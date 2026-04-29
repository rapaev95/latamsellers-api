"""Per-user raw-file storage in the `uploads` table.

Writes/reads `file_bytes` blobs keyed by `(user_id, source_key)`. Used by the
escalar/finance readers to build per-user views without touching shared `vendas/`
and `_data/armazenagem/` filesystem directories.

Dedupe: `ux_uploads_user_hash` on `(user_id, content_sha256)` prevents duplicate
inserts for the same file content. Re-uploading the same bytes updates the
`created_at` so the "most recent" ordering still reflects user intent.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


@dataclass
class StoredFile:
    id: int
    filename: str
    source_key: str
    content_sha256: str
    file_bytes: bytes
    created_at: datetime


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── parsed_meta column (lazy migration) ────────────────────────────────────
#
# `uploads.parsed_meta` (JSONB) holds the structured output of a one-shot
# parser (DAS PDF, etc) so the UI can show extracted fields without re-parsing
# on every list-view. Schema lives in main.py:init_db() — that file was
# intentionally edited by the user, so we don't touch it. Instead we add the
# column lazily here, idempotently, the first time a parser writes to it.
#
# Per memory `feedback_silent_migration_failures`: NO EXCEPTION wrapping.
# `ADD COLUMN IF NOT EXISTS` is itself idempotent and safe to call repeatedly.

_parsed_meta_column_ensured = False


async def _ensure_parsed_meta_column(pool: asyncpg.Pool) -> None:
    """Add `parsed_meta JSONB` to `uploads` if missing. One-shot per process."""
    global _parsed_meta_column_ensured
    if _parsed_meta_column_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE uploads ADD COLUMN IF NOT EXISTS parsed_meta JSONB"
        )
    _parsed_meta_column_ensured = True


async def set_parsed_meta(pool: asyncpg.Pool, upload_id: int, meta: dict[str, Any]) -> None:
    """Store the parser output for an upload row. Idempotent — overwrites any
    previous parse so a re-run of the parser (after a fix) keeps the latest
    interpretation visible."""
    await _ensure_parsed_meta_column(pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE uploads SET parsed_meta = $1::jsonb WHERE id = $2",
            json.dumps(meta, ensure_ascii=False, default=str),
            upload_id,
        )


async def put_file(
    pool: asyncpg.Pool,
    *,
    user_id: int,
    source_key: str,
    filename: str,
    file_bytes: bytes,
) -> int:
    """Insert (or refresh) a user file. Returns upload id.

    Uses the `ux_uploads_user_hash` partial unique index to upsert: re-uploading
    identical content bumps `created_at` + `filename` (in case the user renamed
    the file) but does not create a second row.
    """
    digest = sha256_hex(file_bytes)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO uploads (user_id, source_key, filename, file_bytes, content_sha256, rows, created_at)
            VALUES ($1, $2, $3, $4, $5, 0, NOW())
            ON CONFLICT (user_id, content_sha256)
            DO UPDATE SET
                filename = EXCLUDED.filename,
                source_key = EXCLUDED.source_key,
                created_at = NOW()
            RETURNING id
            """,
            user_id,
            source_key,
            filename,
            file_bytes,
            digest,
        )
    return int(row["id"])


async def fetch_files_by_source(
    pool: asyncpg.Pool,
    user_id: int,
    source_key: str,
) -> list[StoredFile]:
    """Return every stored file for `(user_id, source_key)`, newest first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, source_key, content_sha256, file_bytes, created_at
            FROM uploads
            WHERE user_id = $1 AND source_key = $2 AND file_bytes IS NOT NULL
            ORDER BY created_at DESC
            """,
            user_id,
            source_key,
        )
    return [
        StoredFile(
            id=r["id"],
            filename=r["filename"],
            source_key=r["source_key"],
            content_sha256=r["content_sha256"] or "",
            file_bytes=bytes(r["file_bytes"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


async def list_sources(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    """Count of files per `source_key` for the user (only rows with file_bytes)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_key, COUNT(*) AS n
            FROM uploads
            WHERE user_id = $1 AND file_bytes IS NOT NULL
            GROUP BY source_key
            """,
            user_id,
        )
    return {r["source_key"] or "": int(r["n"]) for r in rows}


async def delete_file(pool: asyncpg.Pool, user_id: int, upload_id: int) -> bool:
    """Delete a single upload row. Returns True if something was deleted."""
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user_id,
        )
    return res.endswith(" 1")


async def get_file(
    pool: asyncpg.Pool,
    user_id: int,
    upload_id: int,
) -> Optional[StoredFile]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, filename, source_key, content_sha256, file_bytes, created_at
            FROM uploads
            WHERE id = $1 AND user_id = $2
            """,
            upload_id,
            user_id,
        )
    if row is None or row["file_bytes"] is None:
        return None
    return StoredFile(
        id=row["id"],
        filename=row["filename"],
        source_key=row["source_key"] or "",
        content_sha256=row["content_sha256"] or "",
        file_bytes=bytes(row["file_bytes"]),
        created_at=row["created_at"],
    )
