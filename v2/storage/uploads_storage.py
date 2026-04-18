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
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import asyncpg


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
