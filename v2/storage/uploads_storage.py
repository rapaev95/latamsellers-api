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
    user_id: Optional[int] = None       # owner (NULL when not selected at SELECT time)
    project_name: Optional[str] = None  # NULL = legacy upload, no project association


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Lazy migrations ────────────────────────────────────────────────────────
#
# Schema for `uploads` lives in main.py:init_db() — that file was intentionally
# edited by the user, so we don't touch it. Instead we add columns lazily here,
# idempotently, the first time a writer touches them.
#
# Per memory `feedback_silent_migration_failures`: NO EXCEPTION wrapping.
# `ADD COLUMN IF NOT EXISTS` is itself idempotent and safe to call repeatedly.

_parsed_meta_column_ensured = False
_project_name_column_ensured = False


async def _ensure_parsed_meta_column(pool: asyncpg.Pool) -> None:
    """Add `parsed_meta JSONB` to `uploads` if missing. One-shot per process.

    `uploads.parsed_meta` holds the structured output of a one-shot parser
    (DAS PDF, etc) so the UI can show extracted fields without re-parsing on
    every list-view.
    """
    global _parsed_meta_column_ensured
    if _parsed_meta_column_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE uploads ADD COLUMN IF NOT EXISTS parsed_meta JSONB"
        )
    _parsed_meta_column_ensured = True


async def ensure_project_name_column(pool: asyncpg.Pool) -> None:
    """Add `project_name TEXT` to `uploads` if missing. One-shot per process.

    `uploads.project_name` is the project key this upload belongs to (e.g.
    "ARTHUR", "GANZA"). NULL for legacy uploads from before the migration —
    those stay visible only to the owner. New uploads should pass project_name
    in put_file() so invited team members can see them through the project
    membership ACL.
    """
    global _project_name_column_ensured
    if _project_name_column_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE uploads ADD COLUMN IF NOT EXISTS project_name TEXT"
        )
        # Partial index — only non-NULL rows since legacy NULLs are all owner-only.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_uploads_project_user "
            "ON uploads(project_name, user_id) WHERE project_name IS NOT NULL"
        )
    _project_name_column_ensured = True


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
    project_name: Optional[str] = None,
) -> int:
    """Insert (or refresh) a user file. Returns upload id.

    Uses the `ux_uploads_user_hash` partial unique index to upsert: re-uploading
    identical content bumps `created_at` + `filename` (in case the user renamed
    the file) but does not create a second row.

    `project_name` (optional) tags the upload to a specific project so invited
    team members of that project can see it. NULL keeps the legacy
    owner-only behavior. On upsert, a non-NULL incoming project_name overwrites
    NULL, but doesn't clear an already-set value — that requires a separate
    update path.
    """
    await ensure_project_name_column(pool)
    digest = sha256_hex(file_bytes)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO uploads (user_id, source_key, filename, file_bytes, content_sha256, rows, project_name, created_at)
            VALUES ($1, $2, $3, $4, $5, 0, $6, NOW())
            ON CONFLICT (user_id, content_sha256)
            DO UPDATE SET
                filename = EXCLUDED.filename,
                source_key = EXCLUDED.source_key,
                project_name = COALESCE(EXCLUDED.project_name, uploads.project_name),
                created_at = NOW()
            RETURNING id
            """,
            user_id,
            source_key,
            filename,
            file_bytes,
            digest,
            project_name,
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
    """Delete a single upload row owned by `user_id`. Returns True if deleted.

    Use this when the caller is known to be the owner. For project-membership
    deletes (caller is a member, not the owner), use delete_file_by_id() after
    enforcing membership + effective_from checks at the route level.
    """
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id,
            user_id,
        )
    return res.endswith(" 1")


async def delete_file_by_id(pool: asyncpg.Pool, upload_id: int) -> bool:
    """Delete a single upload row by id, regardless of owner.

    Caller MUST have already verified authz — this is a primitive used after
    project-membership checks decide that a non-owner member is allowed to
    delete a specific row. Returns True if deleted.
    """
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM uploads WHERE id = $1",
            upload_id,
        )
    return res.endswith(" 1")


async def get_file_by_id(
    pool: asyncpg.Pool,
    upload_id: int,
) -> Optional[StoredFile]:
    """Fetch a single upload by id without filtering on owner.

    Used by routes that need to inspect the row's user_id + project_name +
    created_at before deciding whether the caller is allowed to delete it.
    Returns the row populated with user_id and project_name fields so the
    handler can run project-membership checks.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, filename, source_key, content_sha256,
                   file_bytes, created_at, project_name
              FROM uploads
             WHERE id = $1
            """,
            upload_id,
        )
    if row is None:
        return None
    return StoredFile(
        id=row["id"],
        user_id=row["user_id"],
        filename=row["filename"] or "",
        source_key=row["source_key"] or "",
        content_sha256=row["content_sha256"] or "",
        file_bytes=bytes(row["file_bytes"]) if row["file_bytes"] is not None else b"",
        created_at=row["created_at"],
        project_name=row["project_name"],
    )


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
