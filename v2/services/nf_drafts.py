"""NF publishing drafts — auto-saved state of the long-running publish wizard.

The wizard (8 steps × N families) takes 10-15 minutes to complete. Without
persistence, a single refresh would lose everything. This service backs the
useDraftAutoSave hook on the frontend.

Persistence model: hybrid (localStorage + this DB table).
- localStorage holds the most recent state for instant restore after refresh.
- This table is the source of truth for cross-device / cross-tab recovery.

Optimistic locking: every PUT requires `expected_version`. Mismatch returns
409 with the current row so the client can decide to overwrite or reload.

Lifecycle:
- create()  → status='draft', version=0
- update()  → bumps version, writes state_json + current_step + idx
- mark_abandoned() → status='abandoned' (kept for 60 more days for support)
- mark_published(plan_id) → status='published' (kept for 7 days for audit)
- Stale-cleanup cron in main.py finishes the lifecycle.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

# UUIDs need pgcrypto for gen_random_uuid(); fall back to uuid_generate_v4 if
# pgcrypto is unavailable. We ensure pgcrypto first.
CREATE_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS nf_publishing_drafts (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ls_user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  managed_account_id   INTEGER NULL,
  nf_upload_id         INTEGER NULL,
  state_json           JSONB NOT NULL DEFAULT '{}'::jsonb,
  current_step         TEXT,
  current_family_idx   INTEGER DEFAULT 0,
  version              INTEGER NOT NULL DEFAULT 0,
  status               TEXT NOT NULL DEFAULT 'draft',
  updated_at           TIMESTAMPTZ DEFAULT NOW(),
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nf_drafts_user_status
  ON nf_publishing_drafts(ls_user_id, status);

CREATE INDEX IF NOT EXISTS idx_nf_drafts_updated_status
  ON nf_publishing_drafts(updated_at) WHERE status = 'draft';
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    """Convert asyncpg Record to a JSON-serializable dict. asyncpg returns
    UUID and datetime objects which Pydantic's default JSON encoder can
    handle, but raw FastAPI may not."""
    d = dict(row)
    if "id" in d and d["id"] is not None:
        d["id"] = str(d["id"])
    # state_json is already a dict via asyncpg+JSONB. If for some reason
    # it came back as a string (older driver path), decode it.
    if isinstance(d.get("state_json"), str):
        try:
            d["state_json"] = json.loads(d["state_json"])
        except Exception:  # noqa: BLE001
            pass
    return d


async def create_draft(
    pool: asyncpg.Pool,
    ls_user_id: int,
    *,
    managed_account_id: Optional[int] = None,
    nf_upload_id: Optional[int] = None,
    initial_state: Optional[dict] = None,
    current_step: Optional[str] = "upload",
) -> dict[str, Any]:
    state = initial_state or {}
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO nf_publishing_drafts (
                ls_user_id, managed_account_id, nf_upload_id,
                state_json, current_step
            )
            VALUES ($1, $2, $3, $4::jsonb, $5)
            RETURNING id, ls_user_id, managed_account_id, nf_upload_id,
                      state_json, current_step, current_family_idx,
                      version, status, updated_at, created_at
            """,
            ls_user_id, managed_account_id, nf_upload_id,
            json.dumps(state), current_step,
        )
    return _row_to_dict(row)


async def list_drafts(
    pool: asyncpg.Pool, ls_user_id: int, *, status: str = "draft", limit: int = 50
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, ls_user_id, managed_account_id, nf_upload_id,
                   state_json, current_step, current_family_idx,
                   version, status, updated_at, created_at
            FROM nf_publishing_drafts
            WHERE ls_user_id = $1 AND status = $2
            ORDER BY updated_at DESC
            LIMIT $3
            """,
            ls_user_id, status, limit,
        )
    return [_row_to_dict(r) for r in rows]


async def get_draft(
    pool: asyncpg.Pool, ls_user_id: int, draft_id: str
) -> Optional[dict[str, Any]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, ls_user_id, managed_account_id, nf_upload_id,
                   state_json, current_step, current_family_idx,
                   version, status, updated_at, created_at
            FROM nf_publishing_drafts
            WHERE id = $1::uuid AND ls_user_id = $2
            """,
            draft_id, ls_user_id,
        )
    return _row_to_dict(row) if row else None


class DraftVersionConflict(Exception):
    """Raised when PUT's expected_version doesn't match DB. Carries the
    current row so the API layer can return it to the client."""
    def __init__(self, current_row: dict[str, Any]) -> None:
        super().__init__("version_conflict")
        self.current_row = current_row


async def update_draft(
    pool: asyncpg.Pool,
    ls_user_id: int,
    draft_id: str,
    *,
    expected_version: int,
    state: Optional[dict] = None,
    current_step: Optional[str] = None,
    current_family_idx: Optional[int] = None,
    managed_account_id: Optional[int] = None,
    nf_upload_id: Optional[int] = None,
) -> dict[str, Any]:
    """Optimistic-locked update. Raises DraftVersionConflict on mismatch."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            current = await conn.fetchrow(
                """
                SELECT id, ls_user_id, managed_account_id, nf_upload_id,
                       state_json, current_step, current_family_idx,
                       version, status, updated_at, created_at
                FROM nf_publishing_drafts
                WHERE id = $1::uuid AND ls_user_id = $2
                FOR UPDATE
                """,
                draft_id, ls_user_id,
            )
            if not current:
                raise LookupError("draft_not_found")
            if current["version"] != expected_version:
                # Caller's optimistic check failed — return current state so
                # the client can merge or reload.
                raise DraftVersionConflict(_row_to_dict(current))

            new_state = state if state is not None else current["state_json"]
            new_step = current_step if current_step is not None else current["current_step"]
            new_idx = (
                current_family_idx if current_family_idx is not None
                else current["current_family_idx"]
            )
            new_mgr = (
                managed_account_id if managed_account_id is not None
                else current["managed_account_id"]
            )
            new_nf = nf_upload_id if nf_upload_id is not None else current["nf_upload_id"]

            row = await conn.fetchrow(
                """
                UPDATE nf_publishing_drafts SET
                    state_json = $1::jsonb,
                    current_step = $2,
                    current_family_idx = $3,
                    managed_account_id = $4,
                    nf_upload_id = $5,
                    version = version + 1,
                    updated_at = NOW()
                WHERE id = $6::uuid
                RETURNING id, ls_user_id, managed_account_id, nf_upload_id,
                          state_json, current_step, current_family_idx,
                          version, status, updated_at, created_at
                """,
                json.dumps(new_state) if isinstance(new_state, (dict, list)) else new_state,
                new_step, new_idx, new_mgr, new_nf, draft_id,
            )
    return _row_to_dict(row)


async def mark_abandoned(
    pool: asyncpg.Pool, ls_user_id: int, draft_id: str
) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE nf_publishing_drafts SET status = 'abandoned', updated_at = NOW()
            WHERE id = $1::uuid AND ls_user_id = $2 AND status = 'draft'
            """,
            draft_id, ls_user_id,
        )
    return result.endswith("1")


async def cleanup_stale(pool: asyncpg.Pool, *, abandon_after_days: int = 30,
                        delete_after_days: int = 90) -> dict[str, int]:
    """Cron-callable: mark old drafts abandoned, hard-delete really old ones.
    Returns counts for telemetry."""
    async with pool.acquire() as conn:
        abandoned = await conn.execute(
            f"""
            UPDATE nf_publishing_drafts SET status = 'abandoned', updated_at = NOW()
            WHERE status = 'draft'
              AND updated_at < NOW() - INTERVAL '{abandon_after_days} days'
            """,
        )
        deleted = await conn.execute(
            f"""
            DELETE FROM nf_publishing_drafts
            WHERE status = 'abandoned'
              AND updated_at < NOW() - INTERVAL '{delete_after_days} days'
            """,
        )
    # Result strings look like "UPDATE 5" / "DELETE 3"
    def _n(s: str) -> int:
        try:
            return int(s.rsplit(" ", 1)[-1])
        except Exception:  # noqa: BLE001
            return 0
    return {"abandoned": _n(abandoned), "deleted": _n(deleted)}
