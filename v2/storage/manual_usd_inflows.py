"""Manual external USD inflow records.

Some BRL→USD conversions never touch C6 — Bybit USDT cash-out, CALIZA-Nubank
direct transfer, Cred.Nubank TS, custom in-person câmbio, etc. The user
records each of these manually and we feed them into the same FIFO
inventory the C6-paired entradas use, so the Câmbio banner and per-row
FIFO costs cover ALL the dollar inflows, not just the C6 path.

Storage shape mirrors a single câmbio operation:
  date, usd_received, brl_paid, source label, free-text note.
Computed `rate = brl_paid / usd_received` lives in code, not the column.

One row per inflow event. No upsert / dedupe — same date+amount can legitimately
appear twice (two separate cash-outs at different times). User-driven CRUD
controls duplication.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type, datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


# Allowed `source` labels — keep the list short; "manual" is the catch-all.
# UI surfaces these as a dropdown so users don't accidentally create variant
# spellings like "bybit" / "Bybit" / "BYBIT" that wouldn't aggregate together.
SOURCE_OPTIONS: tuple[str, ...] = (
    "Bybit",
    "CALIZA-Nubank",
    "Cred.Nubank TS",
    "Manual",  # fallback when none of the above fits
)


@dataclass
class ManualInflow:
    id: int
    user_id: int
    date: date_type
    usd_received: float
    brl_paid: float
    source: str
    note: str
    created_at: datetime
    updated_at: datetime
    project_name: Optional[str] = None  # NULL = legacy row, owner-only


_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS manual_usd_inflows (
      id SERIAL PRIMARY KEY,
      user_id BIGINT NOT NULL,
      date DATE NOT NULL,
      usd_received NUMERIC(15, 2) NOT NULL CHECK (usd_received > 0),
      brl_paid NUMERIC(15, 2) NOT NULL CHECK (brl_paid > 0),
      source TEXT NOT NULL DEFAULT 'Manual',
      note TEXT NOT NULL DEFAULT '',
      project_name TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # Lazy add for tables that pre-date the project_name column.
    "ALTER TABLE manual_usd_inflows ADD COLUMN IF NOT EXISTS project_name TEXT",
    "CREATE INDEX IF NOT EXISTS idx_manual_usd_inflows_user_date ON manual_usd_inflows(user_id, date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_manual_usd_inflows_project ON manual_usd_inflows(project_name, user_id) WHERE project_name IS NOT NULL",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Idempotent. Belt-and-suspenders called from each endpoint that touches
    this table — startup registration is optional (memory
    `reference_ad_storage_callers_pattern`)."""
    for stmt in _CREATE_STATEMENTS:
        async with pool.acquire() as conn:
            await conn.execute(stmt)


def _row_to_dataclass(row: asyncpg.Record) -> ManualInflow:
    return ManualInflow(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        date=row["date"],
        usd_received=float(row["usd_received"]),
        brl_paid=float(row["brl_paid"]),
        source=row["source"],
        note=row["note"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        project_name=row["project_name"] if "project_name" in row.keys() else None,
    )


async def list_for_user(pool: asyncpg.Pool, user_id: int) -> list[ManualInflow]:
    """Return all inflows for the user, newest date first."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, date, usd_received, brl_paid, source, note,
                   project_name, created_at, updated_at
            FROM manual_usd_inflows
            WHERE user_id = $1
            ORDER BY date DESC, id DESC
            """,
            user_id,
        )
    return [_row_to_dataclass(r) for r in rows]


async def get_by_id(pool: asyncpg.Pool, inflow_id: int) -> Optional[ManualInflow]:
    """Fetch by id without owner filter. Caller MUST verify authz before
    acting on the row. Used by delete handlers that need the row's
    user_id + project_name + created_at to run project-membership checks."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, date, usd_received, brl_paid, source, note,
                   project_name, created_at, updated_at
            FROM manual_usd_inflows
            WHERE id = $1
            """,
            inflow_id,
        )
    return _row_to_dataclass(row) if row else None


async def create(
    pool: asyncpg.Pool,
    *,
    user_id: int,
    date: date_type,
    usd_received: float,
    brl_paid: float,
    source: str,
    note: str = "",
    project_name: Optional[str] = None,
) -> ManualInflow:
    """Insert and return the freshly-stored row.

    Caller is responsible for sanity-checking inputs (rate band, future-date
    guard) — that's enforced at the endpoint level so backend logic stays
    simple. Database CHECK constraints catch only zero/negative amounts.

    `project_name` (optional) tags the row to a project so invited team
    members can see it. NULL keeps the legacy owner-only behavior.
    """
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO manual_usd_inflows
              (user_id, date, usd_received, brl_paid, source, note, project_name)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, user_id, date, usd_received, brl_paid, source, note,
                      project_name, created_at, updated_at
            """,
            user_id, date, usd_received, brl_paid, source, note, project_name,
        )
    return _row_to_dataclass(row)


async def update(
    pool: asyncpg.Pool,
    *,
    inflow_id: int,
    user_id: int,
    date: date_type,
    usd_received: float,
    brl_paid: float,
    source: str,
    note: str = "",
) -> Optional[ManualInflow]:
    """Update by (id, user_id) — returns None if the row doesn't exist or
    belongs to another user. user_id check prevents tenant leakage."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE manual_usd_inflows
               SET date = $3,
                   usd_received = $4,
                   brl_paid = $5,
                   source = $6,
                   note = $7,
                   updated_at = NOW()
             WHERE id = $1 AND user_id = $2
            RETURNING id, user_id, date, usd_received, brl_paid, source, note,
                      project_name, created_at, updated_at
            """,
            inflow_id, user_id, date, usd_received, brl_paid, source, note,
        )
    return _row_to_dataclass(row) if row else None


async def delete(pool: asyncpg.Pool, *, inflow_id: int, user_id: int) -> bool:
    """Delete by (id, user_id). Returns True if a row was removed.

    Owner-only path. For project-member deletes (caller != owner), use
    delete_by_id() after running project_members.enforce_caller_can_delete
    at the route level.
    """
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM manual_usd_inflows WHERE id = $1 AND user_id = $2",
            inflow_id, user_id,
        )
    return res.endswith(" 1")


async def delete_by_id(pool: asyncpg.Pool, inflow_id: int) -> bool:
    """Delete by id only — caller MUST have already verified authz.
    Returns True if a row was removed."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM manual_usd_inflows WHERE id = $1",
            inflow_id,
        )
    return res.endswith(" 1")
