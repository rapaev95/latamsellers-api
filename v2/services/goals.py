"""Goal tracking — manual quarterly / monthly targets the seller sets
to track progress against (Sprint 7 of the UI architecture).

Single goal kind in MVP: lucro_liquido. Other kinds (receita, units)
can be added later without schema change — `kind` column is text.

`current_value` and `progress_pct` are computed client-side from
existing PnL data; this service just stores the target + period
boundaries.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS escalar_goals (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL,
      project_name TEXT,                    -- NULL = global, else scoped to one project
      kind TEXT NOT NULL DEFAULT 'lucro_liquido',
      target_amount NUMERIC NOT NULL,
      period_type TEXT NOT NULL DEFAULT 'quarter',   -- month | quarter | year | custom
      period_start DATE NOT NULL,
      period_end DATE NOT NULL,
      currency TEXT NOT NULL DEFAULT 'BRL',
      active BOOLEAN NOT NULL DEFAULT TRUE,
      notes TEXT,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_escalar_goals_user_active ON escalar_goals(user_id, active)",
    "CREATE INDEX IF NOT EXISTS idx_escalar_goals_user_period ON escalar_goals(user_id, period_start, period_end)",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    for stmt in _CREATE_STATEMENTS:
        async with pool.acquire() as conn:
            await conn.execute(stmt)


def _quarter_bounds(today: Optional[date] = None) -> tuple[date, date]:
    today = today or date.today()
    q = (today.month - 1) // 3
    start = date(today.year, q * 3 + 1, 1)
    end_year = today.year if q < 3 else today.year
    end_month = q * 3 + 3
    end_day = (date(end_year + (1 if end_month == 12 else 0),
                    1 if end_month == 12 else end_month + 1, 1)
               - timedelta(days=1)).day
    end = date(end_year, end_month, end_day)
    return start, end


def _month_bounds(today: Optional[date] = None) -> tuple[date, date]:
    today = today or date.today()
    start = date(today.year, today.month, 1)
    next_month = (start + timedelta(days=32)).replace(day=1)
    end = next_month - timedelta(days=1)
    return start, end


def _year_bounds(today: Optional[date] = None) -> tuple[date, date]:
    today = today or date.today()
    return date(today.year, 1, 1), date(today.year, 12, 31)


def compute_period_bounds(period_type: str) -> tuple[date, date]:
    """Helper for the create endpoint — auto-fills period_start/end
    when seller picks "this month / quarter / year" instead of custom."""
    if period_type == "month":
        return _month_bounds()
    if period_type == "year":
        return _year_bounds()
    # default + 'custom' caller passes own dates anyway
    return _quarter_bounds()


async def list_active(
    pool: asyncpg.Pool, user_id: int, *, project_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return goals that haven't been deactivated AND whose period hasn't
    fully passed (period_end >= today). Sorted by period_start desc."""
    today = date.today()
    where = "WHERE user_id = $1 AND active = TRUE AND period_end >= $2"
    params: list[Any] = [user_id, today]
    if project_name is not None:
        where += " AND COALESCE(project_name, '') = $3"
        params.append(project_name or '')
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, user_id, project_name, kind, target_amount,
                   period_type, period_start, period_end, currency,
                   active, notes,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
                   to_char(updated_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS updated_at
              FROM escalar_goals
              {where}
             ORDER BY period_start DESC, id DESC
            """,
            *params,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["target_amount"] = float(d["target_amount"]) if d["target_amount"] is not None else 0.0
        d["period_start"] = d["period_start"].isoformat() if d["period_start"] else None
        d["period_end"] = d["period_end"].isoformat() if d["period_end"] else None
        # Days remaining (negative if period already ended)
        if d["period_end"]:
            try:
                end_d = date.fromisoformat(d["period_end"])
                d["days_left"] = max(0, (end_d - today).days)
            except ValueError:
                d["days_left"] = None
        else:
            d["days_left"] = None
        out.append(d)
    return out


async def upsert_goal(
    pool: asyncpg.Pool, user_id: int, *,
    goal_id: Optional[int] = None,
    project_name: Optional[str] = None,
    kind: str = "lucro_liquido",
    target_amount: float = 0.0,
    period_type: str = "quarter",
    period_start: Optional[date] = None,
    period_end: Optional[date] = None,
    currency: str = "BRL",
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Create or update a goal. When period_start/end aren't provided,
    auto-fill based on period_type."""
    if period_start is None or period_end is None:
        ps, pe = compute_period_bounds(period_type)
        period_start = period_start or ps
        period_end = period_end or pe

    if goal_id is not None:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE escalar_goals
                   SET project_name = $2, kind = $3, target_amount = $4,
                       period_type = $5, period_start = $6, period_end = $7,
                       currency = $8, notes = $9, updated_at = NOW()
                 WHERE id = $1 AND user_id = $10
                RETURNING id
                """,
                goal_id, project_name, kind, target_amount,
                period_type, period_start, period_end,
                currency, notes, user_id,
            )
            return {"id": row["id"], "updated": True} if row else {"error": "not_found"}

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO escalar_goals
              (user_id, project_name, kind, target_amount,
               period_type, period_start, period_end, currency, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            user_id, project_name, kind, target_amount,
            period_type, period_start, period_end, currency, notes,
        )
    return {"id": row["id"], "created": True}


async def deactivate_goal(
    pool: asyncpg.Pool, user_id: int, goal_id: int,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE escalar_goals SET active = FALSE, updated_at = NOW() "
            "WHERE id = $1 AND user_id = $2",
            goal_id, user_id,
        )
    return {"deactivated": result.endswith(" 1"), "id": goal_id}
