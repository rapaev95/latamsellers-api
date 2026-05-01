"""Track outcomes of accepted promotions — sales/revenue in N days after
accept_at — so seller knows which promo types actually drive volume.

Use case: seller accepts SMART campaign with -20% deal price. Was it
worth it? Without tracking — guessing. With analytics:
  - 14d sales volume на товар после accept_at
  - Compared with same item's 14d before accept_at
  - Revenue delta + margin delta
  - AI recommendation после 10+ outcomes: «accept SMART for X categorias»

Daily cron 21:00 UTC closes 14-day-old open windows и finalizes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

WINDOW_DAYS = 14  # length of post-accept observation window
BRT = timezone(timedelta(hours=-3))

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_user_promotion_outcomes (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  promotion_id TEXT NOT NULL,
  promotion_type TEXT,
  promotion_name TEXT,
  accepted_at TIMESTAMPTZ NOT NULL,
  deal_price NUMERIC,
  original_price NUMERIC,
  discount_pct NUMERIC,
  -- baseline window (14d before accept)
  baseline_orders INTEGER DEFAULT 0,
  baseline_units INTEGER DEFAULT 0,
  baseline_revenue NUMERIC DEFAULT 0,
  -- treatment window (14d after accept)
  treatment_orders INTEGER DEFAULT 0,
  treatment_units INTEGER DEFAULT 0,
  treatment_revenue NUMERIC DEFAULT 0,
  -- finalize when treatment window closes
  finalized_at TIMESTAMPTZ,
  delta_revenue_brl NUMERIC,
  delta_units INTEGER,
  raw JSONB,
  UNIQUE(user_id, promotion_id)
);
CREATE INDEX IF NOT EXISTS idx_promo_outcomes_user
  ON ml_user_promotion_outcomes(user_id, accepted_at DESC);
CREATE INDEX IF NOT EXISTS idx_promo_outcomes_open
  ON ml_user_promotion_outcomes(accepted_at)
  WHERE finalized_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


async def record_acceptance(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    promotion_id: str,
    promotion_type: str | None = None,
    promotion_name: str | None = None,
    deal_price: float | None = None,
    original_price: float | None = None,
    discount_pct: float | None = None,
) -> None:
    """Called from tg-action accept handler — opens an outcome window.
    Idempotent via UNIQUE(user_id, promotion_id) — duplicate accepts (rare)
    don't create new rows.

    Baseline window aggregated immediately to avoid drift if seller
    re-runs analytics on stale data.
    """
    if pool is None:
        return
    try:
        await ensure_schema(pool)
    except Exception:  # noqa: BLE001
        return
    accepted_at = datetime.now(timezone.utc)
    baseline_start = accepted_at - timedelta(days=WINDOW_DAYS)
    base_orders, base_units, base_rev = await _aggregate_item_window(
        pool, user_id, item_id, baseline_start, accepted_at,
    )
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_user_promotion_outcomes
              (user_id, item_id, promotion_id, promotion_type, promotion_name,
               accepted_at, deal_price, original_price, discount_pct,
               baseline_orders, baseline_units, baseline_revenue)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (user_id, promotion_id) DO NOTHING
            """,
            user_id, item_id.upper(), promotion_id,
            promotion_type, promotion_name, accepted_at,
            deal_price, original_price, discount_pct,
            base_orders, base_units, float(base_rev),
        )


async def _aggregate_item_window(
    pool: asyncpg.Pool, user_id: int, item_id: str,
    start: datetime, end: datetime,
) -> tuple[int, int, float]:
    """Returns (orders, units, revenue) for one item over [start, end]."""
    import json as _json
    item_id_up = item_id.upper()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT items
              FROM ml_user_orders
             WHERE user_id = $1
               AND date_created BETWEEN $2 AND $3
               AND status NOT IN ('cancelled', 'invalid')
               AND items @> $4::jsonb
            """,
            user_id, start, end,
            _json.dumps([{"mlb": item_id_up}]),
        )
    orders = 0
    units = 0
    revenue = 0.0
    for r in rows:
        items_raw = r["items"]
        if isinstance(items_raw, str):
            try:
                items_raw = _json.loads(items_raw)
            except Exception:  # noqa: BLE001
                continue
        if not isinstance(items_raw, list):
            continue
        matched = False
        for it in items_raw:
            if (it or {}).get("mlb") != item_id_up:
                continue
            matched = True
            units += int(it.get("quantity") or 0)
            revenue += float(it.get("revenue") or 0)
        if matched:
            orders += 1
    return orders, units, revenue


async def finalize_due_outcomes(pool: asyncpg.Pool) -> dict[str, int]:
    """Finalize all outcome rows whose treatment window (14d) is past.
    Aggregates ml_user_orders for the post-accept window, computes deltas,
    sets finalized_at.
    """
    if pool is None:
        return {"finalized": 0}
    await ensure_schema(pool)
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, item_id, accepted_at,
                   baseline_revenue, baseline_units
              FROM ml_user_promotion_outcomes
             WHERE finalized_at IS NULL AND accepted_at <= $1
             LIMIT 200
            """,
            cutoff,
        )

    finalized = 0
    for row in rows:
        treat_start = row["accepted_at"]
        treat_end = treat_start + timedelta(days=WINDOW_DAYS)
        orders, units, rev = await _aggregate_item_window(
            pool, row["user_id"], row["item_id"], treat_start, treat_end,
        )
        delta_rev = float(rev) - float(row["baseline_revenue"] or 0)
        delta_units = int(units) - int(row["baseline_units"] or 0)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ml_user_promotion_outcomes
                   SET treatment_orders = $1, treatment_units = $2,
                       treatment_revenue = $3,
                       delta_revenue_brl = $4, delta_units = $5,
                       finalized_at = NOW()
                 WHERE id = $6
                """,
                orders, units, float(rev),
                round(delta_rev, 2), delta_units, row["id"],
            )
        finalized += 1
    return {"finalized": finalized, "considered": len(rows)}


async def get_user_analytics(
    pool: asyncpg.Pool, user_id: int, days: int = 90,
) -> dict[str, Any]:
    """Returns top-3 winners + worst-3 losers + summary."""
    if pool is None:
        return {"error": "no_db"}
    await ensure_schema(pool)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        finalized = await conn.fetch(
            """
            SELECT id, item_id, promotion_id, promotion_type, promotion_name,
                   accepted_at, deal_price, original_price, discount_pct,
                   baseline_units, baseline_revenue,
                   treatment_units, treatment_revenue,
                   delta_units, delta_revenue_brl, finalized_at
              FROM ml_user_promotion_outcomes
             WHERE user_id = $1 AND finalized_at IS NOT NULL
               AND accepted_at >= $2
             ORDER BY delta_revenue_brl DESC NULLS LAST
            """,
            user_id, since,
        )
        open_rows = await conn.fetchval(
            """
            SELECT COUNT(*) FROM ml_user_promotion_outcomes
             WHERE user_id = $1 AND finalized_at IS NULL
            """,
            user_id,
        )

    rows_list = [dict(r) for r in finalized]
    for r in rows_list:
        # serialize datetimes
        for k, v in list(r.items()):
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
            if hasattr(v, "real"):  # Decimal
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass

    total_count = len(rows_list)
    total_delta = sum(float(r.get("delta_revenue_brl") or 0) for r in rows_list)
    winners = [r for r in rows_list if (r.get("delta_revenue_brl") or 0) > 0]
    losers = [r for r in rows_list if (r.get("delta_revenue_brl") or 0) < 0]

    # By promotion_type aggregation
    by_type: dict[str, dict[str, Any]] = {}
    for r in rows_list:
        t = r.get("promotion_type") or "UNKNOWN"
        b = by_type.setdefault(t, {"count": 0, "delta_total": 0.0})
        b["count"] += 1
        b["delta_total"] += float(r.get("delta_revenue_brl") or 0)
    by_type_list = sorted(
        [{"type": k, **v} for k, v in by_type.items()],
        key=lambda x: x["delta_total"], reverse=True,
    )

    return {
        "period_days": days,
        "finalized_count": total_count,
        "open_count": int(open_rows or 0),
        "total_delta_revenue_brl": round(total_delta, 2),
        "winners_count": len(winners),
        "losers_count": len(losers),
        "top_3_winners": rows_list[:3],
        "top_3_losers": list(reversed(rows_list))[:3] if losers else [],
        "by_type": by_type_list,
    }
