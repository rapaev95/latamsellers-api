"""Listing Journey — KPI snapshots, stage detection, change log, anomalies.

Three pieces of the launch→top→maintain workflow live here:

1. **listing_kpi_daily** — daily snapshot per (user, item, date) with
   visits / units / conversion / ad metrics. The table is the single
   trustworthy time-series for trend / anomaly detection — ml_visits
   rotates its rolling windows, this one doesn't.

2. **listing_changelog** — events per item: photo change, price change,
   promo start, video added, manual notes. Source 'auto' means a daily
   cron diffed ml_item_context snapshots; 'manual' means the seller
   pressed "+ Зафиксировать тест" in the UI.

3. **Stage** is computed on-read from age + KPIs (no separate state
   column — keeps the model simple and self-correcting):
     pre_launch / launch / push / climb / top / maintain.

Anomaly detection: runs on-read against listing_kpi_daily. Flags items
where today's value is < 50% of the 7-day median (visits or conversion).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS listing_kpi_daily (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  date DATE NOT NULL,
  visits INTEGER,
  units_sold INTEGER,
  revenue NUMERIC,
  conversion NUMERIC,
  ad_spend NUMERIC,
  drr_pct NUMERIC,
  position INTEGER,            -- if available from positions tracking
  organic_share NUMERIC,
  available_quantity INTEGER,
  captured_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, item_id, date)
);
CREATE INDEX IF NOT EXISTS idx_kpi_daily_user_date ON listing_kpi_daily(user_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_kpi_daily_item ON listing_kpi_daily(user_id, item_id, date DESC);

CREATE TABLE IF NOT EXISTS listing_changelog (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  source TEXT NOT NULL,        -- 'auto' | 'manual'
  event_type TEXT NOT NULL,    -- 'photo_change' | 'price_change' | 'promo_start' | 'video_added' | 'note' | ...
  before_value JSONB,
  after_value JSONB,
  note TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_changelog_item ON listing_changelog(user_id, item_id, created_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── KPI snapshot ──────────────────────────────────────────────────────────────

async def snapshot_today(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    visits_by_item: dict[str, int],
    units_by_item: dict[str, int],
    revenue_by_item: dict[str, float],
    available_qty_by_item: dict[str, int],
) -> int:
    """Insert/update today's KPI row per item from pre-aggregated maps.

    Conversion = units / visits (clamped 0-100%).
    Returns count of upserted rows.
    """
    today = date.today()
    saved = 0
    item_ids = (
        set(visits_by_item.keys()) | set(units_by_item.keys()) |
        set(revenue_by_item.keys()) | set(available_qty_by_item.keys())
    )
    async with pool.acquire() as conn:
        for item_id in item_ids:
            visits = visits_by_item.get(item_id, 0)
            units = units_by_item.get(item_id, 0)
            revenue = revenue_by_item.get(item_id, 0.0)
            avail = available_qty_by_item.get(item_id)
            conversion = (units / visits * 100) if visits > 0 else None
            if conversion is not None:
                conversion = min(max(conversion, 0.0), 100.0)
            await conn.execute(
                """
                INSERT INTO listing_kpi_daily
                  (user_id, item_id, date, visits, units_sold, revenue,
                   conversion, available_quantity)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, item_id, date) DO UPDATE SET
                  visits = EXCLUDED.visits,
                  units_sold = EXCLUDED.units_sold,
                  revenue = EXCLUDED.revenue,
                  conversion = EXCLUDED.conversion,
                  available_quantity = EXCLUDED.available_quantity,
                  captured_at = NOW()
                """,
                user_id, item_id, today,
                int(visits or 0), int(units or 0),
                float(revenue or 0.0),
                float(conversion) if conversion is not None else None,
                int(avail) if avail is not None else None,
            )
            saved += 1
    return saved


async def get_kpi_history(
    pool: asyncpg.Pool, user_id: int, item_id: str, days: int = 30,
) -> list[dict[str, Any]]:
    cutoff = date.today() - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT to_char(date, 'YYYY-MM-DD') AS date,
                   visits, units_sold, revenue, conversion,
                   ad_spend, drr_pct, position, organic_share,
                   available_quantity
              FROM listing_kpi_daily
             WHERE user_id = $1 AND item_id = $2 AND date >= $3
             ORDER BY date ASC
            """,
            user_id, item_id, cutoff,
        )
    return [
        {
            "date": r["date"],
            "visits": r["visits"],
            "unitsSold": r["units_sold"],
            "revenue": float(r["revenue"]) if r["revenue"] is not None else None,
            "conversion": float(r["conversion"]) if r["conversion"] is not None else None,
            "adSpend": float(r["ad_spend"]) if r["ad_spend"] is not None else None,
            "drrPct": float(r["drr_pct"]) if r["drr_pct"] is not None else None,
            "position": r["position"],
            "organicShare": float(r["organic_share"]) if r["organic_share"] is not None else None,
            "availableQuantity": r["available_quantity"],
        }
        for r in rows
    ]


# ── Changelog ─────────────────────────────────────────────────────────────────

async def add_event(
    pool: asyncpg.Pool, user_id: int, item_id: str,
    *, event_type: str, source: str = "manual",
    before_value: Any = None, after_value: Any = None, note: Optional[str] = None,
) -> int:
    async with pool.acquire() as conn:
        evt_id = await conn.fetchval(
            """
            INSERT INTO listing_changelog
              (user_id, item_id, source, event_type, before_value, after_value, note)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
            RETURNING id
            """,
            user_id, item_id, source, event_type,
            json.dumps(before_value) if before_value is not None else None,
            json.dumps(after_value) if after_value is not None else None,
            note,
        )
    return int(evt_id)


async def get_changelog(
    pool: asyncpg.Pool, user_id: int, item_id: str, limit: int = 30,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, source, event_type, before_value, after_value, note,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
              FROM listing_changelog
             WHERE user_id = $1 AND item_id = $2
             ORDER BY created_at DESC
             LIMIT $3
            """,
            user_id, item_id, int(limit),
        )
    out = []
    for r in rows:
        before_v = r["before_value"]
        after_v = r["after_value"]
        for key, val in [("before", before_v), ("after", after_v)]:
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:  # noqa: BLE001
                    pass
            if key == "before":
                before_v = val
            else:
                after_v = val
        out.append({
            "id": int(r["id"]),
            "source": r["source"],
            "eventType": r["event_type"],
            "beforeValue": before_v,
            "afterValue": after_v,
            "note": r["note"],
            "createdAt": r["created_at"],
        })
    return out


# ── Auto-diff: detect changes in ml_item_context vs yesterday ──────────────────

async def detect_context_changes(
    pool: asyncpg.Pool, user_id: int,
) -> dict[str, int]:
    """Compare each item's ml_item_context snapshot today vs yesterday's.
    For changed price / photo count / title — emit a 'auto' changelog event.

    Lightweight: we don't keep historical context snapshots; we use the
    most recent changelog entry of the same event_type per item to avoid
    re-logging the same change daily.
    """
    saved = 0
    async with pool.acquire() as conn:
        items = await conn.fetch(
            """
            SELECT item_id, price, title,
                   COALESCE(jsonb_array_length(pictures), 0) AS pictures_count
              FROM ml_item_context
             WHERE user_id = $1
            """,
            user_id,
        )
        for it in items:
            item_id = it["item_id"]
            cur_price = float(it["price"]) if it["price"] is not None else None
            cur_photos = int(it["pictures_count"] or 0)
            cur_title = it["title"] or ""

            # Last logged snapshot per type
            last_price = await conn.fetchrow(
                """
                SELECT after_value FROM listing_changelog
                 WHERE user_id = $1 AND item_id = $2 AND event_type = 'price_change'
                 ORDER BY created_at DESC LIMIT 1
                """,
                user_id, item_id,
            )
            last_photos = await conn.fetchrow(
                """
                SELECT after_value FROM listing_changelog
                 WHERE user_id = $1 AND item_id = $2 AND event_type = 'photo_change'
                 ORDER BY created_at DESC LIMIT 1
                """,
                user_id, item_id,
            )
            last_title = await conn.fetchrow(
                """
                SELECT after_value FROM listing_changelog
                 WHERE user_id = $1 AND item_id = $2 AND event_type = 'title_change'
                 ORDER BY created_at DESC LIMIT 1
                """,
                user_id, item_id,
            )

            def _last_val(row) -> Any:
                if not row:
                    return None
                v = row["after_value"]
                if isinstance(v, str):
                    try:
                        return json.loads(v)
                    except Exception:  # noqa: BLE001
                        return v
                return v

            prev_price = _last_val(last_price)
            if cur_price is not None and prev_price != cur_price:
                # If no prior log, only record if we see a price now (initial seed
                # is fine — establishes baseline for next compare)
                await conn.execute(
                    """
                    INSERT INTO listing_changelog
                      (user_id, item_id, source, event_type, before_value, after_value)
                    VALUES ($1, $2, 'auto', 'price_change', $3::jsonb, $4::jsonb)
                    """,
                    user_id, item_id,
                    json.dumps(prev_price) if prev_price is not None else None,
                    json.dumps(cur_price),
                )
                saved += 1

            prev_photos = _last_val(last_photos)
            if prev_photos != cur_photos:
                await conn.execute(
                    """
                    INSERT INTO listing_changelog
                      (user_id, item_id, source, event_type, before_value, after_value)
                    VALUES ($1, $2, 'auto', 'photo_change', $3::jsonb, $4::jsonb)
                    """,
                    user_id, item_id,
                    json.dumps(prev_photos) if prev_photos is not None else None,
                    json.dumps(cur_photos),
                )
                saved += 1

            prev_title = _last_val(last_title)
            if isinstance(prev_title, str) and prev_title != cur_title and cur_title:
                await conn.execute(
                    """
                    INSERT INTO listing_changelog
                      (user_id, item_id, source, event_type, before_value, after_value)
                    VALUES ($1, $2, 'auto', 'title_change', $3::jsonb, $4::jsonb)
                    """,
                    user_id, item_id,
                    json.dumps(prev_title)[:500],
                    json.dumps(cur_title)[:500],
                )
                saved += 1
            elif prev_title is None and cur_title:
                # Seed initial title for future diffs
                await conn.execute(
                    """
                    INSERT INTO listing_changelog
                      (user_id, item_id, source, event_type, before_value, after_value, note)
                    VALUES ($1, $2, 'auto', 'title_change', NULL, $3::jsonb, 'baseline')
                    """,
                    user_id, item_id, json.dumps(cur_title)[:500],
                )
                saved += 1
    return {"events_logged": saved}


# ── Stage detection (pure function — no DB persistence) ────────────────────────

def detect_stage(
    *,
    age_days: int,
    position: Optional[int],
    organic_share: Optional[float],
    is_published: bool = True,
) -> str:
    """Compute stage label from item age + KPIs."""
    if not is_published:
        return "pre_launch"
    if age_days < 7:
        return "launch"
    if position is not None and position <= 3 and (organic_share or 0) > 0.6:
        return "top"
    if position is not None and position <= 12 and (organic_share or 0) > 0.3:
        return "climb"
    if age_days < 60:
        return "push"
    return "maintain"


# ── Anomaly detector ──────────────────────────────────────────────────────────

async def detect_anomalies(
    pool: asyncpg.Pool, user_id: int, *,
    visits_drop_threshold: float = 0.5,
    conv_drop_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Find items where today's KPI is far below 7-day median.

    Visits drop > 50%: red flag (could be category shift, sub_status, ad pause).
    Conversion drop > 50% with stable visits: card issue (price, image, sub_status).
    """
    today = date.today()
    cutoff_start = today - timedelta(days=8)
    cutoff_end = today - timedelta(days=1)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH baseline AS (
                SELECT item_id,
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY visits) AS med_visits,
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY conversion) AS med_conv
                  FROM listing_kpi_daily
                 WHERE user_id = $1 AND date BETWEEN $2 AND $3
                 GROUP BY item_id
                HAVING COUNT(*) >= 4
            )
            SELECT k.item_id, k.visits, k.conversion,
                   b.med_visits, b.med_conv
              FROM listing_kpi_daily k
              JOIN baseline b ON b.item_id = k.item_id
             WHERE k.user_id = $1 AND k.date = $4
            """,
            user_id, cutoff_start, cutoff_end, today,
        )

    anomalies: list[dict[str, Any]] = []
    for r in rows:
        item_id = r["item_id"]
        visits = r["visits"] or 0
        med_v = float(r["med_visits"] or 0)
        conv = r["conversion"]
        med_c = float(r["med_conv"]) if r["med_conv"] is not None else None

        flags: list[str] = []
        if med_v >= 10 and visits < med_v * visits_drop_threshold:
            flags.append("visits_drop")
        if (med_c is not None and conv is not None and visits >= 50
                and conv < med_c * conv_drop_threshold):
            flags.append("conversion_drop")

        if flags:
            anomalies.append({
                "itemId": item_id,
                "flags": flags,
                "today": {"visits": visits, "conversion": float(conv) if conv is not None else None},
                "baseline": {
                    "medianVisits": round(med_v, 1),
                    "medianConversion": round(float(med_c), 2) if med_c is not None else None,
                },
            })
    return anomalies


# ── Aggregated journey payload (drawer reads this) ─────────────────────────────

async def get_journey(
    pool: asyncpg.Pool, user_id: int, item_id: str,
) -> dict[str, Any]:
    """One call returns everything the journey drawer needs:
    - current stage
    - last 30 days of KPIs
    - last 30 changelog events
    - any anomaly today
    """
    history = await get_kpi_history(pool, user_id, item_id, days=30)
    changelog = await get_changelog(pool, user_id, item_id, limit=30)

    # Stage estimation — we don't have created_at for items themselves yet;
    # use earliest changelog event or fall back to "push" if we have any
    # KPI history.
    earliest_dt: Optional[datetime] = None
    async with pool.acquire() as conn:
        first_ctx = await conn.fetchval(
            "SELECT MIN(fetched_at) FROM ml_item_context WHERE user_id = $1 AND item_id = $2",
            user_id, item_id,
        )
        if first_ctx:
            earliest_dt = first_ctx
    age_days = 999
    if earliest_dt:
        age_days = (datetime.now(timezone.utc) - earliest_dt.astimezone(timezone.utc)).days

    latest_kpi = history[-1] if history else None
    position = latest_kpi.get("position") if latest_kpi else None
    organic_share = latest_kpi.get("organicShare") if latest_kpi else None

    stage = detect_stage(
        age_days=age_days,
        position=position,
        organic_share=organic_share,
        is_published=True,
    )

    # Anomaly check for THIS item only
    anomalies = await detect_anomalies(pool, user_id)
    item_anomaly = next((a for a in anomalies if a["itemId"] == item_id), None)

    return {
        "itemId": item_id,
        "stage": stage,
        "ageDays": age_days,
        "kpiHistory": history,
        "changelog": changelog,
        "anomaly": item_anomaly,
    }
