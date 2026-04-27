"""Per-user cache of ML Product Ads v2: advertisers, campaigns, ads, daily metrics.

All four tables are scoped by `user_id` (FK → users.id). A hourly APScheduler
job populates them from `/advertising/*` — UI reads from here to keep pages
fast and stay under ML's rate limits.

Legacy endpoints `/advertising/product_ads/*` were decommissioned on 2026-02-26,
so every write here originates from the v2 `/advertising/{SITE}/...` family.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Optional

import asyncpg


def _parse_dt(value: Any) -> Optional[datetime]:
    """ML returns timestamps as ISO-8601 strings with a trailing 'Z'. asyncpg
    expects a datetime for TIMESTAMPTZ columns, so normalise here."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.strip()
        # Python 3.11+ accepts 'Z' — guard older runtimes just in case.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS ml_advertisers (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    advertiser_id BIGINT NOT NULL,
    site_id TEXT,
    advertiser_name TEXT,
    account_name TEXT,
    product_id TEXT NOT NULL DEFAULT 'PADS',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, advertiser_id)
);

CREATE TABLE IF NOT EXISTS ml_ad_campaigns (
    user_id INTEGER NOT NULL,
    advertiser_id BIGINT NOT NULL,
    campaign_id BIGINT NOT NULL,
    name TEXT,
    status TEXT,
    strategy TEXT,
    budget NUMERIC,
    automatic_budget BOOLEAN,
    roas_target NUMERIC,
    channel TEXT,
    date_created TIMESTAMPTZ,
    last_updated TIMESTAMPTZ,
    metrics JSONB,
    metrics_date_from DATE,
    metrics_date_to DATE,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, advertiser_id, campaign_id)
);

CREATE INDEX IF NOT EXISTS ml_ad_campaigns_user_idx
    ON ml_ad_campaigns (user_id, advertiser_id);

CREATE TABLE IF NOT EXISTS ml_ad_campaign_metrics_daily (
    user_id INTEGER NOT NULL,
    advertiser_id BIGINT NOT NULL,
    campaign_id BIGINT NOT NULL,
    date DATE NOT NULL,
    metrics JSONB,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, advertiser_id, campaign_id, date)
);

CREATE INDEX IF NOT EXISTS ml_ad_campaign_metrics_daily_lookup_idx
    ON ml_ad_campaign_metrics_daily (user_id, campaign_id, date DESC);

CREATE TABLE IF NOT EXISTS ml_ad_ads (
    user_id INTEGER NOT NULL,
    advertiser_id BIGINT NOT NULL,
    item_id TEXT NOT NULL,
    campaign_id BIGINT,
    title TEXT,
    status TEXT,
    price NUMERIC,
    thumbnail TEXT,
    permalink TEXT,
    domain_id TEXT,
    brand_value_name TEXT,
    metrics JSONB,
    metrics_date_from DATE,
    metrics_date_to DATE,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, advertiser_id, item_id)
);

CREATE INDEX IF NOT EXISTS ml_ad_ads_user_idx
    ON ml_ad_ads (user_id, advertiser_id, status);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create the four cache tables if missing. Idempotent."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)


def _json(value: Any) -> str:
    """asyncpg expects JSONB as text — dumps with ISO-friendly defaults."""
    return json.dumps(value, default=str, ensure_ascii=False)


# ── Advertisers ───────────────────────────────────────────────────────────

async def upsert_advertisers(
    pool: asyncpg.Pool,
    user_id: int,
    advertisers: list[dict],
    product_id: str = "PADS",
) -> None:
    if not advertisers:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO ml_advertisers
                    (user_id, advertiser_id, site_id, advertiser_name, account_name, product_id, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (user_id, advertiser_id) DO UPDATE SET
                    site_id = EXCLUDED.site_id,
                    advertiser_name = EXCLUDED.advertiser_name,
                    account_name = EXCLUDED.account_name,
                    product_id = EXCLUDED.product_id,
                    updated_at = NOW()
                """,
                [
                    (
                        user_id,
                        int(a["advertiser_id"]),
                        a.get("site_id"),
                        a.get("advertiser_name"),
                        a.get("account_name"),
                        product_id,
                    )
                    for a in advertisers
                    if a.get("advertiser_id") is not None
                ],
            )


async def list_advertisers(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT advertiser_id, site_id, advertiser_name, account_name, product_id, updated_at
              FROM ml_advertisers
             WHERE user_id = $1
             ORDER BY advertiser_name NULLS LAST
            """,
            user_id,
        )
    return [dict(r) for r in rows]


# ── Campaigns ─────────────────────────────────────────────────────────────

async def upsert_campaign_snapshot(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    campaigns: list[dict],
    date_from: Optional[date],
    date_to: Optional[date],
) -> None:
    """Persist the list returned by /campaigns/search. `metrics` from that
    response is stored as-is under the same date window — the window is the
    sync cursor; separate daily rows live in ml_ad_campaign_metrics_daily."""
    if not campaigns:
        return
    rows: list[tuple] = []
    for c in campaigns:
        cid = c.get("id")
        if cid is None:
            continue
        rows.append((
            user_id,
            advertiser_id,
            int(cid),
            c.get("name"),
            c.get("status"),
            c.get("strategy"),
            c.get("budget"),
            c.get("automatic_budget"),
            c.get("roas_target"),
            c.get("channel"),
            _parse_dt(c.get("date_created")),
            _parse_dt(c.get("last_updated")),
            _json(c.get("metrics") or {}),
            date_from,
            date_to,
        ))
    if not rows:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO ml_ad_campaigns
                    (user_id, advertiser_id, campaign_id, name, status, strategy,
                     budget, automatic_budget, roas_target, channel,
                     date_created, last_updated,
                     metrics, metrics_date_from, metrics_date_to, synced_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14,$15,NOW())
                ON CONFLICT (user_id, advertiser_id, campaign_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    strategy = EXCLUDED.strategy,
                    budget = EXCLUDED.budget,
                    automatic_budget = EXCLUDED.automatic_budget,
                    roas_target = EXCLUDED.roas_target,
                    channel = EXCLUDED.channel,
                    date_created = EXCLUDED.date_created,
                    last_updated = EXCLUDED.last_updated,
                    metrics = EXCLUDED.metrics,
                    metrics_date_from = EXCLUDED.metrics_date_from,
                    metrics_date_to = EXCLUDED.metrics_date_to,
                    synced_at = NOW()
                """,
                rows,
            )


async def list_campaigns(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: Optional[int] = None,
) -> list[dict]:
    async with pool.acquire() as conn:
        if advertiser_id is None:
            rows = await conn.fetch(
                """
                SELECT * FROM ml_ad_campaigns
                 WHERE user_id = $1
                 ORDER BY last_updated DESC NULLS LAST
                """,
                user_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT * FROM ml_ad_campaigns
                 WHERE user_id = $1 AND advertiser_id = $2
                 ORDER BY last_updated DESC NULLS LAST
                """,
                user_id, advertiser_id,
            )
    result: list[dict] = []
    for r in rows:
        d = dict(r)
        metrics_raw = d.get("metrics")
        if isinstance(metrics_raw, str):
            try:
                d["metrics"] = json.loads(metrics_raw)
            except (ValueError, TypeError):
                d["metrics"] = {}
        elif metrics_raw is None:
            d["metrics"] = {}
        result.append(d)
    return result


async def get_campaign(
    pool: asyncpg.Pool, user_id: int, campaign_id: int,
) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM ml_ad_campaigns
             WHERE user_id = $1 AND campaign_id = $2
             LIMIT 1
            """,
            user_id, campaign_id,
        )
    if not row:
        return None
    d = dict(row)
    raw = d.get("metrics")
    if isinstance(raw, str):
        try:
            d["metrics"] = json.loads(raw)
        except (ValueError, TypeError):
            d["metrics"] = {}
    elif raw is None:
        d["metrics"] = {}
    return d


async def campaign_staleness(
    pool: asyncpg.Pool, user_id: int, advertiser_id: int,
) -> Optional[datetime]:
    """Latest `synced_at` across this advertiser's campaigns — None if empty."""
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            """
            SELECT MAX(synced_at) FROM ml_ad_campaigns
             WHERE user_id = $1 AND advertiser_id = $2
            """,
            user_id, advertiser_id,
        )
    return val


# ── Daily metrics ─────────────────────────────────────────────────────────

async def upsert_daily_metrics(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    campaign_id: int,
    daily_rows: list[dict],
) -> None:
    if not daily_rows:
        return
    rows: list[tuple] = []
    for r in daily_rows:
        d_str = r.get("date")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(d_str) if isinstance(d_str, str) else d_str
        except ValueError:
            continue
        # drop `date` from stored metrics — it's in a separate column
        metrics = {k: v for k, v in r.items() if k != "date"}
        rows.append((user_id, advertiser_id, campaign_id, d, _json(metrics)))
    if not rows:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO ml_ad_campaign_metrics_daily
                    (user_id, advertiser_id, campaign_id, date, metrics, synced_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
                ON CONFLICT (user_id, advertiser_id, campaign_id, date) DO UPDATE SET
                    metrics = EXCLUDED.metrics,
                    synced_at = NOW()
                """,
                rows,
            )


async def list_daily(
    pool: asyncpg.Pool,
    user_id: int,
    campaign_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT date, metrics FROM ml_ad_campaign_metrics_daily
             WHERE user_id = $1
               AND campaign_id = $2
               AND ($3::date IS NULL OR date >= $3)
               AND ($4::date IS NULL OR date <= $4)
             ORDER BY date ASC
            """,
            user_id, campaign_id, date_from, date_to,
        )
    out: list[dict] = []
    for r in rows:
        raw = r["metrics"]
        if isinstance(raw, str):
            try:
                m = json.loads(raw)
            except (ValueError, TypeError):
                m = {}
        else:
            m = raw or {}
        out.append({"date": r["date"].isoformat(), **m})
    return out


# ── Ads (product_ads) ─────────────────────────────────────────────────────

async def upsert_ads_snapshot(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    ads: list[dict],
    date_from: Optional[date],
    date_to: Optional[date],
) -> None:
    if not ads:
        return
    rows: list[tuple] = []
    for a in ads:
        item_id = a.get("item_id")
        if not item_id:
            continue
        rows.append((
            user_id,
            advertiser_id,
            item_id,
            a.get("campaign_id"),
            a.get("title"),
            a.get("status"),
            a.get("price"),
            a.get("thumbnail"),
            a.get("permalink"),
            a.get("domain_id"),
            a.get("brand_value_name"),
            _json(a.get("metrics") or {}),
            date_from,
            date_to,
        ))
    if not rows:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO ml_ad_ads
                    (user_id, advertiser_id, item_id, campaign_id, title, status, price,
                     thumbnail, permalink, domain_id, brand_value_name,
                     metrics, metrics_date_from, metrics_date_to, synced_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$14,NOW())
                ON CONFLICT (user_id, advertiser_id, item_id) DO UPDATE SET
                    campaign_id = EXCLUDED.campaign_id,
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    price = EXCLUDED.price,
                    thumbnail = EXCLUDED.thumbnail,
                    permalink = EXCLUDED.permalink,
                    domain_id = EXCLUDED.domain_id,
                    brand_value_name = EXCLUDED.brand_value_name,
                    metrics = EXCLUDED.metrics,
                    metrics_date_from = EXCLUDED.metrics_date_from,
                    metrics_date_to = EXCLUDED.metrics_date_to,
                    synced_at = NOW()
                """,
                rows,
            )


async def list_ads(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: Optional[int] = None,
    campaign_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Returns (rows, total_count) for UI pagination."""
    where = ["user_id = $1"]
    args: list[Any] = [user_id]
    if advertiser_id is not None:
        args.append(advertiser_id)
        where.append(f"advertiser_id = ${len(args)}")
    if campaign_id is not None:
        args.append(campaign_id)
        where.append(f"campaign_id = ${len(args)}")
    if status:
        args.append(status)
        where.append(f"status = ${len(args)}")
    sql_where = " AND ".join(where)

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM ml_ad_ads WHERE {sql_where}", *args,
        )
        args.extend([limit, offset])
        rows = await conn.fetch(
            f"""
            SELECT * FROM ml_ad_ads
             WHERE {sql_where}
             ORDER BY synced_at DESC, item_id
             LIMIT ${len(args) - 1} OFFSET ${len(args)}
            """,
            *args,
        )

    out: list[dict] = []
    for r in rows:
        d = dict(r)
        raw = d.get("metrics")
        if isinstance(raw, str):
            try:
                d["metrics"] = json.loads(raw)
            except (ValueError, TypeError):
                d["metrics"] = {}
        elif raw is None:
            d["metrics"] = {}
        out.append(d)
    return out, int(total or 0)


async def all_users_with_ml_tokens(pool: asyncpg.Pool) -> list[int]:
    """Users the sync job should poll. Used by ml_ads_sync."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE refresh_token IS NOT NULL"
        )
    return [int(r["user_id"]) for r in rows]
