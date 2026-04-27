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
    PRIMARY KEY (user_id, advertiser_id, product_id)
);

-- Migrate legacy 2-col PK to include product_id so the same advertiser_id
-- can have separate rows for PADS / DISPLAY / BADS (a seller routinely
-- has different IDs per product type, but ML's APIs sometimes overlap).
DO $migrate_pk$
DECLARE
  pk_cols int;
BEGIN
  SELECT count(*) INTO pk_cols
    FROM information_schema.key_column_usage
   WHERE table_name = 'ml_advertisers'
     AND constraint_name = 'ml_advertisers_pkey';
  IF pk_cols = 2 THEN
    ALTER TABLE ml_advertisers DROP CONSTRAINT ml_advertisers_pkey;
    ALTER TABLE ml_advertisers ADD PRIMARY KEY (user_id, advertiser_id, product_id);
  END IF;
END
$migrate_pk$;

CREATE TABLE IF NOT EXISTS ml_ad_campaigns (
    user_id INTEGER NOT NULL,
    advertiser_id BIGINT NOT NULL,
    campaign_id BIGINT NOT NULL,
    product_id TEXT NOT NULL DEFAULT 'PADS',
    name TEXT,
    status TEXT,
    strategy TEXT,            -- PADS only
    budget NUMERIC,           -- PADS / BADS
    automatic_budget BOOLEAN, -- PADS only
    roas_target NUMERIC,      -- PADS only
    channel TEXT,             -- PADS only
    date_created TIMESTAMPTZ,
    last_updated TIMESTAMPTZ,
    -- Generic time-window for the row (start_date / end_date as ML returns them).
    -- DISPLAY uses these from /display/campaigns; BADS uses them too.
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    -- DISPLAY fields
    campaign_type TEXT,       -- BADS: 'automatic'/'custom'; DISPLAY: 'GUARANTEED'/'PROGRAMMATIC'
    goal TEXT,                -- DISPLAY only
    site_id TEXT,
    -- BADS fields
    headline TEXT,
    cpc NUMERIC,
    currency TEXT,
    official_store_id BIGINT,
    destination_id BIGINT,
    -- Whole campaign payload — keeps type-specific fields without schema churn.
    raw JSONB,
    metrics JSONB,
    metrics_date_from DATE,
    metrics_date_to DATE,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, advertiser_id, campaign_id, product_id)
);

CREATE INDEX IF NOT EXISTS ml_ad_campaigns_user_idx
    ON ml_ad_campaigns (user_id, advertiser_id, product_id);

-- Idempotent migration for installations created before product_id existed.
DO $migrate_camp_pk$
BEGIN
    BEGIN
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS product_id TEXT NOT NULL DEFAULT 'PADS';
    EXCEPTION WHEN duplicate_column THEN NULL;
    END;
    BEGIN
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS start_date TIMESTAMPTZ;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS end_date TIMESTAMPTZ;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS campaign_type TEXT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS goal TEXT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS site_id TEXT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS headline TEXT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS cpc NUMERIC;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS currency TEXT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS official_store_id BIGINT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS destination_id BIGINT;
        ALTER TABLE ml_ad_campaigns ADD COLUMN IF NOT EXISTS raw JSONB;
    EXCEPTION WHEN OTHERS THEN NULL;
    END;
    -- Update PK to include product_id if it's still 3-cols.
    DECLARE
        pk_cols int;
    BEGIN
        SELECT count(*) INTO pk_cols
          FROM information_schema.key_column_usage
         WHERE table_name = 'ml_ad_campaigns'
           AND constraint_name = 'ml_ad_campaigns_pkey';
        IF pk_cols = 3 THEN
            ALTER TABLE ml_ad_campaigns DROP CONSTRAINT ml_ad_campaigns_pkey;
            ALTER TABLE ml_ad_campaigns ADD PRIMARY KEY (user_id, advertiser_id, campaign_id, product_id);
        END IF;
    END;
END
$migrate_camp_pk$;

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
                ON CONFLICT (user_id, advertiser_id, product_id) DO UPDATE SET
                    site_id = EXCLUDED.site_id,
                    advertiser_name = EXCLUDED.advertiser_name,
                    account_name = EXCLUDED.account_name,
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


async def list_advertisers(
    pool: asyncpg.Pool, user_id: int, product_id: Optional[str] = None,
) -> list[dict]:
    """List advertisers for a user. If product_id is provided, filters by type
    (PADS / DISPLAY / BADS). Default = all types."""
    async with pool.acquire() as conn:
        if product_id:
            rows = await conn.fetch(
                """
                SELECT advertiser_id, site_id, advertiser_name, account_name, product_id, updated_at
                  FROM ml_advertisers
                 WHERE user_id = $1 AND product_id = $2
                 ORDER BY advertiser_name NULLS LAST
                """,
                user_id, product_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT advertiser_id, site_id, advertiser_name, account_name, product_id, updated_at
                  FROM ml_advertisers
                 WHERE user_id = $1
                 ORDER BY product_id, advertiser_name NULLS LAST
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
    *,
    product_id: str = "PADS",
) -> None:
    """Persist a list of campaigns under the chosen product type.

    PADS rows come from `/advertising/{site}/.../product_ads/campaigns/search`
    and carry `id, strategy, budget, automatic_budget, roas_target, channel,
    metrics`. DISPLAY rows come from `/advertising/.../display/campaigns` and
    carry `id, type, goal, start_date, end_date, status`. BADS rows come from
    `/advertising/.../brand_ads/campaigns` and carry `campaign_id, campaign_type,
    headline, budget {amount, currency}, cpc, official_store_id, destination_id`.

    All three are stored in the same row schema; type-specific fields are mapped
    to columns where they exist, and the entire payload is duplicated into `raw`
    so the UI can pick out anything we forgot to map without a backfill."""
    if not campaigns:
        return
    rows: list[tuple] = []
    for c in campaigns:
        # PADS uses `id`; BADS uses `campaign_id`; DISPLAY uses `id`.
        cid = c.get("id") or c.get("campaign_id")
        if cid is None:
            continue
        # BADS budget is {amount, currency}; PADS budget is a number.
        budget_raw = c.get("budget")
        if isinstance(budget_raw, dict):
            budget_val = budget_raw.get("amount")
            currency_val = budget_raw.get("currency")
        else:
            budget_val = budget_raw
            currency_val = c.get("currency")
        rows.append((
            user_id,
            advertiser_id,
            int(cid),
            product_id,
            c.get("name"),
            c.get("status"),
            c.get("strategy"),
            float(budget_val) if isinstance(budget_val, (int, float)) else None,
            c.get("automatic_budget"),
            c.get("roas_target"),
            c.get("channel"),
            _parse_dt(c.get("date_created")),
            _parse_dt(c.get("last_updated")),
            _parse_dt(c.get("start_date")),
            _parse_dt(c.get("end_date")),
            # PADS doesn't use campaign_type field; DISPLAY uses 'type'; BADS uses 'campaign_type'.
            c.get("campaign_type") or c.get("type"),
            c.get("goal"),
            c.get("site_id"),
            c.get("headline"),
            float(c["cpc"]) if isinstance(c.get("cpc"), (int, float)) else None,
            currency_val,
            int(c["official_store_id"]) if c.get("official_store_id") is not None else None,
            int(c["destination_id"]) if c.get("destination_id") is not None else None,
            _json(c),
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
                    (user_id, advertiser_id, campaign_id, product_id,
                     name, status, strategy,
                     budget, automatic_budget, roas_target, channel,
                     date_created, last_updated, start_date, end_date,
                     campaign_type, goal, site_id, headline, cpc, currency,
                     official_store_id, destination_id, raw,
                     metrics, metrics_date_from, metrics_date_to, synced_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                        $16,$17,$18,$19,$20,$21,$22,$23,$24::jsonb,
                        $25::jsonb,$26,$27,NOW())
                ON CONFLICT (user_id, advertiser_id, campaign_id, product_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    strategy = EXCLUDED.strategy,
                    budget = EXCLUDED.budget,
                    automatic_budget = EXCLUDED.automatic_budget,
                    roas_target = EXCLUDED.roas_target,
                    channel = EXCLUDED.channel,
                    date_created = EXCLUDED.date_created,
                    last_updated = EXCLUDED.last_updated,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    campaign_type = EXCLUDED.campaign_type,
                    goal = EXCLUDED.goal,
                    site_id = EXCLUDED.site_id,
                    headline = EXCLUDED.headline,
                    cpc = EXCLUDED.cpc,
                    currency = EXCLUDED.currency,
                    official_store_id = EXCLUDED.official_store_id,
                    destination_id = EXCLUDED.destination_id,
                    raw = EXCLUDED.raw,
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
    *,
    product_id: Optional[str] = None,
) -> list[dict]:
    async with pool.acquire() as conn:
        where = ["user_id = $1"]
        params: list = [user_id]
        if advertiser_id is not None:
            where.append(f"advertiser_id = ${len(params) + 1}")
            params.append(advertiser_id)
        if product_id is not None:
            where.append(f"product_id = ${len(params) + 1}")
            params.append(product_id)
        rows = await conn.fetch(
            f"""
            SELECT * FROM ml_ad_campaigns
             WHERE {' AND '.join(where)}
             ORDER BY COALESCE(last_updated, start_date) DESC NULLS LAST
            """,
            *params,
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
    *, product_id: Optional[str] = None,
) -> Optional[datetime]:
    """Latest `synced_at` across this advertiser's campaigns — None if empty.
    Optionally filter by product type so PADS staleness doesn't mask DISPLAY/BADS."""
    async with pool.acquire() as conn:
        if product_id is None:
            val = await conn.fetchval(
                """
                SELECT MAX(synced_at) FROM ml_ad_campaigns
                 WHERE user_id = $1 AND advertiser_id = $2
                """,
                user_id, advertiser_id,
            )
        else:
            val = await conn.fetchval(
                """
                SELECT MAX(synced_at) FROM ml_ad_campaigns
                 WHERE user_id = $1 AND advertiser_id = $2 AND product_id = $3
                """,
                user_id, advertiser_id, product_id,
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
