"""ML notices sync job: pull /communications/notices per user → Railway Postgres → Telegram.

Reads and writes the SAME Railway Postgres that Next.js Escalar reads from:
- ml_user_tokens         (source of per-user OAuth tokens — owned by ml_oauth.py)
- ml_notices             (target: content + dedup + TG dispatch state)
- notification_settings  (per-user TG chat_id + language + notify_ml_news flag)

Token refresh is delegated to ml_oauth.get_valid_access_token() so we share
the same refresh-margin logic as the on-demand `/access-token` endpoint.

Idempotent by (user_id, notice_id): re-runs only touch mutable columns.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc
from . import telegram_notify as tg_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
ML_RATE_LIMIT_SLEEP = 0.05         # 50 req/sec app-wide → 50ms between user calls
TG_MESSAGE_THROTTLE = 1.1          # Telegram: 1 msg/sec per chat, 1.1 for safety
NOTICES_PAGE_LIMIT = 50
NOTICES_MAX_PAGES = 8              # up to 400 per run per user
TG_BATCH_CAP = 20                  # cap per-user TG sends per tick


# ── Schema bootstrap ──────────────────────────────────────────────────────────

CREATE_NOTICES_SQL = """
CREATE TABLE IF NOT EXISTS ml_notices (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  notice_id TEXT NOT NULL,
  label TEXT,
  description TEXT,
  from_date TIMESTAMPTZ,
  tags JSONB DEFAULT '[]'::jsonb,
  actions JSONB DEFAULT '[]'::jsonb,
  raw JSONB,
  topic TEXT,
  resource TEXT,
  first_seen_at TIMESTAMPTZ DEFAULT NOW(),
  read_at TIMESTAMPTZ,
  telegram_sent_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, notice_id)
);
ALTER TABLE ml_notices ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE ml_notices ADD COLUMN IF NOT EXISTS resource TEXT;
CREATE INDEX IF NOT EXISTS idx_ml_notices_user_unread ON ml_notices(user_id, read_at);
CREATE INDEX IF NOT EXISTS idx_ml_notices_user_tg_pending ON ml_notices(user_id) WHERE telegram_sent_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_ml_notices_from_date ON ml_notices(user_id, from_date DESC);
CREATE INDEX IF NOT EXISTS idx_ml_notices_user_topic ON ml_notices(user_id, topic);

CREATE TABLE IF NOT EXISTS notification_settings (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL UNIQUE,
  telegram_chat_id TEXT,
  notify_daily_sales BOOLEAN DEFAULT TRUE,
  notify_acos_change BOOLEAN DEFAULT TRUE,
  notify_ml_news BOOLEAN DEFAULT TRUE,
  acos_threshold NUMERIC DEFAULT 5,
  language TEXT DEFAULT 'pt',
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_NOTICES_SQL)


# ── ML API ────────────────────────────────────────────────────────────────────

async def _fetch_notices(http: httpx.AsyncClient, access_token: str) -> list[dict[str, Any]]:
    """Page through /communications/notices until exhausted (cap NOTICES_MAX_PAGES)."""
    out: list[dict[str, Any]] = []
    offset = 0
    headers = {"Authorization": f"Bearer {access_token}"}
    for _ in range(NOTICES_MAX_PAGES):
        try:
            r = await http.get(
                f"{ML_API_BASE}/communications/notices",
                params={"limit": NOTICES_PAGE_LIMIT, "offset": offset},
                headers=headers,
                timeout=20.0,
            )
        except Exception as err:  # noqa: BLE001
            log.warning("notices fetch error: %s", err)
            break
        if r.status_code != 200:
            log.warning("notices %s: %s", r.status_code, r.text[:200])
            break
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            break
        out.extend(results)
        if len(results) < NOTICES_PAGE_LIMIT:
            break
        offset += NOTICES_PAGE_LIMIT
        await asyncio.sleep(ML_RATE_LIMIT_SLEEP)
    return out


# ── Sync one user ─────────────────────────────────────────────────────────────

async def upsert_normalized(
    pool: asyncpg.Pool,
    user_id: int,
    notice: dict[str, Any],
) -> bool:
    """Upsert a single normalize_event() output row into ml_notices.

    Used by cron jobs that synthesize notices from non-webhook sources (e.g.
    promotions discovered by ml_user_promotions.refresh). Returns True if a
    new row was inserted, False if it already existed (still updates content).
    """
    nid = str(notice.get("notice_id") or "")
    if not nid:
        return False
    async with pool.acquire() as conn:
        existed = await conn.fetchval(
            "SELECT 1 FROM ml_notices WHERE user_id = $1 AND notice_id = $2",
            user_id, nid,
        )
        await conn.execute(
            """
            INSERT INTO ml_notices
              (user_id, notice_id, label, description, from_date,
               tags, actions, raw, topic, resource, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, NOW())
            ON CONFLICT (user_id, notice_id) DO UPDATE SET
              label = EXCLUDED.label,
              description = EXCLUDED.description,
              from_date = EXCLUDED.from_date,
              tags = EXCLUDED.tags,
              actions = EXCLUDED.actions,
              raw = EXCLUDED.raw,
              topic = EXCLUDED.topic,
              resource = EXCLUDED.resource,
              updated_at = NOW()
            """,
            user_id,
            nid,
            notice.get("label"),
            notice.get("description"),
            notice.get("from_date"),
            json.dumps(notice.get("tags") or [], default=str),
            json.dumps(notice.get("actions") or [], default=str),
            json.dumps(notice.get("raw") or {}, default=str),
            notice.get("topic"),
            notice.get("resource"),
        )
    return existed is None


async def _upsert_notices(conn: asyncpg.Connection, user_id: int, notices: list[dict[str, Any]]) -> int:
    saved = 0
    for n in notices:
        nid = n.get("id")
        if not nid:
            continue
        await conn.execute(
            """
            INSERT INTO ml_notices
              (user_id, notice_id, label, description, from_date, tags, actions, raw, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, NOW())
            ON CONFLICT (user_id, notice_id) DO UPDATE SET
              label = EXCLUDED.label,
              description = EXCLUDED.description,
              from_date = EXCLUDED.from_date,
              tags = EXCLUDED.tags,
              actions = EXCLUDED.actions,
              raw = EXCLUDED.raw,
              updated_at = NOW()
            """,
            user_id,
            str(nid),
            n.get("label"),
            n.get("description"),
            n.get("from_date"),
            json.dumps(n.get("tags") or []),
            json.dumps(n.get("actions") or []),
            json.dumps(n),
        )
        saved += 1
    return saved


async def _dispatch_to_telegram(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
) -> int:
    """Send any not-yet-sent notices to this user's Telegram, if enabled."""
    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, notify_ml_news, COALESCE(language, 'pt') AS language
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
        if not settings or not settings["notify_ml_news"] or not settings["telegram_chat_id"]:
            return 0
        chat_id = str(settings["telegram_chat_id"])
        language = settings["language"] or "pt"

        # Skip low-value platform notices (invoices, price_suggestion, etc.) —
        # they come from /communications/notices and flood the chat without
        # useful context. Only send real seller events (orders/questions/
        # claims/items/messages) + anything with a topic webhook topic attached.
        NOISE_PREFIXES = (
            "invoices:",
            "price_suggestion:",
            "payments:",
            "stock-locations:",
            "shipments:",
            "fbm_stock_operations:",
            "catalog_item_competition",
            "catalog_suggestions",
            # Promotions webhook events: useful for triggering refresh
            # (so ml_user_promotions cache + topic=promotions notice get
            # produced), but the events themselves carry only an internal
            # OFFER-/CANDIDATE- id, not the actionable promotion_id. Mark
            # silent — the seller gets ONE rich promotions: notice with
            # Aceitar/Rejeitar buttons instead of a stub here.
            "public_offers:",
            "public_candidates:",
        )
        # First pass: bulk-mark noisy notices as sent so they drop out of the
        # queue without consuming a TG send slot. Idempotent — once cleared,
        # future backfills won't duplicate them.
        noise_where = " OR ".join([f"notice_id LIKE '{p}%'" for p in NOISE_PREFIXES])
        if noise_where:
            await conn.execute(
                f"""
                UPDATE ml_notices
                   SET telegram_sent_at = NOW()
                 WHERE user_id = $1 AND telegram_sent_at IS NULL AND ({noise_where})
                """,
                user_id,
            )

        # Questions are dispatched by ml_questions_dispatch (rich format with
        # AI suggestion + Aprovar/Editar buttons + product context). Mark
        # questions notices as sent here so we don't double-notify.
        await conn.execute(
            """
            UPDATE ml_notices
               SET telegram_sent_at = NOW()
             WHERE user_id = $1
               AND telegram_sent_at IS NULL
               AND topic IN ('questions', 'questions_v2')
            """,
            user_id,
        )

        pending_rows = await conn.fetch(
            """
            SELECT notice_id, label, description, actions, tags, topic, raw
              FROM ml_notices
             WHERE user_id = $1 AND telegram_sent_at IS NULL
             ORDER BY from_date ASC NULLS FIRST
             LIMIT $2
            """,
            user_id, TG_BATCH_CAP,
        )

    sent = 0
    for row in pending_rows:
        notice = {
            "notice_id": row["notice_id"],
            "label": row["label"],
            "description": row["description"],
            "actions": row["actions"] or [],
            "tags": row["tags"] or [],
            "topic": row["topic"],
            "raw": row["raw"] or {},
        }
        ok = await tg_svc.send_notice(chat_id, notice, language, http)
        if ok:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ml_notices SET telegram_sent_at = NOW() WHERE user_id = $1 AND notice_id = $2",
                    user_id, row["notice_id"],
                )
            sent += 1
        await asyncio.sleep(TG_MESSAGE_THROTTLE)
    return sent


async def _sync_one_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
) -> dict[str, int]:
    """One tick:
      1. fetch /communications/notices (may be empty for active sellers — normal)
      2. upsert any new notices into ml_notices
      3. dispatch ALL pending rows to Telegram

    Step 3 must run on every tick regardless of step 1 — webhook is the
    primary source of seller events (orders/questions/items/...) and writes
    directly to ml_notices. If we early-returned on empty /communications,
    webhook-saved rows would never get dispatched (production bug 2026-04-25).
    """
    # Delegate refresh to ml_oauth — handles refresh-margin, invalid_grant, etc.
    try:
        access_token, _expires_at, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("user %s: token refresh failed: %s", user_id, err)
        return {"user_id": user_id, "fetched": 0, "saved": 0, "sent": 0}

    notices = await _fetch_notices(http, access_token)
    saved = 0
    if notices:
        async with pool.acquire() as conn:
            saved = await _upsert_notices(conn, user_id, notices)

    # Always dispatch — there may be pending rows from webhooks.
    sent = await _dispatch_to_telegram(pool, http, user_id)
    return {"user_id": user_id, "fetched": len(notices), "saved": saved, "sent": sent}


# ── Public helpers for schedulers ─────────────────────────────────────────────

async def dispatch_pending_for_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
) -> int:
    """Public wrapper around `_dispatch_to_telegram` for the dispatch-only cron
    job (separate from the fetch loop in sync_all_users_notices)."""
    return await _dispatch_to_telegram(pool, http, user_id)


async def dispatch_all_pending(pool: asyncpg.Pool) -> dict[str, Any]:
    """Drain TG queue for every user with any pending notices.

    Independent of /communications/notices fetch — webhook keeps writing to
    ml_notices in real time, this cron just empties the outbox. Call every
    1-2 min to keep latency low for buyers' questions / new orders / paused
    items without burning ML rate-limits on /communications fetches.
    """
    if pool is None:
        return {"users": 0, "sent": 0}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT n.user_id
              FROM notification_settings n
              JOIN ml_notices m ON m.user_id = n.user_id
                                AND m.telegram_sent_at IS NULL
             WHERE n.notify_ml_news = TRUE
               AND n.telegram_chat_id IS NOT NULL
            """,
        )
    user_ids = [r["user_id"] for r in rows]
    if not user_ids:
        return {"users": 0, "sent": 0}

    total_sent = 0
    async with httpx.AsyncClient() as http:
        for uid in user_ids:
            try:
                sent = await _dispatch_to_telegram(pool, http, uid)
                total_sent += sent
            except Exception as err:  # noqa: BLE001
                log.exception("dispatch_all_pending: user %s failed: %s", uid, err)
            await asyncio.sleep(0.1)
    return {"users": len(user_ids), "sent": total_sent}


# ── Entry point for the APScheduler job ───────────────────────────────────────

async def sync_all_users_notices(pool: asyncpg.Pool) -> dict[str, Any]:
    """Pull notices for every user with stored ML tokens.
    Returns aggregate stats: {users, fetched, saved, sent}."""
    if pool is None:
        log.warning("ml_notices: no pool — skipping tick")
        return {"users": 0, "fetched": 0, "saved": 0, "sent": 0}

    await ensure_schema(pool)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
        )
    user_ids = [r["user_id"] for r in rows]

    totals = {"users": 0, "fetched": 0, "saved": 0, "sent": 0}
    if not user_ids:
        return totals

    async with httpx.AsyncClient() as http:
        for uid in user_ids:
            try:
                res = await _sync_one_user(pool, http, uid)
                totals["users"] += 1
                totals["fetched"] += res["fetched"]
                totals["saved"] += res["saved"]
                totals["sent"] += res["sent"]
            except Exception as err:  # noqa: BLE001
                log.exception("ml_notices: user %s failed: %s", uid, err)
            await asyncio.sleep(ML_RATE_LIMIT_SLEEP)
    return totals
