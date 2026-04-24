"""Account health cache — reputation + unanswered questions + recent orders.

Snapshot per user refreshed on-demand. ML updates `seller_reputation` rarely
(level changes take days/weeks), so TTL=6h is plenty — user doesn't need live
data every time the /escalar dashboard opens. Auto-refresh kicks in when the
frontend detects stale cache (same pattern as ml_quality / ml_visits).

Source ML endpoints (parallel):
  1. GET /users/me            → seller_reputation {level_id, power_seller_status,
                                 transactions{completed, canceled, total, ratings{positive}}}
  2. GET /my/received_questions/search?status=UNANSWERED → total
  3. GET /orders/search/recent?seller=me&order.status=paid → results[].length

Everything flattened into one row keyed by user_id (PRIMARY KEY).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_account_health (
  user_id INTEGER PRIMARY KEY,
  nickname TEXT,
  site_id TEXT,
  level TEXT,
  power_seller_status TEXT,
  positive_rate INTEGER,
  completed INTEGER,
  canceled INTEGER,
  unanswered_questions INTEGER,
  recent_orders INTEGER,
  fetched_at TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── ML API ────────────────────────────────────────────────────────────────────

async def _fetch_me(http: httpx.AsyncClient, token: str) -> dict | None:
    try:
        r = await http.get(
            f"{ML_API_BASE}/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as err:  # noqa: BLE001
        log.warning("fetch /users/me failed: %s", err)
    return None


async def _fetch_unanswered(http: httpx.AsyncClient, token: str) -> int:
    try:
        r = await http.get(
            f"{ML_API_BASE}/my/received_questions/search?status=UNANSWERED&api_version=4",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if r.status_code == 200:
            data = r.json() or {}
            # ML returns `total` (new API) or `questions[].length` (old).
            total = data.get("total")
            if isinstance(total, int):
                return total
            questions = data.get("questions")
            if isinstance(questions, list):
                return len(questions)
    except Exception as err:  # noqa: BLE001
        log.warning("fetch unanswered questions failed: %s", err)
    return 0


async def _fetch_recent_orders(http: httpx.AsyncClient, token: str) -> int:
    try:
        r = await http.get(
            f"{ML_API_BASE}/orders/search/recent?seller=me&order.status=paid&sort=date_desc&limit=50",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if r.status_code == 200:
            data = r.json() or {}
            results = data.get("results")
            if isinstance(results, list):
                return len(results)
    except Exception as err:  # noqa: BLE001
        log.warning("fetch recent orders failed: %s", err)
    return 0


def _flatten_me(me: dict | None) -> dict[str, Any]:
    """Extract the fields we care about from /users/me payload."""
    if not me:
        return {
            "nickname": None, "site_id": None, "level": None,
            "power_seller_status": None, "positive_rate": None,
            "completed": 0, "canceled": 0,
        }
    rep = me.get("seller_reputation") or {}
    tx = rep.get("transactions") or {}
    ratings = tx.get("ratings") or {}
    total = tx.get("total") or 0
    positive = ratings.get("positive") or 0
    positive_rate = int(round((positive / total) * 100)) if total > 0 else None
    return {
        "nickname": me.get("nickname"),
        "site_id": me.get("site_id"),
        "level": rep.get("level_id"),
        "power_seller_status": rep.get("power_seller_status"),
        "positive_rate": positive_rate,
        "completed": tx.get("completed") or 0,
        "canceled": tx.get("canceled") or 0,
    }


# ── Refresh ───────────────────────────────────────────────────────────────────

async def refresh_user_health(pool: asyncpg.Pool, user_id: int) -> dict[str, Any] | None:
    """Fetch all 3 ML endpoints in parallel, upsert one row into ml_account_health."""
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("refresh_user_health: token refresh failed for user %s: %s", user_id, err)
        return None

    async with httpx.AsyncClient() as http:
        me, unanswered, recent = await asyncio.gather(
            _fetch_me(http, token),
            _fetch_unanswered(http, token),
            _fetch_recent_orders(http, token),
        )

    flat = _flatten_me(me)
    row = {
        **flat,
        "unanswered_questions": unanswered,
        "recent_orders": recent,
    }

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_account_health
              (user_id, nickname, site_id, level, power_seller_status,
               positive_rate, completed, canceled,
               unanswered_questions, recent_orders, fetched_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
              nickname             = EXCLUDED.nickname,
              site_id              = EXCLUDED.site_id,
              level                = EXCLUDED.level,
              power_seller_status  = EXCLUDED.power_seller_status,
              positive_rate        = EXCLUDED.positive_rate,
              completed            = EXCLUDED.completed,
              canceled             = EXCLUDED.canceled,
              unanswered_questions = EXCLUDED.unanswered_questions,
              recent_orders        = EXCLUDED.recent_orders,
              fetched_at           = NOW()
            """,
            user_id,
            row["nickname"],
            row["site_id"],
            row["level"],
            row["power_seller_status"],
            row["positive_rate"],
            row["completed"],
            row["canceled"],
            row["unanswered_questions"],
            row["recent_orders"],
        )

    return await get_cached(pool, user_id)


# ── Cache readback ────────────────────────────────────────────────────────────

async def get_cached(pool: asyncpg.Pool, user_id: int) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            """
            SELECT nickname, site_id, level, power_seller_status,
                   positive_rate, completed, canceled,
                   unanswered_questions, recent_orders,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_account_health
             WHERE user_id = $1
            """,
            user_id,
        )
    if not r:
        return None
    return {
        "nickname": r["nickname"],
        "siteId": r["site_id"],
        "reputation": {
            "level": r["level"],
            "powerSeller": r["power_seller_status"],
            "positiveRate": r["positive_rate"],
            "completed": int(r["completed"] or 0),
            "canceled": int(r["canceled"] or 0),
        },
        "unansweredQuestions": int(r["unanswered_questions"] or 0),
        "recentOrders": int(r["recent_orders"] or 0),
        "fetchedAt": r["fetched_at"],
    }
