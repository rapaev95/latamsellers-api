"""ML item context cache — title + attributes + description per item.

Why: AI question replies need product context (attrs, description) to avoid
hallucinating specs. Fetching from ML on every reply is slow (~1-2s per item)
and wasteful — product attrs change rarely. This cache stores them in
Railway Postgres with TTL=24h.

Pattern reference: ml_quality.py (TEST → DB → CACHE).

Schema:
  ml_item_context(
    user_id, item_id (UNIQUE pair),
    title, condition, price, currency, available_quantity, warranty,
    shipping_free, permalink,
    attributes JSONB,  -- [{name, value}, ...] top 30
    description TEXT,  -- plain_text, sliced to 4000 chars
    fetched_at TIMESTAMPTZ
  )

Usage:
  - get_or_refresh(pool, user_id, item_id) — main entry; cache-first, refresh
    on miss/stale (>24h)
  - get_cached(pool, user_id, item_id) — read-only
  - refresh_user_context(pool, user_id, item_ids) — bulk refresh, throttled
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc
from .ml_quality import normalize_item_id

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
RATE_SLEEP = 0.2  # 5 req/sec safe per-user cap
DEFAULT_TTL_HOURS = 24
DESCRIPTION_MAX_CHARS = 4000


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_item_context (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  title TEXT,
  condition TEXT,
  price NUMERIC,
  currency TEXT,
  available_quantity INTEGER,
  warranty TEXT,
  shipping_free BOOLEAN,
  logistic_type TEXT,
  permalink TEXT,
  status TEXT,
  sub_status JSONB DEFAULT '[]'::jsonb,
  attributes JSONB DEFAULT '[]'::jsonb,
  description TEXT DEFAULT '',
  pictures JSONB DEFAULT '[]'::jsonb,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_item_context_user ON ml_item_context(user_id);
CREATE INDEX IF NOT EXISTS idx_ml_item_context_fetched ON ml_item_context(fetched_at);
"""

ALTER_SQL = """
ALTER TABLE ml_item_context
  ADD COLUMN IF NOT EXISTS pictures JSONB DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS status TEXT,
  ADD COLUMN IF NOT EXISTS sub_status JSONB DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS logistic_type TEXT;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)
        await conn.execute(ALTER_SQL)


# ── ML API ────────────────────────────────────────────────────────────────────

async def fetch_from_ml(
    http: httpx.AsyncClient,
    access_token: str,
    item_id: str,
) -> Optional[dict[str, Any]]:
    """Fetch core item + description from ML in parallel. Returns None on
    fatal failure (4xx/5xx for the item itself); description is best-effort.
    """
    mlb = normalize_item_id(item_id)
    if not mlb:
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    attrs_q = ",".join([
        "id", "title", "condition", "price", "currency_id", "permalink",
        "available_quantity", "attributes", "shipping", "warranty",
        "pictures", "status", "sub_status",
    ])

    async def _item():
        try:
            r = await http.get(
                f"{ML_API_BASE}/items/{mlb}?attributes={attrs_q}",
                headers=headers, timeout=10.0,
            )
            if r.status_code == 200:
                return r.json()
            # Surface non-2xx loudly — silent fails here historically left
            # ml_item_context permanently empty for affected items.
            log.warning(
                "ml_item_context: items/%s status=%s body=%s",
                mlb, r.status_code, r.text[:200],
            )
        except Exception as err:  # noqa: BLE001
            log.warning("ml_item_context items/%s exception: %s", mlb, err)
        return None

    async def _desc():
        try:
            r = await http.get(
                f"{ML_API_BASE}/items/{mlb}/description",
                headers=headers, timeout=10.0,
            )
            if r.status_code == 200:
                d = r.json() or {}
                return (d.get("plain_text") or d.get("text") or "")[:DESCRIPTION_MAX_CHARS]
        except Exception as err:  # noqa: BLE001
            log.warning("ml_item_context desc/%s exception: %s", mlb, err)
        return ""

    item, desc = await asyncio.gather(_item(), _desc())
    if not item:
        return None

    attrs: list[dict[str, str]] = []
    for a in (item.get("attributes") or [])[:30]:
        name = a.get("name") or a.get("id") or ""
        val = a.get("value_name")
        if not val and a.get("values"):
            vv = a["values"][0] if a["values"] else None
            val = (vv or {}).get("name")
        if name and val:
            attrs.append({"name": str(name), "value": str(val)})

    pictures: list[dict[str, str]] = []
    for p in (item.get("pictures") or [])[:6]:
        pid = p.get("id")
        if not pid:
            continue
        pictures.append({
            "id": str(pid),
            "url": str(p.get("url") or ""),
            "secure_url": str(p.get("secure_url") or ""),
        })

    sub_status_raw = item.get("sub_status") or []
    if not isinstance(sub_status_raw, list):
        sub_status_raw = []
    sub_status = [str(s) for s in sub_status_raw if s]

    sh = item.get("shipping") or {}
    return {
        "item_id": mlb,
        "title": item.get("title") or "",
        "condition": item.get("condition"),
        "price": item.get("price"),
        "currency": item.get("currency_id") or "BRL",
        "available_quantity": item.get("available_quantity"),
        "warranty": item.get("warranty"),
        "shipping_free": sh.get("free_shipping"),
        "logistic_type": sh.get("logistic_type"),
        "permalink": item.get("permalink"),
        "status": item.get("status"),
        "sub_status": sub_status,
        "attributes": attrs,
        "description": desc or "",
        "pictures": pictures,
    }


# ── DB upsert / get ───────────────────────────────────────────────────────────

UPSERT_SQL = """
INSERT INTO ml_item_context (
  user_id, item_id, title, condition, price, currency, available_quantity,
  warranty, shipping_free, logistic_type, permalink, status, sub_status,
  attributes, description, pictures, fetched_at
)
VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb,
  $14::jsonb, $15, $16::jsonb, NOW()
)
ON CONFLICT (user_id, item_id) DO UPDATE SET
  title = EXCLUDED.title,
  condition = EXCLUDED.condition,
  price = EXCLUDED.price,
  currency = EXCLUDED.currency,
  available_quantity = EXCLUDED.available_quantity,
  warranty = EXCLUDED.warranty,
  shipping_free = EXCLUDED.shipping_free,
  logistic_type = EXCLUDED.logistic_type,
  permalink = EXCLUDED.permalink,
  status = EXCLUDED.status,
  sub_status = EXCLUDED.sub_status,
  attributes = EXCLUDED.attributes,
  description = EXCLUDED.description,
  pictures = EXCLUDED.pictures,
  fetched_at = NOW();
"""


async def upsert(pool: asyncpg.Pool, user_id: int, data: dict[str, Any]) -> None:
    price = data.get("price")
    avail = data.get("available_quantity")
    async with pool.acquire() as conn:
        await conn.execute(
            UPSERT_SQL,
            user_id,
            data["item_id"],
            data.get("title") or None,
            data.get("condition"),
            float(price) if isinstance(price, (int, float)) else None,
            data.get("currency"),
            int(avail) if isinstance(avail, int) else None,
            data.get("warranty"),
            data.get("shipping_free"),
            data.get("logistic_type"),
            data.get("permalink"),
            data.get("status"),
            json.dumps(data.get("sub_status") or []),
            json.dumps(data.get("attributes") or []),
            data.get("description") or "",
            json.dumps(data.get("pictures") or []),
        )


async def get_cached(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
) -> Optional[dict[str, Any]]:
    """Read row from cache. Returns None if absent. Does NOT check TTL —
    callers that care use get_or_refresh()."""
    mlb = normalize_item_id(item_id)
    if not mlb:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT item_id, title, condition, price, currency, available_quantity,
                   warranty, shipping_free, logistic_type, permalink, status, sub_status,
                   attributes, description, pictures, fetched_at
              FROM ml_item_context
             WHERE user_id = $1 AND item_id = $2
            """,
            user_id, mlb,
        )
    if not row:
        return None
    attrs = row["attributes"]
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:  # noqa: BLE001
            attrs = []
    if not isinstance(attrs, list):
        attrs = []
    pictures = row["pictures"]
    if isinstance(pictures, str):
        try:
            pictures = json.loads(pictures)
        except Exception:  # noqa: BLE001
            pictures = []
    if not isinstance(pictures, list):
        pictures = []
    sub_status = row["sub_status"]
    if isinstance(sub_status, str):
        try:
            sub_status = json.loads(sub_status)
        except Exception:  # noqa: BLE001
            sub_status = []
    if not isinstance(sub_status, list):
        sub_status = []
    return {
        "item_id": row["item_id"],
        "title": row["title"] or "",
        "condition": row["condition"],
        "price": float(row["price"]) if row["price"] is not None else None,
        "currency": row["currency"] or "BRL",
        "available_quantity": row["available_quantity"],
        "warranty": row["warranty"],
        "shipping_free": row["shipping_free"],
        "logistic_type": row["logistic_type"],
        "permalink": row["permalink"],
        "status": row["status"],
        "sub_status": sub_status,
        "attributes": attrs,
        "description": row["description"] or "",
        "pictures": pictures,
        "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
    }


async def get_or_refresh(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
    item_id: str,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    access_token: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Cache-first: serve from DB if present and fresh; else refetch + upsert.

    Returns:
      - dict with item context (cached or freshly fetched)
      - None if ML returned nothing AND nothing in cache (item invalid/deleted)
    """
    mlb = normalize_item_id(item_id)
    if not mlb:
        return None

    cached = await get_cached(pool, user_id, mlb)
    if cached:
        # Check TTL
        fetched = cached.get("fetched_at")
        if fetched:
            try:
                ts = datetime.fromisoformat(fetched.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc) - ts
                if age < timedelta(hours=ttl_hours):
                    return cached
            except Exception:  # noqa: BLE001
                pass

    # Cache miss or stale — fetch fresh
    if access_token is None:
        try:
            access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
        except Exception as err:  # noqa: BLE001
            log.warning("ml_item_context oauth fail user=%s: %s", user_id, err)
            return cached  # serve stale if we have it

    if not access_token:
        return cached

    fresh = await fetch_from_ml(http, access_token, mlb)
    if fresh:
        await upsert(pool, user_id, fresh)
        # Re-read so caller gets the fetched_at + same shape as cached
        return await get_cached(pool, user_id, mlb)
    return cached  # serve stale on ML error


# ── Bulk refresh ──────────────────────────────────────────────────────────────

async def refresh_user_context(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
    item_ids: list[str],
    access_token: Optional[str] = None,
) -> dict[str, int]:
    """Throttled bulk refresh. Returns {fetched, saved, failed} counts."""
    if not item_ids:
        return {"fetched": 0, "saved": 0, "failed": 0}

    if access_token is None:
        try:
            access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
        except Exception as err:  # noqa: BLE001
            log.warning("ml_item_context bulk oauth fail user=%s: %s", user_id, err)
            return {"fetched": 0, "saved": 0, "failed": len(item_ids)}

    if not access_token:
        return {"fetched": 0, "saved": 0, "failed": len(item_ids)}

    saved = 0
    failed = 0
    seen: set[str] = set()
    for raw_id in item_ids:
        mlb = normalize_item_id(raw_id)
        if not mlb or mlb in seen:
            continue
        seen.add(mlb)
        try:
            data = await fetch_from_ml(http, access_token, mlb)
            if data:
                await upsert(pool, user_id, data)
                saved += 1
            else:
                failed += 1
        except Exception as err:  # noqa: BLE001
            log.warning("ml_item_context bulk %s exception: %s", mlb, err)
            failed += 1
        await asyncio.sleep(RATE_SLEEP)

    return {"fetched": len(seen), "saved": saved, "failed": failed}
