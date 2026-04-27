"""Items catalog cache — ML `/users/{id}/items/search` + `/items?ids=...`.

Caches the user's full ML listing catalog as a flat table. TTL=6h — listings
change slowly (price/status/stock edits). Full payload preserved as JSONB
so callers don't need new columns whenever ML adds a field.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
RATE_SLEEP = 0.2
BATCH_SIZE = 20  # /items?ids= accepts up to 20 per call


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_user_items (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  title TEXT,
  price NUMERIC,
  currency_id TEXT,
  status TEXT,
  thumbnail TEXT,
  permalink TEXT,
  sold_quantity INTEGER,
  available_quantity INTEGER,
  listing_type_id TEXT,
  category_id TEXT,
  health NUMERIC,
  shipping_json JSONB,
  raw JSONB,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_user_items_user ON ml_user_items(user_id);
CREATE INDEX IF NOT EXISTS idx_ml_user_items_status ON ml_user_items(user_id, status);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── ML API ────────────────────────────────────────────────────────────────────

async def _search_item_ids(http: httpx.AsyncClient, token: str, ml_user_id: int, status: str) -> list[str]:
    """Page through /users/{id}/items/search?status=X with scroll until empty."""
    ids: list[str] = []
    offset = 0
    limit = 100
    while True:
        url = f"{ML_API_BASE}/users/{ml_user_id}/items/search?status={status}&limit={limit}&offset={offset}"
        try:
            r = await http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20.0)
        except Exception as err:  # noqa: BLE001
            log.warning("items/search page failed: %s", err)
            break
        if r.status_code != 200:
            log.warning("items/search status %s: %s", r.status_code, r.text[:200])
            break
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            break
        ids.extend([str(x) for x in results])
        paging = data.get("paging") or {}
        total = paging.get("total", 0)
        offset += limit
        if offset >= total:
            break
        await asyncio.sleep(RATE_SLEEP)
    return ids


async def _fetch_batch(http: httpx.AsyncClient, token: str, item_ids: list[str]) -> list[dict]:
    # attributes + variations + seller_custom_field нужны чтобы вытащить SELLER_SKU
    # (для item без вариаций — в attributes[]/seller_custom_field, для вариаций —
    # в variations[].attributes[] и variations[].attribute_combinations[]).
    # Используется в /finance/sku-mapping чтобы дать ML-ссылку SKU без продаж.
    attrs = (
        "id,title,price,currency_id,status,thumbnail,sold_quantity,available_quantity,"
        "listing_type_id,shipping,category_id,health,permalink,"
        "attributes,variations,seller_custom_field"
    )
    url = f"{ML_API_BASE}/items?ids={','.join(item_ids)}&attributes={attrs}"
    try:
        r = await http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    except Exception as err:  # noqa: BLE001
        log.warning("items batch failed: %s", err)
        return []
    if r.status_code != 200:
        return []
    data = r.json() or []
    out: list[dict] = []
    for entry in data:
        if isinstance(entry, dict):
            body = entry.get("body") or entry
            if isinstance(body, dict) and body.get("id"):
                out.append(body)
    return out


# ── Refresh ───────────────────────────────────────────────────────────────────

async def refresh_user_items(
    pool: asyncpg.Pool,
    user_id: int,
    status: str = "active",
) -> dict[str, int]:
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("refresh_user_items token failed: %s", err)
        return {"fetched": 0, "saved": 0, "failed": 0}

    token_row = await ml_oauth_svc.load_user_tokens(pool, user_id) or {}
    ml_user_id = token_row.get("ml_user_id")
    if not ml_user_id:
        return {"fetched": 0, "saved": 0, "failed": 0, "error": "no_ml_user_id"}

    saved = 0
    failed = 0
    async with httpx.AsyncClient() as http:
        item_ids = await _search_item_ids(http, token, int(ml_user_id), status)
        if not item_ids:
            return {"fetched": 0, "saved": 0, "failed": 0}

        for i in range(0, len(item_ids), BATCH_SIZE):
            batch = item_ids[i:i + BATCH_SIZE]
            details = await _fetch_batch(http, token, batch)
            for item in details:
                try:
                    async with pool.acquire() as conn:
                        await _upsert(conn, user_id, item)
                    saved += 1
                except Exception as err:  # noqa: BLE001
                    log.warning("upsert item %s failed: %s", item.get("id"), err)
                    failed += 1
            await asyncio.sleep(RATE_SLEEP)

    # Сбросить sync кеш MLB→seller_sku — он используется в get_project_by_sku
    # как fallback для vendas-строк без SKU. Стейл после рефреша.
    try:
        from v2.legacy.config import invalidate_mlb_to_sku_from_ml_items
        invalidate_mlb_to_sku_from_ml_items()
    except Exception:
        pass

    return {"fetched": len(item_ids), "saved": saved, "failed": failed}


async def _upsert(conn: asyncpg.Connection, user_id: int, item: dict) -> None:
    shipping = item.get("shipping") or {}
    await conn.execute(
        """
        INSERT INTO ml_user_items
          (user_id, item_id, title, price, currency_id, status, thumbnail, permalink,
           sold_quantity, available_quantity, listing_type_id, category_id, health,
           shipping_json, raw, fetched_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15::jsonb, NOW())
        ON CONFLICT (user_id, item_id) DO UPDATE SET
          title = EXCLUDED.title,
          price = EXCLUDED.price,
          currency_id = EXCLUDED.currency_id,
          status = EXCLUDED.status,
          thumbnail = EXCLUDED.thumbnail,
          permalink = EXCLUDED.permalink,
          sold_quantity = EXCLUDED.sold_quantity,
          available_quantity = EXCLUDED.available_quantity,
          listing_type_id = EXCLUDED.listing_type_id,
          category_id = EXCLUDED.category_id,
          health = EXCLUDED.health,
          shipping_json = EXCLUDED.shipping_json,
          raw = EXCLUDED.raw,
          fetched_at = NOW()
        """,
        user_id,
        str(item.get("id") or ""),
        item.get("title"),
        float(item.get("price")) if item.get("price") is not None else None,
        item.get("currency_id"),
        item.get("status"),
        item.get("thumbnail"),
        item.get("permalink"),
        int(item.get("sold_quantity") or 0),
        int(item.get("available_quantity") or 0),
        item.get("listing_type_id"),
        item.get("category_id"),
        float(item.get("health")) if item.get("health") is not None else None,
        json.dumps(shipping, default=str),
        json.dumps(item, default=str),
    )


# ── Cache readback ────────────────────────────────────────────────────────────

async def get_cached(
    pool: asyncpg.Pool,
    user_id: int,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    where = "WHERE user_id = $1"
    params: list[Any] = [user_id]
    if status and status != "all":
        where += " AND status = $2"
        params.append(status)
    # Total count
    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM ml_user_items {where}", *params)
        params_paged = params + [limit, offset]
        rows = await conn.fetch(
            f"""
            SELECT item_id, title, price, currency_id, status, thumbnail, permalink,
                   sold_quantity, available_quantity, listing_type_id, category_id, health,
                   shipping_json,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_items
              {where}
             ORDER BY sold_quantity DESC NULLS LAST, item_id
             LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """,
            *params_paged,
        )
    items = []
    for r in rows:
        shipping = r["shipping_json"]
        if isinstance(shipping, str):
            shipping = json.loads(shipping or "{}")
        items.append({
            "id": r["item_id"],
            "title": r["title"],
            "price": float(r["price"]) if r["price"] is not None else None,
            "currency_id": r["currency_id"],
            "status": r["status"],
            "thumbnail": r["thumbnail"],
            "permalink": r["permalink"],
            "sold_quantity": int(r["sold_quantity"] or 0),
            "available_quantity": int(r["available_quantity"] or 0),
            "listing_type_id": r["listing_type_id"],
            "category_id": r["category_id"],
            "health": float(r["health"]) if r["health"] is not None else None,
            "shipping": shipping,
        })
    return {
        "total": int(total or 0),
        "items": items,
        "fetchedAt": rows[0]["fetched_at"] if rows else None,
    }


async def get_latest_fetched_at(pool: asyncpg.Pool, user_id: int) -> str | None:
    async with pool.acquire() as conn:
        r = await conn.fetchval(
            """
            SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"')
              FROM ml_user_items
             WHERE user_id = $1
            """,
            user_id,
        )
    return r
