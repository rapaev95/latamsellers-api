"""Physical Full warehouse stock cache via ML `/inventories/{id}/stock/fulfillment`.

Why this exists: `/items/{id}.available_quantity` returns allocated-to-listing
stock (e.g. 133 for one MLB), но физический сток в Full-складе обычно выше
(168-172) и shared между несколькими MLB-листингами. Раньше пользователь
вручную скачивал XLSX из ML UI; этот сервис заменяет ручной экспорт API-pull'ом.

Shape of /inventories/{inventory_id}/stock/fulfillment:
  {
    "inventory_id": "DDQN26485",
    "total": 169,
    "available_quantity": 168,
    "not_available_quantity": 1,
    "not_available_detail": [{"status": "lost", "quantity": 1}],
    "external_references": [
      {"type": "item", "id": "MLB6143605452", "variation_id": null},
      {"type": "item", "id": "MLB6143748308", "variation_id": null}
    ]
  }

`external_references` критичен: один inventory_id может быть привязан к
нескольким MLB (shared listings — обычное дело когда seller делает два
объявления на один и тот же товар). Сохраняем mapping чтобы lookup
по любому MLB давал правильный физический сток.

Pattern follows ml_visits.py / ml_quality.py.
Pt-BR docs: https://developers.mercadolivre.com.br/pt_br/estoque-em-mercado-envios-full
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
RATE_SLEEP = 0.2  # 5 rps per user — matches ml_quality/visits


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_full_inventory (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  inventory_id TEXT NOT NULL,
  total INTEGER NOT NULL DEFAULT 0,
  available_quantity INTEGER NOT NULL DEFAULT 0,
  not_available_quantity INTEGER NOT NULL DEFAULT 0,
  not_available_detail JSONB DEFAULT '[]'::jsonb,
  external_refs JSONB DEFAULT '[]'::jsonb,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, inventory_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_full_inv_user ON ml_full_inventory(user_id);

-- Per-MLB reverse lookup (flattened from external_refs). Один row per (user,mlb)
-- даже если MLB шарит inventory_id с другим листингом.
CREATE TABLE IF NOT EXISTS ml_full_inventory_by_mlb (
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  inventory_id TEXT NOT NULL,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_full_inv_by_mlb_inv
  ON ml_full_inventory_by_mlb(user_id, inventory_id);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── ML API ─────────────────────────────────────────────────────────────────────

async def _fetch_fulfillment_stock(
    http: httpx.AsyncClient, token: str, inventory_id: str,
) -> Optional[dict[str, Any]]:
    """GET /inventories/{inventory_id}/stock/fulfillment. Returns parsed body
    or None on error/403/404. Does not raise."""
    try:
        r = await http.get(
            f"{ML_API_BASE}/inventories/{inventory_id}/stock/fulfillment",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("ML inventory stock fetch %s failed: %s", inventory_id, err)
        return None
    if r.status_code != 200:
        log.warning(
            "ML inventory stock %s → %s: %s",
            inventory_id, r.status_code, r.text[:200],
        )
        return None
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return None


async def _list_user_fulfillment_inventory_ids(
    pool: asyncpg.Pool, user_id: int, http: httpx.AsyncClient, token: str,
) -> dict[str, str]:
    """Return {mlb: inventory_id} for всех Full items этого юзера.

    Источник:
    1. ml_user_items.raw.inventory_id если присутствует
    2. Иначе live fetch /items/{id} per item (только если logistic_type=fulfillment)

    Дедуплицирует inventory_id'ы — один inv может быть на несколько MLB.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT item_id,
                   raw->>'inventory_id' as cached_inv_id,
                   raw->'shipping'->>'logistic_type' as logistic_type
              FROM ml_user_items
             WHERE user_id = $1
               AND raw->'shipping'->>'logistic_type' = 'fulfillment'
            """,
            user_id,
        )

    mlb_to_inv: dict[str, str] = {}
    items_to_resolve: list[str] = []
    for r in rows:
        item_id = r["item_id"]
        cached = (r["cached_inv_id"] or "").strip()
        if cached:
            mlb_to_inv[item_id] = cached
        else:
            items_to_resolve.append(item_id)

    # Live-fetch inventory_id для тех items где наш кэш не содержит его.
    # Обычно это происходит если ml_user_items был fetched старым кодом.
    for item_id in items_to_resolve:
        try:
            r = await http.get(
                f"{ML_API_BASE}/items/{item_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"attributes": "id,inventory_id,shipping"},
                timeout=10.0,
            )
        except Exception as err:  # noqa: BLE001
            log.debug("inventory_id resolve %s failed: %s", item_id, err)
            continue
        if r.status_code != 200:
            continue
        try:
            d = r.json()
        except Exception:  # noqa: BLE001
            continue
        inv = (d.get("inventory_id") or "").strip()
        if inv:
            mlb_to_inv[item_id] = inv
        await asyncio.sleep(RATE_SLEEP)

    return mlb_to_inv


# ── Public API ─────────────────────────────────────────────────────────────────

async def refresh_user_full_inventory(
    pool: asyncpg.Pool, user_id: int,
) -> dict[str, int]:
    """Refresh ml_full_inventory + ml_full_inventory_by_mlb for the user.

    Returns {"inventories": N, "mlbs": M, "failed": K}.
    """
    await ensure_schema(pool)
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("full inventory refresh user=%s skipped: %s", user_id, err)
        return {"inventories": 0, "mlbs": 0, "failed": 0, "skipped": "no_token"}

    async with httpx.AsyncClient(timeout=30.0) as http:
        mlb_to_inv = await _list_user_fulfillment_inventory_ids(
            pool, user_id, http, token,
        )
        # Unique inventory_ids — shared listings share one.
        unique_invs = list(set(mlb_to_inv.values()))
        ok = 0
        failed = 0
        async with pool.acquire() as conn:
            for inv_id in unique_invs:
                body = await _fetch_fulfillment_stock(http, token, inv_id)
                await asyncio.sleep(RATE_SLEEP)
                if not body:
                    failed += 1
                    continue
                ext_refs = body.get("external_references") or []
                await conn.execute(
                    """
                    INSERT INTO ml_full_inventory
                      (user_id, inventory_id, total, available_quantity,
                       not_available_quantity, not_available_detail,
                       external_refs, fetched_at)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, NOW())
                    ON CONFLICT (user_id, inventory_id) DO UPDATE SET
                       total = EXCLUDED.total,
                       available_quantity = EXCLUDED.available_quantity,
                       not_available_quantity = EXCLUDED.not_available_quantity,
                       not_available_detail = EXCLUDED.not_available_detail,
                       external_refs = EXCLUDED.external_refs,
                       fetched_at = NOW()
                    """,
                    user_id, inv_id,
                    int(body.get("total") or 0),
                    int(body.get("available_quantity") or 0),
                    int(body.get("not_available_quantity") or 0),
                    json.dumps(body.get("not_available_detail") or []),
                    json.dumps(ext_refs),
                )
                # Refresh reverse map for every MLB in external_refs
                for ref in ext_refs:
                    if not isinstance(ref, dict):
                        continue
                    if (ref.get("type") or "").lower() != "item":
                        continue
                    mlb_ref = ref.get("id")
                    if not mlb_ref:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO ml_full_inventory_by_mlb
                          (user_id, item_id, inventory_id, fetched_at)
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (user_id, item_id) DO UPDATE SET
                           inventory_id = EXCLUDED.inventory_id,
                           fetched_at = NOW()
                        """,
                        user_id, mlb_ref, inv_id,
                    )
                ok += 1

    return {
        "inventories": ok,
        "mlbs": len(mlb_to_inv),
        "failed": failed,
    }


async def get_total_by_mlb(
    pool: asyncpg.Pool, user_id: int, item_id: str,
) -> Optional[dict[str, Any]]:
    """Return {"total", "available_quantity", "not_available_quantity",
    "inventory_id", "fetched_at"} for given MLB, or None if not cached.

    Used by ml_inventory_forecast._get_stock as priority source (physical
    Full stock) before falling back to /items API or XLSX upload.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT fi.inventory_id, fi.total, fi.available_quantity,
                   fi.not_available_quantity, fi.fetched_at
              FROM ml_full_inventory_by_mlb m
              JOIN ml_full_inventory fi
                ON fi.user_id = m.user_id AND fi.inventory_id = m.inventory_id
             WHERE m.user_id = $1 AND m.item_id = $2
            """,
            user_id, item_id,
        )
    if not row:
        return None
    return {
        "inventory_id": row["inventory_id"],
        "total": int(row["total"]),
        "available_quantity": int(row["available_quantity"]),
        "not_available_quantity": int(row["not_available_quantity"]),
        "fetched_at": row["fetched_at"],
    }
