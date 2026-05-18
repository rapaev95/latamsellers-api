"""Cached fetch of /categories/{id}/sale_terms.

`sale_terms` are a separate attribute set per category — same shape as
`/categories/{id}/attributes` but they go into the item body's `sale_terms`
array, not `attributes`. Common members: WARRANTY_TYPE, WARRANTY_TIME,
MANUFACTURING_TIME, INVOICE, PURCHASE_MAX_QUANTITY.

Cached 7 days; rarely changes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
SALE_TERMS_TTL = timedelta(days=7)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_category_sale_terms (
  category_id    TEXT PRIMARY KEY,
  sale_terms     JSONB NOT NULL,
  fetched_at     TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _is_stale(fetched_at: Optional[datetime], ttl: timedelta) -> bool:
    if fetched_at is None:
        return True
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_at > ttl


async def get_sale_terms(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    category_id: str,
    *,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {category_id, sale_terms, cache_hit, fetched_at}."""
    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT sale_terms, fetched_at FROM ml_category_sale_terms WHERE category_id = $1",
                category_id,
            )
        if row and not _is_stale(row["fetched_at"], SALE_TERMS_TTL):
            st = row["sale_terms"]
            if isinstance(st, str):
                st = json.loads(st)
            return {
                "category_id": category_id,
                "sale_terms": st,
                "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    r = await http.get(
        f"{ML_API_BASE}/categories/{category_id}/sale_terms",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_sale_terms_failed", request=r.request, response=r,
        )
    payload = r.json()
    sale_terms = payload if isinstance(payload, list) else []

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_category_sale_terms (category_id, sale_terms, fetched_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (category_id) DO UPDATE SET
                sale_terms = EXCLUDED.sale_terms,
                fetched_at = NOW()
            """,
            category_id, json.dumps(sale_terms),
        )
    return {
        "category_id": category_id,
        "sale_terms": sale_terms,
        "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
