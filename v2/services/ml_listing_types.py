"""ML listing types + commission rates per category.

Two ML endpoints contribute:
- /users/{seller_id}/available_listing_types?category_id=X — which listing
  types are usable for this seller in this category (some are blocked,
  e.g. `free` after exceeding a transaction limit).
- /sites/{site_id}/listing_types/{id} — details for one listing type
  including sale_fee_criteria (commission %, currency etc).

We assemble both into a single cached read so the UI can render Premium
vs Clássico comparison cards without N round-trips.
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
LISTING_TYPES_TTL = timedelta(hours=24)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_listing_types_per_category (
  site_id            TEXT NOT NULL,
  category_id        TEXT NOT NULL,
  ml_user_id         BIGINT NOT NULL,
  available          JSONB NOT NULL,
  details_by_id      JSONB NOT NULL,
  fetched_at         TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (site_id, category_id, ml_user_id)
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


async def get_listing_types(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    *,
    site_id: str,
    category_id: str,
    ml_user_id: int,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns merged shape:
       { site_id, category_id, ml_user_id,
         types: [{ id, name, available, remaining_listings?, commission_pct?,
                   max_fee_amount?, currency?, listing_exposure?, ... }],
         cache_hit, fetched_at }
    """
    site_id = site_id.upper()

    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT available, details_by_id, fetched_at
                FROM ml_listing_types_per_category
                WHERE site_id = $1 AND category_id = $2 AND ml_user_id = $3
                """,
                site_id, category_id, ml_user_id,
            )
        if row and not _is_stale(row["fetched_at"], LISTING_TYPES_TTL):
            return _merge(
                site_id, category_id, ml_user_id,
                row["available"], row["details_by_id"], row["fetched_at"], True,
            )

    # Fetch available types for this seller+category.
    r = await http.get(
        f"{ML_API_BASE}/users/{ml_user_id}/available_listing_types",
        params={"category_id": category_id},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_available_listing_types_failed", request=r.request, response=r,
        )
    payload = r.json() or {}
    available = payload.get("available") or []

    # Fetch details for each type — site-level, not seller-level.
    details_by_id: dict[str, dict[str, Any]] = {}
    for item in available:
        lid = item.get("id")
        if not lid or lid in details_by_id:
            continue
        rd = await http.get(
            f"{ML_API_BASE}/sites/{site_id}/listing_types/{lid}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15.0,
        )
        if rd.status_code == 200:
            details_by_id[lid] = rd.json() or {}

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_listing_types_per_category (
                site_id, category_id, ml_user_id, available, details_by_id, fetched_at
            )
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, NOW())
            ON CONFLICT (site_id, category_id, ml_user_id) DO UPDATE SET
                available = EXCLUDED.available,
                details_by_id = EXCLUDED.details_by_id,
                fetched_at = NOW()
            """,
            site_id, category_id, ml_user_id,
            json.dumps(available), json.dumps(details_by_id),
        )
    return _merge(
        site_id, category_id, ml_user_id,
        available, details_by_id,
        datetime.now(timezone.utc), False,
    )


def _merge(
    site_id: str, category_id: str, ml_user_id: int,
    available: Any, details_by_id: Any,
    fetched_at: datetime, cache_hit: bool,
) -> dict[str, Any]:
    if isinstance(available, str):
        available = json.loads(available)
    if isinstance(details_by_id, str):
        details_by_id = json.loads(details_by_id)

    types: list[dict[str, Any]] = []
    for av in available or []:
        lid = av.get("id")
        det = (details_by_id or {}).get(lid) or {}
        cfg = det.get("configuration") or {}
        fee = cfg.get("sale_fee_criteria") or {}
        types.append({
            "id": lid,
            "name": av.get("name") or cfg.get("name") or lid,
            "available": True,
            "remaining_listings": av.get("remaining_listings"),
            "commission_pct": fee.get("percentage_of_fee_amount"),
            "min_fee_amount": fee.get("min_fee_amount"),
            "max_fee_amount": fee.get("max_fee_amount"),
            "currency": fee.get("currency"),
            "listing_exposure": cfg.get("listing_exposure"),
            "duration_days": cfg.get("duration_days"),
            "requires_picture": cfg.get("requires_picture"),
            "max_stock_per_item": cfg.get("max_stock_per_item"),
        })

    return {
        "site_id": site_id,
        "category_id": category_id,
        "ml_user_id": ml_user_id,
        "types": types,
        "cache_hit": cache_hit,
        "fetched_at": fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
    }
