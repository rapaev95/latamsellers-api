"""HTTP client for ML Product Ads v2 + orchestration helpers.

Base path: `https://api.mercadolibre.com/advertising/*`
Every call sends `api-version: 2` — required for the v2 suite.

Legacy endpoints (`/advertising/product_ads/*`) were decommissioned on
2026-02-26 — this module deliberately does not expose them.

Token handling is delegated to `v2.services.ml_oauth.get_valid_access_token`,
which auto-refreshes expiring tokens and is serialized by a per-user advisory
lock. We never re-implement refresh here.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Optional

import asyncpg
import httpx

from v2.services import ml_oauth as ml_oauth_svc
from v2.storage import ads_storage

log = logging.getLogger("ml-ads")

ML_BASE = "https://api.mercadolibre.com"

# One canonical metric list for all queries — keep every column populated so
# the cache matches what the docs promise, regardless of which UI tab the
# user opens next.
ML_METRICS_FULL = ",".join([
    "clicks", "prints", "ctr", "cost", "cpc", "acos", "roas", "cvr", "sov",
    "direct_amount", "indirect_amount", "total_amount",
    "direct_units_quantity", "indirect_units_quantity", "units_quantity",
    "direct_items_quantity", "indirect_items_quantity", "advertising_items_quantity",
    "organic_units_quantity", "organic_units_amount", "organic_items_quantity",
])

# Ad-level response drops ctr/cvr/roas/sov — keeping its list separate avoids
# ML returning 400 on unknown-for-endpoint metric names.
ML_METRICS_AD = ",".join([
    "clicks", "prints", "cost", "cpc", "acos",
    "direct_amount", "indirect_amount", "total_amount",
    "direct_units_quantity", "indirect_units_quantity", "units_quantity",
    "direct_items_quantity", "indirect_items_quantity", "advertising_items_quantity",
    "organic_units_quantity", "organic_items_quantity",
])

# Window ML supports for metrics queries.
METRICS_MAX_DAYS = 90

DEFAULT_LOOKBACK_DAYS = 30


class MLAdsError(Exception):
    """Raised on non-2xx ML API responses. Carries `status` so the router can
    translate 401/403/404 → typed HTTPException for the UI."""
    def __init__(self, status: int, message: str, *, payload: Any = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.payload = payload


# ── HTTP primitive ─────────────────────────────────────────────────────────

async def _get(
    access_token: str,
    path: str,
    params: Optional[dict] = None,
    *,
    timeout: float = 30.0,
) -> Any:
    url = f"{ML_BASE}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {access_token}",
                "api-version": "2",
                "Accept": "application/json",
            },
        )
    if r.status_code >= 400:
        try:
            payload = r.json()
        except ValueError:
            payload = {"raw": r.text[:400]}
        raise MLAdsError(r.status_code, str(payload)[:400], payload=payload)
    return r.json()


# ── Raw ML calls ───────────────────────────────────────────────────────────

async def fetch_advertisers(
    access_token: str, product_id: str = "PADS",
) -> list[dict]:
    data = await _get(
        access_token,
        "/advertising/advertisers",
        params={"product_id": product_id},
    )
    return list(data.get("advertisers") or [])


async def fetch_campaigns_search_page(
    access_token: str,
    site_id: str,
    advertiser_id: int,
    date_from: date,
    date_to: date,
    *,
    limit: int = 50,
    offset: int = 0,
    metrics_summary: bool = False,
) -> dict:
    params = {
        "limit": limit,
        "offset": offset,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "metrics": ML_METRICS_FULL,
    }
    if metrics_summary:
        params["metrics_summary"] = "true"
    return await _get(
        access_token,
        f"/advertising/{site_id}/advertisers/{advertiser_id}/product_ads/campaigns/search",
        params=params,
    )


async def fetch_all_campaigns(
    access_token: str,
    site_id: str,
    advertiser_id: int,
    date_from: date,
    date_to: date,
) -> tuple[list[dict], dict]:
    """Paginates through `campaigns/search`. Also returns `metrics_summary`
    (computed on the first page request with metrics_summary=true)."""
    page = await fetch_campaigns_search_page(
        access_token, site_id, advertiser_id, date_from, date_to,
        limit=50, offset=0, metrics_summary=True,
    )
    results: list[dict] = list(page.get("results") or [])
    total = int((page.get("paging") or {}).get("total") or len(results))
    metrics_summary = page.get("metrics_summary") or {}

    offset = len(results)
    while offset < total:
        nxt = await fetch_campaigns_search_page(
            access_token, site_id, advertiser_id, date_from, date_to,
            limit=50, offset=offset, metrics_summary=False,
        )
        batch = list(nxt.get("results") or [])
        if not batch:
            break
        results.extend(batch)
        offset += len(batch)
    return results, metrics_summary


async def fetch_campaign_daily(
    access_token: str,
    site_id: str,
    campaign_id: int,
    date_from: date,
    date_to: date,
) -> list[dict]:
    """Per-day metrics for a single campaign. ML returns a list — not paged."""
    data = await _get(
        access_token,
        f"/advertising/{site_id}/product_ads/campaigns/{campaign_id}",
        params={
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "metrics": ML_METRICS_FULL,
            "aggregation_type": "DAILY",
        },
    )
    # Daily returns either a bare list or {results: [...]} depending on edge
    # cases — normalise.
    if isinstance(data, list):
        return data
    return list((data or {}).get("results") or [])


async def fetch_ads_search_page(
    access_token: str,
    site_id: str,
    advertiser_id: int,
    date_from: date,
    date_to: date,
    *,
    limit: int = 50,
    offset: int = 0,
    filters: Optional[dict] = None,
) -> dict:
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "metrics": ML_METRICS_AD,
    }
    for k, v in (filters or {}).items():
        if v is None or v == "":
            continue
        params[f"filters[{k}]"] = v
    return await _get(
        access_token,
        f"/advertising/{site_id}/advertisers/{advertiser_id}/product_ads/ads/search",
        params=params,
    )


async def fetch_all_ads(
    access_token: str,
    site_id: str,
    advertiser_id: int,
    date_from: date,
    date_to: date,
    *,
    filters: Optional[dict] = None,
    hard_cap: int = 500,
) -> list[dict]:
    """Paged iterator over `ads/search`. `hard_cap` bounds the sync cost —
    a large advertiser with 10k ads shouldn't spam ML every hour. UI-side
    ads listing uses paginated DB reads; this limit only affects how much
    cache we populate on each sync tick."""
    first = await fetch_ads_search_page(
        access_token, site_id, advertiser_id, date_from, date_to,
        limit=50, offset=0, filters=filters,
    )
    results: list[dict] = list(first.get("results") or [])
    total = int((first.get("paging") or {}).get("total") or len(results))
    total = min(total, hard_cap)
    offset = len(results)
    while offset < total:
        nxt = await fetch_ads_search_page(
            access_token, site_id, advertiser_id, date_from, date_to,
            limit=50, offset=offset, filters=filters,
        )
        batch = list(nxt.get("results") or [])
        if not batch:
            break
        results.extend(batch)
        offset += len(batch)
    return results


# ── Orchestration (sync to DB) ─────────────────────────────────────────────

@dataclass
class SyncStats:
    advertisers: int = 0
    campaigns: int = 0
    daily_rows: int = 0
    ads: int = 0


async def sync_user_advertisers(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict]:
    bearer, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    try:
        advertisers = await fetch_advertisers(bearer, product_id="PADS")
    except MLAdsError as err:
        # 404 "No permissions found for user_id" → user hasn't enabled Publicidade
        if err.status == 404:
            log.info(f"[ml-ads] user={user_id} has no PADS permission")
            return []
        raise
    await ads_storage.upsert_advertisers(pool, user_id, advertisers)
    return advertisers


async def sync_advertiser_campaigns(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    site_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> SyncStats:
    bearer, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    dt_to = date.today()
    dt_from = dt_to - timedelta(days=lookback_days)

    campaigns, _summary = await fetch_all_campaigns(
        bearer, site_id, advertiser_id, dt_from, dt_to,
    )
    await ads_storage.upsert_campaign_snapshot(
        pool, user_id, advertiser_id, campaigns, dt_from, dt_to,
    )

    # Daily granularity per campaign. Sequential is fine — ML throttles
    # parallel requests anyway, and sync runs off the hot path.
    daily_rows_total = 0
    for c in campaigns:
        cid = c.get("id")
        if cid is None:
            continue
        try:
            daily = await fetch_campaign_daily(
                bearer, site_id, int(cid), dt_from, dt_to,
            )
        except MLAdsError as err:
            log.warning(f"[ml-ads] daily fetch failed campaign={cid}: {err}")
            continue
        await ads_storage.upsert_daily_metrics(
            pool, user_id, advertiser_id, int(cid), daily,
        )
        daily_rows_total += len(daily)

    return SyncStats(campaigns=len(campaigns), daily_rows=daily_rows_total)


async def sync_advertiser_ads(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    site_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> int:
    bearer, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    dt_to = date.today()
    dt_from = dt_to - timedelta(days=lookback_days)
    ads = await fetch_all_ads(bearer, site_id, advertiser_id, dt_from, dt_to)
    await ads_storage.upsert_ads_snapshot(
        pool, user_id, advertiser_id, ads, dt_from, dt_to,
    )
    return len(ads)


async def sync_user_full(
    pool: asyncpg.Pool, user_id: int,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> SyncStats:
    """Top-level sync entry — advertisers → campaigns+daily → ads."""
    stats = SyncStats()
    try:
        advertisers = await sync_user_advertisers(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.info(f"[ml-ads] user={user_id} no valid ML token: {err}")
        return stats
    except MLAdsError as err:
        log.warning(f"[ml-ads] user={user_id} advertisers fetch failed: {err}")
        return stats
    stats.advertisers = len(advertisers)

    for adv in advertisers:
        advertiser_id = int(adv["advertiser_id"])
        site_id = adv.get("site_id") or "MLB"
        try:
            camp_stats = await sync_advertiser_campaigns(
                pool, user_id, advertiser_id, site_id, lookback_days=lookback_days,
            )
            stats.campaigns += camp_stats.campaigns
            stats.daily_rows += camp_stats.daily_rows
            stats.ads += await sync_advertiser_ads(
                pool, user_id, advertiser_id, site_id, lookback_days=lookback_days,
            )
        except MLAdsError as err:
            log.warning(
                f"[ml-ads] user={user_id} adv={advertiser_id} sync failed: {err}"
            )
            continue
    return stats
