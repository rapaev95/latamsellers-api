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
#
# Share-of-voice family (impression_share/top_*/lost_*/acos_benchmark) are
# available на campaign Detail endpoint (single campaign GET) per file
# ADS-PRODUCT-ML.js line 351. Bulk campaigns endpoint may reject some of
# these — keep them в FULL list, ML просто опустит unknown в response.
ML_METRICS_FULL = ",".join([
    "clicks", "prints", "ctr", "cost", "cpc", "acos", "roas", "cvr", "sov",
    "direct_amount", "indirect_amount", "total_amount",
    "direct_units_quantity", "indirect_units_quantity", "units_quantity",
    "direct_items_quantity", "indirect_items_quantity", "advertising_items_quantity",
    "organic_units_quantity", "organic_units_amount", "organic_items_quantity",
    # Share of Voice / impression-share family — для seller'ского UI
    # «exibido vs concorrência / não exibido por orçamento / não exibido
    # por classificação». ML returns 0..1 fractions (we render as %).
    "impression_share", "top_impression_share",
    "lost_impression_share_by_budget", "lost_impression_share_by_ad_rank",
    "acos_benchmark",
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


# ── DISPLAY product family ────────────────────────────────────────────────
# DISPLAY uses /advertising/advertisers/{id}/display/* (no site_id in path).
# Reference: developers.mercadolibre.* /api-display
# Different shape than PADS — campaigns don't ship metrics in the list, you
# need a separate /metrics call per campaign per date range (max 90d).

async def fetch_display_campaigns(
    access_token: str,
    advertiser_id: int,
    *,
    sort_by: str = "id",
    sort_order: str = "desc",
) -> list[dict]:
    data = await _get(
        access_token,
        f"/advertising/advertisers/{advertiser_id}/display/campaigns",
        params={"sort_by": sort_by, "sort_order": sort_order},
    )
    return list(data.get("results") or [])


async def fetch_display_campaign_metrics(
    access_token: str,
    advertiser_id: int,
    campaign_id: int,
    date_from: date,
    date_to: date,
) -> dict:
    """Returns {metrics: [{date, prints, clicks, ...}], summary: {...}}."""
    data = await _get(
        access_token,
        f"/advertising/advertisers/{advertiser_id}/display/campaigns/{campaign_id}/metrics",
        params={
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        },
    )
    return data or {}


# ── BADS (Brand Ads) family ───────────────────────────────────────────────
# BADS uses /advertising/advertisers/{id}/brand_ads/* (no site_id in path).
# Reference: developers.mercadolibre.* /api-brand-ads
# Campaigns ship items+keywords in the list. Metrics in separate calls.

async def fetch_bads_campaigns(access_token: str, advertiser_id: int) -> list[dict]:
    data = await _get(
        access_token,
        f"/advertising/advertisers/{advertiser_id}/brand_ads/campaigns",
    )
    # BADS uses "campaigns" key; PADS used "results". Accept both for safety.
    return list(data.get("campaigns") or data.get("results") or [])


async def fetch_bads_campaign_metrics(
    access_token: str,
    advertiser_id: int,
    campaign_id: int,
    date_from: date,
    date_to: date,
    *,
    aggregation_type: str = "daily",
) -> dict:
    """Returns {paging, metrics: [{date, prints, clicks, ...}], summary}."""
    data = await _get(
        access_token,
        f"/advertising/advertisers/{advertiser_id}/brand_ads/campaigns/{campaign_id}/metrics",
        params={
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "aggregation_type": aggregation_type,
        },
    )
    return data or {}


async def fetch_bads_advertiser_metrics_aggregate(
    access_token: str,
    advertiser_id: int,
    date_from: date,
    date_to: date,
) -> dict:
    """Aggregate metrics across all of one advertiser's BADS campaigns."""
    data = await _get(
        access_token,
        f"/advertising/advertisers/{advertiser_id}/brand_ads/campaigns/metrics",
        params={
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "aggregation_type": "daily",
        },
    )
    return data or {}


# ── Orchestration (sync to DB) ─────────────────────────────────────────────

@dataclass
class SyncStats:
    advertisers: int = 0
    campaigns: int = 0
    daily_rows: int = 0
    ads: int = 0


async def sync_user_advertisers(
    pool: asyncpg.Pool, user_id: int,
    product_id: str = "PADS",
) -> list[dict]:
    """Sync advertisers for one product type (PADS / DISPLAY / BADS).
    Returns the raw advertisers list. Caller may iterate all 3 product types
    via sync_user_advertisers_all_types()."""
    bearer, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    try:
        advertisers = await fetch_advertisers(bearer, product_id=product_id)
    except MLAdsError as err:
        # 404 "No permissions found for user_id" → user hasn't enabled this
        # product type (or no Publicidade access at all). Treat as empty.
        if err.status == 404:
            log.info(f"[ml-ads] user={user_id} has no {product_id} permission")
            return []
        raise
    await ads_storage.upsert_advertisers(pool, user_id, advertisers, product_id=product_id)
    return advertisers


async def sync_user_advertisers_all_types(
    pool: asyncpg.Pool, user_id: int,
) -> dict[str, list[dict]]:
    """Sync advertisers across all 3 ML product types.
    Returns {product_id: [advertiser, ...]} for caller diagnostics."""
    out: dict[str, list[dict]] = {}
    for ptype in ("PADS", "DISPLAY", "BADS"):
        try:
            out[ptype] = await sync_user_advertisers(pool, user_id, product_id=ptype)
        except (ml_oauth_svc.MLRefreshError, MLAdsError) as err:
            log.warning(f"[ml-ads] user={user_id} {ptype} sync failed: {err}")
            out[ptype] = []
    return out


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


def _flatten_summary_for_ui(summary: dict) -> dict:
    """DISPLAY and BADS metrics nest several core fields inside event_time/
    touch_point. Our existing Pydantic CampaignMetrics expects them at the
    root (so the same UI row-renderer works for all 3 types). Flatten the
    'event_time' branch (purchases attributed to the action date — closer
    to PADS semantics) up to root-level keys our schema knows about."""
    if not isinstance(summary, dict):
        return {}
    out = dict(summary)
    src = summary.get("event_time")
    if isinstance(src, dict):
        # Common: roas, units_quantity, direct_amount.
        for key in ("roas", "units_quantity", "direct_amount"):
            if key in src and key not in out:
                out[key] = src[key]
        # DISPLAY uses direct_amount as revenue. Map to total_amount used by UI.
        if "direct_amount" in src and "total_amount" not in out:
            out["total_amount"] = src["direct_amount"]
    # DISPLAY uses consumed_budget; PADS uses cost. Normalize to cost.
    if "consumed_budget" in out and "cost" not in out:
        out["cost"] = out["consumed_budget"]
    return out


async def sync_display_advertiser_campaigns(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> SyncStats:
    """Pull DISPLAY campaigns + per-day metrics for one advertiser, upsert.
    DISPLAY metrics endpoint requires per-campaign date_from/date_to calls."""
    bearer, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    dt_to = date.today()
    dt_from = dt_to - timedelta(days=lookback_days)

    campaigns = await fetch_display_campaigns(bearer, advertiser_id)
    # Fetch + attach summary metrics so the campaigns table has something
    # to display without a separate UI round-trip.
    enriched: list[dict] = []
    daily_rows_total = 0
    for c in campaigns:
        cid = c.get("id") or c.get("campaign_id")
        if cid is None:
            enriched.append(c)
            continue
        try:
            metrics_data = await fetch_display_campaign_metrics(
                bearer, advertiser_id, int(cid), dt_from, dt_to,
            )
        except MLAdsError as err:
            log.warning(f"[ml-ads:DISPLAY] metrics fetch failed campaign={cid}: {err}")
            metrics_data = {}
        # Store summary on the campaign row (re-using the existing `metrics` JSONB column).
        c2 = dict(c)
        c2["metrics"] = _flatten_summary_for_ui(metrics_data.get("summary") or {})
        enriched.append(c2)
        # Per-day metrics — same daily table as PADS so the UI can re-use chart code.
        daily = list((metrics_data or {}).get("metrics") or [])
        if daily:
            await ads_storage.upsert_daily_metrics(
                pool, user_id, advertiser_id, int(cid), daily,
            )
            daily_rows_total += len(daily)

    await ads_storage.upsert_campaign_snapshot(
        pool, user_id, advertiser_id, enriched, dt_from, dt_to,
        product_id="DISPLAY",
    )
    return SyncStats(campaigns=len(campaigns), daily_rows=daily_rows_total)


async def sync_bads_advertiser_campaigns(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> SyncStats:
    """Pull BADS campaigns + per-campaign metrics for one advertiser."""
    bearer, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    dt_to = date.today()
    dt_from = dt_to - timedelta(days=lookback_days)

    campaigns = await fetch_bads_campaigns(bearer, advertiser_id)
    enriched: list[dict] = []
    daily_rows_total = 0
    for c in campaigns:
        cid = c.get("campaign_id") or c.get("id")
        if cid is None:
            enriched.append(c)
            continue
        try:
            metrics_data = await fetch_bads_campaign_metrics(
                bearer, advertiser_id, int(cid), dt_from, dt_to,
            )
        except MLAdsError as err:
            log.warning(f"[ml-ads:BADS] metrics fetch failed campaign={cid}: {err}")
            metrics_data = {}
        c2 = dict(c)
        c2["metrics"] = _flatten_summary_for_ui(metrics_data.get("summary") or {})
        enriched.append(c2)
        daily = list((metrics_data or {}).get("metrics") or [])
        if daily:
            await ads_storage.upsert_daily_metrics(
                pool, user_id, advertiser_id, int(cid), daily,
            )
            daily_rows_total += len(daily)

    await ads_storage.upsert_campaign_snapshot(
        pool, user_id, advertiser_id, enriched, dt_from, dt_to,
        product_id="BADS",
    )
    return SyncStats(campaigns=len(campaigns), daily_rows=daily_rows_total)


async def sync_advertiser_campaigns_dispatch(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    site_id: str,
    product_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> SyncStats:
    """Single entrypoint for the router — picks the right sync path per product."""
    if product_id == "PADS":
        # PADS already syncs ads alongside campaigns via the existing sync_advertiser_campaigns,
        # but the standalone _ads call lives in the original sync_user_full flow. For
        # cold-cache router calls (GET /campaigns) we just refresh PADS campaigns.
        return await sync_advertiser_campaigns(
            pool, user_id, advertiser_id, site_id, lookback_days=lookback_days,
        )
    if product_id == "DISPLAY":
        return await sync_display_advertiser_campaigns(
            pool, user_id, advertiser_id, lookback_days=lookback_days,
        )
    if product_id == "BADS":
        return await sync_bads_advertiser_campaigns(
            pool, user_id, advertiser_id, lookback_days=lookback_days,
        )
    raise ValueError(f"unknown_product_id: {product_id}")


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
    """Top-level sync entry. Discovers advertisers across all 3 product types
    (PADS / DISPLAY / BADS) so the UI's product-type toggle has data.
    Drill-down (campaigns/daily/ads) only runs for PADS — DISPLAY/BADS use
    different URL patterns and are deferred to a per-type sync path."""
    stats = SyncStats()
    try:
        advertisers_by_type = await sync_user_advertisers_all_types(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.info(f"[ml-ads] user={user_id} no valid ML token: {err}")
        return stats

    pads_advertisers = advertisers_by_type.get("PADS", [])
    stats.advertisers = sum(len(v) for v in advertisers_by_type.values())

    # PADS-only drill-down — campaigns + ads from product_ads/* paths.
    for adv in pads_advertisers:
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
