"""Read-only API for Product Ads v2 data.

Reads from the `ml_ad_*` cache tables. When a table is empty for the caller,
falls back to a live ML sync before answering so the first page view after
connecting ML doesn't show an empty state.

Gated by `require_tier("escalar")` — paywall handled by the frontend.
"""
from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

log = logging.getLogger(__name__)

from v2.deps import CurrentUser, current_user, get_pool, require_admin, require_tier
from v2.schemas.ads import (
    AdCampaign,
    AdItem,
    AdMetrics,
    AdsOut,
    Advertiser,
    AdvertisersOut,
    CampaignDetail,
    CampaignMetrics,
    CampaignsOut,
    SyncResult,
)
from v2.services import ml_ads
from v2.services import ml_ads_sync
from v2.services import ml_oauth as ml_oauth_svc
from v2.storage import ads_storage

router = APIRouter(
    prefix="/ads",
    tags=["ads"],
    dependencies=[Depends(require_tier("escalar"))],
)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _metrics_from_raw(raw: Optional[dict], cls) -> "CampaignMetrics | AdMetrics":
    if not raw:
        return cls()
    # Swallow unknown keys — ML occasionally sneaks new fields into responses.
    known = set(cls.model_fields.keys())
    return cls(**{k: v for k, v in raw.items() if k in known})


def _to_campaign(row: dict) -> AdCampaign:
    return AdCampaign(
        id=int(row["campaign_id"]),
        product_id=row.get("product_id") or "PADS",
        name=row.get("name"),
        status=row.get("status"),
        strategy=row.get("strategy"),
        budget=float(row["budget"]) if row.get("budget") is not None else None,
        automatic_budget=row.get("automatic_budget"),
        roas_target=float(row["roas_target"]) if row.get("roas_target") is not None else None,
        channel=row.get("channel"),
        advertiser_id=int(row["advertiser_id"]),
        date_created=_iso(row.get("date_created")),
        last_updated=_iso(row.get("last_updated")),
        start_date=_iso(row.get("start_date")),
        end_date=_iso(row.get("end_date")),
        campaign_type=row.get("campaign_type"),
        goal=row.get("goal"),
        site_id=row.get("site_id"),
        headline=row.get("headline"),
        cpc=float(row["cpc"]) if row.get("cpc") is not None else None,
        currency=row.get("currency"),
        official_store_id=int(row["official_store_id"]) if row.get("official_store_id") is not None else None,
        destination_id=int(row["destination_id"]) if row.get("destination_id") is not None else None,
        metrics=_metrics_from_raw(row.get("metrics"), CampaignMetrics),
        metrics_date_from=row["metrics_date_from"].isoformat() if row.get("metrics_date_from") else None,
        metrics_date_to=row["metrics_date_to"].isoformat() if row.get("metrics_date_to") else None,
        synced_at=_iso(row.get("synced_at")),
    )


def _to_ad(row: dict) -> AdItem:
    return AdItem(
        item_id=row["item_id"],
        advertiser_id=int(row["advertiser_id"]),
        campaign_id=int(row["campaign_id"]) if row.get("campaign_id") is not None else None,
        title=row.get("title"),
        status=row.get("status"),
        price=float(row["price"]) if row.get("price") is not None else None,
        thumbnail=row.get("thumbnail"),
        permalink=row.get("permalink"),
        domain_id=row.get("domain_id"),
        brand_value_name=row.get("brand_value_name"),
        metrics=_metrics_from_raw(row.get("metrics"), AdMetrics),
        metrics_date_from=row["metrics_date_from"].isoformat() if row.get("metrics_date_from") else None,
        metrics_date_to=row["metrics_date_to"].isoformat() if row.get("metrics_date_to") else None,
    )


def _stale(synced_at: Optional[datetime]) -> bool:
    if synced_at is None:
        return True
    age = (datetime.now(timezone.utc) - synced_at).total_seconds()
    return age > ml_ads_sync.STALE_THRESHOLD_SECONDS


def _handle_ml_error(err: ml_ads.MLAdsError) -> None:
    """Translate ML statuses to frontend-actionable HTTP errors."""
    if err.status == 401:
        raise HTTPException(status_code=428, detail="ml_token_invalid")
    if err.status == 404 and "permissions" in (err.message or "").lower():
        raise HTTPException(status_code=404, detail="pads_not_enabled")
    raise HTTPException(status_code=502, detail=f"ml_error: {err.message[:200]}")


def _safe_call_label(label: str, advertiser_id: int, user_id: int) -> str:
    return f"{label}(advertiser={advertiser_id}, user={user_id})"


def _explain_unhandled(err: Exception, label: str) -> HTTPException:
    """Convert any unexpected exception into a structured 500.

    Without this the Next.js proxy gets `Internal Server Error` plain
    text from Starlette and we have to dig through Railway logs to
    learn what actually broke. With this, the response carries the
    error type + message + the first 1500 chars of the traceback so
    the diagnostic loop closes in one round instead of three.
    """
    log.exception("[ads] %s unhandled exception", label)
    detail = {
        "error": "ads_internal",
        "label": label,
        "exception_type": type(err).__name__,
        "message": str(err)[:500],
        "traceback": traceback.format_exc()[-1500:],
    }
    return HTTPException(status_code=500, detail=detail)


# ── Advertisers ───────────────────────────────────────────────────────────

@router.get("/advertisers", response_model=AdvertisersOut)
async def get_advertisers(
    product_type: str = Query("PADS", pattern="^(PADS|DISPLAY|BADS|ALL)$"),
    refresh: bool = Query(False),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Return advertisers for this user filtered by product type.

    `product_type=PADS|DISPLAY|BADS` filters to a single ML product type.
    `product_type=ALL` returns all rows. On cold cache or `?refresh=1`,
    hits ML `/advertising/advertisers?product_id=...` synchronously and
    populates the per-type rows (DISPLAY/BADS skip drill-down sync —
    they only have advertisers + campaigns list endpoints today)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")

    filter_pid = None if product_type == "ALL" else product_type

    cached = await ads_storage.list_advertisers(pool, user.id, product_id=filter_pid)
    if cached and not refresh:
        return AdvertisersOut(advertisers=[Advertiser(**r) for r in cached])

    # Live fallback — also covers first-time connection.
    try:
        if product_type == "ALL":
            await ml_ads.sync_user_advertisers_all_types(pool, user.id)
        else:
            await ml_ads.sync_user_advertisers(pool, user.id, product_id=product_type)
    except ml_oauth_svc.MLRefreshError as err:
        raise HTTPException(status_code=428, detail=f"ml_oauth_required: {err}")
    except ml_ads.MLAdsError as err:
        _handle_ml_error(err)

    fresh = await ads_storage.list_advertisers(pool, user.id, product_id=filter_pid)
    return AdvertisersOut(advertisers=[Advertiser(**r) for r in fresh])


# ── Campaigns ─────────────────────────────────────────────────────────────

@router.get("/campaigns", response_model=CampaignsOut)
async def get_campaigns(
    advertiser_id: int = Query(..., ge=1),
    site_id: str = Query(..., min_length=3, max_length=4),
    product_type: str = Query("PADS", pattern="^(PADS|DISPLAY|BADS)$"),
    refresh: bool = Query(False),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """List campaigns + rolled-up metrics for the advertiser, scoped to one
    ML product type. Triggers a sync if the cache is stale or empty.

    PADS goes through the legacy /product_ads/campaigns/search path; DISPLAY
    uses /display/campaigns; BADS uses /brand_ads/campaigns."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")

    label = _safe_call_label("get_campaigns", advertiser_id, user.id)
    try:
        synced_at = await ads_storage.campaign_staleness(
            pool, user.id, advertiser_id, product_id=product_type,
        )
        should_sync = refresh or synced_at is None or _stale(synced_at)

        if should_sync:
            try:
                if product_type == "PADS":
                    # PADS still uses the dedicated ensure_fresh helper which
                    # also pulls ads — preserves existing behaviour.
                    await ml_ads_sync.ensure_fresh_for_advertiser(
                        pool, user.id, advertiser_id, site_id,
                        max_age_seconds=0 if refresh else ml_ads_sync.STALE_THRESHOLD_SECONDS,
                    )
                else:
                    await ml_ads.sync_advertiser_campaigns_dispatch(
                        pool, user.id, advertiser_id, site_id, product_type,
                    )
            except ml_oauth_svc.MLRefreshError as err:
                raise HTTPException(status_code=428, detail=f"ml_oauth_required: {err}")
            except ml_ads.MLAdsError as err:
                _handle_ml_error(err)

        rows = await ads_storage.list_campaigns(
            pool, user.id, advertiser_id, product_id=product_type,
        )
        fresh_synced_at = await ads_storage.campaign_staleness(
            pool, user.id, advertiser_id, product_id=product_type,
        )
        return CampaignsOut(
            campaigns=[_to_campaign(r) for r in rows],
            total=len(rows),
            stale=_stale(fresh_synced_at),
            synced_at=_iso(fresh_synced_at),
        )
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        raise _explain_unhandled(err, label)


@router.get("/campaigns/{campaign_id}", response_model=CampaignDetail)
async def get_campaign_detail(
    campaign_id: int,
    daily: bool = Query(True),
    days: int = Query(30, ge=1, le=ml_ads.METRICS_MAX_DAYS),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")

    row = await ads_storage.get_campaign(pool, user.id, campaign_id)
    if not row:
        raise HTTPException(status_code=404, detail="campaign_not_found")

    base = _to_campaign(row)
    daily_rows: list[dict] = []
    if daily:
        dt_to = date.today()
        dt_from = dt_to - timedelta(days=days)
        daily_rows = await ads_storage.list_daily(
            pool, user.id, campaign_id, date_from=dt_from, date_to=dt_to,
        )
    return CampaignDetail(**base.model_dump(), daily=daily_rows)


# ── Ads ───────────────────────────────────────────────────────────────────

@router.get("/ads", response_model=AdsOut)
async def get_ads(
    advertiser_id: int = Query(..., ge=1),
    site_id: str = Query(..., min_length=3, max_length=4),
    campaign_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    refresh: bool = Query(False),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")

    label = _safe_call_label("get_ads", advertiser_id, user.id)
    try:
        synced_at = await ads_storage.campaign_staleness(pool, user.id, advertiser_id)
        should_sync = refresh or synced_at is None or _stale(synced_at)
        if should_sync:
            try:
                await ml_ads_sync.ensure_fresh_for_advertiser(
                    pool, user.id, advertiser_id, site_id,
                    max_age_seconds=0 if refresh else ml_ads_sync.STALE_THRESHOLD_SECONDS,
                )
            except ml_oauth_svc.MLRefreshError as err:
                raise HTTPException(status_code=428, detail=f"ml_oauth_required: {err}")
            except ml_ads.MLAdsError as err:
                _handle_ml_error(err)

        rows, total = await ads_storage.list_ads(
            pool, user.id,
            advertiser_id=advertiser_id,
            campaign_id=campaign_id,
            status=status,
            limit=limit, offset=offset,
        )
        fresh_synced_at = await ads_storage.campaign_staleness(pool, user.id, advertiser_id)
        return AdsOut(
            ads=[_to_ad(r) for r in rows],
            total=total,
            limit=limit,
            offset=offset,
            stale=_stale(fresh_synced_at),
            synced_at=_iso(fresh_synced_at),
        )
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        raise _explain_unhandled(err, label)


# ── Manual trigger (admin only) ───────────────────────────────────────────

@router.post("/sync", response_model=SyncResult)
async def trigger_sync(
    target_user_id: Optional[int] = Query(None),
    _admin: CurrentUser = Depends(require_admin),
    pool=Depends(get_pool),
):
    """Force a full sync for `target_user_id` (or the caller if omitted)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    uid = target_user_id if target_user_id is not None else _admin.id
    try:
        stats = await ml_ads.sync_user_full(pool, uid)
    except Exception as err:  # noqa: BLE001
        return SyncResult(status="error", user_id=uid, message=str(err)[:200])
    return SyncResult(
        status="ok",
        user_id=uid,
        advertisers=stats.advertisers,
        campaigns=stats.campaigns,
        daily_rows=stats.daily_rows,
        ads=stats.ads,
    )
