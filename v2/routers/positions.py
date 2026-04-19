"""Position-in-search endpoints for the Escalar Marketing page.

Calls ML's public `/sites/<site>/search` (no auth) via `services.positions`
and records every check into `position_history` for the authenticated user.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from v2.deps import CurrentUser, current_user, get_pool
from v2.schemas.escalar import (
    PositionCheckOut,
    TrackedKeyword,
    TrackedListOut,
    TrackKeywordIn,
)
from v2.services import ml_oauth as ml_oauth_svc
from v2.services import positions as positions_service
from v2.storage import positions_storage

router = APIRouter(prefix="/escalar/positions", tags=["escalar", "positions"])


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


@router.get("/check", response_model=PositionCheckOut)
async def check(
    item_id: str = Query(..., min_length=3),
    keyword: str = Query(..., min_length=1),
    site_id: str = Query("MLB"),
    category_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """One-shot check: search ML for `keyword`, scan up to 1000 results.

    Reads the user's ML bearer from Railway Postgres `ml_user_tokens`
    (`ml_oauth.get_valid_access_token` auto-refreshes if expiring). No
    Supabase path. If no token, returns 428 → UI deep-links to /escalar/settings.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    try:
        bearer, _expires, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        raise HTTPException(
            status_code=428,  # Precondition Required — "connect ML first"
            detail=f"ml_oauth_required: {err}",
        )

    # Use the seller_id stored at OAuth-exchange time. Avoids a round-trip to
    # /users/me (which returns 403 on non-Developer-Partner apps) and any
    # fragile parsing of the APP_USR token suffix.
    token_row = await ml_oauth_svc.load_user_tokens(pool, user.id) or {}
    seller_id = token_row.get("ml_user_id")

    try:
        result = await positions_service.check_position(
            item_id=item_id,
            keyword=keyword,
            site_id=site_id,
            category_id=category_id,
            bearer_token=bearer,
            seller_id=seller_id,
        )
    except positions_service.PositionCheckError as err:
        raise HTTPException(status_code=502, detail=f"ml_error: {err}")
    if pool is not None:
        await positions_storage.record_check(
            pool,
            user_id=user.id,
            item_id=result.item_id,
            keyword=result.keyword,
            site_id=result.site_id,
            position=result.position,
            total_results=result.total_results,
            found=result.found,
        )
    return PositionCheckOut(
        itemId=result.item_id,
        keyword=result.keyword,
        position=result.position,
        found=result.found,
        totalResults=result.total_results,
        pagesScanned=result.pages_scanned,
        siteId=result.site_id,
        adsAbove=result.ads_above,
    )


@router.post("/track", response_model=TrackedKeyword)
async def track(
    body: TrackKeywordIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    tracked_id = await positions_storage.add_tracked(
        pool,
        user_id=user.id,
        item_id=body.itemId,
        keyword=body.keyword,
        site_id=body.siteId,
        category_id=body.categoryId,
    )
    return TrackedKeyword(
        id=tracked_id,
        itemId=body.itemId,
        keyword=body.keyword,
        siteId=body.siteId,
        categoryId=body.categoryId,
        createdAt=datetime.now().isoformat(),
        lastPosition=None,
        lastCheckedAt=None,
        lastFound=None,
    )


@router.get("/tracked", response_model=TrackedListOut)
async def tracked(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return TrackedListOut(tracked=[])
    rows = await positions_storage.list_tracked(pool, user.id)
    return TrackedListOut(
        tracked=[
            TrackedKeyword(
                id=r["id"],
                itemId=r["item_id"],
                keyword=r["keyword"],
                siteId=r["site_id"],
                categoryId=r["category_id"],
                createdAt=_iso(r["created_at"]) or "",
                lastPosition=r["last_position"],
                lastCheckedAt=_iso(r["last_checked_at"]),
                lastFound=r["last_found"],
            )
            for r in rows
        ]
    )


@router.delete("/track/{tracked_id}")
async def untrack(
    tracked_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    ok = await positions_storage.delete_tracked(pool, user.id, tracked_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not_found")
    return {"deleted": True}
