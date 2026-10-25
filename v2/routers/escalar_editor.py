"""Escalar listing editor, one-click levers and Ads campaigns.

The screens a seller works in: open a listing, fill its technical sheet and
description, watch the quality score move, switch Product Ads on, join a
promotion, open a campaign and change its budget or ROAS target.

Every endpoint takes the user from the cookie and resolves the cabinet
through `ml_access`, so a partner acts on the owner's cabinet only for the
listings of the projects they were invited to, and only with a write role.
A listing outside the scope is 404, not 403 — its existence is not leaked.

Kept out of `escalar.py` on purpose: that module is 7k lines of mixed
webhook, probe and page code; this one is the user-facing write surface.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel

from v2.deps import CurrentUser, current_user, get_pool
from v2.services import ml_access
from v2.services import ml_category_attributes as cat_attrs_svc
from v2.services import ml_oauth as ml_oauth_svc
from v2.services import ml_quality as quality_svc
from v2.services import ml_user_promotions as promo_svc
from v2.storage import ads_storage

log = logging.getLogger(__name__)

router = APIRouter(prefix="/escalar", tags=["escalar-editor"])

ML = "https://api.mercadolibre.com"
ADS_HEADERS = {"api-version": "2"}


# ── plumbing ─────────────────────────────────────────────────────────────────

def _mlb(item_id: str) -> str:
    mlb = quality_svc.normalize_item_id(item_id)
    if not mlb:
        raise HTTPException(status_code=400, detail="bad_item_id")
    return mlb


async def _scope_for_item(pool, user, item_id: str, project: str, *, write: bool):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    mlb = _mlb(item_id)
    scope = await ml_access.resolve_scope(pool, user, project=project or "")
    if not scope.allows_read(mlb):
        raise HTTPException(status_code=404, detail="item_not_found")
    if write and not scope.allows_write(mlb):
        raise HTTPException(status_code=403, detail="read_only")
    return scope, mlb


async def _token(pool, scope: ml_access.MlScope) -> str:
    """Token of the cabinet the scope points at; callers resolved the scope."""
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, scope.owner_id)
    except ml_oauth_svc.MLRefreshError as err:
        raise HTTPException(status_code=409, detail={"code": "ml_oauth_required", "message": str(err)})
    return token


async def _ml(
    http: httpx.AsyncClient, method: str, path: str, token: str, *,
    json: Any = None, params: Optional[dict] = None, headers: Optional[dict] = None,
) -> tuple[int, Any]:
    try:
        r = await http.request(
            method, f"{ML}{path}", json=json, params=params, timeout=20.0,
            headers={"Authorization": f"Bearer {token}", **(headers or {})},
        )
    except Exception as err:  # noqa: BLE001
        return 0, {"message": f"network: {err}"}
    try:
        body = r.json() if r.content else {}
    except ValueError:
        body = {"message": r.text[:500]}
    return r.status_code, body


def _ml_error(status: int, body: Any) -> dict[str, Any]:
    """ML's reason, readable: message plus each `cause` line."""
    body = body if isinstance(body, dict) else {"message": str(body)}
    causes = []
    for c in body.get("cause") or []:
        if isinstance(c, dict):
            causes.append(c.get("message") or c.get("code") or str(c))
        else:
            causes.append(str(c))
    return {
        "ok": False,
        "status": status,
        "message": body.get("message") or body.get("error") or f"HTTP {status}",
        "causes": causes,
    }


async def _refresh_quality(pool, http, token, owner_id: int, mlb: str) -> Optional[dict]:
    row, _err = await quality_svc.fetch_one(http, token, mlb)
    if row is None:
        return None
    async with pool.acquire() as conn:
        await quality_svc._upsert_quality(conn, owner_id, row)
    return {
        "score": quality_svc._coerce_score(row.get("score")),
        "level": row.get("level"),
        "status": row.get("status"),
        "calculatedAt": row.get("calculated_at"),
        "warnings": row.get("warnings") or [],
        "opportunities": row.get("opportunities") or [],
    }


# ── listing editor ───────────────────────────────────────────────────────────

def _merge_attributes(category_attrs: list[dict], item_attrs: list[dict]) -> list[dict]:
    """Category attribute definitions with the listing's current value on each.

    Attributes set on the listing but not in the category list (ML keeps a few
    computed ones) are appended read-only so nothing the seller sees in ML
    goes missing here.
    """
    current = {a.get("id"): a for a in item_attrs or [] if a.get("id")}
    out = []
    for a in category_attrs or []:
        cur = current.pop(a.get("id"), None)
        out.append({
            **a,
            "current": (
                {"value_id": cur.get("value_id"), "value_name": cur.get("value_name")}
                if cur else None
            ),
        })
    for aid, cur in current.items():
        out.append({
            "id": aid,
            "name": cur.get("name"),
            "value_type": cur.get("value_type") or "string",
            "tags": {"read_only": True},
            "current": {"value_id": cur.get("value_id"), "value_name": cur.get("value_name")},
        })
    return out


def _has_value(cur: Optional[dict]) -> bool:
    if not cur:
        return False
    return bool(cur.get("value_id") or (cur.get("value_name") or "").strip())


def _completeness(attrs: list[dict]) -> dict[str, Any]:
    """How much of the technical sheet is filled — the part the seller controls."""
    visible = [
        a for a in attrs
        if not cat_attrs_svc.is_hidden(a) and not cat_attrs_svc.is_fixed(a)
        and not (a.get("tags") or {}).get("read_only")
        and not cat_attrs_svc.is_variation_attribute(a)
    ]
    required = [a for a in visible if cat_attrs_svc.is_required(a)]
    filled = [a for a in visible if _has_value(a.get("current"))]
    return {
        "total": len(visible),
        "filled": len(filled),
        "pct": round(len(filled) / len(visible) * 100) if visible else None,
        "requiredMissing": [a["id"] for a in required if not _has_value(a.get("current"))],
    }


@router.get("/items/{item_id}/editor")
async def get_item_editor(
    item_id: str,
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Everything the listing screen needs, live from ML."""
    scope, mlb = await _scope_for_item(pool, user, item_id, project or "", write=False)
    owner = scope.owner_id
    token = await _token(pool, scope)
    await cat_attrs_svc.ensure_schema(pool)
    await quality_svc.ensure_schema(pool)
    await ads_storage.ensure_schema(pool)

    async with httpx.AsyncClient() as http:
        (st_item, item), (st_desc, desc), quality, offers = await asyncio.gather(
            _ml(http, "GET", f"/items/{mlb}", token, params={"include_attributes": "all"}),
            _ml(http, "GET", f"/items/{mlb}/description", token),
            _refresh_quality(pool, http, token, owner, mlb),
            promo_svc._fetch_offers_for_item(http, token, mlb),
        )
        if st_item != 200:
            raise HTTPException(status_code=502, detail=_ml_error(st_item, item))
        category_id = item.get("category_id")
        category: dict[str, Any] = {}
        if category_id:
            try:
                category = await cat_attrs_svc.get_attributes(pool, http, token, category_id)
            except Exception as err:  # noqa: BLE001 — the sheet still opens read-only
                log.warning("editor %s: category %s attributes failed: %s", mlb, category_id, err)
        _st_cat, cat_info = await _ml(http, "GET", f"/categories/{category_id}", token) if category_id else (0, {})

    attributes = _merge_attributes(category.get("attributes") or [], item.get("attributes") or [])

    async with pool.acquire() as conn:
        ad = await conn.fetchrow(
            """
            SELECT a.advertiser_id, a.campaign_id, a.status, a.metrics, c.name AS campaign_name
              FROM ml_ad_ads a
              LEFT JOIN ml_ad_campaigns c
                ON c.user_id = a.user_id AND c.campaign_id = a.campaign_id
             WHERE a.user_id = $1 AND upper(a.item_id) = $2
             ORDER BY a.synced_at DESC
             LIMIT 1
            """,
            owner, mlb,
        )
        campaigns = await conn.fetch(
            """
            SELECT campaign_id, advertiser_id, name, status, budget, roas_target
              FROM ml_ad_campaigns
             WHERE user_id = $1 AND product_id = 'PADS' AND status <> 'deleted'
             ORDER BY name
            """,
            owner,
        )

    pictures = [
        {"id": p.get("id"), "url": p.get("secure_url") or p.get("url"), "size": p.get("size")}
        for p in item.get("pictures") or []
    ]
    return {
        "item": {
            "id": mlb,
            "title": item.get("title"),
            "price": item.get("price"),
            "originalPrice": item.get("original_price"),
            "currency": item.get("currency_id"),
            "status": item.get("status"),
            "subStatus": item.get("sub_status") or [],
            "tags": item.get("tags") or [],
            "health": item.get("health"),
            "stock": item.get("available_quantity"),
            "sold": item.get("sold_quantity"),
            "permalink": item.get("permalink"),
            "thumbnail": item.get("secure_thumbnail") or item.get("thumbnail"),
            "pictures": pictures,
            "categoryId": category_id,
            "categoryName": (cat_info or {}).get("name"),
            "domainId": item.get("domain_id"),
            "listingType": item.get("listing_type_id"),
            "condition": item.get("condition"),
            "catalogListing": item.get("catalog_listing"),
            "variations": len(item.get("variations") or []),
            "titleEditable": not (item.get("sold_quantity") or 0),
        },
        "description": (desc or {}).get("plain_text") if st_desc == 200 else "",
        "attributes": attributes,
        "completeness": _completeness(attributes),
        "quality": quality,
        "promotions": offers,
        "ads": {
            "current": (
                {
                    "advertiserId": ad["advertiser_id"],
                    "campaignId": ad["campaign_id"],
                    "campaignName": ad["campaign_name"],
                    "status": ad["status"],
                }
                if ad else None
            ),
            "campaigns": [
                {
                    "id": c["campaign_id"], "advertiserId": c["advertiser_id"],
                    "name": c["name"], "status": c["status"],
                    "budget": float(c["budget"]) if c["budget"] is not None else None,
                    "roasTarget": float(c["roas_target"]) if c["roas_target"] is not None else None,
                }
                for c in campaigns
            ],
        },
        "canEdit": scope.allows_write(mlb),
        "project": ml_access.attribution(owner).get(mlb),
    }


class _AttrIn(BaseModel):
    id: str
    value_id: Optional[str] = None
    value_name: Optional[str] = None


class _PictureIn(BaseModel):
    id: Optional[str] = None
    source: Optional[str] = None


class _ItemUpdateIn(BaseModel):
    title: Optional[str] = None
    attributes: Optional[list[_AttrIn]] = None
    pictures: Optional[list[_PictureIn]] = None
    description: Optional[str] = None


@router.put("/items/{item_id}")
async def update_item(
    item_id: str,
    body: _ItemUpdateIn,
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Save the listing: title, technical sheet, pictures, description.

    The item and its description are separate ML resources, so each part
    reports its own result — a rejected title must not hide a saved
    description. Returns the new quality score so the screen shows the effect.
    """
    scope, mlb = await _scope_for_item(pool, user, item_id, project or "", write=True)
    owner = scope.owner_id
    token = await _token(pool, scope)

    payload: dict[str, Any] = {}
    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="empty_title")
        payload["title"] = title
    if body.attributes is not None:
        payload["attributes"] = [
            {k: v for k, v in a.model_dump().items() if v is not None or k == "value_name"}
            for a in body.attributes
        ]
    if body.pictures is not None:
        payload["pictures"] = [
            {k: v for k, v in p.model_dump().items() if v} for p in body.pictures
        ]
        if not payload["pictures"]:
            raise HTTPException(status_code=400, detail="no_pictures")
    if not payload and body.description is None:
        raise HTTPException(status_code=400, detail="empty_payload")

    results: dict[str, Any] = {}
    async with httpx.AsyncClient() as http:
        if payload:
            st, resp = await _ml(http, "PUT", f"/items/{mlb}", token, json=payload)
            results["item"] = {"ok": True} if st in (200, 201) else _ml_error(st, resp)
            if st in (200, 201) and isinstance(resp, dict):
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE ml_user_items
                           SET title = coalesce($3, title),
                               thumbnail = coalesce($4, thumbnail)
                         WHERE user_id = $1 AND upper(item_id) = $2
                        """,
                        owner, mlb, resp.get("title"),
                        resp.get("secure_thumbnail") or resp.get("thumbnail"),
                    )
        if body.description is not None:
            st, resp = await _ml(
                http, "PUT", f"/items/{mlb}/description", token,
                json={"plain_text": body.description}, params={"api_version": "2"},
            )
            results["description"] = {"ok": True} if st in (200, 201) else _ml_error(st, resp)

        # The Q&A assistant reads the listing from this cache; drop it so the
        # next read fetches what was just saved.
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ml_item_context WHERE user_id = $1 AND upper(item_id) = $2",
                owner, mlb,
            )
        quality = await _refresh_quality(pool, http, token, owner, mlb)

    return {
        "ok": all(r.get("ok") for r in results.values()),
        "results": results,
        "quality": quality,
    }


# ── one-click levers: promotions and Product Ads ─────────────────────────────

class _PromoJoinIn(BaseModel):
    promotion_id: str
    promotion_type: str
    deal_price: Optional[float] = None
    top_deal_price: Optional[float] = None
    offer_id: Optional[str] = None


@router.post("/items/{item_id}/promotions")
async def join_promotion(
    item_id: str,
    body: _PromoJoinIn,
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    scope, mlb = await _scope_for_item(pool, user, item_id, project or "", write=True)
    token = await _token(pool, scope)
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    async with httpx.AsyncClient() as http:
        st, resp = await _ml(
            http, "POST", f"/seller-promotions/items/{mlb}", token,
            json=payload, params={"app_version": "v2"},
        )
        if st not in (200, 201):
            raise HTTPException(status_code=422, detail=_ml_error(st, resp))
        offers = await promo_svc._fetch_offers_for_item(http, token, mlb)
    await promo_svc.mark_accepted(pool, scope.owner_id, mlb, body.promotion_id)
    await promo_svc.refresh_user_promotions(pool, scope.owner_id, item_ids=[mlb])
    return {"ok": True, "promotions": offers}


@router.delete("/items/{item_id}/promotions")
async def leave_promotion(
    item_id: str,
    promotion_type: str = Query(...),
    promotion_id: str = Query(...),
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    scope, mlb = await _scope_for_item(pool, user, item_id, project or "", write=True)
    token = await _token(pool, scope)
    async with httpx.AsyncClient() as http:
        st, resp = await _ml(
            http, "DELETE", f"/seller-promotions/items/{mlb}", token,
            params={"app_version": "v2", "promotion_type": promotion_type,
                    "promotion_id": promotion_id},
        )
        if st not in (200, 201, 204):
            raise HTTPException(status_code=422, detail=_ml_error(st, resp))
        offers = await promo_svc._fetch_offers_for_item(http, token, mlb)
    await promo_svc.refresh_user_promotions(pool, scope.owner_id, item_ids=[mlb])
    return {"ok": True, "promotions": offers}


class _AdIn(BaseModel):
    status: str                      # 'active' | 'paused'
    campaign_id: Optional[int] = None


@router.put("/items/{item_id}/ads")
async def set_item_ad(
    item_id: str,
    body: _AdIn,
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Switch Product Ads on or off for one listing, optionally moving it
    into a campaign. ML: PUT /marketplace/advertising/{site}/product_ads/ads/{item}.
    """
    if body.status not in ("active", "paused"):
        raise HTTPException(status_code=400, detail="bad_status")
    scope, mlb = await _scope_for_item(pool, user, item_id, project or "", write=True)
    owner = scope.owner_id
    token = await _token(pool, scope)
    payload: dict[str, Any] = {"status": body.status}
    if body.campaign_id is not None:
        payload["campaign_id"] = int(body.campaign_id)
    site = "MLB"
    async with httpx.AsyncClient() as http:
        st, resp = await _ml(
            http, "PUT", f"/marketplace/advertising/{site}/product_ads/ads/{mlb}", token,
            json=payload, headers=ADS_HEADERS,
        )
    if st not in (200, 201, 204):
        raise HTTPException(status_code=422, detail=_ml_error(st, resp))
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ml_ad_ads
               SET status = $3, campaign_id = coalesce($4, campaign_id), synced_at = NOW()
             WHERE user_id = $1 AND upper(item_id) = $2
            """,
            owner, mlb, body.status, body.campaign_id,
        )
    return {"ok": True, "ml": resp}


# ── Ads campaigns ────────────────────────────────────────────────────────────

async def _campaign_items(pool, owner_id: int, campaign_id: int) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT upper(item_id) AS item_id FROM ml_ad_ads WHERE user_id = $1 AND campaign_id = $2",
            owner_id, campaign_id,
        )
    return [r["item_id"] for r in rows]


def _campaign_out(row: dict) -> dict[str, Any]:
    m = row.get("metrics") or {}
    return {
        "id": int(row["campaign_id"]),
        "advertiserId": int(row["advertiser_id"]),
        "name": row.get("name"),
        "status": row.get("status"),
        "strategy": row.get("strategy"),
        "budget": float(row["budget"]) if row.get("budget") is not None else None,
        "roasTarget": float(row["roas_target"]) if row.get("roas_target") is not None else None,
        "metrics": m,
        "metricsFrom": str(row["metrics_date_from"]) if row.get("metrics_date_from") else None,
        "metricsTo": str(row["metrics_date_to"]) if row.get("metrics_date_to") else None,
        "syncedAt": row["synced_at"].isoformat() if row.get("synced_at") else None,
    }


@router.get("/campaigns")
async def list_campaigns(
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Product Ads campaigns of the cabinet that carry the caller's listings.

    The owner without a project filter sees every campaign. Otherwise a
    campaign shows when at least one of its ads is a listing in scope.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    scope = await ml_access.resolve_scope(pool, user, project=project or "")
    if not scope.is_owner and not scope.projects:
        raise HTTPException(status_code=404, detail="project_not_found")
    owner = scope.owner_id
    await ads_storage.ensure_schema(pool)
    rows = await ads_storage.list_campaigns(pool, owner, product_id="PADS")
    async with pool.acquire() as conn:
        ad_rows = await conn.fetch(
            "SELECT campaign_id, upper(item_id) AS item_id, status FROM ml_ad_ads WHERE user_id = $1",
            owner,
        )
    by_campaign: dict[int, list[dict]] = {}
    for r in ad_rows:
        by_campaign.setdefault(int(r["campaign_id"] or 0), []).append(dict(r))

    out = []
    for row in rows:
        ads = by_campaign.get(int(row["campaign_id"]), [])
        mine = [a for a in ads if scope.allows_read(a["item_id"])]
        if scope.item_ids is not None and not mine:
            continue
        c = _campaign_out(row)
        c["ads"] = len(ads)
        c["adsInScope"] = len(mine)
        c["adsActive"] = sum(1 for a in mine if a["status"] == "active")
        out.append(c)
    out.sort(key=lambda c: (c["status"] != "active", -(c["metrics"].get("cost") or 0)))
    synced = max((c["syncedAt"] for c in out if c["syncedAt"]), default=None)
    return {"campaigns": out, "total": len(out), "syncedAt": synced}


@router.get("/campaigns/{campaign_id}")
async def get_campaign(
    campaign_id: int,
    days: int = Query(30, ge=1, le=90),
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    scope = await ml_access.resolve_scope(pool, user, project=project or "")
    owner = scope.owner_id
    row = await ads_storage.get_campaign(pool, owner, campaign_id)
    if not row:
        raise HTTPException(status_code=404, detail="campaign_not_found")
    items = await _campaign_items(pool, owner, campaign_id)
    if scope.item_ids is not None and not any(scope.allows_read(i) for i in items):
        raise HTTPException(status_code=404, detail="campaign_not_found")

    dt_to = date.today()
    daily = await ads_storage.list_daily(
        pool, owner, campaign_id, date_from=dt_to - timedelta(days=days), date_to=dt_to,
    )
    ads, _total = await ads_storage.list_ads(pool, owner, campaign_id=campaign_id, limit=500)
    ads_out = []
    for a in ads:
        iid = str(a.get("item_id") or "").upper()
        if not scope.allows_read(iid):
            continue
        ads_out.append({
            "itemId": iid,
            "title": a.get("title"),
            "thumbnail": a.get("thumbnail"),
            "permalink": a.get("permalink"),
            "price": float(a["price"]) if a.get("price") is not None else None,
            "status": a.get("status"),
            "metrics": a.get("metrics") or {},
            "canEdit": scope.allows_write(iid),
        })
    out = _campaign_out(row)
    out["daily"] = daily
    out["ads"] = ads_out
    out["adsHidden"] = len(items) - len(ads_out)
    out["canEdit"] = scope.may_write and all(scope.allows_write(i) for i in items)
    return out


class _CampaignUpdateIn(BaseModel):
    budget: Optional[float] = None
    status: Optional[str] = None          # 'active' | 'paused'
    roas_target: Optional[float] = None   # ML stores ROAS; the API takes acos_target = 100/ROAS
    strategy: Optional[str] = None        # 'profitability' | 'increase' | 'visibility'
    name: Optional[str] = None


@router.put("/campaigns/{campaign_id}")
async def update_campaign(
    campaign_id: int,
    body: _CampaignUpdateIn,
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Change budget, status, ROAS target or strategy of a campaign.

    A campaign spends on every listing in it, so a partner may change one
    only when they can write every listing it carries.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    scope = await ml_access.resolve_scope(pool, user, project=project or "")
    owner = scope.owner_id
    row = await ads_storage.get_campaign(pool, owner, campaign_id)
    if not row:
        raise HTTPException(status_code=404, detail="campaign_not_found")
    items = await _campaign_items(pool, owner, campaign_id)
    if scope.item_ids is not None and not any(scope.allows_read(i) for i in items):
        raise HTTPException(status_code=404, detail="campaign_not_found")
    if not scope.may_write or not all(scope.allows_write(i) for i in items):
        raise HTTPException(status_code=403, detail="read_only")

    payload: dict[str, Any] = {}
    if body.budget is not None:
        payload["budget"] = float(max(1.0, body.budget))
    if body.status is not None:
        if body.status not in ("active", "paused"):
            raise HTTPException(status_code=400, detail="bad_status")
        payload["status"] = body.status
    if body.roas_target is not None:
        if body.roas_target <= 0:
            raise HTTPException(status_code=400, detail="bad_roas_target")
        payload["acos_target"] = round(max(3.1, min(499.0, 100.0 / body.roas_target)), 2)
    if body.strategy is not None:
        if body.strategy.lower() not in ("profitability", "increase", "visibility"):
            raise HTTPException(status_code=400, detail="bad_strategy")
        payload["strategy"] = body.strategy.lower()
    if body.name is not None:
        payload["name"] = body.name.strip()[:60]
    if not payload:
        raise HTTPException(status_code=400, detail="empty_payload")

    token = await _token(pool, scope)
    site = (row.get("site_id") or "MLB").upper()
    async with httpx.AsyncClient() as http:
        st, resp = await _ml(
            http, "PUT", f"/marketplace/advertising/{site}/product_ads/campaigns/{campaign_id}",
            token, json=payload, headers=ADS_HEADERS,
        )
    if st not in (200, 201, 204):
        raise HTTPException(status_code=422, detail=_ml_error(st, resp))

    roas_saved = (100.0 / payload["acos_target"]) if "acos_target" in payload else None
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ml_ad_campaigns
               SET budget = coalesce($3, budget),
                   status = coalesce($4, status),
                   roas_target = coalesce($5, roas_target),
                   strategy = coalesce($6, strategy),
                   name = coalesce($7, name),
                   last_updated = NOW()
             WHERE user_id = $1 AND campaign_id = $2
            """,
            owner, campaign_id, payload.get("budget"), payload.get("status"),
            round(roas_saved, 2) if roas_saved else None,
            payload["strategy"].upper() if "strategy" in payload else None,
            payload.get("name"),
        )
    fresh = await ads_storage.get_campaign(pool, owner, campaign_id)
    return {"ok": True, "campaign": _campaign_out(fresh or row), "ml": resp}
