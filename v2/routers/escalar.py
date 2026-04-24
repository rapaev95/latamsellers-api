"""ABC products + snooze endpoints for the Escalar (Promotion) section."""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, Query

from v2.deps import CurrentUser, current_user, get_pool
from v2.legacy import db_storage as legacy_db
from v2.parsers import db_loader
from v2.schemas.escalar import EscalarProductsOut, SnoozeIn, SnoozeOut
from v2.services import abc, ml_backfill as ml_backfill_svc, ml_oauth as ml_oauth_svc, ml_quality as ml_quality_svc, ml_visits as ml_visits_svc, projects
from v2.settings import get_settings
from v2.storage import user_storage

import httpx

router = APIRouter(prefix="/escalar", tags=["escalar"])

SNOOZE_KEY = "escalar_snoozed_skus"


def _parse_days(raw: Optional[str]) -> Union[int, str]:
    if raw is None or raw == "":
        return 30
    if raw == "all":
        return "all"
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return 30


@router.get("/products", response_model=EscalarProductsOut)
async def get_products(
    days: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    import logging as _lg, time as _time
    _log = _lg.getLogger("escalar.products")
    _t0 = _time.perf_counter()
    _step = lambda name: _log.info("  [%5.2fs] %s", _time.perf_counter() - _t0, name)

    days_v = _parse_days(days)
    _step("after parse_days")
    # Bind user_id into legacy db_storage context-var — abc.aggregate calls
    # legacy.sku_catalog.load_catalog() for NCM/Origem/EAN/CSOSN, which is
    # per-user. Without this bind the fiscal fields come back empty.
    legacy_db.set_current_user_id(user.id)
    snoozed = set(await user_storage.get(pool, user.id, SNOOZE_KEY) or [])
    _step(f"after snoozed load ({len(snoozed)} items)")
    resolver = await projects.load_resolver(pool, user.id)
    _step("after projects resolver")

    vendas_rows = None
    storage_map = None
    stock_full_map = None
    vendas_filenames = None
    publicidade_rows = None
    if get_settings().storage_mode == "db" and pool is not None:
        vendas_rows = await db_loader.load_user_vendas(pool, user.id)
        _step(f"after load_user_vendas ({len(vendas_rows)} rows)")
        storage_map = await db_loader.load_user_armazenagem(pool, user.id)
        _step(f"after load_user_armazenagem ({len(storage_map)} skus)")
        stock_full_map = await db_loader.load_user_stock_full(pool, user.id)
        _step(f"after load_user_stock_full ({len(stock_full_map)} skus)")
        vendas_filenames = await db_loader.list_user_vendas_filenames(pool, user.id)
        _step(f"after list_vendas_filenames ({len(vendas_filenames)} files)")
        publicidade_rows = await db_loader.load_user_publicidade(pool, user.id)
        _step(f"after load_user_publicidade ({len(publicidade_rows)} rows)")

    summary = abc.aggregate(
        days=days_v,
        project=project or "",
        snoozed_skus=snoozed,
        resolver=resolver,
        vendas_rows=vendas_rows,
        storage_map=storage_map,
        stock_full_map=stock_full_map,
        vendas_filenames=vendas_filenames,
        publicidade_rows=publicidade_rows,
    )
    _step(f"after abc.aggregate (products={len(summary['products'])})")

    # Join cached listing quality (ml_item_quality) — fast dict lookup by itemId.
    quality_map: dict = {}
    latest_fetched_at = None
    if pool is not None:
        try:
            await ml_quality_svc.ensure_schema(pool)
            async with pool.acquire() as conn:
                quality_map = await ml_quality_svc.get_cached_map(conn, user.id)
                latest_fetched_at = await ml_quality_svc.get_latest_fetched_at(conn, user.id)
            _step(f"after quality map load ({len(quality_map)} items)")
        except Exception as err:  # noqa: BLE001
            _log.warning("quality map load failed: %s", err)

    products_out = summary["products"]
    quality_coverage = 0
    if quality_map:
        for p in products_out:
            key = ml_quality_svc.normalize_item_id(p.get("itemId"))
            if not key:
                continue
            q = quality_map.get(key)
            if not q:
                continue
            quality_coverage += 1
            p["qualityScore"] = q["score"]
            p["qualityLevel"] = q["level"]
            p["qualityStatus"] = q["status"]
            p["qualityCalculatedAt"] = q["calculatedAt"]
            p["qualityFetchedAt"] = q["fetchedAt"]
            p["warningsCount"] = q["warningsCount"]
            p["opportunitiesCount"] = q["opportunitiesCount"]
            p["warnings"] = q["warnings"]
            p["opportunities"] = q["opportunities"]

    # Join cached ML visits (ml_item_visits) — same dict-lookup pattern.
    visits_map: dict = {}
    visits_latest_fetched_at = None
    if pool is not None:
        try:
            await ml_visits_svc.ensure_schema(pool)
            async with pool.acquire() as conn:
                visits_map = await ml_visits_svc.get_cached_map(conn, user.id)
                visits_latest_fetched_at = await ml_visits_svc.get_latest_fetched_at(conn, user.id)
            _step(f"after visits map load ({len(visits_map)} items)")
        except Exception as err:  # noqa: BLE001
            _log.warning("visits map load failed: %s", err)

    visits_coverage = 0
    if visits_map:
        for p in products_out:
            key = ml_quality_svc.normalize_item_id(p.get("itemId"))
            if not key:
                continue
            v = visits_map.get(key)
            if not v:
                continue
            visits_coverage += 1
            p["visits7d"] = v["visits7d"]
            p["visits30d"] = v["visits30d"]
            p["visitsDaily"] = v["daily"]
            p["visitsFetchedAt"] = v["fetchedAt"]

    meta = dict(summary["meta"])
    meta["qualityFetchedAt"] = latest_fetched_at
    meta["qualityCoverage"] = quality_coverage
    meta["visitsFetchedAt"] = visits_latest_fetched_at
    meta["visitsCoverage"] = visits_coverage
    # Diagnostic — surfaces whether the DB cache actually has rows and whether
    # keys align with product itemIds. Temporary until visits UI is verified.
    sample_db_keys = list(visits_map.keys())[:3] if visits_map else []
    sample_product_keys = [
        ml_quality_svc.normalize_item_id(p.get("itemId"))
        for p in products_out[:3]
        if p.get("itemId")
    ]
    meta["visitsDebug"] = {
        "dbRows": len(visits_map),
        "sampleDbKeys": sample_db_keys,
        "sampleProductKeys": sample_product_keys,
        "productsWithItemId": sum(1 for p in products_out if p.get("itemId")),
    }
    summary_meta = meta

    return {
        "products": products_out,
        "hasData": len(products_out) > 0,
        "meta": summary_meta,
    }


@router.post("/snooze", response_model=SnoozeOut)
async def post_snooze(
    body: SnoozeIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    current = set(await user_storage.get(pool, user.id, SNOOZE_KEY) or [])
    if body.snoozed:
        current.add(body.sku)
    else:
        current.discard(body.sku)
    new_list = sorted(current)
    await user_storage.put(pool, user.id, SNOOZE_KEY, new_list)
    return {"snoozedSkus": new_list}


@router.post("/backfill-notices")
async def backfill_notices(
    days: int = Query(30, ge=1, le=90),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Pull last `days` of orders / questions / claims / paused-items / messages
    and upsert into ml_notices. Used by the «Sincronizar histórico» button and
    right after OAuth-connect."""
    async with httpx.AsyncClient() as http:
        result = await ml_backfill_svc.backfill_user(pool, http, user.id, days=days)
    return {"fetched": result["fetched"], "saved": result["saved"]}


@router.post("/notices/mark-all-read")
async def mark_all_notices_read(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Bulk-clear the queue: mark every pending notice as sent without
    actually sending it. Useful after a big backfill that filled the queue
    with old platform notices the user doesn't want flooding their TG.
    """
    if pool is None:
        return {"error": "db_pool_unavailable"}
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH cleared AS (
                UPDATE ml_notices
                   SET telegram_sent_at = NOW()
                 WHERE user_id = $1 AND telegram_sent_at IS NULL
                RETURNING 1
            )
            SELECT COUNT(*) AS n FROM cleared
            """,
            user.id,
        )
    return {"cleared": int(row["n"] if row else 0)}


@router.post("/notices/dispatch-now")
async def dispatch_notices_now(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manually trigger TG dispatch for current user — bypasses cron schedule.

    Diagnoses why messages aren't arriving: runs _dispatch_to_telegram with
    full exception visibility, returns counts + any error.
    """
    from v2.services import ml_notices as ml_notices_svc
    if pool is None:
        return {"error": "db_pool_unavailable"}

    # Pre-check: how many notices are actually pending for this user?
    async with pool.acquire() as conn:
        pending_row = await conn.fetchrow(
            "SELECT COUNT(*) AS n FROM ml_notices WHERE user_id = $1 AND telegram_sent_at IS NULL",
            user.id,
        )
        pending = int(pending_row["n"] if pending_row else 0)
        settings_row = await conn.fetchrow(
            """
            SELECT telegram_chat_id, notify_ml_news, COALESCE(language, 'pt') AS language
              FROM notification_settings WHERE user_id = $1
            """,
            user.id,
        )

    try:
        async with httpx.AsyncClient() as http:
            sent = await ml_notices_svc._dispatch_to_telegram(pool, http, user.id)
        return {
            "pending_before": pending,
            "sent": sent,
            "settings": dict(settings_row) if settings_row else None,
        }
    except Exception as err:  # noqa: BLE001
        import traceback
        return {
            "pending_before": pending,
            "sent": 0,
            "error": str(err),
            "traceback": traceback.format_exc()[:2000],
            "settings": dict(settings_row) if settings_row else None,
        }


@router.get("/products/quality-probe")
async def quality_probe(
    item_id: str = Query(..., min_length=1),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Diagnostic — call ML /item/{id}/performance with this user's token and
    return the RAW response (status + body) without touching the cache.

    Used to figure out why bulk refresh fails en masse (scope / 404 / etc).
    Remove once cache is stable; safe to keep since it's per-user auth-gated.
    """
    mlb = ml_quality_svc.normalize_item_id(item_id)
    if not mlb:
        return {"error": "invalid_item_id", "normalized": None, "raw_input": item_id}
    try:
        access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "token_refresh_failed", "detail": str(err)}

    url = f"https://api.mercadolibre.com/item/{mlb}/performance"
    async with httpx.AsyncClient() as http:
        try:
            r = await http.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=15.0)
        except Exception as err:  # noqa: BLE001
            return {"error": "network", "detail": str(err), "url": url, "normalized": mlb}

    return {
        "raw_input": item_id,
        "normalized": mlb,
        "url": url,
        "status": r.status_code,
        "headers_subset": {
            "content-type": r.headers.get("content-type"),
            "x-request-id": r.headers.get("x-request-id"),
        },
        "body": (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:1000]),
    }


@router.post("/products/refresh-quality")
async def refresh_products_quality(
    limit: int = Query(500, ge=1, le=1000),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Rebuild the ml_item_quality cache for this user's items.

    Triggered by the Next.js UI when the cache is missing or stale (>24h).
    Walks the user's product list (same ABC aggregator as /products), extracts
    unique itemIds and calls /item/{id}/performance for each (throttled).
    """
    # Build the same product list as /products to know which itemIds to refresh.
    snoozed = set(await user_storage.get(pool, user.id, SNOOZE_KEY) or [])
    resolver = await projects.load_resolver(pool, user.id)

    vendas_rows = None
    storage_map = None
    stock_full_map = None
    vendas_filenames = None
    if get_settings().storage_mode == "db" and pool is not None:
        vendas_rows = await db_loader.load_user_vendas(pool, user.id)
        storage_map = await db_loader.load_user_armazenagem(pool, user.id)
        stock_full_map = await db_loader.load_user_stock_full(pool, user.id)
        vendas_filenames = await db_loader.list_user_vendas_filenames(pool, user.id)

    summary = abc.aggregate(
        days="all",
        project="",
        snoozed_skus=snoozed,
        resolver=resolver,
        vendas_rows=vendas_rows,
        storage_map=storage_map,
        stock_full_map=stock_full_map,
        vendas_filenames=vendas_filenames,
    )
    item_ids = [p.get("itemId") for p in summary["products"] if p.get("itemId")]

    await ml_quality_svc.ensure_schema(pool)
    result = await ml_quality_svc.refresh_user_quality(pool, user.id, item_ids, limit=limit)
    return {
        "totalItems": len(item_ids),
        "fetched": result["fetched"],
        "saved": result["saved"],
        "failed": result["failed"],
        "skipped": result.get("skipped", 0),
        "statusCounts": result.get("status_counts", {}),
        "sampleErrors": result.get("sample_errors", []),
    }


@router.post("/products/refresh-visits")
async def refresh_products_visits(
    limit: int = Query(500, ge=1, le=1000),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Rebuild the ml_item_visits cache for this user's items.

    Mirrors refresh-quality: walks the same product list, calls ML
    `/items/{id}/visits/time_window?last=30&unit=day` per item (throttled)
    and upserts into ml_item_visits.
    """
    snoozed = set(await user_storage.get(pool, user.id, SNOOZE_KEY) or [])
    resolver = await projects.load_resolver(pool, user.id)

    vendas_rows = None
    storage_map = None
    stock_full_map = None
    vendas_filenames = None
    if get_settings().storage_mode == "db" and pool is not None:
        vendas_rows = await db_loader.load_user_vendas(pool, user.id)
        storage_map = await db_loader.load_user_armazenagem(pool, user.id)
        stock_full_map = await db_loader.load_user_stock_full(pool, user.id)
        vendas_filenames = await db_loader.list_user_vendas_filenames(pool, user.id)

    summary = abc.aggregate(
        days="all",
        project="",
        snoozed_skus=snoozed,
        resolver=resolver,
        vendas_rows=vendas_rows,
        storage_map=storage_map,
        stock_full_map=stock_full_map,
        vendas_filenames=vendas_filenames,
    )
    item_ids = [p.get("itemId") for p in summary["products"] if p.get("itemId")]

    await ml_visits_svc.ensure_schema(pool)
    result = await ml_visits_svc.refresh_user_visits(pool, user.id, item_ids, limit=limit)
    return {
        "totalItems": len(item_ids),
        "fetched": result["fetched"],
        "saved": result["saved"],
        "failed": result["failed"],
        "skipped": result.get("skipped", 0),
        "statusCounts": result.get("status_counts", {}),
        "sampleErrors": result.get("sample_errors", []),
    }
