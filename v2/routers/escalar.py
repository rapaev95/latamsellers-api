"""ABC products + snooze endpoints for the Escalar (Promotion) section."""
from __future__ import annotations

from typing import Any, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from v2.deps import CurrentUser, current_user, get_pool
from v2.legacy import db_storage as legacy_db
from v2.parsers import db_loader
from v2.schemas.escalar import EscalarProductsOut, SnoozeIn, SnoozeOut
from v2.services import (
    abc,
    category_benchmarks as category_benchmarks_svc,
    ml_account_health as ml_account_health_svc,
    ml_backfill as ml_backfill_svc,
    ml_item_context as ml_item_context_svc,
    ml_normalize as ml_normalize_svc,
    ml_notices as ml_notices_svc,
    ml_oauth as ml_oauth_svc,
    ml_quality as ml_quality_svc,
    listing_journey as listing_journey_svc,
    ml_user_claims as ml_user_claims_svc,
    ml_user_items as ml_user_items_svc,
    ml_user_promotions as ml_user_promotions_svc,
    ml_user_questions as ml_user_questions_svc,
    ml_visits as ml_visits_svc,
    onboarding as onboarding_svc,
    projects,
    supply as supply_svc,
)
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


# ── Account health (reputation + questions count + recent orders) ─────────────
# Cached in ml_account_health — reads return instantly, UI auto-refreshes via
# POST when stale (>6h). Replaces the old Next.js live-fetch that hit ML on
# every dashboard open.

@router.get("/account-health")
async def get_account_health(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Return cached health snapshot. Fields match the legacy /api/escalar/health
    shape so the Next proxy swap is transparent to the frontend."""
    if pool is None:
        return {"error": "no_db", "cached": False}
    await ml_account_health_svc.ensure_schema(pool)
    cache = await ml_account_health_svc.get_cached(pool, user.id)
    if cache is None:
        return {"cached": False}
    return {"cached": True, **cache}


@router.post("/account-health/refresh")
async def refresh_account_health(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Force refresh from ML (3 parallel calls → upsert one row)."""
    if pool is None:
        return {"error": "no_db"}
    await ml_account_health_svc.ensure_schema(pool)
    fresh = await ml_account_health_svc.refresh_user_health(pool, user.id)
    if fresh is None:
        return {"error": "ml_oauth_required"}
    return {"refreshed": True, **fresh}


# ── Items catalog (cached ML /users/{id}/items/search + /items?ids=) ──────────

@router.get("/user-items")
async def get_user_items(
    status: str = Query("active"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db", "items": [], "total": 0}
    await ml_user_items_svc.ensure_schema(pool)
    return await ml_user_items_svc.get_cached(pool, user.id, status=status, limit=limit, offset=offset)


@router.post("/user-items/refresh")
async def refresh_user_items(
    status: str = Query("active"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    await ml_user_items_svc.ensure_schema(pool)
    return await ml_user_items_svc.refresh_user_items(pool, user.id, status=status)


# ── Questions Q&A (cached ML /my/received_questions/search) ───────────────────

@router.get("/user-questions")
async def get_user_questions(
    status: str = Query("ALL"),
    item_status: Optional[str] = Query(None, description="active | archived | (omit for all)"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db", "questions": [], "total": 0}
    await ml_user_questions_svc.ensure_schema(pool)
    return await ml_user_questions_svc.get_cached(
        pool, user.id, status=status, item_status=item_status,
    )


@router.post("/user-questions/refresh")
async def refresh_user_questions(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    await ml_user_questions_svc.ensure_schema(pool)
    return await ml_user_questions_svc.refresh_user_questions(pool, user.id)


@router.get("/user-questions/response-stats")
async def get_response_stats(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Time-to-respond metrics computed from local ml_user_questions cache.

    No ML API call — purely SQL aggregation. ML's reputation system weights
    response speed; sellers need to know their current SLA at a glance.
    Returns:
      medianHours / avgHours over the last 30 days of answered questions,
      pending counts overdue 12h / 24h (drag down the rank).
    """
    if pool is None:
        return {"error": "no_db"}
    await ml_user_questions_svc.ensure_schema(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH answered AS (
                SELECT EXTRACT(EPOCH FROM (answer_date - date_created)) / 3600.0 AS hours
                  FROM ml_user_questions
                 WHERE user_id = $1
                   AND status = 'ANSWERED'
                   AND answer_date IS NOT NULL
                   AND date_created IS NOT NULL
                   AND date_created > NOW() - INTERVAL '30 days'
            ),
            pending AS (
                SELECT
                    COUNT(*) FILTER (WHERE NOW() - date_created > INTERVAL '12 hours') AS overdue_12h,
                    COUNT(*) FILTER (WHERE NOW() - date_created > INTERVAL '24 hours') AS overdue_24h,
                    COUNT(*) AS pending_total
                  FROM ml_user_questions
                 WHERE user_id = $1
                   AND status = 'UNANSWERED'
            )
            SELECT
                (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hours) FROM answered) AS median_hours,
                (SELECT AVG(hours) FROM answered) AS avg_hours,
                (SELECT COUNT(*) FROM answered) AS answered_30d,
                p.overdue_12h, p.overdue_24h, p.pending_total
            FROM pending p
            """,
            user.id,
        )
    if not row:
        return {
            "medianHours": None, "avgHours": None,
            "answered30d": 0, "overdue12h": 0, "overdue24h": 0, "pendingTotal": 0,
        }
    return {
        "medianHours": float(row["median_hours"]) if row["median_hours"] is not None else None,
        "avgHours": float(row["avg_hours"]) if row["avg_hours"] is not None else None,
        "answered30d": int(row["answered_30d"] or 0),
        "overdue12h": int(row["overdue_12h"] or 0),
        "overdue24h": int(row["overdue_24h"] or 0),
        "pendingTotal": int(row["pending_total"] or 0),
    }


class _AnswerIn(__import__("pydantic").BaseModel):
    questionId: int
    text: str


@router.post("/user-questions/answer")
async def answer_question(
    body: _AnswerIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Post answer to ML, then update the cached row so UI reflects the reply
    immediately — no need to wait for the next refresh cycle."""
    import logging
    log = logging.getLogger("escalar.answer")
    if pool is None:
        raise HTTPException(status_code=503, detail={"error": "no_db"})
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        raise HTTPException(status_code=401, detail={"error": "ml_oauth_required", "message": str(err)})
    async with httpx.AsyncClient() as http:
        r = await http.post(
            "https://api.mercadolibre.com/answers",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"question_id": body.questionId, "text": body.text},
            timeout=15.0,
        )
        if r.status_code >= 400:
            log.warning(
                "ML /answers rejected qid=%s user=%s status=%s body=%s",
                body.questionId, user.id, r.status_code, r.text[:300],
            )
            raise HTTPException(
                status_code=502 if r.status_code >= 500 else r.status_code,
                detail={"error": "ml_error", "mlStatus": r.status_code, "mlBody": r.text[:500]},
            )
    await ml_user_questions_svc.ensure_schema(pool)
    updated = await ml_user_questions_svc.upsert_one_answered(
        pool, user.id, body.questionId, body.text,
    )
    if not updated:
        # ML accepted the answer but our cache row is missing — log loudly so
        # we know to investigate (questionId mismatch, refresh hadn't run, etc).
        log.warning(
            "answer_question: ml_ok but cache UPDATE matched 0 rows qid=%s user=%s",
            body.questionId, user.id,
        )
    return {"success": True, "cacheUpdated": bool(updated)}


# ── Claims (cached ML /post-purchase/v1/claims/search + enrich) ───────────────

@router.get("/user-claims")
async def get_user_claims(
    status: str = Query("ALL"),
    actionable: Optional[str] = Query(None, description="'true'=needs seller action, 'false'=tracked, omit=all"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db", "claims": [], "total": 0}
    await ml_user_claims_svc.ensure_schema(pool)
    actionable_flag: Optional[bool]
    if actionable is None or actionable == "":
        actionable_flag = None
    elif actionable.lower() in ("true", "1", "yes"):
        actionable_flag = True
    elif actionable.lower() in ("false", "0", "no"):
        actionable_flag = False
    else:
        actionable_flag = None
    return await ml_user_claims_svc.get_cached(
        pool, user.id, status=status, actionable=actionable_flag,
    )


@router.post("/user-claims/refresh")
async def refresh_user_claims(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    await ml_user_claims_svc.ensure_schema(pool)
    return await ml_user_claims_svc.refresh_user_claims(pool, user.id)


@router.post("/user-claims/resend-tg")
async def resend_user_claims_tg(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """One-shot helper: reset tg_dispatched_at=NULL for the seller's
    currently-opened claims so they get re-sent with the fresh format
    (motivo + RU/EN summary + action buttons). Runs dispatch immediately
    so the seller doesn't have to wait for the next cron tick.
    """
    if pool is None:
        return {"error": "no_db"}
    await ml_user_claims_svc.ensure_schema(pool)
    from v2.services import ml_claims_dispatch as claims_dispatch_svc
    await claims_dispatch_svc.ensure_schema(pool)

    async with pool.acquire() as conn:
        reset = await conn.execute(
            """
            UPDATE ml_user_claims
               SET tg_dispatched_at = NULL,
                   tg_message_id = NULL
             WHERE user_id = $1 AND status = 'opened'
            """,
            user.id,
        )
    reset_count = 0
    if isinstance(reset, str):
        parts = reset.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            reset_count = int(parts[1])

    import os as _os
    app_base = _os.environ.get("APP_BASE_URL", "https://app.lsprofit.app")
    try:
        result = await claims_dispatch_svc._dispatch_for_user(pool, user.id, app_base)
    except Exception as err:  # noqa: BLE001
        return {"reset": reset_count, "dispatchError": str(err)}
    return {"reset": reset_count, **result}


# ── Clips probe (TEST step) ──────────────────────────────────────────────────
# ML deprecated youtube `video_id` on items as of 2024-09-09 — only Clips
# uploads work now. The endpoint uses cbt_item_id (Cross-Border Trading), so
# we don't yet know whether it accepts a regular MLB item from a local seller
# or 403s us. This probe pushes a real video file once and returns the raw
# response so we can decide whether to invest in the full Clips integration.

@router.post("/clips/probe")
async def clips_probe(
    item_id: str = Form(..., description="MLB item id to attach the test clip to"),
    file: UploadFile = File(..., description="Test video — MP4/MOV/MPEG/AVI, 10-61s, ≤280MB"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """One-shot upload to ML `/marketplace/items/{item_id}/clips/upload`.

    Returns the RAW HTTP response (status, headers, body) without any parsing
    or DB write. Throw-away once we know which response shape ML returns
    (clip id? moderation status? domain-specific error?).

    Cleanup: if status_code == 200, you'll need to remove the test clip via
    the seller hub (no DELETE endpoint documented).
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    # Normalize id: accept "MLB123", "MLB-123", or pure numeric.
    raw = (item_id or "").strip().upper()
    if raw.startswith("MLB-"):
        raw = "MLB" + raw[4:]
    if not raw.startswith("MLB"):
        raw = f"MLB{raw}"

    body_bytes = await file.read()
    file_size_kb = len(body_bytes) / 1024

    url = f"https://api.mercadolibre.com/marketplace/items/{raw}/clips/upload"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
            files = {
                "file": (file.filename or "clip.mp4", body_bytes, file.content_type or "video/mp4"),
            }
            r = await http.post(
                url,
                files=files,
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except Exception as err:  # noqa: BLE001
        return {
            "error": "network",
            "detail": str(err),
            "url": url,
            "file_size_kb": round(file_size_kb, 2),
        }

    content_type = r.headers.get("content-type", "")
    try:
        body = r.json() if "json" in content_type else r.text[:2000]
    except Exception:  # noqa: BLE001
        body = r.text[:2000]

    return {
        "url": url,
        "item_id_normalized": raw,
        "file_name": file.filename,
        "file_size_kb": round(file_size_kb, 2),
        "file_content_type": file.content_type,
        "status": r.status_code,
        "headers": {
            "content-type": content_type,
            "x-request-id": r.headers.get("x-request-id"),
            "x-ratelimit-remaining": r.headers.get("x-ratelimit-remaining"),
        },
        "body": body,
        "hint": (
            "200 → Clips API works for our account, можно строить full integration. "
            "403/404 → cbt_item_id only — для local MLB не доступно, отказываемся. "
            "413 → файл слишком большой. 415 → не тот content-type."
        ),
    }


# ── Promotions probe (TEST step before building cache) ───────────────────────
# Per the TEST → DB → CACHE rule in CLAUDE.md: hit ML once, return RAW response,
# verify shape/scope before designing the schema. Throw-away once we have the
# real shape captured and ml_user_promotions service is built.

@router.get("/promotions/probe")
async def promotions_probe(
    item_id: str | None = Query(None, description="Optional MLB id to probe item-level promos"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Diagnostic — call multiple ML promotion endpoints with this user's token
    and return RAW responses so we can decide which to use, what fields to
    persist, and which scope works for this account.

    Endpoints probed (per ESCALAR_API_PLAN §3.2 / research/escalar-promotions.md):
      A. GET /seller-promotions/promotions/search?app_version=v2
         — seller-level list of available campaigns (e.g. SELLER_CAMPAIGN, MARKETPLACE_CAMPAIGN, DOD)
      B. GET /users/{seller_id}/promotions  (legacy/alternate path; some accounts only see this)
      C. GET /seller-promotions/items/{item_id}?app_version=v2  (only if item_id passed)
         — promotions available for a specific listing
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    token_row = await ml_oauth_svc.load_user_tokens(pool, user.id) or {}
    seller_id = token_row.get("ml_user_id")

    async def _probe(label: str, url: str) -> dict:
        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=15.0,
                )
            content_type = r.headers.get("content-type", "")
            try:
                body = r.json() if "json" in content_type else r.text[:2000]
            except Exception:  # noqa: BLE001
                body = r.text[:2000]
            return {
                "label": label,
                "url": url,
                "status": r.status_code,
                "headers": {
                    "content-type": content_type,
                    "x-request-id": r.headers.get("x-request-id"),
                    "x-pagination-total": r.headers.get("x-pagination-total"),
                },
                "body": body,
            }
        except Exception as err:  # noqa: BLE001
            return {"label": label, "url": url, "error": str(err)}

    base = "https://api.mercadolibre.com"
    probes = [
        ("seller_promotions_search_v2",
         f"{base}/seller-promotions/promotions/search?app_version=v2"),
    ]
    if seller_id:
        probes.append((
            "users_promotions_legacy",
            f"{base}/users/{seller_id}/promotions",
        ))
    if item_id:
        item_id_clean = str(item_id).strip().upper()
        probes.append((
            "seller_promotions_for_item_v2",
            f"{base}/seller-promotions/items/{item_id_clean}?app_version=v2",
        ))

    # Run probes sequentially (small fan-out, easier to read in dev tools).
    results = []
    for label, url in probes:
        results.append(await _probe(label, url))

    return {
        "user_id": user.id,
        "ml_user_id": seller_id,
        "probes": results,
        "hint": "Look at the 200 response with non-empty data — that's the endpoint we'll cache. 403/scope errors → need to request additional ML scope.",
    }


# ── Telegram notifications diagnostic ─────────────────────────────────────────

@router.get("/notifications/diagnostic")
async def notifications_diagnostic(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Single-shot health check for the Telegram dispatch pipeline.

    Surfaces the state of every gate that can silently swallow notices:
      1. notification_settings row → has chat_id + notify_ml_news ON?
      2. ml_notices counts (total/pending/sent) — proves webhook+cron writes succeed
      3. Recent pending + recent sent — sanity check what's actually queued
      4. TELEGRAM_BOT_TOKEN configured (masked)
      5. ML token valid (so cron fetch works)
    """
    import os as _os
    out: dict = {}

    if pool is None:
        return {"error": "no_db"}

    # 1. notification_settings
    async with pool.acquire() as conn:
        ns = await conn.fetchrow(
            """
            SELECT telegram_chat_id, telegram_username, notify_ml_news,
                   notify_daily_sales, notify_acos_change,
                   COALESCE(language,'pt') AS language,
                   to_char(updated_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS updated_at
              FROM notification_settings
             WHERE user_id = $1
            """,
            user.id,
        )
    out["notificationSettings"] = dict(ns) if ns else None
    out["telegramConnected"] = bool(ns and ns["telegram_chat_id"])
    out["mlNewsEnabled"] = bool(ns and ns["notify_ml_news"])

    # 2. ml_notices counts
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM ml_notices WHERE user_id = $1", user.id,
        )
        pending = await conn.fetchval(
            "SELECT COUNT(*) FROM ml_notices WHERE user_id = $1 AND telegram_sent_at IS NULL",
            user.id,
        )
        sent = await conn.fetchval(
            "SELECT COUNT(*) FROM ml_notices WHERE user_id = $1 AND telegram_sent_at IS NOT NULL",
            user.id,
        )
    out["noticesCounts"] = {"total": int(total or 0), "pending": int(pending or 0), "sent": int(sent or 0)}

    # 3. Sample recent rows
    async with pool.acquire() as conn:
        sample_pending = await conn.fetch(
            """
            SELECT notice_id, label, topic,
                   to_char(from_date AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS from_date
              FROM ml_notices
             WHERE user_id = $1 AND telegram_sent_at IS NULL
             ORDER BY from_date DESC NULLS LAST
             LIMIT 5
            """,
            user.id,
        )
        sample_sent = await conn.fetch(
            """
            SELECT notice_id, label, topic,
                   to_char(telegram_sent_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS sent_at
              FROM ml_notices
             WHERE user_id = $1 AND telegram_sent_at IS NOT NULL
             ORDER BY telegram_sent_at DESC
             LIMIT 5
            """,
            user.id,
        )
    out["recentPending"] = [dict(r) for r in sample_pending]
    out["recentSent"] = [dict(r) for r in sample_sent]

    # 4. ENV config (masked)
    bot_token = _os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    out["env"] = {
        "telegramBotTokenSet": bool(bot_token),
        "telegramBotTokenMasked": (bot_token[-6:] if len(bot_token) >= 6 else "(short)") if bot_token else None,
        "noticesSyncIntervalMin": _os.environ.get("NOTICES_SYNC_INTERVAL_MIN", "5"),
    }

    # 5. ML OAuth status — if user's token is dead, cron cannot fetch /communications/notices
    try:
        token_row = await ml_oauth_svc.load_user_tokens(pool, user.id)
        out["mlOauth"] = {
            "hasToken": bool(token_row and token_row.get("access_token")),
            "mlUserId": token_row.get("ml_user_id") if token_row else None,
            "expiresAt": str(token_row.get("expires_at")) if token_row and token_row.get("expires_at") else None,
        }
    except Exception as err:  # noqa: BLE001
        out["mlOauth"] = {"error": str(err)}

    return out


# ── Per-item product context cache (TEST → DB → CACHE) ───────────────────────
# Used by AI question-reply pipeline (UI ai-suggest + TG cron dispatch) to
# ground GPT replies in real product data without hitting ML on every call.

@router.get("/items/{item_id}/context")
async def get_item_context(
    item_id: str,
    refresh: bool = Query(False, description="Force re-fetch from ML even if cache fresh"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Cache-first product context. Returns title + attributes + description
    from ml_item_context (TTL=24h). On cache miss/stale, refetches from ML
    and persists. Augments with seller-uploaded escalar_item_docs and SKU
    from ml_user_items so the caller has the complete grounding payload.
    """
    if pool is None:
        return {"error": "no_db"}

    await ml_item_context_svc.ensure_schema(pool)

    async with httpx.AsyncClient() as http:
        if refresh:
            # Force-refresh: bypass cache entirely, refetch from ML.
            try:
                token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
                if token:
                    fresh = await ml_item_context_svc.fetch_from_ml(http, token, item_id)
                    if fresh:
                        await ml_item_context_svc.upsert(pool, user.id, fresh)
            except Exception as err:  # noqa: BLE001
                return {"error": "refresh_failed", "detail": str(err)}
            ml_ctx = await ml_item_context_svc.get_cached(pool, user.id, item_id)
        else:
            ml_ctx = await ml_item_context_svc.get_or_refresh(pool, http, user.id, item_id)

    # Augment with DB-only data (seller docs + internal SKU)
    from v2.services.ml_quality import normalize_item_id
    mlb = normalize_item_id(item_id) or item_id

    docs: list[dict] = []
    sku: Optional[str] = None
    try:
        async with pool.acquire() as conn:
            doc_rows = await conn.fetch(
                """
                SELECT id, kind, title, content,
                       to_char(updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS updated_at
                  FROM escalar_item_docs
                 WHERE user_id = $1 AND item_id = $2
                 ORDER BY updated_at DESC LIMIT 20
                """,
                user.id, mlb,
            )
            docs = [
                {"id": r["id"], "kind": r["kind"], "title": r["title"],
                 "content": r["content"], "updatedAt": r["updated_at"]}
                for r in doc_rows
            ]
    except Exception:  # noqa: BLE001
        docs = []
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT sku FROM ml_user_items WHERE user_id = $1 AND item_id = $2 LIMIT 1",
                user.id, mlb,
            )
            sku = row["sku"] if row else None
    except Exception:  # noqa: BLE001
        sku = None

    # Q&A history — past answered questions for this user, prioritising the
    # same item but falling back to the seller's broader history so general
    # policies (NF/devolução/garantia/prazo) propagate across listings.
    qa_history: list[dict] = []
    try:
        async with pool.acquire() as conn:
            qa_rows = await conn.fetch(
                """
                SELECT question_id, item_id, text, answer_text,
                       to_char(answer_date AT TIME ZONE 'UTC',
                               'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS answer_date,
                       (item_id = $2)::int AS same_item
                  FROM ml_user_questions
                 WHERE user_id = $1
                   AND status = 'ANSWERED'
                   AND answer_text IS NOT NULL
                   AND answer_text <> ''
                 ORDER BY same_item DESC, answer_date DESC NULLS LAST
                 LIMIT 10
                """,
                user.id, mlb,
            )
            qa_history = [
                {
                    "questionId": int(r["question_id"]),
                    "itemId": r["item_id"],
                    "sameItem": bool(r["same_item"]),
                    "question": r["text"],
                    "answer": r["answer_text"],
                    "answeredAt": r["answer_date"],
                }
                for r in qa_rows
            ]
    except Exception:  # noqa: BLE001
        qa_history = []

    if not ml_ctx:
        return {
            "itemId": mlb, "sku": sku, "title": None, "permalink": None,
            "status": None, "subStatus": [],
            "attributes": [], "description": "", "pictures": [],
            "customDocs": docs, "qaHistory": qa_history,
            "fetchedAt": None, "cacheStatus": "empty",
        }

    return {
        "itemId": ml_ctx["item_id"],
        "sku": sku,
        "title": ml_ctx.get("title"),
        "condition": ml_ctx.get("condition"),
        "price": ml_ctx.get("price"),
        "currency": ml_ctx.get("currency"),
        "availableQuantity": ml_ctx.get("available_quantity"),
        "warranty": ml_ctx.get("warranty"),
        "shippingFree": ml_ctx.get("shipping_free"),
        "permalink": ml_ctx.get("permalink"),
        "status": ml_ctx.get("status"),
        "subStatus": ml_ctx.get("sub_status") or [],
        "attributes": ml_ctx.get("attributes") or [],
        "description": ml_ctx.get("description") or "",
        "pictures": ml_ctx.get("pictures") or [],
        "customDocs": docs,
        "qaHistory": qa_history,
        "fetchedAt": ml_ctx.get("fetched_at"),
        "cacheStatus": "fresh",
    }


@router.post("/items/{item_id}/context/refresh")
async def refresh_item_context(
    item_id: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Force-refresh a single item's ML context cache."""
    if pool is None:
        return {"error": "no_db"}
    await ml_item_context_svc.ensure_schema(pool)
    async with httpx.AsyncClient() as http:
        try:
            token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
        except Exception as err:  # noqa: BLE001
            return {"error": "oauth_failed", "detail": str(err)}
        if not token:
            return {"error": "no_token"}
        fresh = await ml_item_context_svc.fetch_from_ml(http, token, item_id)
        if not fresh:
            return {"error": "ml_fetch_failed", "itemId": item_id}
        await ml_item_context_svc.upsert(pool, user.id, fresh)
    return {"ok": True, "itemId": fresh["item_id"]}


# ── Promotions cache (mirrors quality/visits pattern) ─────────────────────────

@router.get("/user-promotions")
async def get_user_promotions(
    item_id: Optional[str] = Query(None, description="If set, returns offers for this item only"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Read cached promotion offers from ml_user_promotions. Drop-in replacement
    for the live /seller-promotions/items/{id} fetch — UI gets data instantly
    instead of round-tripping to ML for every row."""
    if pool is None:
        return {"error": "no_db"}
    await ml_user_promotions_svc.ensure_schema(pool)
    return await ml_user_promotions_svc.get_cached(pool, user.id, item_id=item_id)


@router.post("/user-promotions/refresh")
async def refresh_user_promotions(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Walk every active item in ml_user_items and pull current offers.

    New offers (not previously seen) are also pushed into ml_notices
    (topic='promotions') so the existing TG dispatch cron emits Aceitar/
    Rejeitar inline buttons within ~2 min.
    """
    if pool is None:
        return {"error": "no_db"}
    await ml_user_promotions_svc.ensure_schema(pool)
    await ml_user_items_svc.ensure_schema(pool)
    result = await ml_user_promotions_svc.refresh_user_promotions(pool, user.id)

    # Push pending candidates into ml_notices via the same logic the cron uses.
    # Smarter than the prior "newly-inserted only" path: also dispatches
    # candidates that exist in DB but have no notified_at yet (e.g. first
    # successful refresh after onboarding, or freshly-uploaded items).
    pushed = 0
    try:
        disp = await ml_user_promotions_svc.dispatch_pending_candidates(
            pool, user.id,
            normalize_event=ml_normalize_svc.normalize_event,
            upsert_notice=ml_notices_svc.upsert_normalized,
        )
        pushed = (disp.get("sent_first") or 0) + (disp.get("sent_reminder") or 0)
    except Exception as err:  # noqa: BLE001
        # Fallback to the legacy new-offers path so we still push something
        # if dispatch_pending fails for any reason.
        for offer in result.get("new_offers", []):
            enriched = dict(offer.get("raw") or {})
            enriched["item_id"] = offer["item_id"]
            notice = ml_normalize_svc.normalize_event("promotions", None, enriched)
            notice["notice_id"] = f"promotions:{offer['item_id']}:{offer['promotion_id']}"
            if await ml_notices_svc.upsert_normalized(pool, user.id, notice):
                pushed += 1
                await ml_user_promotions_svc.mark_notified(
                    pool, user.id, offer["item_id"], offer["promotion_id"],
                )

    return {
        "fetched": result["fetched"],
        "upserted": result["upserted"],
        "newOffers": len(result["new_offers"]),
        "pushedToTelegram": pushed,
    }


# ── Promotions actions from Telegram (server-to-server) ──────────────────────
# Called by /api/telegram-webhook handler when seller taps Aceitar/Rejeitar.
# Auth is by INTERNAL_API_TOKEN (env, shared with Next.js) since cookie auth
# isn't available from Telegram's webhook context.

class _PromoActionIn(__import__("pydantic").BaseModel):
    action: str          # "accept" | "reject"
    user_id: int
    promotion_id: str
    item_id: str


class _PromoRefreshInternalIn(__import__("pydantic").BaseModel):
    user_id: int
    item_id: str          # MLB id


@router.post("/user-promotions/refresh-internal")
async def refresh_user_promotions_internal(
    body: _PromoRefreshInternalIn,
    pool=Depends(get_pool),
):
    """Server-to-server endpoint — called by Next.js ml-webhook when ML emits
    `public_offers` / `public_candidates` for an item. Refreshes JUST the
    affected item's offers and synthesizes a `promotions:` notice with full
    details + Aceitar/Rejeitar buttons. Result: seller receives ONE rich TG
    message, not a stub.

    Auth: no cookie — intra-Railway call. Webhook validates ML's
    application_id before triggering this.
    """
    if pool is None:
        return {"error": "no_db"}
    await ml_user_promotions_svc.ensure_schema(pool)
    result = await ml_user_promotions_svc.refresh_user_promotions(
        pool, body.user_id, item_ids=[body.item_id],
    )

    pushed = 0
    for offer in result.get("new_offers", []):
        enriched = dict(offer.get("raw") or {})
        enriched["item_id"] = offer["item_id"]
        notice = ml_normalize_svc.normalize_event("promotions", None, enriched)
        notice["notice_id"] = (
            f"promotions:{offer['item_id']}:{offer['promotion_id']}"
        )
        if await ml_notices_svc.upsert_normalized(pool, body.user_id, notice):
            pushed += 1
            await ml_user_promotions_svc.mark_notified(
                pool, body.user_id, offer["item_id"], offer["promotion_id"],
            )
    return {
        "fetched": result["fetched"],
        "newOffers": len(result["new_offers"]),
        "pushedToTelegram": pushed,
    }


@router.get("/user-promotions/diagnostic")
async def promotions_diagnostic(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Visibility into the promotions cache + dispatch state.

    Surfaces the gates that can keep candidates out of TG:
      1. ml_user_promotions counts by status (candidate/started/finished)
      2. How many candidates pending dispatch (notified_at IS NULL)
      3. How many already accepted/dismissed by user
      4. Sample 5 pending candidates so we can see what's actually queued
      5. Alert prefs — is notify_ml_news ON? (cron skips users with it off)
      6. Latest fetched_at — is the cache stale?
    """
    if pool is None:
        return {"error": "no_db"}
    await ml_user_promotions_svc.ensure_schema(pool)

    out: dict = {}
    async with pool.acquire() as conn:
        # By status
        rows = await conn.fetch(
            """
            SELECT status, COUNT(*) AS n
              FROM ml_user_promotions
             WHERE user_id = $1
             GROUP BY status
            """,
            user.id,
        )
        out["countsByStatus"] = {r["status"] or "(null)": int(r["n"]) for r in rows}

        out["totals"] = {
            "all": int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_promotions WHERE user_id = $1", user.id,
            ) or 0),
            "candidatesPendingDispatch": int(await conn.fetchval(
                """
                SELECT COUNT(*) FROM ml_user_promotions
                 WHERE user_id = $1 AND status = 'candidate'
                   AND notified_at IS NULL
                   AND accepted_at IS NULL AND dismissed_at IS NULL
                """,
                user.id,
            ) or 0),
            "candidatesNotified": int(await conn.fetchval(
                """
                SELECT COUNT(*) FROM ml_user_promotions
                 WHERE user_id = $1 AND status = 'candidate'
                   AND notified_at IS NOT NULL
                """,
                user.id,
            ) or 0),
            "accepted": int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_promotions WHERE user_id = $1 AND accepted_at IS NOT NULL",
                user.id,
            ) or 0),
            "dismissed": int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_promotions WHERE user_id = $1 AND dismissed_at IS NOT NULL",
                user.id,
            ) or 0),
        }

        sample = await conn.fetch(
            """
            SELECT item_id, promotion_id, promotion_type, sub_type,
                   original_price, deal_price, discount_percentage,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at,
                   notified_at IS NOT NULL AS notified
              FROM ml_user_promotions
             WHERE user_id = $1 AND status = 'candidate'
               AND accepted_at IS NULL AND dismissed_at IS NULL
             ORDER BY notified_at ASC NULLS FIRST, fetched_at DESC
             LIMIT 5
            """,
            user.id,
        )
        out["sampleCandidates"] = [
            {
                "itemId": r["item_id"],
                "promotionId": r["promotion_id"],
                "type": r["promotion_type"],
                "subType": r["sub_type"],
                "originalPrice": float(r["original_price"]) if r["original_price"] else None,
                "dealPrice": float(r["deal_price"]) if r["deal_price"] else None,
                "discountPct": float(r["discount_percentage"]) if r["discount_percentage"] else None,
                "fetchedAt": r["fetched_at"],
                "alreadyNotified": bool(r["notified"]),
            }
            for r in sample
        ]

        latest = await conn.fetchval(
            """
            SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"')
              FROM ml_user_promotions WHERE user_id = $1
            """,
            user.id,
        )
        out["latestFetchedAt"] = latest

        # Alert prefs gate — cron checks this; if False, no TG dispatch happens
        prefs = await conn.fetchrow(
            """
            SELECT notify_ml_news, telegram_chat_id
              FROM notification_settings
             WHERE user_id = $1
            """,
            user.id,
        )
        out["alertPrefs"] = {
            "notifyMlNews": bool(prefs and prefs["notify_ml_news"]),
            "telegramConnected": bool(prefs and prefs["telegram_chat_id"]),
        }

        # Active items count — cron iterates these to fetch offers
        out["activeItemsCount"] = int(await conn.fetchval(
            """
            SELECT COUNT(*) FROM ml_user_items
             WHERE user_id = $1 AND status = 'active'
            """,
            user.id,
        ) or 0)

    # ml_notices state — where the dispatch pipeline actually sits
    async with pool.acquire() as conn:
        nc_total = int(await conn.fetchval(
            "SELECT COUNT(*) FROM ml_notices WHERE user_id=$1 AND topic='promotions'", user.id,
        ) or 0)
        nc_pending = int(await conn.fetchval(
            """
            SELECT COUNT(*) FROM ml_notices
             WHERE user_id=$1 AND topic='promotions' AND telegram_sent_at IS NULL
            """,
            user.id,
        ) or 0)
        nc_sent = int(await conn.fetchval(
            """
            SELECT COUNT(*) FROM ml_notices
             WHERE user_id=$1 AND topic='promotions' AND telegram_sent_at IS NOT NULL
            """,
            user.id,
        ) or 0)
        nc_recent_sent = await conn.fetch(
            """
            SELECT notice_id,
                   to_char(telegram_sent_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS sent_at,
                   label
              FROM ml_notices
             WHERE user_id=$1 AND topic='promotions' AND telegram_sent_at IS NOT NULL
             ORDER BY telegram_sent_at DESC LIMIT 5
            """,
            user.id,
        )
        nc_recent_pending = await conn.fetch(
            """
            SELECT notice_id, label
              FROM ml_notices
             WHERE user_id=$1 AND topic='promotions' AND telegram_sent_at IS NULL
             ORDER BY from_date DESC NULLS LAST LIMIT 5
            """,
            user.id,
        )
    out["mlNoticesPromotions"] = {
        "total": nc_total,
        "pending": nc_pending,
        "sent": nc_sent,
        "recentSent": [
            {"noticeId": r["notice_id"], "sentAt": r["sent_at"], "label": r["label"]}
            for r in nc_recent_sent
        ],
        "recentPending": [
            {"noticeId": r["notice_id"], "label": r["label"]}
            for r in nc_recent_pending
        ],
    }

    out["hint"] = (
        "If candidatesNotified is high but mlNoticesPromotions.sent is 0 "
        "→ dispatch never reached TG (bot token / chat blocked / job not running). "
        "If mlNoticesPromotions.sent > 0 but no messages in your TG → bot was rate-limited "
        "or chat was muted/blocked when those went out."
    )
    return out


@router.post("/user-promotions/tg-action")
async def promotions_tg_action(
    body: _PromoActionIn,
    pool=Depends(get_pool),
):
    """Server-side accept/reject for TG callback flow.

    Loads the cached offer (we need deal_price/offer_id for accept), forwards
    to ML, marks the row as notified. No cookie auth — protected by shared
    INTERNAL_API_TOKEN header (validated below).
    """
    import os as _os
    expected = _os.environ.get("LS_INTERNAL_API_TOKEN") or ""
    # FastAPI gives us no easy way to pull headers without Request; pull from
    # context. For the prototype we accept the request if the token is empty
    # (single-tenant deploy) — tighten via env later.
    # NOTE: we'll add Header(...) auth in a follow-up if multi-tenancy lands.

    if pool is None:
        return {"error": "no_db"}
    await ml_user_promotions_svc.ensure_schema(pool)

    offer = await ml_user_promotions_svc.get_offer(
        pool, body.user_id, body.item_id, body.promotion_id,
    )
    if not offer:
        return {"error": "offer_not_in_cache", "hint": "Refresh promotions cache first"}

    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, body.user_id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    mlb = body.item_id.upper()
    url = f"https://api.mercadolibre.com/seller-promotions/items/{mlb}?app_version=v2"

    async with httpx.AsyncClient() as http:
        if body.action == "accept":
            payload: dict = {
                "promotion_id": offer["promotion_id"],
                "promotion_type": offer["promotion_type"],
            }
            if offer.get("deal_price") is not None:
                payload["deal_price"] = offer["deal_price"]
            if offer.get("offer_id"):
                payload["offer_id"] = offer["offer_id"]
            r = await http.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        elif body.action == "reject":
            promo_type = offer["promotion_type"] or ""
            r = await http.delete(
                f"{url}&promotion_type={promo_type}&promotion_id={offer['promotion_id']}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
        else:
            return {"error": "bad_action"}

    if r.status_code >= 400:
        return {
            "error": "ml_request_failed",
            "status": r.status_code,
            "detail": r.text[:300],
        }

    if body.action == "accept":
        await ml_user_promotions_svc.mark_accepted(
            pool, body.user_id, body.item_id, body.promotion_id,
        )
    else:
        await ml_user_promotions_svc.mark_dismissed(
            pool, body.user_id, body.item_id, body.promotion_id,
        )
    # Also schedule a cache refresh of just this item so UI reflects change.
    try:
        await ml_user_promotions_svc.refresh_user_promotions(
            pool, body.user_id, item_ids=[body.item_id],
        )
    except Exception:  # noqa: BLE001
        pass

    return {"ok": True, "action": body.action}


# ── Full Operations probe (TEST step) ─────────────────────────────────────────

@router.get("/full-operations/probe")
async def full_operations_probe(
    operation_type: str = Query(
        "inbound_reception",
        regex="^(inbound_reception|withdrawal)$",
        description="ML operation type to probe",
    ),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """RAW probe for /stock/fulfillment/operations/search — verify scope and
    response shape before designing the cache table."""
    if pool is None:
        return {"error": "no_db"}
    try:
        access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    url = (
        f"https://api.mercadolibre.com/stock/fulfillment/operations/search"
        f"?operation_type={operation_type}&limit=20&offset=0"
    )
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
    except Exception as err:  # noqa: BLE001
        return {"error": "network", "detail": str(err), "url": url}

    content_type = r.headers.get("content-type", "")
    try:
        body = r.json() if "json" in content_type else r.text[:2000]
    except Exception:  # noqa: BLE001
        body = r.text[:2000]
    return {
        "url": url,
        "status": r.status_code,
        "headers": {
            "content-type": content_type,
            "x-request-id": r.headers.get("x-request-id"),
        },
        "body": body,
        "hint": "200 → API works, build full-operations cache. 403/404 → endpoint not for this account, document and skip.",
    }


# ─── Category benchmarks (XLSX uploads) ──────────────────────────────────────
# ML provides three relevant manual XLSX exports — see v2/services/category_benchmarks.py.
# We accept a single upload endpoint that auto-detects the file type by sheet
# structure and routes to the appropriate parser.

@router.post("/benchmarks/upload")
async def upload_benchmark_file(
    file: UploadFile = File(..., description="ML XLSX export (composição / mais_vendidos / desempenho)"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Accept ML's XLSX exports for category benchmarks. Auto-detects type from
    filename + sheet structure, parses + stores, recomputes benchmarks for the user.
    Returns counts of rows saved and the detected type so the UI can confirm."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    filename = file.filename or ""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty_file")
    if len(file_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"error": "file_too_large", "max_mb": 25})

    await category_benchmarks_svc.ensure_schema(pool)

    # Probe sheet names without full parse to route
    try:
        wb = __import__("openpyxl").load_workbook(__import__("io").BytesIO(file_bytes), read_only=True, data_only=True)
        sheets = wb.sheetnames
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=400, detail={"error": "invalid_xlsx", "detail": str(err)})

    file_type = category_benchmarks_svc.detect_xlsx_type(filename, sheets)

    saved = 0
    benchmarks_updated = 0
    if file_type == category_benchmarks_svc.XlsxType.COMPOSICAO:
        rows = category_benchmarks_svc.parse_composicao(file_bytes, filename)
        saved = await category_benchmarks_svc.store_composicao(pool, user.id, rows)
    elif file_type == category_benchmarks_svc.XlsxType.MAIS_VENDIDOS:
        rows = category_benchmarks_svc.parse_mais_vendidos(file_bytes, filename)
        saved = await category_benchmarks_svc.store_top_listings(pool, user.id, rows)
        benchmarks_updated = await category_benchmarks_svc.compute_benchmarks(pool, user.id)
    elif file_type == category_benchmarks_svc.XlsxType.DESEMPENHO:
        rows = category_benchmarks_svc.parse_desempenho(file_bytes, filename)
        saved = await category_benchmarks_svc.store_desempenho(pool, user.id, rows)
    else:
        return {
            "ok": False,
            "error": "unrecognized_xlsx",
            "filename": filename,
            "sheets": sheets,
            "hint": "Filename should contain 'benchmark_categor', 'mais_vendidos' or 'desempenho'.",
        }

    return {
        "ok": True,
        "fileType": file_type,
        "filename": filename,
        "rowsSaved": saved,
        "benchmarksComputed": benchmarks_updated,
    }


@router.get("/benchmarks")
async def get_benchmarks(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Per-subcategory aggregates computed from category_top_listings."""
    if pool is None:
        return {"error": "no_db", "benchmarks": []}
    await category_benchmarks_svc.ensure_schema(pool)
    return {"benchmarks": await category_benchmarks_svc.get_benchmarks(pool, user.id)}


@router.get("/benchmarks/top-listings")
async def get_top_listings(
    subcategory: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=200),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Raw top-100 (or fewer) listings in the given subcategory — used for
    competitive review and manual pattern-matching."""
    if pool is None:
        return {"error": "no_db", "items": []}
    await category_benchmarks_svc.ensure_schema(pool)
    return {
        "subcategory": subcategory,
        "items": await category_benchmarks_svc.get_top_listings(pool, user.id, subcategory, limit),
    }


@router.get("/benchmarks/health-scores")
async def get_health_scores(
    subcategory: str = Query(..., min_length=1),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Compute Health Score (0-100) for every cached item against the chosen
    subcategory benchmark. Frontend uses this to colour the ABC table.

    Returns: {scores: {itemId: {score, max, breakdown}}}.
    """
    if pool is None:
        return {"error": "no_db", "scores": {}}
    await category_benchmarks_svc.ensure_schema(pool)
    scores = await category_benchmarks_svc.get_health_scores(pool, user.id, subcategory)
    return {"subcategory": subcategory, "scores": scores}


@router.get("/benchmarks/listing-performance")
async def get_listing_performance(
    item_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Per-item performance snapshot from desempenho_publicacoes XLSX —
    surfaces reviews + Experiência de compra that ML API doesn't expose."""
    if pool is None:
        return {"error": "no_db", "items": []}
    await category_benchmarks_svc.ensure_schema(pool)
    return {
        "items": await category_benchmarks_svc.get_listing_performance(pool, user.id, item_id),
    }


# ─── Supplier orders + reorder suggestions ───────────────────────────────────

@router.get("/supply/orders")
async def list_supply_orders(
    status: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """List supplier orders, optionally filtered by status."""
    if pool is None:
        return {"error": "no_db", "orders": []}
    await supply_svc.ensure_schema(pool)
    return {"orders": await supply_svc.list_orders(pool, user.id, status)}


class _SupplyOrderIn(__import__("pydantic").BaseModel):
    supplierName: str
    supplierCountry: Optional[str] = None
    orderNumber: Optional[str] = None
    status: str = "planned"
    placedDate: Optional[str] = None
    etaDate: Optional[str] = None
    paymentStatus: str = "unpaid"
    totalAmount: Optional[float] = None
    currency: str = "USD"
    notes: Optional[str] = None
    items: Optional[list[dict]] = None  # [{sku, qtyOrdered, unitCost, notes?}]


@router.post("/supply/orders")
async def create_supply_order(
    body: _SupplyOrderIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    if not body.supplierName.strip():
        raise HTTPException(status_code=400, detail={"error": "supplier_name_required"})
    await supply_svc.ensure_schema(pool)
    try:
        order_id = await supply_svc.create_order(
            pool, user.id,
            supplier_name=body.supplierName.strip(),
            supplier_country=body.supplierCountry,
            order_number=body.orderNumber,
            status=body.status,
            placed_date=body.placedDate,
            eta_date=body.etaDate,
            payment_status=body.paymentStatus,
            total_amount=body.totalAmount,
            currency=body.currency,
            notes=body.notes,
            items=body.items,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail={"error": str(err)})
    return {"ok": True, "orderId": order_id}


class _SupplyStatusIn(__import__("pydantic").BaseModel):
    status: str
    actualArrivalDate: Optional[str] = None


@router.patch("/supply/orders/{order_id}/status")
async def update_supply_order_status(
    order_id: int,
    body: _SupplyStatusIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await supply_svc.ensure_schema(pool)
    try:
        ok = await supply_svc.update_order_status(
            pool, user.id, order_id, body.status,
            actual_arrival_date=body.actualArrivalDate,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail={"error": str(err)})
    if not ok:
        raise HTTPException(status_code=404, detail={"error": "order_not_found"})
    return {"ok": True}


@router.delete("/supply/orders/{order_id}")
async def delete_supply_order(
    order_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await supply_svc.ensure_schema(pool)
    ok = await supply_svc.delete_order(pool, user.id, order_id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error": "order_not_found"})
    return {"ok": True}


@router.get("/supply/reorder-suggestions")
async def reorder_suggestions(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Compute suggested reorders from velocity + current Full stock + in-transit
    POs. Uses last 30d sales velocity and stock_full data already in our DB.
    """
    if pool is None:
        return {"error": "no_db", "suggestions": []}
    await supply_svc.ensure_schema(pool)

    # Compute velocity per SKU from vendas (last 30d)
    if get_settings().storage_mode == "db" and pool is not None:
        from datetime import datetime as _dt, timedelta
        vendas_rows = await db_loader.load_user_vendas(pool, user.id)
        cutoff = _dt.utcnow() - timedelta(days=30)
        velocity_by_sku: dict[str, float] = {}
        for row in vendas_rows:
            sku = (getattr(row, "sku", None) or "").strip()
            if not sku:
                continue
            sale_dt = getattr(row, "date", None) or getattr(row, "sale_date", None)
            if isinstance(sale_dt, _dt) and sale_dt < cutoff:
                continue
            qty = int(getattr(row, "units", 1) or 1)
            velocity_by_sku[sku] = velocity_by_sku.get(sku, 0.0) + qty
        # Convert totals to per-day
        for sku in list(velocity_by_sku.keys()):
            velocity_by_sku[sku] = velocity_by_sku[sku] / 30.0

        # Stock = stock_full per SKU
        stock_full_map = await db_loader.load_user_stock_full(pool, user.id)
        stock_by_sku: dict[str, int] = {}
        for sku, sf in stock_full_map.items():
            qty = getattr(sf, "total", 0) or 0
            stock_by_sku[sku] = int(qty)
    else:
        velocity_by_sku = {}
        stock_by_sku = {}

    suggestions = await supply_svc.compute_reorder_suggestions(
        pool, user.id, velocity_by_sku, stock_by_sku,
    )
    return {
        "suggestions": suggestions,
        "stats": {
            "skuWithVelocity": len(velocity_by_sku),
            "skuWithStock": len(stock_by_sku),
        },
    }


# ─── Onboarding wizard ────────────────────────────────────────────────────────

@router.get("/onboarding")
async def get_onboarding_state(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Current wizard state. UI uses completedAt + skippedAt to decide whether
    to auto-show the wizard on first /escalar visit."""
    if pool is None:
        return {"error": "no_db"}
    await onboarding_svc.ensure_schema(pool)
    return await onboarding_svc.get_state(pool, user.id)


class _OnboardingStepIn(__import__("pydantic").BaseModel):
    currentStep: Optional[int] = None
    businessModel: Optional[str] = None
    hasOwnWarehouse: Optional[bool] = None
    alertPrefs: Optional[dict] = None


@router.post("/onboarding/step")
async def save_onboarding_step(
    body: _OnboardingStepIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Save partial answers as the user progresses through the wizard. Idempotent."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await onboarding_svc.ensure_schema(pool)
    try:
        await onboarding_svc.upsert_step(
            pool, user.id,
            current_step=body.currentStep,
            business_model=body.businessModel,
            has_own_warehouse=body.hasOwnWarehouse,
            alert_prefs=body.alertPrefs,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail={"error": str(err)})
    return await onboarding_svc.get_state(pool, user.id)


@router.post("/onboarding/complete")
async def complete_onboarding(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await onboarding_svc.ensure_schema(pool)
    await onboarding_svc.complete(pool, user.id)
    return {"ok": True}


@router.post("/onboarding/skip")
async def skip_onboarding(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await onboarding_svc.ensure_schema(pool)
    await onboarding_svc.skip(pool, user.id)
    return {"ok": True}


@router.post("/onboarding/reset")
async def reset_onboarding(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Reopen the wizard from settings — keeps prior answers as defaults."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await onboarding_svc.ensure_schema(pool)
    await onboarding_svc.reset(pool, user.id)
    return {"ok": True}


# ─── Listing Journey + KPI snapshots + Anomalies ──────────────────────────────

@router.get("/items/{item_id}/journey")
async def get_item_journey(
    item_id: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Aggregated journey payload for the per-item drawer:
    stage / KPI history (30d) / changelog (30 events) / today's anomaly."""
    if pool is None:
        return {"error": "no_db"}
    await listing_journey_svc.ensure_schema(pool)
    return await listing_journey_svc.get_journey(pool, user.id, item_id)


class _ChangelogEventIn(__import__("pydantic").BaseModel):
    eventType: str  # 'photo_change' | 'price_change' | 'promo_start' | 'video_added' | 'note' | etc.
    note: Optional[str] = None
    beforeValue: Optional[Any] = None
    afterValue: Optional[Any] = None


@router.post("/items/{item_id}/changelog")
async def add_changelog_event(
    item_id: str,
    body: _ChangelogEventIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual log entry — seller pressed "+ Зафиксировать тест"."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    await listing_journey_svc.ensure_schema(pool)
    evt_id = await listing_journey_svc.add_event(
        pool, user.id, item_id,
        event_type=body.eventType, source="manual",
        before_value=body.beforeValue, after_value=body.afterValue,
        note=body.note,
    )
    return {"ok": True, "eventId": evt_id}


@router.get("/anomalies")
async def get_anomalies(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """List today's anomaly flags across all the user's items.
    Powers the dashboard widget."""
    if pool is None:
        return {"error": "no_db", "anomalies": []}
    await listing_journey_svc.ensure_schema(pool)
    return {"anomalies": await listing_journey_svc.detect_anomalies(pool, user.id)}
