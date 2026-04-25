"""ABC products + snooze endpoints for the Escalar (Promotion) section."""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from v2.deps import CurrentUser, current_user, get_pool
from v2.legacy import db_storage as legacy_db
from v2.parsers import db_loader
from v2.schemas.escalar import EscalarProductsOut, SnoozeIn, SnoozeOut
from v2.services import (
    abc,
    ml_account_health as ml_account_health_svc,
    ml_backfill as ml_backfill_svc,
    ml_item_context as ml_item_context_svc,
    ml_oauth as ml_oauth_svc,
    ml_quality as ml_quality_svc,
    ml_user_claims as ml_user_claims_svc,
    ml_user_items as ml_user_items_svc,
    ml_user_questions as ml_user_questions_svc,
    ml_visits as ml_visits_svc,
    projects,
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
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db", "questions": [], "total": 0}
    await ml_user_questions_svc.ensure_schema(pool)
    return await ml_user_questions_svc.get_cached(pool, user.id, status=status)


@router.post("/user-questions/refresh")
async def refresh_user_questions(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    await ml_user_questions_svc.ensure_schema(pool)
    return await ml_user_questions_svc.refresh_user_questions(pool, user.id)


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
    if pool is None:
        return {"error": "no_db"}
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}
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
            return {"error": "ml_error", "status": r.status_code, "body": r.text[:500]}
    await ml_user_questions_svc.ensure_schema(pool)
    await ml_user_questions_svc.upsert_one_answered(pool, user.id, body.questionId, body.text)
    return {"success": True}


# ── Claims (cached ML /post-purchase/v1/claims/search + enrich) ───────────────

@router.get("/user-claims")
async def get_user_claims(
    status: str = Query("ALL"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db", "claims": [], "total": 0}
    await ml_user_claims_svc.ensure_schema(pool)
    return await ml_user_claims_svc.get_cached(pool, user.id, status=status)


@router.post("/user-claims/refresh")
async def refresh_user_claims(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    await ml_user_claims_svc.ensure_schema(pool)
    return await ml_user_claims_svc.refresh_user_claims(pool, user.id)


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
                token = await ml_oauth_svc.get_valid_access_token(pool, user.id)
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

    if not ml_ctx:
        return {
            "itemId": mlb, "sku": sku, "title": None, "permalink": None,
            "attributes": [], "description": "", "customDocs": docs,
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
        "attributes": ml_ctx.get("attributes") or [],
        "description": ml_ctx.get("description") or "",
        "customDocs": docs,
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
            token = await ml_oauth_svc.get_valid_access_token(pool, user.id)
        except Exception as err:  # noqa: BLE001
            return {"error": "oauth_failed", "detail": str(err)}
        if not token:
            return {"error": "no_token"}
        fresh = await ml_item_context_svc.fetch_from_ml(http, token, item_id)
        if not fresh:
            return {"error": "ml_fetch_failed", "itemId": item_id}
        await ml_item_context_svc.upsert(pool, user.id, fresh)
    return {"ok": True, "itemId": fresh["item_id"]}
