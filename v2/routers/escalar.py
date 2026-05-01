"""ABC products + snooze endpoints for the Escalar (Promotion) section."""
from __future__ import annotations

import json
from typing import Any, Optional, Union

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Response, UploadFile

from v2.deps import CurrentUser, current_user, get_pool
from v2.legacy import db_storage as legacy_db
from v2.parsers import db_loader
from v2.schemas.escalar import EscalarProductsOut, SnoozeIn, SnoozeOut
from v2.services import (
    abc,
    finance_cache,
    category_benchmarks as category_benchmarks_svc,
    ml_account_health as ml_account_health_svc,
    ml_backfill as ml_backfill_svc,
    ml_item_context as ml_item_context_svc,
    ml_normalize as ml_normalize_svc,
    ml_notices as ml_notices_svc,
    goals as goals_svc,
    ml_anomalies as ml_anomalies_svc,
    ml_oauth as ml_oauth_svc,
    ml_orders as ml_orders_svc,
    ml_quality as ml_quality_svc,
    ml_scraper_chat as ml_scraper_chat_svc,
    daily_summary_dispatch as daily_summary_dispatch_svc,
    photo_ab_dispatch as photo_ab_dispatch_svc,
    listing_journey as listing_journey_svc,
    project_members as project_members_svc,
    email_brevo as email_brevo_svc,
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
    response: Response,
    days: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    fresh: bool = Query(False, description="Bypass abc cache and recompute from scratch"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    import asyncio as _asyncio
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

    # Pull live listing prices from ml_user_items so abc can compute unit econ
    # at the current selling price (not the historical period average — avg
    # mixes in past discounts and is misleading for "should I take this promo").
    current_prices_map: dict[str, float] = {}
    items_max_fetched_iso: str | None = None
    snooze_updated_at_iso: str | None = None
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                price_rows = await conn.fetch(
                    "SELECT item_id, price FROM ml_user_items WHERE user_id = $1",
                    user.id,
                )
                # MAX(fetched_at) feeds the cache fingerprint via extra_deps so
                # that an items refresh (price change) invalidates ABC, not
                # just an upload or settings change.
                items_max_row = await conn.fetchrow(
                    "SELECT MAX(fetched_at) AS m FROM ml_user_items WHERE user_id = $1",
                    user.id,
                )
                if items_max_row and items_max_row["m"]:
                    items_max_fetched_iso = items_max_row["m"].isoformat()
                # Snoozed-SKU list lives in user_data but we keep it OUT of
                # the base finance fingerprint (reports/matrix don't care).
                # Pulling its updated_at here lets ABC alone invalidate when
                # the user toggles snooze on a SKU.
                snooze_row = await conn.fetchrow(
                    "SELECT updated_at FROM user_data WHERE user_id = $1 AND data_key = $2",
                    user.id, SNOOZE_KEY,
                )
                if snooze_row and snooze_row["updated_at"]:
                    snooze_updated_at_iso = snooze_row["updated_at"].isoformat()
            for r in price_rows:
                if r["price"] is not None:
                    try:
                        current_prices_map[r["item_id"]] = float(r["price"])
                    except (TypeError, ValueError):
                        pass
            _step(f"after current_prices load ({len(current_prices_map)} items)")
        except Exception as err:  # noqa: BLE001
            _log.warning("current_prices load failed: %s", err)

    # ABC compute is the heavy part — wrap only this in the durable cache.
    # Quality/visits/items_meta joins below stay live so item refreshes show up
    # without a force-refresh of ABC.
    def _abc_compute() -> dict:
        return abc.aggregate(
            days=days_v,
            project=project or "",
            snoozed_skus=snoozed,
            resolver=resolver,
            vendas_rows=vendas_rows,
            storage_map=storage_map,
            stock_full_map=stock_full_map,
            vendas_filenames=vendas_filenames,
            publicidade_rows=publicidade_rows,
            current_prices_map=current_prices_map,
        )

    abc_cache_key = f"abc:{project or 'all'}:{days_v}"
    abc_extra_deps = {
        "ml_user_items_max_fetched": items_max_fetched_iso,
        "snoozed_updated_at": snooze_updated_at_iso,
    }
    summary, abc_status = await _asyncio.to_thread(
        finance_cache.cached_compute,
        user.id, abc_cache_key, _abc_compute,
        force=fresh,
        extra_deps=abc_extra_deps,
    )
    response.headers["X-Cache-Abc"] = abc_status
    _step(f"after abc.aggregate cache={abc_status} (products={len(summary['products'])})")

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

    # Join cached ml_user_items meta (status / sub_status / tags) so the
    # /escalar/moderation UI can compute severity client-side without
    # firing N × 2 ML API calls on every page load. Includes paused /
    # closed / under_review items that backfill brings into the cache.
    items_meta_map: dict[str, dict[str, Any]] = {}
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT item_id, status, raw FROM ml_user_items WHERE user_id = $1",
                    user.id,
                )
            import json as _json_mod
            for r in rows:
                raw = r["raw"]
                if isinstance(raw, str):
                    try:
                        raw = _json_mod.loads(raw)
                    except Exception:  # noqa: BLE001
                        raw = {}
                if not isinstance(raw, dict):
                    raw = {}
                items_meta_map[r["item_id"]] = {
                    "status": r["status"],
                    "sub_status": raw.get("sub_status") or [],
                    "tags": raw.get("tags") or [],
                }
            _step(f"after items meta load ({len(items_meta_map)} items)")
        except Exception as err:  # noqa: BLE001
            _log.warning("items meta load failed: %s", err)

    # Severity classification (mirror of /api/escalar/items/[itemId]/moderation
    # so the page sees the same buckets without extra fetches).
    _PROBLEMATIC_TAGS = {
        "incomplete_compatibilities", "incomplete_technical_specs",
        "incomplete_position_compatibilities", "moderation_penalty",
        "catalog_forewarning", "poor_quality_picture", "poor_quality_thumbnail",
        "not_market_price", "lost_me2_by_dimensions",
    }

    def _severity(status: Any, sub_status: Any, tags: Any) -> str:
        subs = [s.lower() for s in (sub_status or []) if isinstance(s, str)]
        if status == "closed":
            return "red"
        if "forbidden" in subs or "waiting_for_patch" in subs:
            return "red"
        if any(s in subs for s in ("pending_documentation", "held", "warning")):
            return "amber"
        if any("picture_download" in s for s in subs):
            return "gray"
        if status == "under_review":
            return "amber"
        if status == "paused":
            return "gray"
        problematic = [t for t in (tags or []) if isinstance(t, str) and t in _PROBLEMATIC_TAGS]
        if problematic:
            return "amber"
        return "ok"

    moderation_coverage = 0
    for p in products_out:
        item_id = p.get("itemId")
        if not item_id:
            continue
        meta = items_meta_map.get(item_id)
        if not meta:
            continue
        moderation_coverage += 1
        p["itemStatus"] = meta["status"]
        p["itemSubStatus"] = meta["sub_status"]
        p["itemTags"] = meta["tags"]
        p["severity"] = _severity(meta["status"], meta["sub_status"], meta["tags"])

    meta = dict(summary["meta"])
    meta["qualityFetchedAt"] = latest_fetched_at
    meta["qualityCoverage"] = quality_coverage
    meta["visitsFetchedAt"] = visits_latest_fetched_at
    meta["visitsCoverage"] = visits_coverage
    meta["moderationCoverage"] = moderation_coverage
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


@router.get("/items/stock-lookup")
async def items_stock_lookup(
    q: str = Query(..., description="MLB-id, часть title или ML attribute SELLER_SKU"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Текущий остаток `available_quantity` из `ml_user_items` для items
    matching `q`. SKU и sub_status извлекаются из `raw` JSONB (отдельных
    колонок нет — см. v2/services/ml_user_items.py CREATE_SQL)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    pat = f"%{q}%"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT item_id, title, available_quantity, sold_quantity, status, raw,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_items
             WHERE user_id = $1
               AND (
                 item_id ILIKE $2
                 OR title ILIKE $2
                 OR raw::text ILIKE $2
               )
             ORDER BY fetched_at DESC NULLS LAST
             LIMIT 50
            """,
            user.id, pat,
        )

    out_items: list[dict[str, Any]] = []
    for r in rows:
        raw = r["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        raw = raw or {}
        sku = ""
        for attr in (raw.get("attributes") or []):
            if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                sku = str(attr.get("value_name") or "")
                break
        sub_status = raw.get("sub_status") or []
        if not isinstance(sub_status, list):
            sub_status = [sub_status]
        out_items.append({
            "item_id": r["item_id"],
            "title": r["title"],
            "sku": sku,
            "available_quantity": r["available_quantity"],
            "sold_quantity": r["sold_quantity"],
            "status": r["status"],
            "sub_status": sub_status,
            "fetched_at": r["fetched_at"],
        })
    return {"query": q, "count": len(out_items), "items": out_items}


@router.get("/items/paused-with-stock")
async def items_paused_with_stock(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Список paused-объявлений у которых остаток > 0 на складе ML.
    Аномалия — paused обычно auto-set при stockout, но иногда юзер ставит
    pause вручную и забывает, либо ML модерация. Эти позиции занимают
    Full без trafficа продаж — нужно либо активировать, либо разобраться."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT item_id, title, available_quantity, sold_quantity, status, raw,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_items
             WHERE user_id = $1
               AND status = 'paused'
               AND available_quantity > 0
             ORDER BY available_quantity DESC, sold_quantity DESC
            """,
            user.id,
        )

    out_items: list[dict[str, Any]] = []
    for r in rows:
        raw = r["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        raw = raw or {}
        sku = ""
        for attr in (raw.get("attributes") or []):
            if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                sku = str(attr.get("value_name") or "")
                break
        sub_status = raw.get("sub_status") or []
        if not isinstance(sub_status, list):
            sub_status = [sub_status]
        out_items.append({
            "item_id": r["item_id"],
            "title": r["title"],
            "sku": sku,
            "available_quantity": r["available_quantity"],
            "sold_quantity": r["sold_quantity"],
            "sub_status": sub_status,
            "fetched_at": r["fetched_at"],
        })
    return {"count": len(out_items), "items": out_items}


@router.post("/items/{item_id}/activate")
async def items_activate(
    item_id: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Активирует paused-объявление через ML API (PUT /items/{id} status=active).

    Pre-flight проверка: товар должен принадлежать юзеру (есть в `ml_user_items`)
    и иметь stock > 0 (иначе ML всё равно не активирует — auto-pause при 0).

    После успеха обновляет `ml_user_items.status = 'active'` локально, чтобы
    UI сразу видел новый статус без ожидания ночного refresh."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT item_id, status, available_quantity
              FROM ml_user_items
             WHERE user_id = $1 AND item_id = $2
            """,
            user.id, item_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail={"error": "item_not_owned"})
    stock = int(row["available_quantity"] or 0)
    if stock <= 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "no_stock", "stock": stock,
                    "hint": "ML auto-pauses listings without stock; refill Full first"},
        )

    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        raise HTTPException(status_code=401,
                            detail={"error": "ml_oauth_required", "message": str(err)})

    item_url = f"https://api.mercadolibre.com/items/{item_id}"
    try:
        async with httpx.AsyncClient() as http:
            pr = await http.put(
                item_url,
                json={"status": "active"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "error": "ml_call_failed", "detail": str(err)}
    if pr.status_code >= 400:
        return {"ok": False, "error": "ml_rejected",
                "status": pr.status_code, "detail": pr.text[:400]}

    new_status = (pr.json() or {}).get("status") or "active"
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE ml_user_items SET status = $1 WHERE user_id = $2 AND item_id = $3",
            new_status, user.id, item_id,
        )

    return {"ok": True, "item_id": item_id, "new_status": new_status, "stock": stock}


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


# ── Moderations: paused items + ML moderation notices ────────────────────────


@router.get("/moderations")
async def moderations_list(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Listings the seller needs to fix — paused/inactive items, sub_status
    reasons, plus matching `item:MLB...` notices.

    Two sources merged into one list:
      1. `ml_user_items` rows with status = 'paused' / 'inactive' / 'closed'
         → tells us WHICH item is down + sub_status array (the ML reason
         tags like 'expired', 'out_of_stock', 'flagged_by_moderation', etc).
      2. `ml_notices` rows whose notice_id starts with 'item:' AND label
         hints at moderation/pausing → captures cases where ML emitted a
         notice but our items cache hasn't refreshed yet.

    De-dupes on item_id across sources (notice wins for label/description,
    item row wins for sub_status + thumbnail).
    """
    if pool is None:
        return {"moderations": [], "total": 0}

    async with pool.acquire() as conn:
        item_rows = await conn.fetch(
            """
            SELECT item_id, title, status, permalink, thumbnail, raw,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_items
             WHERE user_id = $1
               AND status IN ('paused', 'inactive', 'closed', 'under_review')
             ORDER BY fetched_at DESC NULLS LAST
            """,
            user.id,
        )
        notice_rows = await conn.fetch(
            """
            SELECT notice_id, label, description, from_date, tags, raw, read_at,
                   to_char(from_date AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS from_date_iso
              FROM ml_notices
             WHERE user_id = $1
               AND notice_id LIKE 'item:%'
               AND (
                 label ILIKE '%pausa%'
                 OR label ILIKE '%inativ%'
                 OR label ILIKE '%moder%'
                 OR label ILIKE '%bloque%'
                 OR label ILIKE '%infra%'
                 OR label ILIKE '%suspens%'
               )
             ORDER BY from_date DESC NULLS LAST
            """,
            user.id,
        )

    by_item: dict[str, dict[str, Any]] = {}

    for r in item_rows:
        raw = r["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        sub_status = (raw or {}).get("sub_status") or []
        if not isinstance(sub_status, list):
            sub_status = [sub_status]
        by_item[r["item_id"]] = {
            "item_id": r["item_id"],
            "title": r["title"],
            "status": r["status"],
            "sub_status": sub_status,
            "permalink": r["permalink"],
            "thumbnail": r["thumbnail"],
            "fetched_at": r["fetched_at"],
            "label": None,
            "description": None,
            "notice_date": None,
            "source": "items",
        }

    for r in notice_rows:
        nid = (r["notice_id"] or "")
        # notice_id format is 'item:MLB...'
        item_id = nid.split(":", 1)[1] if ":" in nid else nid
        slot = by_item.setdefault(item_id, {
            "item_id": item_id,
            "title": None,
            "status": None,
            "sub_status": [],
            "permalink": None,
            "thumbnail": None,
            "fetched_at": None,
            "source": "notice",
        })
        slot["label"] = r["label"]
        slot["description"] = r["description"]
        slot["notice_date"] = r["from_date_iso"]
        slot["read_at"] = bool(r["read_at"])
        if slot.get("source") == "items":
            slot["source"] = "both"

    moderations = sorted(
        by_item.values(),
        key=lambda m: m.get("notice_date") or m.get("fetched_at") or "",
        reverse=True,
    )
    return {
        "moderations": moderations,
        "total": len(moderations),
        "counts": {
            "paused": sum(1 for m in moderations if m.get("status") == "paused"),
            "inactive": sum(1 for m in moderations if m.get("status") == "inactive"),
            "closed": sum(1 for m in moderations if m.get("status") == "closed"),
            "under_review": sum(1 for m in moderations if m.get("status") == "under_review"),
            "notice_only": sum(1 for m in moderations if m.get("source") == "notice"),
        },
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
    include_paused: bool = Query(True),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Rebuild the ml_item_quality cache for this user's items.

    Source of item_ids: `ml_user_items` (the live ML catalog mirror),
    NOT abc.aggregate which only knows items present in uploaded vendas
    CSVs. Without this fix, items without recent CSV-tracked sales fall
    through quality coverage, leaving qualityCoverage stuck around 50%.

    `include_paused=True` (default) covers paused listings too —
    moderation reasons can be attached to paused items and the seller
    needs to see them.
    """
    if pool is None:
        return {"error": "no_db"}

    statuses = ["active"]
    if include_paused:
        statuses.extend(["paused", "under_review"])

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT item_id FROM ml_user_items
             WHERE user_id = $1 AND status = ANY($2::text[])
             ORDER BY fetched_at DESC NULLS LAST
            """,
            user.id, statuses,
        )
    item_ids = [r["item_id"] for r in rows if r["item_id"]]

    await ml_quality_svc.ensure_schema(pool)
    result = await ml_quality_svc.refresh_user_quality(pool, user.id, item_ids, limit=limit)
    return {
        "totalItems": len(item_ids),
        "source": "ml_user_items",
        "statuses": statuses,
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
    include_paused: bool = Query(True),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Rebuild the ml_item_visits cache for this user's items.

    Source: `ml_user_items` (live ML catalog mirror), same as
    refresh-quality. Items not present in CSV vendas now also get
    visits cached, fixing the visitsCoverage gap. include_paused
    covers paused listings too — useful for «когда было больше визитов
    до паузы» retrospectives.
    """
    if pool is None:
        return {"error": "no_db"}

    statuses = ["active"]
    if include_paused:
        statuses.extend(["paused", "under_review"])

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT item_id FROM ml_user_items
             WHERE user_id = $1 AND status = ANY($2::text[])
             ORDER BY fetched_at DESC NULLS LAST
            """,
            user.id, statuses,
        )
    item_ids = [r["item_id"] for r in rows if r["item_id"]]

    await ml_visits_svc.ensure_schema(pool)
    result = await ml_visits_svc.refresh_user_visits(pool, user.id, item_ids, limit=limit)
    return {
        "totalItems": len(item_ids),
        "source": "ml_user_items",
        "statuses": statuses,
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


@router.get("/user-claims/{claim_id}/messages-probe")
async def claim_messages_probe(
    claim_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Probe ML's /claims/{id}/messages directly so we can see whether the
    enrichment fetch silently 403s. Returns raw status + body — same
    pattern as the TEST step elsewhere."""
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    headers = {"Authorization": f"Bearer {token}"}
    candidates = [
        ("v1_messages", f"{base}/post-purchase/v1/claims/{claim_id}/messages"),
        ("v2_messages", f"{base}/post-purchase/v2/claims/{claim_id}/messages"),
        ("v1_history", f"{base}/post-purchase/v1/claims/{claim_id}/history"),
        ("v1_actions", f"{base}/post-purchase/v1/claims/{claim_id}/actions"),
        ("v2_full", f"{base}/post-purchase/v2/claims/{claim_id}"),
        # For mediation/dispute claims the seller's reply lives on a
        # different thread — these probe URLs reveal which path actually
        # has the data so we know /claims/{id}/messages alone isn't enough.
        ("v1_mediation_full", f"{base}/post-purchase/v1/mediations/{claim_id}"),
        ("v1_mediation_messages", f"{base}/post-purchase/v1/mediations/{claim_id}/messages"),
    ]
    results = []
    async with httpx.AsyncClient() as http:
        for label, url in candidates:
            try:
                r = await http.get(url, headers=headers, timeout=15.0)
                results.append({"label": label, "url": url, "status": r.status_code, "body_preview": r.text[:600]})
            except Exception as err:  # noqa: BLE001
                results.append({"label": label, "url": url, "error": str(err)})
    return {"claim_id": claim_id, "results": results}


@router.get("/user-claims/{claim_id}/file-probe")
async def claim_file_probe(
    claim_id: int,
    filename: str = Query(..., description="Server filename from message attachment"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Probe ML for the actual binary download URL of a claim attachment.
    Each message attachment gives us only metadata (size, type, filename) —
    no URL. We need to find the GET path that returns the bytes.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    headers = {"Authorization": f"Bearer {token}"}
    base_path = f"{base}/post-purchase/v1/claims/{claim_id}/attachments/{filename}"
    candidates = [
        # Confirmed-200 metadata path — re-include to capture FULL JSON
        # so we can spot a download_url-like field.
        ("metadata_full", base_path),
        # Likely binary subpaths
        ("subpath_file", f"{base_path}/file"),
        ("subpath_binary", f"{base_path}/binary"),
        ("subpath_download", f"{base_path}/download"),
        ("subpath_content", f"{base_path}/content"),
        ("subpath_raw", f"{base_path}/raw"),
        # Query-string variants
        ("query_download_true", f"{base_path}?download=true"),
        ("query_source_download", f"{base_path}?source=download"),
        ("query_format_binary", f"{base_path}?format=binary"),
    ]
    results = []
    full_metadata: Any = None
    async with httpx.AsyncClient() as http:
        for label, url in candidates:
            try:
                r = await http.get(url, headers=headers, timeout=15.0, follow_redirects=False)
                ct = (r.headers.get("content-type", "") or "").lower()
                meta: dict[str, Any] = {
                    "label": label,
                    "status": r.status_code,
                    "content_type": ct,
                    "content_length": r.headers.get("content-length", ""),
                    "location": r.headers.get("location", ""),
                }
                if r.status_code == 200 and ("image/" in ct or "octet-stream" in ct):
                    meta["binary_size"] = len(r.content)
                elif r.status_code == 200 and "json" in ct:
                    try:
                        body = r.json()
                        meta["full_body"] = body
                        if label == "metadata_full":
                            full_metadata = body
                    except Exception:  # noqa: BLE001
                        meta["body_preview"] = r.text[:600]
                else:
                    meta["body_preview"] = r.text[:300]
                results.append(meta)
            except Exception as err:  # noqa: BLE001
                results.append({"label": label, "error": str(err)})
    metadata_keys = sorted(full_metadata.keys()) if isinstance(full_metadata, dict) else None
    return {
        "claim_id": claim_id,
        "filename": filename,
        "metadata_keys": metadata_keys,
        "results": results,
    }


@router.get("/user-claims/{claim_id}/attachments-probe")
async def claim_attachments_probe(
    claim_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Probe ML for buyer-uploaded photos/evidences attached to a claim.
    Tries several documented + likely paths plus inspects messages and
    returns for embedded attachments.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    headers = {"Authorization": f"Bearer {token}"}

    out: dict = {"claim_id": claim_id}
    async with httpx.AsyncClient() as http:
        # Direct attachment-style endpoints
        endpoint_results = []
        candidates = [
            ("v1_evidences_with_role", f"{base}/post-purchase/v1/claims/{claim_id}/evidences?role=complainant"),
            ("v1_evidences_with_player", f"{base}/post-purchase/v1/claims/{claim_id}/evidences?player_role=complainant"),
            ("v2_evidences", f"{base}/post-purchase/v2/claims/{claim_id}/evidences"),
            ("v1_messages_attachments", f"{base}/post-purchase/v1/claims/{claim_id}/messages/attachments"),
        ]
        for label, url in candidates:
            try:
                r = await http.get(url, headers=headers, timeout=15.0)
                endpoint_results.append({
                    "label": label, "status": r.status_code, "body_preview": r.text[:500],
                })
            except Exception as err:  # noqa: BLE001
                endpoint_results.append({"label": label, "error": str(err)})
        out["endpoints"] = endpoint_results

        # Full message thread — inspect each message for embedded attachments[]
        try:
            r = await http.get(
                f"{base}/post-purchase/v1/claims/{claim_id}/messages",
                headers=headers, timeout=15.0,
            )
            if r.status_code == 200:
                try:
                    msgs = r.json()
                    if isinstance(msgs, dict):
                        msgs = msgs.get("messages") or []
                    out["messages_full"] = msgs[:5]  # first 5 with all fields
                    # Extract attachment-like keys from each message
                    seen_keys: set[str] = set()
                    for m in (msgs or [])[:10]:
                        if isinstance(m, dict):
                            for k in m.keys():
                                if "attach" in k.lower() or "evidenc" in k.lower() or "media" in k.lower() or "file" in k.lower() or "photo" in k.lower() or "image" in k.lower():
                                    seen_keys.add(k)
                    out["messages_attachment_keys"] = sorted(seen_keys)
                except Exception as err:  # noqa: BLE001
                    out["messages_parse_error"] = str(err)
            else:
                out["messages_status"] = r.status_code
        except Exception as err:  # noqa: BLE001
            out["messages_error"] = str(err)

        # Return record (if any) — buyer evidences often live there
        try:
            r = await http.get(
                f"{base}/post-purchase/v2/claims/{claim_id}/returns",
                headers=headers, timeout=15.0,
            )
            if r.status_code == 200:
                rd = r.json()
                returns = rd.get("data") if isinstance(rd, dict) and isinstance(rd.get("data"), list) else (rd if isinstance(rd, list) else [rd] if rd else [])
                ret_summaries = []
                for ret in (returns or [])[:3]:
                    if not isinstance(ret, dict):
                        continue
                    ret_id = ret.get("id")
                    ret_summaries.append({
                        "id": ret_id,
                        "status": ret.get("status"),
                        "keys": sorted(list(ret.keys())) if isinstance(ret, dict) else [],
                    })
                    # Probe per-return endpoints
                    if ret_id:
                        for label, sub_url in [
                            ("return_evidences", f"{base}/post-purchase/v1/returns/{ret_id}/evidences"),
                            ("return_attachments", f"{base}/post-purchase/v1/returns/{ret_id}/attachments"),
                            ("return_full", f"{base}/post-purchase/v1/returns/{ret_id}"),
                        ]:
                            try:
                                sr = await http.get(sub_url, headers=headers, timeout=15.0)
                                ret_summaries[-1][label] = {
                                    "status": sr.status_code,
                                    "body_preview": sr.text[:400],
                                }
                            except Exception as err:  # noqa: BLE001
                                ret_summaries[-1][label] = {"error": str(err)}
                out["returns"] = ret_summaries
        except Exception as err:  # noqa: BLE001
            out["returns_error"] = str(err)

    return out


@router.get("/user-claims/{claim_id}/decide-probe")
@router.get("/user-claims/{claim_id}/send-message-probe")
async def claim_send_message_probe(
    claim_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Discover the POST endpoint for the `send_message_to_mediator`
    action. Diagnostic only — uses HEAD/OPTIONS so we don't actually
    create a real chat message.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    headers = {"Authorization": f"Bearer {token}"}
    candidates = [
        # Generic actions endpoint patterns
        ("v1_actions", f"{base}/post-purchase/v1/claims/{claim_id}/actions"),
        ("v1_action_named", f"{base}/post-purchase/v1/claims/{claim_id}/actions/send_message_to_mediator"),
        # Specific named endpoints
        ("v1_send_message", f"{base}/post-purchase/v1/claims/{claim_id}/send-message"),
        ("v1_messages_mediator", f"{base}/post-purchase/v1/claims/{claim_id}/messages-to-mediator"),
        ("v1_mediator_message", f"{base}/post-purchase/v1/claims/{claim_id}/mediator-message"),
        ("v1_messages_post", f"{base}/post-purchase/v1/claims/{claim_id}/messages"),
        # v2 / mediations namespace
        ("v2_messages", f"{base}/post-purchase/v2/claims/{claim_id}/messages"),
        ("v1_mediations_messages", f"{base}/post-purchase/v1/mediations/{claim_id}/messages"),
    ]
    results = []
    async with httpx.AsyncClient() as http:
        for label, url in candidates:
            try:
                # Use OPTIONS — non-mutating, returns Allow header.
                r = await http.options(url, headers=headers, timeout=10.0)
                results.append({
                    "method": "OPTIONS",
                    "label": label,
                    "url": url,
                    "status": r.status_code,
                    "allow": r.headers.get("allow", ""),
                    "body_preview": r.text[:200],
                })
            except Exception as err:  # noqa: BLE001
                results.append({"label": label, "error": str(err)})
    return {"claim_id": claim_id, "results": results}


@router.get("/user-claims/{claim_id}/decide-probe")
async def claim_decide_probe(
    claim_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """GET-only probe — try several plausible 'select-resolution' / 'answer'
    endpoints with HEAD-style requests (HEAD or OPTIONS, no body) to see
    which ones exist for this claim. We then know which POST path to wire
    into the TG callback.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    headers = {"Authorization": f"Bearer {token}"}
    # Candidates: GET on listings, OPTIONS on mutating endpoints to discover
    # what verbs they accept without actually mutating anything.
    candidates = [
        # GET — list-style endpoints that might exist
        ("GET", "v1_decisions", f"{base}/post-purchase/v1/claims/{claim_id}/decisions"),
        ("GET", "v1_answers", f"{base}/post-purchase/v1/claims/{claim_id}/answers"),
        ("GET", "v1_resolutions", f"{base}/post-purchase/v1/claims/{claim_id}/resolutions"),
        ("GET", "v1_dispute", f"{base}/post-purchase/v1/claims/{claim_id}/dispute"),
        ("GET", "v1_mediation_actions", f"{base}/post-purchase/v1/mediations/{claim_id}/actions"),
        ("GET", "v1_mediation_decisions", f"{base}/post-purchase/v1/mediations/{claim_id}/decisions"),
        ("GET", "v1_player_actions", f"{base}/post-purchase/v1/claims/{claim_id}/player-actions"),
        # OPTIONS preflight — tells us allowed methods on a path
        ("OPTIONS", "options_messages", f"{base}/post-purchase/v1/claims/{claim_id}/messages"),
        ("OPTIONS", "options_expected_resolutions", f"{base}/post-purchase/v1/claims/{claim_id}/expected-resolutions"),
        ("OPTIONS", "options_decisions", f"{base}/post-purchase/v1/claims/{claim_id}/decisions"),
    ]
    results = []
    async with httpx.AsyncClient() as http:
        for method, label, url in candidates:
            try:
                if method == "GET":
                    r = await http.get(url, headers=headers, timeout=15.0)
                else:  # OPTIONS
                    r = await http.options(url, headers=headers, timeout=15.0)
                ct = (r.headers.get("content-type", "") or "").lower()
                meta: dict[str, Any] = {
                    "method": method,
                    "label": label,
                    "status": r.status_code,
                    "allow": r.headers.get("allow", ""),  # important for OPTIONS
                    "content_type": ct,
                    "body_preview": r.text[:300],
                }
                results.append(meta)
            except Exception as err:  # noqa: BLE001
                results.append({"method": method, "label": label, "error": str(err)})
    return {"claim_id": claim_id, "results": results}


@router.post("/user-claims/{claim_id}/send-message-action")
async def send_message_to_mediator_test(
    claim_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Test endpoint: try POST /claims/{id}/expected-resolutions/send_message_to_mediator
    with various body formats. Returns ML's raw response so we can iterate
    on body shape without polluting the keyboard.
    Body: { "text": "..." } — message content.
    """
    if pool is None:
        return {"error": "no_db"}
    text = (body or {}).get("text") or ""
    if not text:
        return {"error": "text_required"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    url = f"{base}/post-purchase/v1/claims/{claim_id}/expected-resolutions/send_message_to_mediator"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Try several body shapes
    shapes = [
        ("plain", {"message": text}),
        ("text_field", {"text": text}),
        ("body_field", {"body": text}),
        ("array_form", [{"message": text}]),
        ("attachments_empty", {"message": text, "attachments": []}),
    ]
    results = []
    async with httpx.AsyncClient() as http:
        for label, payload in shapes:
            try:
                r = await http.post(url, json=payload, headers=headers, timeout=20.0)
                ct = r.headers.get("content-type", "")
                results.append({
                    "shape": label,
                    "status": r.status_code,
                    "body": r.json() if "json" in ct else r.text[:300],
                })
                # If we hit 200 — stop, we created a real message; don't try more shapes
                if r.status_code < 400:
                    break
            except Exception as err:  # noqa: BLE001
                results.append({"shape": label, "error": str(err)})
    return {"claim_id": claim_id, "url": url, "results": results}


@router.post("/user-claims/{claim_id}/send-message")
async def send_claim_message(
    claim_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Post a message into a claim's mediation chat. For dispute/mediation
    stage claims, this is how the seller "selects" a resolution — by
    replying to the ML mediator's question.

    Body: { "text": "..." } — the message content.

    Returns ML's raw response so we can iterate on the right text format
    ("1" vs "Aceitar devolução" vs "Devolução").
    """
    if pool is None:
        return {"error": "no_db"}
    text = (body or {}).get("text") or ""
    if not text or not isinstance(text, str):
        return {"error": "text_required"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    base = "https://api.mercadolibre.com"
    url = f"{base}/post-purchase/v1/claims/{claim_id}/messages"
    payload = {"message": text}
    async with httpx.AsyncClient() as http:
        try:
            r = await http.post(
                url, json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=20.0,
            )
            ct = r.headers.get("content-type", "")
            return {
                "ok": r.status_code < 400,
                "status": r.status_code,
                "content_type": ct,
                "body": r.json() if "json" in ct else r.text[:600],
            }
        except Exception as err:  # noqa: BLE001
            return {"error": "exception", "detail": str(err)}


@router.post("/photo-experiments")
async def photo_experiment_start(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Start a photo A/B test for an item.

    Body:
      item_id: str (required)
      duration_days: int (3, 7, or 14; default 7)
      new_picture_id: str (optional — ML picture id of the new main photo)
      old_picture_id: str (optional)
      ai_asset_id: int (optional — FK to escalar_generated_assets)
      notes: str (optional)
    """
    if pool is None:
        return {"error": "no_db"}
    item_id = (body or {}).get("item_id")
    if not item_id or not isinstance(item_id, str):
        return {"error": "item_id_required"}
    duration_days = int((body or {}).get("duration_days", 7))
    if duration_days not in (3, 7, 14):
        return {"error": "duration_days_must_be_3_7_or_14"}
    return await photo_ab_dispatch_svc.start_experiment(
        pool, user.id, item_id, duration_days,
        new_picture_id=(body or {}).get("new_picture_id"),
        old_picture_id=(body or {}).get("old_picture_id"),
        ai_asset_id=(body or {}).get("ai_asset_id"),
        notes=(body or {}).get("notes"),
    )


@router.get("/photo-experiments")
async def photo_experiment_list(
    status: Optional[str] = Query(None, description="testing | completed | cancelled"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"experiments": []}
    await photo_ab_dispatch_svc.ensure_schema(pool)
    rows = await photo_ab_dispatch_svc.list_experiments(pool, user.id, status=status)
    return {"experiments": rows, "total": len(rows)}


@router.post("/photo-experiments/{experiment_id}/cancel")
async def photo_experiment_cancel(
    experiment_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    return await photo_ab_dispatch_svc.cancel_experiment(pool, user.id, experiment_id)


@router.post("/photo-experiments/{experiment_id}/close-now")
async def photo_experiment_close_now(
    experiment_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Force-close one experiment immediately (for testing the result-card
    flow without waiting for ends_at). Same code path as the cron uses.
    """
    if pool is None:
        return {"error": "no_db"}
    await photo_ab_dispatch_svc.ensure_schema(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, item_id, ai_asset_id, old_picture_id,
                   new_picture_id, duration_days, started_at, ends_at,
                   baseline_visits, baseline_orders
              FROM escalar_photo_experiments
             WHERE id = $1 AND user_id = $2 AND status = 'testing'
            """,
            experiment_id, user.id,
        )
    if not row:
        return {"error": "not_found_or_not_testing"}
    async with httpx.AsyncClient() as http:
        ok = await photo_ab_dispatch_svc._close_one_experiment(pool, http, dict(row))
    return {"ok": ok, "experiment_id": experiment_id}


# ── Orders cache (ml_user_orders — feeds daily-summary + photo A/B) ──────────


@router.get("/orders-probe")
async def orders_probe(
    days_back: int = Query(2, ge=0, le=60),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """TEST step: hit /orders/search directly for [today-days_back, today]
    BRT and return raw status + sample so we can verify ML token + shape
    before relying on the cache."""
    if pool is None:
        return {"error": "no_db"}
    return await ml_orders_svc.probe(pool, user.id, days_back=days_back)


@router.post("/orders/refresh")
async def orders_refresh(
    days_back: int = Query(14, ge=1, le=60),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Force-pull orders from ML for the last `days_back` BRT days into
    `ml_user_orders`. Daily-summary calls this implicitly with
    days_back=2 before aggregating; photo A/B calls it with
    days_back=duration_days+2 at start and at close. This endpoint is
    for manual smoke-tests / backfill."""
    if pool is None:
        return {"error": "no_db"}
    return await ml_orders_svc.refresh_for_period(pool, user.id, days_back=days_back)


@router.get("/orders")
async def orders_list(
    days: int = Query(7, ge=1, le=60),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Read cached orders for the last N BRT days. Cache-only, no
    ML fetch — call /orders/refresh first if you need fresh data."""
    if pool is None:
        return {"orders": []}
    await ml_orders_svc.ensure_schema(pool)
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    BRT_TZ = _tz(_td(hours=-3))
    end = _dt.now(BRT_TZ)
    start = end - _td(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT order_id, pack_id,
                   to_char(date_created AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS date_created,
                   status, total_amount, currency, items
              FROM ml_user_orders
             WHERE user_id = $1
               AND date_created BETWEEN $2 AND $3
             ORDER BY date_created DESC
             LIMIT 500
            """,
            user.id, start, end,
        )
    return {
        "orders": [dict(r) for r in rows],
        "total": len(rows),
        "fetchedAt": await ml_orders_svc.get_latest_fetched_at(pool, user.id),
    }


@router.delete("/photo-experiments/{experiment_id}")
async def delete_photo_experiment(
    experiment_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Delete a photo A/B experiment row outright. Used for cleanup of
    test rows created during smoke tests. Only deletes rows owned by
    the current user."""
    if pool is None:
        return {"error": "no_db"}
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM escalar_photo_experiments WHERE id = $1 AND user_id = $2",
            experiment_id, user.id,
        )
    deleted = result.endswith(" 1")
    return {"deleted": deleted, "experiment_id": experiment_id}


# ── Goals (manual targets per user/quarter) ──────────────────────────────────


@router.get("/goals")
async def goals_list(
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Active goals for the current user. `project` filters to one
    project (NULL = global goals)."""
    if pool is None:
        return {"goals": []}
    await goals_svc.ensure_schema(pool)
    project_filter = project if project else None
    goals = await goals_svc.list_active(pool, user.id, project_name=project_filter)
    return {"goals": goals, "total": len(goals)}


@router.post("/goals")
async def goals_create(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Create or update a goal. Body keys:
      id?            — update existing if set
      project_name?  — null = global goal
      kind           — 'lucro_liquido' | 'receita' | 'units' (default lucro)
      target_amount  — required, positive number
      period_type    — 'month' | 'quarter' | 'year' | 'custom' (default quarter)
      period_start?  — ISO date, required when period_type=custom
      period_end?    — ISO date, required when period_type=custom
      currency       — default 'BRL'
      notes?         — free text
    """
    if pool is None:
        return {"error": "no_db"}
    await goals_svc.ensure_schema(pool)

    target = (body or {}).get("target_amount")
    try:
        target_f = float(target)
    except (TypeError, ValueError):
        return {"error": "target_amount_required_positive_number"}
    if target_f <= 0:
        return {"error": "target_amount_must_be_positive"}

    from datetime import date as _date
    ps_str = (body or {}).get("period_start")
    pe_str = (body or {}).get("period_end")
    period_start = _date.fromisoformat(ps_str) if ps_str else None
    period_end = _date.fromisoformat(pe_str) if pe_str else None

    return await goals_svc.upsert_goal(
        pool, user.id,
        goal_id=body.get("id"),
        project_name=body.get("project_name"),
        kind=body.get("kind", "lucro_liquido"),
        target_amount=target_f,
        period_type=body.get("period_type", "quarter"),
        period_start=period_start,
        period_end=period_end,
        currency=body.get("currency", "BRL"),
        notes=body.get("notes"),
    )


@router.delete("/goals/{goal_id}")
async def goals_delete(
    goal_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Soft-delete (active=false)."""
    if pool is None:
        return {"error": "no_db"}
    return await goals_svc.deactivate_goal(pool, user.id, goal_id)


@router.post("/_admin/trigger-cron")
async def trigger_cron_admin(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(default={}),
):
    """Manually fire any of the scheduled cron jobs RIGHT NOW for the
    current user. Useful to verify everything works without waiting
    for 23:00 UTC daily summary or 23:30 UTC anomalies.

    Body: `{ jobs: ["daily_summary", "anomalies", "questions", "claims",
                    "messages", "promotions", "all"] }`
    Default `["all"]` — runs every dispatch in sequence.
    """
    if pool is None:
        return {"error": "no_db"}

    requested = (body or {}).get("jobs") or ["all"]
    if isinstance(requested, str):
        requested = [requested]

    results: dict[str, Any] = {}

    async def _run(name: str, coro_factory: Any) -> None:
        try:
            results[name] = await coro_factory()
        except Exception as err:  # noqa: BLE001
            import traceback
            results[name] = {
                "error": type(err).__name__,
                "message": str(err)[:300],
                "traceback": traceback.format_exc()[-800:],
            }

    do_all = "all" in requested

    if do_all or "daily_summary" in requested:
        await _run(
            "daily_summary",
            lambda: daily_summary_dispatch_svc._dispatch_for_user(pool, user.id),
        )

    if do_all or "anomalies" in requested:
        from v2.services import ml_anomalies as _anomalies_svc
        await _run(
            "anomalies",
            lambda: _anomalies_svc._dispatch_for_user(pool, user.id),
        )

    if do_all or "photo_ab" in requested:
        await _run(
            "photo_ab",
            lambda: photo_ab_dispatch_svc.dispatch_pending_results(pool),
        )

    if do_all or "positions" in requested:
        from v2.services import positions_refresh as _pr_svc
        await _run(
            "positions_refresh",
            lambda: _pr_svc._refresh_one_user(pool, user.id),
        )

    return {"user_id": user.id, "ran": list(results.keys()), "results": results}


@router.post("/daily-summary/preview")
async def daily_summary_preview(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(default={}),
):
    """Run the daily-summary dispatch immediately for the current user.

    Lets the seller test the format without waiting for the 20:00 BRT cron.
    Body (optional):
      { date: 'YYYY-MM-DD' }   — target a specific BRT day (default: yesterday)
      { force: true }          — bypass notify_daily_sales toggle
    """
    if pool is None:
        return {"error": "no_db"}

    target_date_str = (body or {}).get("date")
    force = bool((body or {}).get("force", False))
    target_date = None
    if target_date_str:
        try:
            from datetime import date as _date
            target_date = _date.fromisoformat(target_date_str)
        except ValueError:
            return {"error": "invalid_date_format", "expected": "YYYY-MM-DD"}

    try:
        result = await daily_summary_dispatch_svc._dispatch_for_user(
            pool, user.id, target_date=target_date, force=force,
        )
        return {"ok": True, **result}
    except Exception as err:  # noqa: BLE001
        return {"error": "dispatch_failed", "detail": str(err)}


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


# ── Returns probe (TEST step for Devoluções) ────────────────────────────
# ML's Seller Hub "Devoluções → Próximas a serem atendidas" surface returns
# that don't always show up via /post-purchase/v1/claims/search (those are
# claims; standalone returns are a separate post-purchase resource). This
# probe hits the most likely candidate endpoints and reports each one's
# status + body sample so we can pick the right one for the cache.

@router.get("/returns-probe")
async def returns_probe(
    order_id: Optional[str] = Query(None, description="Specific order_id to probe (e.g. one shown in ML Devoluções)"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Probe candidate ML endpoints for "Devoluções (Próximas a serem
    atendidas)". When order_id is passed, queries scoped to that specific
    order — much more likely to surface the right endpoint than blind
    listing.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    async with pool.acquire() as conn:
        ml_user_id = await conn.fetchval(
            "SELECT ml_user_id FROM ml_user_tokens WHERE user_id = $1", user.id,
        )

    headers = {"Authorization": f"Bearer {token}"}
    base = "https://api.mercadolibre.com"

    candidates: list[tuple[str, str]] = []
    # Generic listing — kept for completeness even though prior run showed 404
    candidates.append(("v1_returns_root", f"{base}/post-purchase/v1/returns"))
    candidates.append(("v1_my_returns", f"{base}/post-purchase/v1/my/returns"))
    # Seller scoped via /users/{id}/...
    if ml_user_id:
        candidates.append(("seller_claims_open", f"{base}/post-purchase/v1/claims/search?resource=order&seller={ml_user_id}&limit=5"))
        candidates.append(("seller_returns_search", f"{base}/post-purchase/v1/returns/search?seller={ml_user_id}&limit=5"))

    if order_id:
        # Most informative: target the specific number shown in the seller's
        # ML "Devoluções" view. Prior probe revealed it's NOT an order_id
        # (404 on /orders/{id}), so we now try several interpretations:
        # claim_id, return_id, pack_id, etc.
        candidates.append(("order_full", f"{base}/orders/{order_id}"))
        candidates.append(("packs_full", f"{base}/packs/{order_id}"))
        candidates.append(("claim_direct", f"{base}/post-purchase/v1/claims/{order_id}"))
        candidates.append(("return_direct", f"{base}/post-purchase/v1/returns/{order_id}"))
        candidates.append(("v2_return_direct", f"{base}/post-purchase/v2/returns/{order_id}"))
        candidates.append(("claims_by_pack", f"{base}/post-purchase/v1/claims/search?resource=pack&resource_id={order_id}"))
        candidates.append(("claims_by_shipping", f"{base}/post-purchase/v1/claims/search?resource=shipping&resource_id={order_id}"))
        candidates.append(("claims_by_resource_pair", f"{base}/post-purchase/v1/claims/search?resource=order&resource_id={order_id}"))
        # Player-role search — covers all claims where seller is involved,
        # filtered by the order_id-as-resource. Returns ALL claim types if
        # the resource_id matches anything ML knows about.
        if ml_user_id:
            candidates.append(("claims_by_player_seller", f"{base}/post-purchase/v1/claims/search?player_role=respondent&player_user_id={ml_user_id}&resource_id={order_id}"))
        # If the number is a shipment_id, this may surface it
        candidates.append(("shipments_full", f"{base}/shipments/{order_id}"))
        # Order search by external id (long shot)
        if ml_user_id:
            candidates.append(("orders_search_by_q", f"{base}/orders/search?seller={ml_user_id}&q={order_id}"))

    results = []
    async with httpx.AsyncClient() as http:
        for label, url in candidates:
            try:
                r = await http.get(url, headers=headers, timeout=15.0)
                results.append({
                    "label": label,
                    "url": url,
                    "status": r.status_code,
                    "body_preview": r.text[:600],
                })
            except Exception as err:  # noqa: BLE001
                results.append({"label": label, "url": url, "error": str(err)})
    return {"ml_user_id": ml_user_id, "order_id": order_id, "results": results}


# ── Messages probe (TEST step for Mensagens) ────────────────────────────
# Order chat between buyer↔seller surfaces under ML's "Mensagens" tab.
# Probable endpoint pattern: /messages/packs/{pack_id}/sellers/{seller_id}.
# To probe meaningfully we need an order_id the seller knows is the
# unread one; user passes it in the query string.

@router.get("/scraper/ip-probe")
async def scraper_ip_probe(
    user: CurrentUser = Depends(current_user),
):
    """Probe whether ML's seller-hub URL is reachable from Railway's IP at
    all, BEFORE we ask the seller to extract their storage_state cookies.
    Opens https://www.mercadolivre.com.br/ in headless Chromium and reports
    the page title + first chunk of body — Cloudflare blocks return
    'Access denied' / 'Pardon Our Interruption' / 'Cloudflare', success
    returns the normal homepage.
    """
    import asyncio as _asyncio
    from concurrent.futures import ThreadPoolExecutor as _Pool
    from playwright.sync_api import sync_playwright as _sync_playwright

    def _probe() -> dict[str, Any]:
        with _sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/129.0.0.0 Safari/537.36"
                    ),
                    locale="pt-BR",
                )
                page = ctx.new_page()
                page.set_default_timeout(20000)
                resp = page.goto(
                    "https://www.mercadolivre.com.br/",
                    wait_until="domcontentloaded",
                )
                status = resp.status if resp else 0
                title = page.title() or ""
                body_preview = page.content()[:1500]
                # Cloudflare / bot-detection markers
                lowered = body_preview.lower()
                blocked_markers = [
                    "access denied",
                    "pardon our interruption",
                    "cloudflare",
                    "checking your browser",
                    "captcha",
                    "blocked",
                ]
                blocked_hits = [m for m in blocked_markers if m in lowered]
                ctx.close()
                return {
                    "status": status,
                    "title": title,
                    "blockedMarkers": blocked_hits,
                    "blocked": bool(blocked_hits),
                    "bodyPreview": body_preview[:600],
                }
            finally:
                browser.close()

    try:
        result = await _asyncio.to_thread(_probe)
        return result
    except Exception as err:  # noqa: BLE001
        return {"error": "probe_exception", "detail": str(err)}


@router.get("/scraper/chat-read")
async def scraper_chat_read(
    pack_id: str = Query(..., description="Pack ID from ML — first part of the URL after /mensagens/"),
    claim_id: Optional[str] = Query(None, description="Optional claim_id for mediation chat"),
    user: CurrentUser = Depends(current_user),
):
    """Stage-1 test: drive Chromium against ML's chat URL with the stored
    storage_state and return the visible message bubbles. If ML_SCRAPER_
    STORAGE_STATE_B64 isn't set yet, returns storage_state_missing.
    """
    try:
        result = await ml_scraper_chat_svc.read_chat(pack_id, claim_id)
        return {"pack_id": pack_id, "claim_id": claim_id, "messages": result, "count": len(result)}
    except ml_scraper_chat_svc.ScraperChatError as err:
        return {"error": str(err)}
    except Exception as err:  # noqa: BLE001
        return {"error": "scraper_exception", "detail": str(err)}


@router.post("/scraper/chat-send")
async def scraper_chat_send(
    user: CurrentUser = Depends(current_user),
    body: dict[str, Any] = Body(...),
):
    """Stage-1 test: send a message via Playwright. body shape:
      { pack_id: '...', claim_id: '...' (optional), text: '...' }
    """
    pack_id = (body or {}).get("pack_id")
    claim_id = (body or {}).get("claim_id")
    text = (body or {}).get("text") or ""
    if not pack_id or not text:
        return {"error": "pack_id_and_text_required"}
    try:
        result = await ml_scraper_chat_svc.send_chat(pack_id, text, claim_id)
        return {"pack_id": pack_id, "claim_id": claim_id, **result}
    except ml_scraper_chat_svc.ScraperChatError as err:
        return {"error": str(err)}
    except Exception as err:  # noqa: BLE001
        return {"error": "scraper_exception", "detail": str(err)}


@router.get("/messages-probe")
async def messages_probe(
    order_id: str = Query(..., description="ML order id to probe messages for"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Probe candidate ML endpoints for the order-message chat. Pass an
    order_id the seller has unread messages on (visible in ML's Mensagens
    tab) — different endpoints scope by order vs. pack vs. seller.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    # Pull the seller's ML user_id (need it for /messages/packs/.../sellers/{id})
    async with pool.acquire() as conn:
        ml_user_id = await conn.fetchval(
            "SELECT ml_user_id FROM ml_user_tokens WHERE user_id = $1",
            user.id,
        )

    headers = {"Authorization": f"Bearer {token}"}
    base = "https://api.mercadolibre.com"
    # All 10 prior candidates returned 404 even with the hex resource_id from
    # the webhook payload. Try user-scoped paths + private/marketplace/post
    # variants — common ML conventions we haven't hit yet.
    candidates = [
        # User-scoped
        ("users_messages", f"{base}/users/{ml_user_id}/messages"),
        ("users_messages_inbox", f"{base}/users/{ml_user_id}/messages/inbox"),
        ("users_messages_received", f"{base}/users/{ml_user_id}/messages/received"),
        # Per-resource direct (with hex id from webhook)
        ("message_resource_id", f"{base}/messages/{order_id}"),
        ("messages_v1_resource", f"{base}/messages/v1/{order_id}"),
        ("private_messages", f"{base}/messages/private/{order_id}"),
        ("marketplace_messages", f"{base}/marketplace/messages/{order_id}"),
        ("messaging_v1_resource", f"{base}/messaging/v1/{order_id}"),
        # Site-scoped
        ("site_messages_user", f"{base}/sites/MLB/users/{ml_user_id}/messages"),
        # Pack option variants (the option could be different)
        ("messages_pack_post_sale", f"{base}/messages/packs/{order_id}/sellers/{ml_user_id}/option/post_sale"),
        ("messages_pack_questions", f"{base}/messages/packs/{order_id}/sellers/{ml_user_id}/option/questions"),
        # Conversations
        ("conversations_user", f"{base}/conversations/users/{ml_user_id}"),
        ("conversations_resource", f"{base}/conversations/{order_id}"),
    ]

    results = []
    async with httpx.AsyncClient() as http:
        for label, url in candidates:
            try:
                r = await http.get(url, headers=headers, timeout=15.0)
                results.append({
                    "label": label,
                    "url": url,
                    "status": r.status_code,
                    "body_preview": r.text[:600],
                })
            except Exception as err:  # noqa: BLE001
                results.append({"label": label, "url": url, "error": str(err)})
    return {"results": results, "ml_user_id": ml_user_id}


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

    # Aggregate topic distribution across the user's full notice history.
    # Helps diagnose which event types ML actually pushes to our webhook —
    # e.g. presence of `messages` topic confirms we can build a TG dispatch
    # for buyer-seller chat without scraping ML's restricted /messages API.
    async with pool.acquire() as conn:
        topic_rows = await conn.fetch(
            """
            SELECT COALESCE(topic, '(null)') AS topic, COUNT(*) AS n
              FROM ml_notices
             WHERE user_id = $1
             GROUP BY topic
             ORDER BY n DESC
            """,
            user.id,
        )
    out["topicCounts"] = {r["topic"]: int(r["n"]) for r in topic_rows}

    # Sample raw payloads for the messages topic — needed to fix the empty
    # rich-card we saw in production (cron sent 3 cards with only the title
    # because raw didn't have the fields ml_normalize expected).
    async with pool.acquire() as conn:
        msg_rows = await conn.fetch(
            """
            SELECT notice_id, label, description, raw,
                   to_char(from_date AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS from_date
              FROM ml_notices
             WHERE user_id = $1 AND topic = 'messages'
             ORDER BY from_date DESC NULLS LAST
             LIMIT 3
            """,
            user.id,
        )
    out["messagesSample"] = []
    for r in msg_rows:
        raw = r["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                pass
        out["messagesSample"].append({
            "notice_id": r["notice_id"],
            "label": r["label"],
            "description": r["description"],
            "from_date": r["from_date"],
            "raw_keys": sorted(raw.keys()) if isinstance(raw, dict) else None,
            "raw": raw,
        })

    # 4. ENV config (masked)
    bot_token = _os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    out["env"] = {
        "telegramBotTokenSet": bool(bot_token),
        "telegramBotTokenMasked": (bot_token[-6:] if len(bot_token) >= 6 else "(short)") if bot_token else None,
        "noticesSyncIntervalMin": _os.environ.get("NOTICES_SYNC_INTERVAL_MIN", "5"),
    }

    # 5. ML OAuth status — if user's token is dead, cron cannot fetch /communications/notices.
    # The column name on ml_user_tokens is `access_token_expires_at`, not
    # `expires_at` — earlier diag had this misspelled and always returned null.
    try:
        token_row = await ml_oauth_svc.load_user_tokens(pool, user.id)
        expires_at = token_row.get("access_token_expires_at") if token_row else None
        out["mlOauth"] = {
            "hasToken": bool(token_row and token_row.get("access_token")),
            "mlUserId": token_row.get("ml_user_id") if token_row else None,
            "expiresAt": expires_at.isoformat() if expires_at else None,
            "lastRefreshedAt": (
                token_row["last_refreshed_at"].isoformat()
                if token_row and token_row.get("last_refreshed_at") else None
            ),
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


@router.post("/items/refresh-margins")
async def refresh_item_margins_endpoint(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger ml_item_margin_cache refresh для всех SKU юзера.

    Без этого endpoint margin собирается только nightly cron'ом — после
    добавления нового SKU или первой загрузки ждёшь до утра. Здесь
    запускаем тот же refresh_user_item_margins синхронно (~5-10s для
    ~200 SKU). Вызывать от UI кнопкой или через console fetch:

      fetch('/api/v2-proxy/escalar/items/refresh-margins', {method:'POST'})
        .then(r=>r.json()).then(console.log)

    После 200 OK профит будет в orders_v2 TG-уведомлениях для следующих
    продаж и в /escalar/products.
    """
    if pool is None:
        return {"error": "no_db"}
    legacy_db.set_current_user_id(user.id)
    try:
        from v2.services import ml_item_margin as _margin_svc
        await _margin_svc.ensure_schema(pool)
        result = await _margin_svc.refresh_user_item_margins(pool, user.id, period_months=3)
        return {
            "ok": True,
            "user_id": user.id,
            "computed": result.get("computed"),
            "items_total": result.get("items_total"),
            "projects": result.get("projects"),
        }
    except Exception as err:  # noqa: BLE001
        import traceback as _tb
        return {
            "ok": False,
            "error": str(err),
            "trace": _tb.format_exc()[-500:],
        }


@router.get("/user-promotions/raw-debug")
async def user_promotions_raw_debug(
    item_id: str = Query(..., min_length=1, description="MLB id"),
    promotion_id: str = Query(..., min_length=1, description="Promotion id"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Diagnostic: показывает что в БД для этой promo + что ML отдаёт сейчас.

    Используется когда accept падает с `Offer id is required` или похожим —
    видно есть ли offer_id в текущем кэше и в свежем response от ML, чтобы
    понять надо ли менять _upsert_offer или нет.
    """
    import traceback as _tb
    if pool is None:
        return {"error": "no_db"}
    try:
        legacy_db.set_current_user_id(user.id)
        mlb = item_id.upper().strip()
    except Exception as err:  # noqa: BLE001
        return {"error": "bind_failed", "detail": str(err), "trace": _tb.format_exc()[-500:]}

    # 1. Что у нас в БД (только реально существующие колонки —
    # meli/seller_percentage не сохраняются в schema, они в raw JSON).
    db_row = None
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT promotion_id, promotion_type, sub_type, status, offer_id,
                       original_price, deal_price, discount_percentage,
                       fetched_at, start_date, finish_date,
                       accepted_at, dismissed_at, raw
                  FROM ml_user_promotions
                 WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
                """,
                user.id, mlb, promotion_id.strip(),
            )
        if row:
            db_row = dict(row)
            for k, v in list(db_row.items()):
                if hasattr(v, "isoformat"):
                    db_row[k] = v.isoformat()
            # raw — это JSONB, может прийти как str или dict.
            if isinstance(db_row.get("raw"), str):
                try:
                    db_row["raw"] = json.loads(db_row["raw"])
                except Exception:  # noqa: BLE001
                    pass
    except Exception as err:  # noqa: BLE001
        return {"error": "db_query_failed", "detail": str(err), "trace": _tb.format_exc()[-500:]}

    # 2. Что ML отдаёт сейчас (raw, без обработки).
    fresh = None
    fresh_err = None
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
        url = f"https://api.mercadolibre.com/seller-promotions/items/{mlb}?app_version=v2"
        async with httpx.AsyncClient() as http:
            r = await http.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
        try:
            body_value = r.json() if r.status_code == 200 else r.text[:500]
        except Exception:  # noqa: BLE001
            body_value = r.text[:500]
        fresh = {"status": r.status_code, "body": body_value}
    except Exception as err:  # noqa: BLE001
        fresh_err = f"{type(err).__name__}: {err}"

    # 3. Подобрать matching offer из raw response по promotion_id.
    matching_offer: Optional[dict] = None
    if isinstance(fresh, dict) and isinstance(fresh.get("body"), list):
        for o in fresh["body"]:
            if isinstance(o, dict) and str(o.get("id") or o.get("promotion_id") or "") == promotion_id.strip():
                matching_offer = o
                break

    return {
        "user_id": user.id,
        "item_id": mlb,
        "promotion_id": promotion_id.strip(),
        "db": db_row,
        "ml_fresh": fresh,
        "ml_fresh_error": fresh_err,
        "matching_offer_in_fresh": matching_offer,
        "hint": (
            "Проверь поле 'offer_id' в db и в matching_offer_in_fresh. "
            "Если в БД None но в fresh есть — нужно re-upsert. "
            "Если в fresh тоже None для этого type — ML формирует offer_id "
            "только при отдельном request, нужен другой flow accept."
        ),
    }


class _ItemPriceShiftIn(__import__("pydantic").BaseModel):
    """Сдвиг цены листинга на ±X% — для per-sale TG-кнопок «Поднять/Опустить»."""
    user_id: int
    item_id: str           # MLB id
    delta_pct: float       # +10 / −10 / любой проценты
    # base: 'sale' (от sale_price) | 'listing' (от текущей цены листинга).
    # По дефолту 'sale' — кнопки в TG приходят после конкретной продажи и
    # пользователь думает в терминах sale_price которую он только что увидел.
    base: Optional[str] = "sale"
    sale_price: Optional[float] = None  # обязательно если base="sale"
    # Если true — сначала DELETE pricing-automation для item, потом PUT
    # /items {price}. Используется когда predыдущий вызов вернул
    # needs_dyn_confirm и user подтвердил через TG-кнопку sdyn:.
    disable_dyn: Optional[bool] = False


@router.post("/items/price-shift")
async def items_price_shift(
    body: _ItemPriceShiftIn,
    pool=Depends(get_pool),
):
    """Поднимает или опускает цену листинга на ±X% от sale/listing-base.

    PUT /items/{id} с новой `price`. ML может отбить с 400 если:
      - price ≤ 0
      - изменение > 50% за неделю (limit per regulation)
    Возвращает {ok, old_price, new_price, delta_pct}.

    Auth: server-to-server через webhook → no cookie. Шифрование URL не
    требуется т.к. внутри Railway VPC. Header LS_INTERNAL_API_TOKEN читается
    но пока не enforced (single-tenant deploy).
    """
    if pool is None:
        return {"error": "no_db"}

    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, body.user_id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    mlb = body.item_id.upper()
    item_url = f"https://api.mercadolibre.com/items/{mlb}"

    # Resolve base price.
    if (body.base or "sale").lower() == "sale" and body.sale_price and body.sale_price > 0:
        base_price = float(body.sale_price)
    else:
        try:
            async with httpx.AsyncClient() as http:
                gr = await http.get(
                    item_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
            if gr.status_code >= 400:
                return {"error": "ml_item_fetch_rejected",
                        "status": gr.status_code, "detail": gr.text[:300]}
            base_price = float((gr.json() or {}).get("price") or 0.0)
        except Exception as err:  # noqa: BLE001
            return {"error": "ml_item_fetch_failed", "detail": str(err)}
        if base_price <= 0:
            return {"error": "no_current_price"}

    new_price = round(base_price * (1.0 + float(body.delta_pct) / 100.0), 2)
    if new_price <= 0:
        return {"error": "non_positive_price",
                "base_price": base_price, "attempted": new_price}

    dyn_disabled = False
    async with httpx.AsyncClient() as http:
        # Optional preflight: disable Dynamic Pricing if user confirmed via
        # TG button. Saves prev rule into ml_user_pricing_history.
        if body.disable_dyn:
            try:
                gar = await http.get(
                    f"https://api.mercadolibre.com/pricing-automation/items/{mlb}/automation",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
                if gar.status_code == 200:
                    cur_auto = gar.json() or {}
                    rule = (cur_auto.get("item_rule") or {}).get("rule_id") or "INT_EXT"
                    try:
                        min_p = float(cur_auto.get("min_price") or 0)
                    except (TypeError, ValueError):
                        min_p = 0.0
                    try:
                        max_p = float(cur_auto.get("max_price") or 0)
                    except (TypeError, ValueError):
                        max_p = 0.0
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS ml_user_pricing_history (
                              user_id INTEGER NOT NULL,
                              item_id TEXT NOT NULL,
                              rule_id TEXT,
                              min_price NUMERIC,
                              max_price NUMERIC,
                              raw JSONB,
                              disabled_at TIMESTAMPTZ DEFAULT NOW()
                            );
                            """
                        )
                        await conn.execute(
                            """
                            INSERT INTO ml_user_pricing_history
                              (user_id, item_id, rule_id, min_price, max_price, raw)
                            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                            """,
                            body.user_id, mlb, rule, min_p, max_p,
                            json.dumps(cur_auto, default=str),
                        )
            except Exception:  # noqa: BLE001
                pass
            try:
                dr = await http.delete(
                    f"https://api.mercadolibre.com/pricing-automation/items/{mlb}/automation",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
                if dr.status_code in (200, 204, 404):
                    dyn_disabled = True
            except Exception as err:  # noqa: BLE001
                return {"error": "ml_dyn_disable_failed", "detail": str(err)}

        try:
            pr = await http.put(
                item_url,
                json={"price": new_price},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except Exception as err:  # noqa: BLE001
            return {"error": "ml_price_update_failed", "detail": str(err)}
    if pr.status_code >= 400:
        # Detect Dynamic Pricing block — return needs_dyn_confirm so TG
        # webhook can render confirmation prompt instead of an error toast.
        if pr.status_code == 400 and "item.price.not_modifiable" in (pr.text or ""):
            return {
                "ok": False,
                "needs_dyn_confirm": True,
                "reason": "dynamic_pricing_active",
                "item_id": mlb,
                "old_price": base_price,
                "attempted_new_price": new_price,
                "delta_pct": float(body.delta_pct),
            }
        return {"error": "ml_price_update_rejected",
                "status": pr.status_code, "detail": pr.text[:300],
                "old_price": base_price, "attempted_new_price": new_price,
                "dyn_disabled": dyn_disabled}

    # Audit price-shift event (sales TG button sup:/sdn:).
    try:
        from v2.services import escalar_audit as _audit
        await _audit.log_event(
            pool,
            user_id=body.user_id,
            action="price_raised" if body.delta_pct >= 0 else "price_lowered",
            target_type="item",
            target_id=mlb,
            user_action="sale_price_shift",
            metadata={
                "old_price": base_price,
                "new_price": new_price,
                "delta_pct": float(body.delta_pct),
                "dyn_disabled": dyn_disabled,
            },
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": True,
        "item_id": mlb,
        "old_price": base_price,
        "new_price": new_price,
        "delta_pct": float(body.delta_pct),
        "base": body.base or "sale",
        "dyn_disabled": dyn_disabled,
    }


class _PromoActionIn(__import__("pydantic").BaseModel):
    # actions:
    #   "accept" / "reject" / "accept_with_raise" — для status=candidate
    #     (исходный flow: ML предложил, seller решает).
    #   "exit"  — для status=started: выйти из УЖЕ активной акции (DELETE
    #     /seller-promotions/items/{ITEM_ID}). Используется на SMART/UNHEALTHY
    #     которые ML auto-opt-in включила.
    #   "raise_only" — для status=started: поднять цену листинга (PUT /items/{ID})
    #     без accept/reject. Промо остаётся активной, но cena покупателя растёт
    #     обратно к оригиналу. raise_pct/raise_mode те же что и в accept_with_raise.
    action: str
    user_id: int
    promotion_id: str
    item_id: str
    # accept_with_raise: % uplift applied to listing price BEFORE accept.
    # If raise_pct is explicit, used as-is. Otherwise raise_mode picks:
    #   "linear" (default) → raise_pct = entrada_discount_pct
    #     → effective price ends slightly below original (≈ −D% of D%).
    #   "exact" → raise_pct = D / (1 − D/100)
    #     → effective price lands on original exactly.
    raise_pct: Optional[float] = None
    raise_mode: Optional[str] = None  # "linear" | "exact"


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


class _OrderNoticeFromWebhookIn(__import__("pydantic").BaseModel):
    user_id: int
    topic: str
    resource: Optional[str] = None
    enriched: dict[str, Any]


@router.post("/orders/notice-from-webhook")
async def orders_notice_from_webhook(
    body: _OrderNoticeFromWebhookIn,
    pool=Depends(get_pool),
):
    """Server-to-server endpoint — called by Next.js ml-webhook for orders_v2 /
    orders topics. Routes through the SAME pipeline as ml_backfill (Python
    _enrich_order_with_margin → normalize_event → upsert), so real-time TG
    notifications get the same profit/margin/break-even block as backfill ones.

    Без этого endpoint'а Next.js использовал свой `notice-normalize.ts` (TS) —
    он не знает про ml_item_margin_cache и break-even tracker, поэтому продажи
    приходили без блока маржи. Теперь и backfill, и live webhook идут через
    единый Python pipeline.

    Auth: no cookie — intra-Railway call. Webhook validates ML's
    application_id before triggering this.
    """
    if pool is None:
        return {"error": "no_db", "ok": False}

    enriched = dict(body.enriched or {})
    topic = body.topic
    resource = body.resource

    import logging as _lg
    _log_n = _lg.getLogger("escalar.orders_notice")

    # Pre-enrich with cached unit margin (for orders_v2 / orders).
    # Mirrors ml_backfill._upsert_batch's logic.
    enrichment_status = "skipped"
    if topic in ("orders_v2", "orders"):
        try:
            async with pool.acquire() as conn:
                await ml_backfill_svc._enrich_order_with_margin(
                    conn, body.user_id, enriched, pool=pool,
                )
            enrichment_status = (
                "with_margin_breakeven" if (
                    enriched.get("_margin") and enriched.get("_breakeven")
                ) else (
                    "with_margin" if enriched.get("_margin") else (
                        "no_margin_in_cache" if not enriched.get("_margin") else "partial"
                    )
                )
            )
        except Exception as err:  # noqa: BLE001
            enrichment_status = f"exception:{err}"
            _log_n.warning(
                "enrich order margin failed user=%s resource=%s: %s",
                body.user_id, resource, err,
            )
            try:
                from v2.services import tg_admin_alerts as _alerts
                await _alerts.send_admin_alert(
                    title="Order enrichment exception",
                    detail=(
                        f"user={body.user_id} resource={resource}\n"
                        f"exception: {err}"
                    ),
                    severity="error",
                    service="escalar/orders/notice-from-webhook",
                    deduplicate_key=f"order_enrich_exc:{type(err).__name__}",
                )
            except Exception:  # noqa: BLE001
                pass

    try:
        notice = ml_normalize_svc.normalize_event(topic, resource, enriched)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "error": f"normalize_failed: {err}"}

    # Upsert via shared helper (handles JSON serialization + ON CONFLICT).
    try:
        inserted = await ml_notices_svc.upsert_normalized(pool, body.user_id, notice)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "error": f"upsert_failed: {err}", "notice": notice}

    has_margin_block = bool(notice.get("description") and "Margem variável" in notice["description"])
    _log_n.info(
        "orders notice user=%s resource=%s enrichment=%s has_margin=%s tags=%s",
        body.user_id, resource, enrichment_status, has_margin_block, notice.get("tags"),
    )

    # If we expected a margin block (item is in cache + sale_price>0) but it
    # didn't end up in the rendered notice — that's a regression worth alerting.
    # Skip if cancelled (no margin expected) or if status was missing
    # entirely (enrichment failure, not regression).
    notice_tags_for_alert = notice.get("tags") or []
    if isinstance(notice_tags_for_alert, list):
        notice_tags_set = {str(t).upper() for t in notice_tags_for_alert}
    else:
        notice_tags_set = set()
    skip_alert = (
        "CANCELLED" in notice_tags_set
        or "INVALID" in notice_tags_set
        or enrichment_status == "no_margin_in_cache"  # known cause, no need to alert
    )
    if (
        topic in ("orders_v2", "orders")
        and not has_margin_block
        and not skip_alert
    ):
        try:
            from v2.services import tg_admin_alerts as _alerts
            await _alerts.send_admin_alert(
                title="Order notice without margin block",
                detail=(
                    f"user={body.user_id} resource={resource}\n"
                    f"enrichment={enrichment_status}\n"
                    f"tags={notice_tags_for_alert}\n"
                    f"Means: ml_item_margin_cache miss for item, OR no sale_price, OR normalize regression."
                ),
                severity="warn",
                service="escalar/orders/notice-from-webhook",
                deduplicate_key=f"order_no_margin:{enrichment_status}",
            )
        except Exception:  # noqa: BLE001
            pass

    return {
        "ok": True,
        "inserted": inserted,
        "notice_id": notice.get("notice_id"),
        "tags": notice.get("tags") or [],
        "enrichment_status": enrichment_status,
        "has_margin_block": has_margin_block,
    }


@router.get("/orders/{order_id}/replay-notice")
async def order_replay_notice(
    order_id: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Diagnostic — replays the orders_v2 notice pipeline for an existing
    order and shows where margin/breakeven enrichment landed (or didn't).

    Steps mirrored:
      1. Pull saved notice from ml_notices (by notice_id orders_v2:{order_id}).
      2. Try to fetch full order from ML API to re-enrich (so we see what
         live webhook would inject).
      3. Run _enrich_order_with_margin → log _margin/_breakeven additions.
      4. Run normalize_event → return resulting description.

    Returns enough state to tell whether:
      - item_id is in ml_item_margin_cache (margin block won't render if not)
      - sale_price could be parsed (margin needs unit_price > 0)
      - break-even tracker initialized for project
    """
    if pool is None:
        return {"error": "no_db"}

    notice_id = f"orders_v2:{order_id}"
    out: dict[str, Any] = {"order_id": order_id, "notice_id": notice_id}

    # 1. Saved TG notice
    async with pool.acquire() as conn:
        saved = await conn.fetchrow(
            """
            SELECT description, raw, tags, updated_at
              FROM ml_notices
             WHERE user_id = $1 AND notice_id = $2
            """,
            user.id, notice_id,
        )
    if saved:
        raw = saved["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        out["saved_notice"] = {
            "tags": json.loads(saved["tags"]) if isinstance(saved["tags"], str) else (saved["tags"] or []),
            "description_preview": (saved["description"] or "")[:600],
            "has_margin_block": bool(saved["description"] and "argem" in (saved["description"] or "")),
            "has_breakeven_block": bool(saved["description"] and "reak-even" in (saved["description"] or "")),
            "raw_keys": sorted(list(raw.keys()))[:30] if isinstance(raw, dict) else None,
            "updated_at": saved["updated_at"].isoformat() if saved["updated_at"] else None,
        }
    else:
        out["saved_notice"] = None

    # 2. Fetch full order from ML to replay enrichment
    try:
        token, _exp, _ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {**out, "error": "oauth_failed", "detail": str(err)}

    enriched: dict[str, Any] = {}
    async with httpx.AsyncClient() as http:
        gr = await http.get(
            f"https://api.mercadolibre.com/orders/{order_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if gr.status_code >= 400:
            return {**out, "error": "ml_order_fetch_failed", "status": gr.status_code, "detail": gr.text[:300]}
        try:
            enriched = gr.json() or {}
        except Exception:  # noqa: BLE001
            enriched = {}

    # 3. Extract first item_id + sale_price (same logic as _enrich_order_with_margin)
    items_arr = enriched.get("order_items") or enriched.get("items") or []
    first = items_arr[0] if items_arr and isinstance(items_arr[0], dict) else None
    item_id = ""
    sale_price = 0.0
    if first:
        inner = first.get("item") if isinstance(first.get("item"), dict) else first
        if isinstance(inner, dict):
            item_id = str(inner.get("id") or inner.get("mlb") or "").strip().upper()
        if not item_id:
            item_id = str(first.get("mlb") or "").strip().upper()
        try:
            sale_price = float(first.get("unit_price") or 0.0)
        except (TypeError, ValueError):
            sale_price = 0.0
    out["extracted"] = {
        "item_id": item_id,
        "sale_price": sale_price,
        "items_count": len(items_arr),
        "status": enriched.get("status"),
        "date_created": enriched.get("date_created"),
    }

    # 4. Check margin cache directly
    margin_cache: Optional[dict[str, Any]] = None
    if item_id:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT computed_at FROM ml_item_margin_cache
                 WHERE user_id = $1 AND item_id = $2 AND period_months = 3
                """,
                user.id, item_id,
            )
        if row:
            margin_cache = {
                "exists": True,
                "computed_at": row["computed_at"].isoformat() if row["computed_at"] else None,
            }
        else:
            margin_cache = {"exists": False}
    out["margin_cache_for_item"] = margin_cache

    # 5. Run _enrich_order_with_margin and capture _margin / _breakeven
    enrich_result = {"called": False, "exception": None}
    if item_id and sale_price > 0 and pool is not None:
        try:
            async with pool.acquire() as conn:
                await ml_backfill_svc._enrich_order_with_margin(
                    conn, user.id, enriched, pool=pool,
                )
            enrich_result["called"] = True
        except Exception as err:  # noqa: BLE001
            enrich_result["exception"] = str(err)
    out["enrichment"] = enrich_result
    out["margin_injected"] = bool(enriched.get("_margin"))
    out["breakeven_injected"] = bool(enriched.get("_breakeven"))
    if enriched.get("_margin"):
        m = enriched["_margin"]
        out["margin_summary"] = {
            "project": m.get("project"),
            "margin_variable_pct": (m.get("unit") or {}).get("margin_variable_pct"),
            "profit_variable": (m.get("unit") or {}).get("profit_variable"),
        }
    if enriched.get("_breakeven"):
        out["breakeven_summary"] = enriched["_breakeven"]

    # 6. Run normalize_event to see what description would emerge
    try:
        notice = ml_normalize_svc.normalize_event(
            "orders_v2", f"/orders/{order_id}", enriched,
        )
        out["replayed_notice"] = {
            "label": notice.get("label"),
            "description_preview": (notice.get("description") or "")[:1500],
            "tags": notice.get("tags") or [],
        }
    except Exception as err:  # noqa: BLE001
        out["replayed_notice"] = {"error": str(err)}

    return out


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

    # Resolve the entrada discount % from the offer (either derived from
    # original/deal prices or explicit discount_percentage). This is "D" in
    # the formulas below.
    entrada_pct: Optional[float] = None
    try:
        orig_p = float(offer.get("original_price") or 0)
        deal_p = float(offer.get("deal_price") or 0)
        if orig_p > 0 and 0 < deal_p < orig_p:
            entrada_pct = round((1 - deal_p / orig_p) * 100, 2)
    except (TypeError, ValueError):
        entrada_pct = None
    if entrada_pct is None:
        try:
            entrada_pct = float(offer.get("discount_percentage") or 0) or None
        except (TypeError, ValueError):
            entrada_pct = None

    # raise_pct selection:
    #   1. body.raise_pct override (explicit % from caller) wins
    #   2. raise_mode "exact" → D / (1 − D/100), so post-raise entrada
    #      price = original exactly.
    #   3. raise_mode "linear" (default) → raise_pct = D, simpler mental
    #      model, lands slightly below original.
    if body.raise_pct is not None:
        raise_pct = float(body.raise_pct)
    elif entrada_pct is not None and entrada_pct > 0:
        if (body.raise_mode or "").lower() == "exact" and entrada_pct < 100:
            raise_pct = round(entrada_pct / (1 - entrada_pct / 100.0), 2)
        else:
            raise_pct = round(entrada_pct, 2)
    else:
        raise_pct = 15.0
    raise_info: dict = {}

    async with httpx.AsyncClient() as http:
        # ── Two-step "raise then accept" flow ─────────────────────────────────
        # Lifts the listing price by raise_pct BEFORE accepting the promo so the
        # promo's mandatory discount lands close to the seller's original price.
        # ML auto-recomputes the offer's max_discounted_price/etc. against the
        # new listing price, so we re-fetch after the PUT and accept at the
        # refreshed entrada. If anything fails partway, we keep the already-
        # raised price (rollback would need its own retry-safe path).
        if body.action == "accept_with_raise":
            item_url = f"https://api.mercadolibre.com/items/{mlb}"
            try:
                gr = await http.get(
                    item_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
            except Exception as err:  # noqa: BLE001
                return {"error": "ml_item_fetch_failed", "detail": str(err)}
            if gr.status_code >= 400:
                return {
                    "error": "ml_item_fetch_rejected",
                    "status": gr.status_code,
                    "detail": gr.text[:300],
                }
            try:
                old_price = float((gr.json() or {}).get("price") or 0.0)
            except (TypeError, ValueError):
                old_price = 0.0
            if old_price <= 0:
                return {"error": "no_current_price"}
            new_price = round(old_price * (1.0 + raise_pct / 100.0), 2)
            try:
                pr = await http.put(
                    item_url,
                    json={"price": new_price},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    timeout=15.0,
                )
            except Exception as err:  # noqa: BLE001
                return {"error": "ml_price_update_failed", "detail": str(err)}
            if pr.status_code >= 400:
                return {
                    "error": "ml_price_update_rejected",
                    "status": pr.status_code,
                    "detail": pr.text[:300],
                    "old_price": old_price,
                    "attempted_new_price": new_price,
                }
            raise_info = {
                "old_price": old_price,
                "new_price": new_price,
                "raise_pct": raise_pct,
                "raise_mode": (body.raise_mode or "linear").lower(),
                "entrada_pct": entrada_pct,
            }

            # Re-fetch the offer with the new listing price so we accept at
            # the refreshed entrada (max_discounted_price recomputed by ML).
            try:
                await ml_user_promotions_svc.refresh_user_promotions(
                    pool, body.user_id, item_ids=[mlb],
                )
            except Exception:  # noqa: BLE001
                pass
            offer = await ml_user_promotions_svc.get_offer(
                pool, body.user_id, body.item_id, body.promotion_id,
            )
            if not offer:
                return {
                    "error": "offer_not_in_cache_after_raise",
                    "raise": raise_info,
                    "hint": "Price raised but offer disappeared. Check ML side.",
                }

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
        elif body.action == "accept":
            payload = {
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
        elif body.action == "exit":
            # Exit from an already-active promotion (status=started). Same DELETE
            # call as reject, but used for promotions ML auto-opt-in (e.g. SMART).
            promo_type = offer["promotion_type"] or ""
            r = await http.delete(
                f"{url}&promotion_type={promo_type}&promotion_id={offer['promotion_id']}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
        elif body.action == "reenable_dynamic_pricing":
            # User wants ML Pricing Automation back ON for this item.
            # Reads previous rule (rule_id, min, max) from
            # ml_user_pricing_history (saved when DELETE'ed). If no history
            # — falls back to INT_EXT with ±15% range from current price.
            async with pool.acquire() as conn:
                hist = await conn.fetchrow(
                    """
                    SELECT rule_id, min_price, max_price
                      FROM ml_user_pricing_history
                     WHERE user_id = $1 AND item_id = $2
                     ORDER BY disabled_at DESC LIMIT 1
                    """,
                    body.user_id, mlb,
                )
            rule_id = (hist["rule_id"] if hist else None) or "INT_EXT"
            min_price = float(hist["min_price"]) if hist and hist["min_price"] is not None else None
            max_price = float(hist["max_price"]) if hist and hist["max_price"] is not None else None

            if not min_price or not max_price or min_price <= 0 or max_price <= min_price:
                # Derive ±15% range from current item price.
                gr = await http.get(
                    f"https://api.mercadolibre.com/items/{mlb}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
                cur_price = float((gr.json() or {}).get("price") or 0.0) if gr.status_code < 400 else 0.0
                if cur_price <= 0:
                    return {"error": "no_current_price_for_range"}
                min_price = round(cur_price * 0.85, 2)
                max_price = round(cur_price * 1.15, 2)

            payload = {
                "rule_id": rule_id,
                "min_price": float(min_price),
                "max_price": float(max_price),
            }
            r = await http.post(
                f"https://api.mercadolibre.com/pricing-automation/items/{mlb}/automation",
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
            if r.status_code >= 400:
                return {
                    "error": "ml_dyn_reenable_rejected",
                    "status": r.status_code,
                    "detail": r.text[:300],
                    "attempted_payload": payload,
                }
            raise_info = {
                "dyn_reenabled": True,
                "rule_id": rule_id,
                "min_price": float(min_price),
                "max_price": float(max_price),
            }

        elif body.action in ("raise_only", "raise_with_disable_dyn"):
            # Standalone price raise.
            #   raise_only: just PUT /items {price}. If ML rejects with
            #     `item.price.not_modifiable` (item has Dynamic Pricing enabled),
            #     return needs_dyn_confirm so TG webhook can ask the user to
            #     opt-out first.
            #   raise_with_disable_dyn: explicit confirmation — DELETE the
            #     pricing-automation rule first, then PUT /items {price}.
            item_url = f"https://api.mercadolibre.com/items/{mlb}"
            try:
                gr = await http.get(
                    item_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
            except Exception as err:  # noqa: BLE001
                return {"error": "ml_item_fetch_failed", "detail": str(err)}
            if gr.status_code >= 400:
                return {
                    "error": "ml_item_fetch_rejected",
                    "status": gr.status_code,
                    "detail": gr.text[:300],
                }
            try:
                old_price = float((gr.json() or {}).get("price") or 0.0)
            except (TypeError, ValueError):
                old_price = 0.0
            if old_price <= 0:
                return {"error": "no_current_price"}
            new_price = round(old_price * (1.0 + raise_pct / 100.0), 2)

            dyn_disabled = False
            if body.action == "raise_with_disable_dyn":
                # Explicit user confirmation — first save current automation
                # rule to history (so user can re-enable later via TG button),
                # then DELETE pricing-automation rule.
                # Doc: DELETE /pricing-automation/items/{id}/automation.
                # Errors swallowed/logged: if 404 automation_not_found, ML
                # already cleared it elsewhere — proceed to PUT anyway.
                try:
                    # Read current rule before deleting (best-effort)
                    cur_auto = None
                    try:
                        gar = await http.get(
                            f"https://api.mercadolibre.com/pricing-automation/items/{mlb}/automation",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=15.0,
                        )
                        if gar.status_code == 200:
                            cur_auto = gar.json() or {}
                    except Exception:  # noqa: BLE001
                        cur_auto = None
                    if isinstance(cur_auto, dict) and cur_auto:
                        rule = (cur_auto.get("item_rule") or {}).get("rule_id") or "INT_EXT"
                        try:
                            min_p = float(cur_auto.get("min_price") or 0)
                        except (TypeError, ValueError):
                            min_p = 0.0
                        try:
                            max_p = float(cur_auto.get("max_price") or 0)
                        except (TypeError, ValueError):
                            max_p = 0.0
                        async with pool.acquire() as conn:
                            await conn.execute(
                                """
                                CREATE TABLE IF NOT EXISTS ml_user_pricing_history (
                                  user_id INTEGER NOT NULL,
                                  item_id TEXT NOT NULL,
                                  rule_id TEXT,
                                  min_price NUMERIC,
                                  max_price NUMERIC,
                                  raw JSONB,
                                  disabled_at TIMESTAMPTZ DEFAULT NOW()
                                );
                                CREATE INDEX IF NOT EXISTS idx_pricing_hist_user_item
                                  ON ml_user_pricing_history(user_id, item_id, disabled_at DESC);
                                """,
                            )
                            await conn.execute(
                                """
                                INSERT INTO ml_user_pricing_history
                                  (user_id, item_id, rule_id, min_price, max_price, raw)
                                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                                """,
                                body.user_id, mlb, rule, min_p, max_p,
                                json.dumps(cur_auto, default=str),
                            )

                    dr = await http.delete(
                        f"https://api.mercadolibre.com/pricing-automation/items/{mlb}/automation",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=15.0,
                    )
                    if dr.status_code in (200, 204, 404):
                        dyn_disabled = True
                    elif dr.status_code >= 400:
                        return {
                            "error": "ml_dyn_disable_rejected",
                            "status": dr.status_code,
                            "detail": dr.text[:300],
                        }
                except Exception as err:  # noqa: BLE001
                    return {"error": "ml_dyn_disable_failed", "detail": str(err)}

            try:
                pr = await http.put(
                    item_url,
                    json={"price": new_price},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    timeout=15.0,
                )
            except Exception as err:  # noqa: BLE001
                return {"error": "ml_price_update_failed", "detail": str(err)}
            if pr.status_code >= 400:
                # Detect Dynamic Pricing block — return needs_dyn_confirm so TG
                # webhook can render confirmation prompt instead of an error.
                detail_text = pr.text or ""
                if (
                    body.action == "raise_only"
                    and pr.status_code == 400
                    and "item.price.not_modifiable" in detail_text
                ):
                    return {
                        "ok": False,
                        "needs_dyn_confirm": True,
                        "reason": "dynamic_pricing_active",
                        "old_price": old_price,
                        "attempted_new_price": new_price,
                        "raise_pct": raise_pct,
                    }
                return {
                    "error": "ml_price_update_rejected",
                    "status": pr.status_code,
                    "detail": detail_text[:300],
                    "old_price": old_price,
                    "attempted_new_price": new_price,
                    "dyn_disabled": dyn_disabled,
                }
            raise_info = {
                "old_price": old_price,
                "new_price": new_price,
                "raise_pct": raise_pct,
                "raise_mode": (body.raise_mode or "exact").lower(),
                "entrada_pct": entrada_pct,
                "dyn_disabled": dyn_disabled,
            }
            # raise_only doesn't talk to /seller-promotions endpoint; we synthesize
            # a 200-shaped response so the downstream success-path handles it.
            class _OkResp:
                status_code = 200
                text = ""
            r = _OkResp()  # type: ignore
        else:
            return {"error": "bad_action"}

    if r.status_code >= 400:
        return {
            "error": "ml_request_failed",
            "status": r.status_code,
            "detail": r.text[:300],
            "raise": raise_info or None,
        }

    # Audit: log the seller's choice + AI/promo metadata.
    try:
        from v2.services import escalar_audit as _audit
        audit_action_map = {
            "accept": "promo_accepted",
            "accept_with_raise": "promo_accepted",
            "reject": "promo_rejected",
            "exit": "promo_rejected",
            "raise_only": "price_raised",
            "raise_with_disable_dyn": "dyn_pricing_off",
            "reenable_dynamic_pricing": "dyn_pricing_on",
        }
        await _audit.log_event(
            pool,
            user_id=body.user_id,
            action=audit_action_map.get(body.action or "", body.action or "promo"),
            target_type="promotion",
            target_id=f"{body.item_id}:{body.promotion_id}",
            user_action=body.action,
            metadata={
                "promotion_type": offer.get("promotion_type"),
                "deal_price": offer.get("deal_price"),
                "raise_pct": raise_info.get("raise_pct") if isinstance(raise_info, dict) else None,
            },
        )
    except Exception:  # noqa: BLE001
        pass

    if body.action in ("accept", "accept_with_raise"):
        await ml_user_promotions_svc.mark_accepted(
            pool, body.user_id, body.item_id, body.promotion_id,
        )
        # Open analytics window — track sales after accept vs before.
        try:
            from v2.services import ml_promo_analytics as _promo_a
            disc = None
            if offer.get("discount_percentage") is not None:
                try:
                    disc = float(offer["discount_percentage"])
                except (TypeError, ValueError):
                    disc = None
            await _promo_a.record_acceptance(
                pool, body.user_id, body.item_id, body.promotion_id,
                promotion_type=offer.get("promotion_type"),
                promotion_name=(offer.get("raw") or {}).get("name") if isinstance(offer.get("raw"), dict) else None,
                deal_price=float(offer["deal_price"]) if offer.get("deal_price") is not None else None,
                original_price=float(offer["original_price"]) if offer.get("original_price") is not None else None,
                discount_pct=disc,
            )
        except Exception:  # noqa: BLE001
            pass
    elif body.action in ("reject", "exit"):
        # exit на started SMART = effective dismiss. Помечаем строку чтобы
        # повторное TG-уведомление не приходило.
        await ml_user_promotions_svc.mark_dismissed(
            pool, body.user_id, body.item_id, body.promotion_id,
        )
    # raise_only — оставляем строку как есть (промо ещё активна, цена выросла).
    # Also schedule a cache refresh of just this item so UI reflects change.
    try:
        await ml_user_promotions_svc.refresh_user_promotions(
            pool, body.user_id, item_ids=[body.item_id],
        )
    except Exception:  # noqa: BLE001
        pass

    out: dict = {"ok": True, "action": body.action}
    if raise_info:
        out["raise"] = raise_info
        out["deal_price"] = offer.get("deal_price")
    return out


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
    response shape before designing the cache table.

    The ML endpoint expects seller_id + (optional) date range. Initial
    probe missed seller_id and got 400. This probe tries a small grid of
    parameter combinations and returns the status of each so we can spot
    which shape ML actually wants for THIS user without running through
    the docs every time.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    tokens = await ml_oauth_svc.load_user_tokens(pool, user.id) or {}
    seller_id = tokens.get("ml_user_id")

    base = "https://api.mercadolibre.com/stock/fulfillment/operations/search"
    # Date range default: last 60 days, ML wants ISO with TZ
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    now = _dt.now(_tz.utc)
    date_to = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    date_from = (now - _td(days=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    candidates: list[tuple[str, str]] = [
        ("v1_no_seller", f"{base}?operation_type={operation_type}&limit=20&offset=0"),
    ]
    if seller_id:
        candidates += [
            ("v2_seller_id", f"{base}?operation_type={operation_type}&seller_id={seller_id}&limit=20&offset=0"),
            ("v3_seller_id_dates", f"{base}?operation_type={operation_type}&seller_id={seller_id}&date_from={date_from}&date_to={date_to}&limit=20"),
            # Some ML endpoints prefer `seller=` param name
            ("v4_seller", f"{base}?operation_type={operation_type}&seller={seller_id}&limit=20"),
        ]

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as http:
        for label, url in candidates:
            try:
                r = await http.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=15.0,
                )
                ct = r.headers.get("content-type", "")
                try:
                    body = r.json() if "json" in ct else r.text[:1500]
                except Exception:  # noqa: BLE001
                    body = r.text[:1500]
                results.append({
                    "label": label,
                    "url": url,
                    "status": r.status_code,
                    "content_type": ct,
                    "body_preview": (json.dumps(body, default=str)[:500] if isinstance(body, (dict, list)) else str(body)[:500]),
                })
            except Exception as err:  # noqa: BLE001
                results.append({"label": label, "url": url, "error": str(err)})

    successful = next((r for r in results if r.get("status") == 200), None)
    return {
        "operation_type": operation_type,
        "seller_id": seller_id,
        "results": results,
        "winning_shape": successful["label"] if successful else None,
        "hint": (
            "Use the `winning_shape` parameter set when building the "
            "ml_full_operations cache. If all probes 400/403 → ML hasn't "
            "enabled the API for this account; document and skip."
        ),
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
    days: int = Query(14, ge=1, le=60),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Account-level anomalies (ACOS spike, sales drop, visits drop) for
    the last N days. Reads from `escalar_anomalies` cache populated by
    the cron / preview endpoint. Returns history so the UI can show
    trend, not just today.
    """
    if pool is None:
        return {"error": "no_db", "anomalies": []}
    await ml_anomalies_svc.ensure_schema(pool)
    rows = await ml_anomalies_svc.list_recent(pool, user.id, days=days)
    return {
        "anomalies": rows,
        "total": len(rows),
        "fetchedAt": rows[0]["created_at"] if rows else None,
    }


@router.post("/anomalies/preview")
async def anomalies_preview(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(default={}),
):
    """Run detect+upsert+dispatch for the current user immediately.
    Bypasses notify toggle when force=true. Returns counts so the UI
    knows whether anything was found / sent.
    """
    if pool is None:
        return {"error": "no_db"}
    target_str = (body or {}).get("date")
    target = None
    if target_str:
        try:
            from datetime import date as _date
            target = _date.fromisoformat(target_str)
        except ValueError:
            return {"error": "invalid_date"}
    return await ml_anomalies_svc._dispatch_for_user(pool, user.id, target_date=target)


@router.get("/anomalies/probe")
async def anomalies_probe(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Run detectors WITHOUT persisting or dispatching. For dev — see
    what would have fired today."""
    if pool is None:
        return {"error": "no_db"}
    return {"anomalies": await ml_anomalies_svc.detect_for_user(pool, user.id)}


# ──────────────────────────────────────────────────────────────────────
# Video concat — склейка нескольких Runway-клипов в одно длинное видео
# для FB Reels (Runway даёт максимум 10с за раз; для 15-30с нужно 2-3
# клипа склеить через ffmpeg на сервере).
# ──────────────────────────────────────────────────────────────────────

from pydantic import BaseModel as _CV_BM  # avoid clobbering top-level alias if any
from fastapi import Response as _CV_Response


class _ConcatVideosIn(_CV_BM):
    video_urls: list[str]


@router.post("/videos/concat")
async def concat_videos_endpoint(
    body: _ConcatVideosIn,
    user: CurrentUser = Depends(current_user),
):
    """Скачать N видео по URL, склеить через ffmpeg, отдать MP4.

    Body: {"video_urls": ["https://...mp4", "https://...mp4"]}
    Returns: video/mp4 binary stream (Content-Disposition attachment).

    Лимиты: max 5 URL, каждый <= 200MB. ffmpeg должен быть в системе
    (ставится в Dockerfile). При несовпадении формата — fallback на
    перекодирование через libx264 (медленнее но надёжнее).
    """
    from v2.services.video_concat import concat_videos, VideoConcatError

    if not body.video_urls:
        raise HTTPException(status_code=400, detail="video_urls required")
    try:
        mp4_bytes = await concat_videos(body.video_urls)
    except VideoConcatError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"concat_exception: {err}") from err

    filename = f"concat_user{user.id}_{len(body.video_urls)}clips.mp4"
    return _CV_Response(
        content=mp4_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Project ACL — team members and invitations ──────────────────────────────


def _app_base_url() -> str:
    import os as _os
    return _os.environ.get("APP_BASE_URL", "https://app.lsprofit.app").rstrip("/")


def _build_accept_url(token: str) -> str:
    return f"{_app_base_url()}/auth/invite/{token}"


@router.get("/team/members")
async def team_list_members(
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """List accepted members invited by current user. Optional ?project=NAME filter."""
    if pool is None:
        return {"members": [], "invitations": []}
    await project_members_svc.ensure_schema(pool)
    members = await project_members_svc.list_members(pool, user.id, project_name=project)
    invitations = await project_members_svc.list_pending_invitations(pool, user.id)
    if project:
        invitations = [inv for inv in invitations if inv["project_name"] == project]
    # Strip token from invitations list (only revealed once at creation time
    # via /team/invite or via invitation copy-link in UI).
    invitations_safe = [
        {k: v for k, v in inv.items() if k != "token"} | {
            "accept_url": _build_accept_url(inv["token"]),
        }
        for inv in invitations
    ]
    return {
        "members": members,
        "invitations": invitations_safe,
        "email_configured": email_brevo_svc.is_configured(),
    }


@router.get("/team/my-memberships")
async def team_my_memberships(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Projects the current user has been added to (membership-as-collaborator)."""
    if pool is None:
        return {"memberships": []}
    await project_members_svc.ensure_schema(pool)
    rows = await project_members_svc.list_my_memberships(pool, user.id)
    return {"memberships": rows, "total": len(rows)}


@router.post("/team/invite")
async def team_invite(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Create or reuse an invitation. Body:
      email         — required
      project_name  — required
      role          — viewer | analyst | admin (default viewer)
      send_email    — bool, if true and Brevo configured, send the invite email
    """
    if pool is None:
        return {"error": "no_db"}
    await project_members_svc.ensure_schema(pool)

    email = (body or {}).get("email")
    project_name = (body or {}).get("project_name")
    role = (body or {}).get("role", "viewer")
    send_email = bool((body or {}).get("send_email", True))

    result = await project_members_svc.create_invitation(
        pool, user.id,
        email=email or "",
        project_name=project_name or "",
        role=role,
    )
    if result.get("error"):
        return result

    accept_url = _build_accept_url(result["token"])
    result["accept_url"] = accept_url

    email_status: dict[str, Any] = {"sent": False, "configured": email_brevo_svc.is_configured()}
    if send_email and email_brevo_svc.is_configured():
        try:
            send_res = await email_brevo_svc.send_invitation_email(
                to_email=result["email"],
                project_name=result["project_name"],
                role=result["role"],
                inviter_name=user.name or user.email,
                inviter_email=user.email,
                accept_url=accept_url,
            )
            email_status["sent"] = bool(send_res.get("ok"))
            email_status["raw"] = send_res
        except Exception as err:  # noqa: BLE001
            email_status["sent"] = False
            email_status["error"] = str(err)

    result["email_status"] = email_status
    # Hide raw token from the response — UI uses accept_url for copy-link
    result.pop("token", None)
    return result


@router.post("/team/invitations/{invitation_id}/revoke")
async def team_revoke_invitation(
    invitation_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    return await project_members_svc.revoke_invitation(pool, user.id, invitation_id)


@router.patch("/team/members/{member_id}")
async def team_update_member_role(
    member_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    if pool is None:
        return {"error": "no_db"}
    new_role = (body or {}).get("role")
    if not new_role:
        return {"error": "role_required"}
    return await project_members_svc.update_role(pool, user.id, member_id, new_role)


@router.delete("/team/members/{member_id}")
async def team_remove_member(
    member_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    return await project_members_svc.remove_member(pool, user.id, member_id)


@router.get("/team/invitations/lookup")
async def team_invitation_lookup(
    token: str = Query(..., min_length=10),
    pool=Depends(get_pool),
):
    """Public-ish: returns the invitation details (no auth required) so the
    accept page can display project + inviter before the user logs in.
    Token itself is the credential — caller must already possess it."""
    if pool is None:
        return {"error": "no_db"}
    await project_members_svc.ensure_schema(pool)
    inv = await project_members_svc.get_invitation_by_token(pool, token)
    if not inv:
        return {"error": "invitation_not_found"}
    # Strip token from response (caller already has it)
    return {
        "email": inv["email"],
        "project_name": inv["project_name"],
        "role": inv["role"],
        "inviter_email": inv["inviter_email"],
        "inviter_name": inv["inviter_name"],
        "expires_at": inv["expires_at"],
        "is_active": inv["is_active"],
        "used_at": inv["used_at"],
        "revoked_at": inv["revoked_at"],
    }


@router.post("/team/invitations/accept")
async def team_invitation_accept(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Accept an invitation as the logged-in user. Validates email match."""
    if pool is None:
        return {"error": "no_db"}
    token = (body or {}).get("token")
    if not token:
        return {"error": "token_required"}
    return await project_members_svc.accept_invitation(
        pool, token=token,
        accepting_user_id=user.id,
        accepting_email=user.email,
    )


# ── Diagnostic probes — production bug investigation ─────────────────────────
# Goals:
#   /items/{mlb}/price-shift-probe — почему PUT /items {price} даёт Validation
#       error от кнопки "Поднять +5%". Возвращает full RAW response (без
#       обрезки text[:300]) + active promotions state на товаре.
#   /claims/{claim_id}/probe — какие поля ML использовать чтобы скрыть кнопку
#       "Atender no app" когда ML уже резолвнул жалобу (Type B).
#   /items/ai-question-probe — что именно отправляется Claude при генерации
#       AI suggestion: фото urls, system prompt, attributes context.
#
# Все три используют cookie-auth (current_user) → можно дёргать прямо из
# браузера через /api/v2-proxy/escalar/...

@router.get("/items/{mlb}/price-shift-probe")
async def price_shift_probe(
    mlb: str,
    pct: float = Query(5.0, description="Raise pct"),
    dry: int = Query(1, description="0 = actually execute PUT (be careful)"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Диагностика «Validation error» при PUT /items {price} (raise +5%).

    dry=1 (default): GET item state + active promotions, симуляция payload.
    dry=0: ещё и реальный PUT — full ML response без обрезки.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, _exp, _ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    mlb_up = mlb.upper()
    out: dict[str, Any] = {"item_id": mlb_up, "raise_pct": pct, "dry_run": bool(dry)}

    async with httpx.AsyncClient() as http:
        gr = await http.get(
            f"https://api.mercadolibre.com/items/{mlb_up}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        try:
            item_body = gr.json()
        except Exception:  # noqa: BLE001
            item_body = gr.text
        out["item"] = {"status": gr.status_code, "body": item_body}

        if gr.status_code >= 400 or not isinstance(item_body, dict):
            return out

        old_price = float(item_body.get("price") or 0.0)
        new_price = round(old_price * (1.0 + pct / 100.0), 2)
        out["old_price"] = old_price
        out["new_price"] = new_price
        out["item_status"] = item_body.get("status")
        out["item_sub_status"] = item_body.get("sub_status")
        out["catalog_listing"] = item_body.get("catalog_listing")
        out["channels"] = item_body.get("channels")
        out["price_lock_hint"] = {
            "deal_ids": item_body.get("deal_ids"),
            "tags": item_body.get("tags"),
        }

        # Active promotions on item — candidate offers can lock price.
        pr_url = (
            f"https://api.mercadolibre.com/seller-promotions/items/{mlb_up}"
            f"?app_version=v2"
        )
        pgr = await http.get(
            pr_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        try:
            promos = pgr.json()
        except Exception:  # noqa: BLE001
            promos = pgr.text
        out["promotions_on_item"] = {"status": pgr.status_code, "body": promos}

        if not dry:
            put_url = f"https://api.mercadolibre.com/items/{mlb_up}"
            put_body = {"price": new_price}
            ppr = await http.put(
                put_url,
                json=put_body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
            try:
                put_resp = ppr.json()
            except Exception:  # noqa: BLE001
                put_resp = ppr.text
            out["put_attempt"] = {
                "url": put_url,
                "body_sent": put_body,
                "status": ppr.status_code,
                "response": put_resp,
                "headers": {k: v for k, v in ppr.headers.items() if k.lower() in (
                    "content-type", "x-request-id", "x-error-id",
                )},
            }

    return out


@router.get("/claims/{claim_id}/probe")
async def claim_type_probe(
    claim_id: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Диагностика «Type A (atender) vs Type B (ML resolved)» для PDD claim.

    Returns claim state, seller's available_actions, last message sender —
    these fields drive button rendering in TG dispatch.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, _exp, _ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "ml_oauth_required", "detail": str(err)}

    out: dict[str, Any] = {"claim_id": claim_id}
    base = "https://api.mercadolibre.com"

    async with httpx.AsyncClient() as http:
        cr = await http.get(
            f"{base}/post-purchase/v1/claims/{claim_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        try:
            claim = cr.json()
        except Exception:  # noqa: BLE001
            claim = cr.text
        out["claim"] = {"status": cr.status_code, "body": claim}

        seller_actions: list = []
        last_message_role: Optional[str] = None
        last_message_text: Optional[str] = None

        if isinstance(claim, dict):
            players = claim.get("players") or []
            seller_player: dict = {}
            for p in players:
                role = (p.get("role") or "").lower()
                ptype = (p.get("type") or "").lower()
                if role == "respondent" or ptype == "seller":
                    seller_player = p
                    break
            seller_actions = list(seller_player.get("available_actions") or [])
            out["seller_player"] = seller_player
            out["seller_available_actions"] = seller_actions
            out["claim_status"] = claim.get("status")
            out["claim_stage"] = claim.get("stage")
            out["claim_type"] = claim.get("type")
            out["resolution"] = claim.get("resolution")
            out["fulfilled"] = claim.get("fulfilled")

        # Messages
        mr = await http.get(
            f"{base}/post-purchase/v1/claims/{claim_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        try:
            msgs = mr.json()
        except Exception:  # noqa: BLE001
            msgs = mr.text
        out["messages"] = {"status": mr.status_code, "body": msgs}

        if isinstance(msgs, dict):
            arr = msgs.get("messages") or msgs.get("data") or []
        elif isinstance(msgs, list):
            arr = msgs
        else:
            arr = []
        if arr:
            last = arr[-1] if isinstance(arr[-1], dict) else {}
            from_obj = last.get("from") or {}
            last_message_role = (
                from_obj.get("role") if isinstance(from_obj, dict) else None
            )
            last_message_text = str(
                last.get("message") or last.get("text") or "",
            )[:500]
        out["last_message_role"] = last_message_role
        out["last_message_preview"] = last_message_text

    actionable_actions = {
        "refund", "return_product", "change_product", "partial_refund",
        "return_review_ok", "review_evidence", "open_dispute",
    }
    has_real_action = any(a in actionable_actions for a in seller_actions)
    is_ml_resolved_text = bool(last_message_text and any(
        marker in last_message_text.lower()
        for marker in ("encerrado", "resolvi tudo", "reputação não foi")
    ))
    out["classification"] = {
        "type": "A_actionable" if has_real_action else "B_ml_resolved",
        "has_real_action": has_real_action,
        "ml_mediator_closed_text": is_ml_resolved_text,
        "rule": "Type A if seller_available_actions ∩ {refund,return_product,...} non-empty",
    }
    return out


class _AiQuestionProbeIn(__import__("pydantic").BaseModel):
    item_id: str
    question_text: str = "Qual a cor do produto?"
    invoke: bool = False  # call OpenRouter (costs cents per call)


@router.post("/breakeven/recompute")
async def breakeven_recompute(
    project: str = Query(..., description="Project id e.g. ARTHUR"),
    month: Optional[str] = Query(None, description="YYYY-MM BRT; default current"),
    backfill: bool = Query(True, description="First populate log from ml_user_orders"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Recovery — пересчитывает project_breakeven_state.cumulative.

    flow:
      1. (optional) backfill_log_from_orders — populate breakeven_sale_log
         из ml_user_orders за месяц с margin recompute (per-order).
      2. recompute_state_from_log — sum log → state.cumulative.

    Используется когда cumulative завышено из-за исторических double-
    increments до idempotency fix. Безопасный idempotent reset.
    """
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_breakeven as breakeven_svc
    await breakeven_svc.ensure_schema(pool)
    if not month:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        month = _dt.now(_tz(_td(hours=-3))).strftime("%Y-%m")

    out: dict[str, Any] = {"project": project.upper(), "month": month}
    if backfill:
        out["backfill"] = await breakeven_svc.backfill_log_from_orders(
            pool, user.id, project.upper(), month,
        )
    out["recompute"] = await breakeven_svc.recompute_state_from_log(
        pool, user.id, project.upper(), month,
    )
    return out


@router.post("/retirada-alerts/dispatch-now")
async def retirada_alerts_dispatch_now(
    days_back: int = Query(7, ge=1, le=30),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger: scan retirada records за last N дней и алертит
    о Descarte / Envio para o endereço. Same as daily 13:30 UTC cron.
    """
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_retirada_alerts as retirada_svc
    await retirada_svc.ensure_schema(pool)
    return await retirada_svc.dispatch_retirada_alerts(pool, user.id, days_back)


@router.post("/inventory-alerts/dispatch-now")
async def inventory_alerts_dispatch_now(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger: scan current user's active items and send TG alerts
    for any that hit `critical` (and weren't alerted in last 7 days).

    Used to test the daily cron without waiting for 12 UTC. Returns counts
    of checked / sent / skipped items.
    """
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_inventory_forecast as inv_svc
    await inv_svc.ensure_schema(pool)
    return await inv_svc.dispatch_inventory_alerts(pool, user.id)


@router.get("/items/{mlb}/inventory-probe")
async def inventory_probe(
    mlb: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Diagnostic — compare cached vs live stock data + variations breakdown.

    Use when seller doubts the «📦 Estoque: 0 un.» blocked in TG. Shows:
      - ml_user_items.available_quantity (cached, used by inventory_forecast)
      - Live ML /items/{id} available_quantity / initial_quantity
      - Variations available_quantity sum (if item has variations)
      - ml_stock_full row if exists (Full warehouse delegated stock)
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        token, _exp, _ = await ml_oauth_svc.get_valid_access_token(pool, user.id)
    except Exception as err:  # noqa: BLE001
        return {"error": "oauth_failed", "detail": str(err)}

    item_id = mlb.upper()
    out: dict[str, Any] = {"item_id": item_id}

    # Cached available_quantity (что использует inventory_forecast)
    async with pool.acquire() as conn:
        cached_qty = await conn.fetchval(
            "SELECT available_quantity FROM ml_user_items WHERE user_id = $1 AND item_id = $2",
            user.id, item_id,
        )
    out["cached_available_quantity"] = cached_qty

    # Live ML
    async with httpx.AsyncClient() as http:
        gr = await http.get(
            f"https://api.mercadolibre.com/items/{item_id}"
            "?attributes=id,available_quantity,initial_quantity,sold_quantity,"
            "status,sub_status,variations.available_quantity,"
            "variations.id,shipping.logistic_type,inventory_id",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
    if gr.status_code >= 400:
        out["live_error"] = {"status": gr.status_code, "detail": gr.text[:300]}
        return out
    live = gr.json() or {}
    out["live"] = {
        "available_quantity": live.get("available_quantity"),
        "initial_quantity": live.get("initial_quantity"),
        "sold_quantity": live.get("sold_quantity"),
        "status": live.get("status"),
        "sub_status": live.get("sub_status"),
        "logistic_type": (live.get("shipping") or {}).get("logistic_type"),
        "inventory_id": live.get("inventory_id"),
    }

    # Variations breakdown
    variations = live.get("variations") or []
    if variations:
        out["variations"] = [
            {"id": v.get("id"), "available_quantity": v.get("available_quantity")}
            for v in variations
        ]
        out["variations_sum_available"] = sum(
            int(v.get("available_quantity") or 0) for v in variations
        )

    # Mismatch detection
    cached_v = int(cached_qty or 0)
    live_v = int(live.get("available_quantity") or 0)
    out["mismatch"] = {
        "cached_vs_live": cached_v != live_v,
        "delta": live_v - cached_v,
        "hint": (
            "ml_user_items stale — run /escalar/items/refresh"
            if cached_v != live_v else "consistent"
        ),
    }
    return out


@router.post("/items/{mlb}/photo-descriptions/generate")
async def photo_descriptions_generate(
    mlb: str,
    force: bool = Query(False, description="Re-generate even if cached"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Generate AI descriptions for item's photos (cached in DB).

    Pre-fills ml_item_photo_descriptions so subsequent ai-suggest calls can
    inject description text alongside the vision blocks. Idempotent unless
    force=True.
    """
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_photo_descriptions as photo_svc

    await photo_svc.ensure_schema(pool)
    item_id = mlb.upper()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT pictures FROM ml_item_context
             WHERE user_id = $1 AND item_id = $2
            """,
            user.id, item_id,
        )
    if not row:
        return {
            "error": "no_item_context",
            "hint": f"refresh ml_item_context for {item_id} first",
        }
    pics = row["pictures"]
    if isinstance(pics, str):
        try:
            pics = json.loads(pics)
        except Exception:  # noqa: BLE001
            pics = []
    if not isinstance(pics, list):
        pics = []
    return await photo_svc.generate_descriptions_for_item(
        pool, user.id, item_id, pics, force=force,
    )


@router.post("/items/photo-descriptions/auto-generate")
async def photo_descriptions_auto_generate(
    max_items: int = Query(15, ge=1, le=50),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger: pick top-N most-sold active items without cached
    descriptions, generate batches. Same logic as daily 05:00 UTC cron.

    Use to bootstrap a fresh seller account or top up after adding new
    products without waiting for tomorrow's auto-run.
    """
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_photo_descriptions as photo_svc
    return await photo_svc.auto_generate_for_user(pool, user.id, max_items)


@router.get("/items/{mlb}/photo-descriptions")
async def photo_descriptions_get(
    mlb: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Returns cached descriptions for item (empty list if none generated)."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_photo_descriptions as photo_svc
    descs = await photo_svc.get_descriptions_for_item(pool, user.id, mlb.upper())
    return {"item_id": mlb.upper(), "count": len(descs), "descriptions": descs}


@router.post("/reconciliation/dispatch-now")
async def reconciliation_dispatch_now(
    threshold: float = Query(50.0, ge=0, le=10000),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger weekly reconciliation. Compares fatura ML vs bank_tx
    и Full Express CSV vs bank_tx за last month + MTD. Если delta > R$X
    → admin TG alert. Per-month dedup в reconciliation_alert_log.
    """
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_reconciliation_alerts as recon_svc
    return await recon_svc.reconcile_for_user(pool, user.id, threshold_brl=threshold)


@router.post("/ads-summary/dispatch-now")
async def ads_summary_dispatch_now(
    days: int = Query(14, ge=7, le=90),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger ads recap. Sends per-campaign cards в TG (top 10 by
    revenue last N days). Same logic as Monday 11:00 UTC cron."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_ads_summary as ads_svc
    return await ads_svc.dispatch_for_user(pool, user.id, days=days)


@router.get("/ads-summary/me")
async def ads_summary_me(
    days: int = Query(14, ge=7, le=90),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Per-campaign aggregated metrics для UI dashboard или JSON-debug."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_ads_summary as ads_svc
    campaigns = await ads_svc.aggregate_per_campaign(pool, user.id, days=days)
    return {"days": days, "count": len(campaigns), "campaigns": campaigns}


@router.get("/audit/me")
async def audit_me(
    days: int = Query(7, ge=1, le=90),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Returns escalar_audit summary за last N days. Action counts +
    Q&A approve/edit/regen funnel + 30 most recent events."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import escalar_audit as audit_svc
    return await audit_svc.get_summary(pool, user.id, days=days)


@router.get("/promo-analytics/me")
async def promo_analytics_me(
    days: int = Query(90, ge=7, le=365),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Promotion outcomes за last N days. Returns top-3 winners + losers,
    by_type breakdown, totals. Help seller decide which promo to accept
    next time based on actual sales delta."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_promo_analytics as analytics
    return await analytics.get_user_analytics(pool, user.id, days=days)


@router.post("/promo-analytics/finalize-now")
async def promo_analytics_finalize_now(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual trigger — finalize 14d windows для рядов прошедших cutoff.
    Use this after testing to populate the analytics dashboard immediately."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ml_promo_analytics as analytics
    return await analytics.finalize_due_outcomes(pool)


@router.get("/ai-usage/me")
async def ai_usage_me(
    days: int = Query(7, ge=1, le=90),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Per-user OpenRouter usage за last N days. Calls / tokens / cost
    breakdown по services (questions/claims/news/narrative/photo) и
    moderation hours."""
    if pool is None:
        return {"error": "no_db"}
    from v2.services import ai_usage_tracker as tracker
    return await tracker.get_usage_summary(pool, user.id, days=days)


@router.get("/ai-usage/admin-all")
async def ai_usage_admin_all(
    days: int = Query(7, ge=1, le=90),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Super-admin only — across all users. Auth via LS_ADMIN_TG_CHAT_IDS
    env (same as admin alerts)."""
    if pool is None:
        return {"error": "no_db"}
    import os as _os
    admin_ids = _os.environ.get("LS_ADMIN_TG_CHAT_IDS", "").split(",")
    admin_ids = {c.strip() for c in admin_ids if c.strip()}
    async with pool.acquire() as conn:
        chat_id = await conn.fetchval(
            "SELECT telegram_chat_id FROM notification_settings WHERE user_id = $1",
            user.id,
        )
    if str(chat_id or "") not in admin_ids:
        return {"error": "not_super_admin"}
    from v2.services import ai_usage_tracker as tracker
    return await tracker.get_usage_summary(pool, user_id=None, days=days)


@router.get("/notifications/me")
async def notifications_me_get(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Returns current user's notification settings + admin status.

    Used by Next.js Settings page.
    """
    if pool is None:
        return {"error": "no_db"}
    import os as _os
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT telegram_chat_id,
                   COALESCE(notify_daily_sales, TRUE)  AS notify_daily_sales,
                   COALESCE(notify_acos_change, TRUE)  AS notify_acos_change,
                   COALESCE(notify_ml_news, TRUE)      AS notify_ml_news,
                   COALESCE(acos_threshold, 5)         AS acos_threshold,
                   COALESCE(language, 'pt')            AS language,
                   COALESCE(inventory_window_days, 14) AS inventory_window_days
              FROM notification_settings
             WHERE user_id = $1
            """,
            user.id,
        )
    out: dict[str, Any] = {
        "user_id": user.id,
        "telegram_chat_id": None,
        "telegram_linked": False,
        "notify_daily_sales": True,
        "notify_acos_change": True,
        "notify_ml_news": True,
        "acos_threshold": 5,
        "language": "pt",
        "inventory_window_days": 14,
        "is_super_admin": False,
    }
    if row:
        chat_id = row["telegram_chat_id"]
        out.update({
            "telegram_chat_id": chat_id,
            "telegram_linked": bool(chat_id),
            "notify_daily_sales": bool(row["notify_daily_sales"]),
            "notify_acos_change": bool(row["notify_acos_change"]),
            "notify_ml_news": bool(row["notify_ml_news"]),
            "acos_threshold": float(row["acos_threshold"] or 5),
            "language": row["language"] or "pt",
            "inventory_window_days": int(row["inventory_window_days"] or 14),
        })
    admin_ids = _os.environ.get("LS_ADMIN_TG_CHAT_IDS", "").split(",")
    admin_ids = [c.strip() for c in admin_ids if c.strip()]
    if out["telegram_chat_id"] and str(out["telegram_chat_id"]) in admin_ids:
        out["is_super_admin"] = True
    return out


class _NotificationsUpdateIn(__import__("pydantic").BaseModel):
    notify_daily_sales: Optional[bool] = None
    notify_acos_change: Optional[bool] = None
    notify_ml_news: Optional[bool] = None
    acos_threshold: Optional[float] = None
    language: Optional[str] = None
    inventory_window_days: Optional[int] = None


@router.put("/notifications/me")
async def notifications_me_update(
    body: _NotificationsUpdateIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Updates per-user notification preferences. Builds dynamic UPDATE
    only for fields explicitly provided (None = leave as is)."""
    if pool is None:
        return {"error": "no_db"}
    # Validate inventory_window_days
    if body.inventory_window_days is not None and body.inventory_window_days not in (7, 14, 30):
        return {"error": "invalid_inventory_window", "allowed": [7, 14, 30]}
    if body.language is not None and body.language not in ("pt", "ru", "en", "es"):
        return {"error": "invalid_language", "allowed": ["pt", "ru", "en", "es"]}
    # Ensure schema (notification_settings + inventory_window_days column)
    from v2.services import ml_inventory_forecast as inv_svc
    await inv_svc.ensure_schema(pool)

    fields: list[tuple[str, Any]] = []
    if body.notify_daily_sales is not None:
        fields.append(("notify_daily_sales", bool(body.notify_daily_sales)))
    if body.notify_acos_change is not None:
        fields.append(("notify_acos_change", bool(body.notify_acos_change)))
    if body.notify_ml_news is not None:
        fields.append(("notify_ml_news", bool(body.notify_ml_news)))
    if body.acos_threshold is not None:
        fields.append(("acos_threshold", max(0.0, min(100.0, float(body.acos_threshold)))))
    if body.language is not None:
        fields.append(("language", body.language))
    if body.inventory_window_days is not None:
        fields.append(("inventory_window_days", int(body.inventory_window_days)))
    if not fields:
        return {"ok": True, "updated_fields": 0}

    # Upsert: row may not exist yet for new users.
    set_clause = ", ".join(f"{name} = ${i + 2}" for i, (name, _) in enumerate(fields))
    values = [v for _, v in fields]
    async with pool.acquire() as conn:
        # Insert dummy row first if missing (unique on user_id)
        await conn.execute(
            "INSERT INTO notification_settings (user_id) VALUES ($1) "
            "ON CONFLICT (user_id) DO NOTHING",
            user.id,
        )
        await conn.execute(
            f"UPDATE notification_settings SET {set_clause}, updated_at = NOW() WHERE user_id = $1",
            user.id, *values,
        )
    return {"ok": True, "updated_fields": len(fields)}


@router.post("/admin-alerts/test")
async def admin_alerts_test(
    user: CurrentUser = Depends(current_user),
):
    """Test endpoint — fires a sample admin alert. Use to verify
    LS_ADMIN_TG_CHAT_IDS env var is configured correctly. Returns the
    parsed chat_ids so caller can see what got loaded.
    """
    import os as _os
    from v2.services import tg_admin_alerts as alerts_svc

    raw = _os.environ.get("LS_ADMIN_TG_CHAT_IDS", "")
    chat_ids = [c.strip() for c in raw.split(",") if c.strip()]
    if not chat_ids:
        return {
            "ok": False,
            "error": "no_admin_chat_ids",
            "hint": "Set LS_ADMIN_TG_CHAT_IDS=94675114 in Railway env",
        }
    await alerts_svc.send_admin_alert(
        title="Test alert",
        detail=f"Triggered by user {user.id} ({user.email}). All admin alerts working.",
        severity="warn",
        service="admin-alerts/test",
        deduplicate_key=f"test:{user.id}:{int(__import__('time').time() // 60)}",
    )
    return {"ok": True, "chat_ids": chat_ids, "sent_alert": True}


@router.get("/claims/{claim_id}/render-preview")
async def claim_render_preview(
    claim_id: str,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Diagnostic — runs the same _build_claim_card + _build_keyboard +
    _is_ml_resolved logic that ml_claims_dispatch uses, but returns the
    rendered output as JSON instead of pushing to Telegram.

    Lets us verify Type A vs Type B classification on a real cached claim
    without waiting for the next dispatch tick.
    """
    if pool is None:
        return {"error": "no_db"}
    try:
        cid = int(claim_id)
    except (TypeError, ValueError):
        return {"error": "bad_claim_id"}

    from v2.services.ml_claims_dispatch import (
        _build_claim_card, _build_keyboard, _is_ml_resolved,
        _seller_available_actions,
    )
    import os as _os

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT enriched, status FROM ml_user_claims
                 WHERE user_id = $1 AND claim_id = $2
                """,
                user.id, cid,
            )
    except Exception as err:  # noqa: BLE001
        return {"error": "db_query_failed", "detail": str(err)}
    if not row:
        return {"error": "claim_not_in_cache", "claim_id": cid}

    enriched_raw = row["enriched"]
    if isinstance(enriched_raw, str):
        try:
            enriched = json.loads(enriched_raw)
        except Exception:  # noqa: BLE001
            enriched = {}
    else:
        enriched = enriched_raw or {}

    is_resolved = _is_ml_resolved(enriched)
    actions = sorted(_seller_available_actions(enriched))
    app_base_url = _os.environ.get("APP_BASE_URL", "https://app.lsprofit.app")

    try:
        text = _build_claim_card(enriched, summary=None, summary_lang="ru")
    except Exception as err:  # noqa: BLE001
        text = f"<card render failed: {err}>"
    try:
        keyboard = _build_keyboard(cid, app_base_url, claim=enriched)
    except Exception as err:  # noqa: BLE001
        keyboard = {"error": str(err)}

    return {
        "claim_id": cid,
        "type_classification": "B_ml_resolved" if is_resolved else "A_actionable",
        "is_ml_resolved": is_resolved,
        "seller_available_actions": actions,
        "claim_status_in_cache": row["status"],
        "card_text_preview": text[:1500] if isinstance(text, str) else str(text),
        "keyboard": keyboard,
        "would_skip_ai_summary": is_resolved,
        "enriched_keys_present": sorted(enriched.keys())[:30] if isinstance(enriched, dict) else None,
        "enriched_messages_count": len(enriched.get("messages") or []) if isinstance(enriched, dict) else 0,
    }


@router.post("/items/ai-question-probe")
async def ai_question_probe(
    body: _AiQuestionProbeIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Диагностика multimodal AI suggestion: что отправляется Claude.

    Возвращает: какие фото переданы (URLs), длина description, кол-во
    атрибутов, преамбула system prompt. invoke=True → ещё и реально зовёт
    AI и возвращает ответ.
    """
    if pool is None:
        return {"error": "no_db"}

    item_id = body.item_id.upper()

    ctx_row = None
    async with pool.acquire() as conn:
        ctx_row = await conn.fetchrow(
            """
            SELECT title, description, attributes, pictures, fetched_at
              FROM ml_item_context
             WHERE user_id = $1 AND item_id = $2
            """,
            user.id, item_id,
        )
    if not ctx_row:
        return {
            "error": "no_item_context",
            "hint": (
                f"item {item_id} не в ml_item_context. Trigger refresh "
                "via /escalar/items/refresh-context."
            ),
        }

    def _maybe_load(v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:  # noqa: BLE001
                return v
        return v

    pictures = _maybe_load(ctx_row["pictures"]) or []
    attributes = _maybe_load(ctx_row["attributes"]) or []
    description = ctx_row["description"] or ""

    picture_urls: list[str] = []
    for p in pictures[:6]:
        if isinstance(p, dict):
            url = p.get("secure_url") or p.get("url")
            if url:
                picture_urls.append(url)

    out: dict[str, Any] = {
        "item_id": item_id,
        "title": ctx_row["title"],
        "fetched_at": ctx_row["fetched_at"].isoformat() if ctx_row["fetched_at"] else None,
        "pictures_total": len(pictures),
        "pictures_to_send_to_llm": picture_urls[:3],
        "attributes_count": len(attributes),
        "description_length": len(description),
        "description_preview": description[:600],
        "question_text": body.question_text,
        "would_invoke_openrouter": body.invoke,
        "current_prompt_hint": (
            "ai-suggest/route.ts:134 says: 'Se a resposta NÃO ESTÁ literalmente "
            "no CONTEXTO ou nas FOTOS — responda Vou verificar...'. Photos go "
            "as vision blocks with detail:high. Bug: prompt не инструктирует "
            "ИЗУЧИТЬ фото для подсчёта/измерений → Claude отвечает Vou verificar."
        ),
    }
    if body.invoke:
        out["invocation_result"] = (
            "not_implemented — call /api/escalar/questions/ai-suggest directly "
            "from Next.js with the same item_id and observe response.text."
        )
    return out
