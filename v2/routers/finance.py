"""Finance v2 endpoints — wraps legacy compute_pnl/cashflow/balance from `_admin/`.

Per-request workflow:
  1. `current_user(request)` parses cookie → returns CurrentUser
  2. We mirror that user_id into `legacy.db_storage._current_user_id_var` so the
     copied Streamlit code (config.py, sku_catalog.py) can read user-scoped data
     from PostgreSQL via the same `user_data` table the Streamlit app uses.
"""
from __future__ import annotations

import concurrent.futures
import contextvars
from datetime import date, datetime
from typing import Any, Callable, Optional, TypeVar

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Response, UploadFile

from v2.db import get_pool
from v2.deps import CurrentUser, current_user, _is_superadmin
from v2.legacy import db_storage as legacy_db
from v2.legacy import config as legacy_config
from v2.services import finance_cache
from v2.services import bank_balances as bank_balances_svc
from v2.schemas.finance import (
    ProjectsListOut, ReportsBundleOut,
    SkuMappingOut, SkuBulkSaveIn, SkuBulkSaveOut,
    PnlMatrixOut,
    OrphanPacotesResponse, OrphanSaveIn, OrphanSaveOut,
    RetiradaOverridesIn,
    RetiradaPreviewOut,
    UploadsListOut, UploadSaveOut, SourceCatalogOut,
    RulesOut, RulesSaveIn, TransactionsOut,
    ClassificationSaveIn, ClassificationSaveOut,
    OnboardingState, ProjectCreateIn, ProjectCreateOut,
    ProjectUpdateIn, ProjectMutOut,
    FixedCostsSaveIn,
    UploadPreviewOut,
    ManualCashflowEntryIn, ManualCashflowEntriesOut,
    PlannedPaymentIn, PlannedPaymentsOut, PlannedPaymentMutOut,
    MonthlyPlanOut, RecurringSuggestionsOut,
    MarkPaidIn,
    PublicidadeInvoiceIn, PublicidadeInvoicesListOut,
    PublicidadeReconciliationOut,
    CoverageOut,
    RentalPaymentIn, RentalPaymentsListOut,
    LoanIn, LoanOut, LoansListOut, LoanMutOut,
    DividendIn, DividendOut, DividendsListOut, DividendMutOut,
    APListOut,
    ManualInflowIn, ManualInflowOut, ManualInflowsListOut,
    ServicesPnlOut, ServicesCashflowOut, ServicesBalanceOut, ServicesReportsBundleOut,
)
from v2.storage import uploads_storage
from v2.storage import manual_usd_inflows as manual_inflows_svc

router = APIRouter(prefix="/finance", tags=["finance"])

_COMPUTE_TIMEOUT_SECONDS = 90  # legacy compute can scan many files; cap response time

T = TypeVar("T")


def _run_parallel_with_timeout(
    tasks: dict[str, Callable[[], Any]],
    timeout: int = _COMPUTE_TIMEOUT_SECONDS,
) -> dict[str, tuple[Any | None, str | None]]:
    """Run several sync legacy computes in parallel with a single shared timeout.

    Total wall time is capped at `timeout` (not N × timeout). Orphaned threads
    keep running but we stop waiting on them — Python can't kill threads safely.
    """
    out: dict[str, tuple[Any | None, str | None]] = {label: (None, None) for label in tasks}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tasks)))
    # Each task gets its OWN copy of the parent context (otherwise ctx.run on
    # the same ctx in parallel raises "cannot enter context: already entered").
    future_to_label = {
        pool.submit(contextvars.copy_context().run, fn): label for label, fn in tasks.items()
    }

    try:
        for future in concurrent.futures.as_completed(future_to_label, timeout=timeout):
            label = future_to_label[future]
            try:
                out[label] = (future.result(), None)
            except Exception as e:
                out[label] = (None, f"{type(e).__name__}: {e}")
    except concurrent.futures.TimeoutError:
        pass

    # Mark unfinished tasks with a timeout error
    for future, label in future_to_label.items():
        if not future.done() and out[label][1] is None:
            out[label] = (None, f"{label}_timeout_{timeout}s")
    pool.shutdown(wait=False, cancel_futures=True)
    return out


def _bind_user(user: CurrentUser) -> None:
    """Make legacy code see the current user_id (used by db_storage internals)."""
    legacy_db.set_current_user_id(user.id)


@router.get("/projects", response_model=ProjectsListOut)
def list_projects(user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    """Return the user's projects dict from PostgreSQL (key=`projects` in user_data).

    Mirrors what Streamlit's sidebar project selector reads. If the user hasn't
    configured projects yet, returns empty `{}` — same behaviour as legacy.
    """
    _bind_user(user)
    projects = legacy_config.load_projects() or {}
    return {"projects": projects, "count": len(projects)}


def _parse_iso(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


@router.get("/reports", response_model=ReportsBundleOut)
async def get_reports(
    response: Response,
    project: str = Query(..., description="Project ID, e.g. 'GANZA'"),
    period_from: Optional[str] = Query(None, alias="from"),
    period_to: Optional[str] = Query(None, alias="to"),
    basis: str = Query("accrual", pattern="^(accrual|cash)$"),
    fresh: bool = Query(False, description="Bypass cache and recompute from scratch"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Compute ОПиУ + ДДС + Баланс for one project + period in a single call.

    Returns three sub-objects matching the Streamlit "Отчёты" page tabs.
    Errors from any single computation are reported per-tab, not as 500.

    Wrapped in a durable read-through cache (`finance_compute_cache` table).
    First call after upload / settings change recomputes from scratch and
    stores the bundle; subsequent calls return cached JSONB in ~50ms.
    Pass `?fresh=1` to force recomputation.
    """
    _bind_user(user)
    projects = legacy_config.load_projects()
    if project not in projects:
        raise HTTPException(status_code=404, detail={"error": "project_not_found", "available": list(projects.keys())})

    pf = _parse_iso(period_from)
    pt = _parse_iso(period_to) or date.today()
    if pf is None:
        # default: 30 days back
        pf = date.fromordinal(pt.toordinal() - 30)

    out: dict[str, Any] = {
        "project": project,
        "period": {"from": pf.isoformat(), "to": pt.isoformat()},
        "basis": basis,
    }

    def _compute_bundle() -> dict[str, Any]:
        # Lazy imports — only pay the cost when we actually need to recompute.
        from v2.legacy.finance import compute_pnl, compute_cashflow, compute_balance
        from v2.legacy.reports import load_vendas_ml_report, has_1yr_bank_statements

        # Pre-warm the vendas DataFrame cache in the request context so all three
        # parallel compute tasks below find it hot. Without this the threads race
        # to rebuild the 2208-row df simultaneously (DB + pd.read_csv + df.apply)
        # and the wall time can exceed the shared timeout.
        try:
            load_vendas_ml_report()
        except Exception:
            pass  # individual computes will re-raise properly

        # P&L фильтруется выбранным периодом — показывает доходы/расходы отрезка.
        # ДДС и Баланс — всегда кумулятивно на сегодня: от даты запуска проекта
        # (launch_date / report_period start) или от минимальной найденной даты
        # продаж до today(). Иначе пользователь при узком периоде видит
        # отрицательный «закрывающий остаток», что не имеет экономического смысла.
        today = date.today()
        proj_meta = projects.get(project, {}) or {}
        cumul_start = _parse_iso((proj_meta.get("launch_date") or "").strip()[:10]) or pf
        rp = proj_meta.get("report_period", "")
        if rp and "/" in rp:
            rp_start = _parse_iso(rp.split("/")[0].strip())
            if rp_start and rp_start < (cumul_start or today):
                cumul_start = rp_start
        local_cumul_start = cumul_start or pf

        # Проверяем, есть ли ≥12 мес. банк-выписок — это разрешает применить
        # прогрессивный Simples Anexo I (RBT12). Иначе compute_das откатится
        # на faixa 1 nominal. Флаг `ml_only_revenue` на проекте даёт такое же
        # разрешение (см. legacy/tax_brazil.compute_das).
        has_1yr = has_1yr_bank_statements()

        results = _run_parallel_with_timeout({
            "pnl": lambda: compute_pnl(project, (pf, pt), basis=basis, has_1yr_bank_data=has_1yr),
            "cashflow": lambda: compute_cashflow(project, (local_cumul_start, today)),
            "balance": lambda: compute_balance(project, today, basis=basis, has_1yr_bank_data=has_1yr),
        })

        bundle: dict[str, Any] = {}
        pnl_res, pnl_err = results["pnl"]
        if pnl_res is not None: bundle["pnl"] = _dataclass_to_dict(pnl_res)
        if pnl_err: bundle["pnl_error"] = pnl_err

        cf_res, cf_err = results["cashflow"]
        if cf_res is not None: bundle["cashflow"] = _dataclass_to_dict(cf_res)
        if cf_err: bundle["cashflow_error"] = cf_err

        bal_res, bal_err = results["balance"]
        if bal_res is not None: bundle["balance"] = _dataclass_to_dict(bal_res)
        if bal_err: bundle["balance_error"] = bal_err
        return bundle

    # Pre-fetch every bank statement, parse it, apply user overrides
    # (per-bank `f2_classifications_grouped_*` and per-upload
    # `f2_classifications_*`), and stuff the merged list into a contextvar
    # so the sync `aggregate_classified_by_project` inside compute_*
    # can read it without touching the disk. This is the only way the
    # production app sees user classifications — the disk path stayed
    # broken on Railway because the FS is ephemeral.
    if pool is not None:
        try:
            from v2.services import bank_classifications as _bank_cls
            prefetched = await _bank_cls.prefetch_for_user(pool, user.id)
            _bank_cls.set_prefetched(prefetched)
        except Exception:  # noqa: BLE001
            # Don't block the report on a prefetch failure — fall back to
            # legacy disk path (which simply returns empty in production)
            pass

    cache_key = f"reports:{project}:{pf.isoformat()}:{pt.isoformat()}:{basis}"
    # Don't cache partial / errored bundles — half-computed results would
    # be served indefinitely until next user input change.
    bundle, status = finance_cache.cached_compute(
        user.id, cache_key, _compute_bundle,
        force=fresh,
        should_cache=lambda b: not any(k.endswith("_error") for k in b),
    )
    out.update(bundle)
    response.headers["X-Cache"] = status
    return out


@router.get("/services-reports", response_model=ServicesReportsBundleOut)
async def get_services_reports(
    response: Response,
    project: str = Query(..., description="Project ID, must have type='services'"),
    period_from: Optional[str] = Query(None, alias="from"),
    period_to: Optional[str] = Query(None, alias="to"),
    fresh: bool = Query(False, description="Bypass cache and recompute from scratch"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Invoice-based ОПиУ + ДДС + Balance for services-projects (Estonia /
    GANZA-USD / `compensation_mode='rental'|'profit_share'`).

    Wraps `v2/services/services_reports.compute_for_user` with the same
    durable read-through cache as the ecom `/reports` endpoint
    (`finance_compute_cache` table). First call after data change recomputes;
    subsequent identical requests served from JSONB in ~50ms.

    Returns 404 if project doesn't exist; 400 if it's not a services project.
    Per-tab errors surface as `{pnl_error, cashflow_error, balance_error}`
    instead of a single 500 — same UX pattern as ecom reports.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    # Services-projects are superadmin-only — they expose hand-curated tax
    # brackets, RBT12 baseline decay, and partner contributions that affect
    # invoice-based ОПиУ + DAS calculations. Regular admins should not see
    # or edit them.
    if not _is_superadmin(user):
        raise HTTPException(
            status_code=403,
            detail={"code": "superadmin_required", "feature": "services_reports"},
        )
    _bind_user(user)

    projects = legacy_config.load_projects()
    if project not in projects:
        raise HTTPException(
            status_code=404,
            detail={"error": "project_not_found", "available": list(projects.keys())},
        )
    proj_meta = projects.get(project, {}) or {}
    if proj_meta.get("type") != "services":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "not_services_project",
                "got_type": proj_meta.get("type"),
                "hint": "use /finance/reports for ecom projects",
            },
        )

    pf = _parse_iso(period_from)
    pt = _parse_iso(period_to) or date.today()
    if pf is None:
        pf = date.fromordinal(pt.toordinal() - 365)  # default: 12 months back

    out: dict[str, Any] = {
        "project": project,
        "period": {"from": pf.isoformat(), "to": pt.isoformat()},
    }

    # Lazy import — keeps startup unaware of services_reports until something
    # actually hits this endpoint. If the module is missing (e.g. fresh deploy
    # before services_reports.py was committed), fail soft with 503 instead of
    # crashing the whole router on import.
    try:
        from v2.services import services_reports as services_svc
    except ImportError as err:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"error": "services_reports_unavailable", "reason": str(err)},
        )

    async def _compute_async() -> dict[str, Any]:
        bundle = await services_svc.compute_for_user(
            pool, user.id, project, period_from=pf, period_to=pt,
        )
        return {k: v for k, v in bundle.items() if k not in ("project", "period")}

    cache_key = f"services_reports:{project}:{pf.isoformat()}:{pt.isoformat()}"

    bundle: dict[str, Any]
    cache_status: str
    if fresh:
        bundle = await _compute_async()
        cache_status = "force"
    else:
        fp, deps = finance_cache.compute_fingerprint(user.id)
        cached = (
            finance_cache._read_cached(user.id, cache_key, fp) if fp else None  # noqa: SLF001
        )
        if cached is not None:
            bundle = cached
            cache_status = "hit"
        else:
            bundle = await _compute_async()
            has_error = any(k.endswith("_error") for k in bundle)
            if not has_error and fp:
                stored = finance_cache._write_cached(  # noqa: SLF001
                    user.id, cache_key, bundle, fp, deps,
                )
                cache_status = "miss" if stored else "compute_only"
            else:
                cache_status = "skip_cache"

    out.update(bundle)
    response.headers["X-Cache"] = cache_status
    return out


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass / nested dataclasses → dict for JSON serialisation."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


# ── SKU Mapping ─────────────────────────────────────────────────────────────
# 1:1 port of Streamlit page _admin/app.py:2848-3104 («Маппинг SKU → Проект»).
# - SKU list = vendas SKUs (titles + MLB) overlaid with catalog (project, cost, supplier)
# - Groups by alpha prefix → bulk project assignment per group
# - Per-project tabs → edit unit_cost_brl + supplier_type per row
# Storage: legacy `sku_catalog` user_data key (shared with Streamlit).

import re as _re_sku


async def _query_ml_items_sku_map(pool, user_id: int) -> dict[str, dict[str, str]]:
    """SELECT по ml_user_items + парсинг raw JSONB. Возвращает пустой dict
    если таблица пустая или у item'ов нет attributes/variations (старые кеши)."""
    import json as _json
    if pool is None:
        return {}
    result: dict[str, dict[str, str]] = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT item_id, title, permalink, raw FROM ml_user_items WHERE user_id = $1",
                user_id,
            )
    except Exception:  # noqa: BLE001
        return {}

    for row in rows:
        item_id = (row["item_id"] or "").strip()
        if not item_id:
            continue
        title = (row["title"] or "").strip()
        permalink = (row["permalink"] or "").strip()
        raw = row["raw"]
        if isinstance(raw, str):
            try:
                raw = _json.loads(raw)
            except Exception:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}

        def _add(sku_val: Any) -> None:
            sku = str(sku_val or "").strip()
            if not sku or sku.lower() == "nan":
                return
            nk = sku.upper()
            if nk in result:
                return
            result[nk] = {
                "mlb": item_id,
                "link": permalink,
                "title": title[:80],
            }

        # Легаси-поле: для item без вариаций SKU часто лежит здесь
        if raw.get("seller_custom_field"):
            _add(raw.get("seller_custom_field"))

        for attr in raw.get("attributes") or []:
            if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                _add(attr.get("value_name") or attr.get("value_id"))

        for var in raw.get("variations") or []:
            if not isinstance(var, dict):
                continue
            for attr in var.get("attributes") or []:
                if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                    _add(attr.get("value_name") or attr.get("value_id"))
            # Некоторые варианты хранят SKU в attribute_combinations
            for attr in var.get("attribute_combinations") or []:
                if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                    _add(attr.get("value_name") or attr.get("value_id"))

    return result


async def _load_ml_items_sku_map(pool, user_id: int) -> dict[str, dict[str, str]]:
    """seller_sku → {mlb, link, title} из ml_user_items.

    Если кеш есть, но raw не содержит attributes/variations (старые записи —
    их fetch не запрашивал эти поля), делаем единичный refresh active-листингов
    и пере-парсим. Дальше работает по кешу.
    """
    result = await _query_ml_items_sku_map(pool, user_id)
    if result:
        return result
    # Cache пустой ИЛИ старого формата — один раз обновляем
    try:
        from v2.services import ml_user_items as ml_items_svc
        await ml_items_svc.refresh_user_items(pool, user_id, status="active")
    except Exception:  # noqa: BLE001
        return {}
    return await _query_ml_items_sku_map(pool, user_id)


def _build_sku_mapping(user_id: int, ml_sku_map: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    """Build the merged SKU view: vendas + ml-cache overlay + catalog + groupings.

    Mirrors _admin/app.py:2867-2946 logic. Synchronous (sits inside FastAPI
    threadpool dispatch). `ml_sku_map` приходит из endpoint'а (async fetch
    кеша ml_user_items) и нужен чтобы у SKU без продаж тоже был MLB/link.
    """
    from v2.legacy.reports import load_vendas_ml_report
    from v2.legacy.sku_catalog import load_catalog, normalize_sku
    from v2.legacy.config import get_project_by_sku, mlb_url

    legacy_db.set_current_user_id(user_id)
    ml_sku_map = ml_sku_map or {}

    all_skus: dict[str, dict[str, Any]] = {}

    # 1) Vendas → seed list with titles, MLB, link
    try:
        vdf = load_vendas_ml_report()
        if vdf is not None and "SKU" in vdf.columns:
            title_col = next(
                (c for c in vdf.columns if "título" in c.lower() or "titulo" in c.lower()),
                None,
            )
            mlb_col = next(
                (c for c in vdf.columns
                 if c.startswith("#") and ("anúncio" in c.lower() or "anuncio" in c.lower())),
                None,
            )
            for _, row in vdf.iterrows():
                sku = str(row.get("SKU", "")).strip()
                if not sku or sku.lower() == "nan":
                    continue
                nk = normalize_sku(sku)
                if nk in all_skus:
                    continue
                title = str(row.get(title_col, ""))[:80] if title_col else ""
                mlb = str(row.get(mlb_col, "")).strip() if mlb_col else ""
                all_skus[nk] = {
                    "sku": sku,
                    "title": title if title.lower() != "nan" else "",
                    "mlb": mlb if mlb.lower() != "nan" else "",
                    "link": mlb_url(mlb) if mlb else "",
                    "project": "",
                    "supplier_type": "local",
                    "unit_cost_brl": None,
                    "note": "",
                }
    except Exception:
        pass

    # 1.5) Обогатить из кеша ml_user_items (включая variations).
    # SKU которые не продавались всё равно получат MLB/link/title если они
    # опубликованы как листинг или вариация на ML. Уже заполненные поля из
    # vendas (более свежий title) не перетираем.
    for nk, ml_info in ml_sku_map.items():
        if nk in all_skus:
            entry = all_skus[nk]
            if not entry.get("mlb") and ml_info.get("mlb"):
                entry["mlb"] = ml_info["mlb"]
            if not entry.get("link") and ml_info.get("link"):
                entry["link"] = ml_info["link"]
            if not entry.get("title") and ml_info.get("title"):
                entry["title"] = ml_info["title"]
        else:
            all_skus[nk] = {
                "sku": nk,
                "title": ml_info.get("title", ""),
                "mlb": ml_info.get("mlb", ""),
                "link": ml_info.get("link", ""),
                "project": "",
                "supplier_type": "local",
                "unit_cost_brl": None,
                "note": "",
            }

    # 1.6) Обогатить из stock_full.xlsx — там колонка "Código do Anúncio" даёт
    # sku→MLB напрямую, без ML API. Покрывает SKU которые есть в стоке Full
    # но ещё не продавались (новые партии, свежие вариации).
    try:
        from v2.legacy.reports import load_stock_full
        stock = load_stock_full() or {}
        for proj_block in stock.values():
            sku_mlbs = (proj_block or {}).get("sku_mlbs") or {}
            sku_titles = (proj_block or {}).get("sku_titles") or {}
            for sku_raw, mlb_raw in sku_mlbs.items():
                nk = normalize_sku(sku_raw)
                if not nk:
                    continue
                mlb = str(mlb_raw or "").strip()
                if not mlb or mlb.lower() == "nan":
                    continue
                title = str(sku_titles.get(sku_raw) or "").strip()
                if nk in all_skus:
                    entry = all_skus[nk]
                    if not entry.get("mlb"):
                        entry["mlb"] = mlb
                        entry["link"] = mlb_url(mlb)
                    if not entry.get("title") and title:
                        entry["title"] = title[:80]
                else:
                    all_skus[nk] = {
                        "sku": str(sku_raw).strip(),
                        "title": title[:80],
                        "mlb": mlb,
                        "link": mlb_url(mlb),
                        "project": "",
                        "supplier_type": "local",
                        "unit_cost_brl": None,
                        "note": "",
                    }
    except Exception:
        pass

    # 2) Catalog overlay (project, cost, supplier + Dados Fiscais поля)
    for it in load_catalog():
        nk = normalize_sku(str(it.get("sku", "")))
        if not nk:
            continue
        cat_titulo = (it.get("titulo") or "").strip() if it.get("titulo") else ""
        cat_mlb = (it.get("mlb") or "").strip() if it.get("mlb") else ""

        if nk in all_skus:
            entry = all_skus[nk]
            entry["project"] = it.get("project", "") or ""
            entry["supplier_type"] = it.get("supplier_type", "local")
            entry["unit_cost_brl"] = it.get("unit_cost_brl")
            entry["note"] = it.get("note", "") or ""
            # Dados Fiscais заполняют пустые поля (vendas/cache title — приоритет)
            if not entry.get("title") and cat_titulo:
                entry["title"] = cat_titulo[:80]
            if not entry.get("mlb") and cat_mlb:
                entry["mlb"] = cat_mlb
                entry["link"] = mlb_url(cat_mlb)
        else:
            all_skus[nk] = {
                "sku": str(it.get("sku", "")).strip(),
                "title": cat_titulo[:80] if cat_titulo else "",
                "mlb": cat_mlb,
                "link": mlb_url(cat_mlb) if cat_mlb else "",
                "project": it.get("project", "") or "",
                "supplier_type": it.get("supplier_type", "local"),
                "unit_cost_brl": it.get("unit_cost_brl"),
                "note": it.get("note", "") or "",
            }

        # Пробросить остальные Dados Fiscais поля (если есть в каталоге)
        entry = all_skus[nk]
        for key in (
            "ncm", "origem_type", "origem_code", "ean", "csosn_venda",
            "peso_bruto_kg", "peso_liquido_kg", "supplier_state",
            "lead_time_days", "dados_fiscais_synced_at",
        ):
            val = it.get(key)
            if val not in (None, "") and not entry.get(key):
                entry[key] = val

    # 3) Fill missing projects via prefix/MLB resolver
    for nk, info in all_skus.items():
        cur = info.get("project", "")
        if not cur or cur == "NAO_CLASSIFICADO":
            resolved = get_project_by_sku(info["sku"], info.get("mlb", ""))
            info["project"] = "" if resolved == "NAO_CLASSIFICADO" else resolved

    # 4) Group by alpha prefix
    groups: dict[str, dict[str, Any]] = {}
    for nk, info in all_skus.items():
        sku = info["sku"]
        m = _re_sku.match(r"^([a-zA-Z]+)", sku)
        prefix = m.group(1).upper() if m else f"#{sku[:4]}"
        g = groups.setdefault(prefix, {"prefix": prefix, "skus": [], "sample_title": "", "project": ""})
        g["skus"].append(nk)
        if info["title"] and not g["sample_title"]:
            g["sample_title"] = info["title"][:50]
        if info["project"] and not g["project"]:
            g["project"] = info["project"]
    groups_sorted = sorted(groups.values(), key=lambda g: -len(g["skus"]))

    # 5) Project counts
    project_counts: dict[str, int] = {}
    for info in all_skus.values():
        p = info.get("project") or ""
        project_counts[p] = project_counts.get(p, 0) + 1

    project_ids = sorted(legacy_config.load_projects().keys())

    return {
        "skus": list(all_skus.values()),
        "groups": groups_sorted,
        "project_counts": project_counts,
        "project_ids": project_ids,
        "total": len(all_skus),
    }


@router.get("/cache/stats")
def get_cache_stats(
    scope: str = Query("user", pattern="^(user|global)$"),
    user: CurrentUser = Depends(current_user),
):
    """Diagnostic — what's currently in the finance compute cache.

    `scope=user` (default) → only this user's entries with cache_key + age +
    truncated fingerprint. `scope=global` → totals across all users (admin
    debugging).
    """
    if scope == "global":
        return finance_cache.stats_sync(user_id=None)
    return finance_cache.stats_sync(user_id=user.id)


@router.post("/cache/cleanup")
def post_cache_cleanup(
    max_age_days: int = Query(14, ge=1, le=365),
    user: CurrentUser = Depends(current_user),  # noqa: ARG001 — auth gate
):
    """Manual cleanup of cache rows older than max_age_days. Cron-callable too."""
    deleted = finance_cache.cleanup_stale_sync(max_age_days=max_age_days)
    return {"deleted": deleted, "max_age_days": max_age_days}


@router.get("/sku-mapping", response_model=SkuMappingOut)
async def get_sku_mapping(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    _bind_user(user)
    ml_sku_map = await _load_ml_items_sku_map(pool, user.id)
    return _build_sku_mapping(user.id, ml_sku_map)


@router.get("/pnl-gap-debug")
async def debug_pnl_gap(
    project: str = Query(..., description="Project ID, e.g. 'ARTUR'"),
    month: str = Query(..., description="Месяц YYYY-MM, e.g. '2026-03'"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Диагностика гэпа в PnL: показывает что лежит в vendas для project+month
    + что висит в NAO_CLASSIFICADO/пусто за этот месяц + orphan-пакеты.
    Помогает понять где недостающая выручка."""
    _bind_user(user)
    from v2.legacy.reports import load_vendas_ml_report, parse_brl
    from v2.legacy.config import mlb_url
    import pandas as pd

    try:
        # Принудительно сбросим кеш чтобы получить актуальные __project
        from v2.legacy.reports import invalidate_vendas_cache
        invalidate_vendas_cache(user.id)
    except Exception:
        pass

    vdf = load_vendas_ml_report()
    if vdf is None:
        return {"error": "vendas_ml не загружен"}

    # Парсим Data da venda → YYYY-MM
    pt_months = {
        "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
        "abril": "04", "maio": "05", "junho": "06", "julho": "07",
        "agosto": "08", "setembro": "09", "outubro": "10",
        "novembro": "11", "dezembro": "12",
    }

    def _row_month(date_str: str) -> str:
        s = (date_str or "").strip().lower()
        if not s:
            return ""
        # «24 de março de 2026 22:36 hs.»
        for pt, num in pt_months.items():
            if pt in s:
                # вытащить год — последняя группа из 4 цифр
                import re
                yrs = re.findall(r"\b(20\d{2})\b", s)
                if yrs:
                    return f"{yrs[0]}-{num}"
        return ""

    if "Data da venda" not in vdf.columns or "__project" not in vdf.columns:
        return {"error": "vendas DF без обязательных колонок (Data da venda / __project)"}

    # Считаем revenue per row: «Preço unitário × Unidades»
    def _row_rev(row) -> float:
        try:
            preco = parse_brl(row.get("Preço unitário de venda do anúncio (BRL)", 0))
            u_raw = row.get("Unidades", 0)
            if pd.isna(u_raw) or u_raw == "":
                return 0.0
            unidades = int(float(str(u_raw).strip()))
            return preco * unidades
        except Exception:
            return 0.0

    target_proj = ""
    target_naoclass = []  # список (sku, mlb, revenue, rows_count)
    naoclass_total = 0.0
    naoclass_rows = 0
    naoclass_by_sku: dict[str, dict] = {}
    target_total = 0.0
    target_rows = 0
    other_projects: dict[str, dict] = {}

    # Заранее построим MLB → title из vendas + ml_user_items для красивого вывода
    title_by_mlb: dict[str, str] = {}
    title_col = next(
        (c for c in vdf.columns if "título" in c.lower() or "titulo" in c.lower()),
        None,
    )
    mlb_col_v = next(
        (c for c in vdf.columns
         if c.startswith("#") and ("anúncio" in c.lower() or "anuncio" in c.lower())),
        None,
    )
    if title_col and mlb_col_v:
        for _, row in vdf.iterrows():
            m = str(row.get(mlb_col_v, "") or "").strip()
            t = str(row.get(title_col, "") or "").strip()
            if m and t and m not in title_by_mlb and t.lower() != "nan":
                title_by_mlb[m] = t[:80]
    # Также подтянем из ml_user_items (item_id, title)
    try:
        async with pool.acquire() as conn:
            ml_rows = await conn.fetch(
                "SELECT item_id, title FROM ml_user_items WHERE user_id = $1",
                user.id,
            )
        for r in ml_rows:
            iid = (r["item_id"] or "").strip()
            t = (r["title"] or "").strip()
            if iid and t and iid not in title_by_mlb:
                title_by_mlb[iid] = t[:80]
    except Exception:
        pass

    for _, row in vdf.iterrows():
        rm = _row_month(str(row.get("Data da venda", "")))
        if rm != month:
            continue
        proj = str(row.get("__project", "") or "").strip()
        rev = _row_rev(row)
        sku = str(row.get("SKU", "") or "").strip()
        mlb = str(row.get("# de anúncio", "") or "").strip()

        if proj == project:
            target_total += rev
            target_rows += 1
        elif proj in ("NAO_CLASSIFICADO", "", "nan"):
            naoclass_total += rev
            naoclass_rows += 1
            key = sku.upper() or f"<no-sku> {mlb}"
            if key not in naoclass_by_sku:
                naoclass_by_sku[key] = {
                    "sku": sku,
                    "mlb": mlb,
                    "title": title_by_mlb.get(mlb, ""),
                    "ml_url": mlb_url(mlb) if mlb else "",
                    "revenue": 0.0,
                    "rows": 0,
                }
            naoclass_by_sku[key]["revenue"] += rev
            naoclass_by_sku[key]["rows"] += 1
        else:
            if proj not in other_projects:
                other_projects[proj] = {"revenue": 0.0, "rows": 0}
            other_projects[proj]["revenue"] += rev
            other_projects[proj]["rows"] += 1

    # Топ-20 NAO_CLASSIFICADO SKU по выручке
    top_naoclass = sorted(
        naoclass_by_sku.values(), key=lambda x: -x["revenue"]
    )[:20]

    # Orphan pacotes
    orphans_summary: dict[str, Any] = {"count": 0, "total_brl": 0.0, "items": []}
    try:
        from v2.legacy.reports import list_orphan_pacotes
        orphans = list_orphan_pacotes() or []
        for o in orphans:
            o_month = _row_month(str(o.get("data", "")))
            if o_month != month:
                continue
            assigned = str(o.get("assigned_project") or "").strip()
            o_rev = float(o.get("total_brl", 0) or 0)
            orphans_summary["count"] += 1
            orphans_summary["total_brl"] += o_rev
            if len(orphans_summary["items"]) < 10:
                orphans_summary["items"].append({
                    "order_id": o.get("order_id"),
                    "data": o.get("data"),
                    "estado": o.get("estado"),
                    "total_brl": o_rev,
                    "assigned_project": assigned or None,
                    "ml_url": o.get("ml_url"),
                })
    except Exception as err:  # noqa: BLE001
        orphans_summary["error"] = str(err)

    return {
        "project": project,
        "month": month,
        "vendas_target_project": {
            "revenue_brl": round(target_total, 2),
            "rows": target_rows,
        },
        "vendas_nao_classificado": {
            "revenue_brl": round(naoclass_total, 2),
            "rows": naoclass_rows,
            "top_skus": top_naoclass,
        },
        "vendas_other_projects": {
            p: {"revenue_brl": round(v["revenue"], 2), "rows": v["rows"]}
            for p, v in sorted(other_projects.items(), key=lambda kv: -kv[1]["revenue"])
        },
        "orphan_pacotes": orphans_summary,
        "expected_total_if_all_attributed": round(
            target_total + naoclass_total + orphans_summary["total_brl"], 2
        ),
    }


@router.post("/pnl-gap-fix")
async def fix_pnl_gap(
    project: str = Query(..., description="Project ID кому привязать MLB, e.g. 'ARTHUR'"),
    month: str = Query(..., description="Месяц YYYY-MM, e.g. '2026-03'"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """One-click фикс гэпа: берёт все NAO_CLASSIFICADO MLB для project+month
    и добавляет их в mlb_fallback указанного проекта. Дальше vendas-rows с
    этими MLB будут попадать в нужный проект.

    Возвращает: количество добавленных MLB + те что уже были в fallback.
    """
    _bind_user(user)
    from v2.legacy.reports import load_vendas_ml_report, parse_brl, invalidate_vendas_cache
    from v2.legacy.config import (
        load_projects, update_project, _invalidate_projects_cache,
    )
    import pandas as pd
    import re

    invalidate_vendas_cache(user.id)
    vdf = load_vendas_ml_report()
    if vdf is None:
        return {"error": "vendas_ml не загружен"}

    pt_months = {
        "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
        "abril": "04", "maio": "05", "junho": "06", "julho": "07",
        "agosto": "08", "setembro": "09", "outubro": "10",
        "novembro": "11", "dezembro": "12",
    }

    def _row_month(date_str: str) -> str:
        s = (date_str or "").strip().lower()
        if not s:
            return ""
        for pt, num in pt_months.items():
            if pt in s:
                yrs = re.findall(r"\b(20\d{2})\b", s)
                if yrs:
                    return f"{yrs[0]}-{num}"
        return ""

    if "Data da venda" not in vdf.columns or "__project" not in vdf.columns:
        return {"error": "vendas DF без обязательных колонок"}

    # Собираем все уникальные NAO_CLASSIFICADO MLB за месяц
    naoclass_mlbs: set[str] = set()
    for _, row in vdf.iterrows():
        rm = _row_month(str(row.get("Data da venda", "")))
        if rm != month:
            continue
        proj = str(row.get("__project", "") or "").strip()
        if proj not in ("NAO_CLASSIFICADO", "", "nan"):
            continue
        mlb = str(row.get("# de anúncio", "") or "").strip()
        if mlb and mlb.lower() != "nan":
            naoclass_mlbs.add(mlb)

    if not naoclass_mlbs:
        return {
            "added": 0, "already_present": 0, "added_mlbs": [],
            "message": "Нет NAO_CLASSIFICADO MLB за этот период",
        }

    # Текущие mlb_fallback проекта
    projects = load_projects()
    pid = project.upper()
    if pid not in projects:
        return {"error": f"проект {pid} не найден"}
    current = list(projects[pid].get("mlb_fallback") or [])
    current_set = set(current)

    new_mlbs = [m for m in naoclass_mlbs if m not in current_set]
    already = [m for m in naoclass_mlbs if m in current_set]
    merged = current + new_mlbs

    ok = update_project(pid, {"mlb_fallback": merged})
    if not ok:
        return {"error": "update_project failed"}

    _invalidate_projects_cache()
    invalidate_vendas_cache(user.id)
    # catalog-project-index тоже сбросим — get_project_by_sku его читает
    try:
        from v2.legacy.config import invalidate_catalog_project_index
        invalidate_catalog_project_index()
    except Exception:
        pass

    return {
        "added": len(new_mlbs),
        "already_present": len(already),
        "added_mlbs": sorted(new_mlbs),
        "already_present_mlbs": sorted(already),
        "total_mlb_fallback_now": len(merged),
        "project": pid,
        "month": month,
    }


@router.get("/sku-mapping-debug")
async def debug_sku_mapping(
    sku: str = Query(..., description="SKU для диагностики, e.g. 'A21191-1'"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Диагностика — почему конкретный SKU не получил MLB/link.
    Проверяет все 4 источника + статус OAuth-токена."""
    _bind_user(user)
    from v2.legacy.sku_catalog import normalize_sku, load_catalog
    from v2.legacy.reports import load_vendas_ml_report, load_stock_full
    from v2.services import ml_oauth as ml_oauth_svc
    import json as _json

    nk = normalize_sku(sku)
    report: dict[str, Any] = {
        "sku_input": sku,
        "normalized": nk,
        "sources": {},
    }

    # 1) vendas_ml
    try:
        vdf = load_vendas_ml_report()
        found = False
        mlb = ""
        title = ""
        if vdf is not None and "SKU" in vdf.columns:
            mask = vdf["SKU"].astype(str).str.strip().str.upper() == nk
            matches = vdf[mask]
            if len(matches) > 0:
                found = True
                row = matches.iloc[0]
                mlb_col = next(
                    (c for c in vdf.columns
                     if c.startswith("#") and ("anúncio" in c.lower() or "anuncio" in c.lower())),
                    None,
                )
                if mlb_col:
                    mlb = str(row.get(mlb_col, "")).strip()
                title_col = next(
                    (c for c in vdf.columns if "título" in c.lower() or "titulo" in c.lower()),
                    None,
                )
                if title_col:
                    title = str(row.get(title_col, ""))[:80]
        report["sources"]["vendas_ml"] = {
            "found": found, "mlb": mlb, "title": title,
            "total_rows_in_file": int(len(vdf)) if vdf is not None else 0,
        }
    except Exception as err:  # noqa: BLE001
        report["sources"]["vendas_ml"] = {"error": str(err)}

    # 2) stock_full.xlsx
    try:
        stock = load_stock_full() or {}
        found = False
        mlb = ""
        title = ""
        qty = 0
        proj_found = ""
        for proj_name, block in stock.items():
            sku_mlbs = (block or {}).get("sku_mlbs") or {}
            sku_titles = (block or {}).get("sku_titles") or {}
            by_sku = (block or {}).get("by_sku") or {}
            for s, m in sku_mlbs.items():
                if normalize_sku(s) == nk:
                    found = True
                    mlb = str(m or "").strip()
                    title = str(sku_titles.get(s) or "").strip()[:80]
                    qty = int(by_sku.get(s) or 0)
                    proj_found = proj_name
                    break
            # Также проверим присутствие SKU в by_sku даже если в sku_mlbs нет
            if not found:
                for s in by_sku.keys():
                    if normalize_sku(s) == nk:
                        found = True
                        title = str(sku_titles.get(s) or "").strip()[:80]
                        qty = int(by_sku.get(s) or 0)
                        proj_found = proj_name
                        # mlb остаётся пустой — этого не было в sku_mlbs
                        break
            if found:
                break
        report["sources"]["stock_full"] = {
            "found": found, "mlb": mlb, "title": title, "qty": qty,
            "project": proj_found,
            "total_projects_in_stock": len(stock),
        }
    except Exception as err:  # noqa: BLE001
        report["sources"]["stock_full"] = {"error": str(err)}

    # 3) sku_catalog (legacy/db) — включая Dados Fiscais поля
    try:
        catalog = load_catalog()
        found = False
        record: dict[str, Any] = {}
        for it in catalog:
            if normalize_sku(str(it.get("sku", ""))) == nk:
                found = True
                record = {
                    "sku": it.get("sku", ""),
                    "project": it.get("project", "") or "",
                    "unit_cost_brl": it.get("unit_cost_brl"),
                    "supplier_type": it.get("supplier_type", "local"),
                    "titulo": it.get("titulo"),
                    "mlb": it.get("mlb"),
                    "ncm": it.get("ncm"),
                    "origem_type": it.get("origem_type"),
                    "ean": it.get("ean"),
                    "peso_bruto_kg": it.get("peso_bruto_kg"),
                    "dados_fiscais_synced_at": it.get("dados_fiscais_synced_at"),
                }
                break
        report["sources"]["sku_catalog"] = {
            "found": found, **record,
            "total_entries": len(catalog),
        }
    except Exception as err:  # noqa: BLE001
        report["sources"]["sku_catalog"] = {"error": str(err)}

    # 4) ml_user_items cache
    try:
        cache_summary: dict[str, Any] = {
            "total_items": 0,
            "items_with_attributes": 0,
            "items_with_variations": 0,
            "items_with_seller_custom_field": 0,
            "matched": False,
            "mlb": "",
            "matched_via": "",
            "matched_title": "",
            "sample_seller_skus": [],
        }
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT item_id, title, permalink, raw FROM ml_user_items WHERE user_id = $1",
                user.id,
            )
        cache_summary["total_items"] = len(rows)
        sample_skus: list[str] = []
        for row in rows:
            raw = row["raw"]
            if isinstance(raw, str):
                try:
                    raw = _json.loads(raw)
                except Exception:
                    raw = {}
            if not isinstance(raw, dict):
                continue
            if raw.get("attributes"):
                cache_summary["items_with_attributes"] += 1
            if raw.get("variations"):
                cache_summary["items_with_variations"] += 1
            if raw.get("seller_custom_field"):
                cache_summary["items_with_seller_custom_field"] += 1

            # Поиск SKU + сбор sample
            def _check(sku_val, via: str) -> bool:
                s = str(sku_val or "").strip()
                if not s or s.lower() == "nan":
                    return False
                if len(sample_skus) < 30 and s.upper() not in sample_skus:
                    sample_skus.append(s.upper())
                if s.upper() == nk and not cache_summary["matched"]:
                    cache_summary["matched"] = True
                    cache_summary["mlb"] = row["item_id"]
                    cache_summary["matched_via"] = via
                    cache_summary["matched_title"] = (row["title"] or "")[:80]
                    return True
                return False

            if raw.get("seller_custom_field"):
                _check(raw.get("seller_custom_field"), "seller_custom_field")
            for attr in raw.get("attributes") or []:
                if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                    _check(attr.get("value_name") or attr.get("value_id"), "attributes.SELLER_SKU")
            for var in raw.get("variations") or []:
                if not isinstance(var, dict):
                    continue
                for attr in var.get("attributes") or []:
                    if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                        _check(attr.get("value_name") or attr.get("value_id"), "variations.attributes.SELLER_SKU")
                for attr in var.get("attribute_combinations") or []:
                    if isinstance(attr, dict) and attr.get("id") == "SELLER_SKU":
                        _check(attr.get("value_name") or attr.get("value_id"), "variations.attribute_combinations.SELLER_SKU")

        cache_summary["sample_seller_skus"] = sample_skus[:30]
        report["sources"]["ml_user_items_cache"] = cache_summary
    except Exception as err:  # noqa: BLE001
        report["sources"]["ml_user_items_cache"] = {"error": str(err)}

    # 5) OAuth token status
    try:
        tokens = await ml_oauth_svc.load_user_tokens(pool, user.id)
        if tokens:
            from datetime import datetime as _dt, timezone as _tz
            exp = tokens.get("access_token_expires_at")
            now = _dt.now(_tz.utc)
            valid = bool(exp and exp > now)
            report["ml_oauth"] = {
                "has_token": True,
                "valid_now": valid,
                "expires_at": exp.isoformat() if exp else None,
                "ml_user_id": tokens.get("ml_user_id"),
                "ml_nickname": tokens.get("ml_nickname"),
                "ml_site_id": tokens.get("ml_site_id"),
                "scope": tokens.get("scope"),
                "last_refreshed_at": (
                    tokens.get("last_refreshed_at").isoformat()
                    if tokens.get("last_refreshed_at") else None
                ),
            }
        else:
            report["ml_oauth"] = {"has_token": False}
    except Exception as err:  # noqa: BLE001
        report["ml_oauth"] = {"error": str(err)}

    return report


@router.get("/pnl-matrix", response_model=PnlMatrixOut)
def get_pnl_matrix(
    response: Response,
    project: str = Query(..., description="Project ID, e.g. 'ARTHUR'"),
    fresh: bool = Query(False, description="Bypass cache and recompute from scratch"),
    user: CurrentUser = Depends(current_user),
):
    """Monthly PnL matrix (rows × 12 months): revenue breakdown + expenses +
    summary (op_profit, margin, orders). Mirrors Streamlit `build_monthly_pnl_matrix`.

    Wrapped in the same durable cache as /reports. Pass `?fresh=1` to force.
    On compute timeout returns empty months/rows (and skips caching).
    """
    _bind_user(user)

    def _compute_matrix() -> dict[str, Any]:
        from v2.legacy.reports import build_monthly_pnl_matrix
        results = _run_parallel_with_timeout({
            "matrix": lambda: build_monthly_pnl_matrix(project),
        })
        data, err = results["matrix"]
        if data is None or err:
            import sys
            print(f"[pnl-matrix] project={project} err={err}", file=sys.stderr, flush=True)
            return {"project": project, "months": [], "years": [], "rows": [], "_error": err or "no_data"}
        return {
            "project": project,
            "months": data.get("months", []),
            "years": data.get("years", []),
            "rows": data.get("rows", []),
        }

    cache_key = f"matrix:{project}"
    payload, status = finance_cache.cached_compute(
        user.id, cache_key, _compute_matrix,
        force=fresh,
        should_cache=lambda p: not p.get("_error") and bool(p.get("months")),
    )
    response.headers["X-Cache"] = status
    # Strip internal marker before returning to UI.
    payload.pop("_error", None)
    return payload


@router.post("/sku-mapping/save", response_model=SkuBulkSaveOut)
def save_sku_mapping(body: SkuBulkSaveIn, user: CurrentUser = Depends(current_user)):
    """Bulk-update catalog rows. Each update merges into the existing catalog
    item (or creates a new one). After save, both project-resolver and projects
    cache are invalidated.
    """
    _bind_user(user)
    from v2.legacy.sku_catalog import load_catalog, save_catalog, normalize_sku
    from v2.legacy.config import invalidate_catalog_project_index, _invalidate_projects_cache

    by_key: dict[str, dict[str, Any]] = {}
    for it in load_catalog():
        nk = normalize_sku(str(it.get("sku", "")))
        if nk:
            by_key[nk] = dict(it)

    saved = 0
    for u in body.updates:
        nk = normalize_sku(u.sku)
        if not nk:
            continue
        item = by_key.get(nk) or {
            "sku": u.sku.strip(),
            "project": "",
            "supplier_type": "local",
            "unit_cost_brl": None,
            "note": "",
        }
        if u.project is not None:
            item["project"] = u.project
        if u.supplier_type is not None:
            stp = u.supplier_type.strip().lower()
            item["supplier_type"] = stp if stp in ("local", "import") else "local"
        if u.unit_cost_brl is not None:
            item["unit_cost_brl"] = u.unit_cost_brl
        if u.extra_fixed_cost_brl is not None:
            item["extra_fixed_cost_brl"] = u.extra_fixed_cost_brl
        if u.note is not None:
            item["note"] = u.note
        by_key[nk] = item
        saved += 1

    save_catalog(list(by_key.values()))
    invalidate_catalog_project_index()
    _invalidate_projects_cache()
    # Vendas DataFrame кеширует __project per-row при загрузке. После
    # переназначения SKU→project старые строки в кеше остаются с прежним
    # проектом, поэтому PnL не подхватывал новые маппинги. Сбрасываем.
    from v2.legacy.reports import invalidate_vendas_cache
    invalidate_vendas_cache(user.id)
    return {"saved": saved, "catalog_total": len(by_key)}


# ── Orphan Pacotes (multi-item ML orders with no child-SKU rows) ────────────
# Mirrors Streamlit flow in _admin/report_views.py:1695-1768 + upload_page.py:1126-1204.
# Storage: per-user `f2_orphan_assignments` in user_data (DB mode) or fs fallback.

_ML_ORDER_URL = "https://www.mercadolivre.com.br/vendas/{oid}/detalhe"


@router.get("/orphan-pacotes", response_model=OrphanPacotesResponse)
def get_orphan_pacotes(user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    """List all «Pacote de N produtos» orders with no resolvable SKU.

    Each item includes the ML order-detail link so the user can check the
    cart content manually before assigning a project. `assigned_project`
    reflects the current saved choice (or null).
    """
    _bind_user(user)

    from v2.legacy.reports import list_orphan_pacotes, load_orphan_assignments

    rows = list_orphan_pacotes() or []
    manual = load_orphan_assignments() or {}

    projects = legacy_config.load_projects() or {}
    available = sorted(projects.keys())

    items: list[dict[str, Any]] = []
    unassigned_count = 0
    unassigned_total = 0.0
    for r in rows:
        oid = str(r.get("order_id", ""))
        assigned = manual.get(oid)
        items.append({
            "order_id": oid,
            "data": r.get("data", ""),
            "estado": r.get("estado", ""),
            "bucket": r.get("bucket", ""),
            "comprador": r.get("comprador", ""),
            "total_brl": float(r.get("total_brl", 0) or 0),
            "ml_url": _ML_ORDER_URL.format(oid=oid),
            "assigned_project": assigned,
        })
        if not assigned:
            unassigned_count += 1
            if r.get("bucket") == "delivered":
                unassigned_total += float(r.get("total_brl", 0) or 0)

    return {
        "items": items,
        "unassigned_count": unassigned_count,
        "unassigned_total_brl": round(unassigned_total, 2),
        "available_projects": available,
    }


@router.post("/orphan-pacotes/save", response_model=OrphanSaveOut)
def save_orphan_pacotes(
    body: OrphanSaveIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Bulk-assign orphan pacote → project. null value clears the assignment.

    Returns the number of records that actually changed plus the total number
    of persisted assignments after the save.
    """
    _bind_user(user)

    from v2.legacy.reports import save_orphan_assignments_bulk, load_orphan_assignments

    changed = save_orphan_assignments_bulk(body.assignments or {})
    total = len(load_orphan_assignments() or {})
    return {"saved": changed, "total_assignments": total}


# ── Retirada Overrides (per-row политика «списание / в обороте») ────────────

@router.get("/retirada-overrides")
def get_retirada_overrides(
    project: str = Query(..., min_length=1),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Map текущих override'ов retirada-операций для проекта.

    Если override не задан, в response будет `overrides: {}` — это значит
    `Forma de retirada` берётся из ML-отчёта (Envio/Descarte) без изменений.
    """
    _bind_user(user)
    from v2.legacy.reports import load_retirada_overrides
    raw = load_retirada_overrides(project) or {}
    # Канонизируем для UI: descarte/envio в lowercase.
    canon = {
        cid: ("descarte" if str(forma).strip().lower().startswith("descart") else "envio")
        for cid, forma in raw.items()
    }
    return {"project": project, "overrides": canon, "saved_count": 0}


@router.get("/retirada/preview", response_model=RetiradaPreviewOut)
def get_retirada_preview(
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Все retirada-операции пользователя без project/period-фильтра.

    Используется UI-модалкой после upload Relatorio_Tarifas_Full_*.xlsx —
    пользователь сразу размечает что списано, а что вернётся в оборот.
    Каждая строка содержит уже применённый override (если задан) и
    оригинальную ML forma.
    """
    _bind_user(user)
    from v2.legacy.reports import list_all_retirada_rows
    rows = list_all_retirada_rows() or []
    projects = sorted({str(r.get("project") or "").strip() for r in rows if r.get("project")})
    return {"rows": rows, "rows_count": len(rows), "projects": projects}


@router.post("/retirada-overrides")
def save_retirada_overrides_endpoint(
    body: RetiradaOverridesIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Замена (replace, не merge) override-карты для проекта.

    Передай `overrides: []` чтобы сбросить все overrides проекта целиком.
    legacy/reports.save_retirada_overrides канонизирует значения — допустимо
    только `descarte` или `envio`, всё остальное отбрасывается.
    """
    _bind_user(user)
    from v2.legacy.reports import save_retirada_overrides, load_retirada_overrides
    overrides_map: dict[str, str] = {}
    for item in body.overrides or []:
        cid = str(item.custo_id or "").strip()
        if not cid:
            continue
        overrides_map[cid] = str(item.forma or "").strip().lower()
    ok = save_retirada_overrides(body.project, overrides_map)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save retirada overrides")
    saved = load_retirada_overrides(body.project) or {}
    canon = {
        cid: ("descarte" if str(forma).strip().lower().startswith("descart") else "envio")
        for cid, forma in saved.items()
    }
    return {"project": body.project, "overrides": canon, "saved_count": len(canon)}


# ── Uploads (Phase 4 — /finance/upload) ─────────────────────────────────────
# Thin wrappers around `v2/storage/uploads_storage` — the SHA-dedupe upsert,
# list-by-source query, and per-row delete are already implemented there.

_ALLOWED_SOURCES = set(legacy_config.DATA_SOURCES.keys())


@router.get("/upload-sources", response_model=SourceCatalogOut)
def list_upload_sources(user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    """Catalog of valid `source_key` values with display metadata.

    Used by the upload UI to render the manual-source fallback selector when
    filename-based auto-detection fails.
    """
    _ = user  # auth only — no user-scoped data in this endpoint
    entries = []
    for key, meta in legacy_config.DATA_SOURCES.items():
        entries.append({
            "key": key,
            "name": str(meta.get("name", key)),
            "file_pattern": str(meta.get("file_pattern", "")),
            "frequency": str(meta.get("frequency", "")),
            "type": str(meta.get("type", "")),
            "description": str(meta.get("description", "")),
        })
    return {"sources": entries}


@router.get("/uploads", response_model=UploadsListOut)
async def list_uploads(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """List per-user uploads grouped by source_key, newest first within group."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    source_counts = await uploads_storage.list_sources(pool, user.id)
    groups: list[dict[str, Any]] = []
    total = 0
    for source_key, count in sorted(source_counts.items()):
        files = await uploads_storage.fetch_files_by_source(pool, user.id, source_key)
        items = [
            {
                "id": f.id,
                "filename": f.filename,
                "size_bytes": len(f.file_bytes),
                "created_at": f.created_at.isoformat() if f.created_at else "",
            }
            for f in files
        ]
        groups.append({
            "source_key": source_key or "",
            "count": count,
            "items": items,
        })
        total += count
    return {"sources": groups, "total_count": total}


_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB — comfortable headroom over ~1.5 MB vendas snapshots


@router.post("/uploads", response_model=UploadSaveOut)
async def create_upload(
    file: UploadFile = File(..., description="File to store"),
    source_key: Optional[str] = Form(None, description="Override auto-detect"),
    password: Optional[str] = Form(None, description="Password for encrypted PDF/Excel"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Store a user file in the `uploads` table with SHA256 dedupe.

    For encrypted PDF/Excel: tries user-supplied `password` + per-user known
    passwords + hardcoded defaults. Decrypted bytes land in `uploads`.
    On auth failure returns HTTP 423 with error=password_required.

    `source_key` is resolved in this order:
      1. explicit `source_key` form field (client override)
      2. filename-based auto-detect + column sniff
      3. HTTP 400 if neither produced a valid source_key
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    from v2.legacy.source_detection import detect_source
    from v2.legacy.reports import invalidate_vendas_cache, invalidate_mlb_to_sku_index
    from v2.legacy.unlocker import try_unlock, add_known_password

    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="missing_filename")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="empty_file")
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "max_bytes": _MAX_UPLOAD_BYTES, "got": len(file_bytes)},
        )

    # ── Try password-unlock for PDF / encrypted XLSX ──
    # `try_unlock` returns ("not_encrypted", "unlocked", "failed", "unsupported")
    unlocked_pwd: Optional[str] = None
    fname_ext = filename.lower().rsplit(".", 1)[-1]
    if fname_ext in ("pdf", "xlsx", "xls"):
        decrypted, used_pwd, status = try_unlock(file_bytes, filename, password)
        if status == "failed":
            raise HTTPException(
                status_code=423,  # Locked
                detail={
                    "error": "password_required",
                    "filename": filename,
                    "hint": "file is encrypted; POST with form field `password` to unlock",
                },
            )
        if status == "unlocked" and decrypted is not None:
            file_bytes = decrypted
            unlocked_pwd = used_pwd
            # Persist the password only if user explicitly passed it — avoid
            # double-saving the hardcoded defaults.
            if password and used_pwd == password.strip():
                add_known_password(password)

    detected = False
    resolved_key = (source_key or "").strip() or None
    if resolved_key is None:
        resolved_key = detect_source(filename, file_bytes)
        detected = resolved_key is not None

    if resolved_key is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ambiguous_source",
                "filename": filename,
                "hint": "provide `source_key` form field from GET /finance/upload-sources",
            },
        )
    if resolved_key not in _ALLOWED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_source_key", "source_key": resolved_key},
        )

    # Probe whether the SHA256 already exists for this user so we can report
    # was_duplicate=true (put_file upserts — no way to tell from its return value).
    pre_count = (await uploads_storage.list_sources(pool, user.id)).get(resolved_key, 0)

    upload_id = await uploads_storage.put_file(
        pool,
        user_id=user.id,
        source_key=resolved_key,
        filename=filename,
        file_bytes=file_bytes,
    )

    post_count = (await uploads_storage.list_sources(pool, user.id)).get(resolved_key, 0)
    was_duplicate = post_count == pre_count  # no new row → SHA conflict path fired

    # Invalidate per-user vendas DF cache so /reports + /pnl-matrix refresh
    # without waiting for the 120s TTL (fingerprint already flips on created_at).
    if resolved_key == "vendas_ml":
        invalidate_vendas_cache(user.id)
    # Vendas and stock_full are the sources of the MLB→SKU index used by the
    # ads parser to resolve project for each campaign.
    if resolved_key in ("vendas_ml", "stock_full"):
        invalidate_mlb_to_sku_index(user.id)

    # Dados Fiscais: автоматический sync в sku_catalog
    # Парсит файл (sheet "Produtos Únicos") → мёрджит NCM/Origem/Peso/CSOSN/Custo
    # для всех SKU юзера. overwrite_costs=True — считаем Dados Fiscais авторитетным
    # источником cost (юзер сам заполнил правильные цифры в ML-файле).
    dados_fiscais_sync: Optional[dict[str, Any]] = None
    if resolved_key == "dados_fiscais":
        try:
            from v2.parsers.dados_fiscais import parse_dados_fiscais_bytes
            from v2.legacy.sku_catalog import sync_from_dados_fiscais
            parsed = parse_dados_fiscais_bytes(file_bytes)
            if parsed:
                dados_fiscais_sync = sync_from_dados_fiscais(parsed, overwrite_costs=True)
            else:
                dados_fiscais_sync = {"error": "empty_or_invalid_sheet"}
        except Exception as e:
            dados_fiscais_sync = {"error": f"{type(e).__name__}: {e}"}

    # DAS Simples Nacional: parse PDF → store extracted {month, total, irpj/csll/...}
    # in `uploads.parsed_meta` JSONB. UI shows the parsed values next to the file
    # row instead of just "das_simples.pdf · uploaded 2 min ago".
    das_parsed: Optional[dict[str, Any]] = None
    if resolved_key == "das_simples":
        try:
            from v2.parsers.das_simples import parse_das_simples_bytes
            das_parsed = parse_das_simples_bytes(file_bytes, filename)
            if das_parsed:
                await uploads_storage.set_parsed_meta(pool, upload_id, das_parsed)
        except Exception as e:
            das_parsed = {"error": f"{type(e).__name__}: {e}"}

    return {
        "id": upload_id,
        "filename": filename,
        "source_key": resolved_key,
        "detected": detected,
        "size_bytes": len(file_bytes),
        "was_duplicate": was_duplicate,
        "unlocked": unlocked_pwd is not None,
        "dados_fiscais_sync": dados_fiscais_sync,
        "das_parsed": das_parsed,
    }


@router.delete("/uploads/{upload_id}")
async def delete_upload(
    upload_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Delete a single upload row.

    Authz: owners (uploads.user_id == caller) and admins always pass. Project
    members of the upload's project pass if their role is at least analyst
    AND the upload was created on/after their effective_from cut-off. See
    project_members.enforce_caller_can_delete for the full decision tree.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    from v2.legacy.reports import invalidate_vendas_cache, invalidate_mlb_to_sku_index
    from v2.services import project_members as pm_svc

    # Load the row first — the authz check needs owner_id, project_name and
    # created_at, not just an existence probe.
    target = await uploads_storage.get_file_by_id(pool, upload_id)
    if target is None:
        raise HTTPException(status_code=404, detail="upload_not_found")

    # Raises 403 with a structured detail if the caller can't delete this row.
    await pm_svc.enforce_caller_can_delete(
        pool,
        caller_id=user.id,
        caller_role=user.role,
        record_owner_id=target.user_id,
        record_project_name=target.project_name,
        record_created_at=target.created_at,
    )

    ok = await uploads_storage.delete_file_by_id(pool, upload_id)
    if not ok:
        raise HTTPException(status_code=404, detail="upload_not_found")

    # Invalidate caches under the OWNER's id, not the caller — caches key by
    # the user_id that originally produced the upload, so a member's delete
    # must clear the owner's cache too.
    cache_user = target.user_id if target.user_id is not None else user.id
    if target.source_key == "vendas_ml":
        invalidate_vendas_cache(cache_user)
    if target.source_key in ("vendas_ml", "stock_full"):
        invalidate_mlb_to_sku_index(cache_user)

    return {"deleted": True}


@router.get("/uploads/{upload_id}/preview", response_model=UploadPreviewOut)
async def preview_upload(
    upload_id: int,
    limit: int = Query(20, ge=1, le=200),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Return a lightweight preview of an uploaded file — first N rows, columns,
    and a source-specific summary (date range, row count, totals).

    Used by the /finance/upload page to let the user sanity-check what landed
    in the DB before running reports.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)

    import io
    import pandas as pd

    stored = await uploads_storage.get_file(pool, user.id, upload_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="upload_not_found")

    out: dict[str, Any] = {
        "upload_id": upload_id,
        "filename": stored.filename,
        "source_key": stored.source_key,
        "size_bytes": len(stored.file_bytes),
        "total_rows": None,
        "columns": [],
        "rows": [],
        "summary": {},
        "parse_error": None,
    }

    # Route to the appropriate parser based on source_key
    df: "pd.DataFrame | None" = None
    try:
        if stored.source_key == "vendas_ml":
            from v2.legacy.reports import _parse_one_vendas_file
            df = _parse_one_vendas_file(stored.filename, stored.file_bytes)
        elif stored.source_key in ("extrato_mp", "extrato_nubank", "extrato_c6_brl", "extrato_c6_usd"):
            from v2.legacy.bank_tx import _read_bank_csv
            df = _read_bank_csv(stored.source_key, stored.file_bytes)
        else:
            # Generic CSV/XLSX sniff
            fname = stored.filename.lower()
            if fname.endswith(".csv"):
                for sep in (";", ","):
                    for skip in (0, 1, 4, 5):
                        try:
                            df_try = pd.read_csv(
                                io.BytesIO(stored.file_bytes), sep=sep, skiprows=skip,
                                encoding="utf-8-sig", low_memory=False, nrows=limit + 1,
                            )
                            if len(df_try.columns) > 2:
                                df = df_try
                                break
                        except Exception:
                            continue
                    if df is not None:
                        break
            elif fname.endswith((".xlsx", ".xls")):
                for skip in (0, 4, 5, 6):
                    try:
                        df_try = pd.read_excel(
                            io.BytesIO(stored.file_bytes), sheet_name=0,
                            skiprows=skip, nrows=limit + 1,
                        )
                        if len(df_try.columns) > 1:
                            df = df_try
                            break
                    except Exception:
                        continue
    except Exception as e:
        out["parse_error"] = f"{type(e).__name__}: {e}"
        return out

    if df is None or df.empty:
        out["parse_error"] = "empty_or_unparseable"
        return out

    out["total_rows"] = int(len(df))
    out["columns"] = [str(c) for c in df.columns][:40]

    # Convert first N rows to JSON-safe dicts
    def _clean(v: Any) -> Any:
        if pd.isna(v):
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        return str(v)

    head = df.head(limit)
    out["rows"] = [
        {str(k): _clean(v) for k, v in row.items()}
        for _, row in head.iterrows()
    ]

    # Source-specific summary
    summary: dict[str, Any] = {}
    if stored.source_key == "vendas_ml" and "Data da venda" in df.columns:
        dates = df["Data da venda"].dropna().astype(str)
        if len(dates):
            summary["date_first"] = str(dates.iloc[-1])[:40]
            summary["date_last"] = str(dates.iloc[0])[:40]
        if "Receita por produtos (BRL)" in df.columns:
            summary["bruto_total"] = float(
                pd.to_numeric(df["Receita por produtos (BRL)"], errors="coerce").sum()
            )
        if "Estado" in df.columns:
            summary["statuses"] = {
                str(k): int(v) for k, v in df["Estado"].value_counts().head(5).items()
            }
    elif stored.source_key.startswith("extrato_"):
        if "valor" in [c.lower() for c in df.columns] or "entrada" in [c.lower() for c in df.columns]:
            summary["note"] = "values will be parsed during classification"
    out["summary"] = summary

    return out


# ── Classification + Bank Rules (Phase 5) ───────────────────────────────────
# Thin wrappers around `v2/legacy/config.{load,save}_transaction_rules` + the
# new `v2/legacy/bank_tx.parse_bank_tx_bytes` CSV sniffer.

_BANK_SOURCES = {"extrato_mp", "extrato_nubank", "extrato_c6_brl", "extrato_c6_usd"}


@router.get("/rules", response_model=RulesOut)
def get_rules(user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    """Return the user's transaction-classification rules (user_data key `transaction_rules`)."""
    _bind_user(user)
    from v2.legacy.config import load_transaction_rules
    rules = load_transaction_rules() or []
    return {"rules": rules, "count": len(rules)}


@router.put("/rules", response_model=RulesOut)
def put_rules(body: RulesSaveIn, user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    """Replace the full rules list. Normalization happens inside save_transaction_rules."""
    _bind_user(user)
    from v2.legacy.config import save_transaction_rules, load_transaction_rules
    save_transaction_rules([r.model_dump() for r in body.rules])
    rules = load_transaction_rules() or []
    return {"rules": rules, "count": len(rules)}


def _overrides_key(upload_id: int) -> str:
    return f"f2_classifications_{upload_id}"


@router.get("/transactions/{upload_id}", response_model=TransactionsOut)
async def get_transactions(
    upload_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Parse a stored bank-statement file, apply rules, merge saved overrides.

    Response shape: list of rows with category/project/label + dropdown options.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)

    from v2.legacy.bank_tx import parse_bank_tx_bytes, looks_like_pdf, CATEGORY_OPTIONS
    from v2.legacy.db_storage import db_load

    stored = await uploads_storage.get_file(pool, user.id, upload_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="upload_not_found")
    if stored.source_key not in _BANK_SOURCES:
        raise HTTPException(
            status_code=400,
            detail={"error": "not_a_bank_statement", "source_key": stored.source_key,
                    "supported": sorted(_BANK_SOURCES)},
        )

    # #2: PDFs are handled per-bank inside parse_bank_tx_bytes (currently:
    # extrato_c6_usd via pdfplumber). For other banks the PDF path returns
    # [] and we surface "pdf_not_supported" so the UI shows a clear hint.
    is_pdf = looks_like_pdf(stored.file_bytes)
    rows = parse_bank_tx_bytes(stored.source_key, stored.file_bytes)
    format_error: Optional[str] = None
    if not rows:
        format_error = "pdf_not_supported" if is_pdf else "empty_after_parse"

    # Merge saved overrides (keyed by upload_id, indexed by idx)
    overrides = db_load(_overrides_key(upload_id)) or {}
    if not isinstance(overrides, dict):
        overrides = {}
    for r in rows:
        ov = overrides.get(str(r["idx"]))
        if isinstance(ov, dict):
            if ov.get("category"):
                r["category"] = ov["category"]
                r["confidence"] = "manual"
                r["auto"] = False
            if ov.get("project") is not None:
                r["project"] = ov["project"]
            if ov.get("label"):
                r["label"] = ov["label"]

    projects = sorted((legacy_config.load_projects() or {}).keys())
    return {
        "upload_id": upload_id,
        "source_key": stored.source_key,
        "filename": stored.filename,
        "rows": rows,
        "categories": CATEGORY_OPTIONS,
        "projects": projects,
        "saved_overrides_count": len(overrides),
        "format_error": format_error,
    }


@router.post("/transactions/{upload_id}/save", response_model=ClassificationSaveOut)
async def save_transactions(
    upload_id: int,
    body: ClassificationSaveIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Persist per-row overrides (category/project/label) for a specific upload.

    Empty overrides (null for all fields) clear the saved entry.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)

    from v2.legacy.db_storage import db_load, db_save

    # Ownership check — don't let clients write overrides for uploads they don't own
    stored = await uploads_storage.get_file(pool, user.id, upload_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="upload_not_found")

    current = db_load(_overrides_key(upload_id)) or {}
    if not isinstance(current, dict):
        current = {}

    saved = 0
    for ov in body.overrides:
        key = str(ov.idx)
        entry = current.get(key, {}) if isinstance(current.get(key), dict) else {}
        has_content = any(v for v in (ov.category, ov.project, ov.label))
        if not has_content:
            if key in current:
                current.pop(key, None)
                saved += 1
            continue
        if ov.category is not None:
            entry["category"] = ov.category
        if ov.project is not None:
            entry["project"] = ov.project
        if ov.label is not None:
            entry["label"] = ov.label
        current[key] = entry
        saved += 1

    db_save(_overrides_key(upload_id), current)
    return {"saved": saved, "total_overrides": len(current)}


# ── Grouped (per-bank) transactions (#1) ────────────────────────────────────
# Same data as /transactions/{upload_id} but merged across ALL uploads for
# one source_key, deduplicated by stable hash, with overrides keyed by
# tx_hash so re-uploading the same statement does not lose categorizations.

_GROUPED_OVERRIDES_KEY_PREFIX = "f2_classifications_grouped_"

_BANK_LABELS = {
    "extrato_mp": "Mercado Pago",
    "extrato_nubank": "Nubank",
    "extrato_c6_brl": "C6 BRL",
    "extrato_c6_usd": "C6 USD",
}


def _grouped_overrides_key(source_key: str) -> str:
    return f"{_GROUPED_OVERRIDES_KEY_PREFIX}{source_key}"


@router.get("/bank-transactions/grouped")
async def get_bank_transactions_grouped(
    source_key: str = Query(..., description="extrato_nubank | extrato_mp | extrato_c6_brl | extrato_c6_usd"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Merge transactions across every upload of one bank for the user.

    Dedupe key: SHA1(source_key + date + value + normalized description).
    Overrides are stored under one key per source_key (not per upload), so
    classification persists across re-uploads of the same statement.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    if source_key not in _BANK_SOURCES:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_source", "source_key": source_key,
                    "supported": sorted(_BANK_SOURCES)},
        )
    _bind_user(user)

    from v2.legacy.bank_tx import parse_bank_tx_bytes, looks_like_pdf, CATEGORY_OPTIONS, extract_pdf_summary
    from v2.legacy.db_storage import db_load

    files = await uploads_storage.fetch_files_by_source(pool, user.id, source_key)

    rows_by_hash: dict[str, dict[str, Any]] = {}
    upload_ids: list[int] = []
    format_errors: list[dict] = []
    duplicates_removed = 0
    # Per-file PDF summaries — currently only C6 USD has one. Aggregated below
    # to drive the reconciliation banner in the Classification page.
    pdf_summaries: list[dict[str, Any]] = []

    for f in files:
        upload_ids.append(f.id)
        is_pdf = looks_like_pdf(f.file_bytes)
        if is_pdf:
            s = extract_pdf_summary(source_key, f.file_bytes)
            if s is not None:
                s["upload_id"] = f.id
                s["filename"] = f.filename
                pdf_summaries.append(s)
        parsed = parse_bank_tx_bytes(source_key, f.file_bytes)
        if not parsed:
            format_errors.append({
                "upload_id": f.id, "filename": f.filename,
                "error": "pdf_not_supported" if is_pdf else "empty_after_parse",
            })
            continue
        for r in parsed:
            h = r.get("tx_hash")
            if not h:
                continue
            if h in rows_by_hash:
                duplicates_removed += 1
                # Prefer earliest source upload's date in case of differing formatting
                continue
            r2 = dict(r)
            r2["source_upload_id"] = f.id
            r2["source_filename"] = f.filename
            rows_by_hash[h] = r2

    rows = list(rows_by_hash.values())

    # Apply per-source overrides keyed by tx_hash
    overrides = db_load(_grouped_overrides_key(source_key)) or {}
    if not isinstance(overrides, dict):
        overrides = {}
    for r in rows:
        ov = overrides.get(r["tx_hash"])
        if isinstance(ov, dict):
            if ov.get("category"):
                r["category"] = ov["category"]
                r["confidence"] = "manual"
                r["auto"] = False
            if ov.get("project") is not None:
                r["project"] = ov["project"]
            if ov.get("label"):
                r["label"] = ov["label"]

    # Câmbio enrichment — currently only relevant for C6 USD: pull paired
    # BRL→USD conversions from C6 BRL, build a FIFO inventory of USD lots,
    # consume saídas oldest-first to compute the *real* BRL cost of each
    # USD outflow. Mutates `rows` in place (adds cambio_rate / fifo_brl_cost).
    cambio_summary: Optional[dict[str, Any]] = None
    if source_key == "extrato_c6_usd":
        try:
            from v2.services import cambio as cambio_svc
            cambio_result = await cambio_svc.compute_for_user(pool, user.id, rows)
            cambio_svc.enrich_rows(rows, cambio_result)
            cambio_summary = cambio_result.summary
        except Exception as err:  # noqa: BLE001
            # Cambio is enrichment, not critical — log but never block the page.
            import logging as _log_mod
            _log_mod.getLogger("finance.cambio").warning("cambio compute failed: %s", err)

    # Sort newest first by date string (ISO-friendly when present, falls back
    # to lexicographic — file order is preserved for ties via Python sort stability)
    rows.sort(key=lambda r: (r.get("date") or ""), reverse=True)

    counts = {
        "external": sum(1 for r in rows if r.get("tx_class") == "external"),
        "internal_ml": sum(1 for r in rows if r.get("tx_class") == "internal_ml"),
        "unknown": sum(1 for r in rows if r.get("tx_class") == "unknown"),
    }

    projects = sorted((legacy_config.load_projects() or {}).keys())

    # Aggregate PDF summaries across all uploads of this bank (one bank → many
    # statement files). UI shows a single banner per bank, so we sum expected
    # vs actual and AND the reconciliation flags. Currency is taken from the
    # first non-null entry — every file for the same source_key carries the
    # same currency by construction.
    summary: Optional[dict[str, Any]] = None
    if pdf_summaries:
        def _opt_sum(key: str) -> Optional[float]:
            vals = [s.get(key) for s in pdf_summaries if s.get(key) is not None]
            return round(sum(vals), 2) if vals else None

        def _opt_min(key: str) -> Optional[str]:
            vals = [s.get(key) for s in pdf_summaries if s.get(key)]
            return min(vals) if vals else None

        def _opt_max(key: str) -> Optional[str]:
            vals = [s.get(key) for s in pdf_summaries if s.get(key)]
            return max(vals) if vals else None

        summary = {
            "currency": pdf_summaries[0].get("currency"),
            "expected_entradas": _opt_sum("expected_entradas_usd"),
            "expected_saidas": _opt_sum("expected_saidas_usd"),
            "actual_entradas": round(sum(s.get("actual_entradas_usd") or 0 for s in pdf_summaries), 2),
            "actual_saidas": round(sum(s.get("actual_saidas_usd") or 0 for s in pdf_summaries), 2),
            # Saldo: take from the file with the latest period_to (most recent
            # statement). Falls back to the last entry's saldo if dates missing.
            "saldo_final": (
                max(pdf_summaries, key=lambda s: s.get("period_to") or "").get("saldo_final_usd")
            ),
            "saldo_final_date": (
                max(pdf_summaries, key=lambda s: s.get("period_to") or "").get("saldo_final_date")
            ),
            "period_from": _opt_min("period_from"),
            "period_to": _opt_max("period_to"),
            "reconciliation_ok": all(s.get("reconciliation_ok") for s in pdf_summaries),
            "missing_count_estimate": sum(s.get("missing_count_estimate") or 0 for s in pdf_summaries),
            "files": [
                {"upload_id": s["upload_id"], "filename": s["filename"], "ok": s.get("reconciliation_ok", False)}
                for s in pdf_summaries
            ],
        }

    return {
        "source_key": source_key,
        "bank_label": _BANK_LABELS.get(source_key, source_key),
        "upload_ids": upload_ids,
        "rows": rows,
        "categories": CATEGORY_OPTIONS,
        "projects": projects,
        "saved_overrides_count": len(overrides),
        "format_errors": format_errors,
        "duplicates_removed": duplicates_removed,
        "counts": counts,
        "summary": summary,
        "cambio": cambio_summary,
    }


@router.post("/bank-transactions/grouped/save")
async def save_bank_transactions_grouped(
    body: dict[str, Any],
    source_key: str = Query(...),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Persist tx_hash-keyed overrides for a whole bank.

    Body: `{ overrides: [{tx_hash, category?, project?, label?}, ...] }`.
    Empty (null/empty) values for all three clear the entry.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    if source_key not in _BANK_SOURCES:
        raise HTTPException(status_code=400, detail="unsupported_source")
    _bind_user(user)

    from v2.legacy.db_storage import db_load, db_save

    current = db_load(_grouped_overrides_key(source_key)) or {}
    if not isinstance(current, dict):
        current = {}

    incoming = (body or {}).get("overrides") or []
    saved = 0
    for ov in incoming:
        if not isinstance(ov, dict):
            continue
        h = ov.get("tx_hash")
        if not h or not isinstance(h, str):
            continue
        cat = ov.get("category")
        proj = ov.get("project")
        lbl = ov.get("label")
        has_content = any(v for v in (cat, proj, lbl))
        if not has_content:
            if h in current:
                current.pop(h, None)
                saved += 1
            continue
        entry = current.get(h, {}) if isinstance(current.get(h), dict) else {}
        if cat is not None:
            entry["category"] = cat
        if proj is not None:
            entry["project"] = proj
        if lbl is not None:
            entry["label"] = lbl
        current[h] = entry
        saved += 1

    db_save(_grouped_overrides_key(source_key), current)
    return {"saved": saved, "total_overrides": len(current)}


# ── Upload source-key diagnostic + re-detect ────────────────────────────────
# Filename auto-detect rules evolved over time (e.g. fix(uploads): handle
# spaces in ML export names); old uploads stayed in the DB with their
# original — sometimes wrong — source_key, so loaders silently skipped them.
# These endpoints let the user see the mismatch and fix it without
# re-uploading the file.


@router.get("/uploads/source-debug")
async def uploads_source_debug(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """List every uploaded file with its stored source_key and what the
    current detector would assign now. Mismatches surface as `would_be`
    being different from `current`.

    Returns:
      {
        "uploads": [{id, filename, current, would_be, mismatch}, ...],
        "summary_by_source": {<source_key>: count, ...},
        "mismatches": <int>,
      }
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    from v2.legacy.source_detection import detect_source_from_filename

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, source_key, content_sha256,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
              FROM uploads
             WHERE user_id = $1 AND file_bytes IS NOT NULL
             ORDER BY created_at DESC
            """,
            user.id,
        )

    out: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    mismatches = 0
    for r in rows:
        current = r["source_key"] or ""
        detected = detect_source_from_filename(r["filename"]) or ""
        is_mismatch = bool(detected and detected != current)
        if is_mismatch:
            mismatches += 1
        summary[current or "(empty)"] = summary.get(current or "(empty)", 0) + 1
        out.append({
            "id": r["id"],
            "filename": r["filename"],
            "current": current,
            "would_be": detected,
            "mismatch": is_mismatch,
            "created_at": r["created_at"],
        })

    return {
        "uploads": out,
        "summary_by_source": summary,
        "total_uploads": len(out),
        "mismatches": mismatches,
    }


@router.post("/uploads/source-fix")
async def uploads_source_fix(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    """Re-run filename auto-detect for every upload owned by the user and
    update `source_key` where the new detector returns a different value.

    Body (optional):
      ids: list[int]  — only fix these specific uploads (default: all)
      dry_run: bool   — return planned changes without writing (default false)
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    from v2.legacy.source_detection import detect_source_from_filename

    only_ids = (body or {}).get("ids")
    dry_run = bool((body or {}).get("dry_run"))

    async with pool.acquire() as conn:
        if only_ids and isinstance(only_ids, list):
            rows = await conn.fetch(
                """
                SELECT id, filename, source_key
                  FROM uploads
                 WHERE user_id = $1 AND id = ANY($2::int[])
                """,
                user.id, [int(x) for x in only_ids if isinstance(x, (int, str)) and str(x).isdigit()],
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, filename, source_key
                  FROM uploads
                 WHERE user_id = $1 AND file_bytes IS NOT NULL
                """,
                user.id,
            )

        plan: list[dict[str, Any]] = []
        for r in rows:
            old = r["source_key"] or ""
            new = detect_source_from_filename(r["filename"]) or ""
            if not new or new == old:
                continue
            plan.append({
                "id": r["id"],
                "filename": r["filename"],
                "old": old,
                "new": new,
            })

        if not dry_run and plan:
            async with conn.transaction():
                for entry in plan:
                    await conn.execute(
                        "UPDATE uploads SET source_key = $1 WHERE id = $2 AND user_id = $3",
                        entry["new"], entry["id"], user.id,
                    )

    return {
        "dry_run": dry_run,
        "scanned": len(rows),
        "fixed_count": 0 if dry_run else len(plan),
        "would_fix_count": len(plan) if dry_run else 0,
        "changes": plan,
    }


@router.get("/uploads/diag-retirada")
async def uploads_diag_retirada(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Per-file breakdown of retirada_full uploads — confirms which месяцы
    действительно дают envio/descarte и какие файлы парсер тихо отверг.

    Используется когда в PnL-строке "Списание товара" значения только в
    одном месяце, а в БД 7 файлов retirada — этот endpoint показывает
    rows_count + sum_envio + sum_descarte для каждого upload'а.
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    from v2.legacy.reports import _parse_retirada_estoque_bytes

    files = await uploads_storage.fetch_files_by_source(pool, user.id, "retirada_full")

    out: list[dict[str, Any]] = []
    total_envio = 0.0
    total_descarte_tarifa = 0.0
    total_descarte_units = 0
    for sf in files:
        entry: dict[str, Any] = {
            "upload_id": sf.id,
            "filename": sf.filename,
            "bytes": len(sf.file_bytes or b""),
            "parsed": False,
            "rows_count": 0,
            "envio_rows": 0,
            "descarte_rows": 0,
            "sum_envio_brl": 0.0,
            "sum_descarte_tarifa_brl": 0.0,
            "descarte_units": 0,
            "dates_min": None,
            "dates_max": None,
            "skus_unique": 0,
            "error": None,
        }
        try:
            parsed = _parse_retirada_estoque_bytes(sf.file_bytes or b"", sf.filename)
        except Exception as err:  # noqa: BLE001
            entry["error"] = f"{type(err).__name__}: {err}"
            out.append(entry)
            continue
        if not parsed:
            entry["error"] = "parser_returned_none (sheet name mismatch or required cols missing)"
            out.append(entry)
            continue

        rows = parsed.get("rows") or []
        entry["parsed"] = True
        entry["rows_count"] = len(rows)
        skus: set[str] = set()
        dates: list[str] = []
        for r in rows:
            forma = (r.get("forma") or "").strip()
            valor = float(r.get("valor") or 0.0)
            units = int(r.get("units") or 0)
            if r.get("sku"):
                skus.add(str(r["sku"]))
            if r.get("date"):
                dates.append(str(r["date"]))
            if forma.startswith("Envio"):
                entry["envio_rows"] += 1
                entry["sum_envio_brl"] += valor
                total_envio += valor
            elif forma.startswith("Descarte"):
                entry["descarte_rows"] += 1
                entry["sum_descarte_tarifa_brl"] += valor
                entry["descarte_units"] += units
                total_descarte_tarifa += valor
                total_descarte_units += units
        entry["sum_envio_brl"] = round(entry["sum_envio_brl"], 2)
        entry["sum_descarte_tarifa_brl"] = round(entry["sum_descarte_tarifa_brl"], 2)
        entry["skus_unique"] = len(skus)
        if dates:
            dates.sort()
            entry["dates_min"] = dates[0]
            entry["dates_max"] = dates[-1]
        out.append(entry)

    return {
        "files": out,
        "totals": {
            "files_count": len(files),
            "files_parsed": sum(1 for e in out if e["parsed"]),
            "envio_brl": round(total_envio, 2),
            "descarte_tarifa_brl": round(total_descarte_tarifa, 2),
            "descarte_units": total_descarte_units,
        },
    }


@router.get("/uploads/{upload_id}/inspect")
async def upload_inspect(
    upload_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Show структуру xlsx/csv: sheet_names + первые 12 строк каждого листа.
    Используется когда парсер возвращает None — увидеть как ML обновил
    формат (sheet renamed, header сдвинулся, новая колонка)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    sf = await uploads_storage.get_file(pool, user.id, upload_id)
    if not sf:
        raise HTTPException(status_code=404, detail="upload_not_found")

    file_bytes = sf.file_bytes or b""
    out: dict[str, Any] = {
        "upload_id": sf.id,
        "filename": sf.filename,
        "source_key": sf.source_key,
        "bytes": len(file_bytes),
        "magic": file_bytes[:4].hex() if file_bytes else "",
    }

    if len(file_bytes) >= 4 and file_bytes[:4] == b"PK\x03\x04":
        import io as _io
        import pandas as pd
        try:
            xl = pd.ExcelFile(_io.BytesIO(file_bytes))
        except Exception as err:  # noqa: BLE001
            out["error"] = f"ExcelFile open failed: {type(err).__name__}: {err}"
            return out

        out["format"] = "xlsx"
        out["sheet_names"] = list(xl.sheet_names)
        sheets_preview: dict[str, Any] = {}
        for sname in xl.sheet_names[:5]:
            try:
                df = pd.read_excel(xl, sheet_name=sname, header=None, nrows=12, dtype=object)
                rows: list[list[str]] = []
                for _, r in df.iterrows():
                    row = []
                    for v in r.tolist():
                        if pd.isna(v):
                            row.append("")
                        else:
                            s = str(v)
                            row.append(s[:60] + "…" if len(s) > 60 else s)
                    rows.append(row)
                sheets_preview[sname] = {
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "rows": rows,
                }
            except Exception as err:  # noqa: BLE001
                sheets_preview[sname] = {"error": f"{type(err).__name__}: {err}"}
        out["sheets_preview"] = sheets_preview
        return out

    out["format"] = "csv"
    text: str | None = None
    enc_used = ""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            enc_used = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        out["error"] = "decode_failed"
        return out
    out["encoding"] = enc_used
    import csv as _csv
    import io as _io2
    try:
        rows_csv = list(_csv.reader(_io2.StringIO(text, newline=""), delimiter=";"))
    except Exception as err:  # noqa: BLE001
        out["error"] = f"csv_parse: {type(err).__name__}: {err}"
        return out
    preview: list[list[str]] = []
    for r in rows_csv[:12]:
        preview.append([(c[:60] + "…" if len(c) > 60 else c) for c in r])
    out["rows_total"] = len(rows_csv)
    out["rows_preview"] = preview
    return out


@router.get("/uploads/diag-mappings")
async def uploads_diag_mappings(
    project: str = Query(..., description="Project ID, e.g. ARTUR"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Покажи как armazenagem и retirada SKU маппятся к проектам. Используется
    когда в PnL `Armazenagem` / `Вывоз / Списание` показывают 0 при том что
    данные парсятся (см. diag-armazenagem / diag-retirada). Корень — SKU
    отсутствует в catalog (cost+project mapping)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)

    from v2.legacy.reports import load_armazenagem_report, load_retirada_estoque_report

    out: dict[str, Any] = {"project": project, "armazenagem": {}, "retirada": {}}

    try:
        df_arm = load_armazenagem_report()
    except Exception as err:  # noqa: BLE001
        out["armazenagem"]["error"] = f"{type(err).__name__}: {err}"
        df_arm = None
    if df_arm is not None and not df_arm.empty:
        total_skus = int(df_arm["SKU"].nunique())
        by_proj: dict[str, int] = {}
        for proj in df_arm["__project"].fillna("(empty)"):
            key = str(proj) or "(empty)"
            by_proj[key] = by_proj.get(key, 0) + 1
        target = df_arm[df_arm["__project"] == project]
        unassigned = df_arm[(df_arm["__project"].isna()) | (df_arm["__project"] == "")]
        out["armazenagem"] = {
            "total_skus": total_skus,
            "by_project_row_count": by_proj,
            "skus_for_project": int(target["SKU"].nunique()) if not target.empty else 0,
            "sample_skus_for_project": target["SKU"].head(20).tolist() if not target.empty else [],
            "unassigned_count": int(unassigned["SKU"].nunique()) if not unassigned.empty else 0,
            "sample_unassigned_skus": (
                unassigned[["SKU", "MLB", "anuncio"]].head(15).to_dict(orient="records")
                if not unassigned.empty else []
            ),
            "source_files": list(df_arm.attrs.get("__source_files", [])),
            "daily_cols_total": len(df_arm.attrs.get("__daily_cols", [])),
        }

    try:
        df_ret = load_retirada_estoque_report()
    except Exception as err:  # noqa: BLE001
        out["retirada"]["error"] = f"{type(err).__name__}: {err}"
        df_ret = None
    if df_ret is not None and not df_ret.empty:
        total_rows = int(len(df_ret))
        by_proj_ret: dict[str, int] = {}
        for proj in df_ret["__project"].fillna("(empty)"):
            key = str(proj) or "(empty)"
            by_proj_ret[key] = by_proj_ret.get(key, 0) + 1
        target = df_ret[df_ret["__project"] == project]
        unassigned = df_ret[(df_ret["__project"].isna()) | (df_ret["__project"] == "")]
        out["retirada"] = {
            "total_rows": total_rows,
            "by_project_row_count": by_proj_ret,
            "rows_for_project": int(len(target)),
            "tarifa_for_project": float(target["valor"].sum()) if not target.empty else 0.0,
            "unassigned_rows": int(len(unassigned)),
            "unassigned_tarifa": float(unassigned["valor"].sum()) if not unassigned.empty else 0.0,
            "sample_unassigned": (
                unassigned[["sku", "mlb", "anuncio", "forma", "valor", "units", "date"]]
                .head(10).astype(str).to_dict(orient="records")
                if not unassigned.empty else []
            ),
            "source_files": list(df_ret.attrs.get("__source_files", [])),
        }

    return out


@router.get("/uploads/diag-armazenagem-period")
async def uploads_diag_armazenagem_period(
    project: str = Query(...),
    period_from: str = Query(..., alias="from"),
    period_to: str = Query(..., alias="to"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """For a given (project, period) показывает per-day total, top SKU,
    coverage. Используется для апрельского прочерка — увидеть какие даты
    в df_armazenagem попадают в период и сколько каждый SKU/день даёт."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)

    from datetime import date as _date, datetime as _dt
    from v2.legacy.reports import load_armazenagem_report, get_armazenagem_by_period
    import pandas as pd

    try:
        pf = _date.fromisoformat(period_from)
        pt = _date.fromisoformat(period_to)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad_date")

    df = load_armazenagem_report()
    out: dict[str, Any] = {
        "project": project,
        "period": {"from": period_from, "to": period_to},
        "df_loaded": df is not None,
    }
    if df is None or df.empty:
        return out

    out["df_total_skus"] = int(df["SKU"].nunique())
    out["source_files"] = list(df.attrs.get("__source_files", []))

    daily_cols = list(df.attrs.get("__daily_cols", []))
    out["daily_cols_total"] = len(daily_cols)
    if daily_cols:
        out["daily_cols_first"] = daily_cols[0]
        out["daily_cols_last"] = daily_cols[-1]

    relevant_cols: list[str] = []
    for c in daily_cols:
        try:
            d = _dt.strptime(c, "%d/%m/%Y").date()
        except ValueError:
            continue
        if pf <= d <= pt:
            relevant_cols.append(c)
    out["relevant_cols_count"] = len(relevant_cols)
    if relevant_cols:
        out["relevant_cols_first"] = relevant_cols[0]
        out["relevant_cols_last"] = relevant_cols[-1]

    sub = df[df["__project"] == project]
    out["skus_for_project"] = int(sub["SKU"].nunique()) if not sub.empty else 0

    # Per-day totals для проекта
    daily_totals: dict[str, float] = {}
    for c in relevant_cols:
        if c in sub.columns:
            v = float(pd.to_numeric(sub[c], errors="coerce").fillna(0).sum())
            if v > 0:
                daily_totals[c] = round(v, 4)
    out["daily_totals"] = daily_totals
    out["daily_totals_sum"] = round(sum(daily_totals.values()), 4)

    # Top 5 SKU по сумме за период
    if not sub.empty and relevant_cols:
        sku_sums: list[tuple[str, float]] = []
        for _, row in sub.iterrows():
            sku = str(row["SKU"])
            total_for_sku = 0.0
            for c in relevant_cols:
                if c in row.index:
                    v = pd.to_numeric(row[c], errors="coerce")
                    if not pd.isna(v):
                        total_for_sku += float(v)
            if total_for_sku > 0:
                sku_sums.append((sku, round(total_for_sku, 4)))
        sku_sums.sort(key=lambda t: t[1], reverse=True)
        out["top_skus_in_period"] = [
            {"sku": s, "total": v} for s, v in sku_sums[:10]
        ]

    # Сравнение с officical aggregator
    try:
        rc = get_armazenagem_by_period(project, pf, pt)
        out["get_armazenagem_by_period"] = {
            "total": rc.get("total"),
            "days_in_period": rc.get("days_in_period"),
            "skus_count": rc.get("skus_count"),
        }
    except Exception as err:  # noqa: BLE001
        out["get_armazenagem_by_period_error"] = f"{type(err).__name__}: {err}"

    return out


@router.get("/uploads/diag-retirada-period")
async def uploads_diag_retirada_period(
    project: str = Query(...),
    period_from: str = Query(..., alias="from"),
    period_to: str = Query(..., alias="to"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """For a given (project, period) показывает RAW retirada-rows + repr(forma)
    (для Unicode normalization debug) + результат compute_retirada_cost.
    Сравнение с pnl-matrix позволяет точечно локализовать баг."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)

    from datetime import date as _date
    import unicodedata
    from v2.legacy.reports import load_retirada_estoque_report
    from v2.legacy.finance import compute_retirada_cost

    try:
        pf = _date.fromisoformat(period_from)
        pt = _date.fromisoformat(period_to)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad_date")

    df = load_retirada_estoque_report()
    out: dict[str, Any] = {
        "project": project,
        "period": {"from": period_from, "to": period_to},
        "df_loaded": df is not None,
        "df_total_rows": int(len(df)) if df is not None else 0,
    }
    if df is not None and not df.empty:
        sub = df[(df["__project"] == project) &
                 (df["date"] >= pf) &
                 (df["date"] <= pt)]
        out["rows_for_project_in_period"] = int(len(sub))

        by_forma: dict[str, Any] = {}
        rows_dump: list[dict[str, Any]] = []
        for _, row in sub.iterrows():
            forma_raw = row["forma"] or ""
            f_str = str(forma_raw)
            f_nfc = unicodedata.normalize("NFC", f_str)
            f_nfd = unicodedata.normalize("NFD", f_str)
            key = f_nfc
            b = by_forma.setdefault(key, {
                "count": 0, "tarifa": 0.0, "units": 0,
                "forma_raw": f_str,
                "forma_repr": repr(f_str),
                "is_nfc": f_str == f_nfc,
                "is_nfd": f_str == f_nfd,
                "byte_len": len(f_str.encode("utf-8")),
                "char_count": len(f_str),
                "matches_envio_const": f_str == "Envio para o endereço",
                "matches_descarte_const": f_str == "Descarte",
            })
            b["count"] += 1
            b["tarifa"] += float(row["valor"])
            b["units"] += int(row["units"])
            if len(rows_dump) < 5:
                rows_dump.append({
                    "date": str(row["date"]),
                    "sku": str(row["sku"]),
                    "forma_repr": repr(f_str),
                    "valor": float(row["valor"]),
                    "units": int(row["units"]),
                    "project": str(row["__project"]),
                })
        out["by_forma"] = by_forma
        out["sample_rows"] = rows_dump

    try:
        rc = compute_retirada_cost(project, (pf, pt))
        out["compute_retirada_cost"] = {
            "tarifa_envio": rc.get("tarifa_envio"),
            "tarifa_descarte": rc.get("tarifa_descarte"),
            "cogs_descarte": rc.get("cogs_descarte"),
            "tarifa_other": rc.get("tarifa_other"),
            "units_envio": rc.get("units_envio"),
            "units_descarte": rc.get("units_descarte"),
            "units_other": rc.get("units_other"),
            "rows_count": rc.get("rows_count"),
            "missing_cost_skus_count": len(rc.get("missing_cost_skus") or []),
            "missing_cost_skus": rc.get("missing_cost_skus") or [],
            "fallback_avg_used": rc.get("fallback_avg_used"),
        }
    except Exception as err:  # noqa: BLE001
        out["compute_retirada_cost_error"] = f"{type(err).__name__}: {err}"

    return out


@router.get("/uploads/diag-armazenagem")
async def uploads_diag_armazenagem(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Per-file breakdown of armazenagem_full uploads — показывает range дат
    и SKU count в каждом файле. Используется чтобы увидеть какие файлы
    парсер принимает (и до какой даты они дают coverage).
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    from v2.legacy.reports import _parse_armazenagem_bytes_daily

    files = await uploads_storage.fetch_files_by_source(pool, user.id, "armazenagem_full")

    out: list[dict[str, Any]] = []
    for sf in files:
        ext = "xlsx" if (sf.file_bytes or b"")[:4] == b"PK\x03\x04" else "csv"
        entry: dict[str, Any] = {
            "upload_id": sf.id,
            "filename": sf.filename,
            "ext": ext,
            "bytes": len(sf.file_bytes or b""),
            "parsed": False,
            "skus_count": 0,
            "daily_cols_count": 0,
            "first_date": None,
            "last_date": None,
            "error": None,
        }
        try:
            parsed = _parse_armazenagem_bytes_daily(sf.file_bytes or b"")
        except Exception as err:  # noqa: BLE001
            entry["error"] = f"{type(err).__name__}: {err}"
            out.append(entry)
            continue
        if not parsed:
            entry["error"] = "parser_returned_none (binary blob, missing SKU header, or empty rows)"
            out.append(entry)
            continue
        daily_cols = parsed.get("daily_cols") or []
        entry["parsed"] = True
        entry["skus_count"] = len(parsed.get("by_sku") or {})
        entry["daily_cols_count"] = len(daily_cols)
        if daily_cols:
            sorted_cols = sorted(daily_cols, key=lambda d: tuple(reversed(d.split("/"))))
            entry["first_date"] = sorted_cols[0]
            entry["last_date"] = sorted_cols[-1]
        out.append(entry)

    return {
        "files": out,
        "totals": {
            "files_count": len(files),
            "files_parsed": sum(1 for e in out if e["parsed"]),
        },
    }


# ── Bank balance anchors (manual + reconciliation) ──────────────────────────


@router.get("/bank-balances")
async def get_bank_balance(
    source_key: str = Query(..., description="extrato_nubank | extrato_mp | extrato_c6_brl | extrato_c6_usd"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Active balance anchor for one bank + reconciliation against parsed
    statements. Returns:
      anchor: {balance, currency, balance_date, notes, ...} | null
      reconciliation: {expected_balance, txn_sum_after, txn_count_after, ...} | null
      history: [...]
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    if source_key not in _BANK_SOURCES:
        raise HTTPException(status_code=400, detail="unsupported_source")
    _bind_user(user)
    await bank_balances_svc.ensure_schema(pool)

    anchor = await bank_balances_svc.get_active(pool, user.id, source_key)
    history = await bank_balances_svc.list_history(pool, user.id, source_key, limit=10)

    reconciliation: Optional[dict[str, Any]] = None
    if anchor:
        # Re-parse all uploads for this bank and reconcile against the anchor
        from v2.legacy.bank_tx import parse_bank_tx_bytes

        files = await uploads_storage.fetch_files_by_source(pool, user.id, source_key)
        merged: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for f in files:
            for r in parse_bank_tx_bytes(source_key, f.file_bytes):
                h = r.get("tx_hash") or ""
                if h and h in seen_hashes:
                    continue
                if h:
                    seen_hashes.add(h)
                merged.append(r)

        try:
            recorded_date = date.fromisoformat(anchor["balance_date"])
        except (TypeError, ValueError):
            recorded_date = None
        if recorded_date:
            reconciliation = bank_balances_svc.reconcile(
                recorded_balance=anchor["balance"],
                recorded_date=recorded_date,
                rows=merged,
            )

    return {
        "source_key": source_key,
        "currency": bank_balances_svc.CURRENCY_BY_BANK.get(source_key, "BRL"),
        "anchor": anchor,
        "reconciliation": reconciliation,
        "history": history,
    }


@router.post("/bank-balances")
async def post_bank_balance(
    body: dict[str, Any],
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Set a new anchor balance for a bank.

    Body: { source_key, balance: number, balance_date: 'YYYY-MM-DD', notes?: str }
    The previous active anchor is superseded (history kept).
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    _bind_user(user)
    await bank_balances_svc.ensure_schema(pool)

    source_key = (body or {}).get("source_key")
    if source_key not in _BANK_SOURCES:
        return {"error": "unsupported_source", "source_key": source_key}

    raw_balance = (body or {}).get("balance")
    try:
        balance = float(raw_balance)
    except (TypeError, ValueError):
        return {"error": "balance_required_number"}

    raw_date = (body or {}).get("balance_date")
    try:
        balance_date = date.fromisoformat(str(raw_date))
    except (TypeError, ValueError):
        return {"error": "balance_date_required_iso"}

    return await bank_balances_svc.upsert_balance(
        pool, user.id,
        source_key=source_key,
        balance=balance,
        balance_date=balance_date,
        currency=(body or {}).get("currency"),
        notes=(body or {}).get("notes"),
    )


@router.delete("/bank-balances")
async def delete_bank_balance(
    source_key: str = Query(...),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Clear the active anchor — leaves history intact."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    if source_key not in _BANK_SOURCES:
        raise HTTPException(status_code=400, detail="unsupported_source")
    _bind_user(user)
    return await bank_balances_svc.delete_balance(pool, user.id, source_key)


# ── Manual external USD inflows (Phase 3 câmbio) ─────────────────────────────
# CRUD over the user's hand-entered BRL→USD conversion records (Bybit USDT,
# CALIZA-Nubank direct, custom transfers — anything that doesn't go through
# C6). Read by `v2/services/cambio.py:compute_for_user` and merged with
# C6-paired entradas into a single FIFO inventory.

# Same band as cambio.pair_brl_usd — out-of-range entries are usually a
# unit mistake (R$ vs US$ swapped, missing decimal, etc.) and would distort
# the FIFO costs. Reject loudly so the user notices.
_MANUAL_INFLOW_RATE_MIN = 3.5
_MANUAL_INFLOW_RATE_MAX = 8.0


def _validate_inflow_payload(body: ManualInflowIn) -> tuple[date, float, float, str, str]:
    """Parse + range-check a POST/PUT body. Raises HTTPException on invalid input.

    Pulled out into a helper because the same checks apply to create and update,
    and the error shapes need to be consistent so the UI can surface field-level
    feedback.
    """
    try:
        d = _parse_iso(body.date)
    except Exception:  # noqa: BLE001
        d = None
    if d is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_date", "field": "date", "got": body.date},
        )
    # gt=0 already enforced at Pydantic level; rate range is the cross-field check.
    rate = body.brl_paid / body.usd_received
    if rate < _MANUAL_INFLOW_RATE_MIN or rate > _MANUAL_INFLOW_RATE_MAX:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "rate_out_of_range",
                "rate": round(rate, 4),
                "expected_range": [_MANUAL_INFLOW_RATE_MIN, _MANUAL_INFLOW_RATE_MAX],
                "hint": "проверьте суммы — возможно перепутаны R$ и US$",
            },
        )
    source = (body.source or "Manual").strip() or "Manual"
    note = (body.note or "").strip()
    return d, body.usd_received, body.brl_paid, source, note


def _inflow_to_out(row: manual_inflows_svc.ManualInflow) -> dict[str, Any]:
    """Dataclass → dict shaped for ManualInflowOut. Computes `rate` here so
    the UI doesn't have to re-divide every render."""
    return {
        "id": row.id,
        "date": row.date.isoformat(),
        "usd_received": row.usd_received,
        "brl_paid": row.brl_paid,
        "rate": round(row.brl_paid / row.usd_received, 4),
        "source": row.source,
        "note": row.note,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }


@router.get("/manual-usd-inflows", response_model=ManualInflowsListOut)
async def list_manual_inflows(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """List every manual inflow + roll-up totals + dropdown options."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    rows = await manual_inflows_svc.list_for_user(pool, user.id)
    items = [_inflow_to_out(r) for r in rows]
    total_usd = round(sum(r.usd_received for r in rows), 2)
    total_brl = round(sum(r.brl_paid for r in rows), 2)
    avg_rate = round(total_brl / total_usd, 4) if total_usd > 0 else None
    return {
        "items": items,
        "source_options": list(manual_inflows_svc.SOURCE_OPTIONS),
        "total_usd": total_usd,
        "total_brl": total_brl,
        "avg_rate": avg_rate,
    }


@router.post("/manual-usd-inflows", response_model=ManualInflowOut)
async def create_manual_inflow(
    body: ManualInflowIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Insert a new inflow. Validation: date parseable, amounts > 0, rate in
    sane range. The Câmbio FIFO will pick it up on next /bank-transactions/grouped."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    d, usd, brl, source, note = _validate_inflow_payload(body)
    row = await manual_inflows_svc.create(
        pool,
        user_id=user.id,
        date=d,
        usd_received=usd,
        brl_paid=brl,
        source=source,
        note=note,
    )
    return _inflow_to_out(row)


@router.put("/manual-usd-inflows/{inflow_id}", response_model=ManualInflowOut)
async def update_manual_inflow(
    inflow_id: int,
    body: ManualInflowIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Replace all fields of an existing inflow owned by the caller."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    d, usd, brl, source, note = _validate_inflow_payload(body)
    row = await manual_inflows_svc.update(
        pool,
        inflow_id=inflow_id,
        user_id=user.id,
        date=d,
        usd_received=usd,
        brl_paid=brl,
        source=source,
        note=note,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="inflow_not_found")
    return _inflow_to_out(row)


@router.delete("/manual-usd-inflows/{inflow_id}")
async def delete_manual_inflow(
    inflow_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Delete a manual inflow by id (caller must own it)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    ok = await manual_inflows_svc.delete(pool, inflow_id=inflow_id, user_id=user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="inflow_not_found")
    return {"deleted": True, "id": inflow_id}


# ── Onboarding Wizard (Phase 6) ─────────────────────────────────────────────
# Port of Streamlit _admin/onboarding.py. State lives in two user_data keys:
#   f2_onboarding_step  → {step, completed}
#   f2_onboarding_data  → dict of form fields accumulated across 10 steps

_ONBOARDING_STEP_KEY = "f2_onboarding_step"
_ONBOARDING_DATA_KEY = "f2_onboarding_data"
_TOTAL_STEPS = 10


@router.get("/onboarding/state", response_model=OnboardingState)
def get_onboarding_state(user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    """Return current wizard progress + accumulated form data."""
    _bind_user(user)
    from v2.legacy.db_storage import db_load

    step_blob = db_load(_ONBOARDING_STEP_KEY) or {}
    data_blob = db_load(_ONBOARDING_DATA_KEY) or {}
    step = 1
    completed = False
    if isinstance(step_blob, dict):
        raw_step = step_blob.get("step", 1)
        try:
            step = max(1, min(_TOTAL_STEPS, int(raw_step)))
        except (TypeError, ValueError):
            step = 1
        completed = bool(step_blob.get("completed", False))
    return {
        "step": step,
        "completed": completed,
        "data": data_blob if isinstance(data_blob, dict) else {},
    }


@router.put("/onboarding/state", response_model=OnboardingState)
def put_onboarding_state(
    body: OnboardingState,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Replace wizard progress + data. Called on each step transition / save."""
    _bind_user(user)
    from v2.legacy.db_storage import db_save
    step = max(1, min(_TOTAL_STEPS, int(body.step or 1)))
    db_save(_ONBOARDING_STEP_KEY, {"step": step, "completed": bool(body.completed)})
    db_save(_ONBOARDING_DATA_KEY, body.data or {})
    return {"step": step, "completed": bool(body.completed), "data": body.data or {}}


@router.post("/projects", response_model=ProjectCreateOut)
def create_project(
    body: ProjectCreateIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Create a new project — thin wrapper over legacy.config.add_project.

    Used by onboarding step 2 and the /finance/projects page. Full edit flow
    (baseline, aluguel, rental, etc.) lives on the projects page — here we
    only collect the minimum needed to unblock the wizard.
    """
    _bind_user(user)
    from v2.legacy.config import add_project, load_projects, _invalidate_projects_cache, update_project

    pid = (body.project_id or "").strip().upper()
    if not pid:
        raise HTTPException(status_code=400, detail="project_id_required")

    # Services-projects are superadmin-only — they expose hand-curated tax
    # config (commission_brackets/formula, tomador_cnpj, services_opening)
    # that has direct effect on invoice-based ОПиУ + DAS calculations.
    # Gate ANY services-shaped payload, even if the user just sets
    # project_type='services' without filling the rest.
    requested_type = (body.project_type or "ecom").strip().lower()
    has_services_payload = (
        requested_type == "services"
        or body.tomador_cnpj or body.tomador_name
        or body.commission_brackets
        or body.commission_formula
        or body.services_opening
        or body.das_apportionment
    )
    if has_services_payload and not _is_superadmin(user):
        raise HTTPException(
            status_code=403,
            detail={"code": "superadmin_required", "feature": "services_project_create"},
        )

    existing = load_projects() or {}
    is_new = pid not in existing

    add_project(
        project_id=pid,
        project_type=body.project_type or "ecom",
        description=body.description or "",
        sku_prefixes=body.sku_prefixes or [],
        compensation_mode=body.compensation_mode or "profit_share",
        profit_share_pct=body.profit_share_pct,
    )
    # Stamp tax-block fields from onboarding Step 1 (company-level) onto
    # the new project via the editable-keys whitelist. Keeps add_project()
    # signature stable while letting the wizard preset tax_regime + anexo +
    # ml_only_revenue in one step.
    extra_fields: dict[str, Any] = {}
    if body.tax_regime is not None:
        extra_fields["tax_regime"] = body.tax_regime
    if body.simples_anexo is not None:
        extra_fields["simples_anexo"] = body.simples_anexo
    if body.ml_only_revenue is not None:
        extra_fields["ml_only_revenue"] = bool(body.ml_only_revenue)
    if extra_fields:
        update_project(pid, extra_fields)
    _invalidate_projects_cache()

    total = len(load_projects() or {})
    return {"project_id": pid, "created": is_new, "total_projects": total}


@router.put("/projects/{project_id}", response_model=ProjectMutOut)
def update_project_endpoint(
    project_id: str,
    body: ProjectUpdateIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Edit an existing project. Only whitelisted fields (see `_PROJECT_EDITABLE_KEYS`
    in legacy.config) are applied. `rental_fields` merges into the `rental` subdict.

    Also invalidates per-user vendas DF cache because project → SKU resolution
    depends on `sku_prefixes` / `mlb_fallback`.
    """
    _bind_user(user)
    from v2.legacy.config import update_project, load_projects, _invalidate_projects_cache
    from v2.legacy.reports import invalidate_vendas_cache

    pid = project_id.strip().upper()
    if not pid:
        raise HTTPException(status_code=400, detail="project_id_required")

    # Services-project gate (superadmin only). Trigger if either:
    #  (a) target project is currently type='services' — ANY edit on it,
    #  (b) the new fields try to switch to 'services' or set
    #      services-only config keys (tomador / commission_brackets /
    #      commission_formula / services_opening / das_apportionment).
    fields = body.fields or {}
    services_keys = {
        "tomador_cnpj", "tomador_name",
        "commission_brackets", "commission_formula",
        "services_opening", "das_apportionment",
    }
    target_type_now = (load_projects() or {}).get(pid, {}).get("type") or ""
    target_type_after = str(fields.get("type") or target_type_now or "").lower()
    has_services_payload = (
        target_type_now == "services"
        or target_type_after == "services"
        or any(k in fields for k in services_keys)
    )
    if has_services_payload and not _is_superadmin(user):
        raise HTTPException(
            status_code=403,
            detail={"code": "superadmin_required", "feature": "services_project_edit"},
        )

    ok = update_project(pid, body.fields or {}, body.rental_fields)
    if not ok:
        return {"project_id": pid, "updated": False, "exists": False}

    _invalidate_projects_cache()
    invalidate_vendas_cache(user.id)
    return {"project_id": pid, "updated": True, "exists": True}


@router.get("/projects/{project_id}/fixed-costs")
def get_project_fixed_costs(
    project_id: str,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Per-project monthly fixed costs (6 категорий + total). Используется для
    конфигурации break-even tracker в TG sales notifications.

    Если поле ещё не задано — возвращает все нули.
    """
    _bind_user(user)
    from v2.legacy.config import load_projects, FIXED_COST_CATEGORIES

    pid = project_id.strip().upper()
    if not pid:
        raise HTTPException(status_code=400, detail="project_id_required")
    projects = load_projects() or {}
    proj = projects.get(pid)
    if proj is None:
        raise HTTPException(status_code=404, detail="project_not_found")

    saved = proj.get("fixed_costs_monthly") or {}
    if not isinstance(saved, dict):
        saved = {}
    breakdown: dict[str, float] = {}
    for cat in FIXED_COST_CATEGORIES:
        try:
            breakdown[cat] = round(max(0.0, float(saved.get(cat) or 0)), 2)
        except (TypeError, ValueError):
            breakdown[cat] = 0.0
    return {
        "project_id": pid,
        "fixed_costs_monthly": breakdown,
        "total_monthly": round(sum(breakdown.values()), 2),
    }


@router.put("/projects/{project_id}/fixed-costs")
def save_project_fixed_costs(
    project_id: str,
    body: FixedCostsSaveIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Save per-project monthly fixed costs. Replaces full breakdown (не merge)."""
    _bind_user(user)
    from v2.legacy.config import update_project, FIXED_COST_CATEGORIES, _invalidate_projects_cache

    pid = project_id.strip().upper()
    if not pid:
        raise HTTPException(status_code=400, detail="project_id_required")

    breakdown_in = body.fixed_costs_monthly.model_dump()
    ok = update_project(pid, {"fixed_costs_monthly": breakdown_in})
    if not ok:
        raise HTTPException(status_code=404, detail="project_not_found")
    _invalidate_projects_cache()
    # Re-read для возврата canonical (после sanitize в update_project).
    from v2.legacy.config import load_projects
    proj = (load_projects() or {}).get(pid) or {}
    saved = proj.get("fixed_costs_monthly") or {}
    canonical: dict[str, float] = {
        cat: float(saved.get(cat) or 0) for cat in FIXED_COST_CATEGORIES
    }
    return {
        "project_id": pid,
        "fixed_costs_monthly": canonical,
        "total_monthly": round(sum(canonical.values()), 2),
    }


@router.delete("/projects/{project_id}", response_model=ProjectMutOut)
def delete_project_endpoint(
    project_id: str,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Remove a project from user_data.projects. Raw vendas / uploads are kept."""
    _bind_user(user)
    from v2.legacy.config import delete_project, load_projects, _invalidate_projects_cache
    from v2.legacy.reports import invalidate_vendas_cache

    pid = project_id.strip().upper()
    existed = pid in (load_projects() or {})
    if not existed:
        return {"project_id": pid, "deleted": False, "exists": False}

    delete_project(pid)
    _invalidate_projects_cache()
    invalidate_vendas_cache(user.id)
    return {"project_id": pid, "deleted": True, "exists": True}


# ── Manual cashflow entries (partner / expense / supplier) ─────────────────
# Streamlit DDS tab had a form "Добавить запись вручную" — port here.
# Data lives inside projects[PROJECT][kind] lists, so creating a new entry
# triggers project-cache invalidation.

_MANUAL_CF_KINDS = (
    "partner_contributions", "manual_expenses", "manual_supplier",
    "loan_given", "loan_received",
)


@router.get("/cashflow-entries", response_model=ManualCashflowEntriesOut)
def list_cashflow_entries(
    project: str = Query(...),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Return 3 lists of manual entries for a project."""
    _bind_user(user)
    from v2.legacy.reports import list_manual_cashflow_entries
    buckets = list_manual_cashflow_entries(project)
    return {"project": project.upper(), **buckets}


@router.post("/cashflow-entries", response_model=ManualCashflowEntriesOut)
def add_cashflow_entry(
    entry: ManualCashflowEntryIn,
    project: str = Query(...),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Append a manual cashflow entry (inflow/outflow/supplier) to a project."""
    _bind_user(user)
    from v2.legacy.reports import add_manual_cashflow_entry, list_manual_cashflow_entries
    from v2.legacy.config import _invalidate_projects_cache

    if entry.kind not in _MANUAL_CF_KINDS:
        raise HTTPException(status_code=400, detail={"error": "bad_kind", "allowed": list(_MANUAL_CF_KINDS)})

    payload: dict[str, Any] = {"date": entry.date, "valor": float(entry.valor), "note": entry.note or ""}
    if entry.kind == "partner_contributions" and entry.from_ is not None:
        payload["from"] = entry.from_
    if entry.kind == "partner_contributions":
        # "Тестовая" закупка — остаётся в ДДС как inflow, но исключается из
        # invested/MOIC в compute_balance. Всегда записываем true/false чтобы
        # PATCH мог снять флаг (иначе старое значение застрянет).
        payload["test_only"] = bool(entry.test_only)
    if entry.kind == "manual_expenses" and entry.category is not None:
        payload["category"] = entry.category or "expense"
    if entry.kind == "manual_supplier" and entry.source is not None:
        payload["source"] = entry.source

    # Multi-currency: only persist non-default values to keep payloads clean
    cur = (entry.currency or "BRL").upper()
    if cur != "BRL":
        payload["currency"] = cur
        if entry.rate_brl is None or entry.rate_brl <= 0:
            raise HTTPException(
                status_code=400,
                detail={"error": "rate_required", "message": "rate_brl required when currency != BRL"},
            )
        payload["rate_brl"] = float(entry.rate_brl)

    # Inter-project loans: require counterparty_project (validated downstream)
    if entry.kind in ("loan_given", "loan_received"):
        if not entry.counterparty_project:
            raise HTTPException(
                status_code=400,
                detail={"error": "counterparty_required", "message": "loan entries need counterparty_project"},
            )
        payload["counterparty_project"] = entry.counterparty_project.upper()

    ok = add_manual_cashflow_entry(project, entry.kind, payload)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail={"error": "project_or_counterparty_not_found"},
        )
    _invalidate_projects_cache()

    buckets = list_manual_cashflow_entries(project)
    return {"project": project.upper(), **buckets}


@router.delete("/cashflow-entries", response_model=ManualCashflowEntriesOut)
def remove_cashflow_entry(
    project: str = Query(...),
    kind: str = Query(...),
    index: int = Query(..., ge=0),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Delete one manual entry by (kind, array index)."""
    _bind_user(user)
    from v2.legacy.reports import delete_manual_cashflow_entry, list_manual_cashflow_entries
    from v2.legacy.config import _invalidate_projects_cache

    if kind not in _MANUAL_CF_KINDS:
        raise HTTPException(status_code=400, detail={"error": "bad_kind"})
    ok = delete_manual_cashflow_entry(project, kind, index)
    if not ok:
        raise HTTPException(status_code=404, detail="entry_not_found")
    _invalidate_projects_cache()

    buckets = list_manual_cashflow_entries(project)
    return {"project": project.upper(), **buckets}


@router.patch("/cashflow-entries", response_model=ManualCashflowEntriesOut)
def update_cashflow_entry(
    entry: ManualCashflowEntryIn,
    project: str = Query(...),
    index: int = Query(..., ge=0, description="Array index within the kind's bucket"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Replace one manual entry at (project, kind, index).

    For loan_given/loan_received rewrites the mirror in the counterparty project
    (loan_id preserved — stable across edits, even when counterparty changes).
    Client must send the full entry body — no partial merges.
    """
    _bind_user(user)
    from v2.legacy.reports import update_manual_cashflow_entry, list_manual_cashflow_entries
    from v2.legacy.config import _invalidate_projects_cache

    if entry.kind not in _MANUAL_CF_KINDS:
        raise HTTPException(status_code=400, detail={"error": "bad_kind", "allowed": list(_MANUAL_CF_KINDS)})

    payload: dict[str, Any] = {"date": entry.date, "valor": float(entry.valor), "note": entry.note or ""}
    if entry.kind == "partner_contributions" and entry.from_ is not None:
        payload["from"] = entry.from_
    if entry.kind == "partner_contributions":
        # "Тестовая" закупка — остаётся в ДДС как inflow, но исключается из
        # invested/MOIC в compute_balance. Всегда записываем true/false чтобы
        # PATCH мог снять флаг (иначе старое значение застрянет).
        payload["test_only"] = bool(entry.test_only)
    if entry.kind == "manual_expenses" and entry.category is not None:
        payload["category"] = entry.category or "expense"
    if entry.kind == "manual_supplier" and entry.source is not None:
        payload["source"] = entry.source

    cur = (entry.currency or "BRL").upper()
    if cur != "BRL":
        payload["currency"] = cur
        if entry.rate_brl is None or entry.rate_brl <= 0:
            raise HTTPException(
                status_code=400,
                detail={"error": "rate_required", "message": "rate_brl required when currency != BRL"},
            )
        payload["rate_brl"] = float(entry.rate_brl)

    if entry.kind in ("loan_given", "loan_received"):
        if not entry.counterparty_project:
            raise HTTPException(status_code=400, detail={"error": "counterparty_required"})
        payload["counterparty_project"] = entry.counterparty_project.upper()

    ok = update_manual_cashflow_entry(project, entry.kind, index, payload)
    if not ok:
        raise HTTPException(status_code=404, detail={"error": "entry_or_counterparty_not_found"})
    _invalidate_projects_cache()

    buckets = list_manual_cashflow_entries(project)
    return {"project": project.upper(), **buckets}


# ── Publicidade invoices (ML billing cycle: anchor date + 30-day window) ─────

def _invoice_anchor(entry: dict) -> str:
    """Extract anchor date from entry. New schema uses `date`; legacy had `ate`."""
    return str(entry.get("date") or entry.get("ate") or "")


def _publicidade_list_response(project: str) -> dict[str, Any]:
    from v2.legacy.reports import list_publicidade_invoices
    from v2.legacy.config import load_projects
    items = list_publicidade_invoices(project)
    proj_meta = (load_projects() or {}).get(project.upper(), {}) or {}
    launch_date = proj_meta.get("launch_date") or None
    cycle_day = proj_meta.get("billing_cycle_day")
    try:
        cycle_day = int(cycle_day) if cycle_day is not None else None
    except (TypeError, ValueError):
        cycle_day = None
    window = proj_meta.get("publicidade_csv_window") or None
    if isinstance(window, dict) and window.get("from") and window.get("to"):
        window_out = {"from": str(window["from"])[:10], "to": str(window["to"])[:10]}
    else:
        window_out = None
    return {
        "project": project.upper(),
        "launch_date": launch_date,
        "billing_cycle_day": cycle_day,
        "publicidade_csv_window": window_out,
        "invoices": [
            {
                "index": i,
                "date": _invoice_anchor(e),
                "valor": float(e.get("valor", 0) or 0),
                "note": str(e.get("note", "") or ""),
            }
            for i, e in enumerate(items)
        ],
    }


@router.get("/publicidade/invoices", response_model=PublicidadeInvoicesListOut)
def list_publicidade(
    project: str = Query(..., description="Project ID"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """List manual Mercado Ads invoices (faturas) for a project.

    Each fatura has its own [desde, ate] period — usually a 12-12 billing cycle,
    but arbitrary periods are allowed (backend splits per-day across requested period).
    """
    _bind_user(user)
    return _publicidade_list_response(project)


@router.post("/publicidade/invoices", response_model=PublicidadeInvoicesListOut)
def add_publicidade(
    entry: PublicidadeInvoiceIn,
    project: str = Query(...),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Append one fatura entry to a project's manual_publicidade list.

    Схема:
      - `date` (YYYY-MM-DD) — явный anchor (legacy путь или ручное переопределение)
      - `month` (YYYY-MM) — месяц фатуры, день берётся из project.billing_cycle_day
    Окно [date-29, date] и daily rate = valor/30 считаются в `get_publicidade_by_period`.
    """
    _bind_user(user)
    from v2.legacy.reports import add_publicidade_invoice as _add
    from v2.legacy.config import load_projects

    date_str = (entry.date or "").strip()
    month_str = (entry.month or "").strip()

    if not date_str and month_str:
        # Выводим дату из month + project.billing_cycle_day
        proj_meta = (load_projects() or {}).get(project.upper(), {}) or {}
        cycle_day = proj_meta.get("billing_cycle_day")
        try:
            cycle_day = int(cycle_day) if cycle_day is not None else None
        except (TypeError, ValueError):
            cycle_day = None
        if not cycle_day or not (1 <= cycle_day <= 28):
            raise HTTPException(
                status_code=400,
                detail={"error": "cycle_day_not_set", "message": "project.billing_cycle_day is required when only month is provided"},
            )
        try:
            datetime.strptime(month_str, "%Y-%m")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_month_format", "message": "expected YYYY-MM"},
            )
        date_str = f"{month_str}-{cycle_day:02d}"

    if not date_str:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_date", "message": "provide `date` (YYYY-MM-DD) or `month` (YYYY-MM)"},
        )
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_date_format", "message": "expected YYYY-MM-DD"},
        )

    if entry.valor is None:
        raise HTTPException(status_code=400, detail={"error": "valor_required"})
    payload = {"date": date_str, "valor": float(entry.valor), "note": entry.note or ""}
    ok = _add(project, payload)
    if not ok:
        raise HTTPException(status_code=404, detail="project_not_found")
    return _publicidade_list_response(project)


@router.delete("/publicidade/invoices", response_model=PublicidadeInvoicesListOut)
def remove_publicidade(
    project: str = Query(...),
    index: int = Query(..., ge=0, description="Array index to remove"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Delete one fatura entry by its array index."""
    _bind_user(user)
    from v2.legacy.reports import delete_publicidade_invoice as _del

    ok = _del(project, index)
    if not ok:
        raise HTTPException(status_code=404, detail="entry_not_found")
    return _publicidade_list_response(project)


@router.patch("/publicidade/invoices", response_model=PublicidadeInvoicesListOut)
def patch_publicidade(
    entry: PublicidadeInvoiceIn,
    project: str = Query(...),
    index: int = Query(..., ge=0, description="Array index to update"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Обновить поля существующей фатуры (valor / note / date / month).
    Поля опциональны — что передано, то обновится."""
    _bind_user(user)
    from v2.legacy.reports import update_publicidade_invoice as _upd
    from v2.legacy.config import load_projects

    patch: dict[str, Any] = {}
    if entry.valor is not None:
        patch["valor"] = float(entry.valor)
    if entry.note is not None:
        patch["note"] = entry.note

    date_str = (entry.date or "").strip()
    month_str = (entry.month or "").strip()
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "bad_date_format"})
        patch["date"] = date_str
    elif month_str:
        proj_meta = (load_projects() or {}).get(project.upper(), {}) or {}
        cycle_day = proj_meta.get("billing_cycle_day")
        try:
            cycle_day = int(cycle_day) if cycle_day is not None else None
        except (TypeError, ValueError):
            cycle_day = None
        if cycle_day and 1 <= cycle_day <= 28:
            try:
                datetime.strptime(month_str, "%Y-%m")
                patch["date"] = f"{month_str}-{cycle_day:02d}"
            except ValueError:
                raise HTTPException(status_code=400, detail={"error": "bad_month_format"})

    if not patch:
        raise HTTPException(status_code=400, detail={"error": "no_fields_to_update"})

    ok = _upd(project, index, patch)
    if not ok:
        raise HTTPException(status_code=404, detail="entry_not_found")
    return _publicidade_list_response(project)


# ── Rental payments (cash-basis aluguel schedule per project) ────────────────

def _rental_payments_response(project: str, auto_generate: bool = True) -> dict[str, Any]:
    from v2.legacy.reports import list_rental_payments, _step_date_by_months
    from v2.legacy.config import load_projects
    from datetime import datetime as _dt, timedelta as _td
    payments = list_rental_payments(project, auto_generate=auto_generate) or []
    proj_meta = (load_projects() or {}).get(project.upper(), {}) or {}
    rental = proj_meta.get("rental") or {}
    rate_usd = float(rental.get("rate_usd", 0) or 0)
    period = str(rental.get("period", "month"))
    step_months = 3 if period.lower().startswith("quart") else 1

    out: list[dict[str, Any]] = []
    total_paid = 0.0
    total_pending = 0.0
    last_paid_date = None
    for i, p in enumerate(payments):
        amt_usd = float(p.get("amount_usd", 0) or 0)
        rate = p.get("rate_brl")
        rate_f = float(rate) if rate is not None else 0.0
        amt_brl = amt_usd * rate_f if rate_f > 0 else amt_usd * 5.46
        status = str(p.get("status", "pending")).lower()
        out.append({
            "index": i,
            "date": str(p.get("date", "")),
            "amount_usd": amt_usd,
            "rate_brl": rate_f if rate_f > 0 else None,
            "amount_brl": amt_brl,
            "status": status,
            "note": str(p.get("note", "") or ""),
        })
        if status == "paid":
            total_paid += amt_brl
            # Track latest paid date for paid_until calculation
            try:
                pd = _dt.strptime(str(p.get("date", "")), "%Y-%m-%d").date()
                if last_paid_date is None or pd > last_paid_date:
                    last_paid_date = pd
            except (ValueError, TypeError):
                pass
        else:
            total_pending += amt_brl

    # paid_until = дата последней оплаты + шаг period − 1 день
    paid_until_iso = None
    if last_paid_date is not None:
        next_period_start = _step_date_by_months(last_paid_date, step_months)
        paid_until_iso = (next_period_start - _td(days=1)).isoformat()

    return {
        "project": project.upper(),
        "rate_usd": rate_usd,
        "period": period,
        "payments": out,
        "total_paid_brl": total_paid,
        "total_pending_brl": total_pending,
        "last_paid_date": last_paid_date.isoformat() if last_paid_date else None,
        "paid_until": paid_until_iso,
        "launch_date": proj_meta.get("launch_date") or None,
    }


@router.get("/rental-payments", response_model=RentalPaymentsListOut)
def list_rental(
    project: str = Query(..., description="Project ID"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Список платежей аренды по проекту. При первом вызове с пустым массивом
    автосгенерирует 6 будущих pending-платежей из rental.next_payment_date."""
    _bind_user(user)
    return _rental_payments_response(project, auto_generate=True)


def _rental_payment_payload(entry: RentalPaymentIn) -> dict[str, Any]:
    """Validate + normalize payload for persistence."""
    status = (entry.status or "pending").lower()
    if status not in ("paid", "pending"):
        raise HTTPException(status_code=400, detail={"error": "bad_status", "allowed": ["paid", "pending"]})
    if entry.amount_usd is None or entry.amount_usd <= 0:
        raise HTTPException(status_code=400, detail={"error": "bad_amount"})
    if status == "paid" and (entry.rate_brl is None or entry.rate_brl <= 0):
        raise HTTPException(
            status_code=400,
            detail={"error": "rate_required", "message": "rate_brl required when status=paid"},
        )
    payload: dict[str, Any] = {
        "date": entry.date,
        "amount_usd": float(entry.amount_usd),
        "status": status,
        "note": entry.note or "",
    }
    if entry.rate_brl is not None and entry.rate_brl > 0:
        payload["rate_brl"] = float(entry.rate_brl)
    return payload


@router.post("/rental-payments", response_model=RentalPaymentsListOut)
def add_rental(
    entry: RentalPaymentIn,
    project: str = Query(...),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Добавить платёж аренды (paid или pending)."""
    _bind_user(user)
    from v2.legacy.reports import add_rental_payment as _add
    payload = _rental_payment_payload(entry)
    ok = _add(project, payload)
    if not ok:
        raise HTTPException(status_code=404, detail="project_not_found")
    return _rental_payments_response(project, auto_generate=False)


@router.patch("/rental-payments", response_model=RentalPaymentsListOut)
def update_rental(
    entry: RentalPaymentIn,
    project: str = Query(...),
    index: int = Query(..., ge=0),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Заменить платёж аренды по индексу."""
    _bind_user(user)
    from v2.legacy.reports import update_rental_payment as _upd
    payload = _rental_payment_payload(entry)
    ok = _upd(project, index, payload)
    if not ok:
        raise HTTPException(status_code=404, detail="payment_not_found")
    return _rental_payments_response(project, auto_generate=False)


@router.delete("/rental-payments", response_model=RentalPaymentsListOut)
def remove_rental(
    project: str = Query(...),
    index: int = Query(..., ge=0),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Удалить платёж аренды."""
    _bind_user(user)
    from v2.legacy.reports import delete_rental_payment as _del
    ok = _del(project, index)
    if not ok:
        raise HTTPException(status_code=404, detail="payment_not_found")
    return _rental_payments_response(project, auto_generate=False)


# ── Publicidade reconciliation ──────────────────────────────────────────────

@router.get("/publicidade/reconciliation", response_model=PublicidadeReconciliationOut)
def get_publicidade_reconciliation(
    project: str = Query(...),
    period_from: str = Query(..., alias="from"),
    period_to: str = Query(..., alias="to"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Сверка расхода publicidade за период: CSV vs fatura.

    Вызывает `get_publicidade_by_period` дважды — отфильтрованно по типу источника,
    чтобы показать юзеру два числа и Δ. CSV = реальный дневной расход ML Ads.
    Fatura = то, что ML выставил/юзер закрыл как итог месяца.
    """
    _bind_user(user)
    from v2.legacy.reports import get_publicidade_by_period

    try:
        pf = datetime.strptime(period_from, "%Y-%m-%d").date()
        pt = datetime.strptime(period_to, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="bad_date_format_expected_YYYY-MM-DD")
    if pf > pt:
        raise HTTPException(status_code=400, detail="period_from_after_period_to")

    proj_upper = project.upper()
    csv_data = get_publicidade_by_period(proj_upper, pf, pt, only="csv")
    fatura_data = get_publicidade_by_period(proj_upper, pf, pt, only="fatura")

    csv_total = float(csv_data.get("total") or 0)
    # `fatura_total` = сумма ВСЕХ введённых фатур проекта (не только попадающих
    # в reconciliation-период). Так пользователь видит полную сумму ручного ввода.
    from v2.legacy.reports import list_publicidade_invoices
    all_invoices = list_publicidade_invoices(proj_upper)
    fatura_total = sum(float((inv or {}).get("valor", 0) or 0) for inv in all_invoices)
    return {
        "project": proj_upper,
        "period_from": period_from,
        "period_to": period_to,
        "csv_total": round(csv_total, 2),
        "csv_files_used": csv_data.get("files_used") or [],
        "fatura_total": round(fatura_total, 2),
        "fatura_files_used": fatura_data.get("files_used") or [],
        "delta": round(fatura_total - csv_total, 2),
        "uncovered_days_csv": int(csv_data.get("uncovered_days") or 0),
        "uncovered_days_fatura": int(fatura_data.get("uncovered_days") or 0),
        "total_days": int(csv_data.get("total_days") or 0),
    }


@router.get("/coverage", response_model=CoverageOut)
def get_coverage_endpoint(
    project: str = Query(...),
    period_from: str = Query(..., alias="from"),
    period_to: str = Query(..., alias="to"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Покрытие источниками данных (publicidade + armazenagem) за период.

    Возвращает сегменты дней, покрытых CSV / fatura / uncovered — для таймлайна.
    `csv_raw_range` — весь реальный охват CSV (до сужения слайдером),
    `csv_window` — пользовательское сужение из project.publicidade_csv_window.
    """
    _bind_user(user)
    from v2.legacy.reports import get_coverage

    try:
        pf = datetime.strptime(period_from, "%Y-%m-%d").date()
        pt = datetime.strptime(period_to, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="bad_date_format_expected_YYYY-MM-DD")
    if pf > pt:
        raise HTTPException(status_code=400, detail="period_from_after_period_to")

    return get_coverage(project.upper(), pf, pt)


# ── Planned Payments / DDS Planning ─────────────────────────────────────────
# Port of Streamlit _admin/dds_planning.py. Per-user list lives in
# user_data.f2_planned_payments; monthly grid + recurring detection derived.

@router.get("/planned-payments", response_model=PlannedPaymentsOut)
def get_planned_payments(user: CurrentUser = Depends(current_user)) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.planning import load_payments
    payments = load_payments()
    return {"payments": payments, "count": len(payments)}


@router.post("/planned-payments", response_model=PlannedPaymentMutOut)
def create_planned_payment(
    body: PlannedPaymentIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.planning import add_payment
    row = add_payment(body.model_dump())
    return {"updated": True, "payment": row}


@router.put("/planned-payments/{payment_id}", response_model=PlannedPaymentMutOut)
def put_planned_payment(
    payment_id: int,
    body: PlannedPaymentIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.planning import update_payment, load_payments
    ok = update_payment(payment_id, body.model_dump())
    if not ok:
        raise HTTPException(status_code=404, detail="payment_not_found")
    row = next((p for p in load_payments() if int(p.get("id") or -1) == payment_id), None)
    return {"updated": True, "payment": row}


@router.delete("/planned-payments/{payment_id}", response_model=PlannedPaymentMutOut)
def remove_planned_payment(
    payment_id: int,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.planning import delete_payment
    ok = delete_payment(payment_id)
    if not ok:
        raise HTTPException(status_code=404, detail="payment_not_found")
    return {"deleted": True}


@router.get("/planned-payments/monthly", response_model=MonthlyPlanOut)
def get_monthly_plan(
    months: int = Query(12, ge=1, le=36),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Aggregate all planned payments into next N months. Used by Planning → Monthly grid."""
    _bind_user(user)
    from v2.legacy.planning import build_monthly_plan
    plan = build_monthly_plan(months)
    months_sorted = sorted(plan.keys())
    buckets = {k: {"month": k, **v} for k, v in plan.items()}
    return {"months": months_sorted, "buckets": buckets}


@router.get("/planned-payments/suggest-recurring", response_model=RecurringSuggestionsOut)
def get_recurring_suggestions(
    min_occurrences: int = Query(3, ge=2, le=12),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Scan all bank uploads of the user and suggest recurring payment patterns
    (same label appearing in >= min_occurrences distinct months)."""
    _bind_user(user)
    from v2.legacy.planning import detect_recurring_from_bank_sync
    items = detect_recurring_from_bank_sync(user.id, min_occurrences)
    return {"suggestions": items, "min_occurrences": min_occurrences}


# ── Loans (Balance sheet Liabilities) ───────────────────────────────────────

@router.get("/loans", response_model=LoansListOut)
def list_loans_endpoint(
    project: str = Query(...),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import load_loans, loans_balance
    items = load_loans(project)
    return {
        "project": project.upper(),
        "loans": items,
        "total_outstanding_brl": loans_balance(project, None),
    }


@router.post("/loans", response_model=LoanMutOut)
def create_loan(
    body: LoanIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import add_loan
    row = add_loan(body.model_dump())
    return {"updated": True, "loan": row}


@router.put("/loans/{loan_id}", response_model=LoanMutOut)
def put_loan(
    loan_id: int,
    body: LoanIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import update_loan, load_loans
    ok = update_loan(loan_id, body.model_dump())
    if not ok:
        raise HTTPException(status_code=404, detail="loan_not_found")
    row = next((it for it in load_loans() if int(it.get("id") or -1) == loan_id), None)
    return {"updated": True, "loan": row}


@router.delete("/loans/{loan_id}", response_model=LoanMutOut)
def remove_loan(
    loan_id: int,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import delete_loan
    ok = delete_loan(loan_id)
    if not ok:
        raise HTTPException(status_code=404, detail="loan_not_found")
    return {"deleted": True}


# ── Dividends (Equity reductions) ───────────────────────────────────────────

@router.get("/dividends", response_model=DividendsListOut)
def list_dividends_endpoint(
    project: str = Query(...),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import load_dividends, dividends_total
    items = load_dividends(project)
    return {
        "project": project.upper(),
        "dividends": items,
        "total_amount_brl": dividends_total(project, None),
    }


@router.post("/dividends", response_model=DividendMutOut)
def create_dividend(
    body: DividendIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import add_dividend
    row = add_dividend(body.model_dump())
    return {"updated": True, "dividend": row}


@router.delete("/dividends/{dividend_id}", response_model=DividendMutOut)
def remove_dividend(
    dividend_id: int,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _bind_user(user)
    from v2.legacy.capital import delete_dividend
    ok = delete_dividend(dividend_id)
    if not ok:
        raise HTTPException(status_code=404, detail="dividend_not_found")
    return {"deleted": True}


# ── Accounts Payable (feeds from planned_payments) ──────────────────────────

@router.get("/accounts-payable", response_model=APListOut)
def list_accounts_payable(
    project: str = Query(...),
    as_of: Optional[str] = Query(None, description="ISO YYYY-MM-DD; default: today"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Unpaid expense planned_payments with date <= as_of. Feeds Balance sheet."""
    _bind_user(user)
    from v2.legacy.planning import list_unpaid_ap, unpaid_ap_total
    as_of_date = _parse_iso(as_of) or date.today()
    items = list_unpaid_ap(project, as_of_date)
    return {
        "project": project.upper(),
        "as_of": as_of_date.isoformat(),
        "items": items,
        "total_brl": unpaid_ap_total(project, as_of_date),
    }


@router.post("/planned-payments/{payment_id}/mark-paid", response_model=PlannedPaymentMutOut)
def mark_planned_paid_endpoint(
    payment_id: int,
    body: MarkPaidIn,
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Toggle the paid status on a planned_payment row. UI: «Marcar como paga»."""
    _bind_user(user)
    from v2.legacy.planning import mark_paid, load_payments
    if body.paid is False:
        stamp = False  # clear flag → marks unpaid
    else:
        stamp = body.paid_at  # None → stamp now; explicit ISO → that timestamp
    ok = mark_paid(payment_id, stamp)
    if not ok:
        raise HTTPException(status_code=404, detail="payment_not_found")
    row = next((p for p in load_payments() if int(p.get("id") or -1) == payment_id), None)
    return {"updated": True, "payment": row}


@router.post("/backfill-fs-to-db")
async def backfill_fs_to_db(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Import every FS-stored vendas / armazenagem / stock_full file into the
    `uploads` table for the current user.

    Finance reports have always read from the shared filesystem (legacy).
    Escalar's ABC aggregator uses the per-user `uploads` table (DB mode).
    This endpoint bridges the gap: call it once, then /escalar/products will
    see the same data Finance already shows.

    Idempotent — dedup by content_sha256 on partial unique index.
    """
    from pathlib import Path
    from v2.parsers.vendas_ml import VENDAS_DIR, is_vendas_ml_file
    from v2.parsers.armazenagem import ARMAZENAGEM_DIRS, _is_armazenamento_file
    from v2.parsers.stock_full import list_stock_full_files

    if pool is None:
        raise HTTPException(status_code=500, detail="db_pool_unavailable")

    async def _ingest_dir(directory: Path, source_key: str, predicate) -> int:
        if not directory.exists():
            return 0
        n = 0
        for entry in sorted(directory.iterdir()):
            if not entry.is_file() or not predicate(entry.name):
                continue
            data = entry.read_bytes()
            await uploads_storage.put_file(
                pool,
                user_id=user.id,
                source_key=source_key,
                filename=entry.name,
                file_bytes=data,
            )
            n += 1
        return n

    vendas = await _ingest_dir(VENDAS_DIR, "vendas_ml", is_vendas_ml_file)
    armazenagem = 0
    for d in ARMAZENAGEM_DIRS:
        armazenagem += await _ingest_dir(d, "armazenagem_full", _is_armazenamento_file)

    stock_full = 0
    for p in list_stock_full_files():
        data = p.read_bytes()
        await uploads_storage.put_file(
            pool,
            user_id=user.id,
            source_key="stock_full",
            filename=p.name,
            file_bytes=data,
        )
        stock_full += 1

    return {
        "vendas_ml": vendas,
        "armazenagem_full": armazenagem,
        "stock_full": stock_full,
    }


@router.get("/uploads/debug")
async def uploads_debug(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Show what's in `uploads` for the current user + how many rows each
    vendas file parses into. Diagnoses the "Finance sees data, Escalar
    shows zero" mismatch.
    """
    if pool is None:
        raise HTTPException(status_code=500, detail="db_pool_unavailable")

    async with pool.acquire() as conn:
        by_source = await conn.fetch(
            """
            SELECT source_key, COUNT(*) AS n, MAX(created_at) AS last_upload,
                   SUM(LENGTH(file_bytes)) AS total_bytes
              FROM uploads
             WHERE user_id = $1
             GROUP BY source_key
             ORDER BY n DESC
            """,
            user.id,
        )
        vendas_files = await conn.fetch(
            """
            SELECT id, filename, LENGTH(file_bytes) AS bytes,
                   to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
              FROM uploads
             WHERE user_id = $1 AND source_key = 'vendas_ml'
             ORDER BY created_at DESC
            """,
            user.id,
        )

    # Parse each vendas file via the same parser db_loader uses
    from v2.parsers.vendas_ml import parse_vendas_bytes
    parse_results = []
    async with pool.acquire() as conn:
        for f in vendas_files[:5]:  # first 5 only, to keep payload small
            row = await conn.fetchrow(
                "SELECT file_bytes FROM uploads WHERE id = $1", f["id"]
            )
            if not row:
                continue
            try:
                rows = parse_vendas_bytes(row["file_bytes"], f["filename"])
                parse_results.append({
                    "filename": f["filename"],
                    "bytes": f["bytes"],
                    "parsed_rows": len(rows),
                    "sample_skus": [r.sku for r in rows[:3] if r.sku],
                })
            except Exception as err:  # noqa: BLE001
                parse_results.append({
                    "filename": f["filename"],
                    "bytes": f["bytes"],
                    "parse_error": str(err)[:300],
                })

    return {
        "user_id": user.id,
        "by_source": [dict(r) for r in by_source],
        "vendas_files_count": len(vendas_files),
        "vendas_files_sample": [dict(f) for f in vendas_files[:10]],
        "parse_results": parse_results,
    }
