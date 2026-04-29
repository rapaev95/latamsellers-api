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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile

from v2.db import get_pool
from v2.deps import CurrentUser, current_user
from v2.legacy import db_storage as legacy_db
from v2.legacy import config as legacy_config
from v2.services import finance_cache
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
)
from v2.storage import uploads_storage

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
def get_reports(
    response: Response,
    project: str = Query(..., description="Project ID, e.g. 'GANZA'"),
    period_from: Optional[str] = Query(None, alias="from"),
    period_to: Optional[str] = Query(None, alias="to"),
    basis: str = Query("accrual", pattern="^(accrual|cash)$"),
    fresh: bool = Query(False, description="Bypass cache and recompute from scratch"),
    user: CurrentUser = Depends(current_user),
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

    return {
        "id": upload_id,
        "filename": filename,
        "source_key": resolved_key,
        "detected": detected,
        "size_bytes": len(file_bytes),
        "was_duplicate": was_duplicate,
        "unlocked": unlocked_pwd is not None,
        "dados_fiscais_sync": dados_fiscais_sync,
    }


@router.delete("/uploads/{upload_id}")
async def delete_upload(
    upload_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Delete a single upload row owned by the caller."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    from v2.legacy.reports import invalidate_vendas_cache, invalidate_mlb_to_sku_index

    # Peek at source_key before delete so we know whether to invalidate vendas cache.
    target = await uploads_storage.get_file(pool, user.id, upload_id)
    ok = await uploads_storage.delete_file(pool, user.id, upload_id)
    if not ok:
        raise HTTPException(status_code=404, detail="upload_not_found")

    if target is not None and target.source_key == "vendas_ml":
        invalidate_vendas_cache(user.id)
    if target is not None and target.source_key in ("vendas_ml", "stock_full"):
        invalidate_mlb_to_sku_index(user.id)

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

    from v2.legacy.bank_tx import parse_bank_tx_bytes, CATEGORY_OPTIONS
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

    rows = parse_bank_tx_bytes(stored.source_key, stored.file_bytes)

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

    ok = update_project(pid, body.fields or {}, body.rental_fields)
    if not ok:
        return {"project_id": pid, "updated": False, "exists": False}

    _invalidate_projects_cache()
    invalidate_vendas_cache(user.id)
    return {"project_id": pid, "updated": True, "exists": True}


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
