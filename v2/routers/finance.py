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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from v2.db import get_pool
from v2.deps import CurrentUser, current_user
from v2.legacy import db_storage as legacy_db
from v2.legacy import config as legacy_config
from v2.schemas.finance import (
    ProjectsListOut, ReportsBundleOut,
    SkuMappingOut, SkuBulkSaveIn, SkuBulkSaveOut,
    PnlMatrixOut,
    OrphanPacotesResponse, OrphanSaveIn, OrphanSaveOut,
    UploadsListOut, UploadSaveOut, SourceCatalogOut,
)
from v2.storage import uploads_storage

router = APIRouter(prefix="/finance", tags=["finance"])

_COMPUTE_TIMEOUT_SECONDS = 30  # legacy compute can scan many files; cap response time

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
    project: str = Query(..., description="Project ID, e.g. 'GANZA'"),
    period_from: Optional[str] = Query(None, alias="from"),
    period_to: Optional[str] = Query(None, alias="to"),
    basis: str = Query("accrual", pattern="^(accrual|cash)$"),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    """Compute ОПиУ + ДДС + Баланс for one project + period in a single call.

    Returns three sub-objects matching the Streamlit "Отчёты" page tabs.
    Errors from any single computation are reported per-tab, not as 500.
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

    # Lazy imports — only pay the cost when /reports is actually called
    from v2.legacy.finance import compute_pnl, compute_cashflow, compute_balance
    from v2.legacy.reports import load_vendas_ml_report

    out: dict[str, Any] = {
        "project": project,
        "period": {"from": pf.isoformat(), "to": pt.isoformat()},
        "basis": basis,
    }

    # Pre-warm the vendas DataFrame cache in the request context so all three
    # parallel compute tasks below find it hot. Without this the threads race
    # to rebuild the 2208-row df simultaneously (DB + pd.read_csv + df.apply)
    # and the wall time can exceed the shared timeout.
    try:
        load_vendas_ml_report()
    except Exception:
        pass  # individual computes will re-raise properly

    # Three computes run in parallel under a single shared timeout
    results = _run_parallel_with_timeout({
        "pnl": lambda: compute_pnl(project, (pf, pt), basis=basis),
        "cashflow": lambda: compute_cashflow(project, (pf, pt)),
        "balance": lambda: compute_balance(project, pt, basis=basis),
    })

    pnl_res, pnl_err = results["pnl"]
    if pnl_res is not None: out["pnl"] = _dataclass_to_dict(pnl_res)
    if pnl_err: out["pnl_error"] = pnl_err

    cf_res, cf_err = results["cashflow"]
    if cf_res is not None: out["cashflow"] = _dataclass_to_dict(cf_res)
    if cf_err: out["cashflow_error"] = cf_err

    bal_res, bal_err = results["balance"]
    if bal_res is not None: out["balance"] = _dataclass_to_dict(bal_res)
    if bal_err: out["balance_error"] = bal_err

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


def _build_sku_mapping(user_id: int) -> dict[str, Any]:
    """Build the merged SKU view: vendas + catalog overlay + groupings.

    Mirrors _admin/app.py:2867-2946 logic. Synchronous (sits inside FastAPI
    threadpool dispatch).
    """
    from v2.legacy.reports import load_vendas_ml_report
    from v2.legacy.sku_catalog import load_catalog, normalize_sku
    from v2.legacy.config import get_project_by_sku, mlb_url

    legacy_db.set_current_user_id(user_id)

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

    # 2) Catalog overlay (project, cost, supplier — saved values win)
    for it in load_catalog():
        nk = normalize_sku(str(it.get("sku", "")))
        if not nk:
            continue
        if nk in all_skus:
            all_skus[nk]["project"] = it.get("project", "") or ""
            all_skus[nk]["supplier_type"] = it.get("supplier_type", "local")
            all_skus[nk]["unit_cost_brl"] = it.get("unit_cost_brl")
            all_skus[nk]["note"] = it.get("note", "") or ""
        else:
            all_skus[nk] = {
                "sku": str(it.get("sku", "")).strip(),
                "title": "", "mlb": "", "link": "",
                "project": it.get("project", "") or "",
                "supplier_type": it.get("supplier_type", "local"),
                "unit_cost_brl": it.get("unit_cost_brl"),
                "note": it.get("note", "") or "",
            }

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


@router.get("/sku-mapping", response_model=SkuMappingOut)
def get_sku_mapping(user: CurrentUser = Depends(current_user)):
    _bind_user(user)
    return _build_sku_mapping(user.id)


@router.get("/pnl-matrix", response_model=PnlMatrixOut)
def get_pnl_matrix(
    project: str = Query(..., description="Project ID, e.g. 'ARTHUR'"),
    user: CurrentUser = Depends(current_user),
):
    """Monthly PnL matrix (rows × 12 months): revenue breakdown + expenses +
    summary (op_profit, margin, orders). Mirrors Streamlit `build_monthly_pnl_matrix`.

    Wraps the legacy compute with the same 15s timeout/parallel safeguard used
    by /reports. On timeout returns empty months/rows.
    """
    _bind_user(user)
    from v2.legacy.reports import build_monthly_pnl_matrix

    results = _run_parallel_with_timeout({
        "matrix": lambda: build_monthly_pnl_matrix(project),
    })
    data, err = results["matrix"]
    if data is None or err:
        return {"project": project, "months": [], "years": [], "rows": []}
    return {
        "project": project,
        "months": data.get("months", []),
        "years": data.get("years", []),
        "rows": data.get("rows", []),
    }


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
        if u.note is not None:
            item["note"] = u.note
        by_key[nk] = item
        saved += 1

    save_catalog(list(by_key.values()))
    invalidate_catalog_project_index()
    _invalidate_projects_cache()
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
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Store a user file in the `uploads` table with SHA256 dedupe.

    `source_key` is resolved in this order:
      1. explicit `source_key` form field (client override)
      2. filename-based auto-detect (`v2/legacy/source_detection.detect_source_from_filename`)
      3. HTTP 400 if neither produced a valid source_key
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")

    from v2.legacy.source_detection import detect_source
    from v2.legacy.reports import invalidate_vendas_cache

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

    return {
        "id": upload_id,
        "filename": filename,
        "source_key": resolved_key,
        "detected": detected,
        "size_bytes": len(file_bytes),
        "was_duplicate": was_duplicate,
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

    from v2.legacy.reports import invalidate_vendas_cache

    # Peek at source_key before delete so we know whether to invalidate vendas cache.
    target = await uploads_storage.get_file(pool, user.id, upload_id)
    ok = await uploads_storage.delete_file(pool, user.id, upload_id)
    if not ok:
        raise HTTPException(status_code=404, detail="upload_not_found")

    if target is not None and target.source_key == "vendas_ml":
        invalidate_vendas_cache(user.id)

    return {"deleted": True}
