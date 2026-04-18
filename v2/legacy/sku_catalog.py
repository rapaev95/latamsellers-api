"""
Каталог SKU: тип поставщика (import/local) и себестоимость BRL.
Хранение: _data/sku_catalog.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR

CATALOG_FILENAME = "sku_catalog.json"
VALID_SUPPLIER_TYPES = frozenset({"import", "local"})


def catalog_path() -> Path:
    return DATA_DIR / CATALOG_FILENAME


def normalize_sku(sku: str) -> str:
    return (sku or "").strip().upper()


def _coerce_item(it: dict[str, Any]) -> dict[str, Any] | None:
    sku_raw = str(it.get("sku", "")).strip()
    sku_key = normalize_sku(sku_raw)
    if not sku_key:
        return None
    st = str(it.get("supplier_type") or "local").lower().strip()
    if st not in VALID_SUPPLIER_TYPES:
        st = "local"
    cost_raw = it.get("unit_cost_brl")
    cost: float | None
    try:
        if cost_raw is None or cost_raw == "":
            cost = None
        else:
            c = float(cost_raw)
            cost = c if c >= 0 else None
    except (TypeError, ValueError):
        cost = None
    # Supplier state (UF) — for ICMS calculation
    supplier_state = str(it.get("supplier_state") or "").strip().upper()[:2]
    if supplier_state and supplier_state not in (
        "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS",
        "MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO",
    ):
        supplier_state = ""
    # Lead time (days from order to Full)
    lead_raw = it.get("lead_time_days")
    lead_time: int | None
    try:
        lead_time = int(lead_raw) if lead_raw is not None and lead_raw != "" else None
    except (TypeError, ValueError):
        lead_time = None
    project = str(it.get("project") or "").strip()
    return {
        "sku": sku_raw,
        "project": project,
        "supplier_type": st,
        "unit_cost_brl": cost,
        "supplier_state": supplier_state,
        "lead_time_days": lead_time,
        "note": str(it.get("note") or ""),
    }


def load_catalog() -> list[dict[str, Any]]:
    """Load catalog: PostgreSQL (per-user) if available, else JSON (local dev).
    If DATABASE_URL is set (production) — ONLY use DB, never JSON.
    """
    import os
    has_db_url = bool(os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL"))

    if has_db_url:
        try:
            from .db_storage import db_load, db_is_available
            if db_is_available():
                db_data = db_load("sku_catalog")
                if db_data and isinstance(db_data, dict):
                    items = db_data.get("items", [])
                    return [x for x in items if isinstance(x, dict)]
                elif db_data and isinstance(db_data, list):
                    return [x for x in db_data if isinstance(x, dict)]
        except Exception:
            pass
        return []  # Production: no fallback to shared JSON

    # No DB → local dev, fallback to JSON file
    path = catalog_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("items") if isinstance(raw, dict) else []
        return [x for x in items if isinstance(x, dict)]
    except Exception:
        return []


def save_catalog(items: list[dict[str, Any]]) -> bool:
    """Сохранить каталог; дубликаты SKU — последняя строка побеждает."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        row = _coerce_item(it)
        if row is None:
            continue
        merged[normalize_sku(row["sku"])] = row
    out_items = [merged[k] for k in sorted(merged.keys())]
    payload = {"version": 1, "items": out_items}
    # JSON (local)
    try:
        catalog_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    # PostgreSQL (Railway)
    try:
        from .db_storage import db_save
        db_save("sku_catalog", payload)
    except Exception:
        pass
    # Invalidate cached project index
    try:
        from .config import invalidate_catalog_project_index
        invalidate_catalog_project_index()
    except Exception:
        pass
    return True


def build_catalog_index() -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for it in load_catalog():
        row = _coerce_item(it)
        if row is None:
            continue
        idx[normalize_sku(row["sku"])] = {
            "supplier_type": row["supplier_type"],
            "unit_cost_brl": row["unit_cost_brl"],
            "supplier_state": row.get("supplier_state", ""),
            "lead_time_days": row.get("lead_time_days"),
            "note": row["note"],
        }
    return idx


def get_sku_row(sku: str) -> dict[str, Any] | None:
    return build_catalog_index().get(normalize_sku(sku))


def assess_stock_for_project(
    project: str,
    by_sku: dict[str, int] | None,
    stock_units_external: int,
    avg_cost_per_unit_brl: float | None,
) -> dict[str, Any]:
    """
    Оценка стоимости стока: явные цены из каталога; без цены — avg_cost_per_unit_brl
    (если задан); иначе SKU попадают в missing_*.
    project зарезервирован для будущих фильтров.
    """
    _ = project
    by_sku = by_sku or {}
    idx = build_catalog_index()

    total_val = 0.0
    by_supplier: dict[str, float] = {"import": 0.0, "local": 0.0, "fallback": 0.0}
    units_from_catalog = 0
    units_from_fallback = 0
    missing_skus: list[str] = []
    missing_units = 0

    avg: float | None
    try:
        if avg_cost_per_unit_brl is None:
            avg = None
        else:
            a = float(avg_cost_per_unit_brl)
            avg = a if a > 0 else None
    except (TypeError, ValueError):
        avg = None

    for sku, qty in by_sku.items():
        q = int(qty) if qty else 0
        if q <= 0:
            continue
        key = normalize_sku(str(sku))
        row = idx.get(key)
        cost: float | None = None
        if row and row.get("unit_cost_brl") is not None:
            try:
                c = float(row["unit_cost_brl"])
                if c > 0:
                    cost = c
            except (TypeError, ValueError):
                pass
        if cost is not None:
            line = q * cost
            total_val += line
            units_from_catalog += q
            st = row["supplier_type"]
            if st not in by_supplier:
                st = "local"
            by_supplier[st] = by_supplier.get(st, 0) + line
        elif avg is not None:
            total_val += q * avg
            units_from_fallback += q
            by_supplier["fallback"] += q * avg
        else:
            missing_skus.append(str(sku).strip())
            missing_units += q

    ext = int(stock_units_external or 0)
    if ext > 0:
        if avg is not None:
            total_val += ext * avg
            units_from_fallback += ext
            by_supplier["fallback"] += ext * avg
        else:
            missing_units += ext

    return {
        "stock_value_brl": round(total_val, 2),
        "by_supplier_type": {k: round(v, 2) for k, v in by_supplier.items() if v > 0},
        "units_from_catalog": units_from_catalog,
        "units_from_fallback": units_from_fallback,
        "missing_skus": sorted(set(missing_skus)),
        "missing_units": missing_units,
    }
