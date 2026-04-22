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


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def _coerce_item(it: dict[str, Any]) -> dict[str, Any] | None:
    sku_raw = str(it.get("sku", "")).strip()
    sku_key = normalize_sku(sku_raw)
    if not sku_key:
        return None
    st = str(it.get("supplier_type") or "local").lower().strip()
    if st not in VALID_SUPPLIER_TYPES:
        st = "local"
    cost = _safe_float(it.get("unit_cost_brl"))
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

    # Dados Fiscais fields (всё опционально, default None)
    origem_code = it.get("origem_code")
    try:
        origem_code = int(origem_code) if origem_code is not None and origem_code != "" else None
    except (TypeError, ValueError):
        origem_code = None
    origem_type_raw = str(it.get("origem_type") or "").strip().lower() or None
    if origem_type_raw not in (None, "import", "local"):
        origem_type_raw = None

    return {
        "sku": sku_raw,
        "project": project,
        "supplier_type": st,
        "unit_cost_brl": cost,
        "supplier_state": supplier_state,
        "lead_time_days": lead_time,
        "note": str(it.get("note") or ""),
        # --- Dados Fiscais (ML official) ---
        "mlb": _safe_str(it.get("mlb")),
        "titulo": _safe_str(it.get("titulo")) or _safe_str(it.get("titulo_anuncio")),
        "variacao": _safe_str(it.get("variacao")),
        "ean": _safe_str(it.get("ean")),
        "ncm": _safe_str(it.get("ncm")),
        "cest": _safe_str(it.get("cest")),
        "origem_code": origem_code,
        "origem_type": origem_type_raw,
        "peso_liquido_kg": _safe_float(it.get("peso_liquido_kg")),
        "peso_bruto_kg": _safe_float(it.get("peso_bruto_kg")),
        "unidade": _safe_str(it.get("unidade")),
        "descricao_nfe": _safe_str(it.get("descricao_nfe")),
        "csosn_venda": _safe_str(it.get("csosn_venda")),
        "csosn_transferencia": _safe_str(it.get("csosn_transferencia")),
        "chave_fci": _safe_str(it.get("chave_fci")),
        "regra_tributaria": _safe_str(it.get("regra_tributaria")),
        "dados_fiscais_synced_at": _safe_str(it.get("dados_fiscais_synced_at")),
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


def _normalize_mlb(v: Any) -> str:
    """Нормализует MLB из любого источника к digits-only для мэтчинга.

    stock_full.xlsx хранит MLB как float-строку `4516196937.0` (legacy формат),
    Dados Fiscais — как `MLB5649425304`. Чтобы сопоставить: убираем `.0`-суффикс,
    префикс 'MLB', любой нецифровой символ; убираем ведущие нули.
    """
    s = str(v or "").strip().upper()
    if not s:
        return ""
    # remove trailing ".0" (float-cast of integer from Excel)
    if s.endswith(".0"):
        s = s[:-2]
    # drop everything that isn't digit
    digits = "".join(c for c in s if c.isdigit())
    return digits.lstrip("0")


def build_catalog_mlb_index() -> dict[str, dict[str, Any]]:
    """Индекс каталога по **нормализованному MLB**. Используется как fallback
    для SKU-лукапа, когда внутренние коды стока (sumka5-1) не совпадают с SKU
    в каталоге (HB50173-5), но MLB-объявление одно и то же."""
    idx: dict[str, dict[str, Any]] = {}
    for it in load_catalog():
        row = _coerce_item(it)
        if row is None:
            continue
        mlb_norm = _normalize_mlb(row.get("mlb"))
        if not mlb_norm:
            continue
        # If multiple catalog items share an MLB, the one with unit_cost wins
        existing = idx.get(mlb_norm)
        new_entry = {
            "supplier_type": row["supplier_type"],
            "unit_cost_brl": row["unit_cost_brl"],
            "supplier_state": row.get("supplier_state", ""),
            "lead_time_days": row.get("lead_time_days"),
            "note": row["note"],
        }
        if existing is None:
            idx[mlb_norm] = new_entry
            continue
        # keep entry that has real cost
        if (existing.get("unit_cost_brl") or 0) == 0 and (new_entry.get("unit_cost_brl") or 0) > 0:
            idx[mlb_norm] = new_entry
    return idx


def get_sku_row(sku: str) -> dict[str, Any] | None:
    return build_catalog_index().get(normalize_sku(sku))


def assess_stock_for_project(
    project: str,
    by_sku: dict[str, int] | None,
    stock_units_external: int,
    avg_cost_per_unit_brl: float | None,
    sku_mlbs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Оценка стоимости стока. Порядок лукапа cost для каждого SKU:
      1) catalog[sku] — прямое совпадение кода
      2) catalog[mlb-norm] — fallback через MLB из stock_full (если `sku_mlbs` задан);
         это закрывает случай когда у юзера внутренние коды (sumka5-1, w02-1) в
         stock_full, а каталог содержит SKU из Dados Fiscais (HB50173-9) — оба
         ссылаются на одну и ту же карточку ML, MLB-нормализация их соединяет.
      3) avg_cost_per_unit_brl — если явно задан для проекта
      4) SKU попадает в missing_*

    project зарезервирован для будущих фильтров.
    """
    _ = project
    by_sku = by_sku or {}
    sku_mlbs = sku_mlbs or {}
    idx = build_catalog_index()
    mlb_idx = build_catalog_mlb_index()

    total_val = 0.0
    by_supplier: dict[str, float] = {"import": 0.0, "local": 0.0, "fallback": 0.0}
    units_from_catalog = 0
    units_from_fallback = 0
    missing_skus: list[str] = []
    missing_details: list[dict[str, Any]] = []   # [{sku, mlb, units}] for UI
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
        # Fallback: MLB-based lookup через stock_full entry.mlb
        if cost is None:
            mlb_raw = sku_mlbs.get(sku) or sku_mlbs.get(key) or ""
            mlb_norm = _normalize_mlb(mlb_raw)
            if mlb_norm:
                mlb_row = mlb_idx.get(mlb_norm)
                if mlb_row and mlb_row.get("unit_cost_brl") is not None:
                    try:
                        c = float(mlb_row["unit_cost_brl"])
                        if c > 0:
                            cost = c
                            row = mlb_row  # use matched catalog row для supplier_type
                    except (TypeError, ValueError):
                        pass
        if cost is not None:
            line = q * cost
            total_val += line
            units_from_catalog += q
            st = (row or {}).get("supplier_type") or "local"
            if st not in by_supplier:
                st = "local"
            by_supplier[st] = by_supplier.get(st, 0) + line
        elif avg is not None:
            total_val += q * avg
            units_from_fallback += q
            by_supplier["fallback"] += q * avg
        else:
            sku_clean = str(sku).strip()
            missing_skus.append(sku_clean)
            missing_details.append({
                "sku": sku_clean,
                "mlb": sku_mlbs.get(sku) or sku_mlbs.get(key) or "",
                "units": q,
            })
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
        "missing_sku_details": sorted(missing_details, key=lambda d: -d["units"]),
        "missing_units": missing_units,
    }


# Fields populated from Dados Fiscais — merged into each catalog item on sync.
# `unit_cost_brl` also gets updated, but only when `overwrite_costs=True` (opt-in).
_DADOS_FISCAIS_FIELDS = (
    "mlb", "titulo", "variacao", "ean", "ncm", "cest",
    "origem_code", "origem_type",
    "peso_liquido_kg", "peso_bruto_kg",
    "unidade", "descricao_nfe",
    "csosn_venda", "csosn_transferencia",
    "chave_fci", "regra_tributaria",
)


def sync_from_dados_fiscais(
    parsed: dict[str, dict[str, Any]],
    *,
    overwrite_costs: bool = True,
) -> dict[str, Any]:
    """Merge parsed Dados Fiscais records into sku_catalog.

    Args:
      parsed: `{sku_key: {mlb, custo_brl, ncm, origem_type, peso_*, ...}}`
        — output of `parse_dados_fiscais_bytes`.
      overwrite_costs: if True (default), Custo do Produto из ML перезаписывает
        `unit_cost_brl` в каталоге. Set False чтобы preserve ручные overrides.

    Returns:
      stats = {created, updated_fields, cost_updated, skipped, synced_at}
        — updated_fields counts SKUs where any Dados Fiscais field changed
          (независимо от cost).
    """
    from datetime import datetime, timezone

    if not parsed:
        return {"created": 0, "updated_fields": 0, "cost_updated": 0, "skipped": 0, "synced_at": None}

    synced_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Index current catalog by normalized SKU
    current = {normalize_sku(r.get("sku") or ""): r for r in load_catalog()}

    created = 0
    updated_fields = 0
    cost_updated = 0
    skipped = 0

    for sku_key, parsed_rec in parsed.items():
        if not sku_key:
            skipped += 1
            continue
        existing = current.get(sku_key)

        if existing is None:
            # New SKU
            new_item = {
                "sku": parsed_rec.get("sku") or sku_key,
                "supplier_type": parsed_rec.get("origem_type") or "local",
                "unit_cost_brl": parsed_rec.get("custo_brl"),
                "dados_fiscais_synced_at": synced_at,
            }
            for f in _DADOS_FISCAIS_FIELDS:
                if f in parsed_rec:
                    new_item[f] = parsed_rec[f]
            current[sku_key] = new_item
            created += 1
            continue

        # Update existing — compare and merge Dados Fiscais fields
        field_changed = False
        for f in _DADOS_FISCAIS_FIELDS:
            new_val = parsed_rec.get(f)
            if new_val is None:
                continue  # don't wipe existing data with None
            if existing.get(f) != new_val:
                existing[f] = new_val
                field_changed = True

        # Mirror origem_type into supplier_type if not set explicitly
        if parsed_rec.get("origem_type") and not existing.get("supplier_type"):
            existing["supplier_type"] = parsed_rec["origem_type"]
            field_changed = True

        # Cost update (opt-in)
        new_cost = parsed_rec.get("custo_brl")
        if overwrite_costs and new_cost is not None and new_cost > 0:
            if existing.get("unit_cost_brl") != new_cost:
                existing["unit_cost_brl"] = new_cost
                cost_updated += 1

        if field_changed or overwrite_costs:
            existing["dados_fiscais_synced_at"] = synced_at
        if field_changed:
            updated_fields += 1

    # Persist: save_catalog re-coerces every item + writes DB + JSON
    save_catalog(list(current.values()))

    return {
        "created": created,
        "updated_fields": updated_fields,
        "cost_updated": cost_updated,
        "skipped": skipped,
        "synced_at": synced_at,
        "total_skus": len(parsed),
    }
