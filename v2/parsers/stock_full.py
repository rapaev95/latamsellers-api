"""Stock Full XLSX parser (Mercado Livre "Estoque geral / Full").

The user exports this weekly from ML → Full → Estoque → Exportar relatório geral.
Sheets split inventory by status (Boa qualidade, Para impulsionar, Retido,
Em trânsito, …). Sheet "Resumo" aggregates and is skipped — we re-aggregate
from the per-status sheets to keep per-status totals accessible in meta.

Header row is at index 5 (row 6 in Excel) — same convention as
Streamlit's legacy `reports.load_stock_full` which this module mirrors.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[3]  # …/_admin
STOCK_FULL_DIRS = [
    _BASE.parent / "_data",
    _BASE.parent / "vendas",
]


@dataclass
class StockFullSku:
    sku: str
    title: str = ""
    mlb: str = ""
    total: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


def _is_stock_full_file(filename: str) -> bool:
    lower = filename.lower()
    if not (lower.endswith(".xlsx") or lower.endswith(".xls")):
        return False
    return "stock_full" in lower or "stock_general_full" in lower or "estoque" in lower and "full" in lower


def _find_qty_col(columns: list[str]) -> str | None:
    """ML report puts a "Unidades de boa qualidade" label in row 4 (merged),
    pandas reads the true numeric column as "…Unidades….1". Prefer the `.1`
    variant — it's the integer column. Fall back to any "unidades"/"qtd".
    """
    for c in columns:
        cl = str(c).lower()
        if (".1" in str(c)) and ("unidades" in cl or "estoque" in cl):
            return c
    for c in columns:
        cl = str(c).lower()
        if "unidades" in cl or "qtd" in cl:
            return c
    return None


def _find_title_col(columns: list[str]) -> str | None:
    for c in columns:
        cl = str(c).lower()
        if "título" in cl or "titulo" in cl or "produto" in cl or "nome" in cl:
            return c
    return None


def _find_mlb_col(columns: list[str]) -> str | None:
    for c in columns:
        cl = str(c).lower()
        if "anúncio" in cl or "anuncio" in cl or cl.strip() == "mlb":
            return c
    return None


def parse_stock_full_bytes(file_bytes: bytes) -> dict[str, StockFullSku]:
    """Parse an uploaded `stock_full*.xlsx` blob → {sku: StockFullSku}.

    Aggregates units across every non-Resumo sheet, keyed by SKU.
    Empty bytes or unreadable XLSX → empty dict (caller decides fallback).
    """
    if not file_bytes:
        return {}
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        return {}

    result: dict[str, StockFullSku] = {}

    for sheet in xl.sheet_names:
        if sheet.strip().lower() == "resumo":
            continue
        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=5)
        except Exception:
            continue
        if "SKU" not in df.columns:
            continue
        df = df[df["SKU"].notna()]

        qty_col = _find_qty_col(list(df.columns))
        if qty_col is None:
            continue
        title_col = _find_title_col(list(df.columns))
        mlb_col = _find_mlb_col(list(df.columns))

        for _, row in df.iterrows():
            sku_raw = row.get("SKU", "")
            sku = ("" if sku_raw is None else str(sku_raw)).strip()
            if not sku or sku.lower() == "nan":
                continue
            try:
                qty = int(float(row.get(qty_col, 0) or 0))
            except (ValueError, TypeError):
                qty = 0
            if qty <= 0:
                continue

            entry = result.get(sku)
            if entry is None:
                entry = StockFullSku(sku=sku)
                result[sku] = entry
            entry.total += qty
            entry.by_status[sheet] = entry.by_status.get(sheet, 0) + qty

            if not entry.title and title_col:
                tit = row.get(title_col, "")
                tit = ("" if tit is None else str(tit)).strip()
                if tit and tit.lower() != "nan":
                    entry.title = tit[:220]
            if not entry.mlb and mlb_col:
                m = row.get(mlb_col, "")
                m = ("" if m is None else str(m)).strip()
                if m and m.lower() != "nan":
                    entry.mlb = m

    return result


def read_stock_full_file(path: Path) -> dict[str, StockFullSku]:
    try:
        with open(path, "rb") as f:
            return parse_stock_full_bytes(f.read())
    except OSError:
        return {}


def list_stock_full_files() -> list[Path]:
    """Discover FS-mode stock_full files under _data/**/ and vendas/."""
    found: list[Path] = []
    seen: set[str] = set()
    # Monthly subfolders under _data/
    data_dir = STOCK_FULL_DIRS[0]
    if data_dir.exists():
        # _data/<month>/stock_full*.xlsx — glob one level deep
        for month_dir in data_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for p in month_dir.glob("stock_full*.xlsx"):
                if p.name not in seen:
                    seen.add(p.name)
                    found.append(p)
    # Flat vendas/ fallback
    vendas_dir = STOCK_FULL_DIRS[1]
    if vendas_dir.exists():
        for fn in os.listdir(vendas_dir):
            if _is_stock_full_file(fn) and fn not in seen:
                seen.add(fn)
                found.append(vendas_dir / fn)
    # Newest first by mtime
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def load_all_stock_full() -> dict[str, StockFullSku]:
    """FS-mode loader: parse the newest stock_full file, merge older files
    only for SKUs missing from the newest one."""
    merged: dict[str, StockFullSku] = {}
    for p in list_stock_full_files():
        parsed = read_stock_full_file(p)
        for sku, entry in parsed.items():
            if sku not in merged:
                merged[sku] = entry
    return merged
