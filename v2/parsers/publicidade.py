"""Parser for Mercado Livre's Product Ads report CSV/XLSX.

Source CSV:  `Relatorio_anuncios_patrocinados*.csv` (user uploads via
/finance/upload; source_key = 'ads_publicidade').

Columns we care about (Portuguese; the export is always pt_BR):
  - "Desde" (date_from of the reporting period)
  - "Até"   (date_to)
  - "Campanha"        — campaign name
  - "Título"          — listing title
  - "Código do anúncio" — MLB item id (this is the join key)
  - "Investimento"    — R$ spent on the listing over the period

Returns a flat list of per-row investments; caller aggregates by MLB + period
window. Mirrors `_admin/reports.py:_parse_publicidade_rows`, but operates on
bytes so it fits the same pattern as `parse_vendas_bytes` / `parse_armazenagem_bytes`.
"""
from __future__ import annotations

import csv as _csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PublicidadeRow:
    desde: date
    ate: date
    mlb: str           # "MLB..." — ML item_id
    campanha: str
    titulo: str
    investimento: float


def _parse_pt_date(raw: str) -> Optional[date]:
    """Parse dates in ML's export format.

    ML emits dates as "DD/MM/YY" or "DD/MM/YYYY", sometimes "YYYY-MM-DD".
    """
    s = (raw or "").strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_brl(raw) -> float:
    """Parse Brazilian float: '1.234,56' → 1234.56, 'R$ 12,50' → 12.50."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace("R$", "").strip()
    if not s or s.lower() == "nan":
        return 0.0
    # "1.234,56" → "1234.56"
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _col_index(header: list[str], needle: str) -> Optional[int]:
    for i, c in enumerate(header):
        if needle in (c or ""):
            return i
    return None


def parse_publicidade_bytes(file_bytes: bytes, filename: str = "") -> list[PublicidadeRow]:
    """Parse CSV or XLSX bytes and return per-MLB per-period investments.

    Assumes the ML export convention: header on row index 1 (row 0 is metadata/title).
    Unknown formats return empty list — caller shouldn't crash on bad uploads.
    """
    lower = (filename or "").lower()
    rows: list[list] = []
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        try:
            import pandas as pd  # lazy — pandas already in deps
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            # Prefer sheet with "Anúncios"/"Relat" in name; fall back to last sheet
            # (ML Studio exports put data on the final sheet).
            target_sheet = None
            for sn in xl.sheet_names:
                if "Anúncio" in sn or "Anuncio" in sn or "Relat" in sn:
                    target_sheet = sn
                    break
            if target_sheet is None:
                target_sheet = xl.sheet_names[-1]
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=target_sheet, header=None)
            rows = df.where(df.notna(), None).values.tolist()
        except Exception as err:  # noqa: BLE001
            log.warning("publicidade xlsx parse failed for %s: %s", filename, err)
            return []
    else:
        # CSV — ML exports with ';' delimiter and UTF-8 BOM.
        try:
            text = file_bytes.decode("utf-8-sig", errors="replace")
            rows = list(_csv.reader(io.StringIO(text), delimiter=";"))
        except Exception as err:  # noqa: BLE001
            log.warning("publicidade csv parse failed for %s: %s", filename, err)
            return []

    if len(rows) < 3:
        return []
    # Row index 1 is the header in ML's export (row 0 is the report title line).
    header = [str(c) if c is not None else "" for c in rows[1]]

    i_desde = _col_index(header, "Desde")
    i_ate = _col_index(header, "Até")
    i_mlb = _col_index(header, "Código do anúncio")
    i_inv = _col_index(header, "Investimento")
    i_camp = _col_index(header, "Campanha")
    i_titulo = _col_index(header, "Título")

    if None in (i_desde, i_ate, i_mlb, i_inv):
        log.warning("publicidade %s: missing required columns in header=%s", filename, header)
        return []

    out: list[PublicidadeRow] = []
    for r in rows[2:]:
        if len(r) <= max(i for i in (i_desde, i_ate, i_mlb, i_inv) if i is not None):
            continue
        desde = _parse_pt_date(str(r[i_desde]) if r[i_desde] is not None else "")
        ate = _parse_pt_date(str(r[i_ate]) if r[i_ate] is not None else "")
        mlb = str(r[i_mlb] or "").strip()
        if not desde or not ate or not mlb or mlb.lower() == "nan":
            continue
        # Normalize MLB — upper-case, strip any '.0' float artifacts.
        mlb_up = mlb.upper()
        if "." in mlb_up:
            head, _, tail = mlb_up.partition(".")
            if tail.strip("0") == "":
                mlb_up = head
        inv = _parse_brl(r[i_inv])
        campanha = str(r[i_camp]) if i_camp is not None and len(r) > i_camp and r[i_camp] is not None else ""
        titulo = str(r[i_titulo]) if i_titulo is not None and len(r) > i_titulo and r[i_titulo] is not None else ""
        out.append(PublicidadeRow(
            desde=desde, ate=ate, mlb=mlb_up,
            campanha=campanha, titulo=titulo, investimento=inv,
        ))
    return out
