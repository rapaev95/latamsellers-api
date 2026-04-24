"""Vendas ML CSV parser.

Format (Mercado Livre "Relatório de vendas"):
- Separator: ;
- Lines 1–5: metadata, line 6 (index 5) is the header
- BRL numbers: "1.234,56" or "78,2"
- Date: "24 de março de 2026 22:36 hs."

Mirrors super-calculator-app/lib/escalar/vendas-loader.ts.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

import unicodedata

from v2.parsers.text_cells import excel_scalar_to_clean_str

VENDAS_DIR = Path(__file__).resolve().parents[3].parent / "vendas"  # …/MERCADO LIVRE/vendas

PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}

# ML exports sometimes change punctuation (N.º vs Nº) or encoding of ó/ã.
# First matching column left-to-right wins (important when «Unidades» repeats).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "saleId": ("N.º de venda", "Nº de venda", "Número de venda", "Numero de venda"),
    "date": ("Data da venda",),
    "units": ("Unidades",),
    "receita": ("Receita por produtos (BRL)",),
    "tarifaVenda": ("Tarifa de venda e impostos (BRL)",),
    "tarifaEnvio": ("Tarifas de envio (BRL)",),
    "custoTroca": ("Custo de envio por troca de produto",),
    "cancelamentos": ("Cancelamentos e reembolsos (BRL)",),
    "total": ("Total (BRL)",),
    "ads": ("Venda por publicidade",),
    "sku": ("SKU",),
    "mlb": ("# de anúncio", "# de anuncio", "# de publicación", "# de publicacion"),
    "title": ("Título do anúncio", "Titulo do anuncio"),
    "shipMode": ("Forma de entrega",),
    "status": ("Estado",),
}


def _norm_header(s: str) -> str:
    t = unicodedata.normalize("NFKC", (s or "").strip().strip("\ufeff")).lower()
    return " ".join(t.split())


def _indices_from_header_cells(cells: list[str]) -> dict[str, int]:
    """Map logical keys → 0-based column index (first header match per key)."""
    out: dict[str, int] = {}
    for logical, aliases in COLUMN_ALIASES.items():
        want = {_norm_header(a) for a in aliases}
        for i, h in enumerate(cells):
            if _norm_header(h) in want:
                out[logical] = i
                break
    return out


def _parse_int_units(raw: str | float | int | None) -> int:
    if raw is None:
        return 0
    if isinstance(raw, float):
        if raw != raw:  # NaN
            return 0
        try:
            return int(raw)
        except (ValueError, OverflowError):
            return 0
    s = str(raw).strip().replace(",", ".")
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _vendas_row_from_indices(
    ix: dict[str, int],
    get_cell: Callable[[str], str],
) -> VendasRow | None:
    sku = get_cell("sku").strip()
    if not sku:
        return None
    units = _parse_int_units(get_cell("units"))
    if units <= 0:
        return None

    def cell(k: str) -> str:
        return get_cell(k).strip()

    return VendasRow(
        sale_id=cell("saleId"),
        date_ms=parse_pt_date(cell("date")),
        sku=sku,
        mlb=excel_scalar_to_clean_str(cell("mlb")),
        title=cell("title"),
        units=units,
        receita=parse_brl(cell("receita")),
        tarifa_venda=abs(parse_brl(cell("tarifaVenda"))),
        tarifa_envio=abs(parse_brl(cell("tarifaEnvio"))),
        custo_troca=abs(parse_brl(cell("custoTroca"))),
        cancelamentos=abs(parse_brl(cell("cancelamentos"))),
        total=parse_brl(cell("total")),
        ads=cell("ads").lower() == "sim",
        ship_mode=cell("shipMode"),
        status=cell("status"),
    )


@dataclass
class VendasRow:
    sale_id: str
    date_ms: int  # milliseconds since epoch (0 if unparseable)
    sku: str
    mlb: str
    title: str
    units: int
    receita: float
    tarifa_venda: float
    tarifa_envio: float
    custo_troca: float
    cancelamentos: float
    total: float
    ads: bool
    ship_mode: str
    status: str


def parse_pt_date(s: str) -> int:
    """Parse 'DD de MONTH de YYYY HH:MM hs.' → ms epoch (0 if fails)."""
    if not s:
        return 0
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})(?:\s+(\d{1,2}):(\d{2}))?", s, re.IGNORECASE)
    if not m:
        return 0
    day = int(m.group(1))
    mon = PT_MONTHS.get(m.group(2).lower())
    if not mon:
        return 0
    year = int(m.group(3))
    hour = int(m.group(4)) if m.group(4) else 0
    minute = int(m.group(5)) if m.group(5) else 0
    try:
        return int(datetime(year, mon, day, hour, minute).timestamp() * 1000)
    except ValueError:
        return 0


def parse_brl(s: str | None) -> float:
    if s is None:
        return 0.0
    v = s.strip().replace("R$", "").replace(" ", "")
    if not v:
        return 0.0
    if "," in v and "." in v:
        v = v.replace(".", "").replace(",", ".")
    elif "," in v:
        v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return 0.0


def parse_csv(text: str, sep: str = ";") -> list[list[str]]:
    """RFC-4180-ish parser: quoted fields with embedded ";" or newlines, "" escape."""
    rows: list[list[str]] = []
    row: list[str] = []
    field = ""
    in_quotes = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_quotes:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field += '"'
                    i += 1
                else:
                    in_quotes = False
            else:
                field += c
        else:
            if c == '"':
                in_quotes = True
            elif c == sep:
                row.append(field)
                field = ""
            elif c == "\n":
                row.append(field)
                rows.append(row)
                row = []
                field = ""
            elif c == "\r":
                pass
            else:
                field += c
        i += 1
    if field or row:
        row.append(field)
        rows.append(row)
    return rows


def is_vendas_ml_file(filename: str) -> bool:
    """Accept CSV/XLSX in vendas/ that isn't in the deny list of other-format reports.

    Some month files have corrupted filenames (e.g. `marc╠зo 26.csv` from a
    cp866→utf8 round-trip). A name-pattern accept regex was missing them and
    silently dropping a whole month of sales — so this is a deny-only filter.
    The format itself (Mercado Livre Vendas) is detected by the header row.
    """
    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        return False
    lower = filename.lower()
    # Other ML-related CSV formats that ship into the same folder
    deny = ("anuncios", "patrocinados", "account_statement", "after_collection",
            "armazenamento", "armazenagem", "relat.")
    return not any(p in lower for p in deny)


def parse_vendas_dataframe(df: Any) -> list[VendasRow]:
    """Build VendasRow list from a pandas DataFrame whose columns are ML headers."""
    try:
        import pandas as pd
    except ImportError:
        return []
    if df is None or df.empty:
        return []
    header_cells = [str(c) for c in df.columns]
    ix = _indices_from_header_cells(header_cells)
    if "sku" not in ix or "units" not in ix:
        return []

    out: list[VendasRow] = []
    for _, ser in df.iterrows():
        ncols = len(ser)

        def get_cell(k: str) -> str:
            j = ix.get(k, -1)
            if j < 0 or j >= ncols:
                return ""
            v = ser.iloc[j]
            if pd.isna(v):
                return ""
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v).strip()

        vr = _vendas_row_from_indices(ix, get_cell)
        if vr is not None:
            out.append(vr)
    return out


def _try_pandas_ml_vendas_blob(filename: str, file_bytes: bytes) -> Any:
    """Same heuristics as `v2.legacy.reports._parse_one_vendas_file` (finance preview).

    Escalar used to call only the strict RFC-style CSV parser; ML monthly files
    and some XLSX exports load here when that parser returns empty.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    buf = BytesIO(file_bytes)
    fn = (filename or "").lower()
    if fn.endswith(".csv"):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            buf.seek(0)
            try:
                df = pd.read_csv(
                    buf,
                    sep=";",
                    skiprows=5,
                    encoding=enc,
                    low_memory=False,
                    decimal=",",
                    thousands=".",
                )
                if "Estado" in df.columns or "N.º de venda" in df.columns:
                    return df
            except Exception:
                continue
        return None
    if fn.endswith((".xlsx", ".xls")):
        for _h in (5, 6, 4, 7):
            buf.seek(0)
            try:
                df = pd.read_excel(buf, sheet_name=0, header=_h)
                if "Estado" in df.columns or "N.º de venda" in df.columns:
                    return df
            except Exception:
                continue
        return None
    return None


def parse_vendas_xlsx_bytes(file_bytes: bytes) -> list[VendasRow]:
    """Mercado Livre «Relatório de vendas» as .xlsx (skiprows=5 first)."""
    try:
        import pandas as pd
    except ImportError:
        return []
    try:
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl", skiprows=5, header=0)
    except Exception:
        return []
    return parse_vendas_dataframe(df)


def parse_vendas_text(text: str) -> list[VendasRow]:
    """Parse a Vendas ML CSV already decoded to text. Source-agnostic — used by
    both the FS reader (`read_vendas_file`) and the DB reader (bytes → text)."""
    rows = parse_csv(text)
    if len(rows) < 7:
        return []
    header = rows[5]
    ix = _indices_from_header_cells(header)
    if "sku" not in ix or "units" not in ix:
        return []

    out: list[VendasRow] = []
    for r in range(6, len(rows)):
        row = rows[r]

        def get_cell(k: str) -> str:
            j = ix.get(k, -1)
            if j < 0 or j >= len(row):
                return ""
            return (row[j] or "").strip()

        vr = _vendas_row_from_indices(ix, get_cell)
        if vr is not None:
            out.append(vr)
    return out


def read_vendas_file(path: Path) -> list[VendasRow]:
    """FS-source: CSV (UTF-8 text) or XLSX (openpyxl)."""
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        try:
            data = path.read_bytes()
        except OSError:
            return []
        return parse_vendas_xlsx_bytes(data)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return parse_vendas_text(text)


def parse_vendas_bytes(file_bytes: bytes, filename: str | None = None) -> list[VendasRow]:
    """Decode vendas ML from uploaded bytes (CSV text or XLSX ZIP).

    `filename` enables a pandas fallback aligned with `/finance/uploads/.../preview`
    when the strict line-6 CSV parser yields no rows (common for some ML exports).
    """
    if len(file_bytes) >= 4 and file_bytes[:2] == b"PK":
        xrows = parse_vendas_xlsx_bytes(file_bytes)
        if xrows:
            return xrows
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
        rows = parse_vendas_text(text)
        if rows:
            return rows
    if filename:
        df = _try_pandas_ml_vendas_blob(filename, file_bytes)
        if df is not None and not df.empty:
            alt = parse_vendas_dataframe(df)
            if alt:
                return alt
    return []


def shipping_mode_key(raw: str) -> str | None:
    s = (raw or "").lower()
    if "full" in s:
        return "fulfillment"
    if "flex" in s or "xd_drop" in s:
        return "xd_drop_off"
    if "dbs" in s or "próprio" in s or "proprio" in s:
        return "self"
    return "fulfillment" if raw else None


def list_vendas_files() -> list[Path]:
    if not VENDAS_DIR.exists():
        return []
    return sorted([VENDAS_DIR / f for f in os.listdir(VENDAS_DIR) if is_vendas_ml_file(f)])


def load_all_vendas() -> Iterable[VendasRow]:
    """Yield deduplicated rows across every Vendas ML file in the vendas/ folder.

    Snapshot files overlap with monthly ones — dedupe by sale_id.
    """
    seen: set[str] = set()
    for f in list_vendas_files():
        for row in read_vendas_file(f):
            if row.sale_id and row.sale_id in seen:
                continue
            if row.sale_id:
                seen.add(row.sale_id)
            yield row
