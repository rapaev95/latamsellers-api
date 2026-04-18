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
from pathlib import Path
from typing import Iterable

VENDAS_DIR = Path(__file__).resolve().parents[3].parent / "vendas"  # …/MERCADO LIVRE/vendas

PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}

COLUMN_NAMES = {
    "saleId": "N.º de venda",
    "date": "Data da venda",
    "units": "Unidades",
    "receita": "Receita por produtos (BRL)",
    "tarifaVenda": "Tarifa de venda e impostos (BRL)",
    "tarifaEnvio": "Tarifas de envio (BRL)",
    "custoTroca": "Custo de envio por troca de produto",
    "cancelamentos": "Cancelamentos e reembolsos (BRL)",
    "total": "Total (BRL)",
    "ads": "Venda por publicidade",
    "sku": "SKU",
    "mlb": "# de anúncio",
    "title": "Título do anúncio",
    "shipMode": "Forma de entrega",
    "status": "Estado",
}


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
    """Accept any CSV in vendas/ that isn't in the deny list of other-format reports.

    Some month files have corrupted filenames (e.g. `marc╠зo 26.csv` from a
    cp866→utf8 round-trip). A name-pattern accept regex was missing them and
    silently dropping a whole month of sales — so this is a deny-only filter.
    The format itself (Mercado Livre Vendas) is detected by the header row.
    """
    if not filename.lower().endswith(".csv"):
        return False
    lower = filename.lower()
    # Other ML-related CSV formats that ship into the same folder
    deny = ("anuncios", "patrocinados", "account_statement", "after_collection",
            "armazenamento", "armazenagem", "relat.")
    return not any(p in lower for p in deny)


def parse_vendas_text(text: str) -> list[VendasRow]:
    """Parse a Vendas ML CSV already decoded to text. Source-agnostic — used by
    both the FS reader (`read_vendas_file`) and the DB reader (bytes → text)."""
    rows = parse_csv(text)
    if len(rows) < 7:
        return []
    header = rows[5]

    def find_idx(name: str) -> int:
        for i, h in enumerate(header):
            if (h or "").strip() == name:
                return i
        return -1

    ix = {k: find_idx(v) for k, v in COLUMN_NAMES.items()}
    if ix["sku"] < 0 or ix["units"] < 0:
        return []

    out: list[VendasRow] = []
    for r in range(6, len(rows)):
        row = rows[r]
        if ix["sku"] >= len(row):
            continue
        sku = (row[ix["sku"]] or "").strip()
        if not sku:
            continue
        units_raw = (row[ix["units"]] or "").strip() if ix["units"] < len(row) else ""
        try:
            units = int(units_raw or "0")
        except ValueError:
            units = 0
        if units <= 0:
            continue

        def cell(k: str) -> str:
            j = ix[k]
            if j < 0 or j >= len(row):
                return ""
            return row[j] or ""

        out.append(VendasRow(
            sale_id=cell("saleId").strip(),
            date_ms=parse_pt_date(cell("date").strip()),
            sku=sku,
            mlb=cell("mlb").strip(),
            title=cell("title").strip(),
            units=units,
            receita=parse_brl(cell("receita")),
            tarifa_venda=abs(parse_brl(cell("tarifaVenda"))),
            tarifa_envio=abs(parse_brl(cell("tarifaEnvio"))),
            custo_troca=abs(parse_brl(cell("custoTroca"))),
            cancelamentos=abs(parse_brl(cell("cancelamentos"))),
            total=parse_brl(cell("total")),
            ads=cell("ads").strip().lower() == "sim",
            ship_mode=cell("shipMode").strip(),
            status=cell("status").strip(),
        ))
    return out


def read_vendas_file(path: Path) -> list[VendasRow]:
    """FS-source shim around `parse_vendas_text`. Kept for backward compat."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return parse_vendas_text(text)


def parse_vendas_bytes(file_bytes: bytes) -> list[VendasRow]:
    """DB-source shim: decode raw bytes then parse. Silently skips undecodable."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
        return parse_vendas_text(text)
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
