"""Armazenagem (Mercado Livre Full storage) CSV parser.

File format:
- Separator: ;
- Lines 1-4: metadata, line 5 (index 4) is the header
- Pairs of rows per SKU:
  - cost row:   R$ values per day + SKU/Produto in identification columns
  - units row:  '0 u' / '8 u' values per day, SKU column empty

Mirrors super-calculator-app/lib/escalar/armazenagem-loader.ts.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Two locations searched in order
_BASE = Path(__file__).resolve().parents[3]  # _admin/api/v2/parsers → _admin
ARMAZENAGEM_DIRS = [
    _BASE.parent / "_data" / "armazenagem",
    _BASE.parent / "vendas",
]


@dataclass
class StorageData:
    sku: str
    mlb: str
    produto: str
    status: str
    tarifa_diaria: float
    custos_acumulados: float
    current_stock: int
    days_in_stock: int
    days_out: int
    total_days: int
    avg_stock: float


def _parse_csv(text: str, sep: str = ";") -> list[list[str]]:
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


def _parse_brl(s: str | None) -> float:
    if s is None:
        return 0.0
    v = s.strip().replace("R$", "").replace(" ", "")
    if not v or v == "-":
        return 0.0
    if "," in v and "." in v:
        v = v.replace(".", "").replace(",", ".")
    elif "," in v:
        v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return 0.0


def _parse_units(s: str | None) -> int:
    if s is None:
        return 0
    m = re.search(r"(\d+)\s*u", s.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _parse_dmy(s: str) -> int:
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", s)
    if not m:
        return 0
    try:
        return int(datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).timestamp())
    except ValueError:
        return 0


def _is_armazenamento_file(filename: str) -> bool:
    if not filename.lower().endswith(".csv"):
        return False
    f = filename.lower()
    if "armazenamento" in f or "armazenagem" in f:
        return True
    return "servic" in f and "armazena" in f


def _parse_text(text: str) -> tuple[int, list[StorageData]]:
    """Source-agnostic parser. Returns (end_date_epoch, rows)."""
    rows = _parse_csv(text)
    if len(rows) < 7:
        return 0, []
    header = rows[4]
    if not header or len(header) < 12:
        return 0, []

    # Find first daily column (matches dd/mm/yyyy)
    daily_start = -1
    for j in range(11, len(header)):
        if re.match(r"^\d{2}/\d{2}/\d{4}$", (header[j] or "").strip()):
            daily_start = j
            break
    if daily_start < 0:
        daily_start = 12

    # Last date column → end_date
    end_date = 0
    for j in range(len(header) - 1, daily_start - 1, -1):
        t = _parse_dmy((header[j] or "").strip())
        if t > 0:
            end_date = t
            break

    total_days = len(header) - daily_start
    result: list[StorageData] = []
    i = 5
    while i < len(rows) - 1:
        r = rows[i]
        sku = (r[3] if len(r) > 3 else "").strip()
        if not sku or sku == "N/A":
            i += 1
            continue
        mlb = (r[4] if len(r) > 4 else "").strip()
        produto = (r[5] if len(r) > 5 else "").strip()
        status = (r[9] if len(r) > 9 else "").strip()
        tarifa = _parse_brl(r[10] if len(r) > 10 else "")
        custos = _parse_brl(r[11] if len(r) > 11 else "")

        units_row = rows[i + 1] if i + 1 < len(rows) else None
        current_stock = 0
        days_in_stock = 0
        total_stock = 0
        actual_days = 0
        if units_row is not None and (len(units_row) <= 3 or not (units_row[3] or "").strip()):
            last_non_empty = -1
            for j in range(daily_start, min(len(units_row), len(header))):
                cell = (units_row[j] or "").strip()
                if not cell:
                    continue
                actual_days += 1
                qty = _parse_units(cell)
                if qty > 0:
                    days_in_stock += 1
                    total_stock += qty
                if "u" in cell:
                    last_non_empty = qty
            current_stock = last_non_empty if last_non_empty >= 0 else 0
            i += 2
        else:
            i += 1

        avg_stock = (total_stock / days_in_stock) if days_in_stock > 0 else 0.0
        td = actual_days if actual_days > 0 else total_days
        days_out = td - days_in_stock

        result.append(StorageData(
            sku=sku, mlb=mlb, produto=produto, status=status,
            tarifa_diaria=tarifa, custos_acumulados=custos,
            current_stock=current_stock, days_in_stock=days_in_stock,
            days_out=days_out, total_days=td, avg_stock=avg_stock,
        ))

    return end_date, result


def _read_file(path: Path) -> tuple[int, list[StorageData]]:
    """FS-source shim around `_parse_text`."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0, []
    return _parse_text(text)


def parse_armazenagem_bytes(file_bytes: bytes) -> tuple[int, list[StorageData]]:
    """DB-source shim: decode raw bytes then parse."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
        return _parse_text(text)
    return 0, []


def load_all_armazenagem() -> dict[str, StorageData]:
    """Aggregate across all storage files; keep the FRESHEST file's data per SKU.

    Older files are used only for SKUs the newer file doesn't list at all.
    """
    files: list[tuple[int, list[StorageData]]] = []
    seen_files: set[str] = set()
    for d in ARMAZENAGEM_DIRS:
        if not d.exists():
            continue
        for fn in os.listdir(d):
            if not _is_armazenamento_file(fn):
                continue
            if fn in seen_files:
                continue
            seen_files.add(fn)
            files.append(_read_file(d / fn))
    # Newest first
    files.sort(key=lambda x: x[0], reverse=True)

    merged: dict[str, StorageData] = {}
    for _end_date, rows in files:
        for row in rows:
            if row.sku not in merged:
                merged[row.sku] = row
    return merged
