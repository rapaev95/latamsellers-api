"""Bank-statement transaction parser + classifier.

Port of `_admin/upload_page._classify_bank_transactions` — CSV sniffer for
extrato_mp/extrato_nubank/extrato_c6_* plus per-row `classify_transaction`
invocation. Returns a plain list of dicts so API serialization is trivial.
"""
from __future__ import annotations

import io
from typing import Any, Optional

import pandas as pd

from .config import classify_transaction


# ── BRL amount parser (identical to upload_page._parse_val) ────────────────

def _parse_val(v: Any) -> float:
    """Parse a BRL numeric cell: handles '1.234,56', '-1.234,56', '1234.56', '1234'."""
    if v is None:
        return 0.0
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return 0.0
    # Negative sign can appear as leading '-' or trailing (rare), or as (123,45)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    if s.startswith("-"):
        neg = True
        s = s[1:]
    # BRL format: "1.234,56" → "1234.56"; plain "1234.56" stays.
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return 0.0
    return -val if neg else val


# ── Parser ──────────────────────────────────────────────────────────────────

# Category enum mirrors upload_page.py — must be kept in sync with legacy.config
CATEGORY_OPTIONS = [
    "internal_transfer", "income", "expense", "supplier", "fulfillment",
    "shipping", "ads", "tax", "accounting", "salary", "freelancer",
    "rent", "utilities", "software", "bank_fee", "fx", "loan",
    "investment", "refund", "dividends", "personal", "uncategorized",
]


def _read_bank_csv(source_key: str, file_bytes: bytes) -> Optional[pd.DataFrame]:
    """Sniff CSV structure — extrato_mp has skiprows=3 and ; sep; others vary."""
    if source_key == "extrato_mp":
        for skip in (3, 2, 4):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=";", skiprows=skip, encoding="utf-8")
                if "RELEASE_DATE" in df.columns or "TRANSACTION_TYPE" in df.columns:
                    return df
            except Exception:
                continue
        return None

    # Other extratos: try comma/semicolon, optional skiprows
    for sep in (",", ";"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding="utf-8")
            if len(df.columns) > 2:
                return df
        except Exception:
            continue
    for skip in (5, 8, 10):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=",", skiprows=skip, encoding="utf-8")
            if len(df.columns) > 2:
                return df
        except Exception:
            continue
    return None


def _find_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Locate the description/amount/date columns across BR bank CSV dialects."""
    cols: dict[str, Optional[str]] = {
        "desc": None, "value": None, "entrada": None, "saida": None, "date": None,
    }
    for c in df.columns:
        cl = str(c).lower()
        if cols["desc"] is None and (
            "descri" in cl or "título" in cl or "titulo" in cl or "transaction_type" in cl
        ):
            cols["desc"] = str(c)
        if cols["entrada"] is None and "entrada" in cl:
            cols["entrada"] = str(c)
        if cols["saida"] is None and ("saída" in cl or "saida" in cl):
            cols["saida"] = str(c)
        if cols["value"] is None and (
            ("valor" in cl and "valor do dia" not in cl) or "transaction_net_amount" in cl
        ):
            cols["value"] = str(c)
        if cols["date"] is None and ("data" in cl or "release_date" in cl):
            cols["date"] = str(c)
    return cols


def parse_bank_tx_bytes(source_key: str, file_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a bank-statement file and return classified transactions.

    Each dict contains: `idx`, `date`, `value_brl`, `description`,
    `category`, `project`, `label`, `confidence`, `auto` (bool).
    MP income/liberação rows are intentionally skipped here — they belong
    in the sales pipeline, not the bank-classification flow.
    """
    df = _read_bank_csv(source_key, file_bytes)
    if df is None or df.empty:
        return []

    cols = _find_columns(df)
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        desc = str(row.get(cols["desc"], "") or "") if cols["desc"] else ""
        if cols["entrada"] and cols["saida"]:
            val = _parse_val(row.get(cols["entrada"], 0)) - _parse_val(row.get(cols["saida"], 0))
        elif cols["value"]:
            val = _parse_val(row.get(cols["value"], 0))
        else:
            val = 0.0

        # Skip MP income + liberação — tracked elsewhere (load_collection_mp / vendas)
        if source_key == "extrato_mp":
            dl = desc.lower()
            if val > 0 or "liberação" in dl or "liberacao" in dl:
                continue

        date_val = str(row.get(cols["date"], "") or "") if cols["date"] else ""
        cls = classify_transaction(desc, val)
        rows.append({
            "idx": int(idx) if isinstance(idx, int) else len(rows),
            "date": date_val[:32],
            "value_brl": float(val),
            "description": desc[:200],
            "category": cls["category"],
            "project": cls.get("project") or "",
            "label": cls["label"],
            "confidence": cls.get("confidence", "none"),
            "auto": cls.get("confidence") == "auto",
        })
    return rows
