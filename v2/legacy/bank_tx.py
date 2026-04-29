"""Bank-statement transaction parser + classifier.

Port of `_admin/upload_page._classify_bank_transactions` — CSV sniffer for
extrato_mp/extrato_nubank/extrato_c6_* plus per-row `classify_transaction`
invocation. Returns a plain list of dicts so API serialization is trivial.
"""
from __future__ import annotations

import hashlib
import io
import re
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
    "certification",  # INMETRO / Anatel / product certification fees
    "rent", "utilities", "software", "bank_fee", "fx", "loan",
    "investment", "refund", "dividends", "personal", "uncategorized",
]


def looks_like_pdf(file_bytes: bytes) -> bool:
    """Detect PDF by magic bytes. PDF parsing is not supported yet — caller
    should return a friendly error instead of silently producing empty rows."""
    return file_bytes[:5] == b"%PDF-" if file_bytes else False


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


# ── MP transaction classification (#4) ──────────────────────────────────────
#
# Mercado Pago's CSV stores `TRANSACTION_TYPE` for each row. We split rows
# into "internal_ml" (transfers within MP/ML — refunds, mediations, etc.)
# and "external" (real outflows to other banks/parties — PIX out, withdrawals,
# bank transfers, boletos paid). UI renders external as a separate block
# because that's where the actionable spend lives.

_MP_INTERNAL_TYPES = {
    # received from buyers / income side — already filtered upstream by `val > 0`,
    # but listed here for completeness
    "payment_account_money", "payment_credit_card", "payment_debit_card",
    "payment", "purchase",
    # ML internal money flows
    "refund", "refund_payment",
    "chargeback", "chargeback_partial",
    "mediation_settlement", "dispute",
    "transfer_in",
    "fee", "shipping_fee", "marketplace_fee",
    "tax_collection", "ml_payout_fee",
    "release", "release_funds", "money_release",
    "liberacao", "liberação",
}

_MP_EXTERNAL_TYPES = {
    "transfer_out", "bank_transfer", "withdrawal",
    "cash_out", "money_out",
    "pix", "pix_out", "pix_payment", "pix_transfer",
    "boleto", "boleto_payment", "pagamento_boleto",
    "ted", "ted_out", "doc_out",
    "p2p_transfer",
    "bill_payment", "utility_payment",
}

_MP_EXTERNAL_DESC_HINTS = (
    "pix", "transferência", "transferencia", "ted ", "doc ", "boleto",
    "pagamento de conta", "saque", "withdrawal",
)


def _classify_mp_tx(transaction_type: str, description: str) -> str:
    """Return one of 'internal_ml' / 'external' / 'unknown'."""
    tt = (transaction_type or "").strip().lower()
    if tt in _MP_EXTERNAL_TYPES:
        return "external"
    if tt in _MP_INTERNAL_TYPES:
        return "internal_ml"
    desc_l = (description or "").lower()
    for hint in _MP_EXTERNAL_DESC_HINTS:
        if hint in desc_l:
            return "external"
    return "unknown"


# ── Stable transaction hash (for #1 group-by-bank dedupe) ───────────────────


def compute_tx_hash(source_key: str, date_iso: str, value_brl: float, description: str) -> str:
    """Stable identifier across re-uploads of the same file. Override storage
    keys are bound to this hash, so re-uploading does NOT lose categorizations.

    Components (canonicalized):
      - source_key (so the same hash collisions across banks are rejected)
      - date (first 10 chars to survive different ISO formats)
      - value rounded to cents
      - description, lowercased + whitespace collapsed
    """
    desc_norm = re.sub(r"\s+", " ", (description or "").strip().lower())
    raw = f"{source_key}|{(date_iso or '')[:10]}|{round(float(value_brl or 0), 2):.2f}|{desc_norm}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_bank_tx_bytes(source_key: str, file_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a bank-statement file and return classified transactions.

    Each dict contains: `idx`, `date`, `value_brl`, `description`,
    `category`, `project`, `label`, `confidence`, `auto` (bool).
    MP income/liberação rows are intentionally skipped here — they belong
    in the sales pipeline, not the bank-classification flow.

    Returns [] for unparseable input (incl. PDF — caller should detect
    upstream via `looks_like_pdf()` and surface a friendlier error).
    """
    if looks_like_pdf(file_bytes):
        return []
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

        # #3: skip zero-only rows — Nubank exports include "no movement" lines
        # for posting/scheduling that visually duplicate real transactions.
        if val == 0:
            continue

        # Skip MP income + liberação — tracked elsewhere (load_collection_mp / vendas)
        if source_key == "extrato_mp":
            dl = desc.lower()
            if val > 0 or "liberação" in dl or "liberacao" in dl:
                continue

        date_val = str(row.get(cols["date"], "") or "") if cols["date"] else ""
        cls = classify_transaction(desc, val)
        # #4: classify MP rows as internal vs external relative to the ML ecosystem.
        # For non-MP banks the field stays 'external' (real bank movements always
        # are external), so the UI can use a single condition `tx_class == 'external'`.
        if source_key == "extrato_mp":
            tx_class = _classify_mp_tx(desc, desc)
        else:
            tx_class = "external"

        rows.append({
            "idx": int(idx) if isinstance(idx, int) else len(rows),
            "tx_hash": compute_tx_hash(source_key, date_val, val, desc),
            "date": date_val[:32],
            "value_brl": float(val),
            "description": desc[:200],
            "category": cls["category"],
            "project": cls.get("project") or "",
            "label": cls["label"],
            "confidence": cls.get("confidence", "none"),
            "auto": cls.get("confidence") == "auto",
            "tx_class": tx_class,
        })
    return rows
