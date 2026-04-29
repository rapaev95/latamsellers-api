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
    """Detect PDF by magic bytes."""
    return file_bytes[:5] == b"%PDF-" if file_bytes else False


# ── PDF parsers (per-bank, pdfplumber-based) ────────────────────────────────


_PT_MONTH_TO_NUM = {
    "jan": "01", "janeiro": "01",
    "fev": "02", "fevereiro": "02",
    "mar": "03", "março": "03", "marco": "03",
    "abr": "04", "abril": "04",
    "mai": "05", "maio": "05",
    "jun": "06", "junho": "06",
    "jul": "07", "julho": "07",
    "ago": "08", "agosto": "08",
    "set": "09", "setembro": "09",
    "out": "10", "outubro": "10",
    "nov": "11", "novembro": "11",
    "dez": "12", "dezembro": "12",
}


def _parse_period_year_from_pdf_text(text: str, default_year: str = "2026") -> str:
    """Pull the statement year from the «Período» header line."""
    import re as _re
    m = _re.search(r"per[ií]odo[^\n]*?(20\d{2})", text, _re.IGNORECASE)
    if m:
        return m.group(1)
    years = _re.findall(r"20\d{2}", text)
    return years[-1] if years else default_year


def _ddmm_to_iso(date_ddmm: str, year: str) -> str:
    """'14/04' + '2026' → '2026-04-14'. Bad input → original string."""
    parts = date_ddmm.split("/")
    if len(parts) == 2 and len(year) == 4:
        dd, mm = parts
        return f"{year}-{mm.zfill(2)}-{dd.zfill(2)}"
    return date_ddmm


def _parse_c6_usd_pdf(file_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a C6 Conta Global USD PDF statement into transaction rows.

    Patterns covered (extracted from real C6 USD exports):
      «DD/MM Débito de cartão <merchant> -US$ X.XX»          → expense
      «DD/MM Compra <merchant> -US$ X.XX»                    → expense
      «DD/MM Entrada Transf C6 Conta Global Líquido US$ X.XX» → entry
      «DD/MM Saque US$ X.XX»                                 → outflow
      «DD/MM Tarifa <X> -US$ X.XX»                            → bank_fee

    Year is taken from the «Período» line in the PDF header.
    Returns rows in the same shape as `parse_bank_tx_bytes` (no header dict).
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    try:
        all_lines: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                all_lines.extend(txt.split("\n"))
    except Exception:
        return []

    if not all_lines:
        return []

    full_text = "\n".join(all_lines)
    year = _parse_period_year_from_pdf_text(full_text)

    pat_amount = r"-?US\$\s*([\d.,]+)"

    # Order matters — broader patterns last so specific labels win.
    patterns = [
        # (regex, sign, default_desc)
        (re.compile(r"(\d{2}/\d{2})\s+Débito de cartão\b\s*(.*?)" + pat_amount, re.IGNORECASE), -1, "Débito de cartão"),
        (re.compile(r"(\d{2}/\d{2})\s+Compra\b\s*(.*?)" + pat_amount, re.IGNORECASE), -1, "Compra"),
        (re.compile(r"(\d{2}/\d{2})\s+Saque\b\s*(.*?)" + pat_amount, re.IGNORECASE), -1, "Saque"),
        (re.compile(r"(\d{2}/\d{2})\s+Tarifa\b\s*(.*?)" + pat_amount, re.IGNORECASE), -1, "Tarifa"),
        (re.compile(r"(\d{2}/\d{2})\s+Entrada\s+(.*?)" + pat_amount, re.IGNORECASE), +1, "Entrada"),
        (re.compile(r"(\d{2}/\d{2})\s+Estorno\b\s*(.*?)" + pat_amount, re.IGNORECASE), +1, "Estorno"),
    ]

    rows: list[dict[str, Any]] = []
    for i, raw_line in enumerate(all_lines):
        line = raw_line.strip()
        if not line:
            continue
        for rgx, sign, default_desc in patterns:
            m = rgx.search(line)
            if not m:
                continue
            ddmm = m.group(1)
            mid = (m.group(2) or "").strip(" -·•")
            amt_raw = m.group(3)
            try:
                amount = float(amt_raw.replace(".", "").replace(",", "."))
            except ValueError:
                break

            if amount == 0:
                break

            # Pick a clean description: prefer captured merchant text, fall back to label
            desc = (mid or default_desc).strip()
            # Some PDFs split the merchant name onto the next line; if the
            # captured text is empty/short, peek 1 line ahead for context.
            if (not desc or len(desc) < 4) and i + 1 < len(all_lines):
                nxt = all_lines[i + 1].strip()
                if nxt and not re.search(r"\d{2}/\d{2}", nxt):
                    desc = (default_desc + " · " + nxt).strip()

            iso_date = _ddmm_to_iso(ddmm, year)
            val = sign * amount
            cls = classify_transaction(desc, val)
            rows.append({
                "idx": len(rows),
                "tx_hash": compute_tx_hash("extrato_c6_usd", iso_date, val, desc),
                "date": iso_date,
                "value_brl": val,        # USD here despite the field name; UI labels accordingly
                "description": desc[:200],
                "category": cls["category"],
                "project": cls.get("project") or "",
                "label": cls["label"],
                "confidence": cls.get("confidence", "none"),
                "auto": cls.get("confidence") == "auto",
                "tx_class": "external",  # all bank-statement movements are external by definition
            })
            break  # first matching pattern wins for this line

    return rows


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
    """Locate the description/amount/date columns across BR bank CSV dialects.

    Recognised header tokens (case-insensitive substring match unless noted):
      desc:    descri / título / titulo / transaction_type / title (English)
      value:   valor / transaction_net_amount / amount (English)
      entrada: entrada
      saída:   saída / saida
      date:    data / release_date / exact 'date' (English)
    """
    cols: dict[str, Optional[str]] = {
        "desc": None, "value": None, "entrada": None, "saida": None, "date": None,
    }
    for c in df.columns:
        cl = str(c).strip().lower()
        if cols["desc"] is None and (
            "descri" in cl or "título" in cl or "titulo" in cl
            or "transaction_type" in cl or cl == "title"
        ):
            cols["desc"] = str(c)
        if cols["entrada"] is None and "entrada" in cl:
            cols["entrada"] = str(c)
        if cols["saida"] is None and ("saída" in cl or "saida" in cl):
            cols["saida"] = str(c)
        if cols["value"] is None and (
            ("valor" in cl and "valor do dia" not in cl)
            or "transaction_net_amount" in cl
            or cl == "amount"
        ):
            cols["value"] = str(c)
        if cols["date"] is None and ("data" in cl or "release_date" in cl or cl == "date"):
            cols["date"] = str(c)
    return cols


def _is_nubank_credit_card_csv(df: pd.DataFrame) -> bool:
    """Detect Nubank's credit-card export (English headers, sign-inverted).

    Format: `date,title,amount` where positive = charge, negative = payment-in.
    Different from the Nubank checking-account CSV which uses Portuguese headers.
    """
    headers = {str(c).strip().lower() for c in df.columns}
    return {"date", "title", "amount"}.issubset(headers)


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

    Returns [] for unparseable input (incl. PDFs from banks where the layout
    isn't yet handled). Caller can still detect "is the file a PDF?" via
    `looks_like_pdf()` to surface a friendlier error.
    """
    if looks_like_pdf(file_bytes):
        # Per-bank PDF parsers (extend as more layouts get covered)
        if source_key == "extrato_c6_usd":
            return _parse_c6_usd_pdf(file_bytes)
        return []
    df = _read_bank_csv(source_key, file_bytes)
    if df is None or df.empty:
        return []

    cols = _find_columns(df)
    # Nubank credit-card CSV uses inverted sign convention (charge=positive,
    # payment-in=negative). Flip so our app's "value < 0 = outflow" holds.
    flip_sign = _is_nubank_credit_card_csv(df)

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        desc = str(row.get(cols["desc"], "") or "") if cols["desc"] else ""
        if cols["entrada"] and cols["saida"]:
            val = _parse_val(row.get(cols["entrada"], 0)) - _parse_val(row.get(cols["saida"], 0))
        elif cols["value"]:
            val = _parse_val(row.get(cols["value"], 0))
        else:
            val = 0.0

        if flip_sign:
            val = -val

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
