"""Parser for DAS Simples Nacional PDF (PGDASD).

DAS = «Documento de Arrecadação do Simples Nacional» — monthly tax slip the
Brazilian Receita issues for Simples-regime companies. Comes as encrypted PDF;
encryption is handled upstream by `legacy/unlocker.try_unlock` so this parser
gets the decrypted bytes.

Port of `_admin/reports.py:parse_das_pdf` — same regexes, same return shape,
but bytes-in / dict-out instead of file_path-in (we read from `uploads.file_bytes`,
not the filesystem).

Returns:
    {
      "month": "Fevereiro/2026",         # human-readable
      "month_iso": "2026-02",            # for sorting / cache keys
      "total": 1234.56,                  # «Valor Total do Documento»
      "vencimento": "20/03/2026",        # due date
      "irpj": 0.0, "csll": 0.0, "cofins": 0.0,
      "pis": 0.0,  "inss": 0.0, "iss":   0.0, "icms": 0.0,
      # optional flags when text layer is missing:
      "needs_manual_total": True, "no_text_layer": True,
    }
or None if neither text nor filename yielded a parseable period.
"""
from __future__ import annotations

import io
import re
from typing import Any, Optional


_MONTH_NAMES_PT_NUM: dict[str, str] = {
    "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
    "abril": "04", "maio": "05", "junho": "06", "julho": "07",
    "agosto": "08", "setembro": "09", "outubro": "10",
    "novembro": "11", "dezembro": "12",
}

_MONTH_NUM_TO_NAMES_PT: dict[str, str] = {
    "01": "Janeiro", "02": "Fevereiro", "03": "Março", "04": "Abril",
    "05": "Maio", "06": "Junho", "07": "Julho", "08": "Agosto",
    "09": "Setembro", "10": "Outubro", "11": "Novembro", "12": "Dezembro",
}


# Receita Federal tax codes that show up on a Simples Nacional DAS.
# Slip line example: "1001 IRPJ - SIMPLES NACIONAL ... 573,77"
_TAX_CODES: dict[str, str] = {
    "1001": "irpj",
    "1002": "csll",
    "1004": "cofins",
    "1005": "pis",
    "1006": "inss",
    "1010": "iss",
    "1007": "icms",
}


def _period_from_filename(filename: str) -> Optional[str]:
    """Recover YYYY-MM from a PGDASD-shaped filename.

    The Receita's own export names are like
        `PGDASD-DAS-04032026202602001.pdf`
                          ^^^^^^
                      year+month (YYYYMM) of the period.
    """
    m = re.search(r"PGDASD-DAS-\d{8}(\d{6})\d+", filename)
    if not m:
        return None
    yyyymm = m.group(1)
    return f"{yyyymm[:4]}-{yyyymm[4:6]}"


def parse_das_simples_bytes(file_bytes: bytes, filename: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Bytes-in DAS PDF parser.

    Yields total + per-tax breakdown + period + vencimento. Falls back to
    filename-derived period when the PDF has no text layer (rare, but happens
    when the user re-prints from a viewer that flattens text into images).
    """
    try:
        import pdfplumber
    except ImportError:
        return None

    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        return None

    fname = filename or ""
    filename_period = _period_from_filename(fname) if fname else None

    if not text:
        # No text layer — best we can do is the filename period. Caller (UI)
        # should prompt the user to enter the total manually.
        if filename_period:
            year_num, month_num = filename_period.split("-")
            return {
                "month": f"{_MONTH_NUM_TO_NAMES_PT.get(month_num, month_num)}/{year_num}",
                "month_iso": filename_period,
                "total": 0.0,
                "vencimento": None,
                "irpj": 0.0, "csll": 0.0, "cofins": 0.0,
                "pis": 0.0, "inss": 0.0, "iss": 0.0, "icms": 0.0,
                "needs_manual_total": True,
                "no_text_layer": True,
            }
        return None

    result: dict[str, Any] = {
        "month": None,
        "month_iso": None,
        "total": 0.0,
        "vencimento": None,
        "irpj": 0.0, "csll": 0.0, "cofins": 0.0,
        "pis": 0.0, "inss": 0.0, "iss": 0.0, "icms": 0.0,
    }

    # Period: «Fevereiro/2026»
    period_match = re.search(
        r"(Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro)/(\d{4})",
        text, re.IGNORECASE,
    )
    if period_match:
        month_name = period_match.group(1).lower()
        year = period_match.group(2)
        result["month"] = f"{period_match.group(1).capitalize()}/{year}"
        month_num = _MONTH_NAMES_PT_NUM.get(month_name, "00")
        result["month_iso"] = f"{year}-{month_num}"
    elif filename_period:
        # Period missing in PDF body but filename has it — use that.
        year_num, month_num = filename_period.split("-")
        result["month"] = f"{_MONTH_NUM_TO_NAMES_PT.get(month_num, month_num)}/{year_num}"
        result["month_iso"] = filename_period

    # Vencimento — first DD/MM/YYYY in text. DAS prints the due date
    # prominently near the top, so first match is the right one.
    venc_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if venc_match:
        result["vencimento"] = venc_match.group(1)

    # «Valor Total do Documento  573,77»
    total_match = re.search(r"Valor Total do Documento\s*([\d.,]+)", text)
    if total_match:
        try:
            result["total"] = float(total_match.group(1).replace(".", "").replace(",", "."))
        except ValueError:
            pass

    # Per-tax breakdown — line shape «1001 IRPJ - SIMPLES NACIONAL ... 573,77».
    # The amount is the last BR-formatted decimal on the same logical line.
    for code, key in _TAX_CODES.items():
        pattern = rf"{code}\s+\w+[^\n]*?(\d{{1,3}}(?:\.\d{{3}})*,\d{{2}})"
        m = re.search(pattern, text)
        if m:
            try:
                result[key] = float(m.group(1).replace(".", "").replace(",", "."))
            except ValueError:
                pass

    # If the «Valor Total» line wasn't matched but the breakdown was,
    # synthesize the total from the parts (legacy parity).
    if result["total"] == 0.0:
        result["total"] = round(sum(
            result[k] for k in ("irpj", "csll", "cofins", "pis", "inss", "iss", "icms")
        ), 2)

    return result if result["total"] > 0 else None
