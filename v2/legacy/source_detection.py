"""Source-key detection for uploaded files.

Two strategies, run in order:
  1. Filename pattern match — port of `_admin/upload_page.auto_detect_source:358-400`.
  2. Column signature sniff on the first few rows (CSV only, XLSX is more costly
     and rare for Monthly vendas exports) — port of the same function's
     column-detection branch at lines 329-356.

Returned keys must match entries in `v2/legacy/config.DATA_SOURCES`.
"""
from __future__ import annotations

import io
from typing import Optional


def detect_source(filename: str, file_bytes: bytes = b"") -> Optional[str]:
    """Resolve source_key using filename first, then column / PDF sniff."""
    hit = detect_source_from_filename(filename)
    if hit:
        return hit
    if file_bytes:
        # PDFs first — the column sniff is CSV-only and would return None.
        if (filename or "").lower().endswith(".pdf"):
            pdf_hit = detect_source_from_pdf_content(file_bytes)
            if pdf_hit:
                return pdf_hit
        return detect_source_from_columns(filename, file_bytes)
    return None


def detect_source_from_pdf_content(file_bytes: bytes) -> Optional[str]:
    """Peek inside a PDF to find a source_key hint.

    Handles two common cases where the filename gives nothing useful:
      - NFS-e PDFs named after the 50-digit chave de acesso
        (e.g. `35097002245659520000149...pdf`).
      - DAS Simples PDFs renamed by accountants to arbitrary names.

    Reads only the first page text to keep this cheap on the upload path.
    """
    try:
        import pdfplumber
        from io import BytesIO
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return None
            text = (pdf.pages[0].extract_text() or "")
    except Exception:  # noqa: BLE001 — fall through to "ambiguous" prompt
        return None
    if not text:
        return None
    blob = text.lower()
    if "nfs-e" in blob or "nfse" in blob or "nota fiscal de servi" in blob:
        return "nfse_shps"
    if (
        "pgdasd" in blob
        or "documento de arrecada" in blob
        or ("simples nacional" in blob and "das" in blob)
    ):
        return "das_simples"
    return None


def detect_source_from_filename(filename: str) -> Optional[str]:
    """Return the matching `source_key` for a given filename, or None if unknown.

    Brazilian ML downloads come in two shapes — with spaces (`Relatorio Tarifas
    Full Abril.xlsx`, the default in the web UI) and with underscores (the
    same file after a rename / API export). Lowercased name is normalised
    so checks like `"tarifas_full" in fname` match both.
    """
    if not filename:
        return None
    fname = filename.lower().replace(" ", "_")

    # Vendas / order-feed reports (check before "mercado" so Account Statement doesn't win)
    if ("vendas" in fname or "ventas" in fname) and ("mercado" in fname or "mercado_li" in fname):
        return "vendas_ml"

    # Mercado Pago activity
    if "collection" in fname:
        return "collection_mp"
    if "account_statement" in fname:
        return "extrato_mp"

    # Ads — ML changed the report filename convention in 2026:
    #   old: «Relatorio_anuncios_patrocinados_*.xlsx»  (Padrão)
    #   new: «report-pads_report-<advertiserId>-<reportId>.xlsx»  (Padrão)
    #   new: «report-sales_report-<...>.xlsx»  (Por anúncios vendidos)
    # The sales-report shape lacks the «Investimento» column we need, so
    # we tag it explicitly so callers can refuse it with a useful error.
    if "anuncios" in fname or "patrocinados" in fname or "publicidade" in fname:
        return "ads_publicidade"
    if "report-pads_report" in fname or fname.startswith("report-pads"):
        return "ads_publicidade"
    if "report-sales_report" in fname or fname.startswith("report-sales"):
        return "ads_publicidade"  # let the parser fail with a clearer reason

    # ML fiscal catalog — SKU × MLB × NCM × Origem × Custo
    if "dados_fiscais" in fname or "dados-fiscais" in fname:
        return "dados_fiscais"

    # Logistics / inventory
    # Retirada Full: вывоз/утилизация со склада (отдельно от armazenagem,
    # хотя оба отчёта называются "Tarifas Full" в кабинете ML).
    # Проверяем ДО armazenagem — у retirada-файла имя `Relatorio_Tarifas_Full_*`,
    # а у armazenagem `Custos_por_servico_armazenamento_*` или `armazenagem_full*`.
    if "tarifas_full" in fname or "relatorio_tarifas_full" in fname or "retirada" in fname:
        return "retirada_full"
    if "armazenamento" in fname or "armazenagem" in fname:
        return "armazenagem_full"
    if "stock_general" in fname or "stock_full" in fname or fname.startswith("stock"):
        return "stock_full"
    if "after_collection" in fname or "pos_vendas" in fname:
        return "after_collection"
    if "full_express" in fname or "fullexpress" in fname:
        return "full_express"

    # Invoicing
    if "fatura" in fname or "faturamento" in fname:
        return "fatura_ml"

    # Bank statements
    # Nubank credit-card export: `Nubank_<YYYY-MM-DD>.csv`.
    # Nubank conta corrente export: `NU_<account>_<period>.csv` (e.g.
    # `NU_621252515_31MAR2026_29ABR2026.csv`) — no "nubank" word in name.
    # Both share extrato_nubank source_key.
    import re as _re
    if "nubank" in fname or _re.match(r"^nu_\d+_", fname):
        return "extrato_nubank"
    if "c6" in fname:
        if "usd" in fname or "conta_global_usd" in fname:
            return "extrato_c6_usd"
        return "extrato_c6_brl"
    if fname.startswith("01k") and (fname.endswith(".csv") or fname.endswith(".pdf")):
        return "extrato_c6_brl"

    # Ad networks / exchanges
    if "trafficstars" in fname or "traffic" in fname:
        return "trafficstars"
    if "bybit" in fname:
        return "bybit_history"

    # Tax / invoice docs
    if "pgdasd" in fname or "das-" in fname or ("das" in fname and "simples" in fname):
        return "das_simples"
    if "nfs" in fname or "nfse" in fname or fname.startswith("nf "):
        return "nfse_shps"

    # NFS-e chave de acesso — 44 or 50 digit numeric .pdf (SEFAZ download).
    # Some prefectures auto-name DANFSe PDFs with the access key only;
    # other clues are usually absent. Match conservatively (PDF only).
    if fname.endswith(".pdf"):
        stem = fname.rsplit(".", 1)[0]
        if stem.isdigit() and len(stem) in (44, 50):
            return "nfse_shps"

    return None


def detect_source_from_columns(filename: str, file_bytes: bytes) -> Optional[str]:
    """Column-signature sniff for CSV/XLSX. Returns source_key or None.

    Cheap: reads only the first few rows. Mirrors `auto_detect_source:329-356`.
    For XLSX we also do a sheet-aware probe — ML's modern reports have 3+
    sheets (Ajuda / Glossário / actual data) and the data header sits on
    row 2 (row 1 is a meta title cell).
    """
    fname = (filename or "").lower()

    # ── XLSX path: handle ML reports with multiple sheets + meta title row ──
    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        try:
            import pandas as pd
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            # Pick the most likely data sheet — skip help/glossary.
            target = None
            for sn in xl.sheet_names:
                low = sn.lower()
                if "ajuda" in low or "gloss" in low or "help" in low:
                    continue
                if "relat" in low or "anúnc" in low or "anunc" in low or "patroc" in low:
                    target = sn
                    break
            if target is None:
                # Pick last non-help sheet, falling back to the last sheet.
                non_help = [
                    sn for sn in xl.sheet_names
                    if not any(k in sn.lower() for k in ("ajuda", "gloss", "help"))
                ]
                target = (non_help or xl.sheet_names)[-1]
            # Try header=0 then header=1 — ML's «pads_report» has a meta row.
            for header_row in (0, 1):
                try:
                    df = pd.read_excel(
                        io.BytesIO(file_bytes), sheet_name=target,
                        header=header_row, nrows=3,
                    )
                except Exception:
                    continue
                cols = {str(c).strip().lower() for c in df.columns
                        if not str(c).startswith("Unnamed:")}
                if len(cols) < 2:
                    continue

                def has_xlsx(sub: str) -> bool:
                    return any(sub in c for c in cols)

                if has_xlsx("investimento") and (has_xlsx("acos") or has_xlsx("roas")):
                    return "ads_publicidade"
                if has_xlsx("código do anúncio") or has_xlsx("codigo do anuncio"):
                    return "ads_publicidade"
        except Exception:
            pass
        return None

    if not fname.endswith(".csv"):
        return None
    try:
        import pandas as pd
        # Try semicolon first (ML / MP Brazilian exports), then comma. Sniff
        # multiple skiprows because different ML exports start the header at
        # different rows.
        # `engine="python", on_bad_lines="skip"` is critical: MP Mercado Pago
        # statements have TWO sections with different column counts (4-col
        # summary header on row 1, then 5-col transaction header mid-file).
        # The default C engine reads chunks ahead and raises ParserError on
        # the column-count mismatch even with nrows=2; the Python engine
        # respects nrows strictly and skips mismatched lines silently, so the
        # first-section sniff (`initial_balance + final_balance`) actually
        # gets a chance to match.
        for sep in (";", ","):
            for skip in (5, 0, 1, 4, 6):
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes), sep=sep, skiprows=skip, nrows=2,
                        encoding="utf-8-sig",
                        engine="python", on_bad_lines="skip",
                    )
                except Exception:
                    continue
                cols = {str(c).strip().lower() for c in df.columns}
                if len(cols) < 2:
                    continue

                def has(sub: str) -> bool:
                    return any(sub in c for c in cols)

                if has("# de anúncio") or has("# de anuncio") or has("# de publicación") or has("# de publicacion"):
                    return "vendas_ml"
                if has("net_received_amount") or has("transaction_amount"):
                    return "collection_mp"
                if has("investimento") and (has("acos") or has("roas")):
                    return "ads_publicidade"
                if has("tarifa por unidade"):
                    return "armazenagem_full"
                if has("amount_refunded") and has("shipment_status"):
                    return "after_collection"
                if has("identificador") and has("descrição"):
                    return "extrato_nubank"
                if "release_date" in cols or "transaction_net_amount" in cols or "partial_balance" in cols:
                    return "extrato_mp"
                if "initial_balance" in cols and "final_balance" in cols:
                    return "extrato_mp"
                if "data lançamento" in cols or "data lancamento" in cols:
                    if any("r$" in c for c in cols):
                        return "extrato_c6_brl"
                    if any("us$" in c or "usd" in c for c in cols):
                        return "extrato_c6_usd"
                    return "extrato_c6_brl"
    except Exception:
        return None
    return None
