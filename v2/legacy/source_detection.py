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
    """Resolve source_key using filename first, then column sniff."""
    hit = detect_source_from_filename(filename)
    if hit:
        return hit
    if file_bytes:
        return detect_source_from_columns(filename, file_bytes)
    return None


def detect_source_from_filename(filename: str) -> Optional[str]:
    """Return the matching `source_key` for a given filename, or None if unknown."""
    if not filename:
        return None
    fname = filename.lower()

    # Vendas / order-feed reports (check before "mercado" so Account Statement doesn't win)
    if ("vendas" in fname or "ventas" in fname) and ("mercado" in fname or "mercado_li" in fname):
        return "vendas_ml"

    # Mercado Pago activity
    if "collection" in fname:
        return "collection_mp"
    if "account_statement" in fname:
        return "extrato_mp"

    # Ads
    if "anuncios" in fname or "patrocinados" in fname:
        return "ads_publicidade"

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
    if "extrato" in fname and "nubank" in fname:
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

    return None


def detect_source_from_columns(filename: str, file_bytes: bytes) -> Optional[str]:
    """Column-signature sniff for CSV/XLSX. Returns source_key or None.

    Cheap: reads only the first few rows. Mirrors `auto_detect_source:329-356`.
    Only enabled for CSV (fast to parse); XLSX fallback requires openpyxl which
    is heavier — skip here, let the UI manual selector handle it.
    """
    fname = (filename or "").lower()
    if not fname.endswith(".csv"):
        return None
    try:
        import pandas as pd
        # Try semicolon first (ML / MP Brazilian exports), then comma. Sniff
        # multiple skiprows because different ML exports start the header at
        # different rows.
        for sep in (";", ","):
            for skip in (5, 0, 1, 4, 6):
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes), sep=sep, skiprows=skip, nrows=2,
                        encoding="utf-8-sig", low_memory=False,
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
