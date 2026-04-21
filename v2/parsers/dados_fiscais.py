"""Parser for "Dados Fiscais" Excel from Mercado Livre.

ML выдаёт продавцам официальный файл `Dados_Fiscais-YYYY_MM_DD.xlsx` с данными
всех их объявлений. После заполнения и ре-аплоада это авторитетный источник
себестоимости, NCM, Origem (импорт/локал), веса и налоговых кодов на уровне SKU.

Структура:
  Лист "Produtos Únicos"
    Row 0: section header (skip)
    Row 1: column headers
    Row 2: descriptions (skip)
    Row 3+: data (one row per variation = MLB × variation_id × SKU)

Ключевые колонки:
  1 Código do Anúncio       (MLB)
  5 Variação
  6 VARIATION_ID
  7 Código do Produto       (SKU)
  9 EAN
 11 Origem                  (0-8 code)
 12 NCM
 14 CEST
 17 Descrição do produto para NF-e
 18 Unidade de medida comercial
 19 Peso líquido (kg)
 20 Peso bruto (kg)
 21 Custo do Produto        (BRL, себестоимость)
 22 CSOSN de Venda
  8 CSOSN de Transferência
 23 Chave da FCI
 24 Regra Tributária
"""
from __future__ import annotations

import io
from typing import Any


SHEET_NAME = "Produtos Únicos"
HEADER_ROW = 1   # 0-indexed: row 1 has the column names
DATA_START_ROW = 3


# Origem codes per Mercado Livre fiscal taxonomy
# 0/4/5/8 = Nacional; 1/2/3/6/7 = Estrangeira (importação)
_IMPORT_ORIGEM_CODES = frozenset({1, 2, 3, 6, 7})
_LOCAL_ORIGEM_CODES = frozenset({0, 4, 5, 8})


def _origem_type(code: int | None) -> str | None:
    if code is None:
        return None
    if code in _IMPORT_ORIGEM_CODES:
        return "import"
    if code in _LOCAL_ORIGEM_CODES:
        return "local"
    return None


def _to_float(v: Any) -> float | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(v)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _find_col(header_row: list, name: str) -> int | None:
    """Locate column index by exact header-name match (case-insensitive, trimmed)."""
    target = name.strip().lower()
    for i, h in enumerate(header_row):
        if h is None:
            continue
        if str(h).strip().lower() == target:
            return i
    return None


def parse_dados_fiscais_bytes(file_bytes: bytes) -> dict[str, dict[str, Any]]:
    """Parse Excel bytes → dict keyed by SKU.

    Returns `{}` if the sheet isn't a Dados Fiscais template (missing required
    columns). Multiple rows with the same SKU get merged — last occurrence wins
    (usually variations share SKU in error; user gets the latest data).
    """
    import pandas as pd

    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        return {}
    if SHEET_NAME not in xl.sheet_names:
        return {}

    df = pd.read_excel(
        xl,
        sheet_name=SHEET_NAME,
        header=None,
        dtype=object,
    )
    if df.shape[0] < DATA_START_ROW + 1:
        return {}

    header = df.iloc[HEADER_ROW].tolist()

    col_mlb = _find_col(header, "Código do Anúncio")
    col_sku = _find_col(header, "Código do Produto")
    col_titulo = _find_col(header, "Descrição do Anúncio")
    col_var = _find_col(header, "Variação")
    col_var_id = _find_col(header, "VARIATION_ID")
    col_ean = _find_col(header, "EAN")
    col_origem = _find_col(header, "Origem")
    col_ncm = _find_col(header, "NCM")
    col_cest = _find_col(header, "CEST")
    col_descricao_nfe = _find_col(header, "Descrição do produto para NF-e")
    col_unidade = _find_col(header, "Unidade de medida comercial")
    col_peso_liq = _find_col(header, "Peso líquido")
    col_peso_bruto = _find_col(header, "Peso bruto")
    col_custo = _find_col(header, "Custo do Produto")
    col_csosn_venda = _find_col(header, "CSOSN de Venda")
    col_csosn_transf = _find_col(header, "CSOSN de Transferência")
    col_fci = _find_col(header, "Chave da FCI")
    col_regra = _find_col(header, "Regra Tributária")

    # Minimal required columns: if any is missing, bail out
    if col_sku is None or col_mlb is None or col_custo is None:
        return {}

    def _cell(row, idx: int | None) -> Any:
        if idx is None or idx >= len(row):
            return None
        v = row[idx]
        # pandas uses NaN for empty Excel cells → treat as None
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass
        return v

    out: dict[str, dict[str, Any]] = {}
    for i in range(DATA_START_ROW, df.shape[0]):
        row = df.iloc[i].tolist()
        sku_raw = _to_str(_cell(row, col_sku))
        if not sku_raw:
            continue
        sku_key = sku_raw.upper()

        origem_code = _to_int(_cell(row, col_origem))
        ncm_raw = _cell(row, col_ncm)
        ncm_str = ""
        if ncm_raw is not None:
            # NCM can come as int (80171200) or string ("8471.30.00"); normalize
            if isinstance(ncm_raw, (int, float)):
                try:
                    ncm_str = str(int(ncm_raw))
                except (TypeError, ValueError):
                    ncm_str = _to_str(ncm_raw)
            else:
                ncm_str = _to_str(ncm_raw).replace(".", "")

        record = {
            "sku": sku_raw,
            "mlb": _to_str(_cell(row, col_mlb)),
            "titulo_anuncio": _to_str(_cell(row, col_titulo)),
            "variacao": _to_str(_cell(row, col_var)),
            "variation_id": _to_str(_cell(row, col_var_id)),
            "ean": _to_str(_cell(row, col_ean)),
            "origem_code": origem_code,
            "origem_type": _origem_type(origem_code),
            "ncm": ncm_str or None,
            "cest": _to_str(_cell(row, col_cest)) or None,
            "descricao_nfe": _to_str(_cell(row, col_descricao_nfe)) or None,
            "unidade": _to_str(_cell(row, col_unidade)) or None,
            "peso_liquido_kg": _to_float(_cell(row, col_peso_liq)),
            "peso_bruto_kg": _to_float(_cell(row, col_peso_bruto)),
            "custo_brl": _to_float(_cell(row, col_custo)),
            "csosn_venda": _to_str(_cell(row, col_csosn_venda)) or None,
            "csosn_transferencia": _to_str(_cell(row, col_csosn_transf)) or None,
            "chave_fci": _to_str(_cell(row, col_fci)) or None,
            "regra_tributaria": _to_str(_cell(row, col_regra)) or None,
        }
        out[sku_key] = record

    return out
