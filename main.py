"""
LatAm Sellers Finance — File Analysis API
Accepts uploaded files, auto-detects source type, returns summary.
"""
from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="LatAm Sellers Finance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Source labels ──
SOURCE_LABELS = {
    "vendas_ml": {"name": "Vendas ML", "type": "ecom"},
    "collection_mp": {"name": "Collection MP", "type": "ecom"},
    "extrato_mp": {"name": "Extrato Mercado Pago", "type": "ecom"},
    "fatura_ml": {"name": "Fatura ML", "type": "ecom"},
    "ads_publicidade": {"name": "Anúncios Patrocinados", "type": "ecom"},
    "armazenagem_full": {"name": "Custos Armazenagem Full", "type": "ecom"},
    "stock_full": {"name": "Stock ML Full", "type": "ecom"},
    "after_collection": {"name": "Pós-vendas", "type": "ecom"},
    "extrato_nubank": {"name": "Extrato Nubank", "type": "bank"},
    "extrato_c6_brl": {"name": "Extrato C6 BRL", "type": "bank"},
    "extrato_c6_usd": {"name": "Extrato C6 USD", "type": "bank"},
    "trafficstars": {"name": "TrafficStars", "type": "expense"},
    "bybit_history": {"name": "Bybit History", "type": "crypto"},
    "das_simples": {"name": "DAS Simples Nacional", "type": "tax"},
    "nfse_shps": {"name": "NFS-e (SHPS)", "type": "invoice"},
    "full_express": {"name": "Full Express (Leticia)", "type": "3pl"},
}


def try_read_csv(file_bytes: bytes, nrows: int | None = None) -> pd.DataFrame | None:
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        for sep in [";", ",", "\t"]:
            for skip in [0, 5, 1, 2, 3, 4]:
                try:
                    kw = {"nrows": nrows} if nrows else {}
                    df = pd.read_csv(
                        io.BytesIO(file_bytes), sep=sep, skiprows=skip,
                        encoding=enc, **kw,
                    )
                    if len(df.columns) > 2:
                        return df
                except Exception:
                    continue
    return None


def auto_detect_source(df: pd.DataFrame, filename: str) -> str | None:
    cols = set(c.strip().lower() for c in df.columns) if len(df.columns) > 0 else set()
    fname = filename.lower()

    def has_col(substring):
        return any(substring in c for c in cols)

    # Column-based detection
    if has_col("net_received_amount") or has_col("transaction_amount"):
        return "collection_mp"
    if has_col("# de anúncio") or has_col("# de anuncio"):
        return "vendas_ml"
    if has_col("investimento") and (has_col("acos") or has_col("roas")):
        return "ads_publicidade"
    if has_col("tarifa por unidade"):
        return "armazenagem_full"
    if has_col("amount_refunded") and has_col("shipment_status"):
        return "after_collection"
    if has_col("identificador") and has_col("descrição"):
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

    # Filename-based detection
    if "collection" in fname:
        return "collection_mp"
    if "account_statement" in fname:
        return "extrato_mp"
    if "vendas" in fname and "mercado" in fname:
        return "vendas_ml"
    if "vendas" in fname:
        return "vendas_ml"
    if "anuncios" in fname or "patrocinados" in fname:
        return "ads_publicidade"
    if "armazenamento" in fname or "armazenagem" in fname:
        return "armazenagem_full"
    if "stock_general" in fname or "stock_full" in fname or fname.startswith("stock"):
        return "stock_full"
    if "after_collection" in fname or "pos" in fname:
        return "after_collection"
    if "fatura" in fname or "faturamento" in fname:
        return "fatura_ml"
    if "extrato" in fname and "nubank" in fname:
        return "extrato_nubank"
    if "c6" in fname:
        if "usd" in fname or "global_usd" in fname:
            return "extrato_c6_usd"
        return "extrato_c6_brl"
    if fname.startswith("01k"):
        return "extrato_c6_brl"
    if "trafficstars" in fname or "traffic" in fname:
        return "trafficstars"
    if "bybit" in fname:
        return "bybit_history"
    if "pgdasd" in fname or "das-" in fname:
        return "das_simples"
    if "nfs" in fname or "nfse" in fname:
        return "nfse_shps"
    return None


def detect_period(df: pd.DataFrame) -> str | None:
    """Try to detect date range from DataFrame."""
    date_cols = []
    for c in df.columns:
        cl = c.strip().lower()
        if any(k in cl for k in ["date", "data", "fecha", "release_date", "created_date"]):
            date_cols.append(c)

    if not date_cols:
        # Try first column if it looks like dates
        for c in df.columns:
            sample = df[c].dropna().head(5).astype(str)
            if sample.str.contains(r"\d{4}[-/]\d{2}", regex=True).any():
                date_cols.append(c)
                break
            if sample.str.contains(r"\d{2}/\d{2}/\d{4}", regex=True).any():
                date_cols.append(c)
                break

    for col in date_cols:
        try:
            dates = pd.to_datetime(df[col], dayfirst=True, errors="coerce").dropna()
            if len(dates) > 0:
                mn = dates.min()
                mx = dates.max()
                months_pt = [
                    "", "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
                ]
                if mn.year == mx.year and mn.month == mx.month:
                    return f"{months_pt[mn.month]} {mn.year}"
                return f"{months_pt[mn.month]}-{months_pt[mx.month]} {mx.year}"
        except Exception:
            continue
    return None


def compute_top_categories(df: pd.DataFrame, source: str) -> list[dict]:
    """Extract top expense/income categories depending on source type."""
    cats = []

    # For extrato_mp — group by 'description' or 'source_id'
    if source == "extrato_mp":
        for col_name in ["description", "source_id", "tipo"]:
            if col_name in [c.strip().lower() for c in df.columns]:
                real_col = [c for c in df.columns if c.strip().lower() == col_name][0]
                val_col = None
                for vc in df.columns:
                    vcl = vc.strip().lower()
                    if any(k in vcl for k in ["amount", "valor", "net", "gross"]):
                        val_col = vc
                        break
                if val_col:
                    try:
                        df[val_col] = pd.to_numeric(
                            df[val_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                            errors="coerce",
                        )
                        grouped = df.groupby(real_col)[val_col].sum().abs().sort_values(ascending=False).head(5)
                        for cat, val in grouped.items():
                            cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
                    except Exception:
                        pass
                break

    # For vendas_ml — by SKU or title
    if source == "vendas_ml":
        for col_name in ["sku", "título do anúncio", "titulo do anuncio"]:
            matched = [c for c in df.columns if c.strip().lower() == col_name]
            if matched:
                val_col = None
                for vc in df.columns:
                    vcl = vc.strip().lower()
                    if "receita" in vcl or "net" in vcl or "valor" in vcl:
                        val_col = vc
                        break
                if val_col:
                    try:
                        df[val_col] = pd.to_numeric(
                            df[val_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                            errors="coerce",
                        )
                        grouped = df.groupby(matched[0])[val_col].sum().abs().sort_values(ascending=False).head(5)
                        for cat, val in grouped.items():
                            cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
                    except Exception:
                        pass
                break

    # For fatura_ml — by tariff type
    if source == "fatura_ml":
        val_cols = [c for c in df.columns if any(
            k in c.strip().lower() for k in ["tarifa", "comiss", "valor"]
        )]
        for vc in val_cols[:3]:
            try:
                total = pd.to_numeric(
                    df[vc].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                ).sum()
                if abs(total) > 0:
                    cats.append({"category": vc.strip()[:40], "value": round(abs(float(total)), 2)})
            except Exception:
                pass
        cats.sort(key=lambda x: x["value"], reverse=True)

    # For bank extracts — group by description
    if source in ("extrato_nubank", "extrato_c6_brl", "extrato_c6_usd"):
        desc_col = None
        val_col = None
        for c in df.columns:
            cl = c.strip().lower()
            if any(k in cl for k in ["descri", "histórico", "historico", "descrição"]):
                desc_col = c
            if any(k in cl for k in ["valor", "saída", "saida", "entrada", "r$", "us$", "amount"]):
                if val_col is None:
                    val_col = c
        if desc_col and val_col:
            try:
                df[val_col] = pd.to_numeric(
                    df[val_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                )
                grouped = df.groupby(desc_col)[val_col].sum().abs().sort_values(ascending=False).head(5)
                for cat, val in grouped.items():
                    cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
            except Exception:
                pass

    # Generic fallback — find any numeric column and group by first text column
    if not cats:
        text_col = None
        num_col = None
        for c in df.columns:
            if text_col is None and df[c].dtype == "object":
                text_col = c
            if num_col is None:
                try:
                    vals = pd.to_numeric(
                        df[c].astype(str).str.replace(",", ".").str.replace(" ", ""),
                        errors="coerce",
                    )
                    if vals.notna().sum() > len(df) * 0.5:
                        num_col = c
                except Exception:
                    pass
        if text_col and num_col:
            try:
                df[num_col] = pd.to_numeric(
                    df[num_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                )
                grouped = df.groupby(text_col)[num_col].sum().abs().sort_values(ascending=False).head(5)
                for cat, val in grouped.items():
                    cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
            except Exception:
                pass

    return cats[:5]


@app.get("/health")
async def health():
    return {"status": "ok"}


def parse_brl(val) -> float:
    """Parse BRL value: '1.234,56' or '1234.56' → float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if not s or s in ("nan", "NaN", "None", ""):
        return 0.0
    s = s.replace("R$", "").replace("$", "").replace(" ", "")
    # BRL format: 1.234,56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def build_vendas_ml_opiu(df: pd.DataFrame) -> dict:
    """Build mini P&L (OPiU) from Vendas ML DataFrame."""
    receita_bruta = 0.0
    tarifa_venda = 0.0
    receita_envio = 0.0
    tarifa_envio = 0.0
    cancelamentos = 0.0
    total_net = 0.0
    vendas_count = 0
    ads_count = 0

    # Detect column names (may vary with accents)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "preço unitário" in cl or "preco unitario" in cl:
            col_map["preco"] = c
        if "receita por produtos" in cl:
            col_map["receita_prod"] = c
        if "tarifa de venda" in cl:
            col_map["tarifa_venda"] = c
        if "receita por envio" in cl:
            col_map["receita_envio"] = c
        if "tarifas de envio" in cl:
            col_map["tarifa_envio"] = c
        if "cancelamentos" in cl:
            col_map["cancelamentos"] = c
        if cl == "total (brl)":
            col_map["total"] = c
        if cl == "unidades":
            col_map["unidades"] = c
        if "venda por publicidade" in cl:
            col_map["ads"] = c

    for _, row in df.iterrows():
        # Receita bruta: Preço unitário × Unidades, or Receita por produtos
        if "preco" in col_map and "unidades" in col_map:
            preco = parse_brl(row.get(col_map["preco"], 0))
            try:
                u = row.get(col_map["unidades"], 0)
                if pd.isna(u) or u == "":
                    unidades = 0
                else:
                    unidades = int(float(str(u).strip()))
            except (ValueError, TypeError):
                unidades = 0
            receita_bruta += preco * unidades
        elif "receita_prod" in col_map:
            receita_bruta += parse_brl(row.get(col_map["receita_prod"], 0))
        tarifa_venda += parse_brl(row.get(col_map.get("tarifa_venda", "Tarifa de venda e impostos (BRL)"), 0))
        receita_envio += parse_brl(row.get(col_map.get("receita_envio", "Receita por envio (BRL)"), 0))
        tarifa_envio += parse_brl(row.get(col_map.get("tarifa_envio", "Tarifas de envio (BRL)"), 0))
        cancelamentos += parse_brl(row.get(col_map.get("cancelamentos", "Cancelamentos e reembolsos (BRL)"), 0))
        total_net += parse_brl(row.get(col_map.get("total", "Total (BRL)"), 0))
        vendas_count += 1
        ads_val = str(row.get(col_map.get("ads", "Venda por publicidade"), "")).strip().lower()
        if ads_val == "sim":
            ads_count += 1

    return {
        "receita_bruta": round(receita_bruta, 2),
        "tarifa_venda": round(abs(tarifa_venda), 2),
        "receita_envio": round(receita_envio, 2),
        "tarifa_envio": round(abs(tarifa_envio), 2),
        "cancelamentos": round(abs(cancelamentos), 2),
        "total_net": round(total_net, 2),
        "vendas_count": vendas_count,
        "ads_count": ads_count,
        "ads_pct": round(ads_count / vendas_count * 100, 1) if vendas_count > 0 else 0,
    }


def detect_vendas_period(df: pd.DataFrame) -> str | None:
    """Detect period from 'Data da venda' column (format: '24 de março de 2026 22:36 hs.')."""
    import re
    pt_months = {
        "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
        "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
        "agosto": "Ago", "setembro": "Set", "outubro": "Out",
        "novembro": "Nov", "dezembro": "Dez",
    }
    col = None
    for c in df.columns:
        if "data da venda" in c.strip().lower() or "data" in c.strip().lower():
            col = c
            break
    if col is None:
        return None

    months_found = set()
    year = None
    for val in df[col].dropna().astype(str):
        m = re.match(r'(\d{1,2}) de (\w+) de (\d{4})', val.strip())
        if m:
            mon_pt = m.group(2).lower()
            year = m.group(3)
            short = pt_months.get(mon_pt)
            if short:
                months_found.add(short)

    if not months_found:
        return None

    order = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    sorted_m = sorted(months_found, key=lambda x: order.index(x) if x in order else 99)
    if len(sorted_m) == 1:
        return f"{sorted_m[0]} {year}"
    return f"{sorted_m[0]}–{sorted_m[-1]} {year}"


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Accept a file upload, detect its type, and return analysis summary."""
    try:
        file_bytes = await file.read()
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Try to read as CSV (with multiple skiprows for vendas_ml format)
        df = None
        if ext in ("csv", "txt", "tsv", ""):
            df = try_read_csv(file_bytes, nrows=5)

        # Try xlsx
        if df is None and ext in ("xlsx", "xls"):
            try:
                df = pd.read_excel(io.BytesIO(file_bytes), nrows=5)
            except Exception:
                pass

        if df is None or df.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not parse file", "filename": filename},
            )

        # Detect source
        source = auto_detect_source(df, filename)

        # Read full file
        df_full = None
        if ext in ("csv", "txt", "tsv", ""):
            df_full = try_read_csv(file_bytes)
        elif ext in ("xlsx", "xls"):
            try:
                df_full = pd.read_excel(io.BytesIO(file_bytes))
            except Exception:
                df_full = df

        if df_full is None:
            df_full = df

        source_info = SOURCE_LABELS.get(source, {"name": source or "Desconhecido", "type": "unknown"})

        # Build response based on source type
        result = {
            "filename": filename,
            "source_key": source,
            "source_name": source_info["name"],
            "source_type": source_info["type"],
            "rows": len(df_full),
            "columns": len(df_full.columns),
        }

        # Special handling for vendas_ml — build OPiU
        if source == "vendas_ml":
            opiu = build_vendas_ml_opiu(df_full)
            result["opiu"] = opiu
            result["period"] = detect_vendas_period(df_full)
            result["top_categories"] = [
                {"category": "Receita Bruta", "value": opiu["receita_bruta"]},
                {"category": "Tarifa de Venda", "value": opiu["tarifa_venda"]},
                {"category": "Tarifa de Envio", "value": opiu["tarifa_envio"]},
                {"category": "Cancelamentos", "value": opiu["cancelamentos"]},
                {"category": "Total NET", "value": abs(opiu["total_net"])},
            ]
        else:
            result["period"] = detect_period(df_full)
            result["top_categories"] = compute_top_categories(df_full, source or "")

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "filename": file.filename},
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
