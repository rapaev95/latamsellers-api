"""
LATAMSELLERS — Report generators
Generates OPiU (P&L), DDS (Cash Flow), Balance per project.
"""
import io
import re
import pandas as pd
from pathlib import Path
from .config import (
    DATA_DIR, MONTHS, DATA_SOURCES, get_project_by_sku,
    ESTONIA_TAX_APPLIED, PROJETOS_DIR, load_projects, TRADE_DAS_RATE,
)
from .tax_brazil import compute_das


def parse_brl(val) -> float:
    """Parse Brazilian number format: 1.234,56 → 1234.56"""
    if pd.isna(val) or val == "":
        return 0.0
    s = str(val).strip()
    if "." in s and "," not in s:
        try:
            return float(s)
        except ValueError:
            pass
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def _find_file(month: str, prefix: str) -> Path | None:
    """Find a data file by source prefix in month folder."""
    month_dir = DATA_DIR / month
    if not month_dir.exists():
        return None
    for f in month_dir.iterdir():
        if f.name.startswith(prefix) and f.is_file():
            return f
    return None


def _read_csv_auto(path: Path, **kwargs) -> pd.DataFrame | None:
    """Read CSV trying multiple separators."""
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", **kwargs)
            if len(df.columns) > 2:
                return df
        except Exception:
            continue
    return None


def load_vendas_ml(month: str) -> pd.DataFrame | None:
    """Legacy: читает старый vendas_ml CSV (skiprows=5). Игнорирует xlsx —
    для xlsx используется отдельный load_vendas_ml_report().
    """
    month_dir = DATA_DIR / month
    if not month_dir.exists():
        return None
    for f in month_dir.iterdir():
        if f.name.startswith("vendas_ml") and f.is_file() and f.suffix.lower() == ".csv":
            for _skip in [5, 6, 4, 7]:
                try:
                    _df = pd.read_csv(f, sep=";", skiprows=_skip, encoding="utf-8")
                    if "Estado" in _df.columns or "N.º de venda" in _df.columns:
                        return _df
                except Exception:
                    continue
            return pd.read_csv(f, sep=";", skiprows=5, encoding="utf-8")
    return None


def load_collection_mp(month: str) -> pd.DataFrame | None:
    path = _find_file(month, "collection_mp")
    if not path:
        return None
    return _read_csv_auto(path)


def load_ads(month: str) -> pd.DataFrame | None:
    path = _find_file(month, "ads_publicidade")
    if not path:
        return None
    return _read_csv_auto(path)


def load_extrato_mp(month: str) -> pd.DataFrame | None:
    path = _find_file(month, "extrato_mp")
    if not path:
        return None
    return _read_csv_auto(path)


def load_extrato_nubank(month: str) -> pd.DataFrame | None:
    path = _find_file(month, "extrato_nubank")
    if not path:
        return None
    return _read_csv_auto(path)


def load_extrato_c6(month: str, currency: str = "brl") -> pd.DataFrame | None:
    path = _find_file(month, f"extrato_c6_{currency}")
    if not path:
        return None
    ext = path.suffix.lower()
    if ext == ".csv":
        return _read_csv_auto(path)
    elif ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────
# OPiU (P&L) — ECOM projects
# ─────────────────────────────────────────────

def generate_opiu_from_vendas(months: list[str] | None = None) -> dict:
    """Generate OPiU per project from Vendas ML data."""
    if months is None:
        months = MONTHS

    projects = {}

    # Try newest vendas_ml file from _data/, fallback to legacy
    has_data_files = any(_find_file(m, "vendas_ml") for m in months)
    if not has_data_files:
        from pathlib import Path
        legacy_dir = DATA_DIR.parent / "vendas"
        legacy_files = sorted(legacy_dir.glob("2026*.csv")) if legacy_dir.exists() else []
        # Filter only Vendas ML files (exclude other CSVs)
        legacy_files = [f for f in legacy_files if "Vendas_BR_Mercado_Libre" in f.name]
        if legacy_files:
            try:
                df_all = pd.read_csv(legacy_files[-1], sep=";", skiprows=5, encoding="utf-8")
                # Process all rows, group by month using "Data da venda"
                for _, row in df_all.iterrows():
                    sku = str(row.get("SKU", "")).strip()
                    mlb = str(row.get("# de anúncio", "")).strip()
                    proj = get_project_by_sku(sku, mlb)

                    if proj not in projects:
                        projects[proj] = {
                            "receita_bruta": 0, "tarifa_venda": 0, "receita_envio": 0,
                            "tarifa_envio": 0, "cancelamentos": 0, "total_net": 0,
                            "vendas_count": 0, "ads_count": 0, "by_month": {},
                        }
                    p = projects[proj]
                    # Use Preço unitário × Unidades (matches approved report)
                    preco_unit = parse_brl(row.get("Preço unitário de venda do anúncio (BRL)", 0))
                    try:
                        u_raw = row.get("Unidades", 0)
                        if pd.isna(u_raw) or u_raw == "":
                            unidades = 0
                        else:
                            unidades = int(float(str(u_raw).strip()))
                    except (ValueError, TypeError):
                        unidades = 0
                    bruto_row = preco_unit * unidades

                    p["receita_bruta"] += bruto_row
                    p["tarifa_venda"] += parse_brl(row.get("Tarifa de venda e impostos (BRL)", 0))
                    p["receita_envio"] += parse_brl(row.get("Receita por envio (BRL)", 0))
                    p["tarifa_envio"] += parse_brl(row.get("Tarifas de envio (BRL)", 0))
                    p["cancelamentos"] += parse_brl(row.get("Cancelamentos e reembolsos (BRL)", 0))
                    p["total_net"] += parse_brl(row.get("Total (BRL)", 0))
                    p["vendas_count"] += 1
                    if str(row.get("Venda por publicidade", "")).strip().lower() == "sim":
                        p["ads_count"] += 1

                    # Parse date for monthly breakdown
                    # Format: "24 de março de 2026 22:36 hs."
                    date_str = str(row.get("Data da venda", "")).strip()
                    month_key = None
                    if date_str:
                        pt_months = {
                            "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
                            "abril": "04", "maio": "05", "junho": "06", "julho": "07",
                            "agosto": "08", "setembro": "09", "outubro": "10",
                            "novembro": "11", "dezembro": "12",
                        }
                        import re as re_mod
                        m_match = re_mod.match(r'(\d{1,2}) de (\w+) de (\d{4})', date_str)
                        if m_match:
                            day = int(m_match.group(1))
                            mon_pt = m_match.group(2).lower()
                            year = m_match.group(3)
                            mon_num = pt_months.get(mon_pt)
                            if mon_num:
                                month_key = f"{year}-{mon_num}"

                    if month_key:
                        if month_key not in p["by_month"]:
                            p["by_month"][month_key] = {
                                "receita_bruta": 0, "total_net": 0, "vendas": 0,
                                "tarifa_venda": 0, "tarifa_envio": 0, "cancelamentos": 0,
                            }
                        bm = p["by_month"][month_key]
                        bm["receita_bruta"] += bruto_row  # Preço unitário × Unidades
                        bm["total_net"] += parse_brl(row.get("Total (BRL)", 0))
                        bm["tarifa_venda"] += parse_brl(row.get("Tarifa de venda e impostos (BRL)", 0))
                        bm["tarifa_envio"] += parse_brl(row.get("Tarifas de envio (BRL)", 0))
                        bm["cancelamentos"] += parse_brl(row.get("Cancelamentos e reembolsos (BRL)", 0))
                        bm["vendas"] += 1
                return projects
            except Exception:
                pass

    for month in months:
        df = load_vendas_ml(month)
        if df is None:
            continue

        for _, row in df.iterrows():
            sku = str(row.get("SKU", "")).strip()
            mlb = str(row.get("# de anúncio", "")).strip()
            proj = get_project_by_sku(sku, mlb)

            if proj not in projects:
                projects[proj] = {
                    "receita_bruta": 0, "tarifa_venda": 0, "receita_envio": 0,
                    "tarifa_envio": 0, "cancelamentos": 0, "total_net": 0,
                    "vendas_count": 0, "ads_count": 0, "by_month": {},
                }

            p = projects[proj]
            # Use Preço unitário × Unidades (matches approved report)
            preco_unit = parse_brl(row.get("Preço unitário de venda do anúncio (BRL)", 0))
            try:
                unidades = int(str(row.get("Unidades", "0")).strip() or "0")
            except (ValueError, TypeError):
                unidades = 0
            bruto_row = preco_unit * unidades

            p["receita_bruta"] += bruto_row
            p["tarifa_venda"] += parse_brl(row.get("Tarifa de venda e impostos (BRL)", 0))
            p["receita_envio"] += parse_brl(row.get("Receita por envio (BRL)", 0))
            p["tarifa_envio"] += parse_brl(row.get("Tarifas de envio (BRL)", 0))
            p["cancelamentos"] += parse_brl(row.get("Cancelamentos e reembolsos (BRL)", 0))
            p["total_net"] += parse_brl(row.get("Total (BRL)", 0))
            p["vendas_count"] += 1

            if str(row.get("Venda por publicidade", "")).strip().lower() == "sim":
                p["ads_count"] += 1

            if month not in p["by_month"]:
                p["by_month"][month] = {
                    "receita_bruta": 0, "total_net": 0, "vendas": 0,
                    "tarifa_venda": 0, "tarifa_envio": 0, "cancelamentos": 0,
                }
            bm = p["by_month"][month]
            bm["receita_bruta"] += bruto_row
            bm["total_net"] += parse_brl(row.get("Total (BRL)", 0))
            bm["tarifa_venda"] += parse_brl(row.get("Tarifa de venda e impostos (BRL)", 0))
            bm["tarifa_envio"] += parse_brl(row.get("Tarifas de envio (BRL)", 0))
            bm["cancelamentos"] += parse_brl(row.get("Cancelamentos e reembolsos (BRL)", 0))
            bm["vendas"] += 1

    return projects


# ─────────────────────────────────────────────
# DDS (Cash Flow) — ECOM projects
# ─────────────────────────────────────────────

def generate_dds_ecom(project: str, months: list[str] | None = None) -> dict:
    """
    Generate cash flow for an ECOM project.
    Inflows: vendas NET entering MP
    Outflows: from MP (ads, fees) and from Nubank (purchases, full express)
    """
    if months is None:
        months = MONTHS

    dds = {"by_month": {}, "totals": {"inflows": 0, "outflows": 0}}

    opiu = generate_opiu_from_vendas(months)
    proj_data = opiu.get(project)
    if not proj_data:
        return dds

    for month in months:
        bm = proj_data["by_month"].get(month)
        if not bm:
            continue

        inflows = bm["total_net"]
        # Outflows from ML: fees are already deducted in NET, so outflows = separate costs
        outflows_ads = 0  # TODO: from ads report by project
        outflows_fees = abs(bm["tarifa_venda"]) + abs(bm["tarifa_envio"])

        dds["by_month"][month] = {
            "inflows_vendas_net": inflows,
            "outflows_taxas_ml": -outflows_fees,  # already in NET calculation
            "outflows_ads": outflows_ads,
            "outflows_cancelamentos": bm["cancelamentos"],
            "net_flow": inflows,  # NET already has fees deducted
        }
        dds["totals"]["inflows"] += inflows
        # Note: outflows are already embedded in NET, not separate cash movements

    return dds


# ─────────────────────────────────────────────
# BALANCE — ECOM projects
# ─────────────────────────────────────────────

def generate_balance_ecom(project: str, months: list[str] | None = None) -> dict:
    """
    Generate balance sheet for an ECOM project.
    Assets: MP balance (per project share), stock value
    Liabilities: debts to/from partners
    """
    if months is None:
        months = MONTHS

    opiu = generate_opiu_from_vendas(months)
    proj_data = opiu.get(project, {})

    stock_data = load_stock_full().get(project, {}) or {}
    proj_meta = load_projects().get(project, {}) or {}
    from .sku_catalog import assess_stock_for_project

    _assess = assess_stock_for_project(
        project,
        stock_data.get("by_sku"),
        int(proj_meta.get("stock_units_external", 0) or 0),
        proj_meta.get("avg_cost_per_unit_brl"),
    )
    stock_value = float(_assess.get("stock_value_brl") or 0)

    balance = {
        "assets": {
            "mp_balance": proj_data.get("total_net", 0),  # Simplified: accumulated NET
            "stock_value": stock_value,
        },
        "liabilities": {
            "debt_partner": 0,  # TODO: from Nubank transfers
        },
        "equity": {
            "investments": 0,  # TODO: USDT transfers etc.
            "retained_earnings": proj_data.get("total_net", 0),
        },
    }
    return balance


# ─────────────────────────────────────────────
# OPiU / DDS / BALANCE — SERVICES (Estonia/Ganza)
# ─────────────────────────────────────────────

def generate_opiu_estonia() -> dict:
    """
    Generate OPiU for Estonia from OUR COMPANY perspective.

    Revenue model:
    - Commission = same % as tax rate on each invoice
      (15.50% when tax=15.50%, 16.75% when tax=16.75%)
    - Next bracket: revenue R$720k-R$1.8M → commission 19.75%
    - Rental (aluguel) from partners: $800/month each

    Our P&L:
      + Commission (% of invoice gross)
      + Rental income (aluguel)
      - DAS tax paid
      = Our profit
    """
    # From approved Balanco_Estonia_19_03_2026.csv
    # Invoices split by tax bracket (faixa):
    #   ate 180k → 15.50%
    #   180-360k → 16.75%
    #   360k+ → 18.75%
    # Some invoices (#5, #8) span two brackets
    invoice_lines = [
        {"date": "2025-07-07", "gross": 1537.41,   "tax": 238.30,   "rate": 0.155},
        {"date": "2025-07-09", "gross": 27867.27,  "tax": 4319.47,  "rate": 0.155},
        {"date": "2025-08-08", "gross": 85244.97,  "tax": 13212.97, "rate": 0.155},
        {"date": "2025-09-03", "gross": 64373.03,  "tax": 9977.82,  "rate": 0.155},
        {"date": "2025-11-05", "gross": 977.32,    "tax": 151.48,   "rate": 0.155},   # inv#5 part1
        {"date": "2025-11-05", "gross": 2218.41,   "tax": 371.58,   "rate": 0.1675},  # inv#5 part2
        {"date": "2025-12-02", "gross": 118857.00, "tax": 19908.55, "rate": 0.1675},
        {"date": "2025-10-02", "gross": 11676.91,  "tax": 1955.88,  "rate": 0.1675},
        {"date": "2026-01-14", "gross": 47247.68,  "tax": 7913.99,  "rate": 0.1675},  # inv#8 part1
        {"date": "2026-01-14", "gross": 133472.52, "tax": 25026.10, "rate": 0.1875},  # inv#8 part2
        {"date": "2026-02-02", "gross": 91744.81,  "tax": 17202.15, "rate": 0.1875},
        {"date": "2026-03-03", "gross": 101482.12, "tax": 19027.90, "rate": 0.1875},
    ]

    # ── Подмешиваем загруженные NFS-e (sidecar JSON-ы) ──
    # Берём только инвойсы ПОСЛЕ baseline (competência >= 2026-04), tomador SHPS,
    # авто-определяем bracket по cumulative gross, при необходимости разбиваем на 2 части.
    BRACKETS = [
        (180000.0,  0.1550),   # до 180k
        (360000.0,  0.1675),   # 180k–360k
        (720000.0,  0.1875),   # 360k–720k
        (1800000.0, 0.1975),   # 720k–1.8M
    ]
    BASELINE_CUTOFF = "2026-04"  # competência >= этой → новый инвойс

    def _bracket_split(start_cum: float, gross: float) -> list[tuple[float, float]]:
        """Разбивает инвойс на (gross_part, rate) по bracket'ам."""
        parts: list[tuple[float, float]] = []
        remaining = gross
        cum = start_cum
        for ceiling, rate in BRACKETS:
            if cum >= ceiling:
                continue
            room = ceiling - cum
            take = min(remaining, room)
            if take > 0:
                parts.append((take, rate))
                cum += take
                remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            # Сверх 1.8M — используем последний rate (грубо)
            parts.append((remaining, BRACKETS[-1][1]))
        return parts

    cum_gross = sum(inv["gross"] for inv in invoice_lines)
    loaded_nfs = load_all_nfse()
    # Сортируем по competência → ref_month → numero
    loaded_nfs.sort(key=lambda r: (
        r.get("competencia") or "",
        r.get("ref_month_iso") or "",
        int(r.get("numero") or 0) if str(r.get("numero") or "").isdigit() else 0,
    ))
    for nf in loaded_nfs:
        comp = nf.get("competencia") or ""
        if comp < BASELINE_CUTOFF:
            continue
        tomador = (nf.get("tomador") or "").upper()
        if tomador and "SHPS" not in tomador:
            continue  # не Estonia-инвойс
        gross = float(nf.get("valor") or 0)
        if gross <= 0:
            continue
        # Дата для группировки: competência (период признания, по бухгалтерии).
        # ref_month (Março в descrição) — это лишь пометка о периоде комиссии,
        # выручка признаётся в месяце эмиссии NF.
        ref = nf.get("competencia") or nf.get("ref_month_iso")
        date_str = f"{ref}-15" if ref and len(ref) == 7 else nf.get("data_emissao", "2026-04-01")
        for part_gross, rate in _bracket_split(cum_gross, gross):
            invoice_lines.append({
                "date": date_str,
                "gross": part_gross,
                "tax": round(part_gross * rate, 2),
                "rate": rate,
                "numero": nf.get("numero"),
                "auto_loaded": True,
            })
            cum_gross += part_gross

    # DAS Simples Nacional payments (already paid, from approved report)
    das_payments = [
        {"month": "2025-07", "value": 1802.72, "status": "paid"},
        {"month": "2025-08", "value": 5126.31, "status": "paid"},
        {"month": "2025-09", "value": 4409.82, "status": "paid"},
        {"month": "2025-10", "value": 874.77, "status": "paid"},
        {"month": "2025-11", "value": 1913.72, "status": "paid"},
        {"month": "2025-12", "value": 9811.83, "status": "paid"},
        {"month": "2026-01", "value": 21341.46, "status": "paid"},
    ]

    # Override / extend with loaded DAS PDFs
    loaded_das = load_all_das()
    paid_months = {d["month"] for d in das_payments}
    for d in loaded_das:
        if not d.get("month_iso"):
            continue
        # Replace estimate with real value
        existing = next((x for x in das_payments if x["month"] == d["month_iso"]), None)
        if existing:
            existing["value"] = d["total"]
            existing["value_total_company"] = d["total"]
            existing["status"] = "paid_real"
            existing["iss"] = d.get("iss", 0)
            existing["breakdown"] = d
        else:
            das_payments.append({
                "month": d["month_iso"],
                "value": d["total"],
                "value_total_company": d["total"],
                "status": "paid_real",
                "iss": d.get("iss", 0),
                "breakdown": d,
            })

    # === Subtract trade DAS (товары ML) from total DAS ===
    # For each month: trade_das = ml_revenue × TRADE_DAS_RATE (4.5%)
    # Then: estonia_das = total_das - trade_das
    ml_opiu = generate_opiu_from_vendas()
    ml_revenue_by_month = {}  # month → total NET ML revenue (all projects)
    for proj_name, proj_data in ml_opiu.items():
        if proj_name == "NAO_CLASSIFICADO":
            continue
        for m, mdata in proj_data.get("by_month", {}).items():
            ml_revenue_by_month[m] = ml_revenue_by_month.get(m, 0) + mdata.get("receita_bruta", 0)

    # Load Analise.csv with exact 2025 split
    analise_2025 = load_analise_2025()

    for d in das_payments:
        m = d["month"]
        if "value_total_company" not in d:
            d["value_total_company"] = d["value"]

        # Use exact split from Analise.csv if available (2025 months)
        if m in analise_2025 and analise_2025[m]["das_total"] > 0:
            an = analise_2025[m]
            d["value_total_company"] = an["das_total"]
            d["trade_das"] = an["das_vendas"]
            d["estonia_das"] = an["das_servicos"]
            d["iss"] = an["das_servicos"]  # ISS = IMPOSTO SOBRE SERVIÇOS
            d["ml_revenue"] = an["faturamento"]
            d["source"] = "Analise.csv"
        else:
            # Fallback: estimate trade DAS as 4.5% of ML revenue
            ml_rev = ml_revenue_by_month.get(m, 0)
            trade_das = ml_rev * TRADE_DAS_RATE
            d["trade_das"] = trade_das
            d["ml_revenue"] = ml_rev
            d["estonia_das"] = max(0, d["value_total_company"] - trade_das)
            d["source"] = "estimated"

        d["value"] = d["estonia_das"]
    # Estimate DAS for Feb/Mar 2026 based on invoice tax bracket
    # Avg DAS rate ≈ DAS / invoice gross of paid period
    paid_invoices_gross_until_jan = 1537.41 + 27867.27 + 85244.97 + 64373.03 + 977.32 + 2218.41 + 118857.00 + 11676.91 + 47247.68 + 133472.52
    paid_das_total = sum(d["value"] for d in das_payments)
    avg_das_rate = paid_das_total / paid_invoices_gross_until_jan if paid_invoices_gross_until_jan > 0 else 0.066

    # Estimated pending DAS — динамически для всех месяцев с invoice gross,
    # для которых нет реального DAS PDF. Использует актуальный gross (с учётом
    # подмешанных загруженных NFS-e), а не хардкод.
    loaded_months = {d["month"] for d in das_payments if d.get("status") == "paid_real"}
    paid_months = {d["month"] for d in das_payments}
    gross_by_month: dict[str, float] = {}
    for inv in invoice_lines:
        mk = inv["date"][:7]
        gross_by_month[mk] = gross_by_month.get(mk, 0) + inv["gross"]

    das_pending = []
    for mk, mg in sorted(gross_by_month.items()):
        if mk in loaded_months or mk in paid_months:
            continue
        est = mg * avg_das_rate
        das_pending.append({
            "month": mk, "value": est, "value_total_company": est,
            "status": "estimated", "trade_das": 0, "estonia_das": est,
            "ml_revenue": 0,
        })

    # Rental (aluguel): $700/trimester = $350 × 2
    # From approved CSV section 4:
    rental_payments = [
        {"date": "2025-04-09", "usd": 350, "quarter": "Q1 Abr-Jun/25"},
        {"date": "2025-07-17", "usd": 350, "quarter": "Q1 Abr-Jun/25"},
        {"date": "2025-08-19", "usd": 350, "quarter": "Q2 Jul-Set/25"},
        {"date": "2025-10-29", "usd": 350, "quarter": "Q2 Jul-Set/25"},
        {"date": "2025-11-10", "usd": 350, "quarter": "Q3 Out-Dez/25"},
        {"date": "2026-01-14", "usd": 350, "quarter": "Q3 Out-Dez/25"},
        {"date": "2026-02-09", "usd": 350, "quarter": "Q4 Jan-Mar/26"},
    ]
    rental_paid_usd = sum(p["usd"] for p in rental_payments)  # $2,450
    rental_pending_usd = 350  # Q4 second payment

    total_gross = 0
    total_tax = 0
    total_commission = 0
    by_month = {}

    for inv in invoice_lines:
        commission = inv["tax"]  # Our commission = tax amount (same %)
        total_gross += inv["gross"]
        total_tax += inv["tax"]
        total_commission += commission

        m = inv["date"][:7]
        if m not in by_month:
            by_month[m] = {
                "gross": 0, "commission": 0, "tax": 0,
                "net_client": 0, "count": 0, "das": 0,
            }
        by_month[m]["gross"] += inv["gross"]
        by_month[m]["commission"] += commission
        by_month[m]["tax"] += inv["tax"]
        by_month[m]["net_client"] += inv["gross"] - inv["tax"]

    # Count unique invoice dates (some are split across brackets)
    unique_dates = set(inv["date"] for inv in invoice_lines)
    invoice_count = len(unique_dates)

    # Count months with invoices
    for m in by_month:
        by_month[m]["count"] = len([i for i in invoice_lines if i["date"][:7] == m])

    # Add DAS to months
    for d in das_payments + das_pending:
        if d["month"] in by_month:
            by_month[d["month"]]["das"] = d["value"]

    total_net_client = total_gross - total_tax

    # Balance from approved report
    saldo_inicial = 1044.26
    total_enviado = 395227.80
    debito_estonia = (saldo_inicial + total_net_client) - total_enviado

    # Recalc totals AFTER trade DAS apportion
    total_das_paid = sum(d["value"] for d in das_payments)
    total_das_estimated = sum(d["value"] for d in das_pending)
    total_das = total_das_paid + total_das_estimated
    total_trade_das = sum(d.get("trade_das", 0) for d in das_payments)
    total_company_das = sum(d.get("value_total_company", d["value"]) for d in das_payments) + total_das_estimated

    # Our P&L (Estonia portion only)
    our_revenue_brl = total_commission
    our_profit_brl = our_revenue_brl - total_das
    our_profit_brl_paid_only = our_revenue_brl - total_das_paid

    # Current bracket
    if total_gross > 360000:
        current_bracket = "18,75% (360k+)"
    elif total_gross > 180000:
        current_bracket = "16,75% (180-360k)"
    else:
        current_bracket = "15,50% (até 180k)"

    # Build detailed monthly P&L for our company
    pnl_by_month = []
    all_months = sorted(set([d["month"] for d in das_payments] + [d["month"] for d in das_pending] + list(by_month.keys())))
    for m in all_months:
        bm = by_month.get(m, {"gross": 0, "commission": 0, "count": 0})
        das_paid_item = next((d for d in das_payments if d["month"] == m), None)
        das_pend_item = next((d for d in das_pending if d["month"] == m), None)

        if das_paid_item:
            das_value = das_paid_item["value"]
            das_total = das_paid_item.get("value_total_company", das_value)
            das_trade = das_paid_item.get("trade_das", 0)
            das_status = das_paid_item.get("status", "paid")
            das_iss = das_paid_item.get("iss", 0)
            ml_rev = das_paid_item.get("ml_revenue", 0)
            has_pdf = das_status == "paid_real"
        elif das_pend_item:
            das_value = das_pend_item["value"]
            das_total = das_pend_item.get("value_total_company", das_value)
            das_trade = das_pend_item.get("trade_das", 0)
            das_status = "estimated"
            das_iss = 0
            ml_rev = das_pend_item.get("ml_revenue", 0)
            has_pdf = False
        else:
            das_value = 0
            das_total = 0
            das_trade = 0
            das_status = "no_invoice"
            das_iss = 0
            ml_rev = 0
            has_pdf = False

        pnl_by_month.append({
            "month": m,
            "invoice_count": bm.get("count", 0),
            "invoice_gross": bm.get("gross", 0),
            "commission": bm.get("commission", 0),
            "das": das_value,            # Estonia portion
            "das_total": das_total,      # Full company DAS
            "das_trade": das_trade,      # Trade portion (4.5% × ML revenue)
            "das_iss": das_iss,
            "das_status": das_status,
            "ml_revenue": ml_rev,
            "has_pdf": has_pdf,
            "profit": bm.get("commission", 0) - das_value,
        })

    return {
        # Invoice totals
        "total_gross": total_gross,
        "total_tax_retained": total_tax,
        "total_net_client": total_net_client,
        "invoice_count": invoice_count,
        # Balance
        "saldo_inicial": saldo_inicial,
        "total_enviado": total_enviado,
        "debito_estonia": debito_estonia,
        # OUR company P&L
        "our_commission": total_commission,      # BRL
        "our_rental_paid_usd": rental_paid_usd,  # USD
        "our_rental_pending_usd": rental_pending_usd,  # USD
        "our_revenue_brl": our_revenue_brl,
        "our_das_paid": total_das_paid,
        "our_das_estimated": total_das_estimated,
        "our_das": total_das,
        "total_company_das": total_company_das,
        "total_trade_das": total_trade_das,
        "trade_das_rate": TRADE_DAS_RATE,
        "our_profit_brl": our_profit_brl,
        "our_profit_brl_paid_only": our_profit_brl_paid_only,
        "das_payments": das_payments,
        "das_pending": das_pending,
        "pnl_by_month": pnl_by_month,
        "das_period_paid": "Jul/2025 — Jan/2026",
        "das_period_pending": "Fev/2026 — Mar/2026",
        "invoices_period": "Jul/2025 — Mar/2026",
        # Tax brackets
        "current_bracket": current_bracket,
        "next_bracket": "19,75% (R$ 720k — R$ 1.8M)",
        "cumulative_gross": total_gross,
        # Monthly
        "by_month": by_month,
    }


def generate_dds_estonia() -> dict:
    """Generate cash flow for Estonia project from approved report."""
    opiu = generate_opiu_estonia()

    # Detailed transfers list (from approved CSV section 2)
    transfers = [
        {"n": 1, "date": "22/09/25", "usd": None, "vet": None, "canal": "CALIZA-Nubank", "brl": 56131.26},
        {"n": 2, "date": "30/09/25", "usd": None, "vet": None, "canal": "CALIZA-Nubank", "brl": 61179.60},
        {"n": 3, "date": "24/12/25", "usd": None, "vet": None, "canal": "Bybit 9.000 USDT", "brl": 49689.00},
        {"n": 4, "date": "23/02/26", "usd": None, "vet": None, "canal": "Bybit 6.000 USDT", "brl": 33900.00},
        {"n": 5, "date": "06/02/26", "usd": 20, "vet": 5.68, "canal": "C6 Cambio", "brl": 113.50},
        {"n": 6, "date": "06/02/26", "usd": 100, "vet": 5.68, "canal": "C6 Cambio", "brl": 567.49},
        {"n": 7, "date": "06/02/26", "usd": 2200, "vet": 5.54, "canal": "C6 Cambio", "brl": 12180.79},
        {"n": 8, "date": "09/02/26", "usd": 1600, "vet": 5.48, "canal": "C6 Cambio", "brl": 8764.51},
        {"n": 9, "date": "17/02/26", "usd": 2070, "vet": 5.62, "canal": "C6 Cambio", "brl": 11638.29},
        {"n": 10, "date": "23/02/26", "usd": 2500, "vet": 5.56, "canal": "C6 Cambio", "brl": 13905.20},
        {"n": 11, "date": "26/02/26", "usd": 4100, "vet": 5.42, "canal": "C6 Cambio", "brl": 22217.09},
        {"n": 12, "date": "03/03/26", "usd": 1500, "vet": 5.57, "canal": "C6 Cambio", "brl": 8351.03},
        {"n": 13, "date": "03/03/26", "usd": 4200, "vet": 5.59, "canal": "C6 Cambio", "brl": 23459.09},
        {"n": 14, "date": "06/03/26", "usd": 4000, "vet": 5.67, "canal": "C6 Cambio", "brl": 22668.55},
        {"n": 15, "date": "07/03/26", "usd": 100, "vet": 5.65, "canal": "C6 Cambio", "brl": 564.57},
        {"n": 16, "date": "11/03/26", "usd": 100, "vet": 5.44, "canal": "C6 Cambio", "brl": 544.17},
        {"n": 17, "date": "11/03/26", "usd": 4100, "vet": 5.45, "canal": "C6 Cambio", "brl": 22340.45},
        {"n": 18, "date": "13/03/26", "usd": 4150, "vet": 5.52, "canal": "C6 Cambio", "brl": 22923.96},
        {"n": 19, "date": "18/03/26", "usd": 4100, "vet": 5.59, "canal": "C6 Cambio", "brl": 22907.80},
        {"n": 20, "date": "20/01/26", "usd": 103, "vet": None, "canal": "Cred.Nubank TS", "brl": 598.69},
        {"n": 21, "date": "04/02/26", "usd": 103, "vet": None, "canal": "Cred.Nubank TS", "brl": 582.76},
    ]

    outflows = {
        "caliza_col": 56131.26 + 61179.60,
        "bybit_crypto": 49689.00 + 33900.00,
        "trafficstars_c6": sum(t["brl"] for t in transfers if t["canal"] == "C6 Cambio"),
        "trafficstars_credit": 598.69 + 582.76,
    }

    return {
        "inflows": {
            "saldo_inicial": opiu["saldo_inicial"],
            "invoices_gross": opiu["total_gross"],
            "invoices_tax": -opiu["total_tax_retained"],
            "invoices_net": opiu["total_net_client"],
        },
        "outflows": outflows,
        "transfers": transfers,
        "total_outflows": opiu["total_enviado"],
        "debito_estonia": opiu["debito_estonia"],
        "by_month": opiu["by_month"],
    }


def generate_balance_estonia() -> dict:
    """Generate balance for Estonia project — approved + live FIFO data."""
    opiu = generate_opiu_estonia()

    result = {
        "approved_date": "19/03/2026",
        "saldo_inicial": opiu["saldo_inicial"],
        "total_gross": opiu["total_gross"],
        "total_tax": opiu["total_tax_retained"],
        "total_net_client": opiu["total_net_client"],
        "total_enviado_approved": opiu["total_enviado"],
        "debito_approved": opiu["debito_estonia"],
        "our_commission": opiu["our_commission"],
        "our_rental_paid_usd": opiu["our_rental_paid_usd"],
        "our_rental_pending_usd": opiu["our_rental_pending_usd"],
        "our_das": opiu["our_das"],
        "our_profit_brl": opiu["our_profit_brl"],
        "has_live_data": False,
    }

    # Calculate live balance using FIFO TrafficStars (real money spent)
    fifo = calculate_trafficstars_fifo()
    c6_usd = parse_c6_usd()

    if fifo:
        # Real outflows = direct transfers (not C6 Cambio) + TrafficStars FIFO
        caliza_brl = 56131.26 + 61179.60
        bybit_brl = 49689.00 + 33900.00
        cred_ts_brl = 598.69 + 582.76
        ts_fifo_brl = fifo["total_ts_brl"]

        total_real_enviado = caliza_brl + bybit_brl + cred_ts_brl + ts_fifo_brl
        liquido_total = opiu["saldo_inicial"] + opiu["total_net_client"]
        debito_real = liquido_total - total_real_enviado

        result["has_live_data"] = True
        result["caliza_brl"] = caliza_brl
        result["bybit_brl"] = bybit_brl
        result["cred_ts_brl"] = cred_ts_brl
        result["ts_fifo_brl"] = ts_fifo_brl
        result["ts_fifo_usd"] = fifo["total_ts_usd"]
        result["usd_in_stock"] = fifo["usd_in_stock"]
        result["brl_value_in_stock"] = fifo["brl_value_in_stock"]
        result["total_real_enviado"] = total_real_enviado
        result["debito_real"] = debito_real

    if c6_usd:
        result["saldo_usd"] = c6_usd["saldo_usd"]
        result["trafficstars_total_usd"] = c6_usd["trafficstars_usd"]

    return result


# ─────────────────────────────────────────────
# C6 BANK PARSERS
# ─────────────────────────────────────────────

def parse_c6_brl(months: list[str] | None = None) -> dict | None:
    """Parse C6 BRL extrato from loaded data. Returns summary."""
    if months is None:
        months = MONTHS

    all_rows = []
    for month in months:
        path = _find_file(month, "extrato_c6_brl")
        if not path or path.suffix.lower() != ".csv":
            continue
        try:
            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            # Find header line (starts with "Data Lançamento")
            header_idx = None
            for i, line in enumerate(lines):
                if "Data Lançamento" in line or "Data Lancamento" in line:
                    header_idx = i
                    break
            if header_idx is None:
                continue

            import csv as csv_mod
            reader = csv_mod.DictReader(lines[header_idx:])
            for row in reader:
                all_rows.append(row)
        except Exception:
            continue

    if not all_rows:
        return None

    # Categorize
    summary = {
        "pix_entrada": 0,       # Nubank → C6
        "cambio_usd": 0,        # BRL → USD conversion
        "compras_cartao": 0,    # Card purchases (personal)
        "seguro": 0,            # Insurance
        "outros_saida": 0,      # Other outflows
        "saldo_final": 0,
        "by_date": {},
        "total_entrada": 0,
        "total_saida": 0,
        "rows": len(all_rows),
        "date_min": None,
        "date_max": None,
    }

    for row in all_rows:
        title = (row.get("Título") or "").strip()
        entrada = float(row.get("Entrada(R$)", "0") or "0")
        saida = float(row.get("Saída(R$)", "0") or "0")
        saldo = row.get("Saldo do Dia(R$)", "")
        date = (row.get("Data Lançamento") or "").strip()

        summary["total_entrada"] += entrada
        summary["total_saida"] += saida

        if saldo:
            try:
                summary["saldo_final"] = float(saldo)
            except ValueError:
                pass

        # Track dates
        if date:
            if summary["date_min"] is None or date < summary["date_min"]:
                summary["date_min"] = date
            if summary["date_max"] is None or date > summary["date_max"]:
                summary["date_max"] = date

        # Categorize
        if "Pix recebido" in title:
            summary["pix_entrada"] += entrada
            # Daily tracking
            if date not in summary["by_date"]:
                summary["by_date"][date] = {"pix_in": 0, "cambio_out": 0}
            summary["by_date"][date]["pix_in"] += entrada
        elif "Câmbio" in title or "Cambio" in title:
            summary["cambio_usd"] += saida
            if date not in summary["by_date"]:
                summary["by_date"][date] = {"pix_in": 0, "cambio_out": 0}
            summary["by_date"][date]["cambio_out"] += saida
        elif "DEBITO DE CARTAO" in title:
            summary["compras_cartao"] += saida
        elif "SEGURO" in title:
            summary["seguro"] += saida
        else:
            summary["outros_saida"] += saida

    summary["brl_kept"] = summary["total_entrada"] - summary["cambio_usd"]

    return summary


def parse_c6_usd(months: list[str] | None = None) -> dict | None:
    """Parse C6 USD extrato from PDF. Returns summary."""
    if months is None:
        months = MONTHS

    all_lines = []
    for month in months:
        path = _find_file(month, "extrato_c6_usd")
        if not path:
            continue
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_lines.extend(text.split("\n"))
        except Exception:
            continue

    if not all_lines:
        return None

    import re

    summary = {
        "saldo_usd": 0,
        "total_entradas_usd": 0,
        "total_saidas_usd": 0,
        "trafficstars_usd": 0,
        "transf_entrada_usd": 0,
        "outros_debitos_usd": 0,
        "transactions": [],
        "date_min": None,
        "date_max": None,
    }

    # Determine year from header (for date parsing)
    current_year = "2026"
    month_to_year = {}
    for line in all_lines:
        if "Período" in line or "periodo" in line.lower():
            # "Extrato Período • 05 de fevereiro de 2026 até 06 de abril de 2026"
            years = re.findall(r'20\d{2}', line)
            if years:
                current_year = years[-1]

    # Parse header info
    for line in all_lines:
        if "Saldo do dia" in line:
            m = re.search(r'US\$\s*([\d.,]+)', line)
            if m:
                summary["saldo_usd"] = float(m.group(1).replace(".", "").replace(",", "."))
        if "Entradas" in line and "Saidas" in line:
            entradas = re.search(r'Entradas\s*•\s*US\$\s*([\d.,]+)', line)
            saidas = re.search(r'Saidas\s*•\s*US\$\s*([\d.,]+)', line)
            if entradas:
                summary["total_entradas_usd"] = float(entradas.group(1).replace(".", "").replace(",", "."))
            if saidas:
                summary["total_saidas_usd"] = float(saidas.group(1).replace(".", "").replace(",", "."))

    # Parse transactions
    i = 0
    while i < len(all_lines):
        line = all_lines[i].strip()

        # Pattern: "DD/MM Débito de cartão [desc] -US$ X.XXX,XX DD/MM"
        debit_match = re.search(r'(\d{2}/\d{2})\s+Débito de cartão.*?-US\$\s*([\d.,]+)', line)
        if debit_match:
            date = debit_match.group(1)
            amount = float(debit_match.group(2).replace(".", "").replace(",", "."))

            # Check if TrafficStars (look at previous line or current)
            context = all_lines[i-1] if i > 0 else ""
            is_ts = "trafficstars" in context.lower() or "trafficstars" in line.lower()

            if is_ts:
                summary["trafficstars_usd"] += amount
            else:
                summary["outros_debitos_usd"] += amount

            summary["transactions"].append({
                "date": date, "type": "debit",
                "amount": -amount,
                "desc": "TrafficStars" if is_ts else "Outro débito",
            })

            if summary["date_min"] is None or date < summary["date_min"]:
                summary["date_min"] = date
            if summary["date_max"] is None or date > summary["date_max"]:
                summary["date_max"] = date

            i += 1
            continue

        # Pattern: "DD/MM Entrada Transf C6 Conta Global Líquido US$ X.XXX,XX"
        entry_match = re.search(r'(\d{2}/\d{2})\s+Entrada\s+.*US\$\s*([\d.,]+)', line)
        if entry_match:
            date = entry_match.group(1)
            amount = float(entry_match.group(2).replace(".", "").replace(",", "."))
            summary["transf_entrada_usd"] += amount
            summary["transactions"].append({
                "date": date, "type": "entrada",
                "amount": amount, "desc": "Transf C6 Conta Global",
            })

            if summary["date_min"] is None or date < summary["date_min"]:
                summary["date_min"] = date
            if summary["date_max"] is None or date > summary["date_max"]:
                summary["date_max"] = date

            i += 1
            continue

        i += 1

    return summary


def parse_das_pdf(file_path: Path, original_filename: str | None = None) -> dict | None:
    """
    Parse DAS Simples Nacional PDF.
    Returns: {month, total, irpj, csll, cofins, pis, inss, iss, vencimento}
    """
    try:
        import pdfplumber
        import re

        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

        # Fallback: extract period from filename
        fname = original_filename or file_path.name
        filename_period = None
        m = re.search(r'PGDASD-DAS-\d{8}(\d{6})\d{3}', fname)
        if m:
            yyyymm = m.group(1)
            filename_period = f"{yyyymm[:4]}-{yyyymm[4:6]}"

        # Or extract from parent folder name (e.g. _data/2026-02/das_simples.pdf)
        if not filename_period:
            parent_name = file_path.parent.name
            if re.match(r'^\d{4}-\d{2}$', parent_name):
                filename_period = parent_name

        if not text:
            # No text layer — return what we can extract from filename
            if filename_period:
                year_num, month_num = filename_period.split("-")
                month_names = {"01": "Janeiro", "02": "Fevereiro", "03": "Março", "04": "Abril",
                               "05": "Maio", "06": "Junho", "07": "Julho", "08": "Agosto",
                               "09": "Setembro", "10": "Outubro", "11": "Novembro", "12": "Dezembro"}
                return {
                    "month": f"{month_names.get(month_num, month_num)}/{year_num}",
                    "month_iso": filename_period,
                    "total": 0,
                    "vencimento": None,
                    "irpj": 0, "csll": 0, "cofins": 0, "pis": 0, "inss": 0, "iss": 0, "icms": 0,
                    "needs_manual_total": True,
                    "no_text_layer": True,
                }
            return None

        result = {
            "month": None,
            "month_iso": None,
            "total": 0,
            "vencimento": None,
            "irpj": 0, "csll": 0, "cofins": 0, "pis": 0, "inss": 0, "iss": 0, "icms": 0,
        }

        # Period: "Fevereiro/2026"
        months_pt = {
            "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
            "abril": "04", "maio": "05", "junho": "06", "julho": "07",
            "agosto": "08", "setembro": "09", "outubro": "10",
            "novembro": "11", "dezembro": "12",
        }
        period_match = re.search(r'(Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro)/(\d{4})', text, re.IGNORECASE)
        if period_match:
            month_name = period_match.group(1).lower()
            year = period_match.group(2)
            result["month"] = f"{period_match.group(1)}/{year}"
            month_num = months_pt.get(month_name, "00")
            result["month_iso"] = f"{year}-{month_num}"

        # Vencimento date
        venc_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
        if venc_match:
            result["vencimento"] = venc_match.group(1)

        # Total value: "Valor Total do Documento" then number
        total_match = re.search(r'Valor Total do Documento\s*([\d.,]+)', text)
        if total_match:
            result["total"] = float(total_match.group(1).replace(".", "").replace(",", "."))

        # Tax breakdown by code
        # 1001 IRPJ, 1002 CSLL, 1004 COFINS, 1005 PIS, 1006 INSS, 1010 ISS, 1007 ICMS
        tax_codes = {
            "1001": "irpj", "1002": "csll", "1004": "cofins",
            "1005": "pis", "1006": "inss", "1010": "iss", "1007": "icms",
        }
        for code, key in tax_codes.items():
            # Look for "1001 IRPJ - SIMPLES NACIONAL ... 573,77"
            pattern = rf'{code}\s+\w+[^\n]*?(\d{{1,3}}(?:\.\d{{3}})*,\d{{2}})'
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1).replace(".", "").replace(",", "."))

        # If total is 0 but we have breakdown, sum it
        if result["total"] == 0:
            result["total"] = sum([result["irpj"], result["csll"], result["cofins"],
                                   result["pis"], result["inss"], result["iss"], result["icms"]])

        return result if result["total"] > 0 else None
    except Exception:
        return None


def parse_nfse_pdf(file_path: Path, original_filename: str | None = None) -> dict | None:
    """
    Parse NFS-e (Nota Fiscal de Serviço eletrônica) PDF.
    Returns: {numero, competencia, data_emissao, valor, tomador, descricao, ref_month}
    """
    try:
        import pdfplumber
        import re

        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

        if not text:
            return None

        if "NFS-e" not in text and "Nota Fiscal" not in text:
            return None

        result = {
            "numero": None,
            "competencia": None,           # YYYY-MM (período de emissão)
            "competencia_raw": None,       # raw "MM/YYYY"
            "data_emissao": None,
            "valor": 0,
            "tomador": None,
            "descricao": None,
            "ref_month": None,             # mês a que se refere a comissão (do descrição)
            "ref_month_iso": None,
            "valor_liquido": 0,
        }

        # DANFSe v1.0 имеет колоночный layout: 3 лейбла в одной строке,
        # 3 значения — на следующей. Парсим line-by-line, ищем строку
        # с числом/датой сразу после строки с нужным лейблом (нечувствительно
        # к пробелам, т.к. некоторые PDF извлекаются как "NúmerodaNFS-e").
        lines = [ln for ln in text.split("\n")]
        compact_lines = [re.sub(r"\s+", "", ln) for ln in lines]

        def _next_match(label: str, value_re: str, group: int = 1):
            label_compact = re.sub(r"\s+", "", label)
            for i, cl in enumerate(compact_lines):
                if label_compact in cl:
                    # Берём следующие 1-2 строки и ищем в них pattern
                    for j in (i + 1, i + 2):
                        if j >= len(lines):
                            break
                        m = re.search(value_re, lines[j])
                        if m:
                            return m.group(group)
            return None

        num = _next_match("Número da NFS-e", r"^\s*(\d+)\b")
        if num:
            result["numero"] = num

        comp = _next_match("Competência da NFS-e", r"(\d{2}/\d{2}/\d{4}|\d{2}/\d{4})")
        if comp:
            result["competencia_raw"] = comp
            parts = comp.split("/")
            if len(parts) == 3:        # DD/MM/YYYY
                _, mo, yr = parts
            else:                       # MM/YYYY
                mo, yr = parts
            result["competencia"] = f"{yr}-{mo}"

        emissao = _next_match("Data e Hora da emissão da NFS-e", r"(\d{2}/\d{2}/\d{4})")
        if emissao:
            result["data_emissao"] = emissao

        # Tomador — находим строки с SHPS/LTDA, исключая GANZA
        all_names = re.findall(r'(GANZA[^\n]*|SHPS[^\n]*|[A-Z][A-Z ]*LTDA\.?)', text)
        for name in all_names:
            if "GANZA" in name:
                continue
            result["tomador"] = name.strip()
            break

        # Valor do Serviço — может быть "R$ 207.242,22" или слитно "R$207.242,22"
        valor_str = _next_match("Valor do Serviço", r"R\$\s*([\d.,]+)")
        if valor_str:
            try:
                result["valor"] = float(valor_str.replace(".", "").replace(",", "."))
            except ValueError:
                pass

        # Valor Líquido da NFS-e
        liq_str = _next_match("Valor Líquido da NFS-e", r"R\$\s*([\d.,]+)")
        if liq_str:
            try:
                result["valor_liquido"] = float(liq_str.replace(".", "").replace(",", "."))
            except ValueError:
                pass

        # Descrição do Serviço (e mês de referência)
        m = re.search(r'Descri[çc][ãa]odoServi[çc]o\s*\n\s*([^\n]+)', re.sub(r"[ \t]+", "", text))
        if not m:
            m = re.search(r'Descrição do Serviço\s*\n?\s*([^\n]+)', text)
        if m:
            result["descricao"] = m.group(1).strip()
            # Try to extract reference month: "elegível em Janeiro", "Comissão Janeiro", etc.
            months_pt = {
                "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
                "abril": "04", "maio": "05", "junho": "06", "julho": "07",
                "agosto": "08", "setembro": "09", "outubro": "10",
                "novembro": "11", "dezembro": "12",
            }
            month_match = re.search(r'(janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)', result["descricao"], re.IGNORECASE)
            if month_match:
                month_name = month_match.group(1).lower()
                month_num = months_pt.get(month_name)
                if month_num and result["competencia"]:
                    # Use year from competencia, but month from descrição
                    yr = result["competencia"].split("-")[0]
                    # If reference month > competencia month, it's previous year
                    comp_month = int(result["competencia"].split("-")[1])
                    if int(month_num) > comp_month:
                        yr = str(int(yr) - 1)
                    result["ref_month_iso"] = f"{yr}-{month_num}"
                    result["ref_month"] = f"{month_name.capitalize()}/{yr}"

        return result if result["valor"] > 0 else None
    except Exception:
        return None


def aggregate_classified_by_project(project: str, after_date: str | None = None) -> dict:
    """
    Aggregate classified bank transactions for a project.
    Reads all classifications JSONs from _data/.

    Args:
        project: project ID (ARTUR, GANZA, etc.)
        after_date: only count transactions after this date (DD/MM/YYYY or DD-MM-YYYY)

    Returns dict with categorized totals.
    """
    import json as json_mod_agg

    result = {
        "inflows": 0,
        "outflows": 0,
        "by_category": {},
        "by_source": {},  # source file → total
        "transactions": [],
    }

    cutoff_date = None
    if after_date:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                cutoff_date = pd.Timestamp(pd.to_datetime(after_date, format=fmt))
                break
            except (ValueError, TypeError):
                continue

    sources = ["extrato_nubank", "extrato_c6_brl", "extrato_c6_usd", "extrato_mp"]
    for month in MONTHS:
        for src in sources:
            jp = DATA_DIR / month / f"{src}_classifications.json"
            if not jp.exists():
                continue
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    d = json_mod_agg.load(f)
            except Exception:
                continue

            txs = d.get("transactions", [])
            splits = d.get("full_express_splits", {})

            # Direct project assignments
            for tx in txs:
                proj = tx.get("Проект", "")
                cat = tx.get("Категория", "")
                if proj != project:
                    continue
                # Skip splittable categories — they're handled below
                label_lo = str(tx.get("Класс.", "")).lower()
                in_split = (cat == "fulfillment" or "fatura ml" in label_lo
                            or "retido" in label_lo or "devolu" in label_lo or "reclamaç" in label_lo)
                if in_split:
                    continue

                try:
                    val = float(tx.get("Valor", 0) or 0)
                except (ValueError, TypeError):
                    val = 0

                # Date filter
                if cutoff_date is not None:
                    date_str = str(tx.get("Data", ""))
                    try:
                        tx_date = pd.to_datetime(date_str, dayfirst=True)
                        if tx_date <= cutoff_date:
                            continue
                    except Exception:
                        pass

                if val > 0:
                    result["inflows"] += val
                else:
                    result["outflows"] += abs(val)

                cat_key = cat or "uncategorized"
                if cat_key not in result["by_category"]:
                    result["by_category"][cat_key] = 0
                result["by_category"][cat_key] += val

                src_key = f"{month}/{src}"
                result["by_source"][src_key] = result["by_source"].get(src_key, 0) + val
                result["transactions"].append(tx)

            # Process splits — allocate group totals to project
            for group_key, group_data in splits.items():
                if not isinstance(group_data, dict) or "split" not in group_data:
                    continue
                amt = group_data["split"].get(project, 0)
                if amt > 0:
                    result["outflows"] += amt
                    cat_label = {
                        "fulfillment": "fulfillment_split",
                        "fatura_ml": "fatura_ml_split",
                        "retido": "retido_split",
                        "devolucoes": "devolucoes_split",
                    }.get(group_key, f"{group_key}_split")
                    result["by_category"][cat_label] = result["by_category"].get(cat_label, 0) - amt

    return result


VENDAS_ML_DELIVERED_PATTERNS = [
    "Entregue", "Venda entregue", "No ponto de retirada", "Troca entregue",
    "Mediação finalizada. Te demos o dinheiro",  # медиация в нашу пользу
    "Venda com solicitação de alteração",         # продажа с запросом изменений (деньги остаются)
    "Pacote de",                                   # «Pacote de 2 produtos» — multi-item доставлено
]
VENDAS_ML_RETURNED_PATTERNS = [
    "Cancelada pelo comprador",
    "Devolução",
    "Devolvido",
    "Mediação finalizada com reembolso",   # медиация в пользу покупателя
    "Mediação finalizada. Te demos o dinheiro dessa venda",  # частный случай — потеря денег
    "Liberamos o dinheiro",
    "Pacote cancelado",
    "Venda cancelada",
]


def _classify_estado(estado: str) -> str:
    """Estado → bucket: delivered / returned / in_progress.
    Порядок: returned имеет приоритет, чтобы 'Devolução finalizada com reembolso'
    не был ошибочно отнесён к delivered. НО: 'Mediação finalizada. Te demos o
    dinheiro' (без 'reembolso') = деньги нам, это delivered. Чтобы не путать с
    'Mediação finalizada com reembolso' (returned), проверяем delivered первым
    для строк, начинающихся с 'Mediação'.
    """
    s = (estado or "").strip()
    if not s:
        return "in_progress"
    # Спец-случай: «Mediação finalizada. Te demos o dinheiro» — это delivered
    # (медиация в нашу пользу), а не returned — пропускаем общий returned-чек
    if s.startswith("Mediação finalizada. Te demos o dinheiro") and "dessa venda" not in s:
        return "delivered"
    for p in VENDAS_ML_RETURNED_PATTERNS:
        if p in s:
            return "returned"
    for p in VENDAS_ML_DELIVERED_PATTERNS:
        if p in s:
            return "delivered"
    return "in_progress"


def _parse_one_vendas_file(filename: str, blob: bytes) -> "pd.DataFrame | None":
    """Parse a single Vendas ML file (XLSX or Brazilian CSV ;-sep skip-5)."""
    import io
    buf = io.BytesIO(blob)
    if filename.lower().endswith(".csv"):
        # Brazilian CSV: ;-sep, header on line 6 (skiprows=5), BRL decimal=","
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            buf.seek(0)
            try:
                df = pd.read_csv(
                    buf, sep=";", skiprows=5, encoding=enc, low_memory=False,
                    decimal=",", thousands=".",
                )
                if "Estado" in df.columns or "N.º de venda" in df.columns:
                    return df
            except Exception:
                continue
        return None
    # XLSX (header may be on row 5/6/4/7 — try each)
    for _h in [5, 6, 4, 7]:
        buf.seek(0)
        try:
            df = pd.read_excel(buf, sheet_name=0, header=_h)
            if "Estado" in df.columns or "N.º de venda" in df.columns:
                return df
        except Exception:
            continue
    return None


def _load_vendas_from_db_sync(user_id: int) -> "pd.DataFrame | None":
    """Read the SINGLE NEWEST Vendas ML file the user uploaded and parse it.

    Mirrors legacy Streamlit `load_vendas_ml_report` which picks the most
    recent `vendas_ml*.xlsx` from `_data/{month}/` — one file per report run,
    no cross-file merging / deduping. Each user upload fully replaces the
    prior view.
    """
    import os
    import psycopg2
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")
    if not dsn:
        return None
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            """SELECT filename, file_bytes FROM uploads
               WHERE user_id=%s AND source_key='vendas_ml' AND file_bytes IS NOT NULL
               ORDER BY created_at DESC
               LIMIT 1""",
            (user_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        return None
    if row is None:
        return None

    filename, blob = row
    df = _parse_one_vendas_file(filename, bytes(blob))
    if df is None or df.empty:
        return None
    df.attrs["__source_file"] = filename
    df.attrs["__source_path"] = "<db:uploads>"
    return df


_VENDAS_DF_CACHE: dict = {}  # {(user_id, fingerprint): (timestamp, df)}
_VENDAS_DF_TTL = 120  # seconds


def _vendas_fingerprint_db(user_id: int) -> str | None:
    """Cheap probe of the newest vendas upload for this user — used as cache key
    component so the cache auto-invalidates when the user uploads a new file."""
    import os
    import psycopg2
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")
    if not dsn:
        return None
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            """SELECT id, EXTRACT(EPOCH FROM created_at)::bigint
               FROM uploads
               WHERE user_id=%s AND source_key='vendas_ml' AND file_bytes IS NOT NULL
               ORDER BY created_at DESC LIMIT 1""",
            (user_id,),
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        return f"{row[0]}:{row[1]}" if row else "empty"
    except Exception:
        return None


def invalidate_vendas_cache(user_id: int | None = None) -> None:
    """Clear cached DataFrame for one user (or all users if user_id is None)."""
    if user_id is None:
        _VENDAS_DF_CACHE.clear()
        return
    for k in list(_VENDAS_DF_CACHE.keys()):
        if k[0] == user_id:
            _VENDAS_DF_CACHE.pop(k, None)


def load_vendas_ml_report() -> "pd.DataFrame | None":
    """Загружает Vendas ML из БД (per-user) ИЛИ из FS (legacy fallback).

    Storage mode выбирается через env LS_STORAGE_MODE=db|fs. В DB-режиме
    читаются ВСЕ загруженные пользователем CSV/XLSX файлы и склеиваются в
    один DataFrame с дедупом по `N.º de venda`. Добавляются колонки:
        __bucket  (delivered/returned/in_progress)
        __project (через get_project_by_sku)

    DB-режим держит per-user df-кеш с TTL (инвалидируется по uploads fingerprint).
    FS-режим не кеширует. Вызывающий код ДОЛЖЕН использовать `.copy()` перед
    мутацией — кеш возвращает ссылку на общий DataFrame.
    """
    import os
    import time as _time
    storage_mode = os.environ.get("LS_STORAGE_MODE", "fs").strip().lower()

    # ── Try cache first (DB-mode only) ──
    if storage_mode == "db":
        from .db_storage import _current_user_id
        uid = _current_user_id()
        if uid is not None:
            fp = _vendas_fingerprint_db(uid)
            key = (uid, fp)
            hit = _VENDAS_DF_CACHE.get(key)
            if hit is not None:
                ts, cached_df = hit
                if _time.time() - ts < _VENDAS_DF_TTL:
                    return cached_df

    df: "pd.DataFrame | None" = None

    if storage_mode == "db":
        from .db_storage import _current_user_id
        uid = _current_user_id()
        if uid is not None:
            df = _load_vendas_from_db_sync(uid)

    if df is None:
        # FS fallback: самый свежий vendas_ml*.xlsx из _data/{month}/
        from pathlib import Path
        candidates: list[Path] = []
        for month in MONTHS:
            d = DATA_DIR / month
            if not d.exists():
                continue
            candidates.extend(d.glob("vendas_ml*.xlsx"))
        if not candidates:
            return None
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        for _h in [5, 6, 4, 7]:
            try:
                _df = pd.read_excel(newest, sheet_name=0, header=_h)
                if "Estado" in _df.columns or "N.º de venda" in _df.columns:
                    df = _df
                    break
            except Exception:
                continue
        if df is None:
            return None
        df.attrs["__source_file"] = newest.name
        df.attrs["__source_path"] = str(newest)

    if "Estado" not in df.columns:
        return None
    df["__bucket"] = df["Estado"].apply(_classify_estado)

    # Multi-item orders («Pacote de N produtos»): родительская строка имеет
    # Total и Estado, но пустой SKU — а дочерние строки ниже наоборот: SKU есть,
    # Total пустой. Чтобы определить проект для родительской, протягиваем SKU
    # вниз→вверх по одному N.º de venda.
    order_col = "N.º de venda"
    sku_col = "SKU"
    if order_col in df.columns and sku_col in df.columns:
        # Для каждого order_id найти первый непустой SKU
        order_to_sku: dict = {}
        for _, row in df.iterrows():
            oid = str(row.get(order_col, "") or "").strip()
            sku = str(row.get(sku_col, "") or "").strip()
            if oid and sku and oid not in order_to_sku:
                order_to_sku[oid] = sku

        def _resolve_sku(row):
            sku = str(row.get(sku_col, "") or "").strip()
            if sku:
                return sku
            oid = str(row.get(order_col, "") or "").strip()
            return order_to_sku.get(oid, "")

        df["__sku_resolved"] = df.apply(_resolve_sku, axis=1)
    else:
        df["__sku_resolved"] = df.get(sku_col, "")

    _mlb_col = "# de anúncio" if "# de anúncio" in df.columns else "# de anuncio" if "# de anuncio" in df.columns else None
    df["__project"] = df.apply(
        lambda r: get_project_by_sku(
            str(r.get("__sku_resolved") or "").strip(),
            str(r.get(_mlb_col) or "").strip() if _mlb_col else "",
        ), axis=1,
    )

    # Multi-item «Pacote de N produtos» с пустым SKU — orphan rows из ML.
    # НЕ атрибутируем автоматически: помечаем "PACOTE_SEM_SKU".
    # Ручные назначения хранятся в _data/orphan_assignments.json и
    # перезаписывают __project на выбранный проект.
    if "Estado" in df.columns:
        mask_orphan_pacote = (
            df["Estado"].astype(str).str.startswith("Pacote de", na=False)
            & (df["__sku_resolved"].astype(str).str.strip() == "")
        )
        df.loc[mask_orphan_pacote, "__project"] = "PACOTE_SEM_SKU"

        # Применяем ручные назначения
        manual = load_orphan_assignments()
        if manual:
            for idx, row in df[mask_orphan_pacote].iterrows():
                oid = str(row.get("N.º de venda", "") or "").strip()
                if oid in manual:
                    df.at[idx, "__project"] = manual[oid]

    # ── Populate cache (DB-mode only) ──
    import os as _os
    import time as _time2
    if _os.environ.get("LS_STORAGE_MODE", "fs").strip().lower() == "db":
        from .db_storage import _current_user_id
        uid = _current_user_id()
        if uid is not None:
            fp = _vendas_fingerprint_db(uid)
            _VENDAS_DF_CACHE[(uid, fp)] = (_time2.time(), df)

    return df


def _vendas_ml_title_column(df: pd.DataFrame) -> str | None:
    for c in ("Título do anúncio", "Título do anuncio", "Título", "Titulo do anúncio"):
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if ("título" in cl or "titulo" in cl) and ("anúncio" in cl or "anuncio" in cl):
            return str(c)
    return None


def _is_numeric_like_title(s: str) -> bool:
    """Не использовать как название товара (часто ID объявления)."""
    raw = (s or "").strip()
    if not raw or raw.lower() == "nan":
        return True
    t = raw.replace(".0", "").strip()
    t2 = t.replace(".", "").replace(",", "")
    if t2.isdigit() and len(t2) >= 7:
        return True
    try:
        v = float(t.replace(",", "."))
        if v == int(v) and abs(v) >= 1e6:
            return True
    except ValueError:
        pass
    return False


def _load_latest_vendas_ml_csv_any() -> pd.DataFrame | None:
    """Последний vendas_ml*.csv в _data или legacy vendas/."""
    from pathlib import Path

    cands = list(DATA_DIR.rglob("vendas_ml*.csv"))
    if not cands:
        legacy = DATA_DIR.parent / "vendas"
        if legacy.exists():
            cands = [
                f for f in legacy.glob("*.csv")
                if "Mercado_Libre" in f.name or "Mercado" in f.name
            ]
    if not cands:
        return None
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    for sep in (";", ","):
        try:
            return pd.read_csv(newest, sep=sep, skiprows=5, encoding="utf-8")
        except Exception:
            continue
    return None


def load_sku_titles_from_vendas() -> dict[str, str]:
    """
    SKU (верхний регистр, strip) → название из Vendas ML («Título do anúncio» и т.п.).
    Сначала самый свежий vendas_ml*.xlsx, иначе последний CSV.
    """
    df = load_vendas_ml_report()
    if df is None or df.empty:
        df = _load_latest_vendas_ml_csv_any()
    if df is None or df.empty:
        return {}
    tcol = _vendas_ml_title_column(df)
    if not tcol:
        return {}
    sku_col = "__sku_resolved" if "__sku_resolved" in df.columns else "SKU"
    if sku_col not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sku = str(row.get(sku_col, "") or "").strip()
        if not sku or sku.lower() == "nan":
            continue
        nk = sku.upper()
        tit = str(row.get(tcol, "") or "").strip()
        if not tit or tit.lower() == "nan":
            continue
        if _is_numeric_like_title(tit):
            continue
        prev = out.get(nk)
        if prev is None:
            out[nk] = tit[:240]
        elif _is_numeric_like_title(prev) and not _is_numeric_like_title(tit):
            out[nk] = tit[:240]
        elif len(tit) > len(prev):
            out[nk] = tit[:240]
    return out


_MANUAL_CF_KINDS = ("partner_contributions", "manual_expenses", "manual_supplier")


def add_manual_cashflow_entry(project: str, kind: str, entry: dict) -> bool:
    """Per-user add: appends entry into user_data.projects[PROJECT][kind] list.
    kind ∈ ('partner_contributions', 'manual_expenses', 'manual_supplier').
    """
    if kind not in _MANUAL_CF_KINDS:
        return False
    from .config import load_projects, save_projects, _invalidate_projects_cache
    projects = load_projects() or {}
    pid = project.upper()
    if pid not in projects:
        return False
    lst = projects[pid].get(kind)
    if not isinstance(lst, list):
        lst = []
    lst.append(entry)
    projects[pid][kind] = lst
    save_projects(projects)
    _invalidate_projects_cache()
    return True


def delete_manual_cashflow_entry(project: str, kind: str, index: int) -> bool:
    """Per-user delete: removes user_data.projects[PROJECT][kind][index]."""
    if kind not in _MANUAL_CF_KINDS:
        return False
    from .config import load_projects, save_projects, _invalidate_projects_cache
    projects = load_projects() or {}
    pid = project.upper()
    lst = (projects.get(pid) or {}).get(kind)
    if not isinstance(lst, list) or index < 0 or index >= len(lst):
        return False
    lst.pop(index)
    projects[pid][kind] = lst
    save_projects(projects)
    _invalidate_projects_cache()
    return True


def list_manual_cashflow_entries(project: str) -> dict:
    """Return {kind: [entry]} for a project (3 buckets)."""
    from .config import load_projects
    pid = project.upper()
    p = (load_projects() or {}).get(pid) or {}
    return {k: (p.get(k) if isinstance(p.get(k), list) else []) for k in _MANUAL_CF_KINDS}


# ── Manual Publicidade invoices (Mercado Ads 12-12 billing cycle) ─────────────

def list_publicidade_invoices(project: str) -> list[dict]:
    """Return user-entered publicidade invoices for a project.

    Each invoice: {desde, ate, valor, note} — matches the shape consumed by
    `parse_publicidade_reports` (via projects_db.json[pid][manual_publicidade]).
    """
    from .config import load_projects
    pid = project.upper()
    p = (load_projects() or {}).get(pid) or {}
    lst = p.get("manual_publicidade")
    return lst if isinstance(lst, list) else []


def add_publicidade_invoice(project: str, invoice: dict) -> bool:
    """Append one fatura entry. Input: {desde, ate, valor, note} (desde/ate as YYYY-MM-DD)."""
    from .config import load_projects, save_projects, _invalidate_projects_cache
    projects = load_projects() or {}
    pid = project.upper()
    if pid not in projects:
        return False
    lst = projects[pid].get("manual_publicidade")
    if not isinstance(lst, list):
        lst = []
    lst.append(invoice)
    projects[pid]["manual_publicidade"] = lst
    save_projects(projects)
    _invalidate_projects_cache()
    return True


def delete_publicidade_invoice(project: str, index: int) -> bool:
    """Delete one fatura by its array index within the project."""
    from .config import load_projects, save_projects, _invalidate_projects_cache
    projects = load_projects() or {}
    pid = project.upper()
    lst = (projects.get(pid) or {}).get("manual_publicidade")
    if not isinstance(lst, list) or index < 0 or index >= len(lst):
        return False
    lst.pop(index)
    projects[pid]["manual_publicidade"] = lst
    save_projects(projects)
    _invalidate_projects_cache()
    return True


_ORPHAN_DB_KEY = "f2_orphan_assignments"


def _storage_is_db() -> bool:
    import os
    return os.environ.get("LS_STORAGE_MODE", "fs").strip().lower() == "db"


def load_orphan_assignments() -> dict:
    """Возвращает {order_id: project}. В db-режиме — per-user из user_data.f2_orphan_assignments;
    в fs-режиме — общий _data/orphan_assignments.json (legacy Streamlit)."""
    if _storage_is_db():
        from .db_storage import db_load
        data = db_load(_ORPHAN_DB_KEY)
        return dict(data) if isinstance(data, dict) else {}

    import json as _json
    path = DATA_DIR / "orphan_assignments.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}


def save_orphan_assignment(order_id: str, project: str | None) -> None:
    """Назначает (project=str) или удаляет (project=None) проект для orphan pacote."""
    if _storage_is_db():
        from .db_storage import db_load, db_save, _current_user_id
        data = db_load(_ORPHAN_DB_KEY)
        data = dict(data) if isinstance(data, dict) else {}
        if project:
            data[str(order_id)] = project
        else:
            data.pop(str(order_id), None)
        db_save(_ORPHAN_DB_KEY, data)
        invalidate_vendas_cache(_current_user_id())  # override меняет __project
        return

    import json as _json
    path = DATA_DIR / "orphan_assignments.json"
    data = load_orphan_assignments()
    if project:
        data[str(order_id)] = project
    else:
        data.pop(str(order_id), None)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2, ensure_ascii=False)


def save_orphan_assignments_bulk(assignments: dict) -> int:
    """Bulk-сохранение {order_id: project|None}. Возвращает число записей, реально изменивших state."""
    if _storage_is_db():
        from .db_storage import db_load, db_save, _current_user_id
        data = db_load(_ORPHAN_DB_KEY)
        data = dict(data) if isinstance(data, dict) else {}
        changed = 0
        for oid, proj in assignments.items():
            oid_s = str(oid)
            old = data.get(oid_s)
            if proj:
                if old != proj:
                    data[oid_s] = proj
                    changed += 1
            else:
                if oid_s in data:
                    data.pop(oid_s, None)
                    changed += 1
        if changed:
            db_save(_ORPHAN_DB_KEY, data)
            invalidate_vendas_cache(_current_user_id())
        return changed

    changed = 0
    for oid, proj in assignments.items():
        save_orphan_assignment(str(oid), proj)
        changed += 1
    return changed


def list_orphan_pacotes() -> list[dict]:
    """Все «Pacote de N produtos» orphan-заказы (SKU не резолвится), вне зависимости
    от того, назначен ли уже проект пользователем — чтобы UI мог показать и
    переназначить существующие записи.

    Порт _admin/report_views.py:1354-1372, но с raw-маской, не по `__project`.
    """
    df = load_vendas_ml_report()
    if df is None or df.empty:
        return []
    if "Estado" not in df.columns or "__sku_resolved" not in df.columns:
        return []
    mask = (
        df["Estado"].astype(str).str.startswith("Pacote de", na=False)
        & (df["__sku_resolved"].astype(str).str.strip() == "")
    )
    rows: list[dict] = []
    for _, r in df[mask].iterrows():
        oid = str(r.get("N.º de venda", "") or "").strip()
        if not oid:
            continue
        total_val = pd.to_numeric(r.get("Total (BRL)"), errors="coerce")
        if pd.isna(total_val):
            total_val = pd.to_numeric(r.get("Receita por produtos (BRL)"), errors="coerce")
        rows.append({
            "order_id": oid,
            "data": str(r.get("Data da venda", "") or ""),
            "estado": str(r.get("Estado", "") or ""),
            "comprador": str(r.get("Comprador", "") or ""),
            "total_brl": float(total_val) if not pd.isna(total_val) else 0.0,
            "bucket": str(r.get("__bucket", "") or ""),
        })
    return rows


def get_vendas_ml_by_project(project: str) -> dict:
    """Сводка по проекту из vendas_ml.xlsx.
    Возвращает {delivered, returned, in_progress, total, by_sku, source_file}.
    """
    df = load_vendas_ml_report()
    if df is None or df.empty:
        return {}
    sub = df[df["__project"] == project]
    if sub.empty:
        return {
            "delivered": {"count": 0, "units": 0, "bruto": 0.0, "net": 0.0},
            "returned": {"count": 0, "units": 0, "bruto": 0.0, "net": 0.0},
            "in_progress": {"count": 0, "units": 0, "bruto": 0.0, "net": 0.0},
            "total": {"count": 0, "units": 0, "bruto": 0.0, "net": 0.0},
            "by_sku": [],
            "source_file": df.attrs.get("__source_file", ""),
        }

    bruto_col = "Receita por produtos (BRL)"
    net_col = "Total (BRL)"
    units_col = "Unidades"
    sku_col = "SKU"
    title_col = "Título do anúncio"

    def _agg(rows) -> dict:
        return {
            "count": len(rows),
            "units": int(pd.to_numeric(rows[units_col], errors="coerce").fillna(0).sum()),
            "bruto": float(pd.to_numeric(rows[bruto_col], errors="coerce").fillna(0).sum()),
            "net": float(pd.to_numeric(rows[net_col], errors="coerce").fillna(0).sum()),
        }

    result = {
        "delivered": _agg(sub[sub["__bucket"] == "delivered"]),
        "returned": _agg(sub[sub["__bucket"] == "returned"]),
        "in_progress": _agg(sub[sub["__bucket"] == "in_progress"]),
        "total": _agg(sub),
        "source_file": df.attrs.get("__source_file", ""),
    }

    # by_sku breakdown (delivered + returned counts отдельно)
    by_sku: dict = {}
    for _, row in sub.iterrows():
        sku = str(row.get(sku_col, "") or "").strip() or "(no SKU)"
        if sku not in by_sku:
            by_sku[sku] = {
                "sku": sku,
                "title": str(row.get(title_col, "") or "").strip(),
                "units": 0,
                "bruto": 0.0,
                "net": 0.0,
                "delivered_units": 0,
                "returned_units": 0,
                "in_progress_units": 0,
            }
        try:
            u = int(float(row.get(units_col, 0) or 0))
        except (ValueError, TypeError):
            u = 0
        br = pd.to_numeric(row.get(bruto_col), errors="coerce")
        br = 0.0 if pd.isna(br) else float(br)
        n = pd.to_numeric(row.get(net_col), errors="coerce")
        n = 0.0 if pd.isna(n) else float(n)
        by_sku[sku]["units"] += u
        by_sku[sku]["bruto"] += br
        by_sku[sku]["net"] += n
        bucket = row.get("__bucket", "in_progress")
        by_sku[sku][f"{bucket}_units"] += u
        if not by_sku[sku]["title"]:
            t = str(row.get(title_col, "") or "").strip()
            if t:
                by_sku[sku]["title"] = t

    result["by_sku"] = sorted(by_sku.values(), key=lambda r: -r["net"])

    # By month — парсим "7 de abril de 2026 23:01 hs."
    pt_months = {
        "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
        "abril": "04", "maio": "05", "junho": "06", "julho": "07",
        "agosto": "08", "setembro": "09", "outubro": "10",
        "novembro": "11", "dezembro": "12",
    }
    import re as _re
    by_month: dict = {}
    for _, row in sub.iterrows():
        ds = str(row.get("Data da venda", "") or "")
        m = _re.search(r"de\s+(\w+)\s+de\s+(\d{4})", ds)
        if not m:
            continue
        mname = m.group(1).lower()
        mnum = pt_months.get(mname)
        if not mnum:
            continue
        mkey = f"{m.group(2)}-{mnum}"
        if mkey not in by_month:
            by_month[mkey] = {
                "month": mkey,
                "delivered": 0, "delivered_net": 0.0,
                "returned": 0, "returned_net": 0.0,
                "in_progress": 0, "in_progress_net": 0.0,
                "total": 0, "total_net": 0.0,
            }
        bucket = row.get("__bucket", "in_progress")
        n = pd.to_numeric(row.get(net_col), errors="coerce")
        n = 0.0 if pd.isna(n) else float(n)
        by_month[mkey][bucket] += 1
        by_month[mkey][f"{bucket}_net"] += n
        by_month[mkey]["total"] += 1
        by_month[mkey]["total_net"] += n
    result["by_month"] = sorted(by_month.values(), key=lambda r: r["month"])

    return result


def load_sku_titles() -> dict:
    """SKU → название товара. Берёт из ranking_produtos.csv (Mercado Turbo).
    Если нет — fallback на vendas/setembro 25.csv ... marco 26.csv (Vendas ML).
    """
    import csv as _csv
    titles: dict[str, str] = {}

    # 1) Ranking_produtos.csv в _data/{месяц}/
    for month in MONTHS:
        for path in (DATA_DIR / month).glob("ranking_produtos*.csv") if (DATA_DIR / month).exists() else []:
            try:
                with open(path, encoding="utf-8-sig") as f:
                    rows = list(_csv.reader(f))
            except Exception:
                continue
            for r in rows[1:]:
                if len(r) < 3:
                    continue
                sku = r[1].strip()
                title = r[2].strip()
                if sku and title and sku not in titles:
                    titles[sku] = title

    # 2) Fallback: parsed Vendas ML CSV files
    vendas_dir = DATA_DIR.parent / "vendas"
    if vendas_dir.exists():
        for path in vendas_dir.glob("*.csv"):
            n = path.name.lower()
            if any(skip in n for skip in ("after_collection", "anuncios", "armazenamento", "account_statement", "ranking")):
                continue
            try:
                with open(path, encoding="utf-8-sig", errors="ignore") as f:
                    reader = _csv.reader(f, delimiter=";")
                    rows = list(reader)
            except Exception:
                continue
            if not rows:
                continue
            # ищем заголовок с SKU и Título
            for hdr_idx in range(min(10, len(rows))):
                hdr = rows[hdr_idx]
                sku_i = next((i for i, c in enumerate(hdr) if str(c).strip() == "SKU"), None)
                ttl_i = next((i for i, c in enumerate(hdr) if "Título" in str(c) or "Titulo" in str(c)), None)
                if sku_i is not None and ttl_i is not None:
                    for row in rows[hdr_idx + 1:]:
                        if len(row) <= max(sku_i, ttl_i):
                            continue
                        sku = str(row[sku_i] or "").strip()
                        title = str(row[ttl_i] or "").strip()
                        if sku and title and sku not in titles:
                            titles[sku] = title
                    break
    return titles


def get_products_by_project(project: str) -> list[dict]:
    """Раскладка collection MP по товарам внутри проекта.
    Возвращает список dicts: {sku, title, units, gross, net, mlb}.
    Группирует по SKU, название берёт из колонки `Descrição da operação (reason)`.
    """
    df = load_collection_mp_legacy()
    if df is None or df.empty:
        return []
    sku_col = "SKU do produto (seller_custom_field)"
    item_col = "Código do produto (item_id)"
    title_col = "Descrição da operação (reason)"
    gross_col = "Valor do produto (transaction_amount)"
    net_col = "Valor total recebido (net_received_amount)"
    status_col = "Status da operação (status)"
    order_col = "Número da venda no Mercado Livre (order_id)"

    by_sku: dict = {}
    for _, row in df.iterrows():
        if str(row.get(status_col, "") or "").strip().lower() != "approved":
            continue
        sku = str(row.get(sku_col, "") or "").strip()
        item = str(row.get(item_col, "") or "").strip()
        if get_project_by_sku(sku, item) != project:
            continue
        if not sku:
            sku = "(no SKU)"
        if sku not in by_sku:
            by_sku[sku] = {
                "sku": sku,
                "title": str(row.get(title_col, "") or "").strip(),
                "mlb": item,
                "units": 0,
                "gross": 0.0,
                "net": 0.0,
                "_orders": set(),
            }
        try:
            by_sku[sku]["gross"] += float(row.get(gross_col, 0) or 0)
            by_sku[sku]["net"] += float(row.get(net_col, 0) or 0)
        except (ValueError, TypeError):
            pass
        oid = str(row.get(order_col, "") or "").strip().removesuffix(".0")
        if oid and oid not in by_sku[sku]["_orders"]:
            by_sku[sku]["_orders"].add(oid)
            by_sku[sku]["units"] += 1
        # обновить title если был пустым
        if not by_sku[sku]["title"]:
            t = str(row.get(title_col, "") or "").strip()
            if t:
                by_sku[sku]["title"] = t

    rows = list(by_sku.values())
    for r in rows:
        r.pop("_orders", None)
    rows.sort(key=lambda r: -r["net"])
    return rows


PT_MONTHS_SHORT = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}


def _parse_pt_short_date(s: str):
    """'01-jan-2026' → date(2026, 1, 1). Возвращает None при неудаче."""
    import re as _re
    from datetime import date as _date
    if not s:
        return None
    m = _re.match(r"(\d{1,2})-(\w+)-(\d{4})", s.strip())
    if not m:
        return None
    mn = PT_MONTHS_SHORT.get(m.group(2)[:3].lower())
    if not mn:
        return None
    try:
        return _date(int(m.group(3)), mn, int(m.group(1)))
    except (ValueError, TypeError):
        return None


def _parse_publicidade_rows(rows: list, file_name: str) -> list[dict]:
    """Парсит сырые rows одного отчёта publicidade. rows[1] = header.
    Используется и для csv (csv.reader), и для xlsx (df.values.tolist()).
    """
    if len(rows) < 2:
        return []
    hdr = [str(c) if c is not None else "" for c in rows[1]]

    def col(name: str) -> int | None:
        for i, c in enumerate(hdr):
            if name in c:
                return i
        return None

    i_desde = col("Desde")
    i_ate = col("Até")
    i_camp = col("Campanha")
    i_titulo = col("Título")
    i_mlb = col("Código do anúncio")
    i_inv = col("Investimento")
    if None in (i_desde, i_ate, i_mlb, i_inv):
        return []

    out: list[dict] = []
    for r in rows[2:]:
        max_idx = max(x for x in (i_desde, i_ate, i_mlb, i_inv) if x is not None)
        if len(r) <= max_idx:
            continue
        desde = _parse_pt_short_date(str(r[i_desde]) if r[i_desde] is not None else "")
        ate = _parse_pt_short_date(str(r[i_ate]) if r[i_ate] is not None else "")
        mlb = str(r[i_mlb] or "").strip()
        if not desde or not ate or not mlb or mlb.lower() == "nan":
            continue
        inv_raw = r[i_inv]
        if inv_raw is None:
            inv = 0.0
        elif isinstance(inv_raw, (int, float)):
            inv = float(inv_raw)
        else:
            inv_str = str(inv_raw).strip().replace("R$", "").replace(".", "").replace(",", ".")
            try:
                inv = float(inv_str)
            except ValueError:
                inv = 0.0
        project = get_project_by_sku("", mlb)
        out.append({
            "file_name": file_name,
            "desde": desde,
            "ate": ate,
            "project": project,
            "mlb": mlb,
            "campanha": str(r[i_camp]) if i_camp is not None and len(r) > i_camp and r[i_camp] is not None else "",
            "titulo": str(r[i_titulo]) if i_titulo is not None and len(r) > i_titulo and r[i_titulo] is not None else "",
            "investimento": inv,
        })
    return out


def _rows_from_publicidade_bytes(filename: str, file_bytes: bytes) -> list | None:
    """Decode uploaded publicidade file bytes → rows suitable for `_parse_publicidade_rows`.

    Handles both CSV and XLSX based on filename extension. Mirrors the FS code's
    format detection in `parse_publicidade_reports`.
    """
    import csv as _csv
    import io
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "xlsx":
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            target_sheet = None
            for sn in xl.sheet_names:
                if "Anúncios" in sn or "Anuncios" in sn or "Relat" in sn:
                    target_sheet = sn
            if target_sheet is None:
                target_sheet = xl.sheet_names[-1]
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=target_sheet, header=None)
            return df.where(pd.notna(df), None).values.tolist()
        # default: csv
        text: str | None = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            return None
        return list(_csv.reader(io.StringIO(text), delimiter=";"))
    except Exception:
        return None


def parse_publicidade_reports() -> list[dict]:
    """Парсит ВСЕ отчёты publicidade:
    - LS_STORAGE_MODE=db: файлы из `uploads WHERE source_key='ads_publicidade'` (per-user)
    - LS_STORAGE_MODE=fs (fallback):
        1) `_data/publicidade/*.csv` и `*.xlsx`
        2) `_data/{месяц}/ads_publicidade.{csv,xlsx}` (куда кладёт UI uploader)
    + ручные записи из `projects_db.json[manual_publicidade]`.
    """
    import csv as _csv
    import os
    result: list[dict] = []
    seen: set = set()  # дедуп по имени файла

    storage_mode = os.environ.get("LS_STORAGE_MODE", "fs").strip().lower()
    if storage_mode == "db":
        from .db_storage import _current_user_id
        uid = _current_user_id()
        if uid is not None:
            for filename, file_bytes in _load_files_from_db_sync(uid, "ads_publicidade"):
                if filename in seen:
                    continue
                seen.add(filename)
                rows = _rows_from_publicidade_bytes(filename, file_bytes)
                if rows is None:
                    continue
                result.extend(_parse_publicidade_rows(rows, filename))

    if not result:
        # FS fallback (запускается и при LS_STORAGE_MODE=fs, и если в БД у юзера пусто)
        def _scan_dir(d):
            if not d.exists():
                return []
            return sorted(list(d.glob("*.csv")) + list(d.glob("*.xlsx")))

        paths: list = []
        paths.extend(_scan_dir(DATA_DIR / "publicidade"))
        # _data/{месяц}/ads_publicidade.*
        for month_dir in DATA_DIR.iterdir():
            if not month_dir.is_dir():
                continue
            for p in month_dir.glob("ads_publicidade*"):
                if p.suffix.lower() in (".csv", ".xlsx"):
                    paths.append(p)

        for path in paths:
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                if path.suffix.lower() == ".xlsx":
                    xl = pd.ExcelFile(path)
                    # Ищем sheet с данными "Relatório Anúncios patrocinados" или
                    # содержащий "Anúncios" в названии. Иначе берём последний.
                    target_sheet = None
                    for sn in xl.sheet_names:
                        if "Anúncios" in sn or "Anuncios" in sn or "Relat" in sn:
                            target_sheet = sn
                    if target_sheet is None:
                        target_sheet = xl.sheet_names[-1]
                    df = pd.read_excel(path, sheet_name=target_sheet, header=None)
                    rows = df.where(pd.notna(df), None).values.tolist()
                else:
                    with open(path, encoding="utf-8-sig") as f:
                        rows = list(_csv.reader(f, delimiter=";"))
            except Exception:
                continue
            result.extend(_parse_publicidade_rows(rows, path.name))

    # Ручные записи из projects_db.json[manual_publicidade]
    from datetime import datetime as _dt
    for proj_name, proj_data in (load_projects() or {}).items():
        manual = (proj_data or {}).get("manual_publicidade") or []
        for item in manual:
            try:
                desde = _dt.strptime(item["desde"], "%Y-%m-%d").date()
                ate = _dt.strptime(item["ate"], "%Y-%m-%d").date()
                valor = float(item["valor"])
            except (KeyError, ValueError, TypeError):
                continue
            note = item.get("note", "")
            result.append({
                "file_name": f"manual:{proj_name}:{note}",
                "desde": desde,
                "ate": ate,
                "project": proj_name,
                "mlb": "MANUAL",
                "campanha": note,
                "titulo": note,
                "investimento": valor,
            })
    return result


def get_publicidade_by_period(project: str, period_from, period_to) -> dict:
    """Сумма Investimento всех файлов чей период [Desde, Até] полностью внутри
    [period_from, period_to], для проекта.
    """
    from datetime import timedelta as _td
    rows = parse_publicidade_reports()

    # Группируем по файлу
    by_file: dict = {}
    for r in rows:
        by_file.setdefault(r["file_name"], []).append(r)

    file_meta: dict = {}
    for fname, frows in by_file.items():
        f_desde = min(r["desde"] for r in frows)
        f_ate = max(r["ate"] for r in frows)
        total_days = (f_ate - f_desde).days + 1
        if total_days <= 0:
            continue
        file_meta[fname] = {
            "desde": f_desde,
            "ate": f_ate,
            "total_days": total_days,
            "rows": frows,
        }

    # Сортировка: более узкие первыми (короткий period перекрывает длинный)
    sorted_files = sorted(file_meta.items(), key=lambda kv: kv[1]["total_days"])

    # day_to_file: для каждого дня в выбранном периоде → имя файла
    day_to_file: dict = {}
    cur = period_from
    while cur <= period_to:
        day_to_file[cur] = None
        cur = cur + _td(days=1)

    for fname, meta in sorted_files:
        cur = max(meta["desde"], period_from)
        end = min(meta["ate"], period_to)
        while cur <= end:
            if day_to_file.get(cur) is None:
                day_to_file[cur] = fname
            cur = cur + _td(days=1)

    # Сколько дней использовано из каждого файла
    file_days_used: dict = {}
    for fname in day_to_file.values():
        if fname:
            file_days_used[fname] = file_days_used.get(fname, 0) + 1

    files_used: list[dict] = []
    files_skipped: list[dict] = []
    total = 0.0
    by_sku: dict = {}

    for fname, meta in file_meta.items():
        days_used = file_days_used.get(fname, 0)
        if days_used == 0:
            files_skipped.append({
                "file_name": fname,
                "desde": meta["desde"],
                "ate": meta["ate"],
                "reason": "перекрыт более узким файлом или вне периода",
            })
            continue
        ratio = days_used / meta["total_days"]
        files_used.append({
            "file_name": fname,
            "days_used": days_used,
            "total_days": meta["total_days"],
            "ratio": round(ratio, 3),
        })
        for r in meta["rows"]:
            if r["project"] != project:
                continue
            inv_part = r["investimento"] * ratio
            total += inv_part
            key = r["mlb"]
            if key not in by_sku:
                by_sku[key] = {
                    "mlb": r["mlb"],
                    "campanha": r["campanha"],
                    "titulo": r["titulo"],
                    "investimento": 0.0,
                }
            by_sku[key]["investimento"] += inv_part

    uncovered = sum(1 for v in day_to_file.values() if v is None)
    return {
        "total": total,
        "files_used": sorted(files_used, key=lambda f: -f["days_used"]),
        "files_skipped": files_skipped,
        "uncovered_days": uncovered,
        "total_days": len(day_to_file),
        "by_sku": sorted(by_sku.values(), key=lambda r: -r["investimento"]),
    }


def _parse_brl_money(s) -> float:
    """Парсит "R$ 0,015" / "R$ 11,674" / "R$ 1.234,56" → float."""
    if s is None:
        return 0.0
    s = str(s).strip().replace("R$", "").replace("\xa0", " ").strip()
    if not s or s == "-":
        return 0.0
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _load_files_from_db_sync(user_id: int, source_key: str) -> list[tuple[str, bytes]]:
    """Fetch ALL user-uploaded files for a given source_key from Postgres.

    Used by LS_STORAGE_MODE=db branches of loaders that merge across multiple
    uploads (armazenagem, publicidade). Returns newest-first so callers can
    honour freshness ordering.
    """
    import os
    import psycopg2
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")
    if not dsn:
        return []
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            """SELECT filename, file_bytes FROM uploads
               WHERE user_id=%s AND source_key=%s AND file_bytes IS NOT NULL
               ORDER BY created_at DESC""",
            (user_id, source_key),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        return []
    return [(fn, bytes(blob)) for fn, blob in rows]


def _parse_armazenagem_file(path) -> dict | None:
    """Парсит один Custos_por_servico_armazenamento.csv.
    Возвращает {by_sku: {sku: {MLB, anuncio, values: {date: float}}}, daily_cols: list[str]}.
    """
    import csv as _csv
    try:
        with open(path, encoding="utf-8-sig") as f:
            rows = list(_csv.reader(f, delimiter=";"))
    except Exception:
        return None
    return _parse_armazenagem_rows(rows)


def _parse_armazenagem_bytes_daily(file_bytes: bytes) -> dict | None:
    """Bytes-variant of `_parse_armazenagem_file` for DB-mode (LS_STORAGE_MODE=db).

    Returns the same `{by_sku, daily_cols}` dict shape. Tries utf-8-sig / utf-8 /
    latin-1 encodings to match the FS parser's `encoding="utf-8-sig"` default.
    """
    import csv as _csv
    import io
    text: str | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None
    rows = list(_csv.reader(io.StringIO(text), delimiter=";"))
    return _parse_armazenagem_rows(rows)


def _parse_armazenagem_rows(rows: list) -> dict | None:
    """Source-agnostic armazenagem parser — shared between FS (file path) and DB (bytes) variants."""
    import re as _re
    if len(rows) < 5:
        return None
    hdr = rows[4]
    i_sku = next((i for i, c in enumerate(hdr) if c.strip() == "SKU"), None)
    i_mlb = next((i for i, c in enumerate(hdr) if "Código ML" in c), None)
    i_anuncio = next((i for i, c in enumerate(hdr) if "Número do anúncio" in c), None)
    if i_sku is None:
        return None
    daily_cols: list[tuple[int, str]] = []
    for i, c in enumerate(hdr):
        if _re.match(r"^\d{2}/\d{2}/\d{4}$", c.strip()):
            daily_cols.append((i, c.strip()))

    by_sku: dict = {}
    for r in rows[5:]:
        if len(r) <= i_sku:
            continue
        sku = (r[i_sku] or "").strip()
        if not sku:
            continue
        mlb = (r[i_mlb] or "").strip() if i_mlb is not None and len(r) > i_mlb else ""
        anuncio = (r[i_anuncio] or "").strip() if i_anuncio is not None and len(r) > i_anuncio else ""
        entry = {"MLB": mlb, "anuncio": anuncio, "values": {}}
        for col_i, col_name in daily_cols:
            entry["values"][col_name] = _parse_brl_money(r[col_i]) if len(r) > col_i else 0.0
        by_sku[sku] = entry

    return {"by_sku": by_sku, "daily_cols": [c for _, c in daily_cols]}


def load_armazenagem_report() -> "pd.DataFrame | None":
    """Загружает ВСЕ Custos_por_servico_armazenamento*.csv из БД (LS_STORAGE_MODE=db,
    per-user) или из `_data/armazenagem/` (FS fallback). Файлы объединяются по
    SKU+дате: берётся максимум (свежий перекрывает старый, нули не затирают).
    """
    import os

    # (source_name, parsed_dict) — наполняется либо из БД, либо из ФС
    file_parses: list[tuple[str, dict]] = []

    storage_mode = os.environ.get("LS_STORAGE_MODE", "fs").strip().lower()
    if storage_mode == "db":
        from .db_storage import _current_user_id
        uid = _current_user_id()
        if uid is not None:
            for filename, file_bytes in _load_files_from_db_sync(uid, "armazenagem_full"):
                parsed = _parse_armazenagem_bytes_daily(file_bytes)
                if parsed:
                    file_parses.append((filename, parsed))

    if not file_parses:
        arm_dir = DATA_DIR / "armazenagem"
        if arm_dir.exists():
            files = sorted(arm_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            for path in files:
                parsed = _parse_armazenagem_file(path)
                if parsed:
                    file_parses.append((path.name, parsed))

    if not file_parses:
        return None

    merged: dict = {}     # sku → {SKU, MLB, anuncio, values: {date: max_value}}
    all_dates: set = set()
    sources: list[str] = []
    for source_name, parsed in file_parses:
        sources.append(source_name)
        all_dates.update(parsed["daily_cols"])
        for sku, entry in parsed["by_sku"].items():
            if sku not in merged:
                merged[sku] = {
                    "SKU": sku,
                    "MLB": entry["MLB"],
                    "anuncio": entry["anuncio"],
                    "values": {},
                }
            else:
                if not merged[sku]["MLB"] and entry["MLB"]:
                    merged[sku]["MLB"] = entry["MLB"]
                if not merged[sku]["anuncio"] and entry["anuncio"]:
                    merged[sku]["anuncio"] = entry["anuncio"]
            for d, v in entry["values"].items():
                cur = merged[sku]["values"].get(d, 0.0)
                if v > cur:
                    merged[sku]["values"][d] = v

    if not merged:
        return None

    daily_cols_sorted = sorted(all_dates, key=lambda d: tuple(reversed(d.split("/"))))
    data_rows = []
    for sku, e in merged.items():
        proj = get_project_by_sku(sku if sku != "N/A" else "", e["anuncio"] or e["MLB"])
        row = {"SKU": sku, "MLB": e["MLB"], "anuncio": e["anuncio"], "__project": proj}
        for d in daily_cols_sorted:
            row[d] = e["values"].get(d, 0.0)
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df.attrs["__source_files"] = sources
    df.attrs["__daily_cols"] = daily_cols_sorted
    return df


def get_armazenagem_by_period(project: str, period_from, period_to) -> dict:
    """Сумма дневных стоимостей armazenagem для проекта в дни period."""
    from datetime import datetime as _dt
    df = load_armazenagem_report()
    if df is None or df.empty:
        return {"total": 0.0, "days_in_period": 0, "skus_count": 0, "source_file": ""}
    sources = df.attrs.get("__source_files", [])
    source_str = ", ".join(sources)
    sub = df[df["__project"] == project]
    if sub.empty:
        return {
            "total": 0.0, "days_in_period": 0, "skus_count": 0,
            "source_file": source_str,
        }
    daily_cols = df.attrs.get("__daily_cols", [])
    relevant_cols: list[str] = []
    for c in daily_cols:
        try:
            d = _dt.strptime(c, "%d/%m/%Y").date()
        except ValueError:
            continue
        if period_from <= d <= period_to:
            relevant_cols.append(c)

    total = 0.0
    for c in relevant_cols:
        if c in sub.columns:
            total += float(pd.to_numeric(sub[c], errors="coerce").fillna(0).sum())

    return {
        "total": total,
        "days_in_period": len(relevant_cols),
        "skus_count": int(sub["SKU"].nunique()),
        "source_file": source_str,
    }


def get_devolucoes_by_project() -> dict:
    """Парсит все devolucoes_ml.csv (claims/reclamações Mercado Pago) из _data/.
    Маппит каждую строку через order_id → SKU из collection MP → проект.
    Возвращает {project: {"total": float, "count": int, "by_status": {...}}}.
    Учитываются только статусы approved/closed (opened ещё не разрешены).
    """
    import csv
    result: dict = {}

    # Индекс order_id → project из collection MP
    df = load_collection_mp_legacy()
    if df is None or df.empty:
        return result
    order_col = "Número da venda no Mercado Livre (order_id)"
    sku_col = "SKU do produto (seller_custom_field)"
    item_col = "Código do produto (item_id)"
    if order_col not in df.columns:
        return result

    def _norm_oid(v) -> str:
        s = str(v or "").strip()
        if not s or s == "nan":
            return ""
        # Pandas хранит numeric order_id как "2000015677415262.0" → срезаем .0
        if s.endswith(".0"):
            s = s[:-2]
        return s

    order_to_proj: dict[str, str] = {}
    order_to_sku: dict[str, str] = {}
    order_to_title: dict[str, str] = {}
    title_col = "Descrição da operação (reason)"
    for _, row in df.iterrows():
        oid = _norm_oid(row.get(order_col))
        if not oid:
            continue
        sku = str(row.get(sku_col, "") or "").strip()
        item = str(row.get(item_col, "") or "").strip()
        proj = get_project_by_sku(sku, item)
        if proj and proj != "NAO_CLASSIFICADO":
            order_to_proj[oid] = proj
            if sku and oid not in order_to_sku:
                order_to_sku[oid] = sku
            ttl = str(row.get(title_col, "") or "").strip()
            if ttl and oid not in order_to_title:
                order_to_title[oid] = ttl

    # Парсим все devolucoes_ml.csv
    for month in MONTHS:
        for path in (DATA_DIR / month).glob("devolucoes_ml*.csv") if (DATA_DIR / month).exists() else []:
            try:
                with open(path, encoding="utf-8-sig") as f:
                    reader = csv.reader(f, delimiter=";")
                    rows = list(reader)
            except Exception:
                continue
            if not rows:
                continue
            hdr = rows[0]

            def col(name_part: str) -> int | None:
                for i, c in enumerate(hdr):
                    if name_part.lower() in c.lower():
                        return i
                return None

            i_amt = col("valor (amount)")
            i_status = col("status (status)")
            i_order = col("id do pedido")
            i_motivo = col("motivo detalhado")
            if i_amt is None or i_order is None or i_status is None:
                continue

            for r in rows[1:]:
                if len(r) <= max(i_amt, i_order, i_status):
                    continue
                status = (r[i_status] or "").strip().lower()
                if status not in ("approved", "closed"):
                    continue
                try:
                    amt = float(r[i_amt] or 0)
                except (ValueError, TypeError):
                    amt = 0
                if amt <= 0:
                    continue
                oid = _norm_oid(r[i_order])
                proj = order_to_proj.get(oid, "NAO_CLASSIFICADO")
                if proj not in result:
                    result[proj] = {"total": 0, "count": 0, "by_status": {}, "by_motivo": {}, "by_sku": {}}
                result[proj]["total"] += amt
                result[proj]["count"] += 1
                result[proj]["by_status"][status] = result[proj]["by_status"].get(status, 0) + 1
                if i_motivo is not None and len(r) > i_motivo:
                    motivo = (r[i_motivo] or "?").strip() or "?"
                    result[proj]["by_motivo"][motivo] = result[proj]["by_motivo"].get(motivo, 0) + 1
                # by_sku breakdown
                sku = order_to_sku.get(oid, "(no SKU)")
                if sku not in result[proj]["by_sku"]:
                    result[proj]["by_sku"][sku] = {
                        "sku": sku,
                        "title": order_to_title.get(oid, ""),
                        "count": 0,
                        "amount": 0.0,
                    }
                result[proj]["by_sku"][sku]["count"] += 1
                result[proj]["by_sku"][sku]["amount"] += amt
    return result


def load_all_nfse() -> list:
    """Load all NFS-e sidecar JSONs from _data/ folders. One file per invoice.
    Dedup by numero. Includes legacy single-file `nfse_shps.json` for back-compat.
    """
    import json as json_mod
    by_numero: dict[str, dict] = {}
    seen_no_num: list[dict] = []
    for month in MONTHS:
        month_dir = DATA_DIR / month
        if not month_dir.exists():
            continue
        for jp in sorted(month_dir.glob("nfse_shps*.json")):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    rec = json_mod.load(f)
            except Exception:
                continue
            num = str(rec.get("numero") or "").strip()
            if num:
                by_numero[num] = rec
            else:
                seen_no_num.append(rec)
        # Fallback: парсим PDF-ы напрямую если нет sidecar'а с тем же numero
        for pdf in sorted(month_dir.glob("nfse_shps*.pdf")):
            parsed = parse_nfse_pdf(pdf)
            if not parsed:
                continue
            num = str(parsed.get("numero") or "").strip()
            if num and num in by_numero:
                continue  # уже есть из sidecar
            if num:
                by_numero[num] = parsed
            else:
                seen_no_num.append(parsed)
    return list(by_numero.values()) + seen_no_num


def load_analise_2025() -> dict:
    """Load Analise.csv with monthly breakdown of revenue and DAS for 2025."""
    import glob
    files = glob.glob(str(DATA_DIR.parent / "Ana*.csv"))
    if not files:
        return {}

    try:
        with open(files[0], "r", encoding="utf-8") as f:
            content = f.read()

        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) < 5:
            return {}

        # Parse header (months)
        header = lines[0].split(";")
        months_pt_to_num = {
            "JANEIRO": "01", "FEVEREIRO": "02", "MARÇO": "03", "ABRIL": "04",
            "MAIO": "05", "JUNHO": "06", "JULHO": "07", "AGOSTO": "08",
            "SETEMBRO": "09", "OUTUBRO": "10", "NOVEMBRO": "11", "DEZEMBRO": "12",
        }
        month_keys = []
        for h in header[1:]:
            h_clean = h.strip().upper().replace("\ufeff", "")
            num = months_pt_to_num.get(h_clean)
            if num:
                month_keys.append(f"2025-{num}")
            else:
                month_keys.append(None)

        def parse_row(line):
            parts = line.split(";")
            values = []
            for p in parts[1:]:
                p_clean = p.strip().replace("R$", "").replace(".", "").replace(",", ".").strip()
                if p_clean == "-" or not p_clean:
                    values.append(0.0)
                else:
                    try:
                        values.append(float(p_clean))
                    except ValueError:
                        values.append(0.0)
            return values

        result = {}
        for line in lines[1:]:
            label = line.split(";")[0].strip().upper()
            values = parse_row(line)
            for i, mk in enumerate(month_keys):
                if mk is None or i >= len(values):
                    continue
                if mk not in result:
                    result[mk] = {"faturamento": 0, "das_total": 0, "das_vendas": 0, "das_servicos": 0}
                if "FATURAMENTO" in label:
                    result[mk]["faturamento"] = values[i]
                elif "DAS SIMPLES" in label or "DAS  SIMPLES" in label:
                    result[mk]["das_total"] = values[i]
                elif "VENDAS" in label:
                    result[mk]["das_vendas"] = values[i]
                elif "SERVI" in label:
                    result[mk]["das_servicos"] = values[i]

        return result
    except Exception:
        return {}


def load_all_das() -> list:
    """Load all DAS from _data/ folders. Prefers sidecar JSON over PDF parsing."""
    import json as json_mod
    das_list = []
    for month in MONTHS:
        # Prefer sidecar JSON (manual entries override PDF)
        json_path = DATA_DIR / month / "das_simples.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
                if data.get("total", 0) > 0:
                    das_list.append(data)
                    continue
            except Exception:
                pass

        # Fallback: parse PDF
        path = _find_file(month, "das_simples")
        if not path or path.suffix.lower() != ".pdf":
            continue
        parsed = parse_das_pdf(path)
        if parsed and parsed.get("total", 0) > 0:
            das_list.append(parsed)
    return das_list


def calculate_trafficstars_fifo() -> dict | None:
    """
    Calculate TrafficStars BRL cost using FIFO method.

    Algorithm:
    1. Build queue of USD purchases (date, USD bought, BRL spent, rate)
       - From approved transfers (C6 Cambio rows)
       - From live C6 BRL data (post 19/03)
    2. For each TS debit (from C6 USD PDF), consume USD from oldest purchase
    3. Calculate weighted BRL cost for each TS payment
    """
    usd_data = parse_c6_usd()
    if not usd_data:
        return None

    # ── Step 1: Build USD purchase queue ──
    # From approved transfers (only C6 Cambio rows with USD/BRL pairs)
    # Hardcoded approved data (same as in generate_dds_estonia)
    approved_purchases = [
        {"date": "06/02/2026", "usd": 20, "brl": 113.50},
        {"date": "06/02/2026", "usd": 100, "brl": 567.49},
        {"date": "06/02/2026", "usd": 2200, "brl": 12180.79},
        {"date": "09/02/2026", "usd": 1600, "brl": 8764.51},
        {"date": "17/02/2026", "usd": 2070, "brl": 11638.29},
        {"date": "23/02/2026", "usd": 2500, "brl": 13905.20},
        {"date": "26/02/2026", "usd": 4100, "brl": 22217.09},
        {"date": "03/03/2026", "usd": 1500, "brl": 8351.03},
        {"date": "03/03/2026", "usd": 4200, "brl": 23459.09},
        {"date": "06/03/2026", "usd": 4000, "brl": 22668.55},
        {"date": "07/03/2026", "usd": 100, "brl": 564.57},
        {"date": "11/03/2026", "usd": 100, "brl": 544.17},
        {"date": "11/03/2026", "usd": 4100, "brl": 22340.45},
        {"date": "13/03/2026", "usd": 4150, "brl": 22923.96},
        {"date": "18/03/2026", "usd": 4100, "brl": 22907.80},
    ]

    purchases = []
    for p in approved_purchases:
        d, m, y = p["date"].split("/")
        purchases.append({
            "date": pd.Timestamp(year=int(y), month=int(m), day=int(d)),
            "usd_remaining": p["usd"],
            "usd_total": p["usd"],
            "brl": p["brl"],
            "rate": p["brl"] / p["usd"],
        })

    # Add live purchases from C6 (BRL câmbio + USD entrada match)
    # Use USD entradas after cutoff with matched BRL câmbio
    cutoff = pd.Timestamp("2026-03-19")

    # Get live BRL câmbio per date
    import csv as csv_mod
    live_brl_by_date = {}
    for month in MONTHS:
        path = _find_file(month, "extrato_c6_brl")
        if not path or path.suffix.lower() != ".csv":
            continue
        try:
            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            header_idx = None
            for li, ln in enumerate(lines):
                if "Data Lançamento" in ln:
                    header_idx = li
                    break
            if header_idx is None:
                continue
            reader = csv_mod.DictReader(lines[header_idx:])
            for row in reader:
                date_str = (row.get("Data Lançamento") or "").strip()
                title = (row.get("Título") or "").strip()
                saida = float(row.get("Saída(R$)", "0") or "0")
                if not date_str or "Câmbio" not in title or saida <= 0:
                    continue
                parts = date_str.split("/")
                day, mon, year = int(parts[0]), int(parts[1]), int(parts[2])
                row_date = pd.Timestamp(year=year, month=mon, day=day)
                if row_date > cutoff:
                    key = f"{day:02d}/{mon:02d}"
                    live_brl_by_date[key] = live_brl_by_date.get(key, 0) + saida
        except Exception:
            pass

    # Match USD entradas with BRL câmbio for live period
    # Need year for USD dates — assume 2026 (extrato is fev-abr/26)
    for tx in usd_data["transactions"]:
        if tx["type"] != "entrada":
            continue
        date_key = tx["date"]  # "DD/MM"
        if date_key not in live_brl_by_date:
            continue
        # Already added in approved? Check year
        day, mon = int(date_key.split("/")[0]), int(date_key.split("/")[1])
        usd_date = pd.Timestamp(year=2026, month=mon, day=day)
        if usd_date <= cutoff:
            continue  # already in approved

        # Calculate proportional BRL based on this USD share
        # If multiple entradas same date, sum first
        # For simplicity: use same rate as the matched BRL câmbio
        brl_sum = live_brl_by_date[date_key]
        # Sum all USD entradas this date
        usd_sum = sum(t["amount"] for t in usd_data["transactions"]
                      if t["type"] == "entrada" and t["date"] == date_key)
        if usd_sum == 0:
            continue
        # Add this entrada (proportional)
        usd_share = tx["amount"]
        brl_share = brl_sum * (usd_share / usd_sum)
        purchases.append({
            "date": usd_date,
            "usd_remaining": usd_share,
            "usd_total": usd_share,
            "brl": brl_share,
            "rate": brl_share / usd_share if usd_share > 0 else 0,
        })

    # Sort purchases by date (FIFO order)
    purchases.sort(key=lambda x: x["date"])

    # ── Step 2: Get ALL debits (TS + personal) sorted by date ──
    all_debits = []
    for tx in usd_data["transactions"]:
        if tx["type"] != "debit":
            continue
        day, mon = int(tx["date"].split("/")[0]), int(tx["date"].split("/")[1])
        all_debits.append({
            "date": pd.Timestamp(year=2026, month=mon, day=day),
            "usd": abs(tx["amount"]),
            "is_ts": "TrafficStars" in tx["desc"],
            "desc": tx["desc"],
        })
    all_debits.sort(key=lambda x: x["date"])

    # ── Step 3: FIFO consume for ALL debits, but track TS separately ──
    ts_with_brl = []
    personal_with_brl = []
    p_idx = 0
    for deb in all_debits:
        usd_needed = deb["usd"]
        brl_cost = 0

        while usd_needed > 0 and p_idx < len(purchases):
            p = purchases[p_idx]
            if p["usd_remaining"] <= 0:
                p_idx += 1
                continue

            consume = min(usd_needed, p["usd_remaining"])
            brl_cost += consume * p["rate"]
            p["usd_remaining"] -= consume
            usd_needed -= consume

            if p["usd_remaining"] <= 0:
                p_idx += 1

        avg_rate = brl_cost / deb["usd"] if deb["usd"] > 0 else 0
        item = {
            "date": deb["date"],
            "usd": deb["usd"],
            "brl": brl_cost,
            "rate": avg_rate,
            "desc": deb["desc"],
            "uncovered_usd": usd_needed,
        }
        if deb["is_ts"]:
            ts_with_brl.append(item)
        else:
            personal_with_brl.append(item)

    # ── Step 4: USD remaining in queue ──
    usd_in_stock = sum(p["usd_remaining"] for p in purchases)
    brl_value_in_stock = sum(p["usd_remaining"] * p["rate"] for p in purchases)

    return {
        "ts_payments": ts_with_brl,
        "personal_payments": personal_with_brl,
        "total_ts_usd": sum(t["usd"] for t in ts_with_brl),
        "total_ts_brl": sum(t["brl"] for t in ts_with_brl),
        "total_personal_usd": sum(t["usd"] for t in personal_with_brl),
        "total_personal_brl": sum(t["brl"] for t in personal_with_brl),
        "usd_in_stock": usd_in_stock,
        "brl_value_in_stock": brl_value_in_stock,
        "purchases_count": len(purchases),
    }


# ─────────────────────────────────────────────
# APPROVED REPORT PARSERS
# ─────────────────────────────────────────────

def _parse_brl_from_csv(text: str) -> float:
    """Extract BRL value from CSV cell like '160.726,71' or '18.136,56'."""
    text = text.strip().strip('"')
    return parse_brl(text)


def parse_approved_artur() -> dict | None:
    """Parse approved Artur balance CSV."""
    # Find newest balance file
    artur_dir = PROJETOS_DIR / "ARTUR"
    if not artur_dir.exists():
        return None
    csvs = sorted(artur_dir.glob("Balanco_Artur_*.csv"))
    if not csvs:
        return None

    content = csvs[-1].read_text(encoding="utf-8")
    lines = content.split("\n")

    data = {
        "report_date": "",
        "period": "",
        # Entradas
        "usdt_total": 0,
        "vendas_bruto": 0,
        "vendas_net": 0,
        "taxas_ml": 0,
        "total_entradas": 0,
        # Saidas
        "mercadoria": 0,
        "publicidade": 0,
        "devolucoes": 0,
        "full_express": 0,
        "das": 0,
        "armazenagem": 0,
        "aluguel": 0,
        "total_saidas": 0,
        # Resultado
        "saldo": 0,
        # Divida
        "divida_empresa": 0,
        # Indicadores
        "vendas_count": 0,
    }

    import csv as csv_mod
    reader = csv_mod.reader(content.splitlines())

    for cells in reader:
        cells = [c.strip() for c in cells]
        line = ",".join(cells)

        if "USDT transferência" in line and len(cells) > 2:
            data["usdt_total"] += _parse_brl_from_csv(cells[2])
        elif "Data do relatório" in line:
            data["report_date"] = cells[0].replace("Data do relatório: ", "")
        elif "Período" in line:
            data["period"] = cells[0].replace("Período: ", "")
        elif "Vendas bruto" in line and len(cells) > 2:
            data["vendas_bruto"] = _parse_brl_from_csv(cells[2])
        elif "Vendas NET" in line and len(cells) > 2:
            data["vendas_net"] = _parse_brl_from_csv(cells[2])
        elif "Taxas ML descontadas" in line and len(cells) > 2:
            data["taxas_ml"] = _parse_brl_from_csv(cells[2])
        elif "TOTAL ENTRADAS" in line and len(cells) > 2:
            data["total_entradas"] = _parse_brl_from_csv(cells[2])
        elif "Subtotal mercadoria" in line and len(cells) > 2:
            data["mercadoria"] = _parse_brl_from_csv(cells[2])
        elif "Subtotal publicidade" in line and len(cells) > 2:
            data["publicidade"] = _parse_brl_from_csv(cells[2])
        elif "Subtotal devolu" in line and len(cells) > 2:
            data["devolucoes"] = _parse_brl_from_csv(cells[2])
        elif "Subtotal Full Express" in line and len(cells) > 2:
            data["full_express"] = _parse_brl_from_csv(cells[2])
        elif line.startswith("DAS") and "Simples" in line and len(cells) > 2:
            # First DAS line, may be partial — replaced below if total found
            if data["das"] == 0:
                data["das"] = _parse_brl_from_csv(cells[2])
        elif "Armazenagem Full" in line and len(cells) > 2:
            data["armazenagem"] = _parse_brl_from_csv(cells[2])
        elif "Aluguel empresa" in line and len(cells) > 2:
            data["aluguel"] = _parse_brl_from_csv(cells[2])
        elif "TOTAL SAÍDAS" in line and len(cells) > 2:
            data["total_saidas"] = _parse_brl_from_csv(cells[2])
        elif "= SALDO ARTUR" in line and len(cells) > 2:
            data["saldo"] = _parse_brl_from_csv(cells[2])
        elif "DÍVIDA (empresa pagou" in line and len(cells) > 2:
            data["divida_empresa"] = _parse_brl_from_csv(cells[2])
        elif line.startswith("TOTAL") and "1934" in line:
            # TOTAL,,1934,"121.347,61"
            try:
                data["vendas_count"] = int(cells[2])
            except (ValueError, IndexError):
                pass

    if data["vendas_bruto"] > 0:
        # Recompute DAS as 4.5% × bruto (correct formula for ARTUR)
        # The CSV file may have outdated DAS value (only first period)
        data["das"] = round(data["vendas_bruto"] * 0.045, 2)
        return data
    return None


def load_collection_mp_legacy() -> pd.DataFrame | None:
    """
    Load and merge ALL collection MP files from _data/ + legacy.
    Deduplicates by operation_id, preferring most recent file (newer = correct).
    """
    from pathlib import Path

    all_files = []
    # _data/ folders
    for month in MONTHS:
        path = DATA_DIR / month / "collection_mp.csv"
        if path.exists():
            all_files.append(path)
    # legacy folder
    legacy_dir = DATA_DIR.parent / "vendas mp"
    if legacy_dir.exists():
        for f in legacy_dir.glob("collection-*.csv"):
            all_files.append(f)

    if not all_files:
        return None

    # Sort: oldest first, newest last (so newest overwrites oldest in dedup)
    all_files.sort(key=lambda p: p.stat().st_mtime)

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f, sep=";", encoding="utf-8")
            df["__source_file"] = f.name
            df["__source_mtime"] = f.stat().st_mtime
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate by operation_id (transaction ID), keep newest occurrence
    op_col = None
    for c in combined.columns:
        if "operation_id" in c.lower():
            op_col = c
            break

    if op_col:
        # keep="last" — combined was concat'd in order (oldest→newest), last = newest
        combined = combined.drop_duplicates(subset=[op_col], keep="last")

    return combined


def get_collection_mp_by_project() -> dict:
    """Aggregate collection MP NET amounts by project (ARTUR/ORG/JOOM/GANZA)."""
    df = load_collection_mp_legacy()
    if df is None:
        return {}

    sku_col = "SKU do produto (seller_custom_field)"
    item_col = "Código do produto (item_id)"
    net_col = "Valor total recebido (net_received_amount)"
    date_col = "Data da compra (date_created)"

    if net_col not in df.columns:
        return {}

    result = {}
    for _, row in df.iterrows():
        sku = str(row.get(sku_col, "")).strip()
        item = str(row.get(item_col, "")).strip()
        proj = get_project_by_sku(sku, item)

        try:
            net = float(row.get(net_col, 0) or 0)
        except (ValueError, TypeError):
            net = 0

        if proj not in result:
            result[proj] = {"net_total": 0, "count": 0, "by_month": {}}
        result[proj]["net_total"] += net
        result[proj]["count"] += 1

        # Parse date for monthly breakdown
        date_str = str(row.get(date_col, ""))
        month_key = None
        try:
            d = pd.to_datetime(date_str.split()[0], dayfirst=True)
            month_key = d.strftime("%Y-%m")
        except Exception:
            pass

        if month_key:
            if month_key not in result[proj]["by_month"]:
                result[proj]["by_month"][month_key] = {"net": 0, "count": 0}
            result[proj]["by_month"][month_key]["net"] += net
            result[proj]["by_month"][month_key]["count"] += 1

    return result


def get_collection_mp_credited_by_period(
    project: str,
    period_from,
    period_to,
    date_field: str = "date_approved",
) -> dict:
    """Реальные зачисления на MP за период.

    Берёт collection_mp.csv (load_collection_mp_legacy), фильтрует по project,
    статус approved, дата `date_field` в [period_from..period_to], суммирует
    `net_received_amount`. Также сводит с vendas_ml.xlsx по order_id чтобы
    показать matched / orphan (есть в MP, нет в vendas, и наоборот).

    date_field:
        "date_approved"  → Data de creditação (когда зачислено в кошелёк)
        "date_released"  → Data de liberação (когда стало доступно к выводу)
    """
    df = load_collection_mp_legacy()
    if df is None or df.empty:
        return {}

    sku_col = "SKU do produto (seller_custom_field)"
    item_col = "Código do produto (item_id)"
    net_col = "Valor total recebido (net_received_amount)"
    status_col = "Status da operação (status)"
    order_col = "Número da venda no Mercado Livre (order_id)"
    date_col = {
        "date_approved": "Data de creditação (date_approved)",
        "date_released": "Data de liberação do dinheiro (date_released)",
    }.get(date_field, "Data de creditação (date_approved)")

    if net_col not in df.columns or date_col not in df.columns:
        return {}

    def _pdate(s):
        if not s or str(s) == "nan":
            return None
        try:
            return pd.to_datetime(str(s).split()[0], dayfirst=True).date()
        except Exception:
            return None

    credited_net = 0.0
    credited_count = 0
    by_order: dict = {}  # order_id → {net, count}
    for _, row in df.iterrows():
        if str(row.get(status_col, "") or "").strip().lower() != "approved":
            continue
        sku = str(row.get(sku_col, "") or "").strip()
        item = str(row.get(item_col, "") or "").strip()
        if get_project_by_sku(sku, item) != project:
            continue
        d = _pdate(row.get(date_col))
        if d is None or d < period_from or d > period_to:
            continue
        try:
            net = float(row.get(net_col, 0) or 0)
        except (ValueError, TypeError):
            net = 0.0
        credited_net += net
        credited_count += 1
        oid = str(row.get(order_col, "") or "").strip().removesuffix(".0")
        if oid:
            if oid not in by_order:
                by_order[oid] = {"net": 0.0, "count": 0}
            by_order[oid]["net"] += net
            by_order[oid]["count"] += 1

    # Сверка с vendas_ml.xlsx по order_id (для того же проекта)
    matched_orders = 0
    matched_net = 0.0
    orphan_in_mp = 0
    orphan_in_mp_net = 0.0
    vendas_orders_in_period: set = set()
    try:
        v_df = load_vendas_ml_report()
    except Exception:
        v_df = None
    if v_df is not None and not v_df.empty:
        sub = v_df[v_df["__project"] == project]
        order_v_col = "N.º de venda"
        if order_v_col in sub.columns:
            vendas_orders = {
                str(o).strip().removesuffix(".0")
                for o in sub[order_v_col].dropna().tolist()
                if str(o).strip()
            }
            vendas_orders_in_period = vendas_orders
            for oid, info in by_order.items():
                if oid in vendas_orders:
                    matched_orders += 1
                    matched_net += info["net"]
                else:
                    orphan_in_mp += 1
                    orphan_in_mp_net += info["net"]

    return {
        "date_field": date_field,
        "period_from": str(period_from),
        "period_to": str(period_to),
        "credited_net": credited_net,
        "credited_count": credited_count,
        "unique_orders": len(by_order),
        "matched_with_vendas": matched_orders,
        "matched_net": matched_net,
        "orphan_in_mp": orphan_in_mp,
        "orphan_in_mp_net": orphan_in_mp_net,
    }


def get_monthly_bruto(project: str) -> dict[str, float]:
    """Return {"YYYY-MM": bruto_BRL} для проекта за все месяцы с данными.

    Источник — vendas_ml.xlsx, та же фильтрация что в build_monthly_pnl_matrix
    (delivered + returned buckets, колонка "Receita por produtos (BRL)").
    Используется для RBT12 (rolling 12-мес. сумма bruto для расчёта Faixa
    в Simples Nacional).

    Пустой dict если данных нет.
    """
    import re as _re
    from datetime import date as _date

    df = load_vendas_ml_report()
    if df is None or df.empty:
        return {}
    sub = df[df["__project"] == project]
    if sub.empty:
        return {}

    pt_months = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
        "outubro": 10, "novembro": 11, "dezembro": 12,
    }

    def _ymd(s):
        m = _re.search(r"(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})", str(s))
        if not m:
            return None
        mn = pt_months.get(m.group(2).lower())
        if not mn:
            return None
        try:
            return _date(int(m.group(3)), mn, int(m.group(1)))
        except (ValueError, TypeError):
            return None

    out: dict[str, float] = {}
    for _, row in sub.iterrows():
        if row.get("__bucket") not in ("delivered", "returned"):
            continue
        d = _ymd(row.get("Data da venda"))
        if d is None:
            continue
        mk = f"{d.year:04d}-{d.month:02d}"
        g = pd.to_numeric(row.get("Receita por produtos (BRL)"), errors="coerce")
        out[mk] = out.get(mk, 0.0) + (0.0 if pd.isna(g) else float(g))
    return out


def rolling_rbt12(bruto_by_month: dict[str, float], target_mk: str) -> float:
    """Сумма bruto за 12 месяцев ПЕРЕД target_mk (не включая сам target_mk).

    Соответствует стандарту LC 123/2006: RBT12 = «Receita Bruta dos últimos 12
    meses» — считается до начала текущего месяца.
    """
    if not target_mk or len(target_mk) < 7:
        return 0.0
    y, m = int(target_mk[:4]), int(target_mk[5:7])
    total = 0.0
    for i in range(1, 13):
        y2, m2 = y, m - i
        while m2 <= 0:
            m2 += 12
            y2 -= 1
        key = f"{y2:04d}-{m2:02d}"
        total += float(bruto_by_month.get(key, 0.0) or 0.0)
    return total


def _build_das_tax_info(proj_meta: dict, das_info_by_month: dict, months: list) -> dict | None:
    """Свернуть per-month das_info в row-level tax_info для matrix.

    Regime / anexo одинаковы для всех месяцев (зависят от настроек проекта),
    а faixa/effective_pct/rbt12 меняются по мере роста истории — их кладём
    в by_month map. UI рендерит общий бейдж и может показывать тултип с
    прогрессией faixa.
    """
    if not das_info_by_month:
        return None
    sample = next(iter(das_info_by_month.values()))
    by_month = {}
    for m in months:
        info = das_info_by_month.get(m)
        if not info:
            continue
        by_month[m] = {
            "faixa": info.get("faixa"),
            "effective_pct": round(float(info.get("effective_pct") or 0.0), 4),
            "rbt12": round(float(info.get("rbt12") or 0.0), 2),
        }
    return {
        "regime": sample.get("regime"),
        "anexo": sample.get("anexo"),
        "icms_pct": sample.get("icms_pct"),
        "state": sample.get("state"),
        "display_pt": sample.get("display_pt"),
        "display_ru": sample.get("display_ru"),
        "exceed_limit": sample.get("exceed_limit", False),
        "by_month": by_month,
    }


def build_monthly_pnl_matrix(project: str) -> dict:
    """Помесячный P&L по проекту из vendas_ml.xlsx + publicidade + armazenagem.

    Возвращает {
        "months": ["2025-09", "2025-10", ...],   # отсортированы
        "years": ["2025", "2026"],
        "rows": [{"label": ..., "section": ..., "values": {month: float}, "total": float}],
    }
    """
    import calendar
    from datetime import date as _date
    import re as _re

    df = load_vendas_ml_report()
    if df is None or df.empty:
        return {"months": [], "years": [], "rows": []}

    sub = df[df["__project"] == project].copy()
    if sub.empty:
        return {"months": [], "years": [], "rows": []}

    pt_months = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
        "outubro": 10, "novembro": 11, "dezembro": 12,
    }

    def _ymd(s):
        m = _re.search(r"(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})", str(s))
        if not m:
            return None
        mn = pt_months.get(m.group(2).lower())
        if not mn:
            return None
        try:
            return _date(int(m.group(3)), mn, int(m.group(1)))
        except (ValueError, TypeError):
            return None

    def _num(v):
        x = pd.to_numeric(v, errors="coerce")
        return 0.0 if pd.isna(x) else float(x)

    # Аккумуляторы по month_key "YYYY-MM"
    months_set: set = set()
    rev_gross: dict = {}
    rev_tarifa: dict = {}
    rev_envios: dict = {}
    rev_cancel: dict = {}
    rev_net: dict = {}
    delivered_cnt: dict = {}
    returned_cnt: dict = {}
    ad_cnt: dict = {}
    ad_net: dict = {}

    for _, row in sub.iterrows():
        d = _ymd(row.get("Data da venda"))
        if d is None:
            continue
        bucket = row.get("__bucket")
        if bucket not in ("delivered", "returned"):
            continue
        mk = f"{d.year:04d}-{d.month:02d}"
        months_set.add(mk)
        g = _num(row.get("Receita por produtos (BRL)"))
        n = _num(row.get("Total (BRL)"))
        tv = _num(row.get("Tarifa de venda e impostos (BRL)"))
        cnc = _num(row.get("Cancelamentos e reembolsos (BRL)"))
        rev_gross[mk] = rev_gross.get(mk, 0.0) + g
        rev_tarifa[mk] = rev_tarifa.get(mk, 0.0) + tv
        rev_cancel[mk] = rev_cancel.get(mk, 0.0) + cnc
        rev_net[mk] = rev_net.get(mk, 0.0) + n
        if bucket == "delivered":
            envios = max(g + tv - n, 0.0)  # тождество
            rev_envios[mk] = rev_envios.get(mk, 0.0) - envios  # как расход
            delivered_cnt[mk] = delivered_cnt.get(mk, 0) + 1
            if str(row.get("Venda por publicidade", "")).strip().lower() == "sim":
                ad_cnt[mk] = ad_cnt.get(mk, 0) + 1
                ad_net[mk] = ad_net.get(mk, 0.0) + n
        else:
            returned_cnt[mk] = returned_cnt.get(mk, 0) + 1

    months = sorted(months_set)
    if not months:
        return {"months": [], "years": [], "rows": []}

    years = sorted({m[:4] for m in months})

    # Расходы по месяцам — publicidade, armazenagem, DAS (4.5%×bruto), aluguel (пропорц)
    _all_projects = load_projects() or {}
    proj_meta_raw = _all_projects.get(project, {}) or {}
    # Company-level inheritance для tax_regime/anexo (same company_cnpj).
    from .tax_brazil import resolve_tax_settings as _resolve_tax
    proj_meta = _resolve_tax(proj_meta_raw, _all_projects)
    approved_data = get_approved_data(project) or {}

    # Aluguel — приоритет источников (нормализуем в BRL/мес):
    #   1) project.aluguel_mensal (BRL/мес)
    #   2) project.rental.rate_usd × period (quarter|month) × 5.46 USD→BRL
    #   3) fallback baseline_overrides.aluguel / baseline_days (legacy)
    mensal = float(proj_meta.get("aluguel_mensal", 0) or 0)
    if mensal <= 0:
        rental = proj_meta.get("rental") or {}
        if isinstance(rental, dict):
            rate_usd = float(rental.get("rate_usd", 0) or 0)
            if rate_usd > 0:
                period_kind = (rental.get("period") or "month").lower()
                mensal_usd = rate_usd / 3.0 if period_kind.startswith("quart") else rate_usd
                mensal = mensal_usd * 5.46
    launch_date_obj = None
    launch_str = (proj_meta.get("launch_date") or "").strip()[:10]
    if launch_str:
        try:
            from datetime import datetime as _dt
            launch_date_obj = _dt.strptime(launch_str, "%Y-%m-%d").date()
        except Exception:
            launch_date_obj = None

    aluguel_full = float(approved_data.get("aluguel", 0) or 0)
    # baseline period длительность для пропорции aluguel (legacy fallback)
    bp_str = proj_meta.get("report_period", "")
    baseline_days = 206  # fallback ARTUR
    if bp_str and "/" in bp_str:
        try:
            from datetime import datetime as _dt
            parts = [p.strip() for p in bp_str.split("/")]
            bp_start = _dt.strptime(parts[0], "%Y-%m-%d").date()
            bp_end = _dt.strptime(parts[1], "%Y-%m-%d").date()
            baseline_days = (bp_end - bp_start).days + 1
        except Exception:
            pass

    publi_by_month: dict = {}
    armaz_by_month: dict = {}
    das_by_month: dict = {}
    das_info_by_month: dict = {}  # per-month {faixa, effective_pct, rbt12, ...}
    aluguel_by_month: dict = {}
    for mk in months:
        y, mo = int(mk[:4]), int(mk[5:7])
        last_day = calendar.monthrange(y, mo)[1]
        pf, pt_ = _date(y, mo, 1), _date(y, mo, last_day)
        try:
            publi_by_month[mk] = float(get_publicidade_by_period(project, pf, pt_).get("total", 0.0))
        except Exception:
            publi_by_month[mk] = 0.0
        try:
            armaz_by_month[mk] = float(get_armazenagem_by_period(project, pf, pt_).get("total", 0.0))
        except Exception:
            armaz_by_month[mk] = 0.0
        # DAS — по выбранному tax_regime (Simples Nacional Anexo I/II/III
        # или Lucro Presumido + ICMS). Если режим не задан — legacy 4.5%.
        # RBT12 = сумма bruto за 12 мес. ПЕРЕД данным месяцем.
        _rbt12 = rolling_rbt12(rev_gross, mk)
        _das_info = compute_das(proj_meta, rev_gross.get(mk, 0.0), _rbt12)
        das_by_month[mk] = _das_info["das_brl"]
        das_info_by_month[mk] = _das_info
        # Aluguel — приоритет: project.aluguel_mensal + launch_date, иначе fallback.
        days_in_month = last_day
        if mensal > 0:
            if launch_date_obj and launch_date_obj > pt_:
                # Проект ещё не запущен в этом месяце — аренды нет.
                aluguel_by_month[mk] = 0.0
            elif launch_date_obj and launch_date_obj > pf:
                # Частичный первый месяц — пропорционально дням после запуска.
                accrued_days = (pt_ - launch_date_obj).days + 1
                aluguel_by_month[mk] = round(mensal * accrued_days / days_in_month, 2)
            else:
                aluguel_by_month[mk] = mensal
        else:
            aluguel_by_month[mk] = round(aluguel_full * days_in_month / baseline_days, 2) if aluguel_full > 0 else 0.0

    def _row(label: str, section: str, by_month: dict, sign: int = 1, key: str = "") -> dict:
        values = {m: sign * float(by_month.get(m, 0.0)) for m in months}
        return {
            "key": key,
            "label": label,
            "section": section,
            "values": values,
            "total": sum(values.values()),
        }

    # Деривативы — учитываем все 4 расхода
    op_profit = {
        m: rev_net.get(m, 0.0)
           - publi_by_month.get(m, 0.0)
           - armaz_by_month.get(m, 0.0)
           - das_by_month.get(m, 0.0)
           - aluguel_by_month.get(m, 0.0)
        for m in months
    }
    margin = {
        m: (op_profit[m] / rev_net[m] * 100.0) if rev_net.get(m, 0.0) else 0.0
        for m in months
    }
    # % возвратов = returned / (delivered + returned) × 100
    return_pct = {}
    for m in months:
        d_c = delivered_cnt.get(m, 0)
        r_c = returned_cnt.get(m, 0)
        return_pct[m] = (r_c / (d_c + r_c) * 100.0) if (d_c + r_c) else 0.0
    # % ДРР = publicidade / revenue_net × 100
    drr_pct = {
        m: (publi_by_month.get(m, 0.0) / rev_net[m] * 100.0) if rev_net.get(m, 0.0) else 0.0
        for m in months
    }
    # ROI = (revenue_net − publicidade) / publicidade × 100
    roi_pct = {}
    for m in months:
        pub = publi_by_month.get(m, 0.0)
        net = rev_net.get(m, 0.0)
        roi_pct[m] = ((net - pub) / pub * 100.0) if pub > 0 else 0.0

    rows = [
        _row("pnl_rev_gross", "REVENUE", rev_gross, key="rev_gross"),
        _row("pnl_tarifa_venda", "REVENUE", rev_tarifa, key="tarifa_venda"),
        _row("pnl_envios", "REVENUE", rev_envios, key="envios"),
        _row("pnl_cancelamentos", "REVENUE", rev_cancel, key="cancelamentos"),
        _row("pnl_net_revenue", "REVENUE", rev_net, key="net_revenue"),
        {
            "key": "ads_subtotal",
            "label": "pnl_ads_subtotal",
            "section": "REVENUE",
            "values": {m: float(ad_net.get(m, 0.0)) for m in months},
            "total": sum(ad_net.values()),
            "is_info": True,
        },
        _row("pnl_publicidade", "EXPENSES", publi_by_month, sign=-1, key="publicidade"),
        _row("pnl_armazenagem", "EXPENSES", armaz_by_month, sign=-1, key="armazenagem"),
        {
            **_row("pnl_das", "EXPENSES", das_by_month, sign=-1, key="das"),
            "tax_info": _build_das_tax_info(proj_meta, das_info_by_month, months),
        },
        _row("pnl_aluguel", "EXPENSES", aluguel_by_month, sign=-1, key="aluguel"),
        {
            "key": "op_profit",
            "label": "pnl_op_profit",
            "section": "SUMMARY",
            "values": op_profit,
            "total": sum(op_profit.values()),
            "is_total": True,
        },
        {
            "key": "margin_pct",
            "label": "pnl_margin_pct",
            "section": "SUMMARY",
            "values": margin,
            "total": (sum(op_profit.values()) / sum(rev_net.values()) * 100.0) if sum(rev_net.values()) else 0.0,
            "is_pct": True,
        },
        {
            "key": "orders_delivered",
            "label": "pnl_orders_delivered",
            "section": "SUMMARY",
            "values": {m: float(delivered_cnt.get(m, 0)) for m in months},
            "total": float(sum(delivered_cnt.values())),
            "is_count": True,
        },
        {
            "key": "orders_ads",
            "label": "pnl_orders_ads",
            "section": "SUMMARY",
            "values": {m: float(ad_cnt.get(m, 0)) for m in months},
            "total": float(sum(ad_cnt.values())),
            "is_count": True,
            "is_info": True,
        },
        {
            "key": "orders_returned",
            "label": "pnl_orders_returned",
            "section": "SUMMARY",
            "values": {m: float(returned_cnt.get(m, 0)) for m in months},
            "total": float(sum(returned_cnt.values())),
            "is_count": True,
            "is_info": True,
        },
        {
            "key": "return_pct",
            "label": "pnl_return_pct",
            "section": "SUMMARY",
            "values": return_pct,
            "total": (
                sum(returned_cnt.values()) / (sum(delivered_cnt.values()) + sum(returned_cnt.values())) * 100.0
                if (sum(delivered_cnt.values()) + sum(returned_cnt.values())) else 0.0
            ),
            "is_pct": True,
        },
        {
            "key": "drr_pct",
            "label": "pnl_drr_pct",
            "section": "SUMMARY",
            "values": drr_pct,
            "total": (sum(publi_by_month.values()) / sum(rev_net.values()) * 100.0) if sum(rev_net.values()) else 0.0,
            "is_pct": True,
        },
        {
            "key": "roi_ads",
            "label": "pnl_roi_ads",
            "section": "SUMMARY",
            "values": roi_pct,
            "total": (
                (sum(rev_net.values()) - sum(publi_by_month.values())) / sum(publi_by_month.values()) * 100.0
                if sum(publi_by_month.values()) > 0 else 0.0
            ),
            "is_pct": True,
        },
    ]

    return {
        "months": months,
        "years": years,
        "rows": rows,
    }


def get_mp_credited_for_orders(order_ids) -> dict:
    """Сколько денег уже реально упало в MP для заданного списка order_id.

    Берёт collection_mp.csv (status=approved) БЕЗ фильтра по дате — нужны все
    зачисления, включая те что пришли уже после конца периода.
    """
    df = load_collection_mp_legacy()
    if df is None or df.empty or not order_ids:
        return {"credited_net": 0.0, "matched_orders": 0, "by_order": {}}

    net_col = "Valor total recebido (net_received_amount)"
    status_col = "Status da operação (status)"
    order_col = "Número da venda no Mercado Livre (order_id)"
    if net_col not in df.columns or order_col not in df.columns:
        return {"credited_net": 0.0, "matched_orders": 0, "by_order": {}}

    target = {str(o).strip().removesuffix(".0") for o in order_ids if str(o).strip()}
    by_order: dict = {}
    for _, row in df.iterrows():
        if str(row.get(status_col, "") or "").strip().lower() != "approved":
            continue
        oid = str(row.get(order_col, "") or "").strip().removesuffix(".0")
        if not oid or oid not in target:
            continue
        try:
            net = float(row.get(net_col, 0) or 0)
        except (ValueError, TypeError):
            net = 0.0
        by_order[oid] = by_order.get(oid, 0.0) + net

    return {
        "credited_net": sum(by_order.values()),
        "matched_orders": len(by_order),
        "by_order": by_order,
    }


def _stock_full_title_column(df: pd.DataFrame) -> str | None:
    """Колонка названия в stock Full — только явные «título do anúncio» / nome produto (не числовой ID)."""
    for c in df.columns:
        cl = str(c).lower()
        if ("título" in cl or "titulo" in cl) and ("anúncio" in cl or "anuncio" in cl):
            return str(c)
    for c in df.columns:
        cl = str(c).lower()
        if "nome" in cl and "produto" in cl:
            return str(c)
    return None


def load_stock_full() -> dict:
    """
    Load stock_full XLSX from _data/, parse all sheets and aggregate by project.
    Returns: {project: {"total_units": N, "by_sku": {sku: qty}, "sku_titles": {sku: str}}}
    """
    from pathlib import Path

    # Find newest stock file
    stock_files = []
    for month in MONTHS:
        for path in (DATA_DIR / month).glob("stock_full*.xlsx") if (DATA_DIR / month).exists() else []:
            stock_files.append(path)
    # Also check Downloads or root
    legacy_paths = list(Path.home().glob("Downloads/stock_general_full*.xlsx"))
    stock_files.extend(legacy_paths)

    if not stock_files:
        return {}

    newest = max(stock_files, key=lambda p: p.stat().st_mtime)

    try:
        xl = pd.ExcelFile(newest)
    except Exception:
        return {}

    # Aggregate units across all sheets (Boa qualidade, Para impulsionar, etc)
    all_skus: dict[str, int] = {}  # sku → total qty
    sku_titles: dict[str, str] = {}  # sku → название (первое непустое)
    for sheet in xl.sheet_names:
        if sheet == "Resumo":
            continue
        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=5)
            df = df[df.get("SKU").notna()] if "SKU" in df.columns else df
            if "SKU" not in df.columns:
                continue

            # Find quantity column — typically "Unidades de boa qualidade.1" or similar
            qty_col = None
            for c in df.columns:
                cl = str(c).lower()
                if ("unidades" in cl or "estoque" in cl) and ".1" in str(c):
                    qty_col = c
                    break
            if qty_col is None:
                for c in df.columns:
                    if "unidades" in str(c).lower() or "qtd" in str(c).lower():
                        qty_col = c
                        break
            if qty_col is None:
                continue

            title_col = _stock_full_title_column(df)

            for _, row in df.iterrows():
                sku = str(row.get("SKU", "")).strip()
                if not sku or sku == "nan":
                    continue
                try:
                    qty = float(row.get(qty_col, 0) or 0)
                except (ValueError, TypeError):
                    qty = 0
                if qty > 0:
                    all_skus[sku] = all_skus.get(sku, 0) + int(qty)
                    if title_col and sku not in sku_titles:
                        tit = str(row.get(title_col, "") or "").strip()
                        if tit and tit.lower() != "nan":
                            sku_titles[sku] = tit[:220]
        except Exception:
            continue

    # Group by project
    result: dict[str, dict] = {}
    for sku, qty in all_skus.items():
        proj = get_project_by_sku(sku, "")
        if proj not in result:
            result[proj] = {"total_units": 0, "by_sku": {}, "sku_titles": {}}
        result[proj]["total_units"] += qty
        result[proj]["by_sku"][sku] = result[proj]["by_sku"].get(sku, 0) + qty

    for proj, block in result.items():
        titles_map = {}
        for s in block["by_sku"]:
            titles_map[s] = sku_titles.get(s, "")
        block["sku_titles"] = titles_map

    return result


def get_artur_monthly_pnl() -> list:
    """Build monthly P&L for ARTUR using Vendas ML (bruto) + collection MP (NET)."""
    opiu = generate_opiu_from_vendas()
    artur = opiu.get("ARTUR", {})
    by_month = artur.get("by_month", {})

    # NET берём из collection MP (как в утверждённом отчёте)
    coll_mp = get_collection_mp_by_project()
    artur_mp = coll_mp.get("ARTUR", {})
    mp_by_month = artur_mp.get("by_month", {})

    # Approved data for known monthly costs (publicidade, devoluções, full express)
    detailed = parse_approved_artur_detailed()

    # Index publicidade by month
    pub_by_month = {}
    if detailed:
        for item in detailed.get("publicidade", []):
            desc = item["description"].lower()
            # "publicidade set–nov/25" → distribute over 3 months
            if "set" in desc and "nov" in desc:
                for m in ["2025-09", "2025-10", "2025-11"]:
                    pub_by_month[m] = pub_by_month.get(m, 0) + item["value"] / 3
            elif "dez/25" in desc:
                pub_by_month["2025-12"] = pub_by_month.get("2025-12", 0) + item["value"]
            elif "23/dez" in desc and "mar" in desc:
                for m in ["2025-12", "2026-01", "2026-02", "2026-03"]:
                    pub_by_month[m] = pub_by_month.get(m, 0) + item["value"] / 4

    # Devoluções by month
    dev_by_month = {}
    if detailed:
        month_map = {"set/25": "2025-09", "out/25": "2025-10", "nov/25": "2025-11",
                     "dez/25": "2025-12", "jan/26": "2026-01", "fev/26": "2026-02", "mar/26": "2026-03"}
        for item in detailed.get("devolucoes", []):
            for k, m in month_map.items():
                if k in item["description"].lower():
                    dev_by_month[m] = item["value"]
                    break

    # Full Express by month
    fe_by_month = {}
    if detailed:
        month_map_fe = {"ago/25": "2025-08", "out/25": "2025-10", "dez/25": "2025-12",
                        "jan/26": "2026-01", "fev/26": "2026-02", "mar/26": "2026-03"}
        for item in detailed.get("full_express", []):
            for k, m in month_map_fe.items():
                if k in item["description"].lower():
                    fe_by_month[m] = item["value"]
                    break

    # COGS (Mercadoria) — себестоимость ПРОДАННОГО товара
    # Считаем как % от bruto (по индикатору утверждённого отчёта: 56,2%)
    # Это НЕ закупка товара (она в ДДС), а себестоимость только проданных единиц
    COGS_RATIO_ARTUR = 0.562
    mer_by_month = {}  # будет заполнено в цикле как bruto × COGS_RATIO

    # DAS = 4.5% × bruto каждого месяца (Anexo I Simples Nacional)
    DAS_RATE_ARTUR = 0.045
    das_by_month = {}

    # Build monthly rows
    months_to_show = sorted(set(list(by_month.keys()) + list(pub_by_month.keys()) + list(mer_by_month.keys()) + list(fe_by_month.keys())))
    rows = []
    for m in months_to_show:
        bm = by_month.get(m, {})
        receita_bruta = bm.get("receita_bruta", 0)
        taxas_ml = abs(bm.get("tarifa_venda", 0)) + abs(bm.get("tarifa_envio", 0))
        cancelam = abs(bm.get("cancelamentos", 0))
        # NET берём из collection MP (точное значение в кассе)
        net_ml = mp_by_month.get(m, {}).get("net", 0) or bm.get("total_net", 0)

        # COGS = 56.2% от bruto (себестоимость проданного, не закупка)
        mer = receita_bruta * COGS_RATIO_ARTUR
        mer_by_month[m] = mer
        pub = pub_by_month.get(m, 0)
        dev = dev_by_month.get(m, 0)
        fe = fe_by_month.get(m, 0)
        # DAS = 4.5% от bruto этого месяца
        das = receita_bruta * DAS_RATE_ARTUR
        das_by_month[m] = das

        custo_total = mer + pub + dev + fe + das
        lucro = net_ml - custo_total

        rows.append({
            "Mês": m,
            "Vendas bruto": receita_bruta,
            "(-) Taxas ML": -taxas_ml,
            "= NET (MP)": net_ml,
            "(-) COGS (56,2%)": -mer,
            "(-) Publicidade": -pub,
            "(-) Devoluções": -dev,
            "(-) Full Express": -fe,
            "(-) DAS (4,5%)": -das,
            "= Lucro": lucro,
        })
    return rows


def parse_approved_artur_detailed() -> dict | None:
    """Parse approved Artur balance CSV with all line items by section."""
    artur_dir = PROJETOS_DIR / "ARTUR"
    if not artur_dir.exists():
        return None
    csvs = sorted(artur_dir.glob("Balanco_Artur_*.csv"))
    if not csvs:
        return None

    content = csvs[-1].read_text(encoding="utf-8")
    import csv as csv_mod
    reader = csv_mod.reader(content.splitlines())

    sections = {
        "entradas": [],          # USDT + Vendas
        "mercadoria": [],         # 2.1
        "publicidade": [],        # 2.2
        "devolucoes": [],         # 2.3
        "full_express": [],       # 2.4 Full Express
        "operacional_outros": [], # DAS, Armazenagem, Aluguel
    }

    current = None
    for cells in reader:
        cells = [c.strip() for c in cells]
        line = ",".join(cells)
        if not any(cells):
            continue

        # Section markers
        if "1. ENTRADAS" in line:
            current = "entradas"
            continue
        if "2.1 MERCADORIA" in line:
            current = "mercadoria"
            continue
        if "2.2 PUBLICIDADE" in line:
            current = "publicidade"
            continue
        if "2.3 DEVOLU" in line:
            current = "devolucoes"
            continue
        if "2.4 OPERACIONAL" in line:
            current = "full_express"
            continue
        if "3. RESULTADO" in line or "4. INFORMATIVO" in line or "5. RESUMO" in line:
            current = None
            continue

        # Skip subtotals and totals
        first = cells[0] if cells else ""
        if any(s in first.lower() for s in ["subtotal", "total entradas", "total sa"]):
            continue
        if not first or "═══" in first or first.startswith(","):
            continue

        if current and len(cells) >= 3 and cells[2]:
            try:
                val = parse_brl(cells[2])
                fonte = cells[3] if len(cells) > 3 else ""
                if val > 0:
                    item = {"description": first, "value": val, "source": fonte}

                    # Special handling for "operacional" — split between Full Express and Outros
                    if current == "full_express":
                        if "Full Express" in first:
                            sections["full_express"].append(item)
                        else:
                            sections["operacional_outros"].append(item)
                    else:
                        sections[current].append(item)
            except Exception:
                pass

    return sections


def get_approved_data(project: str) -> dict | None:
    """Get data from approved reports for a project."""
    if project == "ARTUR":
        data = parse_approved_artur()
        # Fallback: fill missing/zero expense values from baseline_overrides in projects_db.json
        overrides = (load_projects().get(project, {}) or {}).get("baseline_overrides", {}) or {}
        if data is None and overrides:
            data = {}
        if data is not None:
            for key, val in overrides.items():
                if not data.get(key):
                    data[key] = val
        return data
    elif project == "ESTONIA":
        # Estonia data is in generate_opiu_estonia()
        opiu = generate_opiu_estonia()
        dds = generate_dds_estonia()
        bal = generate_balance_estonia()
        return {"opiu": opiu, "dds": dds, "balance": bal}
    # Любой другой ecom-проект: возвращаем baseline_overrides если есть
    overrides = (load_projects().get(project, {}) or {}).get("baseline_overrides")
    if overrides:
        return dict(overrides)
    return None


# ─────────────────────────────────────────────
# DATAFRAME CONVERTERS
# ─────────────────────────────────────────────

def opiu_to_dataframe(opiu: dict) -> pd.DataFrame:
    rows = []
    for proj, data in sorted(opiu.items()):
        rows.append({
            "Projeto": proj,
            "Vendas": data["vendas_count"],
            "Receita Bruta": data["receita_bruta"],
            "Tarifa Venda": data["tarifa_venda"],
            "Tarifa Envio": data["tarifa_envio"],
            "Cancelamentos": data["cancelamentos"],
            "NET (MP)": data["total_net"],
            "% Ads": f"{data['ads_count']/data['vendas_count']*100:.0f}%" if data["vendas_count"] else "0%",
        })
    return pd.DataFrame(rows)


def opiu_monthly_to_dataframe(opiu: dict, project: str) -> pd.DataFrame:
    if project not in opiu:
        return pd.DataFrame()
    by_month = opiu[project]["by_month"]
    rows = []
    for month in MONTHS:
        if month in by_month:
            m = by_month[month]
            rows.append({
                "Mes": month,
                "Vendas": m["vendas"],
                "Receita Bruta": m["receita_bruta"],
                "Taxas ML": m["tarifa_venda"] + m["tarifa_envio"],
                "Cancelam.": m["cancelamentos"],
                "NET (MP)": m["total_net"],
            })
        else:
            rows.append({"Mes": month, "Vendas": 0, "Receita Bruta": 0,
                         "Taxas ML": 0, "Cancelam.": 0, "NET (MP)": 0})
    return pd.DataFrame(rows)


def dds_ecom_to_dataframe(dds: dict) -> pd.DataFrame:
    rows = []
    running = 0
    for month in MONTHS:
        bm = dds["by_month"].get(month)
        if bm:
            running += bm["net_flow"]
            rows.append({
                "Mes": month,
                "Entradas (NET)": bm["inflows_vendas_net"],
                "Fluxo Mes": bm["net_flow"],
                "Saldo Acum.": running,
            })
        else:
            rows.append({"Mes": month, "Entradas (NET)": 0, "Fluxo Mes": 0, "Saldo Acum.": running})
    return pd.DataFrame(rows)


def estonia_opiu_monthly_to_dataframe(opiu_est: dict) -> pd.DataFrame:
    rows = []
    for month in MONTHS:
        bm = opiu_est["by_month"].get(month)
        if bm:
            rows.append({
                "Mes": month,
                "Invoices": bm["count"],
                "Bruto": bm["gross"],
                "Comissao (receita)": bm["commission"],
                "DAS (imposto)": -bm.get("das", 0),
                "Lucro mes": bm["commission"] - bm.get("das", 0),
            })
        else:
            rows.append({"Mes": month, "Invoices": 0, "Bruto": 0,
                         "Comissao (receita)": 0, "DAS (imposto)": 0, "Lucro mes": 0})
    return pd.DataFrame(rows)
