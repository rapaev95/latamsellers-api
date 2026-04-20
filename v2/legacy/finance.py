"""
Финансовый слой расчётов: чистый Python поверх reports.py.

Модуль строит явные dataclass-отчёты (P&L, Cash Flow, Balance) для
ecom-проектов. UI (`app.py` / `report_views.py`) только рендерит эти
объекты — никаких вычислений в представлении.

Принципы:
- Нет хардкода денежных значений: всё из reports.py + projects_db.json.
- USDT-инвестиции — это финансирование (Equity), а не выручка (Revenue).
- Баланс проверяется на сходимость: Активы ≈ Капитал + Обязательства.
- Период фильтрации передаётся параметром, не зашит.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .config import load_projects
from .sku_catalog import assess_stock_for_project
from .reports import (
    aggregate_classified_by_project,
    generate_opiu_from_vendas,
    get_approved_data,
    get_collection_mp_by_project,
    get_devolucoes_by_project,
    load_stock_full,
)  # get_collection_mp_by_project нужен для compute_cashflow


# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class PnLLine:
    label: str
    amount_brl: float
    is_total: bool = False
    note: str = ""


@dataclass
class PnLReport:
    project: str
    period: tuple[date, date]
    revenue_gross: float            # Vendas bruto (Preço × Unidades) из Vendas ML
    taxas_ml: float                 # bruto - net (удержания ML)
    revenue_net: float              # NET из collection MP
    operating_expenses: list[PnLLine]
    operating_profit: float         # net - sum(opex), без COGS
    cogs: float | None              # None если нет cost_per_unit
    net_profit: float | None        # operating_profit - cogs
    vendas_count: int = 0
    margin_pct: float = 0.0
    # Расчётные данные по DAS (Simples Nacional / Lucro Presumido / legacy)
    # — faixa, aliquot effective, ICMS. Используется UI для бейджа рядом
    # со строкой DAS в operating_expenses.
    tax_info: dict | None = None


@dataclass
class CashFlowReport:
    project: str
    period: tuple[date, date]
    opening_balance: float          # стартовое сальдо (обычно 0)
    inflows_operating: float        # operating profit за период
    inflows_count: int              # кол-во продаж за период
    inflows_financing: float        # USDT инвестиции собственника
    inflows_partner: float = 0.0    # ручные поступления от партнёра
    outflows_operating: float = 0.0 # supplier (закупки)
    outflows_other: float = 0.0     # ручные прочие расходы
    closing_balance: float = 0.0
    new_transactions: list = field(default_factory=list)  # supplier tx
    partner_txs: list = field(default_factory=list)
    other_expenses_txs: list = field(default_factory=list)


@dataclass
class BalanceReport:
    """Flow-based balance: Входы − Выходы = Сальдо (как в утверждённом ARTUR CSV)."""
    project: str
    as_of: date
    # ВХОДЫ
    inflow_usdt_brl: float
    inflow_usdt_usd: float
    inflow_sales_net: float
    inflow_sales_count: int
    inflows_total: float
    # ВЫХОДЫ
    outflow_mercadoria: float       # товар (закупка)
    outflow_publicidade: float
    outflow_devolucoes: float       # возвраты + логистика
    outflow_full_express: float
    outflow_das: float
    outflow_armazenagem: float
    outflow_aluguel: float
    outflows_total: float
    # САЛЬДО
    saldo: float                    # inflows - outflows
    pending_rental_usd: float
    pending_rental_brl: float
    saldo_final: float              # saldo - pending_rental
    # Метаданные
    cost_per_unit: float | None
    stock_units: int
    # Оценка стока (каталог SKU + опционально avg_cost_per_unit_brl)
    stock_value_brl: float = 0.0
    stock_missing_skus: list = field(default_factory=list)
    stock_missing_units: int = 0
    stock_by_supplier_type: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _to_ddmmyyyy(d: date | None) -> str | None:
    if d is None:
        return None
    return d.strftime("%d/%m/%Y")


def _parse_baseline_date(proj_data: dict) -> date | None:
    raw = proj_data.get("baseline_date")
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def get_project_meta(project: str) -> dict:
    return load_projects().get(project, {}) or {}


def get_baseline_date(project: str) -> date | None:
    return _parse_baseline_date(get_project_meta(project))


def get_project_start_date(project: str) -> date | None:
    """Return the actual start date of the project (beginning of report_period).

    Falls back to baseline_date, then None.
    """
    meta = get_project_meta(project)
    rp = meta.get("report_period", "")
    if rp and "/" in rp:
        try:
            start_str = rp.split("/")[0].strip()
            return datetime.strptime(start_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass
    return _parse_baseline_date(meta)


# ─────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────

def compute_pnl(project: str, period: tuple[date, date], basis: str = "accrual") -> PnLReport:
    """
    Строит P&L из Vendas ML (bruto) + collection MP (net) + утверждённых
    расходных статей (publicidade, devoluções, full_express, das, armazenagem,
    aluguel) с fallback из projects_db.json[baseline_overrides].

    NB: текущая версия использует полные накопленные значения от Vendas ML
    и collection MP; period нужен только для фильтра ДДС/Баланса. Период
    хранится для отображения, точечная фильтрация revenue по period — TODO.
    """
    # Источник истины — vendas_ml.xlsx (Estado=delivered+returned, в выбранном периоде).
    # Та же логика что в Vendas ML вкладке (_vendas_ml_pnl_by_period).
    from .reports import (
        load_vendas_ml_report,
        get_publicidade_by_period,
        get_armazenagem_by_period,
    )
    import re as _re

    pt_months = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
        "outubro": 10, "novembro": 11, "dezembro": 12,
    }

    def _pdate(s):
        g = _re.search(r"(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})", str(s))
        if not g:
            return None
        mn = pt_months.get(g.group(2).lower())
        if not mn:
            return None
        try:
            return date(int(g.group(3)), mn, int(g.group(1)))
        except (ValueError, TypeError):
            return None

    def _num(v) -> float:
        import pandas as _pd2
        x = _pd2.to_numeric(v, errors="coerce")
        return 0.0 if _pd2.isna(x) else float(x)

    # Аггрегируем delivered + returned за период (та же логика как
    # _vendas_ml_pnl_by_period в report_views.py).
    df = load_vendas_ml_report()
    d_gross = d_net = d_tv = 0.0
    r_gross = r_net = r_tv = r_cnc = 0.0
    d_count = 0
    r_count = 0
    period_start, period_end = period
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            if row.get("__project") != project:
                continue
            bucket = row.get("__bucket")
            if bucket not in ("delivered", "returned"):
                continue
            d = _pdate(row.get("Data da venda"))
            if d is None or d < period_start or d > period_end:
                continue
            g = _num(row.get("Receita por produtos (BRL)"))
            n = _num(row.get("Total (BRL)"))
            tv = _num(row.get("Tarifa de venda e impostos (BRL)"))
            cnc = _num(row.get("Cancelamentos e reembolsos (BRL)"))
            if bucket == "delivered":
                d_gross += g
                d_net += n
                d_tv += tv
                d_count += 1
            else:
                r_gross += g
                r_net += n
                r_tv += tv
                r_cnc += cnc
                r_count += 1

    # Revenue bruto = все отгрузки (delivered + returned), как в помесячной
    # матрице `build_monthly_pnl_matrix` (row "Receita por produtos (bruto)").
    # Это стандарт бухгалтерии: bruto = до вычета возвратов. Затем cancel
    # уменьшает NET. KPI-карточки и матрица показывают одно и то же число.
    revenue_gross = d_gross + r_gross
    # NET revenue выводим как bruto − tarifa(обе группы) − envios(delivered) − cancel,
    # что 1-в-1 совпадает с «= Выручка NET» в матрице.
    envios_dif = max(d_gross + d_tv - d_net, 0.0)
    tarifa_venda_abs = abs(d_tv) + abs(r_tv)
    cancel_abs = abs(r_cnc)
    revenue_net = revenue_gross - tarifa_venda_abs - envios_dif - cancel_abs
    vendas_count = d_count + r_count
    taxas_ml = abs(d_tv) + envios_dif  # KPI «taxas» — только по доставленным (как было)

    # Returned: убыток от возвратов (returned NET обычно ~0, потери в Cancelamentos)
    returned_loss = abs(r_cnc) + abs(r_tv) - r_gross  # реальный убыток от возвратов
    if returned_loss < 0:
        returned_loss = 0.0

    approved = get_approved_data(project) or {}

    # DAS — считаем по выбранному налоговому режиму проекта (Simples Anexo
    # I/II/III с настоящими faixas или Lucro Presumido + ICMS по штату). Для
    # проектов без tax_regime сохраняем legacy 4.5% (backward compat).
    # RBT12 = сумма bruto за 12 мес. перед текущим (LC 123/2006).
    proj_meta_for_tax = get_project_meta(project) or {}
    rbt12 = 0.0
    try:
        from .reports import get_monthly_bruto
        bruto_by_month = get_monthly_bruto(project) or {}
        cur_mk = f"{period_end.year:04d}-{period_end.month:02d}"
        # 12 месяцев перед period_end (не включая текущий)
        from .reports import rolling_rbt12
        rbt12 = rolling_rbt12(bruto_by_month, cur_mk)
    except Exception:
        pass

    from .tax_brazil import compute_das as _compute_das, resolve_tax_settings
    from .config import load_projects as _load_projects
    # Наследуем режим от других проектов той же компании (same company_cnpj).
    # Services-проекты форсят Anexo III (см. resolve_tax_settings).
    effective_meta = resolve_tax_settings(proj_meta_for_tax, _load_projects() or {})
    das_info = _compute_das(effective_meta, revenue_gross if revenue_gross > 0 else 0.0, rbt12)
    das_val = das_info["das_brl"]

    # Publicidade — из отчётов publicidade (auto + manual) с фильтром по периоду
    pub_data = get_publicidade_by_period(project, period_start, period_end)
    publicidade_val = float(pub_data.get("total", 0) or 0)

    # Armazenagem — из дневных отчётов с фильтром по периоду
    arm_data = get_armazenagem_by_period(project, period_start, period_end)
    armazenagem_val = float(arm_data.get("total", 0) or 0)

    # Aluguel — приоритет источников (все нормализуем в BRL/месяц):
    #   1) project.aluguel_mensal (форма, BRL/мес) — прямое значение.
    #   2) project.rental.rate_usd + period(quarter|month) — USD, конвертим
    #      в BRL через USD_BRL_RATE = 5.46 (тот же курс что в compute_balance).
    #   3) Fallback на baseline_overrides.aluguel (ARTUR legacy 206 дней).
    # Начисление: от max(launch_date, period_start) до period_end
    # пропорционально дням (30.4375/мес = 365.25/12).
    proj_meta_for_rent = get_project_meta(project) or {}
    mensal_brl = float(proj_meta_for_rent.get("aluguel_mensal", 0) or 0)
    if mensal_brl <= 0:
        rental = proj_meta_for_rent.get("rental") or {}
        if isinstance(rental, dict):
            rate_usd = float(rental.get("rate_usd", 0) or 0)
            if rate_usd > 0:
                period_kind = (rental.get("period") or "month").lower()
                mensal_usd = rate_usd / 3.0 if period_kind.startswith("quart") else rate_usd
                mensal_brl = mensal_usd * 5.46  # USD→BRL (согласован с compute_balance)

    aluguel_val = 0.0
    if mensal_brl > 0:
        accrual_start = period_start
        launch_str = (proj_meta_for_rent.get("launch_date") or "").strip()[:10]
        if launch_str:
            try:
                from datetime import datetime as _dt
                ld = _dt.strptime(launch_str, "%Y-%m-%d").date()
                if ld > period_start:
                    accrual_start = ld
            except Exception:
                pass
        if accrual_start <= period_end:
            days = (period_end - accrual_start).days + 1
            aluguel_val = round(mensal_brl * days / 30.4375, 2)
    else:
        aluguel_full = float(approved.get("aluguel", 0) or 0)
        if aluguel_full > 0:
            baseline_days = 206  # ARTUR baseline period fallback
            period_days = (period_end - period_start).days + 1
            aluguel_val = round(aluguel_full * period_days / baseline_days, 2)

    # NB: envios_dif и returned_loss НЕ добавляем в opex — они уже в taxas_ml
    # (то есть уже вычтены при revenue_net = bruto - taxas).
    # DAS-строка теперь использует регим-aware label («DAS» / «DAS Simples Anexo
    # I» / «DAS Lucro Presumido»). UI рендерит бейдж через tax_info.
    das_label = "DAS (Simples 4,5% × bruto)"
    _regime = das_info.get("regime")
    if _regime == "simples_nacional":
        das_label = f"DAS Simples Nacional (Anexo {das_info.get('anexo', 'I')})"
    elif _regime == "lucro_presumido":
        das_label = "DAS Lucro Presumido + ICMS"

    opex_items = [
        ("Publicidade (Mercado Ads)", publicidade_val),
        (das_label, das_val),
        ("Armazenagem Full", armazenagem_val),
        ("Aluguel empresa (proрационально)", aluguel_val),
    ]
    operating_expenses = [PnLLine(label=lbl, amount_brl=val) for lbl, val in opex_items]
    opex_total = sum(line.amount_brl for line in operating_expenses)
    operating_profit = revenue_net - opex_total

    proj_meta = get_project_meta(project)
    cost_per_unit = proj_meta.get("avg_cost_per_unit_brl")
    cogs: float | None = None
    net_profit: float | None = None
    # COGS требует данных по проданным единицам и себесу — оставляем None,
    # пока эти поля не заполнены. Никаких эвристик "56,2% от bruto" в P&L.

    margin = (operating_profit / revenue_net * 100) if revenue_net else 0.0

    return PnLReport(
        project=project,
        period=period,
        revenue_gross=revenue_gross,
        taxas_ml=taxas_ml,
        revenue_net=revenue_net,
        operating_expenses=operating_expenses,
        operating_profit=operating_profit,
        cogs=cogs,
        net_profit=net_profit,
        vendas_count=vendas_count,
        margin_pct=margin,
        tax_info=das_info,
    )


def compute_cashflow(project: str, period: tuple[date, date]) -> CashFlowReport:
    """УПРОЩЁННАЯ модель ДДС (по продажам, не по зачислениям).

    Старт с нуля. За выбранный период:
        + Operating profit из ОПиУ (то что заработано на продажах после всех расходов)
        + USDT инвестиции собственника (Artur)
        - Bank outflows категории "supplier" (закупки товара)
        = Cash position

    Это НЕ строгая бухгалтерия — операционная прибыль это accrual,
    а supplier — реальный cash. Смешение допущено для управленческого
    отчёта (видеть «сколько у меня осталось после всех операций»).
    """
    proj_meta = get_project_meta(project)
    period_start, period_end = period

    # 1. Operating profit за период (из ОПиУ — vendas_ml.xlsx + расходы)
    pnl = compute_pnl(project, period)
    op_profit = float(pnl.operating_profit or 0)

    # 2. USDT инвестиции собственника — факт, учитываем всё до period_end
    usdt_inv = proj_meta.get("usdt_investments", []) or []
    usdt_total_brl = 0.0
    from datetime import datetime as _dt
    import calendar as _cal
    for inv in usdt_inv:
        ds = str(inv.get("date", "") or "")
        try:
            inv_date = _dt.strptime(ds, "%Y-%m").date()
            last = _cal.monthrange(inv_date.year, inv_date.month)[1]
            inv_date = inv_date.replace(day=last)
        except (ValueError, TypeError):
            try:
                inv_date = _dt.strptime(ds, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
        if inv_date <= period_end:
            usdt_total_brl += float(inv.get("brl", 0) or 0)

    # 2b. Поступления от партнёра — факт, учитываем всё до period_end
    partner_total = 0.0
    partner_txs: list = []
    for item in (proj_meta.get("partner_contributions") or []):
        try:
            d_t = _dt.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_t > period_end:
            continue
        try:
            v = float(item.get("valor", 0) or 0)
        except (ValueError, TypeError):
            v = 0
        partner_total += v
        partner_txs.append({
            "Data": d_t.strftime("%d/%m/%Y"),
            "Valor": v,
            "Descrição": item.get("note", ""),
            "Категория": "partner",
            "Класс.": item.get("from", ""),
        })

    # 4. Manual expenses — факт, учитываем всё до period_end
    other_expenses_total = 0.0
    other_expenses_txs: list = []
    for item in (proj_meta.get("manual_expenses") or []):
        try:
            d_t = _dt.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_t > period_end:
            continue
        try:
            v = abs(float(item.get("valor", 0) or 0))
        except (ValueError, TypeError):
            v = 0
        other_expenses_total += v
        other_expenses_txs.append({
            "Data": d_t.strftime("%d/%m/%Y"),
            "Valor": -v,
            "Descrição": item.get("note", ""),
            "Категория": item.get("category", "expense"),
            "Класс.": "manual",
        })

    # 3. Supplier outflows — факт, учитываем всё до period_end
    from .reports import aggregate_classified_by_project
    import pandas as _pd
    live = aggregate_classified_by_project(project)
    supplier_total = 0.0
    supplier_txs: list = []
    for tx in live.get("transactions", []):
        cat = str(tx.get("Категория", "") or "").lower()
        if cat != "supplier":
            continue
        ds = str(tx.get("Data", ""))
        try:
            td = _pd.to_datetime(ds, dayfirst=True).date()
        except Exception:
            continue
        if td > period_end:
            continue
        try:
            val = abs(float(tx.get("Valor", 0) or 0))
        except (ValueError, TypeError):
            val = 0
        supplier_total += val
        supplier_txs.append(tx)

    # Manual supplier entries — факт, всё до period_end
    from datetime import datetime as _dt
    for item in (proj_meta.get("manual_supplier") or []):
        try:
            tx_date = _dt.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if tx_date > period_end:
            continue
        try:
            val = abs(float(item.get("valor", 0) or 0))
        except (ValueError, TypeError):
            val = 0
        supplier_total += val
        supplier_txs.append({
            "Data": tx_date.strftime("%d/%m/%Y"),
            "Valor": -val,
            "Descrição": item.get("note", ""),
            "Категория": "supplier",
            "Класс.": f"manual ({item.get('source','')})",
        })

    # ── Cash position ──
    opening = 0.0
    closing = (opening + op_profit + usdt_total_brl + partner_total
               - supplier_total - other_expenses_total)

    return CashFlowReport(
        project=project,
        period=period,
        opening_balance=opening,
        inflows_operating=op_profit,
        inflows_count=int(pnl.vendas_count or 0),
        inflows_financing=usdt_total_brl,
        inflows_partner=partner_total,
        outflows_operating=supplier_total,
        outflows_other=other_expenses_total,
        closing_balance=closing,
        new_transactions=supplier_txs,
        partner_txs=partner_txs,
        other_expenses_txs=other_expenses_txs,
    )


def compute_balance(project: str, as_of: date, basis: str = "accrual") -> BalanceReport:
    """Flow-based balance в формате утверждённого ARTUR CSV:
    Входы (USDT + продажи NET) − Выходы (товар, реклама, devoluções, ...)
    = Сальдо. Минус просроченная аренда = САЛЬДО проекта.
    """
    proj_meta = get_project_meta(project)

    # ── ВХОДЫ ──
    usdt_inv = proj_meta.get("usdt_investments", []) or []
    inflow_usdt_brl = sum(float(inv.get("brl", 0) or 0) for inv in usdt_inv)
    inflow_usdt_usd = sum(float(inv.get("usd", 0) or 0) for inv in usdt_inv)

    # Продажи NET за весь период до as_of (через compute_pnl)
    period = (date(2025, 1, 1), as_of)
    pnl = compute_pnl(project, period, basis=basis)
    inflow_sales_net = pnl.revenue_net
    inflow_sales_count = pnl.vendas_count

    inflows_total = inflow_usdt_brl + inflow_sales_net

    # ── ВЫХОДЫ ──
    approved = get_approved_data(project) or {}
    out_mercadoria = float(approved.get("mercadoria", 0) or 0)
    out_publicidade = float(approved.get("publicidade", 0) or 0)
    out_devolucoes = float(approved.get("devolucoes", 0) or 0)
    out_full_express = float(approved.get("full_express", 0) or 0)
    out_das = float(approved.get("das", 0) or 0)
    if out_das == 0 and pnl.revenue_gross > 0:
        # Используем DAS уже посчитанный в compute_pnl с учётом tax_regime.
        # pnl.tax_info.das_brl всегда есть (включая legacy 4.5% fallback).
        tinfo = pnl.tax_info if isinstance(pnl.tax_info, dict) else None
        out_das = float((tinfo or {}).get("das_brl") or round(pnl.revenue_gross * 0.045, 2))
    out_armazenagem = float(approved.get("armazenagem", 0) or 0)
    out_aluguel = float(approved.get("aluguel", 0) or 0)

    outflows_total = (out_mercadoria + out_publicidade + out_devolucoes
                      + out_full_express + out_das + out_armazenagem + out_aluguel)

    saldo = inflows_total - outflows_total

    # ── Просроченная аренда ──
    rental = proj_meta.get("rental") or {}
    pending_usd = sum(float(p.get("usd", 0) or 0)
                      for p in (rental.get("payments") or [])
                      if p.get("status") == "pending")
    pending_brl = pending_usd * 5.46  # курс из утверждённого baseline ($1350 → R$ 7369.38)
    saldo_final = saldo - pending_brl

    # Сток: каталог SKU (_data/sku_catalog.json) + опционально avg_cost_per_unit_brl
    stock_data = (load_stock_full().get(project, {}) or {})
    stock_units_ml = int(stock_data.get("total_units", 0) or 0)
    stock_units_external = int(proj_meta.get("stock_units_external", 0) or 0)
    stock_units = stock_units_ml + stock_units_external
    legacy_avg = proj_meta.get("avg_cost_per_unit_brl")
    assess = assess_stock_for_project(
        project,
        stock_data.get("by_sku"),
        stock_units_external,
        legacy_avg,
    )
    stock_value_brl = float(assess.get("stock_value_brl") or 0)
    stock_missing_skus = list(assess.get("missing_skus") or [])
    stock_missing_units = int(assess.get("missing_units") or 0)
    stock_by_supplier_type = dict(assess.get("by_supplier_type") or {})

    if stock_units > 0 and stock_value_brl > 0:
        cost_per_unit = stock_value_brl / stock_units
    else:
        try:
            cost_per_unit = float(legacy_avg) if legacy_avg is not None else None
        except (TypeError, ValueError):
            cost_per_unit = None

    return BalanceReport(
        project=project,
        as_of=as_of,
        inflow_usdt_brl=inflow_usdt_brl,
        inflow_usdt_usd=inflow_usdt_usd,
        inflow_sales_net=inflow_sales_net,
        inflow_sales_count=inflow_sales_count,
        inflows_total=inflows_total,
        outflow_mercadoria=out_mercadoria,
        outflow_publicidade=out_publicidade,
        outflow_devolucoes=out_devolucoes,
        outflow_full_express=out_full_express,
        outflow_das=out_das,
        outflow_armazenagem=out_armazenagem,
        outflow_aluguel=out_aluguel,
        outflows_total=outflows_total,
        saldo=saldo,
        pending_rental_usd=pending_usd,
        pending_rental_brl=pending_brl,
        saldo_final=saldo_final,
        cost_per_unit=cost_per_unit,
        stock_units=stock_units,
        stock_value_brl=stock_value_brl,
        stock_missing_skus=stock_missing_skus,
        stock_missing_units=stock_missing_units,
        stock_by_supplier_type=stock_by_supplier_type,
    )
