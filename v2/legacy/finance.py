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
# HELPERS
# ─────────────────────────────────────────────

def _entry_valor_brl(item: dict) -> float:
    """BRL-normalized amount for a manual cashflow entry.

    Entries may be recorded in non-BRL currencies (USDT/USD) with a snapshot
    exchange rate for the date of the transaction. This helper returns the BRL
    equivalent used in ДДС aggregations.

    - currency missing or "BRL" → return valor as-is
    - currency != BRL with positive rate_brl → valor × rate_brl
    - currency != BRL but rate_brl missing/zero → fallback: return valor
      (treats amount as already-BRL; a missing rate is a data issue, not a
      show-stopper — caller can surface it in the UI)
    """
    try:
        v = float(item.get("valor", 0) or 0)
    except (ValueError, TypeError):
        return 0.0
    cur = str(item.get("currency", "BRL") or "BRL").upper()
    if cur == "BRL":
        return v
    try:
        rate = float(item.get("rate_brl", 0) or 0)
    except (ValueError, TypeError):
        rate = 0.0
    return v * rate if rate > 0 else v


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
    cogs: float = 0.0               # Σ(unit_cost_brl × qty) по проданным SKU
    net_profit: float = 0.0         # operating_profit - cogs
    vendas_count: int = 0
    margin_pct: float = 0.0
    # COGS diagnostics — SKUs без unit_cost_brl в каталоге (призыв: загрузить
    # Dados Fiscais или заполнить на /finance/sku-mapping).
    cogs_missing_skus: list[str] = field(default_factory=list)
    cogs_missing_units: int = 0
    # Per-SKU детализация: [{sku, mlb, units}] — для объединённого UI-списка
    # missing-cost (вместе с stock_missing_sku_details из Balance).
    cogs_missing_sku_details: list[dict] = field(default_factory=list)
    unit_cost_per_sku: dict[str, float] = field(default_factory=dict)
    # Расчётные данные по DAS (Simples Nacional / Lucro Presumido / override)
    # — faixa, aliquot effective, ICMS, method. Используется UI для бейджа рядом
    # со строкой DAS в operating_expenses.
    tax_info: dict | None = None


@dataclass
class CashFlowReport:
    project: str
    period: tuple[date, date]
    opening_balance: float          # стартовое сальдо (обычно 0)
    # Источник opening_balance: "configured" (project.opening_balance задан),
    # "partner_investments" (fallback из ручных partner-взносов в ДДС),
    # "none" (нет ни того, ни другого). UI подбирает подпись по этому полю.
    opening_balance_source: str = "none"
    inflows_operating: float = 0.0  # operating profit за период
    inflows_count: int = 0          # кол-во продаж за период
    inflows_financing: float = 0.0  # USDT инвестиции собственника
    inflows_partner: float = 0.0    # ручные поступления от партнёра
    outflows_operating: float = 0.0 # supplier (закупки)
    outflows_other: float = 0.0     # ручные прочие расходы
    closing_balance: float = 0.0
    new_transactions: list = field(default_factory=list)  # supplier tx
    partner_txs: list = field(default_factory=list)
    other_expenses_txs: list = field(default_factory=list)


@dataclass
class BalanceReport:
    """Accounting balance: Assets = Liabilities + Equity. `balance_delta`
    surfaces any gap so UI can prompt the user to fill missing rows
    (initial_equity, loans, unrecorded cash, overdue payables…).

    Legacy `inflow_*`/`outflow_*` flow fields are kept for backwards-
    compatibility with existing /pnl-matrix and UI code; they duplicate
    data exposed through the structured `cash_brl` / `inventory_brl` /
    `assets_total` fields and will be retired in a follow-up sweep.
    """
    project: str
    as_of: date

    # ── Assets ───────────────────────────────────────────────────────────
    cash_brl: float = 0.0
    accounts_receivable_brl: float = 0.0   # 0 in A1 model (vendas ≡ cash)
    inventory_brl: float = 0.0             # stock_value_brl
    assets_total: float = 0.0

    # ── Liabilities ──────────────────────────────────────────────────────
    accounts_payable_brl: float = 0.0      # Σ unpaid overdue planned_payments
    loans_balance_brl: float = 0.0         # Σ f2_loans.outstanding_brl
    liabilities_total: float = 0.0

    # ── Equity ───────────────────────────────────────────────────────────
    initial_equity_brl: float = 0.0
    accumulated_profit_brl: float = 0.0    # Σ net_profit (launch → as_of)
    dividends_paid_brl: float = 0.0
    equity_total: float = 0.0

    # ── Reconciliation ───────────────────────────────────────────────────
    balance_delta_brl: float = 0.0         # assets − (liab + equity)

    # ── Investment return (time-weighted MOIC + annualized) ─────────────
    # «Вложил 10K, сейчас 40K → x4» — главная метрика владельца.
    # weighted_avg_invested = Σ(amount × days_worked) / total_days. Это
    # реальный "средний работавший капитал" — если 3K лежали год а 40K
    # всего месяц, весит первое.
    # NAV = assets − liabilities (чистая стоимость капитала сейчас).
    # Total return NAV = NAV + Σ dividends (учитываем уже-выведенные деньги).
    total_invested_brl: float = 0.0        # номинальная сумма (Σ всех взносов)
    weighted_avg_invested_brl: float = 0.0 # time-weighted (честная база для MOIC)
    current_nav_brl: float = 0.0           # assets − liabilities
    total_return_nav_brl: float = 0.0      # NAV + dividends_paid
    moic_simple: float = 0.0               # nav / total_invested (наивный)
    moic_current: float = 0.0              # nav / weighted_avg_invested (time-weighted)
    moic_total_return: float = 0.0         # (nav + divs) / weighted_avg_invested
    annualized_pct: float = 0.0            # CAGR по time-weighted MOIC
    years_since_launch: float = 0.0        # для UI "xN за Y лет"
    launch_date_iso: str | None = None     # для UI подписи
    # Timeline взносов для UI-tooltip: [{date, brl, kind}]
    capital_contributions: list = field(default_factory=list)

    # ── Legacy flow-based fields (back-compat) ───────────────────────────
    inflow_usdt_brl: float = 0.0
    inflow_usdt_usd: float = 0.0
    inflow_sales_net: float = 0.0
    inflow_sales_count: int = 0
    inflows_total: float = 0.0
    outflow_mercadoria: float = 0.0
    outflow_publicidade: float = 0.0
    outflow_devolucoes: float = 0.0
    outflow_full_express: float = 0.0
    outflow_das: float = 0.0
    outflow_armazenagem: float = 0.0
    outflow_aluguel: float = 0.0
    outflows_total: float = 0.0
    saldo: float = 0.0
    pending_rental_usd: float = 0.0
    pending_rental_brl: float = 0.0
    saldo_final: float = 0.0
    cost_per_unit: float | None = None
    stock_units: int = 0
    stock_value_brl: float = 0.0
    stock_missing_skus: list = field(default_factory=list)
    stock_missing_units: int = 0
    # [{sku, mlb, units}] — для UI (ML-ссылка + CTA на /finance/sku-mapping)
    stock_missing_sku_details: list = field(default_factory=list)
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

def compute_pnl(
    project: str,
    period: tuple[date, date],
    basis: str = "accrual",
    has_1yr_bank_data: bool = False,
) -> PnLReport:
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
    # SKU → qty (только delivered; returned вернулся на склад, не COGS).
    # Плюс SKU → MLB mapping: нужен для MLB-fallback lookup в catalog, когда
    # внутренние коды SKU (sumka5-1, w02-1) не совпадают с Dados-Fiscais SKU
    # (HB50173-9), но оба ссылаются на одно ML-объявление.
    sku_qty: dict[str, int] = {}
    sku_mlbs: dict[str, str] = {}
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
                # COGS feed: накопим SKU×Unidades по доставленным
                _sku = str(row.get("SKU") or "").strip()
                if _sku:
                    try:
                        _u = int(float(str(row.get("Unidades") or 0).strip()))
                    except (ValueError, TypeError):
                        _u = 0
                    if _u > 0:
                        sku_qty[_sku] = sku_qty.get(_sku, 0) + _u
                        # MLB mapping для fallback
                        if _sku not in sku_mlbs:
                            _mlb = str(row.get("# de anúncio")
                                       or row.get("# de anuncio")
                                       or "").strip()
                            if _mlb:
                                sku_mlbs[_sku] = _mlb
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
        # RBT12 по CNPJ: суммируем bruto всех проектов одной компании
        from .reports import get_company_monthly_bruto, rolling_rbt12
        from .config import load_projects as _lp
        bruto_by_month = get_company_monthly_bruto(project, _lp() or {}) or {}
        cur_mk = f"{period_end.year:04d}-{period_end.month:02d}"
        rbt12 = rolling_rbt12(bruto_by_month, cur_mk)
    except Exception:
        pass

    from .tax_brazil import compute_das as _compute_das, resolve_tax_settings
    from .config import load_projects as _load_projects
    # Наследуем режим от других проектов той же компании (same company_cnpj).
    # Services-проекты форсят Anexo III (см. resolve_tax_settings).
    effective_meta = resolve_tax_settings(proj_meta_for_tax, _load_projects() or {})
    # `has_1yr_bank_data` прокидывается из /reports endpoint (проверяет
    # `uploads` на наличие extrato_* с MIN(created_at) ≤ now-12mo). Если
    # ни `ml_only_revenue`, ни 12-мес выписка не заданы — compute_das
    # откатится на faixa 1 nominal как безопасный минимум.
    das_info = _compute_das(
        effective_meta,
        revenue_gross if revenue_gross > 0 else 0.0,
        rbt12,
        has_1yr_bank_data=bool(has_1yr_bank_data),
    )
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

    # Fulfillment — manual_expenses[fulfillment] + bank_tx[fulfillment] за период.
    # Note: "R$ X/продажа" если есть продажи (пользователь просил раскладку на шт).
    try:
        from .reports import get_fulfillment_by_period as _fulf_by_period
        fulfillment_val = round(float(_fulf_by_period(project, period_start, period_end)), 2)
    except Exception:
        fulfillment_val = 0.0
    fulfillment_note = ""
    if fulfillment_val > 0 and vendas_count > 0:
        fulfillment_note = f"R$ {fulfillment_val/vendas_count:.2f}/продажа"

    opex_items: list[tuple[str, float, str]] = [
        ("Publicidade (Mercado Ads)", publicidade_val, ""),
        (das_label, das_val, ""),
        ("Armazenagem Full", armazenagem_val, ""),
        ("Fulfillment (доставка до клиента)", fulfillment_val, fulfillment_note),
        ("Aluguel empresa (proрационально)", aluguel_val, ""),
    ]
    operating_expenses = [
        PnLLine(label=lbl, amount_brl=val, note=note)
        for lbl, val, note in opex_items
    ]
    opex_total = sum(line.amount_brl for line in operating_expenses)
    operating_profit = revenue_net - opex_total

    # ── COGS из Dados Fiscais (sku_catalog.unit_cost_brl) ────────────────
    # Porядок lookup cost для каждого SKU (тот же что в assess_stock_for_project
    # на балансе, чтобы ОПиУ и Balance показывали согласованный missing-список):
    #   1) catalog[normalize_sku(sku)] — прямое совпадение кода
    #   2) catalog_mlb_index[normalize_mlb(sku_mlbs[sku])] — fallback через MLB
    #      (закрывает кейс "sumka5-1 в vendas / HB50173 в Dados Fiscais").
    #   3) avg_cost_per_unit_brl на проекте (legacy fallback для ARTUR).
    #   4) SKU попадает в cogs_missing_skus.
    from .sku_catalog import (
        build_catalog_index, build_catalog_mlb_index, normalize_sku, _normalize_mlb,
    )
    proj_meta = get_project_meta(project)
    legacy_avg_cost = proj_meta.get("avg_cost_per_unit_brl")
    try:
        legacy_avg_cost = float(legacy_avg_cost) if legacy_avg_cost is not None else None
    except (ValueError, TypeError):
        legacy_avg_cost = None

    catalog_index = build_catalog_index()
    catalog_mlb_index = build_catalog_mlb_index()

    cogs_total = 0.0
    cogs_missing_skus: list[str] = []
    cogs_missing_units = 0
    cogs_missing_details: list[dict] = []
    unit_cost_per_sku: dict[str, float] = {}
    for sku, qty in sku_qty.items():
        # Step 1: прямой lookup по SKU
        key = normalize_sku(sku)
        row = catalog_index.get(key) or {}
        raw_cost = row.get("unit_cost_brl")
        try:
            unit_cost = float(raw_cost) if raw_cost is not None else None
        except (ValueError, TypeError):
            unit_cost = None
        if not unit_cost or unit_cost <= 0:
            unit_cost = None
            # Step 2: MLB fallback
            mlb_norm = _normalize_mlb(sku_mlbs.get(sku, ""))
            if mlb_norm:
                mrow = catalog_mlb_index.get(mlb_norm) or {}
                raw_mlb_cost = mrow.get("unit_cost_brl")
                try:
                    mlb_cost = float(raw_mlb_cost) if raw_mlb_cost is not None else None
                except (ValueError, TypeError):
                    mlb_cost = None
                if mlb_cost and mlb_cost > 0:
                    unit_cost = mlb_cost
        if not unit_cost or unit_cost <= 0:
            # Step 3: legacy avg_cost_per_unit_brl
            unit_cost = legacy_avg_cost if (legacy_avg_cost and legacy_avg_cost > 0) else None
        if unit_cost and unit_cost > 0:
            cogs_total += unit_cost * qty
            unit_cost_per_sku[sku] = round(float(unit_cost), 4)
        else:
            cogs_missing_skus.append(sku)
            cogs_missing_units += int(qty)
            cogs_missing_details.append({
                "sku": str(sku).strip(),
                "mlb": sku_mlbs.get(sku, "") or "",
                "units": int(qty),
            })
    cogs_total = round(cogs_total, 2)
    net_profit = round(operating_profit - cogs_total, 2)

    margin = (net_profit / revenue_net * 100) if revenue_net else 0.0

    return PnLReport(
        project=project,
        period=period,
        revenue_gross=revenue_gross,
        taxas_ml=taxas_ml,
        revenue_net=revenue_net,
        operating_expenses=operating_expenses,
        operating_profit=operating_profit,
        cogs=cogs_total,
        net_profit=net_profit,
        vendas_count=vendas_count,
        margin_pct=margin,
        cogs_missing_skus=cogs_missing_skus,
        cogs_missing_units=cogs_missing_units,
        cogs_missing_sku_details=sorted(cogs_missing_details, key=lambda d: -d["units"]),
        unit_cost_per_sku=unit_cost_per_sku,
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

    # 2b. Поступления от партнёра — факт, учитываем всё до period_end.
    # Валюта: partner может внести USDT/USD с указанным курсом → _entry_valor_brl
    # нормализует в BRL.
    partner_total = 0.0
    partner_txs: list = []
    for item in (proj_meta.get("partner_contributions") or []):
        try:
            d_t = _dt.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_t > period_end:
            continue
        v_brl = _entry_valor_brl(item)
        partner_total += v_brl
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        partner_txs.append({
            "Data": d_t.strftime("%d/%m/%Y"),
            "Valor": v_brl,
            "Descrição": item.get("note", ""),
            "Категория": "partner",
            "Класс.": item.get("from", ""),
            "Валюта": cur,
            "Сумма_ориг": float(item.get("valor", 0) or 0),
            "Курс": float(item.get("rate_brl", 0) or 0) if cur != "BRL" else None,
        })

    # 2c. Полученные займы от других проектов — inflow, учитываем до period_end
    for item in (proj_meta.get("loans_received") or []):
        try:
            d_t = _dt.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_t > period_end:
            continue
        v_brl = _entry_valor_brl(item)
        partner_total += v_brl  # в inflows_partner (финансирование от «партнёра по займу»)
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        partner_txs.append({
            "Data": d_t.strftime("%d/%m/%Y"),
            "Valor": v_brl,
            "Descrição": item.get("note", "") or "Займ получен",
            "Категория": "loan_received",
            "Класс.": item.get("counterparty_project", ""),
            "Валюта": cur,
            "Сумма_ориг": float(item.get("valor", 0) or 0),
            "Курс": float(item.get("rate_brl", 0) or 0) if cur != "BRL" else None,
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
        v = abs(_entry_valor_brl(item))
        other_expenses_total += v
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        other_expenses_txs.append({
            "Data": d_t.strftime("%d/%m/%Y"),
            "Valor": -v,
            "Descrição": item.get("note", ""),
            "Категория": item.get("category", "expense"),
            "Класс.": "manual",
            "Валюта": cur,
            "Сумма_ориг": float(item.get("valor", 0) or 0),
            "Курс": float(item.get("rate_brl", 0) or 0) if cur != "BRL" else None,
        })

    # 4b. Выданные займы другим проектам — outflow, учитываем до period_end
    for item in (proj_meta.get("loans_given") or []):
        try:
            d_t = _dt.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_t > period_end:
            continue
        v = abs(_entry_valor_brl(item))
        other_expenses_total += v
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        other_expenses_txs.append({
            "Data": d_t.strftime("%d/%m/%Y"),
            "Valor": -v,
            "Descrição": item.get("note", "") or "Займ выдан",
            "Категория": "loan_given",
            "Класс.": item.get("counterparty_project", ""),
            "Валюта": cur,
            "Сумма_ориг": float(item.get("valor", 0) or 0),
            "Курс": float(item.get("rate_brl", 0) or 0) if cur != "BRL" else None,
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
        val = abs(_entry_valor_brl(item))
        supplier_total += val
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        supplier_txs.append({
            "Data": tx_date.strftime("%d/%m/%Y"),
            "Valor": -val,
            "Descrição": item.get("note", ""),
            "Категория": "supplier",
            "Класс.": f"manual ({item.get('source','')})",
            "Валюта": cur,
            "Сумма_ориг": float(item.get("valor", 0) or 0),
            "Курс": float(item.get("rate_brl", 0) or 0) if cur != "BRL" else None,
        })

    # ── Аренда: ДДС = ФАКТ платежей (cash-basis), ОПиУ = прорейт (accrual) ──
    # op_profit из compute_pnl включает прорейт аренды → вычитаем его обратно,
    # чтобы строка «Операционный приток» не дублировалась с реальными платежами.
    aluguel_accrual_brl = 0.0
    fulfillment_accrual_brl = 0.0
    for line in (pnl.operating_expenses or []):
        lbl = (line.label or "").lower()
        if lbl.startswith("aluguel") or "alug" in lbl:
            aluguel_accrual_brl += float(line.amount_brl or 0)
        elif "fulfillment" in lbl or "фулфилмент" in lbl:
            # manual_expenses[fulfillment] уже в other_expenses_total (cash),
            # а compute_pnl вычитает их через operating_expenses. Чтобы не
            # задваивать — возвращаем их обратно в op_profit_cash.
            fulfillment_accrual_brl += float(line.amount_brl or 0)
    op_profit_cash = op_profit + aluguel_accrual_brl + fulfillment_accrual_brl

    # Фактически оплаченные rental.payments в периоде (status=paid)
    rental = proj_meta.get("rental") or {}
    rental_payments = rental.get("payments") or []
    outflows_rental_actual = 0.0
    rental_txs: list = []
    for p_item in rental_payments:
        if str(p_item.get("status", "")).lower() != "paid":
            continue
        try:
            pd_t = _dt.strptime(str(p_item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if pd_t < period_start or pd_t > period_end:
            continue
        amt_usd = float(p_item.get("amount_usd", 0) or 0)
        rate = float(p_item.get("rate_brl", 0) or 0)
        if rate <= 0:
            rate = 5.46  # fallback — консервативный курс, чтобы пропустить запись
        amt_brl = amt_usd * rate
        outflows_rental_actual += amt_brl
        other_expenses_txs.append({
            "Data": pd_t.strftime("%d/%m/%Y"),
            "Valor": -amt_brl,
            "Descrição": p_item.get("note", "") or "Аренда (факт)",
            "Категория": "rental",
            "Класс.": f"USD {amt_usd:.2f} @ {rate:.4f}",
            "Валюта": "USD",
            "Сумма_ориг": amt_usd,
            "Курс": rate,
        })
    # Показываем в outflows_other, чтобы попало в общую разбивку ДДС
    other_expenses_total += outflows_rental_actual

    # ── Cash position ──
    # Opening balance: берём из project.opening_balance (BRL) если задан.
    # Смысл: касса на дату balance_date (например, сальдо на 01.09.2025 при
    # переносе из старой учётной системы). Если balance_date позже period_end
    # — не применяем (ещё не наступила точка отсчёта).
    # Fallback: если opening_balance не задан — используем сумму партнёрских
    # инвестиций (введённых вручную в ДДС) как стартовый капитал, чтобы
    # пользователь не видел "откр. остаток = 0" пока не настроит баланс.
    # Чтобы не удваивать: когда partner идёт в opening, inflows_partner=0.
    opening = 0.0
    opening_from_fallback = False
    ob_set = False
    try:
        ob = proj_meta.get("opening_balance")
        if ob is not None and str(ob).strip() != "":
            ob_f = float(ob)
            bd_str = (proj_meta.get("balance_date") or "").strip()[:10]
            bd = None
            if bd_str:
                try:
                    bd = _dt.strptime(bd_str, "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    bd = None
            if bd is None or bd <= period_end:
                opening = ob_f
                ob_set = True
    except (TypeError, ValueError):
        pass

    if not ob_set and partner_total > 0:
        opening = partner_total
        opening_from_fallback = True

    # Если opening = partner_total (fallback), то не учитываем их ещё раз
    # как inflows_partner — иначе двойной счёт.
    partner_total_for_inflows = 0.0 if opening_from_fallback else partner_total

    # Источник opening_balance для UI-подписи.
    if ob_set:
        opening_source = "configured"
    elif opening_from_fallback:
        opening_source = "partner_investments"
    else:
        opening_source = "none"

    closing = (opening + op_profit_cash + usdt_total_brl + partner_total_for_inflows
               - supplier_total - other_expenses_total)

    return CashFlowReport(
        project=project,
        period=period,
        opening_balance=opening,
        opening_balance_source=opening_source,
        inflows_operating=op_profit_cash,
        inflows_count=int(pnl.vendas_count or 0),
        inflows_financing=usdt_total_brl,
        inflows_partner=partner_total_for_inflows,
        outflows_operating=supplier_total,
        outflows_other=other_expenses_total,
        closing_balance=closing,
        new_transactions=supplier_txs,
        partner_txs=partner_txs,
        other_expenses_txs=other_expenses_txs,
    )


def compute_balance(
    project: str,
    as_of: date,
    basis: str = "accrual",
    has_1yr_bank_data: bool = False,
) -> BalanceReport:
    """Flow-based balance: Входы (USDT + NET продажи) − Выходы = Сальдо.

    Outflows считаются из РЕАЛЬНЫХ источников данных пользователя:
    - mercadoria = stock_full × sku_catalog.unit_cost_brl (оценка склада)
    - publicidade = get_publicidade_by_period(cumulative)
    - armazenagem = get_armazenagem_by_period(cumulative)
    - devoluções = PnL "Cancelamentos e devoluções" (из vendas cancelled)
    - das = PnL.tax_info.das_brl (с учётом tax_regime)
    - aluguel = rental.payments со статусом paid × rate_brl
    - full_express = 0 (пока нет парсера; legacy override возможен)

    baseline_overrides в projects_db — могут переопределить любое значение
    (для legacy-проектов вроде ARTUR с утверждёнными цифрами).
    """
    proj_meta = get_project_meta(project)

    # ── ВХОДЫ ──
    usdt_inv = proj_meta.get("usdt_investments", []) or []
    inflow_usdt_brl = sum(float(inv.get("brl", 0) or 0) for inv in usdt_inv)
    inflow_usdt_usd = sum(float(inv.get("usd", 0) or 0) for inv in usdt_inv)

    # Продажи NET за весь период до as_of (через compute_pnl)
    period = (date(2025, 1, 1), as_of)
    pnl = compute_pnl(project, period, basis=basis, has_1yr_bank_data=has_1yr_bank_data)
    inflow_sales_net = pnl.revenue_net
    inflow_sales_count = pnl.vendas_count

    inflows_total = inflow_usdt_brl + inflow_sales_net

    # ── STOCK (раньше было ПОСЛЕ outflows — переносим сюда: mercadoria = stock_value) ──
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
        sku_mlbs=stock_data.get("sku_mlbs"),
    )
    stock_value_brl = float(assess.get("stock_value_brl") or 0)
    stock_missing_skus = list(assess.get("missing_skus") or [])
    stock_missing_units = int(assess.get("missing_units") or 0)
    stock_missing_sku_details = list(assess.get("missing_sku_details") or [])
    stock_by_supplier_type = dict(assess.get("by_supplier_type") or {})

    if stock_units > 0 and stock_value_brl > 0:
        cost_per_unit = stock_value_brl / stock_units
    else:
        try:
            cost_per_unit = float(legacy_avg) if legacy_avg is not None else None
        except (TypeError, ValueError):
            cost_per_unit = None

    # ── ВЫХОДЫ (из реальных источников) ──

    # 1. Publicidade — весь cumulative период из фатур/CSV
    try:
        from .reports import get_publicidade_by_period as _pub_by_period
        pub_data = _pub_by_period(project, period[0], as_of)
        out_publicidade = float(pub_data.get("total") or 0)
    except Exception:
        out_publicidade = 0.0

    # 2. Armazenagem — cumulative из отчётов хранения
    try:
        from .reports import get_armazenagem_by_period as _arm_by_period
        arm_data = _arm_by_period(project, period[0], as_of)
        out_armazenagem = float(arm_data.get("total") or 0)
    except Exception:
        out_armazenagem = 0.0

    # 3. Devoluções — из PnL cancellations line (накопленная сумма в period)
    out_devolucoes = 0.0
    for line in (pnl.operating_expenses or []):
        lbl = str(getattr(line, "label", "") or "").lower()
        if "cancel" in lbl or "devolu" in lbl:
            # PnL expenses идут с отрицательным знаком → берём abs
            out_devolucoes += abs(float(getattr(line, "amount_brl", 0) or 0))

    # 4. DAS — из tax_info (уже считается в compute_pnl)
    tinfo = pnl.tax_info if isinstance(pnl.tax_info, dict) else None
    out_das = float((tinfo or {}).get("das_brl") or 0)
    if out_das == 0 and pnl.revenue_gross > 0:
        out_das = round(pnl.revenue_gross * 0.045, 2)  # legacy fallback

    # 5. Aluguel — фактически оплаченные rental платежи
    rental = proj_meta.get("rental") or {}
    out_aluguel = 0.0
    for p in (rental.get("payments") or []):
        if str(p.get("status", "")).lower() != "paid":
            continue
        usd = float(p.get("amount_usd") or p.get("usd") or 0)
        rate = float(p.get("rate_brl") or 0)
        if rate <= 0:
            rate = 5.46
        out_aluguel += usd * rate

    # 6. Fulfillment — manual_expenses[category=fulfillment] + bank_tx[fulfillment]
    #    cumulative от 2025-01-01 до as_of (тот же диапазон что для PnL period
    #    в строчке выше). Балансовая строка outflow_full_express теперь показывает
    #    реальные fulfillment-расходы (а не 0).
    try:
        from .reports import get_fulfillment_by_period as _fulf_by_period_bal
        out_full_express = round(float(_fulf_by_period_bal(project, date(2025, 1, 1), as_of)), 2)
    except Exception:
        out_full_express = 0.0

    # 7. Mercadoria = фактические платежи поставщикам (из ДДС/banking + manual_supplier)
    # НЕ stock_value: чтобы не дублировать (stock_value = оставшийся актив,
    # supplier cash = то что юзер реально заплатил поставщикам). Stock value
    # при этом идёт в balance как отдельный актив (stock_value_brl).
    try:
        from .reports import aggregate_classified_by_project as _agg_cls
        import pandas as _pd_bal
        from datetime import datetime as _dt_bal
        live = _agg_cls(project)
        supplier_total = 0.0
        for tx in (live.get("transactions") or []):
            cat = str(tx.get("Категория", "") or "").lower()
            if cat != "supplier":
                continue
            ds = str(tx.get("Data", ""))
            try:
                td = _pd_bal.to_datetime(ds, dayfirst=True).date()
            except Exception:
                continue
            if td > as_of:
                continue
            try:
                val = abs(float(tx.get("Valor", 0) or 0))
            except (ValueError, TypeError):
                val = 0.0
            supplier_total += val
        # Manual supplier entries
        for item in (proj_meta.get("manual_supplier") or []):
            try:
                tx_date = _dt_bal.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if tx_date > as_of:
                continue
            supplier_total += abs(_entry_valor_brl(item))
        out_mercadoria = round(supplier_total, 2)
    except Exception:
        out_mercadoria = 0.0

    # ── baseline_overrides в projects_db.json перекрывают любое значение ──
    # (для legacy-проектов с утверждёнными цифрами, например ARTUR)
    overrides = proj_meta.get("baseline_overrides") or {}
    if isinstance(overrides, dict):
        if "mercadoria" in overrides and overrides["mercadoria"] not in (None, "", 0):
            out_mercadoria = float(overrides["mercadoria"])
        if "publicidade" in overrides and overrides["publicidade"] not in (None, "", 0):
            out_publicidade = float(overrides["publicidade"])
        if "devolucoes" in overrides and overrides["devolucoes"] not in (None, "", 0):
            out_devolucoes = float(overrides["devolucoes"])
        if "full_express" in overrides and overrides["full_express"] not in (None, "", 0):
            out_full_express = float(overrides["full_express"])
        if "das" in overrides and overrides["das"] not in (None, "", 0):
            out_das = float(overrides["das"])
        if "armazenagem" in overrides and overrides["armazenagem"] not in (None, "", 0):
            out_armazenagem = float(overrides["armazenagem"])
        if "aluguel" in overrides and overrides["aluguel"] not in (None, "", 0):
            out_aluguel = float(overrides["aluguel"])

    outflows_total = (out_mercadoria + out_publicidade + out_devolucoes
                      + out_full_express + out_das + out_armazenagem + out_aluguel)

    saldo = inflows_total - outflows_total

    # ── Просроченная аренда: AP только то что ДОЛЖНЫ сейчас ──
    # Правила (по датам, сортированным asc):
    #  1) Последний paid-платёж + step = paid_until (до какой даты покрыто).
    #  2) Pending платежи с датой <= paid_until — УЖЕ ПОКРЫТЫ (оплата охватывает
    #     их период), игнорируем.
    #  3) Pending с датой > today — будущие обязательства, НЕ AP (нет долга).
    #  4) Всё что посередине (paid_until < date <= today) — реальный overdue AP.
    from datetime import datetime as _dt_pending, timedelta as _td_pending
    _payments_raw = (rental.get("payments") or [])
    # Нормализуем и сортируем по date asc
    _payments: list[tuple[date, str, float]] = []  # (date, status, amount_usd)
    for p in _payments_raw:
        try:
            pd = _dt_pending.strptime(str(p.get("date", ""))[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        st = str(p.get("status", "pending")).lower()
        amt = float(p.get("amount_usd") or p.get("usd") or 0)
        _payments.append((pd, st, amt))
    _payments.sort(key=lambda x: x[0])

    # Последняя оплаченная дата + шаг периода = paid_until
    from .reports import _step_date_by_months as _step_dm
    _period = str(rental.get("period", "month")).lower()
    _step_months = 3 if _period.startswith("quart") else 1
    _last_paid = max((d for d, st, _ in _payments if st == "paid"), default=None)
    _paid_until: date | None = None
    if _last_paid is not None:
        nps = _step_dm(_last_paid, _step_months)
        _paid_until = nps - _td_pending(days=1)

    pending_usd = 0.0
    for pd, st, amt in _payments:
        if st != "pending":
            continue
        if _paid_until is not None and pd <= _paid_until:
            continue  # уже покрыт paid-периодом
        if pd > as_of:
            continue  # будущая — не AP
        pending_usd += amt
    pending_brl = pending_usd * 5.46  # курс fallback
    saldo_final = saldo - pending_brl

    # ─────────────────────────────────────────────────────────────────────
    # НОВАЯ бухгалтерская часть: Assets = Liabilities + Equity
    # ─────────────────────────────────────────────────────────────────────
    from .capital import loans_balance as _loans_balance, dividends_total as _dividends_total
    from .planning import unpaid_ap_total as _unpaid_ap_total

    # Assets
    # cash_brl = ровно closing_balance из ДДС (compute_cashflow) — чтобы
    # "ДДС.закр.остаток + товар − долги" совпадало с NAV на балансе.
    # Старая формула `saldo = inflows − outflows` давала расхождение на
    # величину partner_contributions, opening_balance и прорейта аренды.
    try:
        cf = compute_cashflow(project, (date(2025, 1, 1), as_of))
        cash_brl = round(float(cf.closing_balance or 0), 2)
    except Exception:
        cash_brl = round(saldo, 2)  # fallback на legacy
    inventory_brl = round(stock_value_brl, 2)
    accounts_receivable_brl = 0.0
    assets_total = round(cash_brl + accounts_receivable_brl + inventory_brl, 2)

    # Liabilities — AP из planned_payments (непогашенные, overdue), loans
    # из f2_loans (активные), pending rental — считаем тоже AP.
    try:
        accounts_payable_brl = round(_unpaid_ap_total(project, as_of) + pending_brl, 2)
    except Exception:
        accounts_payable_brl = round(pending_brl, 2)
    try:
        loans_balance_brl = round(_loans_balance(project, as_of), 2)
    except Exception:
        loans_balance_brl = 0.0
    liabilities_total = round(accounts_payable_brl + loans_balance_brl, 2)

    # Equity — initial + accumulated_profit − dividends. accumulated_profit
    # = net_profit за весь период (launch → as_of) — используем тот же pnl
    # инстанс что и выше (не зовём compute_pnl повторно).
    initial_equity_brl = 0.0
    try:
        initial_equity_brl = round(float(proj_meta.get("initial_equity_brl") or 0), 2)
    except (ValueError, TypeError):
        initial_equity_brl = 0.0
    accumulated_profit_brl = round(float(getattr(pnl, "net_profit", 0) or 0), 2)
    try:
        dividends_paid_brl = round(_dividends_total(project, as_of), 2)
    except Exception:
        dividends_paid_brl = 0.0
    equity_total = round(
        initial_equity_brl + accumulated_profit_brl - dividends_paid_brl, 2
    )

    balance_delta_brl = round(assets_total - (liabilities_total + equity_total), 2)

    # ── Investment return: time-weighted MOIC + annualized CAGR ──────────
    # Собираем все взносы капитала с датами в один timeline, нормализуем в BRL.
    # "Время работы" каждого взноса = (as_of - date_of_contribution). Это дает
    # честный weighted-avg invested, который не искажается поздними вливаниями.
    #
    # Источники:
    #   - initial_equity_brl — считаем что зашёл на launch_date (или, если
    #     launch_date нет, игнорируем: без даты не знаем сколько дней работал).
    #   - usdt_investments — из проекта, каждый со своей date.
    #   - partner_contributions — из проекта, каждый со своей date + валютной
    #     нормализацией через _entry_valor_brl.
    contributions: list[dict] = []

    launch_iso_str = None
    launch_date_parsed: date | None = None
    launch_raw = str(proj_meta.get("launch_date") or "").strip()[:10]
    if launch_raw:
        try:
            launch_date_parsed = datetime.strptime(launch_raw, "%Y-%m-%d").date()
            launch_iso_str = launch_date_parsed.isoformat()
        except (ValueError, TypeError):
            launch_date_parsed = None

    if initial_equity_brl > 0 and launch_date_parsed is not None:
        contributions.append({
            "date": launch_date_parsed,
            "brl": initial_equity_brl,
            "kind": "initial_equity",
        })

    import calendar as _cal_tw
    for inv in (proj_meta.get("usdt_investments") or []):
        ds = str(inv.get("date") or "")
        inv_date: date | None = None
        try:
            inv_date = datetime.strptime(ds, "%Y-%m").date()
            last = _cal_tw.monthrange(inv_date.year, inv_date.month)[1]
            inv_date = inv_date.replace(day=last)
        except (ValueError, TypeError):
            try:
                inv_date = datetime.strptime(ds, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                inv_date = None
        if inv_date is None or inv_date > as_of:
            continue
        # "Тестовая закупка" для usdt_investments — тот же флаг что и для
        # partner_contributions: остаётся в ДДС как inflow_usdt_brl, но не
        # считается инвестицией для MOIC/NAV.
        if bool(inv.get("test_only")):
            continue
        amt_brl = float(inv.get("brl", 0) or 0)
        if amt_brl > 0:
            contributions.append({"date": inv_date, "brl": amt_brl, "kind": "usdt"})

    for item in (proj_meta.get("partner_contributions") or []):
        try:
            d_t = datetime.strptime(str(item.get("date", "")), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_t > as_of:
            continue
        # "Тестовая закупка" (test_only=true) остаётся в ДДС как inflow,
        # но не считается инвестицией для MOIC. Пример: положил 3K на тест
        # → 40K на основное. В invested/NAV идёт только 40K.
        if bool(item.get("test_only")):
            continue
        amt_brl = _entry_valor_brl(item)
        if amt_brl > 0:
            contributions.append({"date": d_t, "brl": amt_brl, "kind": "partner"})

    contributions.sort(key=lambda c: c["date"])

    # Номинальная сумма (для справки, используется в simple MOIC)
    total_invested_brl = round(sum(c["brl"] for c in contributions), 2)

    # Time-weighted average invested = Σ(amount × days_worked) / total_period_days
    weighted_avg_invested_brl = 0.0
    years_since_launch = 0.0
    if contributions:
        earliest = contributions[0]["date"]
        # Период = от первого взноса (или launch_date, если он раньше) до as_of
        period_start = min(earliest, launch_date_parsed) if launch_date_parsed else earliest
        total_days = max(1, (as_of - period_start).days)
        weighted_sum = sum(
            c["brl"] * max(0, (as_of - c["date"]).days) for c in contributions
        )
        weighted_avg_invested_brl = round(weighted_sum / total_days, 2)
        years_since_launch = round(total_days / 365.25, 3)

    # NAV
    current_nav_brl = round(assets_total - liabilities_total, 2)
    total_return_nav_brl = round(current_nav_brl + dividends_paid_brl, 2)

    moic_simple = (
        round(current_nav_brl / total_invested_brl, 3) if total_invested_brl > 0 else 0.0
    )
    moic_current = (
        round(current_nav_brl / weighted_avg_invested_brl, 3)
        if weighted_avg_invested_brl > 0 else 0.0
    )
    moic_total_return = (
        round(total_return_nav_brl / weighted_avg_invested_brl, 3)
        if weighted_avg_invested_brl > 0 else 0.0
    )

    # Annualized CAGR на time-weighted базе. Для периода < 1 мес — пропускаем
    # (annualized на 7 днях даёт бессмысленно огромные цифры).
    annualized_pct = 0.0
    if years_since_launch >= 0.083 and moic_total_return > 0:
        annualized_pct = round(
            (moic_total_return ** (1.0 / years_since_launch) - 1.0) * 100.0, 2
        )

    # Timeline для UI tooltip
    capital_contributions_out = [
        {
            "date": c["date"].isoformat(),
            "brl": round(c["brl"], 2),
            "kind": c["kind"],
            "days_worked": max(0, (as_of - c["date"]).days),
        }
        for c in contributions
    ]

    return BalanceReport(
        project=project,
        as_of=as_of,
        # Assets
        cash_brl=cash_brl,
        accounts_receivable_brl=accounts_receivable_brl,
        inventory_brl=inventory_brl,
        assets_total=assets_total,
        # Liabilities
        accounts_payable_brl=accounts_payable_brl,
        loans_balance_brl=loans_balance_brl,
        liabilities_total=liabilities_total,
        # Equity
        initial_equity_brl=initial_equity_brl,
        accumulated_profit_brl=accumulated_profit_brl,
        dividends_paid_brl=dividends_paid_brl,
        equity_total=equity_total,
        # Reconciliation
        balance_delta_brl=balance_delta_brl,
        # Investment return (time-weighted MOIC + simple for comparison)
        total_invested_brl=total_invested_brl,
        weighted_avg_invested_brl=weighted_avg_invested_brl,
        current_nav_brl=current_nav_brl,
        total_return_nav_brl=total_return_nav_brl,
        moic_simple=moic_simple,
        moic_current=moic_current,
        moic_total_return=moic_total_return,
        annualized_pct=annualized_pct,
        years_since_launch=years_since_launch,
        launch_date_iso=launch_iso_str,
        capital_contributions=capital_contributions_out,
        # Legacy (back-compat)
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
        stock_missing_sku_details=stock_missing_sku_details,
        stock_by_supplier_type=stock_by_supplier_type,
    )
