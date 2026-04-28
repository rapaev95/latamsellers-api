"""Pydantic models for /api/v2/finance/*.

Mirrors the dataclasses in `_admin/finance.py` (PnLReport, CashFlowReport, BalanceReport).
Field names kept snake_case to match Python originals — TypeScript side mirrors that
to avoid silent renames.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── ОПиУ ────────────────────────────────────────────────────────────────────

class PnLLineOut(BaseModel):
    label: str
    amount_brl: float
    is_total: bool = False
    note: str = ""


class PnLReportOut(BaseModel):
    """Mirrors finance.PnLReport. `period` (tuple of dates) lives at the bundle
    level — not duplicated here.
    """
    model_config = {"extra": "ignore"}

    project: str
    revenue_gross: float = 0
    taxas_ml: float = 0
    revenue_net: float = 0
    operating_expenses: list[PnLLineOut] = []
    operating_profit: float = 0
    cogs: Optional[float] = None
    net_profit: Optional[float] = None
    vendas_count: int = 0
    vendas_delivered_count: int = 0
    vendas_returned_count: int = 0
    margin_pct: float = 0
    # COGS diagnostics for the UI — SKUs without unit_cost_brl in sku_catalog
    # (user needs to load Dados Fiscais or fill them manually on /finance/sku-mapping).
    cogs_missing_skus: list[str] = []
    cogs_missing_units: int = 0
    # [{sku, mlb, units}] — per-SKU детализация проданных SKU без себестоимости.
    cogs_missing_sku_details: list[dict[str, Any]] = []
    unit_cost_per_sku: dict[str, float] = {}
    # DAS/Simples/LP metadata for UI badge (regime, anexo, faixa, effective %).
    # Shape: см. compute_das() в v2/legacy/tax_brazil.py.
    tax_info: Optional[dict] = None
    # Retirada de estoque Full breakdown (вывоз/утилизация). См. legacy/finance.compute_retirada_cost.
    # Shape: {tarifa_envio, tarifa_descarte, cogs_descarte, tarifa_other, units_*,
    # missing_cost_skus: [{sku,units,mlb,titulo}], fallback_avg_used,
    # by_sku: {sku → {forma, units, tarifa, cogs, cost_source, mlb, titulo, custo_ids}},
    # by_custo_id: {custo_id → {custo_id, date, sku, mlb, titulo, variacao, units, valor,
    #               original_forma, effective_forma, overridden}},
    # overrides_applied: int — сколько строк было вручную переключено через
    # POST /finance/retirada-overrides,
    # rows_count, source_files}.
    retirada_summary: Optional[dict[str, Any]] = None


# ── ДДС ─────────────────────────────────────────────────────────────────────

class CashFlowReportOut(BaseModel):
    model_config = {"extra": "ignore"}

    project: str
    opening_balance: float = 0
    # Источник opening_balance: "configured" (project.opening_balance),
    # "partner_investments" (fallback из partner_txs в ДДС), "none".
    opening_balance_source: str = "none"
    inflows_operating: float = 0
    inflows_count: int = 0
    inflows_financing: float = 0
    inflows_partner: float = 0
    outflows_operating: float = 0
    outflows_other: float = 0
    closing_balance: float = 0
    new_transactions: list[dict[str, Any]] = []
    partner_txs: list[dict[str, Any]] = []
    other_expenses_txs: list[dict[str, Any]] = []


# ── Баланс ───────────────────────────────────────────────────────────────────

class BalanceReportOut(BaseModel):
    """Full accounting balance: Assets = Liabilities + Equity. `balance_delta`
    surfaces rounding / missing-data gaps to the UI.

    Legacy flow-based fields (inflow_* / outflow_*) are kept for backwards
    compatibility with existing /pnl-matrix callers — they duplicate the new
    structured fields and will be removed in a follow-up version.
    """
    model_config = {"extra": "ignore"}

    project: str
    as_of: Optional[str] = None              # ISO YYYY-MM-DD (coerce from date)

    @field_validator("as_of", mode="before")
    @classmethod
    def _coerce_as_of(cls, v):
        # BalanceReport dataclass хранит as_of как date, схема — как str.
        if isinstance(v, date):
            return v.isoformat()
        return v

    # ── Assets ───────────────────────────────────────────────────────────
    cash_brl: float = 0                      # opening + Σ cashflow until as_of
    accounts_receivable_brl: float = 0       # always 0 in A1 model (vendas ≡ cash)
    inventory_brl: float = 0                 # stock × unit_cost (= stock_value_brl)
    assets_total: float = 0

    # ── Liabilities ──────────────────────────────────────────────────────
    accounts_payable_brl: float = 0          # Σ unpaid overdue planned_payments
    loans_balance_brl: float = 0             # Σ f2_loans.outstanding_brl
    liabilities_total: float = 0

    # ── Equity ───────────────────────────────────────────────────────────
    initial_equity_brl: float = 0            # project.initial_equity_brl
    accumulated_profit_brl: float = 0        # Σ net_profit (launch → as_of)
    dividends_paid_brl: float = 0            # Σ f2_dividends
    equity_total: float = 0

    # ── Reconciliation ───────────────────────────────────────────────────
    balance_delta_brl: float = 0             # assets − (liab + equity). |Δ| > 100 → warn

    # ── Investment return (time-weighted MOIC + annualized) ─────────────
    # «Вложил 10K, сейчас 40K → x4» — главная метрика владельца.
    total_invested_brl: float = 0            # Σ всех взносов (nominal, для simple MOIC)
    weighted_avg_invested_brl: float = 0     # time-weighted — честная база для MOIC
    current_nav_brl: float = 0               # assets − liabilities
    total_return_nav_brl: float = 0          # NAV + dividends_paid
    moic_simple: float = 0                   # NAV / total_invested
    moic_current: float = 0                  # NAV / weighted_avg_invested (primary)
    moic_total_return: float = 0             # (NAV + divs) / weighted_avg_invested
    annualized_pct: float = 0                # CAGR по time-weighted MOIC
    years_since_launch: float = 0
    launch_date_iso: Optional[str] = None
    # Timeline: [{date, brl, kind, days_worked}]
    capital_contributions: list[dict[str, Any]] = []

    # ── Legacy flow fields (kept for back-compat) ────────────────────────
    inflow_usdt_brl: float = 0
    inflow_usdt_usd: float = 0
    inflow_sales_net: float = 0
    inflow_sales_count: int = 0
    inflows_total: float = 0
    outflow_mercadoria: float = 0
    outflow_publicidade: float = 0
    outflow_devolucoes: float = 0
    outflow_full_express: float = 0
    outflow_das: float = 0
    outflow_armazenagem: float = 0
    outflow_aluguel: float = 0
    outflows_total: float = 0
    saldo: float = 0
    pending_rental_usd: float = 0
    pending_rental_brl: float = 0
    saldo_final: float = 0
    cost_per_unit: Optional[float] = None
    stock_units: int = 0
    stock_value_brl: float = 0
    stock_missing_skus: list[str] = []
    stock_missing_units: int = 0
    # [{sku, mlb, units}] — для UI (ML-ссылка + deep link на /finance/sku-mapping).
    stock_missing_sku_details: list[dict[str, Any]] = []
    stock_by_supplier_type: dict[str, Any] = {}


# ── Bundle (single endpoint returns three reports) ──────────────────────────

class ReportsBundleOut(BaseModel):
    project: str
    period: dict[str, str]  # {"from": "...", "to": "..."}
    basis: str
    pnl: Optional[PnLReportOut] = None
    pnl_error: Optional[str] = None
    cashflow: Optional[CashFlowReportOut] = None
    cashflow_error: Optional[str] = None
    balance: Optional[BalanceReportOut] = None
    balance_error: Optional[str] = None


class ProjectsListOut(BaseModel):
    projects: dict[str, dict[str, Any]]
    count: int


# ── SKU Маппинг ──────────────────────────────────────────────────────────────

class SkuRow(BaseModel):
    sku: str
    title: str = ""
    mlb: str = ""
    link: str = ""
    project: str = ""
    supplier_type: str = "local"  # 'local' | 'import'
    unit_cost_brl: Optional[float] = None
    note: str = ""


class SkuGroup(BaseModel):
    """Auto-detected group of SKUs sharing the same alpha prefix."""
    prefix: str
    sample_title: str = ""
    project: str = ""  # current dominant project assignment
    skus: list[str]    # normalized (uppercase) keys


class SkuMappingOut(BaseModel):
    skus: list[SkuRow]                     # all merged SKUs (vendas + catalog)
    groups: list[SkuGroup]                 # grouped by alpha prefix, sorted by size desc
    project_counts: dict[str, int]         # {project_id: n_skus} (+ "" / "—" for unassigned)
    project_ids: list[str]                 # available project keys
    total: int


class SkuUpdate(BaseModel):
    sku: str
    project: Optional[str] = None
    supplier_type: Optional[str] = None    # 'local' | 'import'
    unit_cost_brl: Optional[float] = None
    extra_fixed_cost_brl: Optional[float] = None  # manual extra per-unit cost
    note: Optional[str] = None


class SkuBulkSaveIn(BaseModel):
    updates: list[SkuUpdate]


class SkuBulkSaveOut(BaseModel):
    saved: int
    catalog_total: int


# ── PnL Monthly Matrix ──────────────────────────────────────────────────────

class PnlMatrixRow(BaseModel):
    key: str
    label: str                # i18n key like "pnl_rev_gross"
    section: str              # "REVENUE" | "EXPENSES" | "SUMMARY"
    values: dict[str, float]  # { "YYYY-MM": amount }
    total: float
    is_total: bool = False
    is_pct: bool = False
    is_count: bool = False
    is_info: bool = False
    # Дополнительная мета — используется только для строки DAS (regime, anexo,
    # по-месячная faixa/effective/RBT12). None для остальных строк.
    tax_info: Optional[dict] = None


class PnlMatrixOut(BaseModel):
    project: str
    months: list[str]    # sorted YYYY-MM
    years: list[str]
    rows: list[PnlMatrixRow]


# ── Retirada Overrides (per-row политика «списание / в обороте») ────────────
#
# Хранится в user_data JSONB (per-user, per-project, per-custo_id).
# По умолчанию forma берётся из ML-отчёта (Forma de retirada). Override
# переопределяет это значение — например, ML записал Envio para o endereço,
# но реально товар утилизирован → пользователь ставит forma="descarte"
# через POST /finance/retirada-overrides. Логика (Envio = только тариф,
# Descarte = тариф + COGS) — в legacy/finance.compute_retirada_cost.

class RetiradaOverrideItem(BaseModel):
    """Один override для retirada-операции. forma — финальное значение."""
    custo_id: str
    forma: str  # "descarte" | "envio"


class RetiradaOverridesIn(BaseModel):
    """POST body. Заменяет override-map для проекта целиком (replace, не merge).
    Передавай пустой `overrides: []` чтобы сбросить все overrides проекта.
    """
    project: str
    overrides: list[RetiradaOverrideItem]


class RetiradaOverridesOut(BaseModel):
    project: str
    # custo_id → forma ("descarte" | "envio"). Возвращается отфильтрованным:
    # только legitimate values (legacy/reports.load_retirada_overrides
    # канонизирует и отбрасывает мусор).
    overrides: dict[str, str]
    saved_count: int = 0     # сколько записей сохранилось (на ответе POST)


# ── Orphan Pacotes (multi-item ML orders without per-SKU rows) ──────────────

class OrphanPacoteOut(BaseModel):
    order_id: str
    data: str = ""
    estado: str = ""
    bucket: str = ""           # "delivered" | "returned" | "in_progress"
    comprador: str = ""
    total_brl: float = 0
    ml_url: str                # link to mercadolivre.com.br order detail
    assigned_project: Optional[str] = None


class OrphanPacotesResponse(BaseModel):
    items: list[OrphanPacoteOut]
    unassigned_count: int
    unassigned_total_brl: float
    available_projects: list[str]


class OrphanSaveIn(BaseModel):
    # Mapping of order_id → project_slug; `null` means "clear the assignment".
    assignments: dict[str, Optional[str]]


class OrphanSaveOut(BaseModel):
    saved: int
    total_assignments: int


# ── Uploads (Phase 4 — /finance/upload) ─────────────────────────────────────

class UploadItem(BaseModel):
    id: int
    filename: str
    size_bytes: int
    created_at: str                  # ISO-8601 UTC


class UploadSourceGroup(BaseModel):
    source_key: str
    count: int
    items: list[UploadItem]


class UploadsListOut(BaseModel):
    sources: list[UploadSourceGroup]
    total_count: int


class UploadSaveOut(BaseModel):
    id: int
    filename: str
    source_key: str                  # resolved key
    detected: bool                   # True = server auto-detected; False = client provided
    size_bytes: int
    was_duplicate: bool              # True = SHA already existed; created_at refreshed
    unlocked: bool = False           # True = file was encrypted and we decrypted it
    # Только для source_key="dados_fiscais" — статистика sync в sku_catalog.
    # {created, updated_fields, cost_updated, skipped, synced_at, total_skus}
    # или {"error": "..."} при failure парсинга.
    dados_fiscais_sync: Optional[dict[str, Any]] = None


class SourceCatalogEntry(BaseModel):
    key: str
    name: str = ""
    file_pattern: str = ""
    frequency: str = ""              # "monthly" | "daily" | ...
    type: str = ""                   # "ecom" | "bank" | "tax" | ...
    description: str = ""


class SourceCatalogOut(BaseModel):
    sources: list[SourceCatalogEntry]


# ── Classification + Bank Rules (Phase 5) ───────────────────────────────────

class TransactionRule(BaseModel):
    keywords: list[str] = []
    category: str = "uncategorized"
    project: Optional[str] = None
    label: str = ""


class RulesOut(BaseModel):
    rules: list[TransactionRule]
    count: int


class RulesSaveIn(BaseModel):
    rules: list[TransactionRule]


class BankTxRow(BaseModel):
    idx: int                  # row index in the source DataFrame
    date: str
    value_brl: float
    description: str
    category: str
    project: str = ""
    label: str = ""
    confidence: str = "none"  # "auto" | "none" | "manual"
    auto: bool = False


class TransactionsOut(BaseModel):
    upload_id: int
    source_key: str
    filename: str
    rows: list[BankTxRow]
    categories: list[str]     # enum for UI dropdown
    projects: list[str]       # user project slugs
    saved_overrides_count: int = 0


class TransactionOverride(BaseModel):
    idx: int
    category: Optional[str] = None
    project: Optional[str] = None
    label: Optional[str] = None


class ClassificationSaveIn(BaseModel):
    overrides: list[TransactionOverride] = []


class ClassificationSaveOut(BaseModel):
    saved: int
    total_overrides: int


# ── Onboarding Wizard (Phase 6) ─────────────────────────────────────────────

class OnboardingState(BaseModel):
    """Per-user wizard state. `data` accumulates form fields across steps."""
    step: int = 1             # 1..10
    completed: bool = False
    data: dict[str, Any] = {}


class ProjectCreateIn(BaseModel):
    """Minimal project create payload — wizard step 2. Catches the 90% case;
    full project editing stays on /finance/projects."""
    project_id: str
    project_type: str = "ecom"           # "ecom" | "service" | "other"
    description: str = ""
    sku_prefixes: list[str] = []
    compensation_mode: str = "profit_share"
    profit_share_pct: Optional[float] = None
    # Inherited from onboarding Step 1 (company-level), stamped onto every
    # project from this wizard run so DAS/Balance compute correctly without
    # a second trip to /finance/projects settings.
    tax_regime: Optional[str] = None     # "simples_nacional" | "lucro_presumido" | ""
    simples_anexo: Optional[str] = None  # "I" | "II" | "III" | ""
    ml_only_revenue: Optional[bool] = None  # true → прогрессивный Simples без RBT12-warning


class ProjectCreateOut(BaseModel):
    project_id: str
    created: bool
    total_projects: int


class ProjectUpdateIn(BaseModel):
    """Full project edit payload. `fields` are merged using the whitelist in
    legacy.config._PROJECT_EDITABLE_KEYS; `rental_fields` merged into rental dict."""
    fields: dict[str, Any] = {}
    rental_fields: Optional[dict[str, Any]] = None


class ProjectMutOut(BaseModel):
    """Return for PUT / DELETE — confirms target + outcome."""
    project_id: str
    updated: bool = False
    deleted: bool = False
    exists: bool = True


# ── Upload Preview ──────────────────────────────────────────────────────────

class UploadPreviewOut(BaseModel):
    upload_id: int
    filename: str
    source_key: str
    size_bytes: int
    total_rows: Optional[int] = None
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}       # source-specific aggregates
    parse_error: Optional[str] = None


# ── Manual Cashflow Entries ─────────────────────────────────────────────────

class ManualCashflowEntryIn(BaseModel):
    kind: str                     # "partner_contributions" | "manual_expenses" | "manual_supplier" | "loan_given" | "loan_received"
    date: str                     # ISO YYYY-MM-DD
    valor: float                  # в исходной валюте (см. currency)
    note: str = ""
    # Extra kind-specific fields the UI may populate. `from_` aliases to JSON "from"
    # (Python reserved word). `populate_by_name` lets clients send either name.
    from_: Optional[str] = Field(default=None, alias="from")
    category: Optional[str] = None
    source: Optional[str] = None
    # Multi-currency support (partner_contributions and loans often come in USDT/USD).
    # currency defaults to "BRL" for backward compat; rate_brl required when currency ≠ BRL.
    currency: Optional[str] = "BRL"         # "BRL" | "USDT" | "USD"
    rate_brl: Optional[float] = None         # курс к BRL на дату (если currency != BRL)
    # Inter-project loans: who's on the other side. Required for loan_given / loan_received.
    # Backend writes a mirror entry in the counterparty project, linked by auto-generated loan_id.
    counterparty_project: Optional[str] = None
    # Partner contribution flag: "тестовая закупка, не считать как инвестицию".
    # Only applies to kind=partner_contributions. When True, the entry stays in
    # ДДС as an inflow (cash came into the project), but compute_balance skips
    # it from total_invested_brl — so MOIC, прирост, % годовых, все считаются
    # без этих тестовых денег. Default False = old behaviour.
    test_only: Optional[bool] = False

    model_config = {"populate_by_name": True}


class ManualCashflowEntriesOut(BaseModel):
    project: str
    partner_contributions: list[dict[str, Any]] = []
    manual_expenses: list[dict[str, Any]] = []
    manual_supplier: list[dict[str, Any]] = []
    loans_given: list[dict[str, Any]] = []       # этот проект ВЫДАЛ займ другому (outflow)
    loans_received: list[dict[str, Any]] = []    # этот проект ПОЛУЧИЛ займ от другого (inflow)


# ── Planned Payments (DDS Planning) ─────────────────────────────────────────

class PlannedPayment(BaseModel):
    id: Optional[int] = None
    date: str                               # ISO YYYY-MM-DD
    amount: float
    direction: str = "expense"              # "expense" | "income"
    recurrence: str = "once"                # "once" | "monthly" | "quarterly" | "yearly"
    contragent: str = ""
    category: str = ""
    note: str = ""
    project: Optional[str] = None
    paid_at: Optional[str] = None           # ISO timestamp — null = pending (AP feed)
    created_at: Optional[str] = None


class PlannedPaymentIn(BaseModel):
    date: str
    amount: float
    direction: str = "expense"
    recurrence: str = "once"
    contragent: str = ""
    category: str = ""
    note: str = ""
    project: Optional[str] = None


class MarkPaidIn(BaseModel):
    """Toggle-paid body: `paid=true` stamps now (or uses supplied `paid_at`);
    `paid=false` clears the flag (payment goes back to AP)."""
    paid: bool = True
    paid_at: Optional[str] = None           # optional explicit ISO timestamp


class PlannedPaymentsOut(BaseModel):
    payments: list[PlannedPayment] = []
    count: int = 0


class PlannedPaymentMutOut(BaseModel):
    updated: bool = False
    deleted: bool = False
    payment: Optional[PlannedPayment] = None


class MonthlyPlanBucket(BaseModel):
    month: str                              # YYYY-MM
    income: list[dict[str, Any]] = []
    expense: list[dict[str, Any]] = []
    total_in: float = 0
    total_out: float = 0
    net: float = 0


class MonthlyPlanOut(BaseModel):
    months: list[str]
    buckets: dict[str, MonthlyPlanBucket]


class RecurringSuggestion(BaseModel):
    contragent: str
    category: str = ""
    direction: str                          # "expense" | "income"
    avg_amount: float
    min_amount: float
    max_amount: float
    months_count: int
    total_txs: int


class RecurringSuggestionsOut(BaseModel):
    suggestions: list[RecurringSuggestion]
    min_occurrences: int


# ── Rental payments (per-project cash-basis aluguel schedule) ───────────────

class RentalPaymentIn(BaseModel):
    date: str                            # ISO YYYY-MM-DD (дата платежа или планируемая)
    amount_usd: float                    # исходная сумма в USD
    rate_brl: Optional[float] = None     # курс на дату (обязательно для status=paid)
    status: str                          # "paid" | "pending"
    note: str = ""


class RentalPaymentOut(BaseModel):
    index: int                           # позиция в projects_db[pid][rental][payments]
    date: str
    amount_usd: float
    rate_brl: Optional[float] = None
    amount_brl: float                    # вычислено: amount_usd × rate_brl
    status: str
    note: str = ""


class RentalPaymentsListOut(BaseModel):
    project: str
    rate_usd: float                      # из projects_db (эталонная сумма платежа, для автозаполнения)
    period: str                          # "month" | "quarter"
    payments: list[RentalPaymentOut]
    total_paid_brl: float
    total_pending_brl: float
    # Последняя оплата + период = дата до которой аренда закрыта фактически.
    last_paid_date: Optional[str] = None    # ISO YYYY-MM-DD
    paid_until: Optional[str] = None        # ISO YYYY-MM-DD (last_paid_date + period − 1 день)
    launch_date: Optional[str] = None       # ISO YYYY-MM-DD — старт проекта, для UI


# ── Publicidade invoices (manual Mercado Ads faturas, 12-12 billing cycle) ──

class PublicidadeInvoiceIn(BaseModel):
    """Вход: либо полная дата (anchor), либо только месяц — тогда день берётся из project.billing_cycle_day.

    Правила:
      - если есть `date` (YYYY-MM-DD) — используется как anchor.
      - иначе если есть `month` (YYYY-MM) — anchor = month + project.billing_cycle_day.
      - если cycle_day не настроен и есть только month → 400.

    Для PATCH: все поля опциональны (что прислал, то обновится).
    """
    date: Optional[str] = None          # ISO YYYY-MM-DD — anchor (legacy/ручной ввод)
    month: Optional[str] = None         # ISO YYYY-MM — месяц фатуры, день подставится из проекта
    valor: Optional[float] = None       # invoice amount in BRL (optional для PATCH)
    note: Optional[str] = None


class PublicidadeInvoiceOut(BaseModel):
    index: int           # position in projects_db.json[pid][manual_publicidade]
    date: str            # ISO YYYY-MM-DD — anchor
    valor: float
    note: str = ""


class PubCsvWindow(BaseModel):
    from_: str = Field(..., alias="from")   # ISO YYYY-MM-DD
    to: str                                  # ISO YYYY-MM-DD

    model_config = {"populate_by_name": True}


class PublicidadeInvoicesListOut(BaseModel):
    project: str
    invoices: list[PublicidadeInvoiceOut]
    launch_date: Optional[str] = None
    billing_cycle_day: Optional[int] = None          # день закрытия цикла (1..28)
    publicidade_csv_window: Optional[PubCsvWindow] = None


# ── Publicidade reconciliation ───────────────────────────────────────────────

class FileUsageInfo(BaseModel):
    """Один источник publicidade (CSV или ручная fatura) с его вкладом в total."""
    file_name: str
    days_used: int
    total_days: int                           # CSV: период файла; fatura: всегда 30
    ratio: float
    contribution: float                       # BRL, вклад в total этого файла за период
    is_fatura: bool
    kind: str                                 # "csv" | "fatura"


class PublicidadeReconciliationOut(BaseModel):
    """Сверка publicidade за период: CSV (реальный дневной расход) vs fatura (anchor + /30)."""
    project: str
    period_from: str                          # YYYY-MM-DD
    period_to: str                            # YYYY-MM-DD
    csv_total: float
    csv_files_used: list[FileUsageInfo]
    fatura_total: float
    fatura_files_used: list[FileUsageInfo]
    delta: float                              # fatura_total − csv_total
    uncovered_days_csv: int
    uncovered_days_fatura: int
    total_days: int


# ── Coverage timeline (publicidade + armazenagem, per-day как сегменты) ─────

class CoverageRange(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    model_config = {"populate_by_name": True}


class CoveragePublicidade(BaseModel):
    csv_segments: list[CoverageRange]
    fatura_segments: list[CoverageRange]
    uncovered_segments: list[CoverageRange]
    avg_segments: list[CoverageRange] = []      # дни, заполненные средним (если flag on)
    csv_raw_range: Optional[CoverageRange] = None
    csv_window: Optional[CoverageRange] = None
    fill_avg: bool = False                       # текущее состояние чекбокса


class CoverageArmazenagem(BaseModel):
    csv_segments: list[CoverageRange]
    uncovered_segments: list[CoverageRange]
    avg_segments: list[CoverageRange] = []
    csv_raw_range: Optional[CoverageRange] = None
    fill_avg: bool = False


class CoverageOut(BaseModel):
    project: str
    period_from: str
    period_to: str
    launch_date: Optional[str] = None
    publicidade: CoveragePublicidade
    armazenagem: CoverageArmazenagem


# ── Capital & Obligations (loans / dividends / AP feed) ─────────────────────

class LoanIn(BaseModel):
    project: str
    name: str
    principal_brl: float = 0
    outstanding_brl: float = 0
    monthly_payment_brl: float = 0
    rate_pct: float = 0
    start_date: Optional[str] = None        # ISO YYYY-MM-DD
    closed_at: Optional[str] = None         # null = active
    note: str = ""


class LoanOut(LoanIn):
    id: int
    created_at: Optional[str] = None


class LoansListOut(BaseModel):
    project: str
    loans: list[LoanOut] = []
    total_outstanding_brl: float = 0


class LoanMutOut(BaseModel):
    updated: bool = False
    deleted: bool = False
    loan: Optional[LoanOut] = None


class DividendIn(BaseModel):
    project: str
    date: str                                # ISO YYYY-MM-DD
    amount_brl: float
    note: str = ""


class DividendOut(DividendIn):
    id: int
    created_at: Optional[str] = None


class DividendsListOut(BaseModel):
    project: str
    dividends: list[DividendOut] = []
    total_amount_brl: float = 0


class DividendMutOut(BaseModel):
    updated: bool = False
    deleted: bool = False
    dividend: Optional[DividendOut] = None


class APListOut(BaseModel):
    """Accounts Payable feed — unpaid planned_payments with date <= as_of."""
    project: str
    as_of: str                               # ISO YYYY-MM-DD
    items: list[PlannedPayment] = []
    total_brl: float = 0
