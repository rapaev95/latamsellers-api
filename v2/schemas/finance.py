"""Pydantic models for /api/v2/finance/*.

Mirrors the dataclasses in `_admin/finance.py` (PnLReport, CashFlowReport, BalanceReport).
Field names kept snake_case to match Python originals — TypeScript side mirrors that
to avoid silent renames.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


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
    margin_pct: float = 0


# ── ДДС ─────────────────────────────────────────────────────────────────────

class CashFlowReportOut(BaseModel):
    model_config = {"extra": "ignore"}

    project: str
    opening_balance: float = 0
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
    model_config = {"extra": "ignore"}

    project: str
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


class PnlMatrixOut(BaseModel):
    project: str
    months: list[str]    # sorted YYYY-MM
    years: list[str]
    rows: list[PnlMatrixRow]


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


class SourceCatalogEntry(BaseModel):
    key: str
    name: str = ""
    file_pattern: str = ""
    frequency: str = ""              # "monthly" | "daily" | ...
    type: str = ""                   # "ecom" | "bank" | "tax" | ...
    description: str = ""


class SourceCatalogOut(BaseModel):
    sources: list[SourceCatalogEntry]
