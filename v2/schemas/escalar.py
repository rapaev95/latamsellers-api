"""Pydantic models for /api/v2/escalar/*."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel


class ABCProduct(BaseModel):
    sku: str
    title: str
    itemId: Optional[str] = None
    units: int
    revenue: float
    avgPrice: float
    commPerUnit: float
    shipPerUnit: float
    adPerUnit: float
    storagePerUnit: float
    refundPerUnit: float
    unitCost: float
    margin: float
    marginPct: float
    roi: float
    abcGrade: Literal["A", "B", "C"]
    revenuePct: float
    cumulativePct: float
    currentStock: int
    salesPerDay: float
    daysOfStock: float
    leadTime: int
    batchSize: int
    liberacao: int
    fullCycle: float
    turnsPerYear: float
    annualRoi: float
    reorderStatus: Literal["crit", "warn", "ok"]
    orderByDate: Optional[str] = None
    shippingMode: Optional[str] = None
    project: Optional[str] = None
    returnPct: float
    adPct: float
    # Dados Fiscais (ML официальный каталог) — заполняется из sku_catalog
    ncm: Optional[str] = None
    origem_type: Optional[Literal["import", "local"]] = None
    peso_liquido_kg: Optional[float] = None
    peso_bruto_kg: Optional[float] = None
    ean: Optional[str] = None
    csosn_venda: Optional[str] = None
    descricao_nfe: Optional[str] = None


class ABCMeta(BaseModel):
    files: list[str]
    totalRows: int
    uniqueSales: int
    totalRevenue: float
    totalUnits: int
    storageSkus: int
    storageFiles: int
    projects: list[str]
    days: Union[int, Literal["all"]]
    periodFrom: Optional[str] = None
    periodTo: Optional[str] = None
    snoozedSkus: list[str]


class EscalarProductsOut(BaseModel):
    products: list[ABCProduct]
    hasData: bool
    meta: ABCMeta


class SnoozeIn(BaseModel):
    sku: str
    snoozed: bool


class SnoozeOut(BaseModel):
    snoozedSkus: list[str]


# ── Position tracking (Marketing) ─────────────────────────────────────────

class PositionCheckOut(BaseModel):
    itemId: str
    keyword: str
    position: Optional[int] = None  # 1-based organic rank; None when not found or only in ads
    found: bool
    totalResults: int
    pagesScanned: int
    siteId: str = "MLB"
    adsAbove: Optional[int] = None  # sponsored slots encountered before the organic hit


class TrackKeywordIn(BaseModel):
    itemId: str
    keyword: str
    siteId: str = "MLB"
    categoryId: Optional[str] = None


class TrackedKeyword(BaseModel):
    id: int
    itemId: str
    keyword: str
    siteId: str
    categoryId: Optional[str] = None
    createdAt: str
    lastPosition: Optional[int] = None
    lastCheckedAt: Optional[str] = None
    lastFound: Optional[bool] = None


class TrackedListOut(BaseModel):
    tracked: list[TrackedKeyword]
