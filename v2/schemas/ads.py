"""Pydantic models for /api/v2/ads/*.

Field shapes mirror ML Product Ads v2 response bodies
(see https://api.mercadolibre.com/advertising/*). `roas_target` replaces the
retired `acos_target` (decommissioned 2026-03-30); we never expose acos_target
to the UI.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Advertiser(BaseModel):
    advertiser_id: int
    site_id: Optional[str] = None
    advertiser_name: Optional[str] = None
    account_name: Optional[str] = None
    product_id: str = "PADS"


class AdvertisersOut(BaseModel):
    advertisers: list[Advertiser]


# ── Metrics ────────────────────────────────────────────────────────────────
# Campaign-level metrics include roas/sov/cvr/ctr that aren't present on the
# per-ad response, but the shared bits let the UI share a row-renderer.

class CampaignMetrics(BaseModel):
    clicks: float = 0
    prints: float = 0
    ctr: float = 0
    cost: float = 0
    cpc: float = 0
    acos: float = 0
    roas: float = 0
    sov: float = 0
    cvr: float = 0
    direct_amount: float = 0
    indirect_amount: float = 0
    total_amount: float = 0
    direct_units_quantity: float = 0
    indirect_units_quantity: float = 0
    units_quantity: float = 0
    direct_items_quantity: float = 0
    indirect_items_quantity: float = 0
    advertising_items_quantity: float = 0
    organic_units_quantity: float = 0
    organic_units_amount: float = 0
    organic_items_quantity: float = 0


class AdMetrics(BaseModel):
    clicks: float = 0
    prints: float = 0
    cost: float = 0
    cpc: float = 0
    acos: float = 0
    direct_amount: float = 0
    indirect_amount: float = 0
    total_amount: float = 0
    direct_units_quantity: float = 0
    indirect_units_quantity: float = 0
    units_quantity: float = 0
    organic_units_quantity: float = 0
    organic_items_quantity: float = 0
    direct_items_quantity: float = 0
    indirect_items_quantity: float = 0
    advertising_items_quantity: float = 0


# ── Campaigns ──────────────────────────────────────────────────────────────

class AdCampaign(BaseModel):
    id: int
    product_id: str = "PADS"          # PADS | DISPLAY | BADS
    name: Optional[str] = None
    status: Optional[str] = None
    strategy: Optional[str] = None    # PADS only — PROFITABILITY/INCREASE/VISIBILITY
    budget: Optional[float] = None
    automatic_budget: Optional[bool] = None  # PADS only
    roas_target: Optional[float] = None      # PADS only
    channel: Optional[str] = None
    advertiser_id: int
    date_created: Optional[str] = None
    last_updated: Optional[str] = None
    # Generic time window — DISPLAY (start_date/end_date) and BADS use these
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # DISPLAY: type=GUARANTEED/PROGRAMMATIC, goal=awareness/consideration/...
    # BADS: campaign_type=automatic|custom
    campaign_type: Optional[str] = None
    goal: Optional[str] = None
    site_id: Optional[str] = None
    # BADS-specific
    headline: Optional[str] = None
    cpc: Optional[float] = None
    currency: Optional[str] = None
    official_store_id: Optional[int] = None
    destination_id: Optional[int] = None
    metrics: CampaignMetrics = CampaignMetrics()
    metrics_date_from: Optional[str] = None
    metrics_date_to: Optional[str] = None
    synced_at: Optional[str] = None


class CampaignsOut(BaseModel):
    campaigns: list[AdCampaign]
    total: int
    stale: bool  # True if cache is older than STALE_THRESHOLD (UI hint)
    synced_at: Optional[str] = None


class CampaignDetail(AdCampaign):
    daily: list[dict] = []  # [{date, clicks, ...}]


# ── Ads (product_ads) ──────────────────────────────────────────────────────

class AdItem(BaseModel):
    item_id: str
    advertiser_id: int
    campaign_id: Optional[int] = None
    title: Optional[str] = None
    status: Optional[str] = None
    price: Optional[float] = None
    thumbnail: Optional[str] = None
    permalink: Optional[str] = None
    domain_id: Optional[str] = None
    brand_value_name: Optional[str] = None
    metrics: AdMetrics = AdMetrics()
    metrics_date_from: Optional[str] = None
    metrics_date_to: Optional[str] = None


class AdsOut(BaseModel):
    ads: list[AdItem]
    total: int
    limit: int
    offset: int
    stale: bool
    synced_at: Optional[str] = None


# ── Sync ──────────────────────────────────────────────────────────────────

class SyncResult(BaseModel):
    status: Literal["ok", "skipped", "error"]
    user_id: int
    advertisers: int = 0
    campaigns: int = 0
    daily_rows: int = 0
    ads: int = 0
    message: Optional[str] = None
