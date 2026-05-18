"""Per-category attribute schema cache.

Each ML category has its own «ficha técnica» — list of attributes with
value_type, allowed values, tags, etc. The Sprint 3 wizard renders this
as a dynamic form per category.

Shape of /categories/{id}/attributes (relevant fields):
  [
    {
      "id": "BRAND",
      "name": "Marca",
      "value_type": "string" | "number" | "number_unit" | "list" | "boolean" | "grid_id" | "grid_row_id",
      "tags": {
        "required": true,
        "conditional_required": true,
        "allow_variations": true,
        "variation_attribute": true,
        "grid_template_required": true,
        "catalog_required": true,
        "fixed": true,
        "hidden": true,
        "multivalued": true,
        ...
      },
      "values": [{"id": "...", "name": "..."}, ...],     // for list types
      "allowed_units": [{"id": "...", "name": "..."}],   // for number_unit
      "default_unit": "...",
      "hierarchy": "PARENT_PK" | "CHILD_PK" | "ITEM" | "FAMILY" | "SALE_TERMS",
      "value_max_length": int,
      "relevance": 1..5,
      "attribute_group_id": "MAIN" | "OTHERS" | ...,
      "attribute_group_name": "..."
    },
    ...
  ]

We cache the raw JSON 24h. Categories' attribute schema does change (ML
adds new attrs, marks others required), so a daily refresh keeps us fresh.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
ATTRIBUTES_TTL = timedelta(hours=24)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_category_attributes (
  category_id     TEXT PRIMARY KEY,
  attributes      JSONB NOT NULL,       -- raw array from ML
  required_ids    JSONB,                -- denormalized for fast filter
  fetched_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ──────────────────────────────────────────────────────────────────────────────
# Tag/value helpers — used by both backend filters and the UI to decide how
# to render an attribute. The truth table is documented in
# `_AGENT_INDEX.md` §5.2 and `Identificadores de produtos ML.md` §«Lógica».
# ──────────────────────────────────────────────────────────────────────────────

def _tags(attr: dict[str, Any]) -> dict[str, Any]:
    """Tags can come back as either a dict {key: true} or a list of strings.
    Normalize to dict for predictable access."""
    raw = attr.get("tags")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {t: True for t in raw if isinstance(t, str)}
    return {}


def is_required(attr: dict[str, Any]) -> bool:
    """Hard-required: must be present in POST /items."""
    return bool(_tags(attr).get("required"))


def is_conditional_required(attr: dict[str, Any]) -> bool:
    """Soft-required: required when other context applies (brand has 30+
    GTINs, category has special flag etc.). For GTIN this is the gate that
    allows `EMPTY_GTIN_REASON` as an alternative."""
    return bool(_tags(attr).get("conditional_required"))


def is_allow_variations(attr: dict[str, Any]) -> bool:
    """Attribute can be used as a variation axis (COLOR, SIZE, ...).
    Goes into `attribute_combinations` of each variation."""
    return bool(_tags(attr).get("allow_variations"))


def is_variation_attribute(attr: dict[str, Any]) -> bool:
    """Per-variation attribute (e.g. GTIN per color). Goes inside each
    variation's `attributes`, not in `attribute_combinations`."""
    return bool(_tags(attr).get("variation_attribute"))


def is_grid_template_required(attr: dict[str, Any]) -> bool:
    """Drives the tabela de medidas chart search (Sprint 4 work)."""
    return bool(_tags(attr).get("grid_template_required"))


def hierarchy(attr: dict[str, Any]) -> str:
    """Returns 'ITEM' | 'PARENT_PK' | 'CHILD_PK' | 'FAMILY' | 'SALE_TERMS' |
    'UNCLASSIFIED'. UPTIN family grouping uses PARENT_PK/CHILD_PK."""
    h = attr.get("hierarchy")
    if isinstance(h, str) and h:
        return h
    # Some attributes don't declare hierarchy explicitly — default to ITEM
    # which is the most common case.
    return "ITEM"


def is_sale_term(attr: dict[str, Any]) -> bool:
    """Attribute with hierarchy=SALE_TERMS goes into body.sale_terms[],
    NOT body.attributes[]. WARRANTY_TYPE/_TIME, MANUFACTURING_TIME, etc."""
    return hierarchy(attr) == "SALE_TERMS"


def is_hidden(attr: dict[str, Any]) -> bool:
    return bool(_tags(attr).get("hidden"))


def is_fixed(attr: dict[str, Any]) -> bool:
    """ML pre-fills + locks this attribute — we shouldn't render an input
    (e.g. some catalog-listing attrs come pre-set)."""
    return bool(_tags(attr).get("fixed"))


def required_ids(attrs: list[dict[str, Any]]) -> list[str]:
    return [a["id"] for a in attrs if is_required(a) and a.get("id")]


# ──────────────────────────────────────────────────────────────────────────────
# Fetch + cache
# ──────────────────────────────────────────────────────────────────────────────

def _is_stale(fetched_at: Optional[datetime], ttl: timedelta) -> bool:
    if fetched_at is None:
        return True
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_at > ttl


async def _fetch_attributes(
    http: httpx.AsyncClient, access_token: str, category_id: str
) -> list[dict[str, Any]]:
    r = await http.get(
        f"{ML_API_BASE}/categories/{category_id}/attributes",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_category_attributes_failed", request=r.request, response=r,
        )
    payload = r.json()
    return payload if isinstance(payload, list) else []


async def get_attributes(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    category_id: str,
    *,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {category_id, attributes: [...], required_ids: [...],
    cache_hit, fetched_at}.
    """
    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT attributes, required_ids, fetched_at
                FROM ml_category_attributes WHERE category_id = $1
                """,
                category_id,
            )
        if row and not _is_stale(row["fetched_at"], ATTRIBUTES_TTL):
            attrs = row["attributes"]
            if isinstance(attrs, str):
                attrs = json.loads(attrs)
            reqs = row["required_ids"]
            if isinstance(reqs, str):
                reqs = json.loads(reqs)
            return {
                "category_id": category_id,
                "attributes": attrs,
                "required_ids": reqs or [],
                "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    attrs = await _fetch_attributes(http, access_token, category_id)
    reqs = required_ids(attrs)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_category_attributes (category_id, attributes, required_ids, fetched_at)
            VALUES ($1, $2::jsonb, $3::jsonb, NOW())
            ON CONFLICT (category_id) DO UPDATE SET
                attributes = EXCLUDED.attributes,
                required_ids = EXCLUDED.required_ids,
                fetched_at = NOW()
            """,
            category_id, json.dumps(attrs), json.dumps(reqs),
        )
    return {
        "category_id": category_id,
        "attributes": attrs,
        "required_ids": reqs,
        "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# BRAND top-values — used by BrandPicker autocomplete in the UI.
# Endpoint: /catalog_domains/{DOMAIN_ID}/attributes/{ATTRIBUTE_ID}/top_values
# Used for any list-typed attribute, not just BRAND. We pass through.
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_top_values(
    http: httpx.AsyncClient,
    access_token: str,
    *,
    domain_id: str,
    attribute_id: str,
    body: Optional[dict] = None,
) -> dict[str, Any]:
    """POST /catalog_domains/{DOMAIN}/attributes/{ATTR}/top_values.

    Body is usually empty; ML accepts known_attributes for context-aware
    suggestions (e.g. MODEL filtered by BRAND).
    """
    r = await http.post(
        f"{ML_API_BASE}/catalog_domains/{domain_id}/attributes/{attribute_id}/top_values",
        json=body or {},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=15.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_top_values_failed", request=r.request, response=r,
        )
    payload = r.json()
    return {
        "domain_id": domain_id,
        "attribute_id": attribute_id,
        "body": payload,
    }
