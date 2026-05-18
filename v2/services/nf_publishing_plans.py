"""NF publishing plans — Sprint 5.

A `plan` is the resolved blueprint of «N families × cores × sizes → M Items»
derived from a parsed NF + the user's groupings/attribute choices. Once
ready it's the immediate input to Sprint 7's publisher.

Shape (state_json):
{
  "site_id": "MLB",
  "families": [
    {
      "family_id": "f-1",              // client-generated stable id
      "source_line_ns": [1,2,3],       // NF lines this family covers
      "category_id": "MLB...", "category_name": "...", "domain_id": "MLB-SNEAKERS",
      "family_name": "Tênis Unissex Casual",
      "attribute_values": {            // shared by ALL Items in family (PARENT_PK)
        "BRAND": {"value_id":"...","value_name":"Adidas"},
        ...
      },
      "keywords": ["casual", "dia a dia"],
      "cores":   ["Vermelho","Verde","Preto","Cáqui"],
      "tamanhos":["37","38","39","40"],
      "size_grid_id": "232382",
      "size_grid_rows": { "37":"row-101", "38":"row-102", ... },
      "ups": [                         // computed = cores × tamanhos
        {
          "up_id": "f-1-Vermelho-37",
          "color": "Vermelho", "size": "37",
          "available_qty": 7,           // user-editable; auto-spread default
          "size_grid_row_id": "row-101"
        }, ...
      ],
      "items_per_up": 1,               // Sprint 7 may go 1→N (Premium+Classico)
      "listing_type_id": "gold_special",
      "price_brl": null                // filled in Sprint 6
    }
  ]
}

Auto-grouping: greedy bucketing by (NCM, normalised xProd token-set
similarity ≥ 0.6). Ungrouped lines become singleton families.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS nf_publishing_plans (
  id              SERIAL PRIMARY KEY,
  ls_user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  draft_id        UUID NULL REFERENCES nf_publishing_drafts(id) ON DELETE SET NULL,
  nf_upload_id    INTEGER NULL,
  state_json      JSONB NOT NULL DEFAULT '{}'::jsonb,
  status          TEXT NOT NULL DEFAULT 'draft',  -- draft | published | failed
  updated_at      TIMESTAMPTZ DEFAULT NOW(),
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_nf_plans_user_status
  ON nf_publishing_plans(ls_user_id, status);

-- Attempts table — Sprint 7 publisher will fill this in. Defined here so
-- the schema migration is cohesive.
CREATE TABLE IF NOT EXISTS nf_publishing_attempts (
  id              SERIAL PRIMARY KEY,
  plan_id         INTEGER NOT NULL REFERENCES nf_publishing_plans(id) ON DELETE CASCADE,
  family_id       TEXT,
  up_id           TEXT,
  ml_item_id      TEXT,                   -- MLB... after success
  request_body    JSONB,
  response_body   JSONB,
  cause_array     JSONB,
  status          TEXT NOT NULL,          -- pending | success | warning | error
  http_status     INTEGER,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_nf_attempts_plan
  ON nf_publishing_attempts(plan_id, status);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ──────────────────────────────────────────────────────────────────────────────
# Auto-grouping (Sprint 5)
# ──────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "de", "do", "da", "dos", "das", "com", "para", "sem", "e", "a", "o",
    "cor", "tamanho", "tamanhos",
}


def _normalize_for_match(s: str) -> set[str]:
    """Tokenise an `xProd` description into a comparable set.
    Drops accents/case/punct/stopwords. Tokens with length<3 are kept only
    if they're digits (sizes like «37»)."""
    if not s:
        return set()
    nfd = unicodedata.normalize("NFKD", s)
    ascii_s = nfd.encode("ascii", "ignore").decode("ascii").lower()
    tokens = re.findall(r"[a-z0-9]+", ascii_s)
    return {
        t for t in tokens
        if t not in _STOPWORDS and (len(t) >= 3 or t.isdigit())
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = a & b
    union = a | b
    return len(inter) / max(1, len(union))


def auto_group_lines(
    lines: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.55,
) -> list[list[int]]:
    """Greedy buckets of NF line n_items, grouped by (ncm, xProd similarity).

    Returns: a list of buckets where each bucket is a list of `n_item` ids.
    Empty input → []. Each line ends up in exactly one bucket.
    """
    indexed = []
    for ln in lines:
        n = ln.get("n_item")
        if n is None:
            continue
        indexed.append({
            "n_item": int(n),
            "ncm": ln.get("ncm") or "",
            "tokens": _normalize_for_match(ln.get("description") or ""),
        })
    if not indexed:
        return []

    buckets: list[dict[str, Any]] = []
    for ln in indexed:
        placed = False
        for b in buckets:
            if b["ncm"] != ln["ncm"]:
                continue
            sim = _jaccard(b["centroid_tokens"], ln["tokens"])
            if sim >= similarity_threshold:
                b["members"].append(ln["n_item"])
                # Merge centroid by token-union (cheap; for small NFs fine).
                b["centroid_tokens"] |= ln["tokens"]
                placed = True
                break
        if not placed:
            buckets.append({
                "ncm": ln["ncm"],
                "centroid_tokens": set(ln["tokens"]),
                "members": [ln["n_item"]],
            })
    return [b["members"] for b in buckets]


# ──────────────────────────────────────────────────────────────────────────────
# UP/Item derivation
# ──────────────────────────────────────────────────────────────────────────────

def build_up_matrix(
    cores: list[str], tamanhos: list[str], *,
    family_id: str,
    available_qty_total: int = 0,
    size_grid_rows: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Cartesian product cores × tamanhos → UP rows.

    Stock spread strategy: even integer division. Remainder lands on the
    first UPs. Caller can override per-UP afterwards.
    """
    if not cores:
        cores = [""]
    if not tamanhos:
        tamanhos = [""]
    total_cells = len(cores) * len(tamanhos)
    base_qty = available_qty_total // max(1, total_cells)
    remainder = available_qty_total - base_qty * total_cells
    grid_rows = size_grid_rows or {}

    ups: list[dict[str, Any]] = []
    for ci, color in enumerate(cores):
        for si, size in enumerate(tamanhos):
            qty = base_qty + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            up_id = f"{family_id}-{ci}-{si}"
            ups.append({
                "up_id": up_id,
                "color": color,
                "size": size,
                "available_qty": int(qty),
                "size_grid_row_id": grid_rows.get(size),
            })
    return ups


# ──────────────────────────────────────────────────────────────────────────────
# Build Item bodies (Sprint 5 deliverable: preview-ready JSON)
# ──────────────────────────────────────────────────────────────────────────────

def _attr_values_to_ml(
    values: dict[str, dict[str, Any]],
    *,
    skip_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Convert wizard's `{attr_id: {value_id, value_name}}` shape into ML's
    `[{id, value_id?, value_name?}]` body. Drops empty ones."""
    skip = skip_ids or set()
    out: list[dict[str, Any]] = []
    for attr_id, av in values.items():
        if attr_id in skip:
            continue
        if not isinstance(av, dict):
            continue
        vid = av.get("value_id")
        vname = av.get("value_name")
        if not vid and not (vname and str(vname).strip()):
            continue
        item: dict[str, Any] = {"id": attr_id}
        if vid:
            item["value_id"] = vid
        if vname:
            item["value_name"] = vname
        out.append(item)
    return out


# Attributes that go into sale_terms, not into attributes[].
_SALE_TERM_IDS = {
    "WARRANTY_TYPE", "WARRANTY_TIME", "MANUFACTURING_TIME",
    "INVOICE", "PURCHASE_MAX_QUANTITY",
}

# Attributes that go on each variation/up — not the family.
_VARIATION_AXIS_IDS = {"COLOR", "SIZE"}


def build_item_bodies_for_family(
    family: dict[str, Any], *,
    site_id: str = "MLB",
) -> list[dict[str, Any]]:
    """For one family, returns one ML Item body per UP (UPTIN-shape).

    Sprint 5 produces a draft body — Sprint 6 fills in price, Sprint 7
    fills in pictures/description. Right now we expose what's known so
    the user can review the structure end-to-end.
    """
    out: list[dict[str, Any]] = []
    family_attrs = family.get("attribute_values") or {}
    family_name = family.get("family_name") or "(sem nome)"
    title_suffix_keywords = family.get("keywords") or []
    category_id = family.get("category_id")
    listing_type = family.get("listing_type_id") or "gold_special"
    grid_id = family.get("size_grid_id")

    # Per-family static attributes block (PARENT_PK + everything except
    # variation axes and sale_terms).
    parent_attrs = _attr_values_to_ml(
        family_attrs,
        skip_ids=_SALE_TERM_IDS | _VARIATION_AXIS_IDS | {"GTIN", "EMPTY_GTIN_REASON"},
    )
    sale_terms = _attr_values_to_ml(
        {k: v for k, v in family_attrs.items() if k in _SALE_TERM_IDS},
    )
    # GTIN/EMPTY_GTIN_REASON belong to attributes (per ML docs) but only
    # one of the two should be present.
    gtin = family_attrs.get("GTIN") or {}
    empty_reason = family_attrs.get("EMPTY_GTIN_REASON") or {}
    if gtin.get("value_name"):
        parent_attrs.append({"id": "GTIN", "value_name": gtin["value_name"]})
    elif empty_reason.get("value_id"):
        parent_attrs.append({
            "id": "EMPTY_GTIN_REASON",
            "value_id": empty_reason["value_id"],
            "value_name": empty_reason.get("value_name"),
        })

    if grid_id:
        parent_attrs.append({"id": "SIZE_GRID_ID", "value_id": str(grid_id)})

    photos_by_color = family.get("photos_by_color") or {}
    photos_family = family.get("photos") or []

    for up in (family.get("ups") or []):
        color = up.get("color") or ""
        size = up.get("size") or ""
        size_grid_row = up.get("size_grid_row_id")
        title_parts = [family_name]
        if color: title_parts.append(color)
        if size: title_parts.append(f"Tam {size}")
        title = " ".join(title_parts).strip()

        # Per-UP attributes (variation axes + grid_row + GTIN-per-up if needed).
        up_attrs: list[dict[str, Any]] = list(parent_attrs)
        if color:
            up_attrs.append({"id": "COLOR", "value_name": color})
        if size:
            up_attrs.append({"id": "SIZE", "value_name": size})
        if size_grid_row:
            up_attrs.append({"id": "SIZE_GRID_ROW_ID", "value_id": str(size_grid_row)})

        # Pictures: prefer color-scoped, fallback to family-wide.
        color_pics = photos_by_color.get(color) if color else None
        pic_urls = color_pics if isinstance(color_pics, list) and color_pics else photos_family
        pictures = [{"source": url} for url in pic_urls if isinstance(url, str) and url]

        body: dict[str, Any] = {
            "title": title[:60],            # ML max_title_length is 60 for most cats
            "category_id": category_id,
            "currency_id": "BRL",
            "available_quantity": int(up.get("available_qty") or 0),
            "buying_mode": "buy_it_now",
            "condition": "new",
            "listing_type_id": listing_type,
            "site_id": site_id,
            "family_name": family_name,
            "attributes": up_attrs,
        }
        # Sprint 7 — shipping. Default FBM (fulfillment); Flex uses self_service.
        ship_mode = family.get("shipping_mode") or "fbm"
        if ship_mode == "flex":
            body["shipping"] = {
                "mode": "me2",
                "local_pick_up": False,
                "free_shipping": False,
                "tags": ["self_service_in"],
            }
        elif ship_mode == "fbm":
            body["shipping"] = {
                "mode": "me2",
                "local_pick_up": False,
                "logistic_type": "fulfillment",
            }
        if pictures:
            body["pictures"] = pictures
        if sale_terms:
            body["sale_terms"] = sale_terms
        if family.get("price_brl") is not None:
            body["price"] = family["price_brl"]
        if family.get("description"):
            body["_description"] = family["description"]     # Sprint 7 ships via separate POST /items/$ID/description
        if title_suffix_keywords:
            body["_keywords_hint"] = title_suffix_keywords    # for UI only; not sent to ML

        # Trace fields for the UI preview — stripped before publish.
        body["_lsp_meta"] = {
            "family_id": family.get("family_id"),
            "up_id": up.get("up_id"),
            "source_line_ns": family.get("source_line_ns") or [],
        }
        out.append(body)
    return out


def build_plan_preview(state: dict[str, Any]) -> dict[str, Any]:
    """Flatten the whole plan into a flat list of Item bodies + summary stats."""
    site_id = (state.get("site_id") or "MLB").upper()
    families = state.get("families") or []
    items: list[dict[str, Any]] = []
    for fam in families:
        items.extend(build_item_bodies_for_family(fam, site_id=site_id))

    total_qty = sum(int(it.get("available_quantity") or 0) for it in items)
    return {
        "site_id": site_id,
        "families_count": len(families),
        "items_count": len(items),
        "total_quantity": total_qty,
        "items": items,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

async def upsert_plan(
    pool: asyncpg.Pool,
    *,
    ls_user_id: int,
    draft_id: Optional[str] = None,
    nf_upload_id: Optional[int] = None,
    state: dict[str, Any],
    plan_id: Optional[int] = None,
) -> dict[str, Any]:
    payload = json.dumps(state)
    async with pool.acquire() as conn:
        if plan_id:
            row = await conn.fetchrow(
                """
                UPDATE nf_publishing_plans SET
                    state_json = $1::jsonb, updated_at = NOW(),
                    draft_id = COALESCE($2::uuid, draft_id),
                    nf_upload_id = COALESCE($3, nf_upload_id)
                WHERE id = $4 AND ls_user_id = $5
                RETURNING id, draft_id, nf_upload_id, state_json,
                          status, created_at, updated_at
                """,
                payload, draft_id, nf_upload_id, plan_id, ls_user_id,
            )
            if not row:
                raise LookupError("plan_not_found")
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO nf_publishing_plans (
                    ls_user_id, draft_id, nf_upload_id, state_json
                )
                VALUES ($1, $2::uuid, $3, $4::jsonb)
                RETURNING id, draft_id, nf_upload_id, state_json,
                          status, created_at, updated_at
                """,
                ls_user_id, draft_id, nf_upload_id, payload,
            )
    d = dict(row)
    if d.get("id") is not None:
        d["id"] = int(d["id"])
    if d.get("draft_id") is not None:
        d["draft_id"] = str(d["draft_id"])
    return d


async def get_plan(
    pool: asyncpg.Pool, ls_user_id: int, plan_id: int,
) -> Optional[dict[str, Any]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, draft_id, nf_upload_id, state_json, status,
                   created_at, updated_at
            FROM nf_publishing_plans
            WHERE id = $1 AND ls_user_id = $2
            """,
            plan_id, ls_user_id,
        )
    if not row:
        return None
    d = dict(row)
    if d.get("draft_id") is not None:
        d["draft_id"] = str(d["draft_id"])
    return d
