"""ML grid charts (tabela de medidas) — search + cache for fashion publishing.

ML endpoints used:
- `GET /catalog/charts/{SITE_ID}/configurations/active_domains` — which domains
  on this site support tabela de medidas at all.
- `POST /catalog/charts/search?offset=0&limit=100` — list BRAND / STANDARD /
  SPECIFIC charts for a domain, optionally narrowed by known_attributes.
- `GET /catalog/charts/{chart_id}` — full chart details with rows.

Three chart types (from РАБОТА С КАРТОЧКАМИ ОДЕЖДЫ И ОБУВИ ML.md):
- BRAND: pre-loaded by ML for brands like Nike/Adidas (preferred when match).
- STANDARD: ML default per domain (fallback when BRAND not found).
- SPECIFIC: created by sellers themselves (V2 — out of Sprint 4 scope).

We cache active_domains (7d) and chart details (24h). Charts themselves
are cached as a single «search snapshot» keyed by (site, domain, type,
known_attrs_hash) — TTL 24h, ML rarely updates these.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"

ACTIVE_DOMAINS_TTL = timedelta(days=7)
CHARTS_SEARCH_TTL = timedelta(hours=24)
CHART_DETAILS_TTL = timedelta(hours=24)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_grid_active_domains (
  site_id        TEXT PRIMARY KEY,
  domains        JSONB NOT NULL,
  fetched_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_grid_charts_search (
  id             SERIAL PRIMARY KEY,
  site_id        TEXT NOT NULL,
  domain_id      TEXT NOT NULL,
  chart_type     TEXT,                    -- 'BRAND' | 'STANDARD' | 'SPECIFIC' | NULL = all
  known_attrs_hash TEXT NOT NULL,
  charts         JSONB NOT NULL,          -- ML response body
  fetched_at     TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(site_id, domain_id, chart_type, known_attrs_hash)
);
CREATE INDEX IF NOT EXISTS idx_grid_charts_search_lookup
  ON ml_grid_charts_search(site_id, domain_id, chart_type);

CREATE TABLE IF NOT EXISTS ml_grid_chart_details (
  chart_id       TEXT PRIMARY KEY,
  body           JSONB NOT NULL,
  fetched_at     TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_stale(fetched_at: Optional[datetime], ttl: timedelta) -> bool:
    if fetched_at is None:
        return True
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_at > ttl


def _hash_known_attrs(known: Optional[list[dict[str, Any]]]) -> str:
    """Stable hash of `known_attributes` array for cache keying. Order-
    insensitive, only id / value_id / value_name matter."""
    if not known:
        return "empty"
    norm = sorted(
        ({"id": k.get("id"), "value_id": k.get("value_id"),
          "value_name": k.get("value_name")} for k in known),
        key=lambda d: (d.get("id") or "", d.get("value_id") or "", d.get("value_name") or ""),
    )
    return hashlib.sha1(json.dumps(norm).encode("utf-8")).hexdigest()


def _strip_site_prefix(domain_id: str) -> str:
    """ML expects `SNEAKERS` in `/catalog/charts/search`, not `MLB-SNEAKERS`
    (confirmed in `РАБОТА С КАРТОЧКАМИ ОДЕЖДЫ И ОБУВИ ML.md:899`)."""
    if not domain_id:
        return domain_id
    for prefix in ("MLB-", "MLA-", "MLM-", "MLC-", "MCO-", "MPE-", "MLU-"):
        if domain_id.startswith(prefix):
            return domain_id[len(prefix):]
    return domain_id


# ──────────────────────────────────────────────────────────────────────────────
# 1. Active domains per site
# ──────────────────────────────────────────────────────────────────────────────

async def get_active_domains(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    site_id: str,
    *,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {site_id, domains: [{domain_id}, ...], cache_hit, fetched_at}."""
    site_id = site_id.upper()

    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT domains, fetched_at FROM ml_grid_active_domains WHERE site_id = $1",
                site_id,
            )
        if row and not _is_stale(row["fetched_at"], ACTIVE_DOMAINS_TTL):
            domains = row["domains"]
            if isinstance(domains, str):
                domains = json.loads(domains)
            return {
                "site_id": site_id,
                "domains": domains,
                "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    r = await http.get(
        f"{ML_API_BASE}/catalog/charts/{site_id}/configurations/active_domains",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_active_domains_failed", request=r.request, response=r,
        )
    payload = r.json() or {}
    # ML returns either {"domains":[...]} or just a list — normalise.
    if isinstance(payload, list):
        domains = payload
    else:
        domains = payload.get("domains") or []

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_grid_active_domains (site_id, domains, fetched_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (site_id) DO UPDATE SET
                domains = EXCLUDED.domains,
                fetched_at = NOW()
            """,
            site_id, json.dumps(domains),
        )
    return {
        "site_id": site_id,
        "domains": domains,
        "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def domain_supports_grids(
    pool: asyncpg.Pool, http: httpx.AsyncClient, access_token: str,
    *, site_id: str, domain_id: str,
) -> bool:
    """Cheap «do we even need to show a grid picker» check."""
    out = await get_active_domains(pool, http, access_token, site_id)
    targets = {_strip_site_prefix(domain_id), domain_id}
    for d in out["domains"]:
        if not isinstance(d, dict):
            continue
        did = d.get("domain_id") or d.get("id")
        if did and did in targets:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# 2. Search charts
# ──────────────────────────────────────────────────────────────────────────────

async def search_charts(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    *,
    site_id: str,
    domain_id: str,
    chart_type: Optional[str] = None,   # 'BRAND' | 'STANDARD' | 'SPECIFIC' | None
    known_attributes: Optional[list[dict[str, Any]]] = None,
    seller_id: Optional[int] = None,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {site_id, domain_id, chart_type, charts: [...], cache_hit,
    fetched_at}."""
    site_id = site_id.upper()
    domain_norm = _strip_site_prefix(domain_id)
    known_hash = _hash_known_attrs(known_attributes)
    ct = chart_type if chart_type in ("BRAND", "STANDARD", "SPECIFIC") else None

    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT charts, fetched_at FROM ml_grid_charts_search
                WHERE site_id = $1 AND domain_id = $2 AND chart_type IS NOT DISTINCT FROM $3
                  AND known_attrs_hash = $4
                """,
                site_id, domain_norm, ct, known_hash,
            )
        if row and not _is_stale(row["fetched_at"], CHARTS_SEARCH_TTL):
            ch = row["charts"]
            if isinstance(ch, str):
                ch = json.loads(ch)
            return {
                "site_id": site_id, "domain_id": domain_norm, "chart_type": ct,
                "charts": ch, "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    body: dict[str, Any] = {"domain_id": domain_norm, "site_id": site_id}
    if ct:
        body["type"] = ct
    if known_attributes:
        body["known_attributes"] = known_attributes
    if seller_id is not None:
        body["seller_id"] = seller_id

    r = await http.post(
        f"{ML_API_BASE}/catalog/charts/search?offset=0&limit=100",
        json=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=20.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_charts_search_failed", request=r.request, response=r,
        )
    payload = r.json() or {}
    charts = payload.get("charts") if isinstance(payload, dict) else payload

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_grid_charts_search (
                site_id, domain_id, chart_type, known_attrs_hash, charts, fetched_at
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
            ON CONFLICT (site_id, domain_id, chart_type, known_attrs_hash) DO UPDATE SET
                charts = EXCLUDED.charts,
                fetched_at = NOW()
            """,
            site_id, domain_norm, ct, known_hash, json.dumps(charts or []),
        )
    return {
        "site_id": site_id, "domain_id": domain_norm, "chart_type": ct,
        "charts": charts or [], "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Chart details
# ──────────────────────────────────────────────────────────────────────────────

async def get_chart_details(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    chart_id: str,
    *,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {chart_id, body, cache_hit, fetched_at}."""
    cid = str(chart_id)
    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT body, fetched_at FROM ml_grid_chart_details WHERE chart_id = $1",
                cid,
            )
        if row and not _is_stale(row["fetched_at"], CHART_DETAILS_TTL):
            body = row["body"]
            if isinstance(body, str):
                body = json.loads(body)
            return {
                "chart_id": cid, "body": body, "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    r = await http.get(
        f"{ML_API_BASE}/catalog/charts/{cid}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_chart_details_failed", request=r.request, response=r,
        )
    body = r.json() or {}
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_grid_chart_details (chart_id, body, fetched_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (chart_id) DO UPDATE SET
                body = EXCLUDED.body,
                fetched_at = NOW()
            """,
            cid, json.dumps(body),
        )
    return {
        "chart_id": cid, "body": body, "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. SPECIFIC chart creation — Sprint 4 bonus follow-up
# ──────────────────────────────────────────────────────────────────────────────


async def fetch_chart_spec(
    http: httpx.AsyncClient, access_token: str,
    *,
    domain_id: str,
    site_id: str,
    known_attributes: Optional[list[dict[str, Any]]] = None,
    seller_id: Optional[int] = None,
) -> dict[str, Any]:
    """POST /domains/{domain_id}/technical_specs?section=grids — returns the
    «ficha técnica» for creating a SPECIFIC chart in this domain narrowed by
    known_attributes (typically BRAND + GENDER).

    Response contains the attribute schema (main_attributes, attributes, rows
    structure) that the editor must render.
    """
    domain_norm = _strip_site_prefix(domain_id)
    body: dict[str, Any] = {
        "site_id": site_id.upper(),
        "domain_id": domain_norm,
    }
    if known_attributes:
        body["known_attributes"] = known_attributes
    if seller_id is not None:
        body["seller_id"] = seller_id

    r = await http.post(
        f"{ML_API_BASE}/domains/{domain_id}/technical_specs?section=grids",
        json=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=20.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError(
            "ml_chart_spec_failed", request=r.request, response=r,
        )
    return r.json() or {}


async def create_chart(
    http: httpx.AsyncClient, access_token: str,
    *, body: dict[str, Any],
) -> dict[str, Any]:
    """POST /catalog/charts — create a new SPECIFIC chart.

    Caller provides the fully-formed body matching the schema returned by
    `fetch_chart_spec`. Returns ML's response, typically `{id, ...}`.
    """
    r = await http.post(
        f"{ML_API_BASE}/catalog/charts",
        json=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    try:
        parsed = r.json()
    except Exception:  # noqa: BLE001
        parsed = {"raw": r.text[:1000]}
    return {
        "http_status": r.status_code,
        "body": parsed,
        "id": parsed.get("id") if isinstance(parsed, dict) and r.status_code < 400 else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. find_chart_for_listing — Sprint 4 deliverable
# ──────────────────────────────────────────────────────────────────────────────

async def find_chart_for_listing(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    *,
    site_id: str,
    domain_id: str,
    known_attributes: Optional[list[dict[str, Any]]] = None,
    seller_id: Optional[int] = None,
) -> dict[str, Any]:
    """Try BRAND first → STANDARD second → null. Returns:
       { suggestion: { chart, type } | null,
         brand: { charts: [...], cache_hit, fetched_at },
         standard: { charts: [...], cache_hit, fetched_at } }.

    The UI uses this to render a preselected suggestion + the alternative
    list. If both buckets are empty the UI shows a «no chart available»
    banner — Sprint 4 doesn't build SPECIFIC editor yet (V2).
    """
    # BRAND first
    brand = await search_charts(
        pool, http, access_token,
        site_id=site_id, domain_id=domain_id, chart_type="BRAND",
        known_attributes=known_attributes, seller_id=seller_id,
    )
    standard = await search_charts(
        pool, http, access_token,
        site_id=site_id, domain_id=domain_id, chart_type="STANDARD",
        known_attributes=known_attributes, seller_id=seller_id,
    )

    suggestion: Optional[dict[str, Any]] = None
    brand_list = brand.get("charts") or []
    standard_list = standard.get("charts") or []

    if brand_list:
        suggestion = {"chart": brand_list[0], "type": "BRAND"}
    elif standard_list:
        suggestion = {"chart": standard_list[0], "type": "STANDARD"}

    return {
        "suggestion": suggestion,
        "brand": brand,
        "standard": standard,
    }
