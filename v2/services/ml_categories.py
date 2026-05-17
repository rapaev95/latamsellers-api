"""ML categories + domain_discovery cache.

Two read paths:
- `predict_for_query(http, token, site_id, q)` → ML `/sites/{site}/domain_discovery/search`
  with caching keyed by (site_id, normalized_query). TTL 30 days because
  predictor results don't fluctuate often.
- `get_category(pool, http, token, category_id)` → ML `/categories/{id}` with
  caching. TTL 7 days; category tree changes rarely.

Tables live alongside other ml_* caches; created lazily on first call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"

# TTLs
PREDICT_TTL = timedelta(days=30)
CATEGORY_TTL = timedelta(days=7)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_categories (
  category_id     TEXT PRIMARY KEY,
  name            TEXT,
  domain_id       TEXT,
  path_from_root  JSONB,
  settings_json   JSONB,
  raw_json        JSONB NOT NULL,
  fetched_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ml_categories_domain ON ml_categories(domain_id);

CREATE TABLE IF NOT EXISTS ml_domain_discovery_cache (
  id            SERIAL PRIMARY KEY,
  site_id       TEXT NOT NULL,
  q_norm        TEXT NOT NULL,
  q_hash        TEXT NOT NULL,
  predictions   JSONB NOT NULL,
  fetched_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(site_id, q_hash)
);
CREATE INDEX IF NOT EXISTS idx_ml_domain_disc_site ON ml_domain_discovery_cache(site_id, q_hash);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_query(q: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Keeps the cache hit-rate
    high when same product comes through with slight typographic variations."""
    s = unicodedata.normalize("NFKD", q).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _q_hash(site_id: str, q_norm: str) -> str:
    return hashlib.sha1(f"{site_id}\x1f{q_norm}".encode("utf-8")).hexdigest()


def _is_stale(fetched_at: Optional[datetime], ttl: timedelta) -> bool:
    if fetched_at is None:
        return True
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_at > ttl


# ──────────────────────────────────────────────────────────────────────────────
# Predictor (domain_discovery)
# ──────────────────────────────────────────────────────────────────────────────

async def _fetch_predictions(
    http: httpx.AsyncClient, access_token: str, *, site_id: str, q: str, limit: int = 3,
) -> list[dict[str, Any]]:
    r = await http.get(
        f"{ML_API_BASE}/sites/{site_id}/domain_discovery/search",
        params={"q": q, "limit": limit},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    if r.status_code != 200:
        # Bubble up just the status; caller decides whether to fall back to cache.
        raise httpx.HTTPStatusError("ml_domain_discovery_failed", request=r.request, response=r)
    payload = r.json()
    return payload if isinstance(payload, list) else []


async def predict_for_query(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    *,
    site_id: str,
    q: str,
    limit: int = 3,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {predictions: [...], cache_hit: bool, fetched_at: iso}.

    Each prediction has the ML shape {domain_id, domain_name, category_id,
    category_name, attributes:[...]}.
    """
    site_id = site_id.upper()
    q_norm = _normalize_query(q)
    qh = _q_hash(site_id, q_norm)

    if not q_norm:
        return {"predictions": [], "cache_hit": False, "fetched_at": None}

    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT predictions, fetched_at FROM ml_domain_discovery_cache
                WHERE site_id = $1 AND q_hash = $2
                """,
                site_id, qh,
            )
        if row and not _is_stale(row["fetched_at"], PREDICT_TTL):
            preds = row["predictions"]
            if isinstance(preds, str):
                preds = json.loads(preds)
            return {
                "predictions": preds,
                "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    predictions = await _fetch_predictions(http, access_token, site_id=site_id, q=q, limit=limit)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_domain_discovery_cache (site_id, q_norm, q_hash, predictions, fetched_at)
            VALUES ($1, $2, $3, $4::jsonb, NOW())
            ON CONFLICT (site_id, q_hash) DO UPDATE SET
                predictions = EXCLUDED.predictions,
                q_norm = EXCLUDED.q_norm,
                fetched_at = NOW()
            """,
            site_id, q_norm, qh, json.dumps(predictions),
        )
    return {
        "predictions": predictions,
        "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single category lookup (/categories/{id})
# ──────────────────────────────────────────────────────────────────────────────

async def _fetch_category(
    http: httpx.AsyncClient, access_token: str, category_id: str,
) -> dict[str, Any]:
    r = await http.get(
        f"{ML_API_BASE}/categories/{category_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15.0,
    )
    if r.status_code != 200:
        raise httpx.HTTPStatusError("ml_category_failed", request=r.request, response=r)
    payload = r.json()
    if not isinstance(payload, dict):
        raise ValueError("ml_category_unexpected_shape")
    return payload


async def get_category(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    access_token: str,
    category_id: str,
    *,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    """Returns {category_id, name, domain_id, path_from_root, settings, cache_hit, fetched_at}."""
    if not bypass_cache:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT category_id, name, domain_id, path_from_root,
                       settings_json, raw_json, fetched_at
                FROM ml_categories WHERE category_id = $1
                """,
                category_id,
            )
        if row and not _is_stale(row["fetched_at"], CATEGORY_TTL):
            return {
                "category_id": row["category_id"],
                "name": row["name"],
                "domain_id": row["domain_id"],
                "path_from_root": row["path_from_root"],
                "settings": row["settings_json"],
                "raw": row["raw_json"],
                "cache_hit": True,
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
            }

    payload = await _fetch_category(http, access_token, category_id)
    name = payload.get("name")
    settings = payload.get("settings") or {}
    domain_id = (settings.get("catalog_domain") if isinstance(settings, dict) else None) or None
    path = payload.get("path_from_root") or []
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_categories (
                category_id, name, domain_id, path_from_root, settings_json, raw_json, fetched_at
            )
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, NOW())
            ON CONFLICT (category_id) DO UPDATE SET
                name = EXCLUDED.name,
                domain_id = EXCLUDED.domain_id,
                path_from_root = EXCLUDED.path_from_root,
                settings_json = EXCLUDED.settings_json,
                raw_json = EXCLUDED.raw_json,
                fetched_at = NOW()
            """,
            category_id, name, domain_id, json.dumps(path),
            json.dumps(settings), json.dumps(payload),
        )
    return {
        "category_id": category_id,
        "name": name,
        "domain_id": domain_id,
        "path_from_root": path,
        "settings": settings,
        "raw": payload,
        "cache_hit": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
