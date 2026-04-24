"""Listing quality cache — via ML /item/{id}/performance.

Why a cache: quality rarely changes (only after manual edit or moderation),
so TTL=24h is plenty. Fetching 100+ items on every products page load would
hammer ML's rate limits for nothing.

Shape of /item/{id}/performance response (as of 2026-02, post /health migration):
  {
    entity_type: "ITEM" | "USER_PRODUCT",
    entity_id: str,
    score: float (0..100),
    level: "Basic" | "Standard" | "Professional",
    level_wording: str (localized),
    status: "PENDING" | "COMPLETED",
    calculated_at: ISO,
    buckets: [
      {
        key: "CHARACTERISTICS" | "PICTURES" | "DESCRIPTION" | "VIDEO" | "TECHNICAL_SPECIFICATION",
        score, title, status, calculated_at,
        variables: [
          { key, status, score, mode: "OPPORTUNITY" | "WARNING", title, description }
        ]
      }
    ]
  }

We flatten `buckets[].variables[]` into two lists on our side: WARNINGs
(blocking — score-reducers) and OPPORTUNITYs (nice-to-have improvements).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"

# Throttle: ML rate limits are ~10k/hr app-wide. 5 req/sec = 18k/hr, but we
# run many users/modules in parallel — 5 rps per user is a safe single-user cap.
RATE_SLEEP = 0.2  # 200ms = 5 req/sec

# Never block the /products endpoint on an enormous catalog — batching caller
# can always call refresh again for the rest.
DEFAULT_REFRESH_LIMIT = 500


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_item_quality (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  score NUMERIC,
  level TEXT,
  status TEXT,
  calculated_at TIMESTAMPTZ,
  warnings JSONB DEFAULT '[]'::jsonb,
  opportunities JSONB DEFAULT '[]'::jsonb,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_quality_user ON ml_item_quality(user_id);
"""

# Legacy cleanup: before normalize_item_id() stripped '.0' suffixes, we stored
# rows with item_id like 'MLB123.0' that never matched anything on the lookup
# side. Delete them so refresh can repopulate with clean keys.
CLEANUP_LEGACY_SQL = "DELETE FROM ml_item_quality WHERE item_id LIKE '%.%'"


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)
        # Idempotent — becomes a no-op once cleaned. Safe to run every start.
        try:
            await conn.execute(CLEANUP_LEGACY_SQL)
        except Exception as err:  # noqa: BLE001
            log.warning("ml_item_quality legacy cleanup failed: %s", err)


# ── ML API ────────────────────────────────────────────────────────────────────

def normalize_item_id(item_id: str) -> str:
    """Normalize ML item id to 'MLB<digits>' form.

    Handles three real-world messy inputs:
    - 'MLB1234' → 'MLB1234' (already canonical)
    - '1234' → 'MLB1234' (pure numeric)
    - '1234.0' → 'MLB1234' (CSV-parsed float artifact from pandas)
    - 'MLB1234.0' → 'MLB1234' (same, with prefix)

    ML's /item/{id}/performance rejects any trailing '.0' — so we strip the
    decimal part defensively. Returns empty string for unparseable input.
    """
    s = str(item_id or "").strip().upper()
    if not s:
        return ""
    numeric = s[3:] if s.startswith("MLB") else s
    # Strip float formatting: '1234.0' → '1234'. Only if the part after '.' is
    # all zeros (safe — real MLB ids don't contain '.').
    if "." in numeric:
        head, _, tail = numeric.partition(".")
        if tail.strip("0") == "":
            numeric = head
        else:
            # Unexpected non-zero fraction — treat as invalid.
            return ""
    if not numeric.isdigit():
        return ""
    return f"MLB{numeric}"


# Backwards-compat alias in case anything still calls the private name.
_normalize_item_id = normalize_item_id


def _flatten_buckets(buckets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rules across all buckets → variables → rules into (warnings, opportunities).

    ML structure (verified via /quality-probe):
      buckets[] → variables[] → rules[{mode, status, progress, wordings{title,label,link}}]
    The `mode` lives on `rules[]` — NOT on the variable itself.
    """
    warnings: list[dict] = []
    opportunities: list[dict] = []
    for b in buckets or []:
        bucket_key = b.get("key") or ""
        variables = b.get("variables") or b.get("actions") or []
        for v in variables:
            if not isinstance(v, dict):
                continue
            var_title = v.get("title")
            rules = v.get("rules") or []
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                mode = (rule.get("mode") or "").upper()
                wordings = rule.get("wordings") or {}
                entry = {
                    "bucket": bucket_key,
                    "key": rule.get("key") or v.get("key"),
                    "title": wordings.get("title") or var_title,
                    "description": wordings.get("label"),
                    "link": wordings.get("link"),
                    "status": rule.get("status"),
                }
                if mode == "WARNING":
                    warnings.append(entry)
                elif mode == "OPPORTUNITY":
                    opportunities.append(entry)
    return warnings, opportunities


async def fetch_one(
    http: httpx.AsyncClient,
    access_token: str,
    item_id: str,
) -> Optional[dict]:
    """One call to /item/{id}/performance. Returns normalized dict or None on failure.

    None means "skip upsert for this item" — 404, auth issue, transport error.
    We deliberately do not distinguish here to keep the caller simple; the
    bulk worker counts failures.
    """
    mlb = normalize_item_id(item_id)
    if not mlb:
        log.debug("skip invalid item_id: %r", item_id)
        return None
    try:
        r = await http.get(
            f"{ML_API_BASE}/item/{mlb}/performance",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20.0,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("performance GET %s failed: %s", mlb, err)
        return None
    if r.status_code == 404:
        # Some items (e.g. draft or unsupported categories) have no performance.
        # Still write a row with null score so UI can show "—".
        return {
            "item_id": mlb,
            "score": None,
            "level": None,
            "status": None,
            "calculated_at": None,
            "warnings": [],
            "opportunities": [],
        }
    if r.status_code != 200:
        log.warning("performance GET %s → %s: %s", mlb, r.status_code, r.text[:200])
        return None
    try:
        data = r.json() or {}
    except Exception:  # noqa: BLE001
        return None

    warnings, opportunities = _flatten_buckets(data.get("buckets") or [])
    # Prefer localized level_wording ('Profissional', 'Normal', 'Básico', ...) over
    # the internal level code ('good'/'regular'/'bad') — it's what the user sees
    # in ML Seller Center and is fed straight to our UI badge.
    level_display = data.get("level_wording") or data.get("level")
    return {
        "item_id": mlb,
        "score": data.get("score"),
        "level": level_display,
        "status": data.get("status"),
        "calculated_at": data.get("calculated_at"),
        "warnings": warnings,
        "opportunities": opportunities,
    }


# ── Bulk refresh ──────────────────────────────────────────────────────────────

async def _upsert_quality(conn: asyncpg.Connection, user_id: int, row: dict) -> None:
    await conn.execute(
        """
        INSERT INTO ml_item_quality
          (user_id, item_id, score, level, status, calculated_at,
           warnings, opportunities, fetched_at)
        VALUES ($1, $2, $3, $4, $5, $6::timestamptz, $7::jsonb, $8::jsonb, NOW())
        ON CONFLICT (user_id, item_id) DO UPDATE SET
          score = EXCLUDED.score,
          level = EXCLUDED.level,
          status = EXCLUDED.status,
          calculated_at = EXCLUDED.calculated_at,
          warnings = EXCLUDED.warnings,
          opportunities = EXCLUDED.opportunities,
          fetched_at = NOW()
        """,
        user_id,
        row["item_id"],
        row.get("score"),
        row.get("level"),
        row.get("status"),
        row.get("calculated_at"),
        json.dumps(row.get("warnings") or []),
        json.dumps(row.get("opportunities") or []),
    )


async def refresh_user_quality(
    pool: asyncpg.Pool,
    user_id: int,
    item_ids: list[str],
    limit: int = DEFAULT_REFRESH_LIMIT,
) -> dict[str, int]:
    """Fetch /performance for up to `limit` items, upsert into ml_item_quality.

    Throttles to ~5 req/sec per the module constant. Callers should pass unique
    itemIds already — we do a dedup + truncate here defensively.
    """
    # Normalize + dedup upfront so callers don't need to. Invalid ids are dropped.
    seen: set[str] = set()
    unique_ids: list[str] = []
    dropped_invalid = 0
    for raw in item_ids:
        if not raw:
            continue
        mlb = normalize_item_id(raw)
        if not mlb:
            dropped_invalid += 1
            continue
        if mlb in seen:
            continue
        seen.add(mlb)
        unique_ids.append(mlb)
    unique_ids = unique_ids[:limit]
    if not unique_ids:
        return {"fetched": 0, "saved": 0, "failed": 0, "skipped": dropped_invalid}

    try:
        access_token, _expires_at, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("refresh_user_quality: token refresh failed for user %s: %s", user_id, err)
        return {"fetched": 0, "saved": 0, "failed": 0, "skipped": len(unique_ids)}

    fetched = 0
    saved = 0
    failed = 0

    async with httpx.AsyncClient() as http:
        for iid in unique_ids:
            row = await fetch_one(http, access_token, iid)
            fetched += 1
            if row is None:
                failed += 1
            else:
                try:
                    async with pool.acquire() as conn:
                        await _upsert_quality(conn, user_id, row)
                    saved += 1
                except Exception as err:  # noqa: BLE001
                    log.exception("upsert quality %s failed: %s", iid, err)
                    failed += 1
            await asyncio.sleep(RATE_SLEEP)

    return {"fetched": fetched, "saved": saved, "failed": failed, "skipped": dropped_invalid}


# ── Cache readback (for /products join) ───────────────────────────────────────

async def get_cached_map(
    conn: asyncpg.Connection,
    user_id: int,
) -> dict[str, dict[str, Any]]:
    """Return dict keyed by `item_id` (normalized upper-case MLB) with quality rows.

    Keys match `_normalize_item_id(product.itemId)` so caller can do a simple
    dict lookup when joining.
    """
    rows = await conn.fetch(
        """
        SELECT item_id, score, level, status,
               to_char(calculated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS calculated_at,
               to_char(fetched_at    AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at,
               warnings, opportunities
          FROM ml_item_quality
         WHERE user_id = $1
        """,
        user_id,
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        warnings = r["warnings"] if isinstance(r["warnings"], list) else json.loads(r["warnings"] or "[]")
        opportunities = r["opportunities"] if isinstance(r["opportunities"], list) else json.loads(r["opportunities"] or "[]")
        out[r["item_id"].upper()] = {
            "score": float(r["score"]) if r["score"] is not None else None,
            "level": r["level"],
            "status": r["status"],
            "calculatedAt": r["calculated_at"],
            "fetchedAt": r["fetched_at"],
            "warnings": warnings,
            "opportunities": opportunities,
            "warningsCount": len(warnings),
            "opportunitiesCount": len(opportunities),
        }
    return out


async def get_latest_fetched_at(conn: asyncpg.Connection, user_id: int) -> Optional[str]:
    """Returns ISO timestamp of the most recently fetched row, or None if cache empty."""
    row = await conn.fetchrow(
        """
        SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS latest
          FROM ml_item_quality
         WHERE user_id = $1
        """,
        user_id,
    )
    return row["latest"] if row else None
