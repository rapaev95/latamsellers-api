"""Item visits cache — via ML `/items/{id}/visits/time_window`.

Why a cache: visits change daily but we don't need live data for the
Products table. Fetching visits for 100+ items on every page load would
burn ML's rate limits for no reason. Mirrors the `ml_quality.py` pattern.

Shape of /items/{id}/visits/time_window response (last=30&unit=day):
  {
    item_id: "MLB123",
    date_from: "2026-03-25T00:00:00Z",
    date_to: "2026-04-24T00:00:00Z",
    total_visits: 842,
    last: 30,
    unit: "day",
    results: [
      {"date": "2026-03-25T00:00:00Z", "total": 24, "visits_detail": [...]},
      ...
    ]
  }

Docs: https://developers.mercadolivre.com.br/pt_br/publicacao-de-produtos/recurso-visits
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc
from .ml_quality import normalize_item_id  # reuse — same MLB-id cleanup rules

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"

# Same 5 rps per-user cap as ml_quality — ML rate limits are app-wide.
RATE_SLEEP = 0.2

DEFAULT_REFRESH_LIMIT = 500

# Window: 30 days of daily buckets. visits_7d is derived client-side on the
# server by summing the last 7 entries; visits_30d = total_visits.
WINDOW_DAYS = 30


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_item_visits (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  visits_7d INTEGER,
  visits_30d INTEGER,
  daily JSONB DEFAULT '[]'::jsonb,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_visits_user ON ml_item_visits(user_id);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── ML API ────────────────────────────────────────────────────────────────────

def _parse_daily(results: list) -> tuple[int, list[dict]]:
    """Extract (visits_7d, normalized_daily_list) from ML `results` array.

    Normalized entry: {"date": "YYYY-MM-DD", "total": int}. Drops visits_detail
    (noisy, not used in UI sparkline). visits_7d = sum of last 7 buckets by date.
    """
    if not results:
        return 0, []
    buckets: list[dict] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        raw_date = r.get("date") or ""
        # ML returns "YYYY-MM-DDTHH:MM:SSZ" — keep just the date part.
        date = str(raw_date).split("T", 1)[0] if raw_date else ""
        total = r.get("total")
        try:
            total_int = int(total or 0)
        except (TypeError, ValueError):
            total_int = 0
        if date:
            buckets.append({"date": date, "total": total_int})
    # Sort chronologically just in case ML ever returns unordered.
    buckets.sort(key=lambda b: b["date"])
    # Last 7 days by position (already chronological).
    last7 = buckets[-7:] if len(buckets) > 7 else buckets
    visits_7d = sum(b["total"] for b in last7)
    return visits_7d, buckets


async def fetch_one(
    http: httpx.AsyncClient,
    access_token: str,
    item_id: str,
) -> tuple[Optional[dict], Optional[dict]]:
    """One call to /items/{id}/visits/time_window?last=30&unit=day.

    Returns (row, error_info) — same contract as ml_quality.fetch_one.
    """
    mlb = normalize_item_id(item_id)
    if not mlb:
        return None, {"status": 0, "body": f"invalid_item_id: {item_id!r}"}
    url = f"{ML_API_BASE}/items/{mlb}/visits/time_window?last={WINDOW_DAYS}&unit=day"
    try:
        r = await http.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20.0,
        )
    except Exception as err:  # noqa: BLE001
        return None, {"status": 0, "body": f"network: {err}"}

    # 400/404: item never had visits data (private/deactivated). Persist a null
    # row so UI shows "—" and we don't retry on every refresh.
    if r.status_code in (400, 404):
        return {
            "item_id": mlb,
            "visits_7d": 0,
            "visits_30d": 0,
            "daily": [],
        }, None
    if r.status_code != 200:
        return None, {"status": r.status_code, "body": r.text[:300]}
    try:
        data = r.json() or {}
    except Exception as err:  # noqa: BLE001
        return None, {"status": 200, "body": f"json_parse: {err}"}

    visits_7d, daily = _parse_daily(data.get("results") or [])
    # ML's total_visits is the authoritative 30d number; fall back to sum if
    # field is missing/invalid.
    total_raw = data.get("total_visits")
    try:
        visits_30d = int(total_raw) if total_raw is not None else sum(b["total"] for b in daily)
    except (TypeError, ValueError):
        visits_30d = sum(b["total"] for b in daily)

    return {
        "item_id": mlb,
        "visits_7d": visits_7d,
        "visits_30d": visits_30d,
        "daily": daily,
    }, None


# ── Bulk refresh ──────────────────────────────────────────────────────────────

async def _upsert_visits(conn: asyncpg.Connection, user_id: int, row: dict) -> None:
    await conn.execute(
        """
        INSERT INTO ml_item_visits
          (user_id, item_id, visits_7d, visits_30d, daily, fetched_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
        ON CONFLICT (user_id, item_id) DO UPDATE SET
          visits_7d  = EXCLUDED.visits_7d,
          visits_30d = EXCLUDED.visits_30d,
          daily      = EXCLUDED.daily,
          fetched_at = NOW()
        """,
        user_id,
        row["item_id"],
        int(row.get("visits_7d") or 0),
        int(row.get("visits_30d") or 0),
        json.dumps(row.get("daily") or [], default=str),
    )


async def refresh_user_visits(
    pool: asyncpg.Pool,
    user_id: int,
    item_ids: list[str],
    limit: int = DEFAULT_REFRESH_LIMIT,
) -> dict[str, int]:
    """Fetch visits for up to `limit` items, upsert into ml_item_visits.

    Throttled to ~5 req/sec. Dedup + normalize item_ids defensively.
    """
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
        access_token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("refresh_user_visits: token refresh failed for user %s: %s", user_id, err)
        return {"fetched": 0, "saved": 0, "failed": 0, "skipped": len(unique_ids)}

    fetched = 0
    saved = 0
    failed = 0
    status_counts: dict[str, int] = {}
    sample_errors: list[dict] = []

    async with httpx.AsyncClient() as http:
        for iid in unique_ids:
            row, err = await fetch_one(http, access_token, iid)
            fetched += 1
            if row is None:
                failed += 1
                key = str(err.get("status") if err else "unknown")
                status_counts[key] = status_counts.get(key, 0) + 1
                if err and len(sample_errors) < 3:
                    sample_errors.append({"item_id": iid, **err})
            else:
                try:
                    async with pool.acquire() as conn:
                        await _upsert_visits(conn, user_id, row)
                    saved += 1
                except Exception as upsert_err:  # noqa: BLE001
                    log.exception("upsert visits %s failed: %s", iid, upsert_err)
                    failed += 1
                    key = "upsert_error"
                    status_counts[key] = status_counts.get(key, 0) + 1
                    if len(sample_errors) < 3:
                        sample_errors.append({"item_id": iid, "body": str(upsert_err)})
            await asyncio.sleep(RATE_SLEEP)

    return {
        "fetched": fetched,
        "saved": saved,
        "failed": failed,
        "skipped": dropped_invalid,
        "status_counts": status_counts,
        "sample_errors": sample_errors,
    }


# ── Cache readback (for /products join) ───────────────────────────────────────

async def get_cached_map(
    conn: asyncpg.Connection,
    user_id: int,
) -> dict[str, dict[str, Any]]:
    """Return dict keyed by normalized item_id (upper-case MLB)."""
    rows = await conn.fetch(
        """
        SELECT item_id, visits_7d, visits_30d, daily,
               to_char(fetched_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
          FROM ml_item_visits
         WHERE user_id = $1
        """,
        user_id,
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        daily = r["daily"] if isinstance(r["daily"], list) else json.loads(r["daily"] or "[]")
        out[r["item_id"].upper()] = {
            "visits7d": int(r["visits_7d"] or 0),
            "visits30d": int(r["visits_30d"] or 0),
            "daily": daily,
            "fetchedAt": r["fetched_at"],
        }
    return out


async def get_latest_fetched_at(conn: asyncpg.Connection, user_id: int) -> Optional[str]:
    row = await conn.fetchrow(
        """
        SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS latest
          FROM ml_item_visits
         WHERE user_id = $1
        """,
        user_id,
    )
    return row["latest"] if row else None
