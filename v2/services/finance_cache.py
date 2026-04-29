"""Durable read-through cache for compute-heavy finance/escalar endpoints.

Why a durable cache: in-memory dicts in `legacy/reports.py` (_VENDAS_DF_CACHE,
_MATRIX_CACHE) don't survive uvicorn restarts and aren't shared between
workers, so PnL/Cashflow/Balance/ABC get recomputed cold on every page load.
Users wait 5-30s on every refresh.

How it works:
  1. `compute_fingerprint(user_id)` — SHA256 of MAX(uploads.created_at) per
     `source_key` + updated_at per affected `user_data` key. Any user input
     change → new fingerprint.
  2. `cached_compute(user_id, cache_key, compute_fn, force=False)` — read-through:
     SELECT row by (user_id, cache_key); if fingerprint matches → return cached
     payload; otherwise call `compute_fn()` and UPSERT result.
  3. `COMPUTE_VERSION` const — bump it when compute_pnl/cashflow/balance shape
     changes; every cached payload becomes stale automatically.

Same pattern as ml_quality / ml_user_orders (memory `feedback_ml_api_db_first`):
TEST → DB → CACHE. UI never bypasses this layer.

Sync API (psycopg2) so it works inside synchronous FastAPI endpoints
(/finance/reports, /finance/pnl-matrix) — these are sync because the legacy
compute_* functions are sync. Async callers (/escalar/summary) wrap via
`await asyncio.to_thread(cached_compute_sync, ...)`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

# Bump when compute_pnl / compute_cashflow / compute_balance / abc.aggregate
# change shape (added / removed / renamed fields). Bumping invalidates every
# cached payload at once — no manual `invalidate_*` calls needed.
COMPUTE_VERSION = "v1"

# Upload source_keys that influence any finance/abc compute. MAX(created_at)
# per source feeds the fingerprint.
UPLOAD_SOURCES_AFFECTING_FINANCE: tuple[str, ...] = (
    "vendas_ml",
    "ads_publicidade",
    "armazenagem_full",
    "retirada_full",
    "stock_full",
    "collection_mp",
    "extrato_mp",
    "extrato_nubank",
    "extrato_c6_brl",
    "extrato_c6_usd",
    "fatura_ml",
    "after_collection",
    "das_simples",
    "nfse_shps",
    "full_express",
    "trafficstars",
    "bybit_history",
)

# user_data keys that influence finance/abc compute. updated_at per key feeds
# the fingerprint. Includes both legacy (no prefix) and f2_ namespaces because
# the codebase still reads either (see services/projects.py).
USER_DATA_KEYS_AFFECTING_FINANCE: tuple[str, ...] = (
    "projects",
    "f2_projects",
    "sku_catalog",
    "f2_sku_catalog",
    "transaction_rules",
    "retirada_overrides",
    "f2_retirada_overrides",
    "f2_planned_payments",
    "f2_loans",
    "f2_dividends",
    "f2_orphan_assignments",
    "f2_publicidade_invoices",
    "f2_rental_payments",
)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS finance_compute_cache (
    user_id BIGINT NOT NULL,
    cache_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    fingerprint TEXT NOT NULL,
    deps_snapshot JSONB,
    cached_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, cache_key)
);
CREATE INDEX IF NOT EXISTS idx_finance_cache_age ON finance_compute_cache(cached_at);
"""


# ── Connection helper ────────────────────────────────────────────────────────

def _get_dsn() -> str | None:
    return os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")


def _connect():
    dsn = _get_dsn()
    if not dsn:
        return None
    try:
        return psycopg2.connect(dsn)
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: connect failed: %s", err)
        return None


# ── Schema ──────────────────────────────────────────────────────────────────

async def ensure_schema(pool) -> None:
    """Async variant — wired into FastAPI startup alongside other services."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def ensure_schema_sync() -> bool:
    """Sync variant — belt-and-suspenders inside endpoints (memory
    `reference_ad_storage_callers_pattern`). Returns True on success."""
    conn = _connect()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(CREATE_SQL)
        conn.commit()
        cur.close()
        return True
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: ensure_schema_sync failed: %s", err)
        return False
    finally:
        conn.close()


# ── Fingerprint ─────────────────────────────────────────────────────────────

def compute_fingerprint(
    user_id: int,
    extra_deps: dict[str, Any] | None = None,
) -> tuple[str, dict]:
    """SHA256 of all upload+user_data state that influences finance compute.

    Returns (fingerprint_hex, deps_snapshot). The snapshot is stored next to
    the cached payload so we can later answer "what changed on this miss?".
    Returns ("", {}) if DB is unavailable — caller should treat that as
    cache-miss-with-no-store (compute and return without trying to UPSERT).

    `extra_deps` lets specific cache_keys add their own fingerprint inputs
    on top of the base set. For example, ABC reads current_prices from
    `ml_user_items.fetched_at` — that's not an upload or user_data key, so
    we'd pass `{"ml_user_items_max_fetched": "<iso>"}` here. Keeps base
    fingerprint cheap (single query each) while letting niche caches add
    their own trip-wires.
    """
    conn = _connect()
    if conn is None:
        return "", {}
    deps: dict[str, Any] = {
        "version": COMPUTE_VERSION,
        "uploads": {},
        "user_data": {},
        "extra": extra_deps or {},
    }
    try:
        cur = conn.cursor()
        # uploads: one query, all sources
        cur.execute(
            """SELECT source_key, MAX(created_at) AS max_ts
               FROM uploads
               WHERE user_id = %s AND source_key = ANY(%s)
               GROUP BY source_key""",
            (user_id, list(UPLOAD_SOURCES_AFFECTING_FINANCE)),
        )
        per_source = {row[0]: row[1] for row in cur.fetchall()}
        for src in UPLOAD_SOURCES_AFFECTING_FINANCE:
            ts = per_source.get(src)
            deps["uploads"][src] = ts.isoformat() if ts else None
        # user_data: one query, all keys
        cur.execute(
            """SELECT data_key, updated_at
               FROM user_data
               WHERE user_id = %s AND data_key = ANY(%s)""",
            (user_id, list(USER_DATA_KEYS_AFFECTING_FINANCE)),
        )
        per_key = {row[0]: row[1] for row in cur.fetchall()}
        for key in USER_DATA_KEYS_AFFECTING_FINANCE:
            ts = per_key.get(key)
            deps["user_data"][key] = ts.isoformat() if ts else None
        cur.close()
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: compute_fingerprint failed: %s", err)
        return "", {}
    finally:
        conn.close()
    fp = hashlib.sha256(
        json.dumps(deps, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fp, deps


# ── Cache I/O ───────────────────────────────────────────────────────────────

def _read_cached(user_id: int, cache_key: str, fingerprint: str) -> Any | None:
    """SELECT and return payload only if fingerprint matches; else None."""
    conn = _connect()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT payload, fingerprint
               FROM finance_compute_cache
               WHERE user_id = %s AND cache_key = %s""",
            (user_id, cache_key),
        )
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        payload, stored_fp = row
        if stored_fp != fingerprint:
            return None
        if isinstance(payload, (dict, list)):
            return payload
        if isinstance(payload, str):
            return json.loads(payload)
        return payload
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: _read_cached failed: %s", err)
        return None
    finally:
        conn.close()


def _write_cached(
    user_id: int,
    cache_key: str,
    payload: Any,
    fingerprint: str,
    deps_snapshot: dict,
) -> bool:
    """UPSERT payload + fingerprint + deps_snapshot. Returns True on success."""
    conn = _connect()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO finance_compute_cache
                 (user_id, cache_key, payload, fingerprint, deps_snapshot, cached_at)
               VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, NOW())
               ON CONFLICT (user_id, cache_key)
               DO UPDATE SET payload = EXCLUDED.payload,
                             fingerprint = EXCLUDED.fingerprint,
                             deps_snapshot = EXCLUDED.deps_snapshot,
                             cached_at = NOW()""",
            (
                user_id,
                cache_key,
                json.dumps(payload, ensure_ascii=False, default=str),
                fingerprint,
                json.dumps(deps_snapshot, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: _write_cached failed: %s", err)
        return False
    finally:
        conn.close()


# ── Public read-through API ─────────────────────────────────────────────────

def cached_compute(
    user_id: int,
    cache_key: str,
    compute_fn: Callable[[], Any],
    *,
    force: bool = False,
    should_cache: Callable[[Any], bool] | None = None,
    extra_deps: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Read-through cache. Returns (payload, status) where status ∈
    {"hit", "miss", "stale", "force", "no_db", "compute_only", "skip_cache"}.

    - "hit" — fingerprint matched, payload from DB.
    - "miss" — no row at all; computed + stored.
    - "stale" — row exists with mismatched fingerprint; recomputed + overwritten.
    - "force" — caller passed force=True; recomputed + overwritten.
    - "no_db" — DB unreachable; computed + returned, NOT stored.
    - "compute_only" — UPSERT failed; computed + returned, NOT stored.
    - "skip_cache" — `should_cache(payload)` returned False; computed +
      returned, NOT stored. Use this for partial / errored results so half-
      computed bundles don't poison the cache.

    Cache key naming convention:
        "reports:{project}:{period_from}:{period_to}:{basis}"
        "matrix:{project}"
        "abc:{project}:{days}"
    """
    # Lazy schema (belt-and-suspenders per memory). One-shot per process is
    # enough — Postgres caches the IF NOT EXISTS check at near-zero cost.
    ensure_schema_sync()

    fingerprint, deps_snapshot = compute_fingerprint(user_id, extra_deps=extra_deps)
    if not fingerprint:
        # DB unreachable or fingerprint compute failed. Don't block the user;
        # run the compute synchronously and return without caching.
        return compute_fn(), "no_db"

    if not force:
        cached = _read_cached(user_id, cache_key, fingerprint)
        if cached is not None:
            return cached, "hit"
        # Determine miss vs stale for telemetry — quick second probe by key only.
        was_present = _has_any_row(user_id, cache_key)
        status_hint = "stale" if was_present else "miss"
    else:
        status_hint = "force"

    payload = compute_fn()
    if should_cache is not None and not should_cache(payload):
        return payload, "skip_cache"
    stored = _write_cached(user_id, cache_key, payload, fingerprint, deps_snapshot)
    if not stored:
        return payload, "compute_only"
    return payload, status_hint


def _has_any_row(user_id: int, cache_key: str) -> bool:
    conn = _connect()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM finance_compute_cache WHERE user_id = %s AND cache_key = %s",
            (user_id, cache_key),
        )
        return cur.fetchone() is not None
    except Exception:
        return False
    finally:
        conn.close()


# ── Maintenance ─────────────────────────────────────────────────────────────

def cleanup_stale_sync(max_age_days: int = 14) -> int:
    """Delete rows older than max_age_days. Cron-safe.

    Stale-by-fingerprint rows accumulate naturally — fingerprint mismatch on
    read returns no value, but the row remains until cleanup. Returns the
    number of rows deleted.
    """
    conn = _connect()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM finance_compute_cache "
            "WHERE cached_at < NOW() - INTERVAL '%s days'" % int(max_age_days),
        )
        deleted = cur.rowcount or 0
        conn.commit()
        cur.close()
        return deleted
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: cleanup_stale_sync failed: %s", err)
        return 0
    finally:
        conn.close()


def stats_sync(user_id: int | None = None) -> dict:
    """Diagnostic snapshot — used by /finance/cache/stats endpoint.

    If user_id is given, scoped to that user; otherwise global.
    """
    conn = _connect()
    if conn is None:
        return {"available": False}
    try:
        cur = conn.cursor()
        if user_id is None:
            cur.execute(
                """SELECT COUNT(*) AS n,
                          COUNT(DISTINCT user_id) AS users,
                          MIN(cached_at) AS oldest,
                          MAX(cached_at) AS newest
                   FROM finance_compute_cache"""
            )
            n, users, oldest, newest = cur.fetchone()
            cur.close()
            return {
                "available": True,
                "rows": int(n or 0),
                "users": int(users or 0),
                "oldest": oldest.isoformat() if oldest else None,
                "newest": newest.isoformat() if newest else None,
                "compute_version": COMPUTE_VERSION,
            }
        cur.execute(
            """SELECT cache_key, fingerprint, cached_at
               FROM finance_compute_cache
               WHERE user_id = %s
               ORDER BY cached_at DESC""",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return {
            "available": True,
            "user_id": user_id,
            "compute_version": COMPUTE_VERSION,
            "entries": [
                {
                    "cache_key": r[0],
                    "fingerprint": r[1][:12] + "…" if r[1] else None,
                    "cached_at": r[2].isoformat() if r[2] else None,
                }
                for r in rows
            ],
        }
    except Exception as err:  # noqa: BLE001
        log.warning("finance_cache: stats_sync failed: %s", err)
        return {"available": False, "error": str(err)}
    finally:
        conn.close()
