"""DB-sourced counterparts to `vendas_ml.load_all_vendas` and
`armazenagem.load_all_armazenagem`.

Reads raw `file_bytes` from the `uploads` table (per-user), feeds them through
the same source-agnostic parsers (`parse_vendas_bytes`, `parse_armazenagem_bytes`),
and yields the same dataclasses the FS loaders produce. This is the per-user
alternative selected by `LS_STORAGE_MODE=db` (see `v2.settings`).
"""
from __future__ import annotations

import time
from typing import Any, Iterable

import asyncpg

from v2.parsers.armazenagem import StorageData, parse_armazenagem_bytes
from v2.parsers.publicidade import PublicidadeRow, parse_publicidade_bytes
from v2.parsers.stock_full import StockFullSku, parse_stock_full_bytes
from v2.parsers.vendas_ml import VendasRow, parse_vendas_bytes
from v2.storage import uploads_storage

VENDAS_SOURCE_KEY = "vendas_ml"
ARMAZENAGEM_SOURCE_KEY = "armazenagem_full"
STOCK_FULL_SOURCE_KEY = "stock_full"
PUBLICIDADE_SOURCE_KEY = "ads_publicidade"


# ── Parsed-dataset cache ──────────────────────────────────────────────────────
# Re-parsing every uploaded file (15 vendas + armazenagem + stock + ads) on
# every /escalar/products load was the page's dominant latency. Cache the
# parsed result per (user, source), keyed by a cheap fingerprint of the source's
# upload rows (id + created_at, no bytes fetched). A new/removed/replaced upload
# changes the fingerprint → automatic rebuild, so there's nothing to invalidate
# from the upload path. TTL is only a safety cap on staleness of the fingerprint
# itself. Callers treat the result as read-only (they build new structures from
# it), so returning the shared reference is safe.
_PARSE_CACHE: dict[tuple[int, str], tuple[str, float, Any]] = {}
_PARSE_CACHE_TTL = 1800  # seconds
_MISS = object()


async def _source_fingerprint(pool: asyncpg.Pool, user_id: int, source_key: str) -> str:
    """Cheap freshness probe for a source's uploads — ids + created_at, no bytes."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, EXTRACT(EPOCH FROM created_at)::bigint AS ts
                 FROM uploads
                WHERE user_id = $1 AND source_key = $2 AND file_bytes IS NOT NULL
                ORDER BY id""",
            user_id, source_key,
        )
    return ";".join(f"{r['id']}:{r['ts']}" for r in rows) or "empty"


def _cache_get(user_id: int, source_key: str, fp: str) -> Any:
    hit = _PARSE_CACHE.get((user_id, source_key))
    if hit and hit[0] == fp and (time.monotonic() - hit[1]) < _PARSE_CACHE_TTL:
        return hit[2]
    return _MISS


def _cache_put(user_id: int, source_key: str, fp: str, result: Any) -> Any:
    _PARSE_CACHE[(user_id, source_key)] = (fp, time.monotonic(), result)
    return result


def invalidate_parse_cache(user_id: int | None = None) -> None:
    """Drop cached parses for one user (or all). The fingerprint already
    self-invalidates on upload; this is a belt-and-suspenders hook for callers
    that want an immediate, explicit clear."""
    if user_id is None:
        _PARSE_CACHE.clear()
        return
    for k in [k for k in _PARSE_CACHE if k[0] == user_id]:
        _PARSE_CACHE.pop(k, None)


async def load_user_vendas(pool: asyncpg.Pool, user_id: int) -> list[VendasRow]:
    """All Vendas ML rows across the user's uploaded files, deduped by sale_id.

    Sale_id dedupe matches the FS loader — snapshot files (90d rolling) overlap
    with monthly exports and must not double-count.
    """
    fp = await _source_fingerprint(pool, user_id, VENDAS_SOURCE_KEY)
    cached = _cache_get(user_id, VENDAS_SOURCE_KEY, fp)
    if cached is not _MISS:
        return cached
    files = await uploads_storage.fetch_files_by_source(pool, user_id, VENDAS_SOURCE_KEY)
    seen: set[str] = set()
    out: list[VendasRow] = []
    # Newest first (fetch_files_by_source orders by created_at DESC) → dedupe
    # prefers the most recently uploaded copy when sale_ids collide.
    for sf in files:
        for row in parse_vendas_bytes(sf.file_bytes, sf.filename):
            if row.sale_id and row.sale_id in seen:
                continue
            if row.sale_id:
                seen.add(row.sale_id)
            out.append(row)
    return _cache_put(user_id, VENDAS_SOURCE_KEY, fp, out)


async def load_user_armazenagem(pool: asyncpg.Pool, user_id: int) -> dict[str, StorageData]:
    """Merge every armazenagem file the user uploaded; freshest end_date wins per SKU."""
    fp = await _source_fingerprint(pool, user_id, ARMAZENAGEM_SOURCE_KEY)
    cached = _cache_get(user_id, ARMAZENAGEM_SOURCE_KEY, fp)
    if cached is not _MISS:
        return cached
    files = await uploads_storage.fetch_files_by_source(pool, user_id, ARMAZENAGEM_SOURCE_KEY)
    parsed: list[tuple[int, list[StorageData]]] = [
        parse_armazenagem_bytes(sf.file_bytes) for sf in files
    ]
    parsed.sort(key=lambda x: x[0], reverse=True)

    merged: dict[str, StorageData] = {}
    for _end_date, rows in parsed:
        for row in rows:
            if row.sku not in merged:
                merged[row.sku] = row
    return _cache_put(user_id, ARMAZENAGEM_SOURCE_KEY, fp, merged)


async def list_user_vendas_filenames(pool: asyncpg.Pool, user_id: int) -> list[str]:
    files = await uploads_storage.fetch_files_by_source(pool, user_id, VENDAS_SOURCE_KEY)
    return [sf.filename for sf in files]


async def load_user_stock_full(pool: asyncpg.Pool, user_id: int) -> dict[str, StockFullSku]:
    """Merge every stock_full file the user uploaded; newest-first wins per SKU.

    `fetch_files_by_source` orders by `created_at DESC`, so the first parsed
    dict overrides older ones — mirroring the FS loader's newest-wins policy.
    """
    fp = await _source_fingerprint(pool, user_id, STOCK_FULL_SOURCE_KEY)
    cached = _cache_get(user_id, STOCK_FULL_SOURCE_KEY, fp)
    if cached is not _MISS:
        return cached
    files = await uploads_storage.fetch_files_by_source(pool, user_id, STOCK_FULL_SOURCE_KEY)
    merged: dict[str, StockFullSku] = {}
    for sf in files:
        parsed = parse_stock_full_bytes(sf.file_bytes)
        for sku, entry in parsed.items():
            if sku not in merged:
                merged[sku] = entry
    return _cache_put(user_id, STOCK_FULL_SOURCE_KEY, fp, merged)


async def load_user_publicidade(pool: asyncpg.Pool, user_id: int) -> list[PublicidadeRow]:
    """All Product Ads rows from the user's uploaded reports, deduped by
    (mlb, desde, ate, investimento). ML can emit overlapping monthly exports
    the same way vendas does — same period values would double-count otherwise.
    """
    fp = await _source_fingerprint(pool, user_id, PUBLICIDADE_SOURCE_KEY)
    cached = _cache_get(user_id, PUBLICIDADE_SOURCE_KEY, fp)
    if cached is not _MISS:
        return cached
    files = await uploads_storage.fetch_files_by_source(pool, user_id, PUBLICIDADE_SOURCE_KEY)
    seen: set[tuple] = set()
    out: list[PublicidadeRow] = []
    for sf in files:
        for row in parse_publicidade_bytes(sf.file_bytes, sf.filename):
            key = (row.mlb, row.desde.isoformat(), row.ate.isoformat(), row.investimento)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    return _cache_put(user_id, PUBLICIDADE_SOURCE_KEY, fp, out)


def dedupe_vendas_rows(rows: Iterable[VendasRow]) -> list[VendasRow]:
    """Standalone dedupe helper (used by the backfill script)."""
    seen: set[str] = set()
    out: list[VendasRow] = []
    for row in rows:
        if row.sale_id and row.sale_id in seen:
            continue
        if row.sale_id:
            seen.add(row.sale_id)
        out.append(row)
    return out
