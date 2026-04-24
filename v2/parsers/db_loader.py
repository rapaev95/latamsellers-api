"""DB-sourced counterparts to `vendas_ml.load_all_vendas` and
`armazenagem.load_all_armazenagem`.

Reads raw `file_bytes` from the `uploads` table (per-user), feeds them through
the same source-agnostic parsers (`parse_vendas_bytes`, `parse_armazenagem_bytes`),
and yields the same dataclasses the FS loaders produce. This is the per-user
alternative selected by `LS_STORAGE_MODE=db` (see `v2.settings`).
"""
from __future__ import annotations

from typing import Iterable

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


async def load_user_vendas(pool: asyncpg.Pool, user_id: int) -> list[VendasRow]:
    """All Vendas ML rows across the user's uploaded files, deduped by sale_id.

    Sale_id dedupe matches the FS loader — snapshot files (90d rolling) overlap
    with monthly exports and must not double-count.
    """
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
    return out


async def load_user_armazenagem(pool: asyncpg.Pool, user_id: int) -> dict[str, StorageData]:
    """Merge every armazenagem file the user uploaded; freshest end_date wins per SKU."""
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
    return merged


async def list_user_vendas_filenames(pool: asyncpg.Pool, user_id: int) -> list[str]:
    files = await uploads_storage.fetch_files_by_source(pool, user_id, VENDAS_SOURCE_KEY)
    return [sf.filename for sf in files]


async def load_user_stock_full(pool: asyncpg.Pool, user_id: int) -> dict[str, StockFullSku]:
    """Merge every stock_full file the user uploaded; newest-first wins per SKU.

    `fetch_files_by_source` orders by `created_at DESC`, so the first parsed
    dict overrides older ones — mirroring the FS loader's newest-wins policy.
    """
    files = await uploads_storage.fetch_files_by_source(pool, user_id, STOCK_FULL_SOURCE_KEY)
    merged: dict[str, StockFullSku] = {}
    for sf in files:
        parsed = parse_stock_full_bytes(sf.file_bytes)
        for sku, entry in parsed.items():
            if sku not in merged:
                merged[sku] = entry
    return merged


async def load_user_publicidade(pool: asyncpg.Pool, user_id: int) -> list[PublicidadeRow]:
    """All Product Ads rows from the user's uploaded reports, deduped by
    (mlb, desde, ate, investimento). ML can emit overlapping monthly exports
    the same way vendas does — same period values would double-count otherwise.
    """
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
    return out


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
