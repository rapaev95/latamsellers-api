"""Per-user JSONB storage in `user_data` table, namespaced with `f2_` prefix.

Streamlit writes keys like `projects`, `sku_catalog`, `transaction_rules`.
Finance 2.0 writes the same logical data under `f2_projects`, `f2_sku_catalog`, …
so Streamlit data is never overwritten. An "Import from legacy" action will copy
non-`f2_` rows to `f2_` versions on demand.
"""
from __future__ import annotations

import json
from typing import Any

import asyncpg

PREFIX = "f2_"


def _full_key(key: str) -> str:
    """Add f2_ prefix unless caller already passed a fully-qualified key."""
    return key if key.startswith(PREFIX) else f"{PREFIX}{key}"


async def get(pool: asyncpg.Pool, user_id: int, key: str) -> Any | None:
    """Read a JSONB value for the user. Returns None when missing."""
    full = _full_key(key)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data_value FROM user_data WHERE user_id = $1 AND data_key = $2",
            user_id,
            full,
        )
    if row is None:
        return None
    raw = row["data_value"]
    # asyncpg already decodes JSONB into Python types when configured with type codecs;
    # if not, fall back to json.loads for safety.
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None
    return raw


async def put(pool: asyncpg.Pool, user_id: int, key: str, value: Any) -> None:
    """Upsert a JSONB value for the user."""
    full = _full_key(key)
    payload = json.dumps(value, ensure_ascii=False, default=str)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_data (user_id, data_key, data_value, updated_at)
            VALUES ($1, $2, $3::jsonb, NOW())
            ON CONFLICT (user_id, data_key)
            DO UPDATE SET data_value = EXCLUDED.data_value, updated_at = NOW()
            """,
            user_id,
            full,
            payload,
        )


async def delete(pool: asyncpg.Pool, user_id: int, key: str) -> None:
    full = _full_key(key)
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM user_data WHERE user_id = $1 AND data_key = $2",
            user_id,
            full,
        )


async def list_keys(pool: asyncpg.Pool, user_id: int) -> list[str]:
    """List all f2_* keys for the user (without the prefix)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT data_key FROM user_data WHERE user_id = $1 AND data_key LIKE 'f2_%'",
            user_id,
        )
    return [r["data_key"][len(PREFIX):] for r in rows]


async def get_legacy(pool: asyncpg.Pool, user_id: int, legacy_key: str) -> Any | None:
    """Read a Streamlit-era key WITHOUT the f2_ prefix. Used by import-legacy flow."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data_value FROM user_data WHERE user_id = $1 AND data_key = $2",
            user_id,
            legacy_key,
        )
    if row is None:
        return None
    raw = row["data_value"]
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None
    return raw
