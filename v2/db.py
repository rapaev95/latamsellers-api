"""asyncpg connection pool for Finance 2.0.

Single shared pool, lazy-created. v1 keeps its own per-request psycopg2 connection
in main.py — they coexist on the same DATABASE_URL.
"""
from __future__ import annotations

from typing import Optional

import asyncpg

from v2.settings import get_settings

_pool: Optional[asyncpg.Pool] = None


async def create_pool() -> asyncpg.Pool | None:
    """Initialise the pool. Called on FastAPI startup; safe to call repeatedly."""
    global _pool
    if _pool is not None:
        return _pool
    dsn = get_settings().database_url
    if not dsn:
        return None
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_pool() -> asyncpg.Pool | None:
    """Dependency: returns the pool, creating it lazily if startup didn't run."""
    if _pool is None:
        await create_pool()
    return _pool
