"""Per-user persistence for tracked keywords + position_history (Postgres).

Scoped to one user via `user_id` on every query — never trust the client.
"""
from __future__ import annotations

from typing import Optional

import asyncpg


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tracked_keywords (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    site_id TEXT NOT NULL DEFAULT 'MLB',
    category_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, item_id, keyword, site_id)
);

CREATE TABLE IF NOT EXISTS position_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    site_id TEXT NOT NULL DEFAULT 'MLB',
    position INTEGER,
    total_results INTEGER,
    found BOOLEAN DEFAULT FALSE,
    checked_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_position_history_user_item_kw
    ON position_history(user_id, item_id, keyword, checked_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create tracked_keywords + position_history if missing. Idempotent."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)


async def add_tracked(
    pool: asyncpg.Pool,
    *,
    user_id: int,
    item_id: str,
    keyword: str,
    site_id: str = "MLB",
    category_id: Optional[str] = None,
) -> int:
    """Upsert a tracked (item_id, keyword) pair for the user. Returns row id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO tracked_keywords (user_id, item_id, keyword, site_id, category_id)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id, item_id, keyword, site_id)
            DO UPDATE SET category_id = EXCLUDED.category_id
            RETURNING id
            """,
            user_id, item_id, keyword, site_id, category_id,
        )
    return int(row["id"])


async def delete_tracked(pool: asyncpg.Pool, user_id: int, tracked_id: int) -> bool:
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM tracked_keywords WHERE id = $1 AND user_id = $2",
            tracked_id, user_id,
        )
    return res.endswith(" 1")


async def list_tracked(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    """Return each tracked keyword + latest `position_history` row joined in.

    Uses a LATERAL subquery so every tracked row gets exactly its own most
    recent history (avoids GROUP BY over the wider column set).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                t.id,
                t.item_id,
                t.keyword,
                t.site_id,
                t.category_id,
                t.created_at,
                h.position AS last_position,
                h.found AS last_found,
                h.checked_at AS last_checked_at
            FROM tracked_keywords t
            LEFT JOIN LATERAL (
                SELECT position, found, checked_at
                FROM position_history
                WHERE user_id = t.user_id
                  AND item_id = t.item_id
                  AND keyword = t.keyword
                  AND site_id = t.site_id
                ORDER BY checked_at DESC
                LIMIT 1
            ) h ON TRUE
            WHERE t.user_id = $1
            ORDER BY t.created_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


async def record_check(
    pool: asyncpg.Pool,
    *,
    user_id: int,
    item_id: str,
    keyword: str,
    site_id: str,
    position: Optional[int],
    total_results: int,
    found: bool,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO position_history
              (user_id, item_id, keyword, site_id, position, total_results, found)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            user_id, item_id, keyword, site_id, position, total_results, found,
        )
