"""ML test users — POST /users/test_user wrapper + cache.

Used to bootstrap publishing-flow smoke tests without polluting real seller
accounts. Each LS user can create up to 10 test users per real ML account
(ML platform limit).

Test users live in production ML environment but their items don't index for
real buyers and they can't receive real payments. Credentials returned by ML
are static — save them once on creation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_test_users (
  id              SERIAL PRIMARY KEY,
  ls_user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  parent_ml_user_id BIGINT,
  test_user_id    BIGINT NOT NULL,
  nickname        TEXT NOT NULL,
  password        TEXT NOT NULL,
  email           TEXT,
  site_id         TEXT NOT NULL,
  site_status     TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(ls_user_id, test_user_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_test_users_ls_user ON ml_test_users(ls_user_id);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


async def create_test_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    ls_user_id: int,
    site_id: str = "MLB",
) -> dict[str, Any]:
    """Create one ML test user under the current LS user's OAuth token.

    Returns the saved row (including the test user's password — store it,
    ML doesn't expose it again).
    """
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, ls_user_id)
    except ml_oauth_svc.MLRefreshError as err:
        raise RuntimeError(f"ml_oauth_required: {err}") from err

    tokens = await ml_oauth_svc.load_user_tokens(pool, ls_user_id)
    parent_ml_user_id = tokens.get("ml_user_id") if tokens else None

    resp = await http.post(
        f"{ML_API_BASE}/users/test_user",
        json={"site_id": site_id},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=15.0,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"ml_rejected:{resp.status_code}:{resp.text[:300]}")

    payload = resp.json() or {}
    test_user_id = int(payload.get("id"))
    nickname = payload.get("nickname") or ""
    password = payload.get("password") or ""
    email = payload.get("email")
    site_status = payload.get("site_status")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ml_test_users (
                ls_user_id, parent_ml_user_id, test_user_id,
                nickname, password, email, site_id, site_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ls_user_id, test_user_id) DO UPDATE SET
                nickname = EXCLUDED.nickname,
                password = EXCLUDED.password,
                email = EXCLUDED.email,
                site_status = EXCLUDED.site_status
            RETURNING id, ls_user_id, parent_ml_user_id, test_user_id,
                      nickname, password, email, site_id, site_status, created_at
            """,
            ls_user_id, parent_ml_user_id, test_user_id,
            nickname, password, email, site_id, site_status,
        )
    return dict(row)


async def list_test_users(
    pool: asyncpg.Pool, ls_user_id: int
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, parent_ml_user_id, test_user_id, nickname, password,
                   email, site_id, site_status, created_at
            FROM ml_test_users
            WHERE ls_user_id = $1
            ORDER BY created_at DESC
            """,
            ls_user_id,
        )
    return [dict(r) for r in rows]


async def delete_test_user(
    pool: asyncpg.Pool, ls_user_id: int, test_user_id: int
) -> bool:
    """Drops the row from our cache. ML has no delete endpoint for test users —
    they remain in ML's system but unused."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM ml_test_users WHERE ls_user_id = $1 AND test_user_id = $2",
            ls_user_id, test_user_id,
        )
    return result.endswith("1")
