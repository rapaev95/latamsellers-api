"""Токены доступа для MCP-клиентов.

MCP-клиент — это не браузер: cookie `ls_auth` ему негде взять и незачем
хранить. Поэтому у партнёра есть отдельный ключ, который он вставляет в
конфиг своего клиента.

Хранится только хеш. Открытый ключ показывается один раз, при создании: если
партнёр его потерял, правильный ответ — выдать новый, а не искать старый.
Ключ ничего не разрешает сам по себе — он лишь называет пользователя, а что
тому видно, решает `v2.services.ml_access`.
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Any, Optional

import asyncpg

TOKEN_PREFIX = "ls_mcp_"
_TOKEN_BYTES = 32

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS mcp_tokens (
  id            SERIAL PRIMARY KEY,
  user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash    TEXT NOT NULL UNIQUE,
  name          TEXT,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  last_used_at  TIMESTAMPTZ,
  revoked_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_mcp_tokens_user ON mcp_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tokens_hash ON mcp_tokens(token_hash)
  WHERE revoked_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create(
    pool: asyncpg.Pool, user_id: int, name: str = "",
) -> dict[str, Any]:
    """Выдать новый ключ. Открытое значение возвращается только здесь."""
    await ensure_schema(pool)
    token = TOKEN_PREFIX + secrets.token_urlsafe(_TOKEN_BYTES)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO mcp_tokens (user_id, token_hash, name)
            VALUES ($1, $2, $3)
            RETURNING id, name, created_at
            """,
            user_id, _hash(token), (name or "").strip() or None,
        )
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "token": token,
    }


async def resolve(pool: asyncpg.Pool, token: str) -> Optional[dict[str, Any]]:
    """Ключ → пользователь, или None. Отозванные и чужие ключи не проходят.

    Сравнение идёт по хешу в индексе, поэтому в логи и в сообщения об ошибке
    открытый ключ не попадает.
    """
    if not token or not token.startswith(TOKEN_PREFIX):
        return None
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT t.id AS token_id, u.id, u.email, u.name, u.role, u.tiers,
                   u.blocked
              FROM mcp_tokens t
              JOIN users u ON u.id = t.user_id
             WHERE t.token_hash = $1 AND t.revoked_at IS NULL
            """,
            _hash(token),
        )
        if row is None:
            return None
        await conn.execute(
            "UPDATE mcp_tokens SET last_used_at = NOW() WHERE id = $1",
            row["token_id"],
        )
    if row["blocked"]:
        return None
    return {
        "token_id": row["token_id"],
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "role": row["role"],
        "tiers": row["tiers"],
    }


async def list_for_user(pool: asyncpg.Pool, user_id: int) -> list[dict[str, Any]]:
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, created_at, last_used_at
              FROM mcp_tokens
             WHERE user_id = $1 AND revoked_at IS NULL
             ORDER BY created_at DESC
            """,
            user_id,
        )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "last_used_at": (
                r["last_used_at"].isoformat() if r["last_used_at"] else None
            ),
        }
        for r in rows
    ]


async def revoke(pool: asyncpg.Pool, user_id: int, token_id: int) -> bool:
    """Отозвать свой ключ. Чужой отозвать нельзя — отсюда user_id в WHERE."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        res = await conn.execute(
            """
            UPDATE mcp_tokens SET revoked_at = NOW()
             WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL
            """,
            token_id, user_id,
        )
    return res.endswith("1")
