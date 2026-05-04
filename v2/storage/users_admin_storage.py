"""Admin-only CRUD on the `users` table for roles/tiers/blocked.

Called from `routers/admin_users.py`. All writes are scoped to the target
user_id — callers must enforce `require_admin()` / `require_superadmin()`
on the route. Nothing here checks authorization.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import asyncpg


ALLOWED_ROLES: tuple[str, ...] = ("user", "admin", "superadmin")


def _parse_tiers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
        except (ValueError, TypeError):
            return []
    else:
        v = raw
    return [str(x) for x in (v or []) if isinstance(x, str)]


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"] or "",
        "role": str(row["role"] or "user"),
        "tiers": _parse_tiers(row["tiers"]),
        "blocked": bool(row["blocked"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


_SELECT_COLS = """
    id, email, name,
    COALESCE(role, 'user')       AS role,
    COALESCE(tiers, '[]'::jsonb) AS tiers,
    COALESCE(blocked, false)     AS blocked,
    created_at
"""


async def list_users(
    pool: asyncpg.Pool,
    *,
    q: Optional[str] = None,
    role: Optional[str] = None,
    tier: Optional[str] = None,
    blocked: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated list with optional filters. Returns {items, total, limit, offset}.

    `q` is a case-insensitive LIKE on email + name. `tier` matches JSONB
    containment so `tier='finance'` returns users whose tiers include it.
    """
    limit = max(1, min(200, int(limit)))
    offset = max(0, int(offset))

    where: list[str] = []
    params: list[Any] = []

    def _p(v: Any) -> str:
        params.append(v)
        return f"${len(params)}"

    if q:
        needle = f"%{q.strip().lower()}%"
        where.append(f"(LOWER(email) LIKE {_p(needle)} OR LOWER(COALESCE(name, '')) LIKE {_p(needle)})")
    if role:
        where.append(f"COALESCE(role, 'user') = {_p(role)}")
    if tier:
        where.append(f"COALESCE(tiers, '[]'::jsonb) @> {_p(json.dumps([tier]))}::jsonb")
    if blocked is not None:
        where.append(f"COALESCE(blocked, false) = {_p(bool(blocked))}")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM users {where_sql}", *params)
        rows = await conn.fetch(
            f"""
            SELECT {_SELECT_COLS}
              FROM users
              {where_sql}
             ORDER BY id DESC
             LIMIT {limit} OFFSET {offset}
            """,
            *params,
        )

    return {
        "items": [_row_to_dict(r) for r in rows],
        "total": int(total or 0),
        "limit": limit,
        "offset": offset,
    }


async def get_user(pool: asyncpg.Pool, user_id: int) -> Optional[dict[str, Any]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {_SELECT_COLS} FROM users WHERE id = $1",
            int(user_id),
        )
    return _row_to_dict(row) if row else None


async def update_user(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    role: Optional[str] = None,
    tiers: Optional[list[str]] = None,
    blocked: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Patch any subset of role/tiers/blocked. Returns the updated row or None."""
    sets: list[str] = []
    params: list[Any] = []

    def _p(v: Any) -> str:
        params.append(v)
        return f"${len(params)}"

    if role is not None:
        if role not in ALLOWED_ROLES:
            raise ValueError(f"invalid_role: {role}")
        sets.append(f"role = {_p(role)}")
    if tiers is not None:
        clean = sorted({str(t) for t in tiers if isinstance(t, str) and t})
        sets.append(f"tiers = {_p(json.dumps(clean))}::jsonb")
    if blocked is not None:
        sets.append(f"blocked = {_p(bool(blocked))}")

    if not sets:
        return await get_user(pool, user_id)

    params.append(int(user_id))
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE users
               SET {', '.join(sets)}
             WHERE id = ${len(params)}
            RETURNING {_SELECT_COLS}
            """,
            *params,
        )
    return _row_to_dict(row) if row else None


async def grant_tier(pool: asyncpg.Pool, user_id: int, tier: str) -> Optional[dict[str, Any]]:
    """Add `tier` to tiers JSONB if not already present. Idempotent."""
    tier = tier.strip()
    if not tier:
        raise ValueError("empty_tier")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE users
               SET tiers = CASE
                    WHEN COALESCE(tiers, '[]'::jsonb) @> $2::jsonb THEN tiers
                    ELSE COALESCE(tiers, '[]'::jsonb) || $2::jsonb
               END
             WHERE id = $1
            RETURNING {_SELECT_COLS}
            """,
            int(user_id),
            json.dumps([tier]),
        )
    return _row_to_dict(row) if row else None


async def revoke_tier(pool: asyncpg.Pool, user_id: int, tier: str) -> Optional[dict[str, Any]]:
    """Remove `tier` from tiers JSONB. Idempotent."""
    tier = tier.strip()
    if not tier:
        raise ValueError("empty_tier")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE users
               SET tiers = COALESCE(tiers, '[]'::jsonb) - $2::text
             WHERE id = $1
            RETURNING {_SELECT_COLS}
            """,
            int(user_id),
            tier,
        )
    return _row_to_dict(row) if row else None
