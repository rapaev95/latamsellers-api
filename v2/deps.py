"""Shared dependencies for v2 routers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import unquote

from fastapi import Depends, HTTPException, Request, status

from v2.db import get_pool


@dataclass
class CurrentUser:
    id: int
    email: str
    name: str
    role: str = "user"                    # "user" | "admin" | "superadmin"
    tiers: list[str] = field(default_factory=list)   # paywall tiers: finance/calculator/escalar
    blocked: bool = False


def _parse_tiers(raw) -> list[str]:
    """tiers в БД — jsonb (list[str]) или null."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw if t]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed if t]
        except (ValueError, TypeError):
            pass
    return []


async def current_user(
    request: Request,
    pool=Depends(get_pool),
) -> CurrentUser:
    """Authenticate via the `ls_auth` cookie set by Next.js / shared with Streamlit.

    Cookie value is JSON `{id, email, name}` (httpOnly, sameSite=lax, host-only).
    Validates user_id against `users` table — stale cookie → 401.
    Returns role, tiers, blocked so callers can gate paywall / admin routes.
    """
    raw = request.cookies.get("ls_auth")
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="no_cookie")

    # Next.js (cookies API) URL-encodes the value; raw form may also work.
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        try:
            payload = json.loads(unquote(raw))
        except (ValueError, TypeError):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad_cookie")

    user_id = payload.get("id")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="no_user_id")

    if pool is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no_db")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, name, role, tiers, blocked FROM users WHERE id = $1",
            user_id,
        )

    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unknown_user")

    if row["blocked"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user_blocked")

    return CurrentUser(
        id=row["id"],
        email=row["email"],
        name=row["name"] or "",
        role=(row["role"] or "user"),
        tiers=_parse_tiers(row["tiers"]),
        blocked=bool(row["blocked"]),
    )


async def maybe_current_user(
    request: Request,
    pool=Depends(get_pool),
) -> Optional[CurrentUser]:
    """Same as `current_user` but returns None instead of 401. Useful for /me."""
    try:
        return await current_user(request, pool)
    except HTTPException:
        return None


def _is_admin(user: CurrentUser) -> bool:
    return user.role in ("admin", "superadmin")


def require_tier(tier: str):
    """FastAPI dependency factory enforcing a per-feature tier.
    Admins bypass; non-tiered users get 403 with a paywall payload."""
    async def _dep(user: CurrentUser = Depends(current_user)) -> CurrentUser:
        if _is_admin(user):
            return user
        if tier in user.tiers:
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "tier_required", "tier": tier},
        )
    return _dep


async def require_admin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    if not _is_admin(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "admin_required"},
        )
    return user
