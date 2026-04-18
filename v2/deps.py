"""Shared dependencies for v2 routers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote

from fastapi import Depends, HTTPException, Request, status

from v2.db import get_pool


@dataclass
class CurrentUser:
    id: int
    email: str
    name: str


async def current_user(
    request: Request,
    pool=Depends(get_pool),
) -> CurrentUser:
    """Authenticate via the `ls_auth` cookie set by Next.js / shared with Streamlit.

    Cookie value is JSON `{id, email, name}` (httpOnly, sameSite=lax, host-only).
    Validates user_id against `users` table — stale cookie → 401.
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
            "SELECT id, email, name FROM users WHERE id = $1",
            user_id,
        )

    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unknown_user")

    return CurrentUser(id=row["id"], email=row["email"], name=row["name"] or "")


async def maybe_current_user(
    request: Request,
    pool=Depends(get_pool),
) -> Optional[CurrentUser]:
    """Same as `current_user` but returns None instead of 401. Useful for /me."""
    try:
        return await current_user(request, pool)
    except HTTPException:
        return None
