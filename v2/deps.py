"""Shared dependencies for v2 routers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Optional
from urllib.parse import unquote

from fastapi import Depends, HTTPException, Query, Request, status

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
    Admins bypass; non-tiered users get 403 with a paywall payload.

    Inherits tiers via project memberships: a user who's been invited to a
    project owned by someone with `tier` also passes. Falls back to own
    tiers only if the membership lookup hiccups.
    """
    async def _dep(
        user: CurrentUser = Depends(current_user),
        pool=Depends(get_pool),
    ) -> CurrentUser:
        if _is_admin(user):
            return user
        if tier in user.tiers:
            return user
        if pool is not None:
            try:
                # Late import — project_members imports asyncpg-typed pool;
                # avoids circular dep at module load.
                from v2.services import project_members as _pm_svc
                effective = await _pm_svc.get_effective_tiers(pool, user.id)
                if tier in effective:
                    return user
            except Exception:
                # If membership query fails, deny just like before — caller
                # already saw the bare tier check fail.
                pass
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


def _is_superadmin(user: CurrentUser) -> bool:
    return user.role == "superadmin"


async def require_superadmin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    """Stricter gate than `require_admin` — used for services-projects flow
    (Estonia / GANZA) which exposes hand-curated tax brackets, RBT12 baseline
    decay, partner contributions, approved transfers. Regular admins can't
    create or edit these — they're owner-only because mistakes propagate
    into invoice-based ОПиУ and DAS calculations."""
    if not _is_superadmin(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "superadmin_required"},
        )
    return user


# ──────────────────────────────────────────────────────────────────────────────
# ML context resolver — dual-mode (self-service vs managed) Sprint 1
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MLContext:
    """Resolved ML credentials for the current request.

    - `source = 'self'`  → token from ml_user_tokens (Type B, owns own OAuth)
    - `source = 'managed'` → token from ml_managed_accounts (Type A, LS-managed)
    """
    ml_user_id: int
    access_token: str
    source: Literal["self", "managed"]
    managed_account_id: Optional[int] = None


async def resolve_ml_context(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    managed_account_id: Optional[int] = Query(None, alias="managed_account_id"),
) -> MLContext:
    """Decide which ML OAuth context to use for this request.

    - If `managed_account_id` query param is present AND caller is admin/
      superadmin → use the managed account token (Type A).
    - Otherwise → use caller's own ml_user_tokens token (Type B).

    Raises HTTPException 401 (ml_oauth_required) or 403 (permission denied).

    Late imports below avoid circular dependencies — these services may
    themselves depend on shared deps in this module.
    """
    if managed_account_id is not None:
        if not _is_admin(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "managed_account_requires_admin"},
            )
        from v2.services import ml_managed_accounts as _mgr
        try:
            token, _exp, _refreshed = await _mgr.get_valid_access_token(pool, managed_account_id)
        except Exception as err:  # noqa: BLE001 — surface as 401 with detail
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ml_managed_oauth_failed", "message": str(err)},
            )
        account = await _mgr.get_account(pool, managed_account_id)
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "managed_account_not_found"},
            )
        return MLContext(
            ml_user_id=int(account["ml_user_id"]),
            access_token=token,
            source="managed",
            managed_account_id=managed_account_id,
        )

    # Self-service path — read user's personal token.
    from v2.services import ml_oauth as _oauth
    try:
        token, *_ = await _oauth.get_valid_access_token(pool, user.id)
    except _oauth.MLRefreshError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "ml_oauth_required", "message": str(err)},
        )
    tokens = await _oauth.load_user_tokens(pool, user.id)
    ml_user_id = (tokens or {}).get("ml_user_id")
    if not ml_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "no_ml_user_id"},
        )
    return MLContext(
        ml_user_id=int(ml_user_id),
        access_token=token,
        source="self",
    )
