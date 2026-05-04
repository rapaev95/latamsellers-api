"""Admin-only user management: list/patch/block + tier grant/revoke.

Every route requires `require_admin`. Role changes additionally require
`require_superadmin` — admins can grant tiers but cannot promote other admins.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from v2.db import get_pool
from v2.deps import CurrentUser, require_admin
from v2.storage import users_admin_storage as store


router = APIRouter(
    prefix="/admin/users",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


# ── Schemas ────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: int
    email: str
    name: str = ""
    role: str = "user"
    tiers: list[str] = Field(default_factory=list)
    blocked: bool = False
    created_at: Optional[str] = None


class UsersListOut(BaseModel):
    items: list[UserOut]
    total: int
    limit: int
    offset: int


class UserPatchIn(BaseModel):
    role: Optional[str] = None      # user | admin | superadmin  (requires superadmin)
    tiers: Optional[list[str]] = None
    blocked: Optional[bool] = None


class TierIn(BaseModel):
    tier: str


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("", response_model=UsersListOut)
async def list_users_endpoint(
    q: Optional[str] = Query(None, description="Case-insensitive match on email/name"),
    role: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    blocked: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Paginated list of all users with filters. Admin+."""
    return await store.list_users(
        pool, q=q, role=role, tier=tier, blocked=blocked, limit=limit, offset=offset,
    )


@router.get("/{user_id}", response_model=UserOut)
async def get_user_endpoint(user_id: int, pool=Depends(get_pool)) -> dict[str, Any]:
    row = await store.get_user(pool, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="user_not_found")
    return row


@router.patch("/{user_id}", response_model=UserOut)
async def patch_user_endpoint(
    user_id: int,
    body: UserPatchIn,
    actor: CurrentUser = Depends(require_admin),
    pool=Depends(get_pool),
) -> dict[str, Any]:
    """Update role/tiers/blocked. Role changes require superadmin."""
    if body.role is not None and actor.role != "superadmin":
        raise HTTPException(
            status_code=403,
            detail={"code": "superadmin_required", "reason": "role_change"},
        )
    try:
        row = await store.update_user(
            pool, user_id, role=body.role, tiers=body.tiers, blocked=body.blocked,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if row is None:
        raise HTTPException(status_code=404, detail="user_not_found")
    return row


@router.post("/{user_id}/tiers/grant", response_model=UserOut)
async def grant_tier_endpoint(
    user_id: int, body: TierIn, pool=Depends(get_pool),
) -> dict[str, Any]:
    try:
        row = await store.grant_tier(pool, user_id, body.tier)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if row is None:
        raise HTTPException(status_code=404, detail="user_not_found")
    return row


@router.post("/{user_id}/tiers/revoke", response_model=UserOut)
async def revoke_tier_endpoint(
    user_id: int, body: TierIn, pool=Depends(get_pool),
) -> dict[str, Any]:
    try:
        row = await store.revoke_tier(pool, user_id, body.tier)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if row is None:
        raise HTTPException(status_code=404, detail="user_not_found")
    return row
