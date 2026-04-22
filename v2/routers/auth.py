"""GET /api/v2/me — current user from cookie. Proves SSO chain works.

Frontend (useAuth) reads role/tiers here to gate PaywallGate and admin routes —
без этих полей superadmin не распознаётся и UI показывает lock на всех tier-ах.
"""
from fastapi import APIRouter, Depends

from v2.deps import CurrentUser, current_user

router = APIRouter(tags=["auth"])


@router.get("/me")
async def me(user: CurrentUser = Depends(current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,            # "user" | "admin" | "superadmin"
        "tiers": user.tiers,          # e.g. ["finance", "calculator"]
        "blocked": user.blocked,
    }
