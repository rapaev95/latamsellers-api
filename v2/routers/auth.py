"""GET /api/v2/me — current user from cookie. Proves SSO chain works.

Frontend (useAuth) reads role/tiers here to gate PaywallGate and admin routes —
без этих полей superadmin не распознаётся и UI показывает lock на всех tier-ах.

`effective_tiers` adds tiers inherited from project memberships so an invited
collaborator can see the sections the project owner has access to. The raw
`tiers` array stays as the user's OWN tiers for places that need to know
"is this their own subscription vs inherited" (UI sidebar may hide Finance
for inheritance-only users since finance is owner-only by design).
"""
from fastapi import APIRouter, Depends

from v2.db import get_pool
from v2.deps import CurrentUser, current_user
from v2.services import project_members as pm_svc

router = APIRouter(tags=["auth"])


@router.get("/me")
async def me(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    # Compute effective tiers (own + inherited via memberships) and the list
    # of accepted memberships so the UI can show "you're a member of project X".
    effective_tiers: list[str] = list(user.tiers)
    memberships: list[dict] = []
    if pool is not None:
        try:
            await pm_svc.ensure_schema(pool)
            effective_tiers = await pm_svc.get_effective_tiers(pool, user.id)
            memberships = await pm_svc.list_my_memberships(pool, user.id)
        except Exception:
            # Don't fail the auth payload if membership tables hiccup.
            # Falls back to own tiers only — same behavior as before this feature.
            effective_tiers = list(user.tiers)
            memberships = []
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,                  # "user" | "admin" | "superadmin"
        "tiers": user.tiers,                # own tiers (for "is mine vs inherited" checks)
        "effective_tiers": effective_tiers, # own ∪ inherited — for PaywallGate
        "memberships": memberships,         # projects this user was invited to
        "blocked": user.blocked,
    }
