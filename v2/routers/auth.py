"""GET /api/v2/me — current user from cookie. Proves SSO chain works."""
from fastapi import APIRouter, Depends

from v2.deps import CurrentUser, current_user

router = APIRouter(tags=["auth"])


@router.get("/me")
async def me(user: CurrentUser = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name}
