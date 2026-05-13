"""Project ACL — team members and invitations.

Originally lived inside the Escalar router under `/api/v2/escalar/team/*` because
the ACL was prototyped as part of the Escalar surface. The feature scopes
Finance + Calculator + Escalar, so the endpoints now live under `/api/v2/team/*`.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Query

from v2.deps import CurrentUser, current_user, get_pool
from v2.services import (
    email_brevo as email_brevo_svc,
    project_members as project_members_svc,
)

router = APIRouter(prefix="/team", tags=["team"])


def _app_base_url() -> str:
    return os.environ.get("APP_BASE_URL", "https://app.lsprofit.app").rstrip("/")


def _build_accept_url(token: str) -> str:
    return f"{_app_base_url()}/auth/invite/{token}"


@router.get("/members")
async def team_list_members(
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """List accepted members invited by current user. Optional ?project=NAME filter."""
    if pool is None:
        return {"members": [], "invitations": []}
    await project_members_svc.ensure_schema(pool)
    members = await project_members_svc.list_members(pool, user.id, project_name=project)
    invitations = await project_members_svc.list_pending_invitations(pool, user.id)
    if project:
        invitations = [inv for inv in invitations if inv["project_name"] == project]
    invitations_safe = [
        {k: v for k, v in inv.items() if k != "token"} | {
            "accept_url": _build_accept_url(inv["token"]),
        }
        for inv in invitations
    ]
    return {
        "members": members,
        "invitations": invitations_safe,
        "email_configured": email_brevo_svc.is_configured(),
    }


@router.get("/my-memberships")
async def team_my_memberships(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Projects the current user has been added to (membership-as-collaborator)."""
    if pool is None:
        return {"memberships": []}
    await project_members_svc.ensure_schema(pool)
    rows = await project_members_svc.list_my_memberships(pool, user.id)
    return {"memberships": rows, "total": len(rows)}


@router.post("/invite")
async def team_invite(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Create or reuse an invitation. Body:
      email         — required
      project_name  — required
      role          — viewer | analyst | admin (default viewer)
      send_email    — bool, if true and Brevo configured, send the invite email
    """
    if pool is None:
        return {"error": "no_db"}
    await project_members_svc.ensure_schema(pool)

    email = (body or {}).get("email")
    project_name = (body or {}).get("project_name")
    role = (body or {}).get("role", "viewer")
    send_email = bool((body or {}).get("send_email", True))

    result = await project_members_svc.create_invitation(
        pool, user.id,
        email=email or "",
        project_name=project_name or "",
        role=role,
    )
    if result.get("error"):
        return result

    accept_url = _build_accept_url(result["token"])
    result["accept_url"] = accept_url

    email_status: dict[str, Any] = {"sent": False, "configured": email_brevo_svc.is_configured()}
    if send_email and email_brevo_svc.is_configured():
        try:
            send_res = await email_brevo_svc.send_invitation_email(
                to_email=result["email"],
                project_name=result["project_name"],
                role=result["role"],
                inviter_name=user.name or user.email,
                inviter_email=user.email,
                accept_url=accept_url,
            )
            email_status["sent"] = bool(send_res.get("ok"))
            email_status["raw"] = send_res
        except Exception as err:  # noqa: BLE001
            email_status["sent"] = False
            email_status["error"] = str(err)

    result["email_status"] = email_status
    result.pop("token", None)
    return result


@router.post("/invitations/{invitation_id}/revoke")
async def team_revoke_invitation(
    invitation_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    return await project_members_svc.revoke_invitation(pool, user.id, invitation_id)


@router.patch("/members/{member_id}")
async def team_update_member_role(
    member_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    if pool is None:
        return {"error": "no_db"}
    new_role = (body or {}).get("role")
    if not new_role:
        return {"error": "role_required"}
    return await project_members_svc.update_role(pool, user.id, member_id, new_role)


@router.delete("/members/{member_id}")
async def team_remove_member(
    member_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        return {"error": "no_db"}
    return await project_members_svc.remove_member(pool, user.id, member_id)


@router.get("/invitations/lookup")
async def team_invitation_lookup(
    token: str = Query(..., min_length=10),
    pool=Depends(get_pool),
):
    """Public-ish: returns the invitation details (no auth required) so the
    accept page can display project + inviter before the user logs in.
    Token itself is the credential — caller must already possess it."""
    if pool is None:
        return {"error": "no_db"}
    await project_members_svc.ensure_schema(pool)
    inv = await project_members_svc.get_invitation_by_token(pool, token)
    if not inv:
        return {"error": "invitation_not_found"}
    return {
        "email": inv["email"],
        "project_name": inv["project_name"],
        "role": inv["role"],
        "inviter_email": inv["inviter_email"],
        "inviter_name": inv["inviter_name"],
        "expires_at": inv["expires_at"],
        "is_active": inv["is_active"],
        "used_at": inv["used_at"],
        "revoked_at": inv["revoked_at"],
    }


@router.post("/invitations/accept")
async def team_invitation_accept(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
    body: dict[str, Any] = Body(...),
):
    """Accept an invitation as the logged-in user. Validates email match."""
    if pool is None:
        return {"error": "no_db"}
    token = (body or {}).get("token")
    if not token:
        return {"error": "token_required"}
    return await project_members_svc.accept_invitation(
        pool, token=token,
        accepting_user_id=user.id,
        accepting_email=user.email,
    )
