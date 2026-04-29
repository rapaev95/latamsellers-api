"""Project ACL — invite team members to specific projects with a role.

Two tables:
  escalar_project_members  — accepted memberships (user_id × project_name × role)
  escalar_invitations      — pending email invitations with one-time token

Roles: owner | admin | analyst | viewer
  owner    — full access, only the user who registered the project (implicit)
  admin    — full access except member management + billing
  analyst  — read-write on positions/AB tests, read-only elsewhere
  viewer   — read-only across all features

Owner-membership is implicit: the user who created/uploaded a project is its
owner; we do NOT insert a row for the owner. Membership rows describe added
collaborators only.

Token format: 32 url-safe bytes (43 chars), single-use, expires in 7 days.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

ROLE_ORDER = ["viewer", "analyst", "admin", "owner"]
INVITE_TTL_DAYS = 7
TOKEN_BYTES = 32

_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS escalar_project_members (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL,
      project_name TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'viewer',
      invited_by INTEGER,
      invited_at TIMESTAMPTZ DEFAULT NOW(),
      accepted_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(user_id, project_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_escalar_members_user ON escalar_project_members(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_escalar_members_project ON escalar_project_members(project_name)",
    """
    CREATE TABLE IF NOT EXISTS escalar_invitations (
      id SERIAL PRIMARY KEY,
      email TEXT NOT NULL,
      project_name TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'viewer',
      token TEXT NOT NULL UNIQUE,
      invited_by INTEGER NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      expires_at TIMESTAMPTZ NOT NULL,
      used_at TIMESTAMPTZ,
      used_by_user_id INTEGER,
      revoked_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_escalar_invites_email_pending ON escalar_invitations(lower(email)) WHERE used_at IS NULL AND revoked_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_escalar_invites_inviter ON escalar_invitations(invited_by)",
    "CREATE INDEX IF NOT EXISTS idx_escalar_invites_token ON escalar_invitations(token)",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    for stmt in _CREATE_STATEMENTS:
        async with pool.acquire() as conn:
            await conn.execute(stmt)


def _normalize_role(role: Optional[str]) -> str:
    role_l = (role or "").strip().lower()
    if role_l in {"viewer", "analyst", "admin", "owner"}:
        return role_l
    return "viewer"


def _gen_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


# ── Members ──────────────────────────────────────────────────────────────


async def list_members(
    pool: asyncpg.Pool, owner_user_id: int, project_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """All accepted members invited BY this owner. Optional project filter."""
    where = "WHERE m.invited_by = $1"
    params: list[Any] = [owner_user_id]
    if project_name:
        where += " AND m.project_name = $2"
        params.append(project_name)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT m.id, m.user_id, m.project_name, m.role,
                   m.invited_by, m.invited_at, m.accepted_at,
                   u.email, u.name
              FROM escalar_project_members m
              LEFT JOIN users u ON u.id = m.user_id
              {where}
             ORDER BY m.project_name, m.role, m.accepted_at DESC NULLS LAST, m.id
            """,
            *params,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "email": r["email"],
            "name": r["name"] or "",
            "project_name": r["project_name"],
            "role": r["role"],
            "invited_at": r["invited_at"].isoformat() if r["invited_at"] else None,
            "accepted_at": r["accepted_at"].isoformat() if r["accepted_at"] else None,
        })
    return out


async def list_my_memberships(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict[str, Any]]:
    """Projects this user has been added to (not the projects they own)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT m.id, m.project_name, m.role, m.invited_at, m.accepted_at,
                   u.email AS owner_email, u.name AS owner_name
              FROM escalar_project_members m
              LEFT JOIN users u ON u.id = m.invited_by
             WHERE m.user_id = $1 AND m.accepted_at IS NOT NULL
             ORDER BY m.project_name
            """,
            user_id,
        )
    return [
        {
            "id": r["id"],
            "project_name": r["project_name"],
            "role": r["role"],
            "owner_email": r["owner_email"],
            "owner_name": r["owner_name"] or "",
            "invited_at": r["invited_at"].isoformat() if r["invited_at"] else None,
            "accepted_at": r["accepted_at"].isoformat() if r["accepted_at"] else None,
        }
        for r in rows
    ]


async def update_role(
    pool: asyncpg.Pool, owner_user_id: int, member_id: int, new_role: str,
) -> dict[str, Any]:
    new_role = _normalize_role(new_role)
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE escalar_project_members
               SET role = $1, updated_at = NOW()
             WHERE id = $2 AND invited_by = $3
            """,
            new_role, member_id, owner_user_id,
        )
    return {"updated": result.endswith(" 1"), "id": member_id, "role": new_role}


async def remove_member(
    pool: asyncpg.Pool, owner_user_id: int, member_id: int,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM escalar_project_members
             WHERE id = $1 AND invited_by = $2
            """,
            member_id, owner_user_id,
        )
    return {"removed": result.endswith(" 1"), "id": member_id}


async def get_user_role_for_project(
    pool: asyncpg.Pool, user_id: int, project_name: str,
) -> Optional[str]:
    """Return user's role for a project, or None if no membership.
    Owner is implicit: caller decides if user owns the project (e.g. via
    `escalar_projects.owner_user_id` or by uploads ownership) — this function
    only checks the explicit membership table."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT role FROM escalar_project_members
             WHERE user_id = $1 AND project_name = $2 AND accepted_at IS NOT NULL
            """,
            user_id, project_name,
        )
    return row["role"] if row else None


# ── Invitations ──────────────────────────────────────────────────────────


async def list_pending_invitations(
    pool: asyncpg.Pool, owner_user_id: int,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, email, project_name, role, token,
                   created_at, expires_at, used_at, revoked_at
              FROM escalar_invitations
             WHERE invited_by = $1 AND used_at IS NULL AND revoked_at IS NULL
               AND expires_at > NOW()
             ORDER BY created_at DESC
            """,
            owner_user_id,
        )
    return [
        {
            "id": r["id"],
            "email": r["email"],
            "project_name": r["project_name"],
            "role": r["role"],
            "token": r["token"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
        }
        for r in rows
    ]


async def create_invitation(
    pool: asyncpg.Pool, owner_user_id: int, *,
    email: str, project_name: str, role: str = "viewer",
) -> dict[str, Any]:
    """Create a pending invitation. If the email already has an active
    invitation for this same project, return it instead of duplicating
    (idempotent re-invite)."""
    email_norm = (email or "").strip().lower()
    project_norm = (project_name or "").strip()
    role_norm = _normalize_role(role)

    if not email_norm or "@" not in email_norm:
        return {"error": "invalid_email"}
    if not project_norm:
        return {"error": "project_required"}
    if role_norm == "owner":
        return {"error": "cannot_invite_as_owner"}

    expires = datetime.now(timezone.utc) + timedelta(days=INVITE_TTL_DAYS)

    async with pool.acquire() as conn:
        # Same-project + same-email + still active → reuse
        existing = await conn.fetchrow(
            """
            SELECT id, token, expires_at FROM escalar_invitations
             WHERE invited_by = $1
               AND lower(email) = $2
               AND project_name = $3
               AND used_at IS NULL AND revoked_at IS NULL
               AND expires_at > NOW()
             ORDER BY created_at DESC
             LIMIT 1
            """,
            owner_user_id, email_norm, project_norm,
        )
        if existing:
            return {
                "id": existing["id"],
                "token": existing["token"],
                "email": email_norm,
                "project_name": project_norm,
                "role": role_norm,
                "expires_at": existing["expires_at"].isoformat(),
                "reused": True,
            }

        token = _gen_token()
        row = await conn.fetchrow(
            """
            INSERT INTO escalar_invitations
              (email, project_name, role, token, invited_by, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, expires_at
            """,
            email_norm, project_norm, role_norm, token, owner_user_id, expires,
        )

    return {
        "id": row["id"],
        "token": token,
        "email": email_norm,
        "project_name": project_norm,
        "role": role_norm,
        "expires_at": row["expires_at"].isoformat(),
        "reused": False,
    }


async def revoke_invitation(
    pool: asyncpg.Pool, owner_user_id: int, invitation_id: int,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE escalar_invitations
               SET revoked_at = NOW()
             WHERE id = $1 AND invited_by = $2
               AND used_at IS NULL AND revoked_at IS NULL
            """,
            invitation_id, owner_user_id,
        )
    return {"revoked": result.endswith(" 1"), "id": invitation_id}


async def get_invitation_by_token(
    pool: asyncpg.Pool, token: str,
) -> Optional[dict[str, Any]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT i.id, i.email, i.project_name, i.role, i.token,
                   i.invited_by, i.expires_at, i.used_at, i.revoked_at,
                   u.email AS inviter_email, u.name AS inviter_name
              FROM escalar_invitations i
              LEFT JOIN users u ON u.id = i.invited_by
             WHERE i.token = $1
            """,
            token,
        )
    if not row:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "project_name": row["project_name"],
        "role": row["role"],
        "token": row["token"],
        "invited_by": row["invited_by"],
        "inviter_email": row["inviter_email"],
        "inviter_name": row["inviter_name"] or "",
        "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
        "used_at": row["used_at"].isoformat() if row["used_at"] else None,
        "revoked_at": row["revoked_at"].isoformat() if row["revoked_at"] else None,
        "is_active": (
            row["used_at"] is None
            and row["revoked_at"] is None
            and (row["expires_at"] is None or row["expires_at"] > datetime.now(timezone.utc))
        ),
    }


async def accept_invitation(
    pool: asyncpg.Pool, token: str, accepting_user_id: int, accepting_email: str,
) -> dict[str, Any]:
    """Bind an invitation to the logged-in user. Validates that:
      - invitation exists, not used, not revoked, not expired
      - email match (case-insensitive) — invited email must equal user's email
    Then upserts a member row and marks the invitation used.
    """
    inv = await get_invitation_by_token(pool, token)
    if not inv:
        return {"error": "invitation_not_found"}
    if inv["used_at"]:
        return {"error": "invitation_already_used"}
    if inv["revoked_at"]:
        return {"error": "invitation_revoked"}
    if not inv["is_active"]:
        return {"error": "invitation_expired"}
    if (accepting_email or "").strip().lower() != (inv["email"] or "").strip().lower():
        return {"error": "email_mismatch"}
    if inv["invited_by"] == accepting_user_id:
        return {"error": "cannot_accept_own_invite"}

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO escalar_project_members
                  (user_id, project_name, role, invited_by, invited_at, accepted_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                ON CONFLICT (user_id, project_name)
                DO UPDATE SET role = EXCLUDED.role,
                              invited_by = EXCLUDED.invited_by,
                              accepted_at = NOW(),
                              updated_at = NOW()
                """,
                accepting_user_id, inv["project_name"], inv["role"], inv["invited_by"],
            )
            await conn.execute(
                """
                UPDATE escalar_invitations
                   SET used_at = NOW(), used_by_user_id = $1
                 WHERE id = $2
                """,
                accepting_user_id, inv["id"],
            )

    return {
        "accepted": True,
        "project_name": inv["project_name"],
        "role": inv["role"],
        "invited_by_email": inv["inviter_email"],
    }


# ── Permission helper ────────────────────────────────────────────────────


def role_has_at_least(role: Optional[str], minimum: str) -> bool:
    """Check if `role` is at least `minimum` in the role hierarchy.
    None counts as below all roles."""
    if not role:
        return False
    try:
        return ROLE_ORDER.index(role) >= ROLE_ORDER.index(minimum)
    except ValueError:
        return False
