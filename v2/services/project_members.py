"""Project ACL — invite team members to specific projects with a role.

Two tables:
  project_members      — accepted memberships (user_id × project_name × role)
  project_invitations  — pending email invitations with one-time token

Roles: owner | admin | analyst | viewer
  owner    — full access, only the user who registered the project (implicit)
  admin    — full access except member management + billing
  analyst  — read-write on positions/AB tests, read-only elsewhere
  viewer   — read-only across all features

Owner-membership is implicit: the user who created/uploaded a project is its
owner; we do NOT insert a row for the owner. Membership rows describe added
collaborators only.

`effective_from` — the cut-off timestamp for delete operations. A non-owner
member can only delete uploads/records whose `created_at >= effective_from`.
Set to `accepted_at` on accept; nullable for legacy rows (NULL = no cut-off,
member can delete anything within their role).

Token format: 32 url-safe bytes (43 chars), single-use, expires in 7 days.

Migration history:
  v1: tables `escalar_project_members` + `escalar_invitations`
  v2: renamed to `project_members` + `project_invitations`,
      added `effective_from` column.
  ensure_schema() handles the rename idempotently — safe to run repeatedly.
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

# ── Migration: rename old tables, add effective_from ──────────────────────
# Each statement is idempotent. `to_regclass` returns NULL if the relation
# doesn't exist, so we can branch without EXCEPTION blocks.
_MIGRATION_STATEMENTS = [
    # 1. Rename project_members table if it still has the old escalar-prefixed name.
    """
    DO $migrate$
    BEGIN
      IF to_regclass('public.escalar_project_members') IS NOT NULL THEN
        IF to_regclass('public.project_members') IS NOT NULL THEN
          RAISE EXCEPTION 'Both escalar_project_members and project_members exist; manual merge required';
        END IF;
        ALTER TABLE escalar_project_members RENAME TO project_members;
      END IF;
    END
    $migrate$;
    """,
    # 2. Rename invitations table.
    """
    DO $migrate$
    BEGIN
      IF to_regclass('public.escalar_invitations') IS NOT NULL THEN
        IF to_regclass('public.project_invitations') IS NOT NULL THEN
          RAISE EXCEPTION 'Both escalar_invitations and project_invitations exist; manual merge required';
        END IF;
        ALTER TABLE escalar_invitations RENAME TO project_invitations;
      END IF;
    END
    $migrate$;
    """,
    # 3. Rename old indexes to match new table prefix (no-op if they already
    # have the new name — IF EXISTS handles that without EXCEPTION blocks).
    "ALTER INDEX IF EXISTS idx_escalar_members_user RENAME TO idx_project_members_user",
    "ALTER INDEX IF EXISTS idx_escalar_members_project RENAME TO idx_project_members_project",
    "ALTER INDEX IF EXISTS idx_escalar_invites_email_pending RENAME TO idx_project_invites_email_pending",
    "ALTER INDEX IF EXISTS idx_escalar_invites_inviter RENAME TO idx_project_invites_inviter",
    "ALTER INDEX IF EXISTS idx_escalar_invites_token RENAME TO idx_project_invites_token",
]

_CREATE_STATEMENTS = [
    # Create with new names if absent (fresh installs).
    """
    CREATE TABLE IF NOT EXISTS project_members (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL,
      project_name TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'viewer',
      invited_by INTEGER,
      invited_at TIMESTAMPTZ DEFAULT NOW(),
      accepted_at TIMESTAMPTZ,
      effective_from TIMESTAMPTZ,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE(user_id, project_name)
    )
    """,
    # Add column for tables that existed before the migration introduced it.
    "ALTER TABLE project_members ADD COLUMN IF NOT EXISTS effective_from TIMESTAMPTZ",
    # Backfill: legacy rows get effective_from = accepted_at so existing
    # members can delete anything they could before the schema change.
    """
    UPDATE project_members
       SET effective_from = accepted_at
     WHERE effective_from IS NULL AND accepted_at IS NOT NULL
    """,
    "CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_project_members_project ON project_members(project_name)",
    """
    CREATE TABLE IF NOT EXISTS project_invitations (
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
    "CREATE INDEX IF NOT EXISTS idx_project_invites_email_pending ON project_invitations(lower(email)) WHERE used_at IS NULL AND revoked_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_project_invites_inviter ON project_invitations(invited_by)",
    "CREATE INDEX IF NOT EXISTS idx_project_invites_token ON project_invitations(token)",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Idempotent: runs migrations first, then ensures fresh schema exists.

    Safe to call from startup AND per-endpoint (belt-and-suspenders pattern):
    every statement uses IF NOT EXISTS / IF EXISTS or branches on to_regclass(),
    so repeated calls do nothing on a fully-migrated database.
    """
    async with pool.acquire() as conn:
        for stmt in _MIGRATION_STATEMENTS:
            await conn.execute(stmt)
        for stmt in _CREATE_STATEMENTS:
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
                   m.invited_by, m.invited_at, m.accepted_at, m.effective_from,
                   u.email, u.name
              FROM project_members m
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
            "effective_from": r["effective_from"].isoformat() if r["effective_from"] else None,
        })
    return out


async def list_my_memberships(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict[str, Any]]:
    """Projects this user has been added to (not the projects they own)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT m.id, m.project_name, m.role,
                   m.invited_at, m.accepted_at, m.effective_from,
                   u.email AS owner_email, u.name AS owner_name
              FROM project_members m
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
            "effective_from": r["effective_from"].isoformat() if r["effective_from"] else None,
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
            UPDATE project_members
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
            DELETE FROM project_members
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
            SELECT role FROM project_members
             WHERE user_id = $1 AND project_name = $2 AND accepted_at IS NOT NULL
            """,
            user_id, project_name,
        )
    return row["role"] if row else None


async def get_membership(
    pool: asyncpg.Pool, user_id: int, project_name: str,
) -> Optional[dict[str, Any]]:
    """Full membership row for a user/project, or None.
    Used by delete-time gates that need both `role` and `effective_from`.
    Returns only accepted memberships."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, project_name, role,
                   invited_by, invited_at, accepted_at, effective_from
              FROM project_members
             WHERE user_id = $1 AND project_name = $2 AND accepted_at IS NOT NULL
            """,
            user_id, project_name,
        )
    if row is None:
        return None
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "project_name": row["project_name"],
        "role": row["role"],
        "invited_by": row["invited_by"],
        "invited_at": row["invited_at"],
        "accepted_at": row["accepted_at"],
        "effective_from": row["effective_from"],
    }


# ── Invitations ──────────────────────────────────────────────────────────


async def list_pending_invitations(
    pool: asyncpg.Pool, owner_user_id: int,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, email, project_name, role, token,
                   created_at, expires_at, used_at, revoked_at
              FROM project_invitations
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
            SELECT id, token, expires_at FROM project_invitations
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
            INSERT INTO project_invitations
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
            UPDATE project_invitations
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
              FROM project_invitations i
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

    Sets `effective_from = NOW()` — this is the cut-off the delete-time gate
    uses to decide if the member can remove a given record (uploads/entries
    older than this stay read-only for the member).
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
                INSERT INTO project_members
                  (user_id, project_name, role, invited_by,
                   invited_at, accepted_at, effective_from)
                VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW())
                ON CONFLICT (user_id, project_name)
                DO UPDATE SET role = EXCLUDED.role,
                              invited_by = EXCLUDED.invited_by,
                              accepted_at = NOW(),
                              effective_from = NOW(),
                              updated_at = NOW()
                """,
                accepting_user_id, inv["project_name"], inv["role"], inv["invited_by"],
            )
            await conn.execute(
                """
                UPDATE project_invitations
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


# ── Effective tiers (paywall inheritance) ────────────────────────────────


async def get_visible_user_ids(
    pool: asyncpg.Pool, caller_id: int, project_name: str,
) -> list[int]:
    """Whose data is visible to `caller_id` within `project_name`.

    Always includes the caller's own user_id. If the caller is an accepted
    member of the project, also includes every owner who invited them
    (project_members.invited_by). Members can in theory be invited by
    multiple owners to the same project key — we return all distinct
    inviter ids.

    Used by read-routes that want to scope queries to "rows visible to me
    within this project" — e.g. `WHERE user_id = ANY($visible_ids)`.

    Caller-only result (no membership) is intentional — non-members and
    non-owners get just their own uploads; the existing user_id-scoping
    keeps tenant isolation.
    """
    visible: list[int] = [caller_id]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT invited_by
              FROM project_members
             WHERE user_id = $1
               AND project_name = $2
               AND accepted_at IS NOT NULL
               AND invited_by IS NOT NULL
            """,
            caller_id, project_name,
        )
    for r in rows:
        owner_id = r["invited_by"]
        if owner_id and owner_id not in visible:
            visible.append(owner_id)
    return visible


async def get_effective_tiers(pool: asyncpg.Pool, user_id: int) -> list[str]:
    """All paywall tiers this user effectively has access to.

    Effective = own user.tiers ∪ tiers inherited from project memberships
    (the inviter/owner of each project the user is a member of contributes
    their tiers to the member's effective set).

    Rationale: an invited collaborator should see the paywall-gated sections
    that the project owner has paid for. Without inheritance the invited user
    has tiers=[] and PaywallGate blocks them out of every section, making
    the team feature useless.

    Note: this is a paywall-feature gate, not a data-scope gate. Owner-only
    data filters (uploads.user_id, etc.) still restrict what the member
    actually sees inside a section.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH own_tiers AS (
              SELECT jsonb_array_elements_text(COALESCE(tiers, '[]'::jsonb)) AS t
                FROM users
               WHERE id = $1
            ),
            inherited AS (
              SELECT jsonb_array_elements_text(COALESCE(u.tiers, '[]'::jsonb)) AS t
                FROM project_members m
                JOIN users u ON u.id = m.invited_by
               WHERE m.user_id = $1 AND m.accepted_at IS NOT NULL
            )
            SELECT DISTINCT t FROM (
              SELECT t FROM own_tiers
              UNION ALL
              SELECT t FROM inherited
            ) AS combined
             WHERE t IS NOT NULL AND t <> ''
            """,
            user_id,
        )
    return [r["t"] for r in rows]


# ── Permission helpers ───────────────────────────────────────────────────


def role_has_at_least(role: Optional[str], minimum: str) -> bool:
    """Check if `role` is at least `minimum` in the role hierarchy.
    None counts as below all roles."""
    if not role:
        return False
    try:
        return ROLE_ORDER.index(role) >= ROLE_ORDER.index(minimum)
    except ValueError:
        return False


def can_delete_record(
    record_created_at: Optional[datetime],
    membership: Optional[dict[str, Any]],
) -> bool:
    """True if a member with this membership row can delete a record created at
    `record_created_at`. Owners (membership=None) bypass — they can always delete.

    Rule: members can only delete records created at or after their
    `effective_from`. NULL effective_from means no cut-off (legacy rows).
    NULL record_created_at conservatively returns False.
    """
    # Owner / no membership row passed = caller is the owner (implicit).
    if membership is None:
        return True
    eff = membership.get("effective_from")
    if eff is None:
        # Legacy member without cut-off — same behavior as before this feature.
        return True
    if record_created_at is None:
        return False
    return record_created_at >= eff


async def enforce_caller_can_delete(
    pool: asyncpg.Pool,
    *,
    caller_id: int,
    caller_role: str,
    record_owner_id: Optional[int],
    record_project_name: Optional[str],
    record_created_at: Optional[datetime],
) -> None:
    """Generic delete-time gate for project-scoped records (uploads,
    manual_usd_inflows, planned_payments, …). Raises HTTPException if the
    caller is not allowed to delete this record. Returns None on success.

    Decision tree:
      1. Caller is the record owner (record_owner_id == caller_id) → allow.
      2. Caller is admin/superadmin → allow.
      3. Record has no project_name (legacy) → only owner could delete it
         (covered by #1) — anyone else gets 403.
      4. Caller is a member of the record's project:
         a. role=viewer → 403 (viewer is read-only).
         b. role=analyst/admin → check effective_from cut-off.
      5. Caller has no relationship to the record → 403.

    Raising here (instead of returning bool) keeps route handlers compact —
    the handler only writes the happy path after `await enforce_caller_can_delete(...)`.
    """
    from fastapi import HTTPException, status

    # 1. Owner bypass.
    if record_owner_id is not None and record_owner_id == caller_id:
        return

    # 2. Admin / superadmin bypass.
    if caller_role in ("admin", "superadmin"):
        return

    # 3. Legacy record (no project) — non-owners cannot touch it.
    if not record_project_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "not_authorized", "reason": "legacy_record_no_project"},
        )

    # 4. Member path.
    membership = await get_membership(pool, caller_id, record_project_name)
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "not_a_project_member", "project": record_project_name},
        )
    if membership["role"] == "viewer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "viewer_cannot_delete"},
        )
    if not can_delete_record(record_created_at, membership):
        eff = membership.get("effective_from")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "before_membership_cutoff",
                "effective_from": eff.isoformat() if eff else None,
                "record_created_at": record_created_at.isoformat() if record_created_at else None,
            },
        )
