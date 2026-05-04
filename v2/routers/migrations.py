"""Recovery endpoints for schema migrations.

Each migration is run automatically on startup, but startup failures (Railway
container crash before the bootstrap finishes, transient DB unavailability,
etc.) leave the schema half-migrated. These endpoints let a superadmin
re-trigger any single migration on demand without redeploying.

All migrations are idempotent — calling them on a fully-migrated DB is a no-op.
Superadmin-only. Returns the service name and a structured `ok=true` on
success, otherwise raises with the underlying SQL error in `detail`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from v2.db import get_pool
from v2.deps import require_superadmin

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/_migrate",
    tags=["migrations"],
    dependencies=[Depends(require_superadmin)],
)


@router.post("/project_members")
async def migrate_project_members(pool=Depends(get_pool)) -> dict:
    """Rename `escalar_project_members` → `project_members` and
    `escalar_invitations` → `project_invitations`, plus add the
    `effective_from` column. Idempotent."""
    from v2.services import project_members as svc
    try:
        await svc.ensure_schema(pool)
    except Exception as err:
        log.exception("project_members migration failed")
        raise HTTPException(status_code=500, detail={"step": "project_members", "error": str(err)})
    return {"ok": True, "service": "project_members"}
