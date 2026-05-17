"""ML OAuth endpoints: /api/v2/ml-oauth/*.

Ownership model:
- App credentials (client_id / client_secret / redirect_uri) — shared singleton in `ml_app_config`.
- Tokens (access/refresh) — per-user in `ml_user_tokens`, keyed by users.id.

OAuth initiation + callback remain in Next.js; this service only stores + refreshes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from v2.deps import CurrentUser, current_user, get_pool, require_admin, require_superadmin
from v2.schemas.ml_oauth import (
    MLAccessTokenOut,
    MLConfigIn,
    MLConfigOut,
    MLDeleteResult,
    MLExchangeCodeIn,
    MLExchangeCodeOut,
    MLRefreshOut,
    MLSaveResult,
    MLStatusOut,
    MLTokensIn,
)
from v2.services import ml_oauth as ml_svc

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-oauth", tags=["ml-oauth"])


# ──────────────────────────────────────────────────────────────────────────────
# App credentials (shared singleton)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/config", response_model=MLConfigOut)
async def get_config(
    _user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Return masked app credentials."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)
    cfg = await ml_svc.load_app_config(pool)
    return ml_svc.mask_config(cfg)


@router.post("/config", response_model=MLSaveResult)
async def post_config(
    body: MLConfigIn,
    _user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Save non-None fields to ml_app_config (singleton)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)

    if not (body.clientId or body.clientSecret or body.redirectUri):
        raise HTTPException(
            status_code=400,
            detail="no_fields — send at least one of clientId, clientSecret, redirectUri",
        )

    await ml_svc.save_app_config(
        pool,
        client_id=body.clientId,
        client_secret=body.clientSecret,
        redirect_uri=body.redirectUri,
    )
    cfg = await ml_svc.load_app_config(pool)
    return {"saved": True, "config": ml_svc.mask_config(cfg)}


# ──────────────────────────────────────────────────────────────────────────────
# User tokens
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=MLStatusOut)
async def get_status(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Connection status for the current user."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)

    cfg = await ml_svc.load_app_config(pool)
    has_creds = bool(cfg.get("client_id") and cfg.get("client_secret"))

    tokens = await ml_svc.load_user_tokens(pool, user.id)
    if not tokens:
        return MLStatusOut(
            connected=False,
            hasCredentials=has_creds,
            message=(
                "Credentials configured, but user not authorized yet"
                if has_creds
                else "No app credentials — save client_id/secret first"
            ),
        )

    now = datetime.now(timezone.utc)
    expired = tokens["access_token_expires_at"] <= now

    # nickname/site_id may be null on ml_user_tokens — OAuth callbacks
    # before we started persisting these fields, plus some refresh paths
    # don't carry the nickname. Account-health table caches the same
    # values from /users/me with a 6h TTL, so fall back to it. Without
    # this, ml-status reads as "connected: true, nickname: null" which
    # confuses callers expecting a display name.
    nickname = tokens.get("ml_nickname")
    site_id = tokens.get("ml_site_id")
    if not nickname or not site_id:
        async with pool.acquire() as conn:
            ah = await conn.fetchrow(
                "SELECT nickname, site_id FROM ml_account_health WHERE user_id = $1",
                user.id,
            )
        if ah:
            nickname = nickname or ah["nickname"]
            site_id = site_id or ah["site_id"]

    return MLStatusOut(
        connected=not expired,
        hasCredentials=has_creds,
        mlUserId=tokens.get("ml_user_id"),
        nickname=nickname,
        siteId=site_id,
        expiresAt=tokens.get("access_token_expires_at"),
        lastRefreshedAt=tokens.get("last_refreshed_at"),
        message="token_expired" if expired else None,
    )


@router.post("/exchange-code", response_model=MLExchangeCodeOut)
async def post_exchange_code(
    body: MLExchangeCodeIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Server-side PKCE code → token exchange. Preferred over /tokens because
    client_secret never leaves FastAPI."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)
    try:
        payload = await ml_svc.exchange_authorization_code(
            pool, user.id,
            code=body.code,
            code_verifier=body.codeVerifier,
            redirect_uri=body.redirectUri,
        )
    except ml_svc.MLRefreshError as err:
        raise HTTPException(status_code=400, detail=str(err))

    return MLExchangeCodeOut(
        saved=True,
        mlUserId=payload.get("user_id"),
        scope=payload.get("scope"),
        expiresIn=int(payload.get("expires_in", 21600)),
    )


@router.post("/tokens", response_model=MLStatusOut)
async def post_tokens(
    body: MLTokensIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manual token upload. Deprecated in favor of /exchange-code but kept for
    migration/debug. Called by Next.js OAuth callback only if exchange-code
    endpoint isn't available."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)

    await ml_svc.save_user_tokens(
        pool, user.id,
        access_token=body.accessToken,
        refresh_token=body.refreshToken,
        expires_in=body.expiresIn,
        ml_user_id=body.mlUserId,
        nickname=body.nickname,
        site_id=body.siteId,
        scope=body.scope,
    )
    log.info("[ml-oauth] Tokens saved for user_id=%s (ml_user_id=%s)", user.id, body.mlUserId)

    # Return status inline so the caller doesn't need a second round-trip
    cfg = await ml_svc.load_app_config(pool)
    tokens = await ml_svc.load_user_tokens(pool, user.id)
    if tokens is None:
        raise HTTPException(status_code=500, detail="tokens_not_persisted")
    return MLStatusOut(
        connected=True,
        hasCredentials=bool(cfg.get("client_id") and cfg.get("client_secret")),
        mlUserId=tokens.get("ml_user_id"),
        nickname=tokens.get("ml_nickname"),
        siteId=tokens.get("ml_site_id"),
        expiresAt=tokens.get("access_token_expires_at"),
        lastRefreshedAt=tokens.get("last_refreshed_at"),
    )


@router.get("/access-token", response_model=MLAccessTokenOut)
async def get_access_token(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Return a valid access_token, refreshing synchronously if expiring soon."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)
    try:
        token, expires_at, refreshed = await ml_svc.get_valid_access_token(pool, user.id)
    except ml_svc.MLRefreshError as err:
        # invalid_grant → wipe stale tokens so UI shows disconnected
        if "invalid_grant" in str(err):
            await ml_svc.delete_user_tokens(pool, user.id)
        raise HTTPException(status_code=401, detail=str(err))
    return MLAccessTokenOut(
        accessToken=token,
        expiresAt=expires_at,
        refreshed=refreshed,
    )


@router.post("/refresh", response_model=MLRefreshOut)
async def post_refresh(
    all_users: bool = Query(False, alias="all"),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Manually trigger refresh. `?all=true` refreshes every expiring user (admin-ish, but
    currently any authenticated user can call it — tighten before prod if needed)."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)

    if all_users:
        result = await ml_svc.refresh_all_expiring_tokens(pool)
        return MLRefreshOut(**result)

    # Refresh only current user
    try:
        await ml_svc.refresh_user_token(pool, user.id)
        return MLRefreshOut(refreshed=1, failed=0, details=[{"user_id": user.id, "status": "ok"}])
    except ml_svc.MLRefreshError as err:
        if "invalid_grant" in str(err):
            await ml_svc.delete_user_tokens(pool, user.id)
        return MLRefreshOut(
            refreshed=0, failed=1,
            details=[{"user_id": user.id, "status": "failed", "error": str(err)}],
        )


@router.delete("/tokens", response_model=MLDeleteResult)
async def delete_tokens(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Disconnect the current user's ML account."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    await ml_svc.ensure_schema(pool)
    deleted = await ml_svc.delete_user_tokens(pool, user.id)
    return MLDeleteResult(deleted=deleted)


# ──────────────────────────────────────────────────────────────────────────────
# Managed accounts (Type A — Sprint 1)
# Same OAuth code-exchange contract as personal /exchange-code, but the result
# lands in `ml_managed_accounts` instead of `ml_user_tokens`. The Next.js admin
# page reuses its own OAuth callback flow (state + PKCE) and just routes to a
# different exchange endpoint when the OAuth was initiated as «managed».
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/managed/exchange-code")
async def post_managed_exchange_code(
    body: MLExchangeCodeIn,
    note: Optional[str] = Query(None, max_length=200),
    user: CurrentUser = Depends(require_superadmin),
    pool=Depends(get_pool),
):
    """Server-side PKCE exchange that lands tokens into ml_managed_accounts.

    Superadmin-only — managed credentials power Type A clients and can
    publish on behalf of multiple LS managers, so the install gesture
    is owner-scope.

    Body: { code, codeVerifier, redirectUri }  (same as personal flow).
    Query: ?note=Optional+human+label  (e.g. "Ganza", "Carlos Al").
    """
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    from v2.services import ml_managed_accounts as _mgr
    await ml_svc.ensure_schema(pool)
    await _mgr.ensure_schema(pool)

    try:
        row = await _mgr.exchange_authorization_code_for_managed(
            pool,
            owner_ls_user_id=user.id,
            code=body.code,
            code_verifier=body.codeVerifier,
            redirect_uri=body.redirectUri,
            note=note,
        )
    except ml_svc.MLRefreshError as err:
        raise HTTPException(status_code=400, detail={"error": "ml_exchange_failed",
                                                     "message": str(err)})
    return {"ok": True, "account": row}


@router.get("/managed/accounts")
async def get_managed_accounts(
    user: CurrentUser = Depends(require_admin),
    pool=Depends(get_pool),
):
    """List managed ML accounts (no tokens). Admin+superadmin can view —
    so an admin can pick which account to act as in ProjectFilter."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    from v2.services import ml_managed_accounts as _mgr
    await _mgr.ensure_schema(pool)
    rows = await _mgr.list_accounts(pool)
    return {"accounts": rows}


@router.delete("/managed/accounts/{account_id}")
async def delete_managed_account(
    account_id: int,
    user: CurrentUser = Depends(require_superadmin),
    pool=Depends(get_pool),
):
    """Disconnect a managed account (hard delete row). Superadmin-only."""
    if pool is None:
        raise HTTPException(status_code=503, detail="no_db")
    from v2.services import ml_managed_accounts as _mgr
    await _mgr.ensure_schema(pool)
    deleted = await _mgr.delete_account(pool, account_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"error": "account_not_found"})
    return {"ok": True}
