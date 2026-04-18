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

from v2.deps import CurrentUser, current_user, get_pool
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
    return MLStatusOut(
        connected=not expired,
        hasCredentials=has_creds,
        mlUserId=tokens.get("ml_user_id"),
        nickname=tokens.get("ml_nickname"),
        siteId=tokens.get("ml_site_id"),
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
