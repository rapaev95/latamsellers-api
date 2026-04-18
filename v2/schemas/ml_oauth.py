"""Pydantic models for /api/v2/ml-oauth/*."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class MLConfigIn(BaseModel):
    """POST /ml-oauth/config — save app credentials."""
    clientId: Optional[str] = Field(None, min_length=1)
    clientSecret: Optional[str] = Field(None, min_length=1)
    redirectUri: Optional[str] = Field(None, min_length=1)


class MLConfigOut(BaseModel):
    """GET /ml-oauth/config — masked view of app credentials."""
    clientId: str = ""
    clientSecretSet: bool = False
    clientSecretPreview: str = ""
    redirectUri: str = ""
    source: dict  # {clientId: 'db'|'none', clientSecret: ..., redirectUri: ...}


class MLTokensIn(BaseModel):
    """POST /ml-oauth/tokens — called by Next.js after OAuth callback."""
    accessToken: str
    refreshToken: str
    expiresIn: int  # seconds until expiry (6h typical)
    mlUserId: Optional[int] = None
    nickname: Optional[str] = None
    siteId: Optional[str] = None
    scope: Optional[str] = None


class MLExchangeCodeIn(BaseModel):
    """POST /ml-oauth/exchange-code — server-side code→token exchange.

    Lets Next.js perform the OAuth callback without ever touching client_secret.
    """
    code: str = Field(..., min_length=1)
    codeVerifier: str = Field(..., min_length=1)
    redirectUri: str = Field(..., min_length=1)


class MLExchangeCodeOut(BaseModel):
    saved: bool
    mlUserId: Optional[int] = None
    scope: Optional[str] = None
    expiresIn: int


class MLStatusOut(BaseModel):
    """GET /ml-oauth/status — connection state for current user."""
    connected: bool
    hasCredentials: bool
    mlUserId: Optional[int] = None
    nickname: Optional[str] = None
    siteId: Optional[str] = None
    expiresAt: Optional[datetime] = None
    lastRefreshedAt: Optional[datetime] = None
    message: Optional[str] = None


class MLAccessTokenOut(BaseModel):
    """GET /ml-oauth/access-token — fresh access_token for current user."""
    accessToken: str
    expiresAt: datetime
    refreshed: bool  # True if we refreshed synchronously on this request


class MLRefreshOut(BaseModel):
    """POST /ml-oauth/refresh — refresh result."""
    refreshed: int  # number of users refreshed
    failed: int
    details: list[dict] = []


class MLSaveResult(BaseModel):
    saved: bool
    config: MLConfigOut


class MLDeleteResult(BaseModel):
    deleted: bool
