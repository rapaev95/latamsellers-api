"""Managed ML accounts — shared OAuth tokens owned by LS Profit (not by an
individual end-user).

Why a separate table from `ml_user_tokens`:
- `ml_user_tokens` is 1:1 with `users.id` (each LS user owns exactly one ML
  account). Designed for Type B (self-service).
- Managed accounts (Type A) are LS-controlled credentials used to publish
  on behalf of clients who rented our cabinet. One LS-superadmin connects
  the account once; many LS managers can use it.

Auth model:
- `owner_ls_user_id` — who connected it (audit).
- Access control to read/use the token is enforced at the router level —
  for Sprint 1, only admins/superadmins.

OAuth flow (managed):
- /ml-oauth/managed/start → generates state, redirects to ML authorize URL.
- /ml-oauth/managed/callback → exchanges code, persists here via
  `save_account_from_payload`.
- Refresh logic mirrors ml_oauth.refresh_user_token but writes to this table.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc  # reuse ML_TOKEN_URL + MLRefreshError + load_app_config

log = logging.getLogger(__name__)

# Refresh when token expires within this window
SYNC_REFRESH_MARGIN = timedelta(minutes=10)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_managed_accounts (
  id                       SERIAL PRIMARY KEY,
  owner_ls_user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  ml_user_id               BIGINT NOT NULL UNIQUE,
  ml_nickname              TEXT,
  ml_site_id               TEXT NOT NULL DEFAULT 'MLB',
  access_token             TEXT NOT NULL,
  refresh_token            TEXT NOT NULL,
  access_token_expires_at  TIMESTAMPTZ NOT NULL,
  last_refreshed_at        TIMESTAMPTZ,
  scope                    TEXT,
  note                     TEXT,
  created_at               TIMESTAMPTZ DEFAULT NOW(),
  updated_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_managed_accounts_owner
  ON ml_managed_accounts(owner_ls_user_id);

CREATE INDEX IF NOT EXISTS idx_ml_managed_accounts_expires
  ON ml_managed_accounts(access_token_expires_at);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _row_to_dict(row: asyncpg.Record, *, expose_tokens: bool = False) -> dict[str, Any]:
    """Convert to dict; by default strips access_token/refresh_token from
    the response (they should never leak to the frontend listing UI)."""
    d = dict(row)
    if not expose_tokens:
        d.pop("access_token", None)
        d.pop("refresh_token", None)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

async def save_account_from_payload(
    pool: asyncpg.Pool,
    *,
    owner_ls_user_id: int,
    ml_user_id: int,
    access_token: str,
    refresh_token: str,
    expires_in: int,
    ml_nickname: Optional[str] = None,
    ml_site_id: str = "MLB",
    scope: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Insert or update a managed account row. Returns the row (no tokens)."""
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ml_managed_accounts (
                owner_ls_user_id, ml_user_id, ml_nickname, ml_site_id,
                access_token, refresh_token, access_token_expires_at,
                last_refreshed_at, scope, note, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8, $9, NOW())
            ON CONFLICT (ml_user_id) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                access_token_expires_at = EXCLUDED.access_token_expires_at,
                ml_nickname = COALESCE(EXCLUDED.ml_nickname, ml_managed_accounts.ml_nickname),
                ml_site_id = COALESCE(EXCLUDED.ml_site_id, ml_managed_accounts.ml_site_id),
                scope = COALESCE(EXCLUDED.scope, ml_managed_accounts.scope),
                note = COALESCE(EXCLUDED.note, ml_managed_accounts.note),
                last_refreshed_at = NOW(),
                updated_at = NOW()
            RETURNING id, owner_ls_user_id, ml_user_id, ml_nickname, ml_site_id,
                      access_token_expires_at, last_refreshed_at, scope, note,
                      created_at, updated_at
            """,
            owner_ls_user_id, ml_user_id, ml_nickname, ml_site_id,
            access_token, refresh_token, expires_at, scope, note,
        )
    return dict(row)


async def list_accounts(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """List all managed accounts. Tokens are NOT exposed."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, owner_ls_user_id, ml_user_id, ml_nickname, ml_site_id,
                   access_token_expires_at, last_refreshed_at, scope, note,
                   created_at, updated_at
            FROM ml_managed_accounts
            ORDER BY created_at DESC
            """,
        )
    return [dict(r) for r in rows]


async def get_account(
    pool: asyncpg.Pool, account_id: int, *, with_tokens: bool = False
) -> Optional[dict[str, Any]]:
    fields = (
        "id, owner_ls_user_id, ml_user_id, ml_nickname, ml_site_id, "
        "access_token, refresh_token, access_token_expires_at, "
        "last_refreshed_at, scope, note, created_at, updated_at"
        if with_tokens else
        "id, owner_ls_user_id, ml_user_id, ml_nickname, ml_site_id, "
        "access_token_expires_at, last_refreshed_at, scope, note, "
        "created_at, updated_at"
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT {fields} FROM ml_managed_accounts WHERE id = $1",
            account_id,
        )
    return dict(row) if row else None


async def delete_account(pool: asyncpg.Pool, account_id: int) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM ml_managed_accounts WHERE id = $1", account_id,
        )
    return result.endswith(" 1")


# ──────────────────────────────────────────────────────────────────────────────
# OAuth code exchange — Next.js admin page handles redirect/callback,
# then calls us with (code, codeVerifier, redirectUri).
# ──────────────────────────────────────────────────────────────────────────────

async def exchange_authorization_code_for_managed(
    pool: asyncpg.Pool,
    *,
    owner_ls_user_id: int,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Exchange an OAuth authorization code (PKCE) for tokens and persist
    them into ml_managed_accounts. Returns the saved row.

    Reuses client_id/client_secret from `ml_app_config` (singleton shared
    with personal OAuth — the ML app is the same).
    """
    cfg = await ml_oauth_svc.load_app_config(pool)
    client_id = cfg.get("client_id")
    client_secret = cfg.get("client_secret")
    if not client_id or not client_secret:
        raise ml_oauth_svc.MLRefreshError("no_app_credentials")

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            ml_oauth_svc.ML_TOKEN_URL,
            headers={
                "accept": "application/json",
                "content-type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
    try:
        payload = r.json()
    except ValueError:
        raise ml_oauth_svc.MLRefreshError(f"non_json: HTTP {r.status_code}")
    if r.status_code != 200:
        err = payload.get("error") or f"http_{r.status_code}"
        desc = payload.get("error_description", "")
        raise ml_oauth_svc.MLRefreshError(f"{err}: {desc}")

    ml_user_id = payload.get("user_id")
    if not ml_user_id:
        raise ml_oauth_svc.MLRefreshError("ml_payload_missing_user_id")

    # Fetch nickname so the admin UI can show something human.
    nickname = None
    site_id = "MLB"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            ur = await client.get(
                f"https://api.mercadolibre.com/users/{ml_user_id}",
                headers={"Authorization": f"Bearer {payload['access_token']}"},
            )
        if ur.status_code == 200:
            up = ur.json() or {}
            nickname = up.get("nickname")
            site_id = up.get("site_id") or site_id
    except Exception as err:  # noqa: BLE001
        log.warning("managed_oauth: nickname lookup failed for ml_user_id=%s: %s",
                    ml_user_id, err)

    return await save_account_from_payload(
        pool,
        owner_ls_user_id=owner_ls_user_id,
        ml_user_id=int(ml_user_id),
        access_token=payload["access_token"],
        refresh_token=payload["refresh_token"],
        expires_in=int(payload.get("expires_in", 21600)),
        ml_nickname=nickname,
        ml_site_id=site_id,
        scope=payload.get("scope"),
        note=note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Token refresh — mirrors ml_oauth.refresh_user_token
# ──────────────────────────────────────────────────────────────────────────────

async def _refresh_account_token(pool: asyncpg.Pool, account_id: int) -> dict:
    """Refresh tokens for a managed account. Returns the new ML payload."""
    cfg = await ml_oauth_svc.load_app_config(pool)
    client_id = cfg.get("client_id")
    client_secret = cfg.get("client_secret")
    if not client_id or not client_secret:
        raise ml_oauth_svc.MLRefreshError("no_app_credentials")

    account = await get_account(pool, account_id, with_tokens=True)
    if not account:
        raise ml_oauth_svc.MLRefreshError("no_managed_account")

    # Advisory lock — keyed by ml_user_id to keep parallel callers serial.
    async with pool.acquire() as conn:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtext('ml_refresh_managed_' || $1))",
            str(account["ml_user_id"]),
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                ml_oauth_svc.ML_TOKEN_URL,
                headers={
                    "accept": "application/json",
                    "content-type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": account["refresh_token"],
                },
            )

    try:
        payload = r.json()
    except ValueError:
        raise ml_oauth_svc.MLRefreshError(f"non_json: HTTP {r.status_code}")

    if r.status_code != 200:
        err = payload.get("error") or f"http_{r.status_code}"
        desc = payload.get("error_description", "")
        raise ml_oauth_svc.MLRefreshError(f"{err}: {desc}")

    await save_account_from_payload(
        pool,
        owner_ls_user_id=account["owner_ls_user_id"],
        ml_user_id=account["ml_user_id"],
        access_token=payload["access_token"],
        refresh_token=payload["refresh_token"],
        expires_in=int(payload.get("expires_in", 21600)),
        scope=payload.get("scope"),
    )
    return payload


async def get_valid_access_token(
    pool: asyncpg.Pool, account_id: int
) -> tuple[str, datetime, bool]:
    """Returns (access_token, expires_at, refreshed_now).
    Refreshes synchronously if expiry is within SYNC_REFRESH_MARGIN.
    """
    account = await get_account(pool, account_id, with_tokens=True)
    if not account:
        raise ml_oauth_svc.MLRefreshError("no_managed_account")

    now = datetime.now(timezone.utc)
    expires_at = account["access_token_expires_at"]
    refreshed = False

    if expires_at <= now + SYNC_REFRESH_MARGIN:
        await _refresh_account_token(pool, account_id)
        account = await get_account(pool, account_id, with_tokens=True)
        if not account:
            raise ml_oauth_svc.MLRefreshError("account_disappeared")
        expires_at = account["access_token_expires_at"]
        refreshed = True

    return account["access_token"], expires_at, refreshed
