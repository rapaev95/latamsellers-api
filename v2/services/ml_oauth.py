"""ML OAuth token management: storage + refresh + schema bootstrap.

Tables (auto-created on first pool acquisition):
- `ml_app_config` — singleton row with shared client_id / client_secret / redirect_uri
- `ml_user_tokens` — per-user tokens (FK to users.id), refreshed before 6h expiry.

Token refresh:
- Sync: when /access-token is called and expires_at < now + 10 min → refresh on-the-fly
- Async: APScheduler every 5h wakes up and refreshes all expiring tokens
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

# Mercado Libre OAuth token endpoint (generic — works for .com/.com.br/etc.)
ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"

# Refresh when token expires within this window (sync refresh on /access-token)
SYNC_REFRESH_MARGIN = timedelta(minutes=10)

# Scheduler picks up tokens expiring within this window (catches everything before 6h)
SCHEDULER_REFRESH_MARGIN = timedelta(minutes=30)


# ──────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ──────────────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS ml_app_config (
  id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  client_id TEXT,
  client_secret TEXT,
  redirect_uri TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO ml_app_config (id) VALUES (1) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS ml_user_tokens (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  access_token TEXT NOT NULL,
  refresh_token TEXT NOT NULL,
  access_token_expires_at TIMESTAMPTZ NOT NULL,
  ml_user_id BIGINT,
  ml_nickname TEXT,
  ml_site_id TEXT,
  scope TEXT,
  last_refreshed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ml_user_tokens_expires_idx
  ON ml_user_tokens (access_token_expires_at)
  WHERE refresh_token IS NOT NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create ML OAuth tables if missing. Safe to call on every startup."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)


# ──────────────────────────────────────────────────────────────────────────────
# App config (client_id / client_secret / redirect_uri)
# ──────────────────────────────────────────────────────────────────────────────

async def load_app_config(pool: asyncpg.Pool) -> dict:
    """Return {client_id, client_secret, redirect_uri} — may contain None values."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT client_id, client_secret, redirect_uri FROM ml_app_config WHERE id = 1"
        )
    if row is None:
        return {"client_id": None, "client_secret": None, "redirect_uri": None}
    return dict(row)


async def save_app_config(
    pool: asyncpg.Pool,
    *,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    redirect_uri: Optional[str] = None,
) -> None:
    """Update only non-None fields on the singleton row (id=1).

    `ensure_schema()` guarantees the row exists, so plain UPDATE works.
    """
    sets: list[str] = []
    args: list[Any] = []
    if client_id is not None:
        args.append(client_id)
        sets.append(f"client_id = ${len(args)}")
    if client_secret is not None:
        args.append(client_secret)
        sets.append(f"client_secret = ${len(args)}")
    if redirect_uri is not None:
        args.append(redirect_uri)
        sets.append(f"redirect_uri = ${len(args)}")

    if not sets:
        return

    sets.append("updated_at = NOW()")
    sql = f"UPDATE ml_app_config SET {', '.join(sets)} WHERE id = 1"
    async with pool.acquire() as conn:
        await conn.execute(sql, *args)


def mask_config(cfg: dict) -> dict:
    """Build frontend-safe view of app config.

    redirect_uri falls back to env ML_REDIRECT_URI then to a hardcoded
    default for the prod app — empty string in DB used to leak through
    to the UI which complicated the OAuth flow debugging.
    """
    import os
    secret = cfg.get("client_secret") or ""
    client_id = cfg.get("client_id") or ""
    redirect_uri = cfg.get("redirect_uri") or ""
    redirect_source = "db" if redirect_uri else "none"
    if not redirect_uri:
        env_redirect = os.environ.get("ML_REDIRECT_URI", "").strip()
        if env_redirect:
            redirect_uri = env_redirect
            redirect_source = "env"
        else:
            # Last-resort default — what's currently registered in
            # developers.mercadolivre.com.br for this app.
            redirect_uri = "https://app.lsprofit.app/api/ml-oauth/callback"
            redirect_source = "default"
    return {
        "clientId": client_id,
        "clientSecretSet": bool(secret),
        "clientSecretPreview": (secret[:6] + "...") if secret else "",
        "redirectUri": redirect_uri,
        "source": {
            "clientId": "db" if client_id else "none",
            "clientSecret": "db" if secret else "none",
            "redirectUri": redirect_source,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# User tokens
# ──────────────────────────────────────────────────────────────────────────────

async def save_user_tokens(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    access_token: str,
    refresh_token: str,
    expires_in: int,
    ml_user_id: Optional[int] = None,
    nickname: Optional[str] = None,
    site_id: Optional[str] = None,
    scope: Optional[str] = None,
) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_user_tokens (
                user_id, access_token, refresh_token, access_token_expires_at,
                ml_user_id, ml_nickname, ml_site_id, scope,
                last_refreshed_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                access_token_expires_at = EXCLUDED.access_token_expires_at,
                ml_user_id = COALESCE(EXCLUDED.ml_user_id, ml_user_tokens.ml_user_id),
                ml_nickname = COALESCE(EXCLUDED.ml_nickname, ml_user_tokens.ml_nickname),
                ml_site_id = COALESCE(EXCLUDED.ml_site_id, ml_user_tokens.ml_site_id),
                scope = COALESCE(EXCLUDED.scope, ml_user_tokens.scope),
                last_refreshed_at = NOW(),
                updated_at = NOW()
            """,
            user_id, access_token, refresh_token, expires_at,
            ml_user_id, nickname, site_id, scope,
        )


async def load_user_tokens(pool: asyncpg.Pool, user_id: int) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_id, access_token, refresh_token, access_token_expires_at,
                   ml_user_id, ml_nickname, ml_site_id, scope, last_refreshed_at
            FROM ml_user_tokens WHERE user_id = $1
            """,
            user_id,
        )
    return dict(row) if row else None


async def delete_user_tokens(pool: asyncpg.Pool, user_id: int) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM ml_user_tokens WHERE user_id = $1", user_id
        )
    return result.endswith(" 1")


# ──────────────────────────────────────────────────────────────────────────────
# Refresh
# ──────────────────────────────────────────────────────────────────────────────

class MLRefreshError(Exception):
    """Raised when ML refresh endpoint returns an error (invalid_grant, etc.)."""


async def _post_refresh_to_ml(
    client_id: str, client_secret: str, refresh_token: str
) -> dict:
    """POST to ML /oauth/token with grant_type=refresh_token. Return token payload."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            ML_TOKEN_URL,
            headers={
                "accept": "application/json",
                "content-type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            },
        )
    try:
        payload = r.json()
    except ValueError:
        raise MLRefreshError(f"Non-JSON response from ML: HTTP {r.status_code}")
    if r.status_code != 200:
        err = payload.get("error") or f"http_{r.status_code}"
        desc = payload.get("error_description", "")
        raise MLRefreshError(f"{err}: {desc}")
    return payload


async def exchange_authorization_code(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> dict:
    """Exchange an OAuth authorization code for tokens (PKCE flow).

    Looks up client_id + client_secret from ml_app_config, calls ML /oauth/token
    with grant_type=authorization_code, persists tokens to ml_user_tokens.
    Returns the raw ML token payload (for the caller to forward ml_user_id etc.).
    """
    cfg = await load_app_config(pool)
    client_id = cfg.get("client_id")
    client_secret = cfg.get("client_secret")
    if not client_id or not client_secret:
        raise MLRefreshError("no_app_credentials")

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            ML_TOKEN_URL,
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
        raise MLRefreshError(f"Non-JSON response from ML: HTTP {r.status_code}")

    if r.status_code != 200:
        err = payload.get("error") or f"http_{r.status_code}"
        desc = payload.get("error_description", "")
        raise MLRefreshError(f"{err}: {desc}")

    await save_user_tokens(
        pool, user_id,
        access_token=payload["access_token"],
        refresh_token=payload["refresh_token"],
        expires_in=int(payload.get("expires_in", 21600)),
        ml_user_id=payload.get("user_id"),
        scope=payload.get("scope"),
    )

    # Kick off a 30-day backfill in the background so the UI shows history
    # immediately after OAuth-connect. Fire-and-forget; failures are logged
    # but don't block the OAuth response.
    try:
        from . import ml_backfill as _ml_backfill  # local import to avoid circular
        asyncio.create_task(_backfill_after_oauth(pool, user_id, _ml_backfill))
    except Exception as err:  # noqa: BLE001
        log.warning("backfill kickoff failed for user_id=%s: %s", user_id, err)

    return payload


async def _backfill_after_oauth(pool: asyncpg.Pool, user_id: int, backfill_module) -> None:
    try:
        async with httpx.AsyncClient() as http:
            await backfill_module.backfill_user(pool, http, user_id, days=30)
    except Exception as err:  # noqa: BLE001
        log.warning("post-OAuth backfill failed for user_id=%s: %s", user_id, err)


async def refresh_user_token(
    pool: asyncpg.Pool, user_id: int
) -> dict:
    """Refresh a single user's token. Returns new token payload.

    Raises MLRefreshError on failure (caller should invalidate tokens on invalid_grant).
    """
    cfg = await load_app_config(pool)
    client_id = cfg.get("client_id")
    client_secret = cfg.get("client_secret")
    if not client_id or not client_secret:
        raise MLRefreshError("no_app_credentials")

    tokens = await load_user_tokens(pool, user_id)
    if not tokens:
        raise MLRefreshError("no_user_tokens")

    # Advisory lock to prevent refresh race between sync + scheduler
    # hashtext is stable across sessions; use user_id as lock scope
    async with pool.acquire() as conn:
        locked = await conn.fetchval(
            "SELECT pg_try_advisory_xact_lock(hashtext('ml_refresh_' || $1))",
            str(user_id),
        )
        if not locked:
            # Another worker is refreshing — just wait and re-read
            await asyncio.sleep(2)
            fresh = await load_user_tokens(pool, user_id)
            if fresh and fresh["access_token_expires_at"] > datetime.now(timezone.utc) + SYNC_REFRESH_MARGIN:
                return {
                    "access_token": fresh["access_token"],
                    "refresh_token": fresh["refresh_token"],
                    "expires_in": int((fresh["access_token_expires_at"] - datetime.now(timezone.utc)).total_seconds()),
                }

    payload = await _post_refresh_to_ml(
        client_id, client_secret, tokens["refresh_token"]
    )

    await save_user_tokens(
        pool, user_id,
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token", tokens["refresh_token"]),
        expires_in=int(payload.get("expires_in", 21600)),
        ml_user_id=payload.get("user_id") or tokens.get("ml_user_id"),
        scope=payload.get("scope") or tokens.get("scope"),
    )
    return payload


async def get_valid_access_token(
    pool: asyncpg.Pool, user_id: int
) -> tuple[str, datetime, bool]:
    """Return (access_token, expires_at, refreshed_now).

    If token expires within SYNC_REFRESH_MARGIN, refreshes synchronously.
    Raises MLRefreshError if no tokens or refresh fails.
    """
    tokens = await load_user_tokens(pool, user_id)
    if not tokens:
        raise MLRefreshError("no_user_tokens")

    expires_at: datetime = tokens["access_token_expires_at"]
    now = datetime.now(timezone.utc)

    if expires_at > now + SYNC_REFRESH_MARGIN:
        return tokens["access_token"], expires_at, False

    # Needs refresh
    await refresh_user_token(pool, user_id)
    refreshed = await load_user_tokens(pool, user_id)
    if not refreshed:
        raise MLRefreshError("token_disappeared_after_refresh")
    return refreshed["access_token"], refreshed["access_token_expires_at"], True


async def refresh_all_expiring_tokens(pool: Optional[asyncpg.Pool]) -> dict:
    """Scheduled job: refresh all user tokens expiring within SCHEDULER_REFRESH_MARGIN.

    Returns {refreshed: int, failed: int, details: [...]}. Never raises.
    """
    if pool is None:
        log.warning("[ml-oauth-scheduler] No DB pool — skipping")
        return {"refreshed": 0, "failed": 0, "details": []}

    try:
        cfg = await load_app_config(pool)
        if not cfg.get("client_id") or not cfg.get("client_secret"):
            log.info("[ml-oauth-scheduler] No app credentials configured — skipping")
            return {"refreshed": 0, "failed": 0, "details": []}

        cutoff = datetime.now(timezone.utc) + SCHEDULER_REFRESH_MARGIN
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT user_id FROM ml_user_tokens
                WHERE access_token_expires_at < $1
                  AND refresh_token IS NOT NULL
                """,
                cutoff,
            )

        if not rows:
            log.info("[ml-oauth-scheduler] No tokens need refresh (cutoff=%s)", cutoff.isoformat())
            return {"refreshed": 0, "failed": 0, "details": []}

        log.info("[ml-oauth-scheduler] Refreshing %d tokens", len(rows))

        details: list[dict] = []
        refreshed_count = 0
        failed_count = 0

        for row in rows:
            uid = row["user_id"]
            try:
                await refresh_user_token(pool, uid)
                refreshed_count += 1
                details.append({"user_id": uid, "status": "ok"})
            except MLRefreshError as err:
                failed_count += 1
                details.append({"user_id": uid, "status": "failed", "error": str(err)})
                log.error("[ml-oauth-scheduler] user_id=%s refresh failed: %s", uid, err)
                # If invalid_grant → wipe tokens so UI shows disconnected
                if "invalid_grant" in str(err):
                    await delete_user_tokens(pool, uid)
                    log.warning("[ml-oauth-scheduler] user_id=%s tokens wiped (invalid_grant)", uid)
            except Exception as err:  # noqa: BLE001
                failed_count += 1
                details.append({"user_id": uid, "status": "error", "error": str(err)})
                log.exception("[ml-oauth-scheduler] user_id=%s unexpected error", uid)

        return {"refreshed": refreshed_count, "failed": failed_count, "details": details}
    except Exception as err:  # noqa: BLE001
        log.exception("[ml-oauth-scheduler] Top-level failure: %s", err)
        return {"refreshed": 0, "failed": 0, "details": [{"error": str(err)}]}
