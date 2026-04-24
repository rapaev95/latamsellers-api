"""Claims cache — ML `/post-purchase/v1/claims/search` + enriched fields.

Refresh paginates opened+closed claims, then enriches each with full payload,
returns and reviews (same pipeline as the old Next.js direct-fetch). Stores
the entire enriched object as JSONB so UI can read whatever it needs.

TTL: 6h. Claim state changes by the hour (seller/buyer replies), but full-fetch
takes ~20-40s with enrichment — worth caching.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
RATE_SLEEP = 0.2
ENRICH_CONCURRENCY = 10


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_user_claims (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  claim_id BIGINT NOT NULL,
  resource_id TEXT,
  type TEXT,
  status TEXT,
  stage TEXT,
  date_created TIMESTAMPTZ,
  enriched JSONB,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_user_claims_user_status ON ml_user_claims(user_id, status);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _parse_dt(s: Any) -> Any:
    from datetime import datetime
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    try:
        raw = str(s).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


# ── ML API ────────────────────────────────────────────────────────────────────

async def _search_claims(http: httpx.AsyncClient, token: str, status: str) -> list[dict]:
    url = f"{ML_API_BASE}/post-purchase/v1/claims/search?status={status}&sort=date_created&limit=50"
    try:
        r = await http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20.0)
    except Exception as err:  # noqa: BLE001
        log.warning("claims/search status=%s failed: %s", status, err)
        return []
    if r.status_code != 200:
        return []
    data = r.json() or {}
    return data.get("data") or []


async def _enrich_one(http: httpx.AsyncClient, token: str, claim: dict) -> dict:
    cid = claim.get("id")
    if not cid:
        return claim
    # 1. Full claim payload
    try:
        r = await http.get(
            f"{ML_API_BASE}/post-purchase/v1/claims/{cid}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if r.status_code == 200:
            claim = {**claim, **(r.json() or {})}
    except Exception:  # noqa: BLE001
        return claim

    # 2. Returns (devolução) if referenced
    related = claim.get("related_entities") or []
    has_returns = any(
        (isinstance(e, str) and e == "returns")
        or (isinstance(e, dict) and e.get("type") == "returns")
        for e in related
    )
    if has_returns:
        try:
            r = await http.get(
                f"{ML_API_BASE}/post-purchase/v2/claims/{cid}/returns",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
            if r.status_code == 200:
                rd = r.json()
                returns = rd.get("data") if isinstance(rd, dict) and isinstance(rd.get("data"), list) else (rd if isinstance(rd, list) else [rd] if rd else [])
                # 3. Reviews per return
                for ret in returns:
                    ret_related = (ret or {}).get("related_entities") or []
                    has_reviews = any(
                        (isinstance(e, str) and e == "reviews")
                        or (isinstance(e, dict) and e.get("type") == "reviews")
                        for e in ret_related
                    )
                    if has_reviews and ret.get("id"):
                        try:
                            rr = await http.get(
                                f"{ML_API_BASE}/post-purchase/v1/returns/{ret['id']}/reviews",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=10.0,
                            )
                            if rr.status_code == 200:
                                rev = rr.json() or {}
                                ret["reviews"] = rev.get("reviews") or (rev if isinstance(rev, list) else [])
                        except Exception:  # noqa: BLE001
                            pass
                claim["returns"] = returns
        except Exception:  # noqa: BLE001
            pass
    return claim


# ── Refresh ───────────────────────────────────────────────────────────────────

async def refresh_user_claims(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError:
        return {"fetched": 0, "saved": 0}

    async with httpx.AsyncClient() as http:
        opened, closed = await asyncio.gather(
            _search_claims(http, token, "opened"),
            _search_claims(http, token, "closed"),
        )
        all_claims = opened + closed

        # Enrich in batches to respect rate limits (10 concurrent).
        enriched: list[dict] = []
        for i in range(0, len(all_claims), ENRICH_CONCURRENCY):
            batch = all_claims[i:i + ENRICH_CONCURRENCY]
            results = await asyncio.gather(
                *[_enrich_one(http, token, c) for c in batch]
            )
            enriched.extend(results)
            await asyncio.sleep(RATE_SLEEP)

    saved = 0
    for c in enriched:
        try:
            async with pool.acquire() as conn:
                await _upsert(conn, user_id, c)
            saved += 1
        except Exception as err:  # noqa: BLE001
            log.warning("upsert claim %s failed: %s", c.get("id"), err)

    return {"fetched": len(all_claims), "saved": saved}


async def _upsert(conn: asyncpg.Connection, user_id: int, c: dict) -> None:
    cid = c.get("id")
    if cid is None:
        return
    await conn.execute(
        """
        INSERT INTO ml_user_claims
          (user_id, claim_id, resource_id, type, status, stage, date_created,
           enriched, fetched_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, NOW())
        ON CONFLICT (user_id, claim_id) DO UPDATE SET
          resource_id = EXCLUDED.resource_id,
          type = EXCLUDED.type,
          status = EXCLUDED.status,
          stage = EXCLUDED.stage,
          date_created = EXCLUDED.date_created,
          enriched = EXCLUDED.enriched,
          fetched_at = NOW()
        """,
        user_id,
        int(cid),
        str(c.get("resource_id") or "") or None,
        c.get("type"),
        c.get("status"),
        c.get("stage"),
        _parse_dt(c.get("date_created")),
        json.dumps(c, default=str),
    )


# ── Cache readback ────────────────────────────────────────────────────────────

async def get_cached(
    pool: asyncpg.Pool,
    user_id: int,
    status: str = "ALL",
) -> dict[str, Any]:
    where = "WHERE user_id = $1"
    params: list[Any] = [user_id]
    if status and status.upper() != "ALL":
        # Accept "open" → "opened" like legacy route did.
        ml_status = "opened" if status.lower() == "open" else status.lower()
        where += " AND status = $2"
        params.append(ml_status)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT claim_id, enriched,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_claims
              {where}
             ORDER BY date_created DESC NULLS LAST
            """,
            *params,
        )
        # Per-status totals (for UI badges)
        totals = {}
        for s in ("opened", "closed"):
            t = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_claims WHERE user_id = $1 AND status = $2",
                user_id, s,
            )
            totals[s] = int(t or 0)

    claims = []
    for r in rows:
        enriched = r["enriched"]
        if isinstance(enriched, str):
            enriched = json.loads(enriched or "{}")
        claims.append(enriched)
    return {
        "total": len(claims),
        "totals": totals,
        "claims": claims,
        "fetchedAt": rows[0]["fetched_at"] if rows else None,
    }


async def get_latest_fetched_at(pool: asyncpg.Pool, user_id: int) -> str | None:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"')
              FROM ml_user_claims
             WHERE user_id = $1
            """,
            user_id,
        )
