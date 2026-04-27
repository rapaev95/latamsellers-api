"""Claims cache — ML `/post-purchase/v1/claims/search` + enriched fields.

Refresh paginates opened+closed claims, then enriches each with full payload,
returns and reviews (same pipeline as the old Next.js direct-fetch). Stores
the entire enriched object as JSONB so UI can read whatever it needs.

TTL: 6h. Claim state changes by the hour (seller/buyer replies), but full-fetch
takes ~20-40s with enrichment — worth caching.

Staleness reconciliation: after each refresh we look for cached rows whose
status='opened' but were NOT seen in this refresh (i.e. they fell off ML's
"opened" search results page). For each one we hit /claims/{id} directly to
learn its real current state and update accordingly. Without this step,
once a claim is opened in our cache it stays opened forever even if ML
already moved it to closed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

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

async def _search_claims(
    http: httpx.AsyncClient,
    token: str,
    status: str,
    max_pages: int = 4,
) -> list[dict]:
    """Paginate /claims/search until we exhaust results or hit max_pages.
    50/page × 4 pages = 200 claims max per status. Old single-page fetch
    capped at 50 silently dropped users with bigger histories.
    """
    out: list[dict] = []
    offset = 0
    headers = {"Authorization": f"Bearer {token}"}
    for _ in range(max_pages):
        url = (
            f"{ML_API_BASE}/post-purchase/v1/claims/search"
            f"?status={status}&sort=date_created&limit=50&offset={offset}"
        )
        try:
            r = await http.get(url, headers=headers, timeout=20.0)
        except Exception as err:  # noqa: BLE001
            log.warning("claims/search status=%s offset=%s failed: %s", status, offset, err)
            break
        if r.status_code != 200:
            log.warning("claims/search %s status=%s body=%s", status, r.status_code, r.text[:200])
            break
        data = r.json() or {}
        page = data.get("data") or []
        if not page:
            break
        out.extend(page)
        if len(page) < 50:
            break
        offset += 50
        await asyncio.sleep(RATE_SLEEP)
    return out


async def _fetch_one_claim(http: httpx.AsyncClient, token: str, claim_id: int) -> dict | None:
    """Fetch a single claim by id (used by reconciliation)."""
    try:
        r = await http.get(
            f"{ML_API_BASE}/post-purchase/v1/claims/{claim_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return {"id": claim_id, "status": "deleted", "_not_found": True}
        log.warning("claims/%s status=%s body=%s", claim_id, r.status_code, r.text[:200])
    except Exception as err:  # noqa: BLE001
        log.warning("claims/%s exception: %s", claim_id, err)
    return None


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

    # 2. Returns (devolução) if referenced.
    # ML uses "return" (singular) in related_entities — accept both forms;
    # the original strict check was the reason claim["returns"] was always
    # missing in cached payloads.
    related = claim.get("related_entities") or []
    has_returns = any(
        (isinstance(e, str) and e in ("returns", "return"))
        or (isinstance(e, dict) and e.get("type") in ("returns", "return"))
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
                # 3. Reviews per return — same singular/plural quirk.
                for ret in returns:
                    ret_related = (ret or {}).get("related_entities") or []
                    has_reviews = any(
                        (isinstance(e, str) and e in ("reviews", "review"))
                        or (isinstance(e, dict) and e.get("type") in ("reviews", "review"))
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

    # 4. Mediation details (only for type=mediations / stage=dispute claims).
    # The base /claims/{id} response is sparse for disputes — the rich data
    # (due_date, affects_reputation, last_action_player, expected_actions)
    # lives on /post-purchase/v1/mediations/{id}. Without this we cannot
    # filter "needs my action" the way ML's UI does.
    if claim.get("type") == "mediations" or claim.get("stage") == "dispute":
        try:
            r = await http.get(
                f"{ML_API_BASE}/post-purchase/v1/mediations/{cid}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
            if r.status_code == 200:
                claim["mediation"] = r.json()
            elif r.status_code not in (403, 404):
                log.info("mediations/%s status=%s body=%s", cid, r.status_code, r.text[:200])
        except Exception as err:  # noqa: BLE001
            log.warning("mediations/%s exception: %s", cid, err)

    return claim


# ── Refresh ───────────────────────────────────────────────────────────────────

async def refresh_user_claims(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError:
        return {"fetched": 0, "saved": 0, "reconciled": 0}

    async with httpx.AsyncClient() as http:
        opened, closed = await asyncio.gather(
            _search_claims(http, token, "opened"),
            _search_claims(http, token, "closed"),
        )
        all_claims = opened + closed
        seen_ids: set[int] = set()
        for c in all_claims:
            try:
                cid = int(c.get("id"))
            except (TypeError, ValueError):
                continue
            seen_ids.add(cid)

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

        # ── Reconcile stale "opened" rows ───────────────────────────────────
        # Cached claims marked opened that didn't appear in this refresh's
        # opened search (and weren't picked up by the closed search either)
        # are stale — ML moved them off the opened list but our cache still
        # shows them. Hit /claims/{id} directly to learn their real state.
        async with pool.acquire() as conn:
            stale_rows = await conn.fetch(
                """
                SELECT claim_id FROM ml_user_claims
                 WHERE user_id = $1 AND status = 'opened'
                """,
                user_id,
            )
        stale_ids = [int(r["claim_id"]) for r in stale_rows if int(r["claim_id"]) not in seen_ids]

        reconciled = 0
        if stale_ids:
            log.info(
                "claims reconcile user=%s stale_opened=%s (will fetch individually)",
                user_id, len(stale_ids),
            )
            for i in range(0, len(stale_ids), ENRICH_CONCURRENCY):
                batch = stale_ids[i:i + ENRICH_CONCURRENCY]
                fetched = await asyncio.gather(
                    *[_fetch_one_claim(http, token, cid) for cid in batch]
                )
                for cid, payload in zip(batch, fetched):
                    if payload is None:
                        continue
                    if payload.get("_not_found"):
                        # ML returned 404 — claim was deleted/purged. Drop it.
                        async with pool.acquire() as conn:
                            await conn.execute(
                                "DELETE FROM ml_user_claims WHERE user_id = $1 AND claim_id = $2",
                                user_id, cid,
                            )
                        reconciled += 1
                        continue
                    # Re-enrich + upsert with fresh status
                    fresh = await _enrich_one(http, token, payload)
                    try:
                        async with pool.acquire() as conn:
                            await _upsert(conn, user_id, fresh)
                        reconciled += 1
                    except Exception as err:  # noqa: BLE001
                        log.warning("reconcile upsert claim %s failed: %s", cid, err)
                await asyncio.sleep(RATE_SLEEP)

    return {"fetched": len(all_claims), "saved": saved, "reconciled": reconciled}


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

def _compute_needs_action(claim: dict[str, Any]) -> bool:
    """Decide whether this claim needs immediate seller action.

    Rules derived from ML's "Reclamações" view (where the seller sees only
    cards with the prominent "Atender reclamação" CTA):

    1. If status != opened → no action needed (already resolved)
    2. If returns array is empty/missing → seller hasn't chosen a solution
       yet, ML is asking for one → ACTION
    3. If returns[0].status == 'delivered' → return parcel arrived back at
       the seller, must inspect + decide refund/replace → ACTION
    4. Otherwise (label_generated / shipped / in_transit / etc.) → return
       in motion, no immediate action required → NO ACTION
    """
    if (claim.get("status") or "").lower() != "opened":
        return False
    returns = claim.get("returns")
    if not returns or not isinstance(returns, list):
        return True
    head = returns[0] if returns else {}
    if not isinstance(head, dict):
        return True
    head_status = (head.get("status") or "").lower()
    if head_status == "delivered":
        return True
    return False


async def get_cached(
    pool: asyncpg.Pool,
    user_id: int,
    status: str = "ALL",
    actionable: Optional[bool] = None,
) -> dict[str, Any]:
    """Read cached claims.

    `actionable`:
      - None → no filter (return all matching the status filter)
      - True → only claims where _compute_needs_action() is True
      - False → only claims where it's False
    """
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
    actionable_count = 0
    for r in rows:
        enriched = r["enriched"]
        if isinstance(enriched, str):
            enriched = json.loads(enriched or "{}")
        if not isinstance(enriched, dict):
            continue
        needs = _compute_needs_action(enriched)
        # Stash on the payload so the UI can render per-card without
        # recomputing the rule client-side.
        enriched["needsAction"] = needs
        if needs:
            actionable_count += 1
        if actionable is True and not needs:
            continue
        if actionable is False and needs:
            continue
        claims.append(enriched)

    # Compute the count separately by walking ALL opened rows (not just the
    # filtered ones) so the UI badge for "Требуют действия" is correct
    # regardless of which filter is active.
    if actionable is None:
        opened_actionable = actionable_count if status.lower() in ("opened", "open") else None
    else:
        # Need to walk again across all opened rows to get true total.
        opened_actionable = None

    if opened_actionable is None:
        async with pool.acquire() as conn:
            opened_rows = await conn.fetch(
                """
                SELECT enriched FROM ml_user_claims
                 WHERE user_id = $1 AND status = 'opened'
                """,
                user_id,
            )
        opened_actionable = 0
        for r in opened_rows:
            payload = r["enriched"]
            if isinstance(payload, str):
                payload = json.loads(payload or "{}")
            if isinstance(payload, dict) and _compute_needs_action(payload):
                opened_actionable += 1

    totals["actionable"] = opened_actionable

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
