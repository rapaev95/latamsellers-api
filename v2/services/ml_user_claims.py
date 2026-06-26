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
ML_MAX_CONCURRENCY = 5      # global ceiling on simultaneous ML requests
ML_MAX_RETRIES = 3          # retries on 429 before giving up

# Global (process-wide) on purpose: ML rate-limits per token/app, so we cap
# concurrent requests across ALL claims and all users, not per refresh.
_ml_sem = asyncio.Semaphore(ML_MAX_CONCURRENCY)


async def _ml_get(
    http: httpx.AsyncClient, token: str, url: str, *, timeout: float = 15.0
) -> httpx.Response:
    """GET against ML with a global concurrency ceiling and 429 backoff.

    Returns the httpx.Response so callers keep their existing status checks.
    Respects the Retry-After header; falls back to exponential backoff.
    """
    headers = {"Authorization": f"Bearer {token}"}
    backoff = 0.5
    last: httpx.Response | None = None
    for attempt in range(ML_MAX_RETRIES + 1):
        async with _ml_sem:
            last = await http.get(url, headers=headers, timeout=timeout)
        if last.status_code != 429:
            return last
        retry_after = last.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else backoff
        except ValueError:
            delay = backoff
        log.warning("ML 429 %s attempt=%s sleep=%.1fs", url, attempt + 1, delay)
        await asyncio.sleep(delay)
        backoff *= 2
    return last  # last 429 — existing status checks downstream handle it


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
    for _ in range(max_pages):
        url = (
            f"{ML_API_BASE}/post-purchase/v1/claims/search"
            f"?status={status}&sort=date_created&limit=50&offset={offset}"
        )
        try:
            r = await _ml_get(http, token, url, timeout=20.0)
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
        r = await _ml_get(
            http, token,
            f"{ML_API_BASE}/post-purchase/v1/claims/{claim_id}",
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
        r = await _ml_get(
            http, token,
            f"{ML_API_BASE}/post-purchase/v1/claims/{cid}",
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
            r = await _ml_get(
                http, token,
                f"{ML_API_BASE}/post-purchase/v2/claims/{cid}/returns",
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
                            rr = await _ml_get(
                                http, token,
                                f"{ML_API_BASE}/post-purchase/v1/returns/{ret['id']}/reviews",
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

    # 4. Mediation details — REMOVED 2026-05-16.
    # ML deprecated /post-purchase/v1/mediations/{id}; the endpoint now
    # returns 404 for every claim, generating ~50% of the refresh-loop
    # traffic for no payoff and causing 429 rate limits on legitimate
    # /claims/{id}/messages calls.
    # If a future ML version embeds mediation data inline in /claims/{id}
    # response, the `claim.get("mediation")` reader in section (c) below
    # will pick it up automatically. No separate fetch needed.

    # 5. Messages thread — buyer's complaint + ML mediator notes + seller
    # replies. Two sources we merge:
    #   a) /claims/{id}/messages — the regular thread
    #   b) claim.mediation.messages — sometimes the mediation payload
    #      itself embeds the message list, no separate fetch needed.
    # NB: /post-purchase/v1/mediations/{id}/messages was a third source
    # but ML deprecated it (always 404); see section 4 above.
    # Both remaining sources are merged + de-duped so _compute_needs_action
    # and the Telegram dispatcher see the latest reply regardless of source.

    def _absorb(msg_list: Any) -> list[dict]:
        out: list[dict] = []
        if isinstance(msg_list, dict):
            inner = msg_list.get("messages")
            if isinstance(inner, list):
                msg_list = inner
            else:
                msg_list = [msg_list]
        if not isinstance(msg_list, list):
            return out
        for m in msg_list:
            if isinstance(m, dict):
                out.append(m)
        return out

    aggregated: list[dict] = []
    # (a) claims/{id}/messages
    try:
        r = await _ml_get(
            http, token,
            f"{ML_API_BASE}/post-purchase/v1/claims/{cid}/messages",
            timeout=15.0,
        )
        if r.status_code == 200:
            aggregated.extend(_absorb(r.json()))
        elif r.status_code not in (403, 404):
            log.info("claims/%s/messages status=%s body=%s", cid, r.status_code, r.text[:200])
    except Exception as err:  # noqa: BLE001
        log.warning("claims/%s/messages exception: %s", cid, err)

    # (b) mediations/{id}/messages — REMOVED 2026-05-16 (ML deprecated;
    # always returned 404, causing rate-limit pressure). See section 4.

    # (c) embedded inside mediation payload (best-effort — keys vary).
    # `claim["mediation"]` is no longer populated by us (the /mediations/
    # fetch was dropped), but if ML embeds mediation data inline in the
    # /claims/{id} response, this block picks it up automatically.
    mediation = claim.get("mediation")
    if isinstance(mediation, dict):
        for key in ("messages", "message_history", "thread"):
            v = mediation.get(key)
            if v:
                aggregated.extend(_absorb(v))

    # De-dupe: prefer message id; fall back to (sender_role, date_created, text)
    if aggregated:
        seen: set[Any] = set()
        unique: list[dict] = []
        for m in aggregated:
            mid = m.get("id")
            if mid is None:
                key = (
                    m.get("sender_role") or (m.get("sender") or {}).get("role") if isinstance(m.get("sender"), dict) else m.get("sender_role"),
                    m.get("date_created") or m.get("date"),
                    (m.get("message") or m.get("text") or "")[:80],
                )
            else:
                key = ("id", mid)
            if key in seen:
                continue
            seen.add(key)
            unique.append(m)
        claim["messages"] = unique

    # 6. Order item — surface the product title in the TG card so the seller
    # sees WHAT the claim is about, not just the order id. resource_id is
    # the order_id; one /orders/{id} call gives us the title + item_id.
    order_id = claim.get("resource_id")
    if order_id:
        try:
            r = await _ml_get(
                http, token,
                f"{ML_API_BASE}/orders/{order_id}",
                timeout=15.0,
            )
            if r.status_code == 200:
                order_data = r.json() or {}
                items = order_data.get("order_items") or []
                # Capture pack_id — ML's mediation URL uses pack_id, not
                # order_id: /vendas/novo/mensagens/{pack_id}/mediacao/{claim_id}.
                # Single-order packs return null pack_id; fall back to order_id.
                claim["order_pack_id"] = order_data.get("pack_id") or order_data.get("id")
                if items and isinstance(items[0], dict):
                    item_inner = items[0].get("item") or {}
                    buyer = order_data.get("buyer") or {}
                    claim["order_item"] = {
                        "id": item_inner.get("id"),
                        "title": item_inner.get("title"),
                        "variation_id": item_inner.get("variation_id"),
                        "category_id": item_inner.get("category_id"),
                        "quantity": items[0].get("quantity"),
                        "unit_price": items[0].get("unit_price"),
                    }
                    if buyer:
                        claim["order_buyer"] = {
                            "id": buyer.get("id"),
                            "nickname": buyer.get("nickname"),
                            "first_name": buyer.get("first_name"),
                        }
            elif r.status_code not in (403, 404):
                log.info("orders/%s status=%s body=%s", order_id, r.status_code, r.text[:200])
        except Exception as err:  # noqa: BLE001
            log.warning("orders/%s exception: %s", order_id, err)

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

    0. Last message in the thread is from the seller (`respondent`)
       → ball is in the buyer's/mediator's court → NO ACTION. This rule
       fires first because it's the most reliable "I already replied"
       signal — without it, the inbox keeps showing claims where the
       seller already answered via ML's UI but no return has been
       triggered yet.
    1. If status != opened → already resolved → NO ACTION
    2. If returns array is empty/missing → seller hasn't chosen a solution
       yet, ML is asking for one → ACTION
    3. If returns[0].status == 'delivered' → return parcel arrived back at
       the seller, must inspect + decide refund/replace → ACTION
    4. Otherwise (label_generated / shipped / in_transit / etc.) → return
       in motion, no immediate action required → NO ACTION
    """
    # Rule 0 — seller already replied in the thread.
    messages = claim.get("messages")
    if isinstance(messages, list) and messages:
        # Pick the message with the latest date_created. Don't trust list
        # ordering — ML returns oldest-first sometimes, newest-first others.
        last_msg = None
        last_dt = None
        for m in messages:
            if not isinstance(m, dict):
                continue
            dt = _parse_dt(m.get("date_created") or m.get("date"))
            if last_dt is None or (dt and dt > last_dt):
                last_dt = dt
                last_msg = m
        if isinstance(last_msg, dict):
            sender = last_msg.get("sender_role") or ""
            if not sender and isinstance(last_msg.get("sender"), dict):
                sender = last_msg["sender"].get("role") or ""
            if sender == "respondent":
                return False

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
