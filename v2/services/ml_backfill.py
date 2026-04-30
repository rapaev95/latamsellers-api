"""Backfill ML events for a user from 5 listing endpoints.

Called:
- Manually via POST /api/v2/escalar/backfill-notices  (days=30)
- Automatically after OAuth-connect             (days=30)
- APScheduler daily job                        (days=1, catch-up for webhook gaps)

Normalizes every fetched record through ml_normalize.normalize_event and
bulk-upserts into ml_notices (same UNIQUE(user_id, notice_id) constraint).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_normalize
from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
PER_ENDPOINT_LIMIT = 50
RATE_SLEEP = 0.05


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _get_ml(http: httpx.AsyncClient, access_token: str, path: str) -> Any:
    try:
        r = await http.get(
            f"{ML_API_BASE}{path}" if path.startswith("/") else path,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20.0,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("ML GET %s failed: %s", path, err)
        return None
    if r.status_code != 200:
        log.warning("ML GET %s → %s: %s", path, r.status_code, r.text[:200])
        return None
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return None


async def _get_ml_user_id(pool: asyncpg.Pool, user_id: int) -> int | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ml_user_id FROM ml_user_tokens WHERE user_id = $1", user_id,
        )
    return row["ml_user_id"] if row else None


# ── Source fetchers ───────────────────────────────────────────────────────────

async def _fetch_orders(http: httpx.AsyncClient, token: str, ml_user_id: int, since: datetime) -> list[dict]:
    iso = since.strftime("%Y-%m-%dT%H:%M:%S.000-03:00")
    data = await _get_ml(
        http, token,
        f"/orders/search?seller={ml_user_id}&order.date_created.from={iso}&limit={PER_ENDPOINT_LIMIT}",
    )
    return (data or {}).get("results") or []


async def _fetch_questions(http: httpx.AsyncClient, token: str) -> list[dict]:
    # ML /my/received_questions/search accepts ONE status at a time — fetch both
    # UNANSWERED and ANSWERED separately and merge (+ sort by date_created DESC).
    unanswered, answered = await asyncio.gather(
        _get_ml(
            http, token,
            f"/my/received_questions/search?status=UNANSWERED"
            f"&sort_fields=date_created&sort_types=DESC"
            f"&api_version=4&limit={PER_ENDPOINT_LIMIT}",
        ),
        _get_ml(
            http, token,
            f"/my/received_questions/search?status=ANSWERED"
            f"&sort_fields=date_created&sort_types=DESC"
            f"&api_version=4&limit={PER_ENDPOINT_LIMIT}",
        ),
    )
    merged: list[dict] = []
    for bucket in (unanswered, answered):
        if bucket and isinstance(bucket, dict):
            merged.extend(bucket.get("questions") or [])
    return merged


async def _enrich_claim_with_returns(http: httpx.AsyncClient, token: str, claim: dict) -> dict:
    """Fetch full claim + returns + triage reviews. Returns enriched copy."""
    claim_id = claim.get("id")
    if not claim_id:
        return claim
    full = await _get_ml(http, token, f"/post-purchase/v1/claims/{claim_id}") or {}
    merged = {**claim, **full}

    related = merged.get("related_entities") or []
    has_return = any(
        (isinstance(e, str) and e == "returns")
        or (isinstance(e, dict) and (e.get("type") == "returns" or e.get("name") == "returns"))
        for e in related
    )
    if not has_return:
        return merged

    returns_data = await _get_ml(http, token, f"/post-purchase/v2/claims/{claim_id}/returns")
    if not returns_data:
        return merged
    returns_list = returns_data.get("data") if isinstance(returns_data, dict) else returns_data
    if isinstance(returns_list, dict):
        returns_list = [returns_list]
    if not isinstance(returns_list, list):
        returns_list = []

    for ret in returns_list:
        ret_related = ret.get("related_entities") or []
        has_reviews = any(
            (isinstance(e, str) and e == "reviews")
            or (isinstance(e, dict) and (e.get("type") == "reviews" or e.get("name") == "reviews"))
            for e in ret_related
        )
        if has_reviews and ret.get("id"):
            rev = await _get_ml(http, token, f"/post-purchase/v1/returns/{ret['id']}/reviews")
            if rev:
                ret["reviews"] = rev.get("reviews") if isinstance(rev, dict) else rev

    merged["returns"] = returns_list
    return merged


async def _fetch_claims(http: httpx.AsyncClient, token: str) -> list[dict]:
    data = await _get_ml(
        http, token,
        f"/post-purchase/v1/claims/search?player_role=seller&limit={PER_ENDPOINT_LIMIT}",
    )
    claims = (data or {}).get("data") or (data or {}).get("results") or []
    if not claims:
        return []
    # Enrich each claim with full payload + returns + reviews (batched by 10).
    enriched: list[dict] = []
    for i in range(0, len(claims), 10):
        batch = claims[i:i + 10]
        results = await asyncio.gather(
            *[_enrich_claim_with_returns(http, token, c) for c in batch],
            return_exceptions=True,
        )
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                enriched.append(batch[idx])  # keep bare row on error
            else:
                enriched.append(r)
    return enriched


async def _fetch_paused_closed_items(http: httpx.AsyncClient, token: str, ml_user_id: int) -> list[dict]:
    async def _ids_for_status(status: str) -> list[str]:
        data = await _get_ml(
            http, token,
            f"/users/{ml_user_id}/items/search?status={status}&limit={PER_ENDPOINT_LIMIT}",
        )
        return (data or {}).get("results") or []

    ids_paused, ids_closed = await asyncio.gather(
        _ids_for_status("paused"),
        _ids_for_status("closed"),
    )
    ids = list(dict.fromkeys([*ids_paused, *ids_closed]))[:100]
    if not ids:
        return []

    details: list[dict] = []
    # Batch /items?ids=... (up to 20 per call)
    for i in range(0, len(ids), 20):
        batch = ids[i:i + 20]
        data = await _get_ml(
            http, token,
            f"/items?ids={','.join(batch)}&attributes=id,status,sub_status,title,permalink,last_updated",
        )
        if isinstance(data, list):
            for r in data:
                if r and r.get("code") == 200 and r.get("body"):
                    details.append(r["body"])
    return details


async def _fetch_messages(http: httpx.AsyncClient, token: str, ml_user_id: int) -> list[dict]:
    data = await _get_ml(
        http, token,
        f"/messages/actions?seller_id={ml_user_id}&limit={PER_ENDPOINT_LIMIT}",
    )
    if not data:
        return []
    # API shape varies — accept either a list or {results: [...]}.
    if isinstance(data, list):
        return data
    return data.get("results") or data.get("messages") or []


# ── Upsert ────────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO ml_notices
  (user_id, notice_id, label, description, from_date, tags, actions, raw,
   topic, resource, updated_at)
VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, NOW())
ON CONFLICT (user_id, notice_id) DO UPDATE SET
  label = EXCLUDED.label,
  description = EXCLUDED.description,
  from_date = EXCLUDED.from_date,
  tags = EXCLUDED.tags,
  actions = EXCLUDED.actions,
  raw = EXCLUDED.raw,
  topic = EXCLUDED.topic,
  resource = EXCLUDED.resource,
  updated_at = NOW()
"""


async def _enrich_order_with_margin(
    conn: asyncpg.Connection, user_id: int, enriched: dict, period_months: int = 3,
    pool: Optional[asyncpg.Pool] = None,
) -> None:
    """Injects `_margin` (apply_hypothetical_price от sale_price) в order payload.

    Mutates `enriched` in-place. Без модификации БД. Если cache empty или
    нет item_id / unit_price — просто не инжектит, normalize-ветка покажет
    «Margem indisponível». Для SMART/auto-activated promotions accept этот
    block не нужен — это только для orders_v2 уведомлений.

    Также инжектит `_breakeven` — state break-even tracker'а после этой
    продажи (cumulative variable margin + target + breakeven_reached). Это
    показывается в TG normalize как "📈 Прогресс месяца".
    """
    from . import ml_item_margin as ml_margin_svc
    from . import ml_breakeven as breakeven_svc

    items_arr = enriched.get("order_items") or enriched.get("items") or []
    if not items_arr:
        return
    first = items_arr[0] if isinstance(items_arr[0], dict) else None
    if not first:
        return
    inner = first.get("item") if isinstance(first.get("item"), dict) else first
    item_id = ""
    if isinstance(inner, dict):
        item_id = str(inner.get("id") or inner.get("mlb") or "").strip().upper()
    if not item_id:
        item_id = str(first.get("mlb") or "").strip().upper()
    try:
        sale_price = float(first.get("unit_price") or 0.0)
    except (TypeError, ValueError):
        sale_price = 0.0
    if not item_id or sale_price <= 0:
        return

    # Inline-query (тот же что в get_cached_margin), потому что у нас уже
    # есть conn. Это копия нескольких строк ради избегания двойного pool-acquire.
    row = await conn.fetchrow(
        """
        SELECT payload, computed_at
          FROM ml_item_margin_cache
         WHERE user_id = $1 AND item_id = $2 AND period_months = $3
        """,
        user_id, item_id, period_months,
    )
    if not row:
        return
    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:  # noqa: BLE001
            return
    if not isinstance(payload, dict):
        return
    payload["computed_at"] = row["computed_at"].isoformat() if row["computed_at"] else None
    try:
        recomputed = ml_margin_svc.apply_hypothetical_price(payload, sale_price)
        enriched["_margin"] = recomputed
    except Exception as err:  # noqa: BLE001
        log.debug("margin recompute failed for %s: %s", item_id, err)
        return

    # Break-even tracker: добавить эту продажу в cumulative проекта/месяца.
    # Project определяется из margin payload (тот же что в /escalar/products).
    project = recomputed.get("project") or payload.get("project")
    if not project:
        return
    unit = recomputed.get("unit") or {}
    profit_variable = unit.get("profit_variable")
    qty = 1
    try:
        qty = int(first.get("quantity") or 1)
    except (TypeError, ValueError):
        qty = 1
    if profit_variable is None:
        return
    try:
        total_profit_var = float(profit_variable) * qty
    except (TypeError, ValueError):
        return

    # parse sale date (BRT) для year_month identification.
    from datetime import datetime as _dt
    sale_dt: Optional[_dt] = None
    raw_dt = enriched.get("date_created") or enriched.get("last_updated")
    if raw_dt:
        try:
            sale_dt = _dt.fromisoformat(str(raw_dt).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            sale_dt = None

    if pool is None:
        return  # без pool не можем — graceful skip
    try:
        await breakeven_svc.ensure_schema(pool)
        be_state = await breakeven_svc.add_sale_and_check_breakeven(
            pool, user_id, str(project), total_profit_var, sale_dt,
        )
        if be_state:
            enriched["_breakeven"] = be_state
    except Exception as err:  # noqa: BLE001
        log.debug("breakeven update failed for project=%s: %s", project, err)


async def _upsert_batch(
    conn: asyncpg.Connection,
    user_id: int,
    items: list[tuple[str, str | None, dict]],
    pool: Optional[asyncpg.Pool] = None,
) -> int:
    saved = 0
    for topic, resource, enriched in items:
        # For orders — pre-enrich with cached unit margin (apply_hypothetical_price
        # at sale_price) so normalize-ветка orders_v2 рисует profit/breakdown.
        # `pool` нужен enricher'у для вторичного pool.acquire() в breakeven
        # tracker'е (не можем reuse `conn` потому что он внутри transaction'а
        # caller'а — `add_sale_and_check_breakeven` делает свой commit).
        if topic in ("orders_v2", "orders"):
            try:
                await _enrich_order_with_margin(conn, user_id, enriched, pool=pool)
            except Exception as err:  # noqa: BLE001
                log.debug("enrich order margin failed: %s", err)
        try:
            notice = ml_normalize.normalize_event(topic, resource, enriched)
        except Exception as err:  # noqa: BLE001
            log.warning("normalize failed for %s: %s", topic, err)
            continue
        try:
            from . import ml_notices as _ml_notices_svc
            await conn.execute(
                _UPSERT_SQL,
                user_id,
                notice["notice_id"],
                notice.get("label"),
                notice.get("description"),
                _ml_notices_svc._coerce_to_datetime(notice.get("from_date")),
                json.dumps(notice.get("tags") or []),
                json.dumps(notice.get("actions") or []),
                json.dumps(notice.get("raw") or {}),
                notice.get("topic"),
                notice.get("resource"),
            )
            saved += 1
        except Exception as err:  # noqa: BLE001
            log.warning("upsert failed for %s: %s", notice.get("notice_id"), err)
    return saved


# ── Entry points ──────────────────────────────────────────────────────────────

async def backfill_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
    days: int = 30,
) -> dict[str, int]:
    """Fetch 5 source endpoints, normalize, bulk-upsert. Returns counts."""
    try:
        access_token, _expires, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("backfill user %s: no valid token: %s", user_id, err)
        return {"user_id": user_id, "fetched": 0, "saved": 0}

    ml_user_id = await _get_ml_user_id(pool, user_id)
    if not ml_user_id:
        return {"user_id": user_id, "fetched": 0, "saved": 0}

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # All 5 in parallel
    orders, questions, claims, items_detail, messages = await asyncio.gather(
        _fetch_orders(http, access_token, ml_user_id, since),
        _fetch_questions(http, access_token),
        _fetch_claims(http, access_token),
        _fetch_paused_closed_items(http, access_token, ml_user_id),
        _fetch_messages(http, access_token, ml_user_id),
        return_exceptions=False,
    )

    batch: list[tuple[str, str | None, dict]] = []
    for o in orders:
        oid = o.get("id")
        if oid:
            batch.append(("orders_v2", f"/orders/{oid}", o))
    for q in questions:
        qid = q.get("id")
        if qid:
            batch.append(("questions", f"/questions/{qid}", q))
    for c in claims:
        cid = c.get("id") or c.get("resource_id")
        if cid:
            batch.append(("claims", f"/post-purchase/v1/claims/{cid}", c))
    for it in items_detail:
        iid = it.get("id")
        if iid:
            batch.append(("items", f"/items/{iid}", it))
    for m in messages:
        mid = m.get("id") or m.get("message_id") or m.get("pack_id")
        if mid:
            batch.append(("messages", f"/messages/{mid}", m))

    fetched = len(batch)
    saved = 0
    if batch:
        async with pool.acquire() as conn:
            saved = await _upsert_batch(conn, user_id, batch, pool=pool)

    log.info("backfill user=%s days=%s fetched=%s saved=%s", user_id, days, fetched, saved)
    return {"user_id": user_id, "fetched": fetched, "saved": saved}


async def backfill_all_users(pool: asyncpg.Pool, days: int = 1) -> dict[str, int]:
    """APScheduler daily catch-up. Runs backfill_user for every known ML-connected user."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
        )
    totals = {"users": 0, "fetched": 0, "saved": 0}
    if not rows:
        return totals

    async with httpx.AsyncClient() as http:
        for r in rows:
            try:
                res = await backfill_user(pool, http, r["user_id"], days=days)
                totals["users"] += 1
                totals["fetched"] += res["fetched"]
                totals["saved"] += res["saved"]
            except Exception as err:  # noqa: BLE001
                log.exception("daily backfill user %s failed: %s", r["user_id"], err)
            await asyncio.sleep(RATE_SLEEP)
    return totals
