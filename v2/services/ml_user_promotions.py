"""Promotions cache — ML `/seller-promotions/items/{item_id}?app_version=v2`.

Mirrors the ml_user_items / ml_user_questions pattern. Caches per-item offers
so /escalar/promotions UI reads instantly from DB instead of hitting ML for
each row. New `candidate` offers get pushed into ml_notices (topic=promotions)
which the existing TG dispatch cron sends to the seller with inline buttons.

ML response shape (verified empirically — see /promotions/probe):
  GET /seller-promotions/items/MLB123?app_version=v2 → array of offer dicts:
    [{ "id": "MLB-PROMO-...", "type": "DOD" | "LIGHTNING" | "SELLER_CAMPAIGN" | ...,
       "sub_type": "FLEXIBLE_PERCENTAGE" | "FIXED_AMOUNT" | ...,
       "status": "candidate" | "started" | "finished" | "pending",
       "offer_id": ..., "deal_price": 79.90, "original_price": 99.90,
       "discount_percentage": 20.0,
       "start_date": "2026-04-25T00:00:00Z", "finish_date": "...",
       ... }]

When the item has no eligible offers (or returns 400 'Item status is not
allowed'), we just record nothing for that item.
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
RATE_SLEEP = 0.2  # 5 req/sec per user — same as ml_quality

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_user_promotions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  promotion_id TEXT NOT NULL,
  promotion_type TEXT,
  sub_type TEXT,
  status TEXT,
  offer_id TEXT,
  original_price NUMERIC,
  deal_price NUMERIC,
  discount_percentage NUMERIC,
  start_date TIMESTAMPTZ,
  finish_date TIMESTAMPTZ,
  raw JSONB,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  notified_at TIMESTAMPTZ,
  UNIQUE(user_id, item_id, promotion_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_promo_user_status ON ml_user_promotions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_ml_promo_user_item ON ml_user_promotions(user_id, item_id);
"""

ALTER_SQL = """
ALTER TABLE ml_user_promotions
  ADD COLUMN IF NOT EXISTS dismissed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_reminder_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS reminder_count INT NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_ml_promo_pending_dispatch
  ON ml_user_promotions(user_id, notified_at, last_reminder_at)
  WHERE status = 'candidate' AND dismissed_at IS NULL AND accepted_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)
        await conn.execute(ALTER_SQL)


# ── ML API helpers ────────────────────────────────────────────────────────────

def _parse_dt(v: Any):
    from datetime import datetime
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        s = str(v).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _parse_num(v: Any):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_mlb(raw: Any) -> str:
    """Same mlb cleanup as ml_quality.normalize_item_id but inlined."""
    s = str(raw or "").strip().upper()
    if not s:
        return ""
    numeric = s[3:] if s.startswith("MLB") else s
    if "." in numeric:
        head, _, tail = numeric.partition(".")
        if tail.strip("0") == "":
            numeric = head
        else:
            return ""
    if not numeric.isdigit():
        return ""
    return f"MLB{numeric}"


async def _fetch_offers_for_item(
    http: httpx.AsyncClient,
    token: str,
    mlb: str,
) -> list[dict]:
    """Returns offer dicts. 400 (item ineligible) → []. Network errors → []."""
    url = f"{ML_API_BASE}/seller-promotions/items/{mlb}?app_version=v2"
    try:
        r = await http.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("promo fetch %s failed: %s", mlb, err)
        return []
    if r.status_code == 400:
        # Item under_review / paused / closed — no offers possible. Normal.
        return []
    if r.status_code != 200:
        log.warning("promo fetch %s status=%s body=%s", mlb, r.status_code, r.text[:200])
        return []
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        return []
    # ML responds as array, sometimes wrapped {offers:[...]} or {results:[...]}
    if isinstance(data, list):
        return [o for o in data if isinstance(o, dict)]
    if isinstance(data, dict):
        for key in ("offers", "results", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return [o for o in v if isinstance(o, dict)]
    return []


# ── Refresh ───────────────────────────────────────────────────────────────────

async def _upsert_offer(
    conn: asyncpg.Connection,
    user_id: int,
    item_id: str,
    offer: dict,
) -> bool:
    """Returns True if this offer is NEW (never seen before for this user)."""
    promo_id = offer.get("id") or offer.get("promotion_id")
    if not promo_id:
        return False
    promo_id = str(promo_id)
    promo_type = offer.get("type") or offer.get("promotion_type")
    sub_type = offer.get("sub_type")
    status = offer.get("status")
    offer_id = offer.get("offer_id")
    if offer_id is not None:
        offer_id = str(offer_id)
    original_price = _parse_num(offer.get("original_price"))
    # ML returns the discounted price under different keys per offer type:
    #   DEAL/SELLER_CAMPAIGN  → deal_price
    #   LIGHTNING/SMART/PRICE_MATCHING/UNHEALTHY_STOCK → price (current promo
    #     price) or suggested_discounted_price (recommended)
    deal_price = _parse_num(
        offer.get("deal_price")
        or offer.get("price")
        or offer.get("suggested_discounted_price")
    )
    discount_pct = _parse_num(offer.get("discount_percentage"))
    # Some offer types only carry meli_percentage + seller_percentage;
    # the effective buyer-facing discount is the sum.
    if discount_pct is None:
        meli_pct = _parse_num(offer.get("meli_percentage"))
        seller_pct = _parse_num(offer.get("seller_percentage"))
        if meli_pct is not None or seller_pct is not None:
            discount_pct = (meli_pct or 0.0) + (seller_pct or 0.0)
    # Fall back to computing discount from the price pair when ML gave neither.
    if discount_pct is None and original_price and deal_price:
        try:
            discount_pct = round(
                (1 - deal_price / original_price) * 100, 1,
            )
        except (ZeroDivisionError, TypeError):
            discount_pct = None
    start_date = _parse_dt(offer.get("start_date"))
    finish_date = _parse_dt(offer.get("finish_date"))

    # Detect "is new" by checking if row exists before upsert.
    existed = await conn.fetchval(
        "SELECT 1 FROM ml_user_promotions WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3",
        user_id, item_id, promo_id,
    )
    is_new = existed is None

    await conn.execute(
        """
        INSERT INTO ml_user_promotions
          (user_id, item_id, promotion_id, promotion_type, sub_type, status,
           offer_id, original_price, deal_price, discount_percentage,
           start_date, finish_date, raw, fetched_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb, NOW())
        ON CONFLICT (user_id, item_id, promotion_id) DO UPDATE SET
          promotion_type      = EXCLUDED.promotion_type,
          sub_type            = EXCLUDED.sub_type,
          status              = EXCLUDED.status,
          offer_id            = EXCLUDED.offer_id,
          original_price      = EXCLUDED.original_price,
          deal_price          = EXCLUDED.deal_price,
          discount_percentage = EXCLUDED.discount_percentage,
          start_date          = EXCLUDED.start_date,
          finish_date         = EXCLUDED.finish_date,
          raw                 = EXCLUDED.raw,
          fetched_at          = NOW()
        """,
        user_id, item_id, promo_id,
        promo_type, sub_type, status,
        offer_id, original_price, deal_price, discount_pct,
        start_date, finish_date,
        json.dumps(offer, default=str),
    )
    return is_new


async def refresh_user_promotions(
    pool: asyncpg.Pool,
    user_id: int,
    item_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Scan all (or given) user items for promo offers, upsert into DB.

    Returns dict with:
      - fetched: how many items probed
      - upserted: how many offer rows touched
      - new_offers: list of dicts {item_id, promo_id, promo_type, raw} for
        offers that didn't previously exist — caller pushes these to ml_notices
        for TG dispatch.
    """
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError:
        return {"fetched": 0, "upserted": 0, "new_offers": []}

    # If item_ids not provided, pull from ml_user_items cache (active items only).
    if item_ids is None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT item_id FROM ml_user_items
                 WHERE user_id = $1 AND (status = 'active' OR status IS NULL)
                """,
                user_id,
            )
        item_ids = [r["item_id"] for r in rows]

    # Normalize + dedupe.
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in item_ids:
        m = _normalize_mlb(raw)
        if m and m not in seen:
            seen.add(m)
            cleaned.append(m)

    if not cleaned:
        return {"fetched": 0, "upserted": 0, "new_offers": []}

    fetched = 0
    upserted = 0
    new_offers: list[dict] = []
    async with httpx.AsyncClient() as http:
        for mlb in cleaned:
            offers = await _fetch_offers_for_item(http, token, mlb)
            fetched += 1
            if not offers:
                await asyncio.sleep(RATE_SLEEP)
                continue
            try:
                fetched_promo_ids: list[str] = []
                async with pool.acquire() as conn:
                    for offer in offers:
                        is_new = await _upsert_offer(conn, user_id, mlb, offer)
                        upserted += 1
                        promo_id_str = str(offer.get("id") or offer.get("promotion_id") or "")
                        if promo_id_str:
                            fetched_promo_ids.append(promo_id_str)
                        if is_new:
                            new_offers.append({
                                "item_id": mlb,
                                "promotion_id": promo_id_str,
                                "promotion_type": offer.get("type") or offer.get("promotion_type") or "",
                                "sub_type": offer.get("sub_type") or "",
                                "status": offer.get("status") or "",
                                "deal_price": offer.get("deal_price"),
                                "original_price": offer.get("original_price"),
                                "discount_percentage": offer.get("discount_percentage"),
                                "raw": offer,
                            })
                    # Candidates that ML no longer returns for this item were
                    # accepted/rejected outside TG (web UI, app, API). Mark
                    # them dismissed so the dispatcher stops sending reminders.
                    await conn.execute(
                        """
                        UPDATE ml_user_promotions
                           SET dismissed_at = NOW()
                         WHERE user_id = $1 AND item_id = $2
                           AND status = 'candidate'
                           AND dismissed_at IS NULL AND accepted_at IS NULL
                           AND promotion_id != ALL($3::text[])
                        """,
                        user_id, mlb, fetched_promo_ids,
                    )
            except Exception as err:  # noqa: BLE001
                log.exception("upsert promo for %s failed: %s", mlb, err)
            await asyncio.sleep(RATE_SLEEP)

    return {"fetched": fetched, "upserted": upserted, "new_offers": new_offers}


# ── Cache readback ────────────────────────────────────────────────────────────

async def get_cached(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: Optional[str] = None,
) -> dict[str, Any]:
    """If item_id given → returns single-item shape `{itemId, offers, fetchedAt}`.
    If omitted → returns map `{items: {item_id: [offers]}, totalOffers, fetchedAt}`.
    """
    where = "WHERE user_id = $1"
    params: list[Any] = [user_id]
    if item_id:
        norm = _normalize_mlb(item_id)
        if norm:
            where += " AND item_id = $2"
            params.append(norm)
        else:
            return {"itemId": item_id, "offers": [], "fetchedAt": None}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT item_id, promotion_id, promotion_type, sub_type, status,
                   offer_id, original_price, deal_price, discount_percentage,
                   to_char(start_date AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS start_date,
                   to_char(finish_date AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS finish_date,
                   raw,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_promotions
              {where}
             ORDER BY status NULLS LAST, fetched_at DESC
            """,
            *params,
        )

    def _row_to_offer(r) -> dict:
        raw = r["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        # Return ML-shape compatible with existing UI:
        return {
            "id": r["promotion_id"],
            "type": r["promotion_type"],
            "sub_type": r["sub_type"],
            "status": r["status"],
            "offer_id": r["offer_id"],
            "original_price": float(r["original_price"]) if r["original_price"] is not None else None,
            "deal_price": float(r["deal_price"]) if r["deal_price"] is not None else None,
            "discount_percentage": float(r["discount_percentage"]) if r["discount_percentage"] is not None else None,
            "start_date": r["start_date"],
            "finish_date": r["finish_date"],
            **(raw if isinstance(raw, dict) else {}),
        }

    if item_id:
        offers = [_row_to_offer(r) for r in rows]
        return {
            "itemId": _normalize_mlb(item_id),
            "offers": offers,
            "fetchedAt": rows[0]["fetched_at"] if rows else None,
        }

    items_map: dict[str, list[dict]] = {}
    latest = None
    for r in rows:
        items_map.setdefault(r["item_id"], []).append(_row_to_offer(r))
        if latest is None or (r["fetched_at"] and r["fetched_at"] > latest):
            latest = r["fetched_at"]
    return {
        "items": items_map,
        "totalOffers": len(rows),
        "fetchedAt": latest,
    }


async def mark_notified(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    promotion_id: str,
) -> None:
    """Stamp the row as having been pushed into ml_notices, so cron loops
    don't re-push it on every tick."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ml_user_promotions
               SET notified_at = NOW()
             WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
            """,
            user_id, item_id, promotion_id,
        )


async def mark_accepted(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    promotion_id: str,
) -> None:
    """User accepted via TG button. Stops further reminders for this offer."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ml_user_promotions
               SET accepted_at = NOW(), notified_at = COALESCE(notified_at, NOW())
             WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
            """,
            user_id, item_id, promotion_id,
        )


async def mark_dismissed(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    promotion_id: str,
) -> None:
    """User rejected via TG button. Stops further reminders for this offer."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ml_user_promotions
               SET dismissed_at = NOW(), notified_at = COALESCE(notified_at, NOW())
             WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
            """,
            user_id, item_id, promotion_id,
        )


async def dispatch_pending_candidates(
    pool: asyncpg.Pool,
    user_id: int,
    normalize_event,
    upsert_notice,
    limit: int = 15,
    reminder_hours: float = 24.0,
) -> dict[str, int]:
    """Push first-time + reminder TG notices for `candidate` offers the user
    has not yet acted on.

    Sends in two cases:
      - never sent: notified_at IS NULL
      - reminder due: last_reminder_at IS NULL ⇒ notified_at + reminder_hours < NOW()
                      OR last_reminder_at + reminder_hours < NOW()

    Stops as soon as the user accepts (accepted_at), rejects (dismissed_at), or
    the offer transitions out of `candidate` (e.g., started/finished/expired).

    `normalize_event` and `upsert_notice` are passed in to avoid a circular
    import with v2.services.ml_normalize / ml_notices (this module is a leaf
    in the dependency graph).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT p.item_id, p.promotion_id, p.promotion_type, p.sub_type,
                   p.status, p.offer_id, p.original_price, p.deal_price,
                   p.discount_percentage,
                   p.start_date,
                   p.finish_date,
                   p.raw,
                   p.notified_at,
                   p.last_reminder_at,
                   p.reminder_count,
                   i.title AS item_title,
                   i.permalink AS item_permalink,
                   i.thumbnail AS item_thumbnail,
                   i.price AS item_price
              FROM ml_user_promotions p
              LEFT JOIN ml_user_items i
                ON i.user_id = p.user_id AND i.item_id = p.item_id
             WHERE p.user_id = $1
               AND p.status = 'candidate'
               AND p.dismissed_at IS NULL
               AND p.accepted_at IS NULL
               AND (
                 p.notified_at IS NULL
                 OR (p.last_reminder_at IS NULL
                     AND p.notified_at + ($2 * INTERVAL '1 hour') < NOW())
                 OR p.last_reminder_at + ($2 * INTERVAL '1 hour') < NOW()
               )
             ORDER BY p.notified_at ASC NULLS FIRST, p.fetched_at ASC
             LIMIT $3
            """,
            user_id, reminder_hours, limit,
        )

    sent_first = 0
    sent_reminder = 0
    for row in rows:
        raw = row["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        if not isinstance(raw, dict):
            raw = {}

        enriched: dict = dict(raw)
        enriched["item_id"] = row["item_id"]
        enriched["id"] = row["promotion_id"]
        enriched["promotion_id"] = row["promotion_id"]
        enriched["type"] = row["promotion_type"]
        enriched["sub_type"] = row["sub_type"]
        enriched["status"] = row["status"]
        enriched["original_price"] = (
            float(row["original_price"]) if row["original_price"] is not None
            else (float(row["item_price"]) if row["item_price"] is not None else None)
        )
        enriched["deal_price"] = (
            float(row["deal_price"]) if row["deal_price"] is not None else None
        )
        enriched["discount_percentage"] = (
            float(row["discount_percentage"])
            if row["discount_percentage"] is not None else None
        )
        enriched["start_date"] = row["start_date"]
        enriched["finish_date"] = row["finish_date"]
        enriched["_item_title"] = row["item_title"] or ""
        enriched["_item_permalink"] = row["item_permalink"] or ""
        enriched["_item_thumbnail"] = row["item_thumbnail"] or ""
        is_reminder = row["notified_at"] is not None
        enriched["_is_reminder"] = is_reminder
        enriched["_reminder_count"] = int(row["reminder_count"] or 0)

        notice = normalize_event("promotions", None, enriched)
        notice["notice_id"] = (
            f"promotions:{row['item_id']}:{row['promotion_id']}"
        )
        # Reminders share the same notice_id ⇒ ON CONFLICT in ml_notices would
        # update the existing row without re-sending. Suffix with reminder
        # count so each reminder is treated as a fresh notice.
        if is_reminder:
            notice["notice_id"] = (
                f"{notice['notice_id']}:r{int(row['reminder_count'] or 0) + 1}"
            )

        try:
            pushed = await upsert_notice(pool, user_id, notice)
        except Exception as err:  # noqa: BLE001
            log.exception(
                "promo notice upsert failed user=%s promo=%s: %s",
                user_id, row["promotion_id"], err,
            )
            continue
        if not pushed:
            continue

        async with pool.acquire() as conn:
            if is_reminder:
                await conn.execute(
                    """
                    UPDATE ml_user_promotions
                       SET last_reminder_at = NOW(),
                           reminder_count = reminder_count + 1
                     WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
                    """,
                    user_id, row["item_id"], row["promotion_id"],
                )
                sent_reminder += 1
            else:
                await conn.execute(
                    """
                    UPDATE ml_user_promotions
                       SET notified_at = NOW()
                     WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
                    """,
                    user_id, row["item_id"], row["promotion_id"],
                )
                sent_first += 1

    return {"sent_first": sent_first, "sent_reminder": sent_reminder}


async def get_offer(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    promotion_id: str,
) -> Optional[dict]:
    """Single-row lookup — used by TG callback handler to recover deal_price
    and offer_id when accepting from the keyboard."""
    norm = _normalize_mlb(item_id)
    if not norm:
        return None
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            """
            SELECT promotion_id, promotion_type, sub_type, status, offer_id,
                   original_price, deal_price, discount_percentage, raw
              FROM ml_user_promotions
             WHERE user_id = $1 AND item_id = $2 AND promotion_id = $3
            """,
            user_id, norm, promotion_id,
        )
    if not r:
        return None
    raw = r["raw"]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            raw = {}
    return {
        "promotion_id": r["promotion_id"],
        "promotion_type": r["promotion_type"],
        "sub_type": r["sub_type"],
        "status": r["status"],
        "offer_id": r["offer_id"],
        "original_price": float(r["original_price"]) if r["original_price"] is not None else None,
        "deal_price": float(r["deal_price"]) if r["deal_price"] is not None else None,
        "discount_percentage": float(r["discount_percentage"]) if r["discount_percentage"] is not None else None,
        "raw": raw,
    }
