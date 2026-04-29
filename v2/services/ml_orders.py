"""ML orders cache — `/orders/search` paginated + per-day aggregation helpers.

Why this exists: daily_summary_dispatch and photo_ab_dispatch were both
reading orders from `db_loader.load_user_vendas` which is a parser over
manually-uploaded CSV files. As soon as the seller stops uploading
fresh CSVs, "yesterday's" data is missing and summary cards arrive in
TG saying "0 sales" while visits/ads (which DO come from live ML
caches) show normal numbers. That breaks the product's "точность из
файлов" promise.

This module pulls orders directly from ML and caches them in
`ml_user_orders`. Two consumers:
  - daily_summary_dispatch._aggregate_user_metrics → get_orders_for_day
  - photo_ab_dispatch._aggregate_window_for_item   → get_orders_for_window

Pattern: TEST → БД → КЭШ (per CLAUDE.md). The probe endpoint lives in
escalar.py; this file owns schema + refresh + getters.

Cancel filter: ML `status` values "cancelled" / "invalid" are skipped
when summing revenue/orders (refunded post-delivery is a separate
concept handled via Vendas + retirada — out of scope for this module).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
PAGE_LIMIT = 50  # ML's max
MAX_PAGES = 40   # 50 * 40 = 2000 orders per refresh window — generous
RATE_SLEEP = 0.2

# Brazil timezone — matches daily_summary_dispatch / photo_ab_dispatch.
BRT = timezone(timedelta(hours=-3))

# ML statuses that mean the order didn't actually convert. Anything else
# (paid / payment_in_process / payment_required / partially_refunded) is
# counted as a real sale for summary purposes.
EXCLUDED_STATUSES = {"cancelled", "invalid"}


# ── Schema ─────────────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_user_orders (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  order_id BIGINT NOT NULL,
  pack_id BIGINT,
  date_created TIMESTAMPTZ NOT NULL,
  date_closed TIMESTAMPTZ,
  status TEXT,
  status_detail TEXT,
  total_amount NUMERIC,
  currency TEXT,
  buyer_id BIGINT,
  -- Items denormalized as JSONB. One order has 1..N items. Each entry:
  --   { mlb, title, variation_id, quantity, unit_price, sale_fee, category_id }
  -- We flatten what's needed for top/worst-by-revenue + per-item baseline.
  items JSONB NOT NULL DEFAULT '[]'::jsonb,
  tags JSONB,
  raw JSONB,                  -- full payload for debugging / future fields
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  -- Per-sale TG notification tracking. NULL = ещё не нотифицирован, dispatcher
  -- отправит. Чтобы при первой массовой загрузке истории НЕ спамить TG старыми
  -- ордерами — refresh-функция marks all backfilled orders как уже
  -- notified_at=fetched_at. Только реально новые сделки (date_created в окне
  -- последних 24ч) проходят отдельный mark и попадают в очередь.
  notified_at TIMESTAMPTZ,
  UNIQUE(user_id, order_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_user_orders_user_date
  ON ml_user_orders(user_id, date_created DESC);
CREATE INDEX IF NOT EXISTS idx_ml_user_orders_user_status
  ON ml_user_orders(user_id, status);

-- Idempotent migration для existing deploy: добавление колонки — metadata-only
-- в Postgres 11+, мгновенное даже на больших таблицах.
ALTER TABLE ml_user_orders ADD COLUMN IF NOT EXISTS notified_at TIMESTAMPTZ;

-- ВАЖНО: partial index на notified_at IS NULL вынесен из этого блока
-- специально. CREATE INDEX без CONCURRENTLY блокирует таблицу при сборке;
-- на существующей продуктивной ml_user_orders это занимает >20s и
-- Railway healthcheck падает по timeout. Если таблица станет очень большой
-- и dispatch_pending_sales начнёт тормозить — добавить отдельной миграцией:
--   CREATE INDEX CONCURRENTLY idx_ml_user_orders_pending_notify
--     ON ml_user_orders(user_id, date_created DESC)
--     WHERE notified_at IS NULL;
-- (CONCURRENTLY нельзя в transaction, поэтому отдельным execute, не здесь.)
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── ML API helpers ─────────────────────────────────────────────────────────────


def _iso_brt(dt: datetime) -> str:
    """ML wants ISO with offset, e.g. 2026-04-25T00:00:00.000-03:00."""
    return dt.astimezone(BRT).strftime("%Y-%m-%dT%H:%M:%S.000-03:00")


async def _search_orders_page(
    http: httpx.AsyncClient,
    token: str,
    seller_id: int,
    date_from_brt: datetime,
    date_to_brt: datetime,
    offset: int,
) -> tuple[list[dict], int]:
    """One page of /orders/search. Returns (results, total_count_or_-1)."""
    url = (
        f"{ML_API_BASE}/orders/search"
        f"?seller={seller_id}"
        f"&order.date_created.from={_iso_brt(date_from_brt)}"
        f"&order.date_created.to={_iso_brt(date_to_brt)}"
        f"&offset={offset}&limit={PAGE_LIMIT}"
        f"&sort=date_desc"
    )
    try:
        r = await http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20.0)
    except Exception as err:  # noqa: BLE001
        log.warning("orders/search seller=%s offset=%s failed: %s", seller_id, offset, err)
        return [], -1
    if r.status_code != 200:
        log.warning("orders/search seller=%s offset=%s status=%s body=%s",
                    seller_id, offset, r.status_code, r.text[:200])
        return [], -1
    data = r.json() or {}
    results = data.get("results") or []
    total = (data.get("paging") or {}).get("total", -1)
    return results, total


def _parse_dt(s: Any) -> Optional[datetime]:
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


def _slim_items(order: dict) -> list[dict]:
    """Reduce ML's order_items[] to just the fields we use later. Avoids
    storing 5x the bytes per row when we only ever read mlb / units /
    revenue / sale_fee."""
    out: list[dict] = []
    for raw_item in order.get("order_items") or []:
        if not isinstance(raw_item, dict):
            continue
        inner = raw_item.get("item") or {}
        try:
            quantity = int(raw_item.get("quantity") or 0)
        except (TypeError, ValueError):
            quantity = 0
        try:
            unit_price = float(raw_item.get("unit_price") or 0)
        except (TypeError, ValueError):
            unit_price = 0.0
        try:
            sale_fee = float(raw_item.get("sale_fee") or 0)
        except (TypeError, ValueError):
            sale_fee = 0.0
        out.append({
            "mlb": inner.get("id"),
            "title": inner.get("title"),
            "category_id": inner.get("category_id"),
            "variation_id": inner.get("variation_id"),
            "seller_sku": inner.get("seller_sku"),
            "quantity": quantity,
            "unit_price": unit_price,
            "sale_fee": sale_fee,
            "revenue": unit_price * quantity,
        })
    return out


async def _upsert_order(conn: asyncpg.Connection, user_id: int, order: dict) -> None:
    """Upsert ml_user_orders row.

    Notify-tracking: только реально свежие orders (date_created в окне последних
    24ч от now) идут с `notified_at = NULL` — они потом подхватятся диспатчером.
    Старые orders (история, бэкфилл) сразу marked notified_at=NOW(), чтобы
    при первой массовой загрузке за неделю/месяц TG-чат не получил 100+
    устаревших уведомлений. Если order уже existed — оставляем notified_at
    как было (не сбрасываем).
    """
    order_id = order.get("id")
    if order_id is None:
        return
    items = _slim_items(order)
    buyer = order.get("buyer") or {}
    date_created = _parse_dt(order.get("date_created"))
    # Защита от спама: backfill orders помечаем как уже notified.
    is_recent = False
    if date_created is not None:
        try:
            now_utc = datetime.now(timezone.utc)
            is_recent = (now_utc - date_created.astimezone(timezone.utc)).total_seconds() < 86400
        except Exception:  # noqa: BLE001
            is_recent = False
    initial_notified_at = None if is_recent else datetime.now(timezone.utc)
    await conn.execute(
        """
        INSERT INTO ml_user_orders
          (user_id, order_id, pack_id, date_created, date_closed,
           status, status_detail, total_amount, currency, buyer_id,
           items, tags, raw, fetched_at, notified_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11::jsonb, $12::jsonb, $13::jsonb, NOW(), $14)
        ON CONFLICT (user_id, order_id) DO UPDATE SET
          pack_id = EXCLUDED.pack_id,
          date_created = EXCLUDED.date_created,
          date_closed = EXCLUDED.date_closed,
          status = EXCLUDED.status,
          status_detail = EXCLUDED.status_detail,
          total_amount = EXCLUDED.total_amount,
          currency = EXCLUDED.currency,
          buyer_id = EXCLUDED.buyer_id,
          items = EXCLUDED.items,
          tags = EXCLUDED.tags,
          raw = EXCLUDED.raw,
          fetched_at = NOW()
          -- ВАЖНО: notified_at НЕ обновляем — сохраняем существующее значение
        """,
        user_id,
        int(order_id),
        int(order["pack_id"]) if order.get("pack_id") else None,
        date_created,
        _parse_dt(order.get("date_closed")),
        order.get("status"),
        order.get("status_detail"),
        float(order["total_amount"]) if order.get("total_amount") is not None else None,
        order.get("currency_id"),
        int(buyer["id"]) if buyer.get("id") else None,
        json.dumps(items, default=str),
        json.dumps(order.get("tags") or [], default=str),
        json.dumps(order, default=str),
        initial_notified_at,
    )


# ── Public refresh ─────────────────────────────────────────────────────────────


async def refresh_for_period(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    days_back: int = 2,
    end_date_brt: Optional[date] = None,
) -> dict[str, int]:
    """Pull ML orders for [end - days_back, end] BRT and upsert to cache.

    Defaults to "yesterday + buffer" (days_back=2 → yesterday + today).
    For Photo A/B baseline pass days_back = duration_days + 2 so we
    cover the full baseline window plus a small tail for late-arriving
    orders (ML can backfill order metadata for a few hours).
    """
    await ensure_schema(pool)

    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("orders refresh user=%s skipped: %s", user_id, err)
        return {"fetched": 0, "saved": 0, "pages": 0, "skipped": "no_token"}

    tokens = await ml_oauth_svc.load_user_tokens(pool, user_id) or {}
    seller_id = tokens.get("ml_user_id")
    if not seller_id:
        log.warning("orders refresh user=%s skipped: no ml_user_id in tokens", user_id)
        return {"fetched": 0, "saved": 0, "pages": 0, "skipped": "no_seller_id"}

    if end_date_brt is None:
        end_date_brt = datetime.now(BRT).date()

    # End of `end_date_brt` (inclusive of the whole day).
    end_dt = datetime.combine(end_date_brt, datetime.max.time(), tzinfo=BRT)
    start_dt = datetime.combine(
        end_date_brt - timedelta(days=days_back), datetime.min.time(), tzinfo=BRT,
    )

    fetched = 0
    saved = 0
    pages = 0
    async with httpx.AsyncClient() as http:
        offset = 0
        for _ in range(MAX_PAGES):
            results, total = await _search_orders_page(
                http, token, int(seller_id), start_dt, end_dt, offset,
            )
            pages += 1
            if not results:
                break
            fetched += len(results)
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for order in results:
                        try:
                            await _upsert_order(conn, user_id, order)
                            saved += 1
                        except Exception as err:  # noqa: BLE001
                            log.warning("upsert order %s failed: %s",
                                        order.get("id"), err)
            if len(results) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT
            await asyncio.sleep(RATE_SLEEP)

    log.info("orders refresh user=%s seller=%s window=[%s..%s] pages=%s fetched=%s saved=%s",
             user_id, seller_id, start_dt.date(), end_date_brt, pages, fetched, saved)

    # Hook: после refresh — пробуем разослать TG-уведомления для свежих
    # заказов которых ещё не было в БД (notified_at IS NULL). Backfill orders
    # уже помечены NOW() в _upsert_order, так что в выборку не попадут.
    try:
        notify_stats = await dispatch_pending_sales(pool, user_id)
    except Exception as err:  # noqa: BLE001
        log.warning("dispatch_pending_sales user=%s failed: %s", user_id, err)
        notify_stats = {"error": str(err)}

    return {
        "fetched": fetched, "saved": saved, "pages": pages,
        "window_from": start_dt.isoformat(), "window_to": end_dt.isoformat(),
        "notify": notify_stats,
    }


# ── Per-sale TG dispatch ──────────────────────────────────────────────────────

async def dispatch_pending_sales(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    max_per_run: int = 20,
) -> dict[str, int]:
    """Push TG notification for orders с notified_at IS NULL.

    Backfill-защита: сами refresh уже помечают «старые» orders
    notified_at=NOW() (см. _upsert_order is_recent). В выборке остаются
    только реально свежие сделки последних 24ч которых не было в БД до
    последнего refresh.

    Per-sale render: revenue/profit/налоги через
    `ml_item_margin.apply_hypothetical_price(cached, sale_price)` —
    переиспользует ту же формулу что показывает promotions notifications
    (DAS, ML fees, frete, fulfillment, COGS, armazenagem).

    Status filter: пропускаем `cancelled`/`invalid` orders — они
    помечаются notified_at=NOW() чтобы не попасть в следующий retry.

    Возвращает {sent, skipped_cancelled, skipped_no_item, marked}.
    """
    from . import ml_normalize as ml_norm
    from . import ml_notices as ml_notices_svc
    from . import ml_item_margin as ml_margin_svc

    sent = 0
    skipped_cancelled = 0
    skipped_no_item = 0
    marked = 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT order_id, status, total_amount, currency, items, date_created
              FROM ml_user_orders
             WHERE user_id = $1
               AND notified_at IS NULL
             ORDER BY date_created DESC
             LIMIT $2
            """,
            user_id, max_per_run,
        )

    for row in rows:
        order_id = int(row["order_id"])
        status = (row["status"] or "").lower()
        if status in EXCLUDED_STATUSES:
            # Cancel/invalid — не нотифицируем и помечаем чтобы не проверять снова.
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ml_user_orders SET notified_at = NOW() "
                    " WHERE user_id = $1 AND order_id = $2",
                    user_id, order_id,
                )
            skipped_cancelled += 1
            marked += 1
            continue

        # Берём первый item (самый частый кейс — один item per order). Multi-item
        # orders редкие и сложные в TG — для них пока показываем только первый.
        items = row["items"] or []
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except json.JSONDecodeError:
                items = []
        if not items:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ml_user_orders SET notified_at = NOW() "
                    " WHERE user_id = $1 AND order_id = $2",
                    user_id, order_id,
                )
            skipped_no_item += 1
            marked += 1
            continue

        first = items[0]
        item_id = str(first.get("mlb") or "").upper()
        title = str(first.get("title") or "")
        qty = int(first.get("quantity") or 1)
        try:
            sale_price = float(first.get("unit_price") or 0)
        except (TypeError, ValueError):
            sale_price = 0.0
        try:
            ml_fee = float(first.get("sale_fee") or 0)
        except (TypeError, ValueError):
            ml_fee = 0.0

        # Pull cached unit margin (3-month average) and re-derive at sale_price.
        cached = None
        try:
            cached = await ml_margin_svc.get_cached_margin(pool, user_id, item_id)
        except Exception as err:  # noqa: BLE001
            log.debug("margin load %s failed: %s", item_id, err)
        margin_re: dict | None = None
        if cached and sale_price > 0:
            try:
                margin_re = ml_margin_svc.apply_hypothetical_price(cached, sale_price)
            except Exception as err:  # noqa: BLE001
                log.debug("margin recompute %s failed: %s", item_id, err)

        permalink = f"https://www.mercadolivre.com.br/vendas/{order_id}/detalhe"
        enriched = {
            "order_id": order_id,
            "item_id": item_id,
            "title": title,
            "qty": qty,
            "sale_price": sale_price,
            "ml_fee": ml_fee,
            "currency": row["currency"] or "BRL",
            "total_amount": float(row["total_amount"] or 0),
            "date_created": row["date_created"].isoformat() if row["date_created"] else None,
            "_margin": margin_re,
            "_permalink": permalink,
        }
        notice = ml_norm.normalize_event(
            "sales", str(order_id), enriched,
        )
        try:
            await ml_notices_svc.upsert_normalized(pool, user_id, notice)
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ml_user_orders SET notified_at = NOW() "
                    " WHERE user_id = $1 AND order_id = $2",
                    user_id, order_id,
                )
            sent += 1
            marked += 1
        except Exception as err:  # noqa: BLE001
            log.warning("dispatch sale order=%s failed: %s", order_id, err)

    log.info(
        "dispatch_pending_sales user=%s sent=%s cancelled=%s no_item=%s",
        user_id, sent, skipped_cancelled, skipped_no_item,
    )
    return {
        "sent": sent,
        "skipped_cancelled": skipped_cancelled,
        "skipped_no_item": skipped_no_item,
        "marked": marked,
    }


# ── Public getters ─────────────────────────────────────────────────────────────


def _brt_day_bounds(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=BRT)
    end = start + timedelta(days=1) - timedelta(microseconds=1)
    return start, end


async def get_orders_for_day(
    pool: asyncpg.Pool,
    user_id: int,
    target_date_brt: date,
) -> dict[str, Any]:
    """All non-cancelled orders for one BRT day, aggregated.

    Returned shape mirrors what daily_summary_dispatch._aggregate_user_metrics
    used to build manually:
      orders_count, revenue, items_per_mlb [{mlb, title, units, revenue}]
    Excludes cancelled/invalid statuses.
    """
    start, end = _brt_day_bounds(target_date_brt)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT order_id, total_amount, status, items
              FROM ml_user_orders
             WHERE user_id = $1
               AND date_created BETWEEN $2 AND $3
            """,
            user_id, start, end,
        )

    orders_count = 0
    revenue = 0.0
    items_per_mlb: dict[str, dict[str, Any]] = {}

    for row in rows:
        status = (row["status"] or "").lower()
        if status in EXCLUDED_STATUSES:
            continue
        orders_count += 1
        revenue += float(row["total_amount"] or 0)
        items_raw = row["items"]
        if isinstance(items_raw, str):
            try:
                items_raw = json.loads(items_raw)
            except Exception:  # noqa: BLE001
                items_raw = []
        if not isinstance(items_raw, list):
            continue
        for it in items_raw:
            mlb = (it or {}).get("mlb")
            if not mlb:
                continue
            slot = items_per_mlb.setdefault(mlb, {
                "mlb": mlb,
                "title": it.get("title") or "",
                "units": 0,
                "revenue": 0.0,
            })
            slot["units"] += int(it.get("quantity") or 0)
            slot["revenue"] += float(it.get("revenue") or 0)
            if not slot["title"] and it.get("title"):
                slot["title"] = it["title"]

    return {
        "orders_count": orders_count,
        "revenue": revenue,
        "items_per_mlb": items_per_mlb,
    }


async def get_orders_for_window(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    """Total orders/units/revenue for a specific MLB over [start, end].

    Used by photo_ab_dispatch for baseline + treatment windows. Filters
    on items JSONB (any element with .mlb == item_id), excludes
    cancelled/invalid statuses.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT status, items
              FROM ml_user_orders
             WHERE user_id = $1
               AND date_created BETWEEN $2 AND $3
               AND items @> $4::jsonb
            """,
            user_id, start, end,
            json.dumps([{"mlb": item_id}]),
        )

    orders = 0
    units = 0
    revenue = 0.0
    for row in rows:
        status = (row["status"] or "").lower()
        if status in EXCLUDED_STATUSES:
            continue
        items_raw = row["items"]
        if isinstance(items_raw, str):
            try:
                items_raw = json.loads(items_raw)
            except Exception:  # noqa: BLE001
                continue
        if not isinstance(items_raw, list):
            continue
        # Order counts once even if it has multiple items for this MLB
        # (rare but possible with variations).
        matched = False
        for it in items_raw:
            if (it or {}).get("mlb") != item_id:
                continue
            matched = True
            units += int(it.get("quantity") or 0)
            revenue += float(it.get("revenue") or 0)
        if matched:
            orders += 1

    return {"orders": orders, "units": units, "revenue": revenue}


async def get_latest_fetched_at(pool: asyncpg.Pool, user_id: int) -> Optional[str]:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"')
              FROM ml_user_orders
             WHERE user_id = $1
            """,
            user_id,
        )


# ── Probe (TEST step in TEST→БД→КЭШ) ───────────────────────────────────────────


async def probe(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    days_back: int = 2,
) -> dict[str, Any]:
    """Hit /orders/search directly and return raw status + body slice.
    Used by /escalar/orders-probe to verify ML token + endpoint shape
    before relying on the cache. Mirrors the messages-probe pattern.
    """
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"error": "oauth_failed", "detail": str(err)}

    tokens = await ml_oauth_svc.load_user_tokens(pool, user_id) or {}
    seller_id = tokens.get("ml_user_id")
    if not seller_id:
        return {"error": "no_seller_id"}

    end_date = datetime.now(BRT).date()
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=BRT)
    start_dt = datetime.combine(
        end_date - timedelta(days=days_back), datetime.min.time(), tzinfo=BRT,
    )
    url = (
        f"{ML_API_BASE}/orders/search"
        f"?seller={seller_id}"
        f"&order.date_created.from={_iso_brt(start_dt)}"
        f"&order.date_created.to={_iso_brt(end_dt)}"
        f"&offset=0&limit={PAGE_LIMIT}"
    )
    async with httpx.AsyncClient() as http:
        try:
            r = await http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20.0)
        except Exception as err:  # noqa: BLE001
            return {"error": "http_failed", "detail": str(err), "url": url}

    body_text = r.text[:1500]
    body_json: Any = None
    try:
        body_json = r.json()
    except Exception:  # noqa: BLE001
        pass

    paging = (body_json or {}).get("paging") if isinstance(body_json, dict) else None
    sample = (body_json or {}).get("results", [])[:1] if isinstance(body_json, dict) else None
    sample_keys = list(sample[0].keys()) if sample else []

    return {
        "url": url,
        "seller_id": seller_id,
        "status": r.status_code,
        "paging": paging,
        "sample_keys": sample_keys,
        "sample_first": sample[0] if sample else None,
        "body_preview": body_text,
    }
