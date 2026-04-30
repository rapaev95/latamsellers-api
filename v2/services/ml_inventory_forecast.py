"""Inventory forecast for sales TG notifications.

Computes per-item snapshot:
  - current stock (from ml_user_items.available_quantity — covers Full + MEnvios)
  - sales in window (from ml_user_orders, excludes cancelled/invalid)
  - average daily velocity
  - days left at current pace
  - level: critical (<7 days) / low (<14 days) / ok / no_history (no sales in window)

Window length is per-user (notification_settings.inventory_window_days,
default 14). Memory: project_inventory_forecast_in_tg.md.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

from . import ml_orders as ml_orders_svc

log = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 14
ALLOWED_WINDOWS = (7, 14, 30)
CRITICAL_DAYS = 7
LOW_DAYS = 14


CREATE_INVENTORY_SETTINGS_SQL = """
ALTER TABLE notification_settings
  ADD COLUMN IF NOT EXISTS inventory_window_days INTEGER DEFAULT 14;
"""

CREATE_DISPATCH_LOG_SQL = """
CREATE TABLE IF NOT EXISTS inventory_alert_log (
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  level TEXT NOT NULL,
  days_left INTEGER,
  dispatched_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_inv_alert_dispatched
  ON inventory_alert_log(user_id, dispatched_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_INVENTORY_SETTINGS_SQL)
        await conn.execute(CREATE_DISPATCH_LOG_SQL)


async def _get_window_days(pool: asyncpg.Pool, user_id: int) -> int:
    try:
        async with pool.acquire() as conn:
            v = await conn.fetchval(
                "SELECT inventory_window_days FROM notification_settings WHERE user_id = $1",
                user_id,
            )
        if v in ALLOWED_WINDOWS:
            return int(v)
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_WINDOW_DAYS


async def _get_stock(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    variation_id: Optional[str] = None,
) -> tuple[Optional[int], Optional[str]]:
    """Returns (stock_int, variation_label).

    If variation_id is provided, drills into ml_user_items.raw.variations[]
    and returns the matched variation's available_quantity + a short
    label derived from variation attributes. Falls back to item-level
    available_quantity if variation not found in cached raw.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT available_quantity, raw FROM ml_user_items "
                "WHERE user_id = $1 AND item_id = $2",
                user_id, item_id,
            )
        if row is None:
            return None, None
        item_total = int(row["available_quantity"] or 0)
        if not variation_id:
            return item_total, None

        raw = row["raw"]
        if isinstance(raw, str):
            try:
                import json as _json
                raw = _json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        variations = (raw or {}).get("variations") if isinstance(raw, dict) else None
        if not isinstance(variations, list):
            return item_total, None
        for v in variations:
            if not isinstance(v, dict):
                continue
            if str(v.get("id") or "") != str(variation_id):
                continue
            v_qty = int(v.get("available_quantity") or 0)
            # Build short label from attribute_combinations
            label_parts: list[str] = []
            for attr in (v.get("attribute_combinations") or []):
                name = attr.get("value_name") or attr.get("name") or ""
                if name and len(label_parts) < 3:
                    label_parts.append(str(name))
            label = " · ".join(label_parts) if label_parts else f"var {variation_id}"
            return v_qty, label
        # variation_id supplied but not found in cache → fall back to item total
        return item_total, None
    except Exception as err:  # noqa: BLE001
        log.debug("get_stock failed user=%s item=%s var=%s: %s",
                  user_id, item_id, variation_id, err)
        return None, None


def _level(days_left: Optional[float], avg_daily: float) -> str:
    if avg_daily <= 0:
        return "no_history"
    if days_left is None:
        return "ok"
    if days_left < CRITICAL_DAYS:
        return "critical"
    if days_left < LOW_DAYS:
        return "low"
    return "ok"


async def get_inventory_snapshot(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    variation_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Returns inventory snapshot dict, or None if stock data unavailable.

    Shape:
      {
        "stock": int,
        "sold_in_window": int,
        "window_days": int,
        "avg_daily": float (rounded to 2 dec),
        "days_left": int | None,    # None when avg_daily == 0
        "level": "critical" | "low" | "ok" | "no_history",
        "variation_id": str | None,
        "variation_label": str | None,  # e.g. "Preto · M" if variation matched
      }

    Stock — per-variation if variation_id supplied AND found in cached raw,
    else item-level total. Sales window — item-level (per-variation sales
    needs deeper join into order items, kept simple for now).

    None if item not in ml_user_items (no stock to forecast against). The
    caller (normalize.orders_v2) skips the inventory block in that case.
    """
    stock, var_label = await _get_stock(pool, user_id, item_id, variation_id)
    if stock is None:
        return None

    window_days = await _get_window_days(pool, user_id)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=window_days)
    try:
        win = await ml_orders_svc.get_orders_for_window(
            pool, user_id, item_id, start, end,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("get_orders_for_window failed: %s", err)
        win = {"orders": 0, "units": 0, "revenue": 0.0}

    sold = int(win.get("units") or 0)
    avg_daily = round(sold / window_days, 2) if window_days > 0 else 0.0
    days_left: Optional[int] = None
    if avg_daily > 0:
        days_left = int(stock / avg_daily) if stock >= 0 else 0
    level = _level(float(days_left) if days_left is not None else None, avg_daily)

    return {
        "stock": stock,
        "sold_in_window": sold,
        "window_days": window_days,
        "avg_daily": avg_daily,
        "days_left": days_left,
        "level": level,
        "variation_id": str(variation_id) if variation_id else None,
        "variation_label": var_label,
    }


# ── Daily critical-stock alerts ────────────────────────────────────────────

ALERT_DEDUP_DAYS = 7
ALERT_BATCH_SLEEP = 0.3  # seconds between TG messages, anti-rate-limit


async def _list_active_items(pool: asyncpg.Pool, user_id: int) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT item_id FROM ml_user_items
             WHERE user_id = $1 AND COALESCE(status, 'active') = 'active'
            """,
            user_id,
        )
    return [r["item_id"] for r in rows]


async def _was_recently_alerted(
    pool: asyncpg.Pool, user_id: int, item_id: str,
) -> bool:
    async with pool.acquire() as conn:
        ts = await conn.fetchval(
            """
            SELECT dispatched_at FROM inventory_alert_log
             WHERE user_id = $1 AND item_id = $2
            """,
            user_id, item_id,
        )
    if ts is None:
        return False
    age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).days
    return age < ALERT_DEDUP_DAYS


async def _record_alert(
    pool: asyncpg.Pool, user_id: int, item_id: str, level: str, days_left: Optional[int],
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO inventory_alert_log
              (user_id, item_id, level, days_left, dispatched_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (user_id, item_id) DO UPDATE SET
              level = EXCLUDED.level,
              days_left = EXCLUDED.days_left,
              dispatched_at = NOW()
            """,
            user_id, item_id, level, days_left,
        )


async def dispatch_inventory_alerts(
    pool: asyncpg.Pool, user_id: int,
) -> dict[str, int]:
    """For one user: scan active items, send TG alert for each `critical`
    item not alerted in last 7 days. Returns counts.

    Triggered daily via APScheduler. Per-item dedup keeps TG quiet — same
    item won't trigger more than once per week even if stock stays low.
    """
    import asyncio
    import os

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"checked": 0, "sent": 0, "skipped": 0, "error": "no_bot_token"}
    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(notify_daily_sales, TRUE) AS notify_on,
                   COALESCE(language, 'pt') AS language
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
    if not settings or not settings["telegram_chat_id"] or not settings["notify_on"]:
        return {"checked": 0, "sent": 0, "skipped": 0}
    chat_id = str(settings["telegram_chat_id"])
    lang = settings["language"] or "pt"

    items = await _list_active_items(pool, user_id)
    sent = 0
    skipped = 0
    checked = 0
    if not items:
        return {"checked": 0, "sent": 0, "skipped": 0}

    import httpx
    async with httpx.AsyncClient(timeout=15.0) as http:
        for item_id in items:
            checked += 1
            try:
                snap = await get_inventory_snapshot(pool, user_id, item_id)
            except Exception as err:  # noqa: BLE001
                log.debug("snapshot failed item=%s: %s", item_id, err)
                continue
            if not snap or snap["level"] != "critical":
                continue
            if await _was_recently_alerted(pool, user_id, item_id):
                skipped += 1
                continue

            # Fetch item title для useful TG-message
            async with pool.acquire() as conn:
                title = await conn.fetchval(
                    "SELECT title FROM ml_user_items WHERE user_id = $1 AND item_id = $2",
                    user_id, item_id,
                )
            title_short = (title or item_id)[:80]
            days_left = snap["days_left"] if snap["days_left"] is not None else 0
            stock = snap["stock"]
            avg = snap["avg_daily"]

            if lang == "ru":
                text = (
                    f"⚠️ *Скоро закончится товар*\n\n"
                    f"📦 {title_short}\n"
                    f"🆔 `{item_id}`\n\n"
                    f"Остаток: {stock} ед. · скорость: {avg:.1f}/день\n"
                    f"⏰ Хватит на ~{days_left} дней\n\n"
                    f"_Закажи Full / пополни запас, чтобы не потерять продажи._"
                )
            elif lang == "en":
                text = (
                    f"⚠️ *Stock running out*\n\n"
                    f"📦 {title_short}\n"
                    f"🆔 `{item_id}`\n\n"
                    f"Stock: {stock} un. · pace: {avg:.1f}/day\n"
                    f"⏰ Lasts only ~{days_left} days\n\n"
                    f"_Replenish Full to avoid lost sales._"
                )
            else:
                text = (
                    f"⚠️ *Estoque acabando*\n\n"
                    f"📦 {title_short}\n"
                    f"🆔 `{item_id}`\n\n"
                    f"Estoque: {stock} un. · ritmo: {avg:.1f}/dia\n"
                    f"⏰ Dura apenas ~{days_left} dias\n\n"
                    f"_Reponha o Full para não perder vendas._"
                )

            try:
                r = await http.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True,
                    },
                )
                if r.status_code == 200:
                    await _record_alert(pool, user_id, item_id, snap["level"], days_left)
                    sent += 1
                else:
                    log.warning("inventory alert TG failed user=%s item=%s status=%s",
                                user_id, item_id, r.status_code)
            except Exception as err:  # noqa: BLE001
                log.warning("inventory alert exception user=%s item=%s: %s",
                            user_id, item_id, err)
            await asyncio.sleep(ALERT_BATCH_SLEEP)

    return {"checked": checked, "sent": sent, "skipped": skipped}


async def dispatch_inventory_alerts_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Daily cron entrypoint — iterate all TG-linked users."""
    if pool is None:
        return {"users": 0, "sent": 0}
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT t.user_id
              FROM ml_user_tokens t
              JOIN notification_settings n ON n.user_id = t.user_id
             WHERE t.access_token IS NOT NULL
               AND n.telegram_chat_id IS NOT NULL
            """,
        )
    totals = {"users": 0, "sent": 0, "checked": 0, "skipped": 0}
    for r in rows:
        try:
            res = await dispatch_inventory_alerts(pool, r["user_id"])
            totals["users"] += 1
            totals["sent"] += res.get("sent", 0)
            totals["checked"] += res.get("checked", 0)
            totals["skipped"] += res.get("skipped", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("inventory alerts user=%s failed: %s", r["user_id"], err)
    return totals
