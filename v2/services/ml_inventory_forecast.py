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


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_INVENTORY_SETTINGS_SQL)


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


async def _get_stock(pool: asyncpg.Pool, user_id: int, item_id: str) -> Optional[int]:
    try:
        async with pool.acquire() as conn:
            v = await conn.fetchval(
                "SELECT available_quantity FROM ml_user_items WHERE user_id = $1 AND item_id = $2",
                user_id, item_id,
            )
        if v is None:
            return None
        return int(v)
    except Exception as err:  # noqa: BLE001
        log.debug("get_stock failed user=%s item=%s: %s", user_id, item_id, err)
        return None


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
      }

    None if item not in ml_user_items (no stock to forecast against). The
    caller (normalize.orders_v2) skips the inventory block in that case.
    """
    stock = await _get_stock(pool, user_id, item_id)
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
    }
