"""Aging-stock / not-supported / lost units alerts for Full warehouse.

Two distinct cases mirroring ML UI:
  1. "Não aptas para venda" — units with status `notSupported` / `lost` /
     `transfer` (stuck). ML eventually discards them if not retrieved.
     Source: ml_full_inventory.not_available_detail.
  2. "Estoque excedente" — Aptas para venda > 60 days at current velocity.
     ML may discard via auto-disposal program for slow-movers.
     Computed: available_quantity / avg_daily_velocity (from ml_user_orders).

Dedup: per-item per-alert-type, 7-day rolling — seller gets ONE alert per
item-type per week, not flood. Acceptance is via «Открыть в ML UI» button.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

# How many days of sales velocity until "excedente" alert fires.
EXCEDENTE_DAYS_THRESHOLD = 60
# How often we'll send a TG alert per item per type (rolling window).
DEDUP_WINDOW_DAYS = 7

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_aging_alerts_log (
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  alert_type TEXT NOT NULL,  -- 'not_supported' | 'lost' | 'transfer' | 'excedente'
  last_alerted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  alert_count INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (user_id, item_id, alert_type)
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


async def _was_recently_alerted(
    pool: asyncpg.Pool, user_id: int, item_id: str, alert_type: str,
) -> bool:
    """True if the same alert fired within DEDUP_WINDOW_DAYS for this item."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=DEDUP_WINDOW_DAYS)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT last_alerted_at FROM ml_aging_alerts_log
             WHERE user_id = $1 AND item_id = $2 AND alert_type = $3
               AND last_alerted_at > $4
            """,
            user_id, item_id, alert_type, cutoff,
        )
    return row is not None


async def _record_alert(
    pool: asyncpg.Pool, user_id: int, item_id: str, alert_type: str,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml_aging_alerts_log (user_id, item_id, alert_type, last_alerted_at, alert_count)
            VALUES ($1, $2, $3, NOW(), 1)
            ON CONFLICT (user_id, item_id, alert_type) DO UPDATE SET
              last_alerted_at = NOW(),
              alert_count = ml_aging_alerts_log.alert_count + 1
            """,
            user_id, item_id, alert_type,
        )


def _build_alert_payload(
    item_id: str, item_title: str, alert_type: str,
    qty: int, total: int, available: int,
    deadline_hint: Optional[str] = None,
    days_left: Optional[int] = None,
) -> dict[str, Any]:
    """Compose a notice row payload that ml_notices will dispatch to TG."""
    type_friendly = {
        "not_supported": "Não aptas (notSupported)",
        "lost": "Perdidas no Full",
        "transfer": "Em transferência",
        "excedente": "Estoque excedente",
    }.get(alert_type, alert_type)

    title_short = item_title if len(item_title) <= 60 else item_title[:57] + "..."

    lines: list[str] = [f"📦 {title_short}", f"🆔 {item_id}", ""]
    if alert_type == "excedente":
        lines.append(f"⚠️ *Estoque parado*: {available} un. para vender")
        if days_left is not None:
            lines.append(f"📊 No ritmo atual, dura ~{days_left} dias")
        lines.append("")
        lines.append(
            "ML pode descartar excedente — considere baixar preço / "
            "campanha publicitária / retirar do Full."
        )
    else:
        lines.append(f"⚠️ *{qty} un. {type_friendly}* de {total} total")
        lines.append("")
        lines.append(
            "ML descartará essas unidades se não forem retiradas. "
            "Abra «Gerenciar estoque Full» → «Retirar unidades» para evitar."
        )
    if deadline_hint:
        lines.append("")
        lines.append(f"⏰ {deadline_hint}")
    lines.append("")
    lines.append(
        "🔗 [Gerenciar Full](https://www.mercadolivre.com.br/anuncios/lista/"
        "space_management?filters=aging-stock)"
    )

    label = f"Estoque Full: {qty} {type_friendly}" if alert_type != "excedente" else \
            f"Estoque Full: {available} excedente ({item_id})"

    return {
        "notice_id": f"aging:{item_id}:{alert_type}",
        "label": label,
        "description": "\n".join(lines),
        "tags": ["AGING_STOCK", alert_type.upper()],
        "actions": [],
        "from_date": datetime.now(timezone.utc),
        "topic": "aging_stock",
        "resource": f"/full/aging/{item_id}",
        "raw": {
            "item_id": item_id,
            "alert_type": alert_type,
            "quantity_flagged": qty,
            "total": total,
            "available": available,
            "days_left": days_left,
        },
    }


async def scan_and_push_alerts(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    """Scan ml_full_inventory + velocity for aging issues and push notices.

    Returns counts: {"not_supported": N, "lost": N, "transfer": N,
    "excedente": N, "skipped_dedup": N}.
    """
    await ensure_schema(pool)
    from . import ml_notices as _ml_notices_svc

    stats = {"not_supported": 0, "lost": 0, "transfer": 0, "excedente": 0, "skipped_dedup": 0}

    # ── Part 1: not_available items (from ml_full_inventory) ──────────────────
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT fi.inventory_id, fi.total, fi.available_quantity,
                   fi.not_available_quantity, fi.not_available_detail
              FROM ml_full_inventory fi
             WHERE fi.user_id = $1 AND fi.not_available_quantity > 0
            """,
            user_id,
        )
        # Map inventory_id → list of MLBs and titles
        items_by_inv: dict[str, list[tuple[str, str]]] = {}
        if rows:
            inv_ids = list({r["inventory_id"] for r in rows})
            mlb_rows = await conn.fetch(
                """
                SELECT bm.inventory_id, bm.item_id,
                       COALESCE(ui.title, bm.item_id) AS title
                  FROM ml_full_inventory_by_mlb bm
                  LEFT JOIN ml_user_items ui
                    ON ui.user_id = bm.user_id AND ui.item_id = bm.item_id
                 WHERE bm.user_id = $1 AND bm.inventory_id = ANY($2::text[])
                """,
                user_id, inv_ids,
            )
            for mr in mlb_rows:
                items_by_inv.setdefault(mr["inventory_id"], []).append(
                    (mr["item_id"], mr["title"] or mr["item_id"])
                )

    for r in rows:
        inv_id = r["inventory_id"]
        detail = r["not_available_detail"]
        if isinstance(detail, str):
            try: detail = json.loads(detail)
            except Exception: detail = []
        if not isinstance(detail, list):
            detail = []
        item_list = items_by_inv.get(inv_id, [])
        if not item_list:
            continue
        item_id, item_title = item_list[0]
        for d in detail:
            if not isinstance(d, dict):
                continue
            status = (d.get("status") or "").lower()
            qty = int(d.get("quantity") or 0)
            if qty <= 0 or status not in ("notsupported", "lost", "transfer"):
                continue
            alert_type = "not_supported" if status == "notsupported" else status
            if await _was_recently_alerted(pool, user_id, item_id, alert_type):
                stats["skipped_dedup"] += 1
                continue
            payload = _build_alert_payload(
                item_id, item_title, alert_type,
                qty=qty, total=int(r["total"]), available=int(r["available_quantity"]),
            )
            try:
                await _ml_notices_svc.upsert_normalized(pool, user_id, payload)
                await _record_alert(pool, user_id, item_id, alert_type)
                stats[alert_type] = stats.get(alert_type, 0) + 1
            except Exception as err:  # noqa: BLE001
                log.warning("aging alert upsert failed for %s/%s: %s", item_id, alert_type, err)

    # ── Part 2: «excedente» — slow-moving items with > N days of stock ────────
    # Need velocity (from ml_user_orders last 30d) + current Full stock.
    async with pool.acquire() as conn:
        cand_rows = await conn.fetch(
            """
            SELECT fi.inventory_id, fi.available_quantity, bm.item_id,
                   COALESCE(ui.title, bm.item_id) AS title
              FROM ml_full_inventory fi
              JOIN ml_full_inventory_by_mlb bm
                ON bm.user_id = fi.user_id AND bm.inventory_id = fi.inventory_id
              LEFT JOIN ml_user_items ui
                ON ui.user_id = bm.user_id AND ui.item_id = bm.item_id
             WHERE fi.user_id = $1
               AND fi.available_quantity > 0
            """,
            user_id,
        )
        if cand_rows:
            mlbs = list({r["item_id"] for r in cand_rows})
            # 30-day units sold per MLB (excludes cancelled).
            sold_rows = await conn.fetch(
                """
                SELECT items_elem->>'mlb' AS mlb,
                       SUM((items_elem->>'quantity')::int) AS units
                  FROM ml_user_orders, jsonb_array_elements(items) items_elem
                 WHERE user_id = $1
                   AND status NOT IN ('cancelled', 'invalid')
                   AND date_created >= NOW() - INTERVAL '30 days'
                   AND items_elem->>'mlb' = ANY($2::text[])
                 GROUP BY items_elem->>'mlb'
                """,
                user_id, mlbs,
            )
        else:
            sold_rows = []
    sold_30d = {r["mlb"]: int(r["units"] or 0) for r in sold_rows}

    for r in cand_rows:
        item_id = r["item_id"]
        item_title = r["title"] or item_id
        available = int(r["available_quantity"])
        sold = sold_30d.get(item_id, 0)
        avg_daily = sold / 30.0 if sold > 0 else 0.0
        if avg_daily <= 0:
            # No history — already covered by paused/closed alerts. Skip.
            continue
        days_left = int(available / avg_daily) if avg_daily > 0 else 999
        if days_left < EXCEDENTE_DAYS_THRESHOLD:
            continue
        if await _was_recently_alerted(pool, user_id, item_id, "excedente"):
            stats["skipped_dedup"] += 1
            continue
        payload = _build_alert_payload(
            item_id, item_title, "excedente",
            qty=0, total=available, available=available,
            days_left=days_left,
        )
        try:
            await _ml_notices_svc.upsert_normalized(pool, user_id, payload)
            await _record_alert(pool, user_id, item_id, "excedente")
            stats["excedente"] += 1
        except Exception as err:  # noqa: BLE001
            log.warning("excedente alert upsert failed for %s: %s", item_id, err)

    return stats


async def scan_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Scheduler entry — run scan for every ML-connected user. Returns total
    counts across users."""
    await ensure_schema(pool)
    totals = {"users": 0, "not_supported": 0, "lost": 0, "transfer": 0, "excedente": 0, "skipped_dedup": 0}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
        )
    for r in rows:
        try:
            stats = await scan_and_push_alerts(pool, r["user_id"])
            totals["users"] += 1
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as err:  # noqa: BLE001
            log.exception("aging scan user=%s failed: %s", r["user_id"], err)
    return totals
