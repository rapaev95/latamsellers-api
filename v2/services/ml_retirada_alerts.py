"""Retirada de estoque Full alerts — Descarte (утилизация) / Envio para o
endereço (вывоз со склада).

User feedback: «Получать уведомления что нужно вывести товар или
утилизировать. Чтобы не иметь убытки retirada».

Source: Relatorio_Tarifas_Full_*.xlsx parsed by `legacy/reports.py:
get_retirada_by_period`. Each row has unique `Nº do custo` (custo_id) —
that's our dedup key.

Cron daily 13:30 UTC = 10:30 BRT. Per-row alert одного раза:
  - Descarte rows ВСЕГДА critical (товар реально удаляется + COGS списан)
  - Envio rows — informational unless tariff > R$50

Dedup table `retirada_alert_log(user_id, custo_id)` — PRIMARY KEY ensures
one alert per row per user, ever.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone, date
from typing import Any

import asyncpg
import httpx

log = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))
TG_API_BASE = "https://api.telegram.org"
ALERT_BATCH_SLEEP = 0.5

CREATE_LOG_SQL = """
CREATE TABLE IF NOT EXISTS retirada_alert_log (
  user_id INTEGER NOT NULL,
  custo_id TEXT NOT NULL,
  forma TEXT,
  project TEXT,
  units INTEGER,
  tarifa NUMERIC,
  alerted_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, custo_id)
);
CREATE INDEX IF NOT EXISTS idx_retirada_alert_user
  ON retirada_alert_log(user_id, alerted_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_LOG_SQL)


async def _list_user_projects(pool: asyncpg.Pool, user_id: int) -> list[str]:
    """All project IDs configured for this user. Reads legacy projects_db
    via legacy.config (per-user via db_storage.set_current_user_id)."""
    try:
        from v2.legacy import db_storage as _legacy_db
        from v2.legacy.config import load_projects
        _legacy_db.set_current_user_id(user_id)
        projects = load_projects() or {}
        return [str(k).upper() for k in projects.keys()]
    except Exception as err:  # noqa: BLE001
        log.warning("list user projects failed user=%s: %s", user_id, err)
        return []


async def _was_alerted(pool: asyncpg.Pool, user_id: int, custo_id: str) -> bool:
    async with pool.acquire() as conn:
        v = await conn.fetchval(
            "SELECT 1 FROM retirada_alert_log WHERE user_id = $1 AND custo_id = $2",
            user_id, str(custo_id),
        )
    return v is not None


async def _record_alert(
    pool: asyncpg.Pool, user_id: int, custo_id: str,
    forma: str, project: str, units: int, tarifa: float,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO retirada_alert_log
              (user_id, custo_id, forma, project, units, tarifa)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, custo_id) DO NOTHING
            """,
            user_id, str(custo_id), forma, project,
            int(units or 0), float(tarifa or 0),
        )


def _format_alert_pt(forma: str, project: str, row: dict[str, Any]) -> str:
    """Markdown-formatted TG message in Portuguese (default)."""
    titulo = (row.get("titulo") or row.get("title") or "").strip()
    sku = row.get("sku") or "N/A"
    units = int(row.get("units") or 0)
    tarifa = float(row.get("tarifa") or 0)
    custo_id = row.get("custo_id") or ""
    if "descarte" in forma.lower():
        return (
            f"🚨 \\[ALERTA\\] *Descarte Full*\n\n"
            f"📦 {titulo[:80]}\n"
            f"🏷 SKU: `{sku}` · projeto: *{project}*\n"
            f"⚠ {units} un. utilizadas — COGS perdido\n"
            f"💸 Tarifa Full: R$ {tarifa:.2f}\n"
            f"🆔 Custo: `{custo_id}`\n\n"
            f"_Verifique se faz sentido reabastecer ou pausar o anúncio._"
        )
    return (
        f"📦 \\[RETIRADA\\] *Envio do Full ao endereço*\n\n"
        f"🏷 SKU: `{sku}` · projeto: *{project}*\n"
        f"📤 {units} un. saíram do Full (volta para você)\n"
        f"💸 Tarifa Full: R$ {tarifa:.2f}\n"
        f"🆔 Custo: `{custo_id}`\n\n"
        f"_Itens devolvidos não saíram do estoque — só do Full._"
    )


async def dispatch_retirada_alerts(
    pool: asyncpg.Pool, user_id: int, days_back: int = 7,
) -> dict[str, int]:
    """Per-user: scan retirada rows за last N дней for каждый project,
    send first-time alert per custo_id. Returns {checked, sent, skipped}."""
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

    projects = await _list_user_projects(pool, user_id)
    if not projects:
        return {"checked": 0, "sent": 0, "skipped": 0}

    today = datetime.now(BRT).date()
    period_from = today - timedelta(days=days_back)

    sent = 0
    skipped = 0
    checked = 0
    try:
        from v2.legacy.reports import get_retirada_by_period
    except ImportError:
        log.warning("retirada parser unavailable — skip")
        return {"checked": 0, "sent": 0, "skipped": 0, "error": "no_parser"}

    async with httpx.AsyncClient(timeout=15.0) as http:
        for project in projects:
            try:
                report = get_retirada_by_period(project, period_from, today)
            except Exception as err:  # noqa: BLE001
                log.debug("retirada period failed user=%s project=%s: %s",
                          user_id, project, err)
                continue
            by_custo_id: dict = report.get("by_custo_id") or {}
            for custo_id, row in by_custo_id.items():
                checked += 1
                if not custo_id:
                    continue
                forma = row.get("forma") or row.get("forma_effective") or ""
                if await _was_alerted(pool, user_id, custo_id):
                    skipped += 1
                    continue

                text = _format_alert_pt(forma, project, row)
                try:
                    r = await http.post(
                        f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": text,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": True,
                        },
                    )
                    if r.status_code == 200:
                        await _record_alert(
                            pool, user_id, custo_id, forma, project,
                            int(row.get("units") or 0),
                            float(row.get("tarifa") or 0),
                        )
                        sent += 1
                    else:
                        log.warning("retirada TG send failed user=%s status=%s",
                                    user_id, r.status_code)
                except Exception as err:  # noqa: BLE001
                    log.warning("retirada exception user=%s custo=%s: %s",
                                user_id, custo_id, err)
                await asyncio.sleep(ALERT_BATCH_SLEEP)

    return {"checked": checked, "sent": sent, "skipped": skipped}


async def dispatch_retirada_alerts_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Daily cron entrypoint."""
    if pool is None:
        return {"users": 0}
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
    totals = {"users": 0, "checked": 0, "sent": 0, "skipped": 0}
    for r in rows:
        try:
            res = await dispatch_retirada_alerts(pool, r["user_id"])
            totals["users"] += 1
            totals["checked"] += res.get("checked", 0)
            totals["sent"] += res.get("sent", 0)
            totals["skipped"] += res.get("skipped", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("retirada alerts user=%s failed: %s", r["user_id"], err)
    return totals
