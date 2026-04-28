"""Daily positions refresh + scraper-health monitor.

Walks `tracked_keywords` for each user with TG enabled, runs
`check_position` with throttling, persists results to
`position_history`. Two side-benefits:

  1. position_drop anomaly detector (ml_anomalies.py) gets fresh data
     to compare against — without this cron, position_history only
     gets points when the seller manually clicks "Check" in the UI.
  2. Session-health monitoring: when ALL the user's checks fail for
     consecutive days, ML session probably expired (2FA challenged
     or password rotated). We send ONE TG card with «refresh your
     ML_SCRAPER_STORAGE_STATE_B64» instructions and don't spam it.

Cadence: once per day at 09:00 BRT (a quiet hour for ML's gateway,
also the seller's morning so action-items land at a useful time).

Throttling: positions.check_position itself sleeps between pages.
We add an outer per-user delay (~10-30s with jitter) so multiple
keywords for the same user don't burst.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import positions as positions_svc
from .positions import PositionCheckError

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
BRT = timezone(timedelta(hours=-3))

PER_USER_INTER_KW_S = float(os.environ.get("POSITIONS_PER_USER_INTER_KW_S", "12"))
PER_USER_INTER_KW_JITTER_S = float(os.environ.get("POSITIONS_PER_USER_INTER_KW_JITTER_S", "5"))
PER_USER_MAX_KW = int(os.environ.get("POSITIONS_PER_USER_MAX_KW", "30"))
HEALTH_FAILURE_THRESHOLD = float(os.environ.get("POSITIONS_HEALTH_FAILURE_THRESHOLD", "0.7"))


# ── MarkdownV2 helpers (replicating the daily-summary pattern) ─────────────

_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
_MD2_UNESCAPE_RE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


def _esc(text: Any) -> str:
    return (str(text or "")).translate(_MD_ESCAPE)


def _strip_md2(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"(?<!\\)[*_~`]", "", text)
    return _MD2_UNESCAPE_RE.sub(r"\1", out)


# ── Per-user refresh ──────────────────────────────────────────────────────

async def _polite_inter_keyword_sleep() -> None:
    base = PER_USER_INTER_KW_S
    jitter = random.uniform(-PER_USER_INTER_KW_JITTER_S, PER_USER_INTER_KW_JITTER_S)
    await asyncio.sleep(max(2.0, base + jitter))


async def _refresh_one_user(
    pool: asyncpg.Pool, user_id: int,
) -> dict[str, Any]:
    """Walk tracked_keywords for this user, run check_position for each,
    record into position_history. Returns counts of ok/fail/skipped."""
    from v2.storage import positions_storage  # late import to avoid circular

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, item_id, keyword, site_id, category_id
              FROM tracked_keywords
             WHERE user_id = $1
             ORDER BY created_at ASC
             LIMIT $2
            """,
            user_id, PER_USER_MAX_KW,
        )
    tracked = [dict(r) for r in rows]
    if not tracked:
        return {"tracked": 0, "ok": 0, "fail": 0, "skipped": 0}

    # Get bearer once for all checks (auto-refreshed inside check_position
    # if it expires mid-run)
    from . import ml_oauth as _oauth_svc
    try:
        bearer, _exp, _refreshed = await _oauth_svc.get_valid_access_token(pool, user_id)
    except _oauth_svc.MLRefreshError as err:
        log.warning("positions refresh user=%s: oauth failed (%s) — skipping", user_id, err)
        return {"tracked": len(tracked), "ok": 0, "fail": 0, "skipped": len(tracked), "reason": "oauth"}
    tokens_row = await _oauth_svc.load_user_tokens(pool, user_id) or {}
    seller_id = tokens_row.get("ml_user_id")

    ok = 0
    fail = 0
    for tk in tracked:
        try:
            result = await positions_svc.check_position(
                item_id=tk["item_id"],
                keyword=tk["keyword"],
                site_id=tk["site_id"] or "MLB",
                category_id=tk["category_id"],
                bearer_token=bearer,
                seller_id=seller_id,
            )
            await positions_storage.record_check(
                pool,
                user_id=user_id,
                item_id=result.item_id,
                keyword=result.keyword,
                site_id=result.site_id,
                position=result.position,
                total_results=result.total_results,
                found=result.found,
            )
            ok += 1
        except PositionCheckError as err:
            log.info("positions refresh user=%s kw=%r failed: %s", user_id, tk["keyword"], err)
            fail += 1
        except Exception as err:  # noqa: BLE001
            log.warning("positions refresh user=%s kw=%r exception: %s", user_id, tk["keyword"], err)
            fail += 1
        # Polite inter-keyword pause regardless of success/failure
        await _polite_inter_keyword_sleep()

    return {"tracked": len(tracked), "ok": ok, "fail": fail, "skipped": 0}


# ── Health alert ──────────────────────────────────────────────────────────

async def _send_health_alert(
    pool: asyncpg.Pool, user_id: int, summary: dict[str, Any],
) -> bool:
    """Send a TG card warning the seller their ML scraper auth might
    have expired. Throttles itself: we mark `last_health_alert_at` on
    notification_settings (lazy-added column) so we send at most once
    per 48h.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return False

    async with pool.acquire() as conn:
        try:
            await conn.execute(
                "ALTER TABLE notification_settings "
                "ADD COLUMN IF NOT EXISTS last_health_alert_at TIMESTAMPTZ"
            )
        except Exception:  # noqa: BLE001
            pass
        row = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language,
                   last_health_alert_at
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )

    if not row or not row["telegram_chat_id"]:
        return False

    last = row["last_health_alert_at"]
    if last is not None:
        age = (datetime.now(timezone.utc) - last).total_seconds()
        if age < 48 * 3600:
            log.info("health_alert user=%s suppressed — last sent %.1fh ago", user_id, age / 3600)
            return False

    lang = (row["language"] or "pt").lower()
    if lang == "ru":
        title = "🚨 *Сессия ML может быть истекла*"
        body = (
            f"Сегодня замер позиций провалился: {summary['fail']} из {summary['tracked']} keywords.\n\n"
            "Возможные причины:\n"
            "• ML запросил 2FA \\(suspicious activity\\)\n"
            "• Пароль был сменён\n"
            "• Cookies истекли\n\n"
            "Что сделать:\n"
            "1\\. Залогинься в [mercadolivre\\.com\\.br](https://www.mercadolivre.com.br) в обычном Chrome\n"
            "2\\. Cookie\\-Editor extension → Export → JSON\n"
            "3\\. Запусти конвертер в DevTools Console \\(см\\. инструкцию\\)\n"
            "4\\. Обнови `ML_SCRAPER_STORAGE_STATE_B64` в Railway"
        )
    else:
        title = "🚨 *Sessão ML pode ter expirado*"
        body = (
            f"A medição de hoje falhou: {summary['fail']} de {summary['tracked']} keywords.\n\n"
            "Possíveis causas:\n"
            "• ML pediu 2FA \\(atividade suspeita\\)\n"
            "• Senha foi alterada\n"
            "• Cookies expiraram\n\n"
            "O que fazer:\n"
            "1\\. Faça login em [mercadolivre\\.com\\.br](https://www.mercadolivre.com.br)\n"
            "2\\. Cookie\\-Editor extension → Export → JSON\n"
            "3\\. Rode o conversor no DevTools Console\n"
            "4\\. Atualize `ML_SCRAPER_STORAGE_STATE_B64` no Railway"
        )

    text = f"{title}\n\n{body}"

    async with httpx.AsyncClient() as http:
        sent_id = None
        try:
            r = await http.post(
                f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                json={
                    "chat_id": str(row["telegram_chat_id"]),
                    "text": text,
                    "parse_mode": "MarkdownV2",
                    "disable_web_page_preview": True,
                },
                timeout=15.0,
            )
            if r.status_code == 200:
                data = r.json() or {}
                sent_id = (data.get("result") or {}).get("message_id")
            else:
                log.warning("health_alert MD2 failed: %s %s", r.status_code, r.text[:200])
                # Plain fallback
                r2 = await http.post(
                    f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                    json={"chat_id": str(row["telegram_chat_id"]), "text": _strip_md2(text)},
                    timeout=15.0,
                )
                if r2.status_code == 200:
                    sent_id = (r2.json().get("result") or {}).get("message_id")
        except Exception as err:  # noqa: BLE001
            log.exception("health_alert send failed: %s", err)

    if sent_id:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE notification_settings SET last_health_alert_at = NOW() "
                "WHERE user_id = $1",
                user_id,
            )
        return True
    return False


# ── Cron entry ────────────────────────────────────────────────────────────

async def dispatch_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Walk every user that has tracked_keywords and a TG chat. Refresh
    their positions; if failure-rate exceeds threshold, fire a health
    alert (suppressed if already sent within 48h)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT t.user_id
              FROM tracked_keywords t
              JOIN notification_settings n ON n.user_id = t.user_id
             WHERE n.telegram_chat_id IS NOT NULL
            """,
        )
    user_ids = [int(r["user_id"]) for r in rows]
    total_tracked = 0
    total_ok = 0
    total_fail = 0
    alerts_sent = 0

    for uid in user_ids:
        try:
            summary = await _refresh_one_user(pool, uid)
        except Exception as err:  # noqa: BLE001
            log.exception("positions refresh user=%s exception: %s", uid, err)
            continue
        total_tracked += summary.get("tracked", 0)
        total_ok += summary.get("ok", 0)
        total_fail += summary.get("fail", 0)

        tracked = summary.get("tracked", 0)
        fail = summary.get("fail", 0)
        if tracked >= 3 and fail / tracked >= HEALTH_FAILURE_THRESHOLD:
            sent = await _send_health_alert(pool, uid, summary)
            if sent:
                alerts_sent += 1

        # Spread users across the cron tick so we don't hammer ML in
        # parallel. Two-pass safety: per-user pause is intra-user; this
        # is between users.
        await asyncio.sleep(random.uniform(3.0, 8.0))

    log.info(
        "positions refresh tick: users=%s tracked=%s ok=%s fail=%s alerts=%s",
        len(user_ids), total_tracked, total_ok, total_fail, alerts_sent,
    )
    return {
        "users": len(user_ids),
        "tracked": total_tracked,
        "ok": total_ok,
        "fail": total_fail,
        "alerts_sent": alerts_sent,
    }
