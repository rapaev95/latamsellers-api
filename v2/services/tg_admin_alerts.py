"""Critical-error alerts to super-admins via Telegram.

Triggered from places where a failure blocks user-facing functionality and
needs human attention (OpenRouter out of credits, ML OAuth refresh broken
for many users, DB migration failed, etc).

Configuration via env vars (Railway):
  LS_ADMIN_TG_CHAT_IDS  CSV of numeric chat_ids — empty disables alerts.
  TELEGRAM_BOT_TOKEN    same token used for normal user notifications.

Rate-limiting is in-memory per-process via `_last_sent[key] = ts`. With
multiple gunicorn workers each will dedupe independently — fine, we'd
rather get N copies during a real incident than miss one. Dedup key is
caller-provided so callers control granularity (per-user, per-error-code,
per-service).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

import httpx

log = logging.getLogger(__name__)

Severity = Literal["warn", "error", "critical"]

_TG_API_BASE = "https://api.telegram.org"
_DEDUP_WINDOW_SEC = 30 * 60  # 30 minutes — quiet enough for repeating issues
_last_sent: dict[str, float] = {}


def _get_admin_chat_ids() -> list[str]:
    raw = os.environ.get("LS_ADMIN_TG_CHAT_IDS", "").strip()
    if not raw:
        return []
    return [c.strip() for c in raw.split(",") if c.strip()]


def _severity_emoji(severity: Severity) -> str:
    return {"warn": "⚠️", "error": "🟠", "critical": "🚨"}.get(severity, "ℹ️")


def _format_alert(
    title: str, detail: str, severity: Severity, service: Optional[str] = None,
) -> str:
    from datetime import datetime, timezone, timedelta
    brt = timezone(timedelta(hours=-3))
    ts = datetime.now(brt).strftime("%Y-%m-%d %H:%M BRT")
    emoji = _severity_emoji(severity)
    lines = [
        f"{emoji} *{severity.upper()}* — {title}",
    ]
    if service:
        lines.append(f"_Service:_ `{service}`")
    if detail:
        # Trim — TG message limit 4096 but keep alerts concise.
        clip = detail[:1500]
        if len(detail) > 1500:
            clip += "…"
        lines.append("")
        lines.append(clip)
    lines.append("")
    lines.append(f"`{ts}`")
    return "\n".join(lines)


async def send_admin_alert(
    title: str,
    detail: str,
    severity: Severity = "critical",
    service: Optional[str] = None,
    deduplicate_key: Optional[str] = None,
) -> None:
    """Fire-and-forget alert to all super-admins.

    Never raises — callers shouldn't have their happy path broken by a TG
    delivery failure. Rate-limited per `deduplicate_key` (defaults to
    `severity:title` — if you want stricter, pass per-user/per-code key).
    """
    chat_ids = _get_admin_chat_ids()
    if not chat_ids:
        return  # silent — env not configured
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        log.warning("admin alert skipped: no TELEGRAM_BOT_TOKEN")
        return

    key = deduplicate_key or f"{severity}:{title}"
    now = time.time()
    last = _last_sent.get(key, 0.0)
    if now - last < _DEDUP_WINDOW_SEC:
        log.debug("admin alert deduped: key=%s age=%.0fs", key, now - last)
        return
    _last_sent[key] = now

    text = _format_alert(title, detail, severity, service)
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            for chat_id in chat_ids:
                try:
                    await http.post(
                        f"{_TG_API_BASE}/bot{bot_token}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": text,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": True,
                        },
                    )
                except Exception as err:  # noqa: BLE001
                    log.warning("admin alert send to %s failed: %s", chat_id, err)
    except Exception as err:  # noqa: BLE001
        log.warning("admin alert client failed: %s", err)


# ── Specific alert helpers (callers can use these to keep call sites short) ──

async def alert_openrouter_failure(
    status_code: int, response_text: str, service: str,
) -> None:
    """OpenRouter non-200 — covers 402 (no credits), 401 (bad key),
    429 (rate limit), 500 (provider down).
    """
    severity: Severity = "critical" if status_code in (401, 402) else "warn"
    await send_admin_alert(
        title=f"OpenRouter {status_code}",
        detail=f"Status: {status_code}\nResponse: {response_text[:800]}",
        severity=severity,
        service=service,
        deduplicate_key=f"openrouter:{status_code}:{service}",
    )


async def alert_ml_oauth_broken(user_id: int, detail: str) -> None:
    """ML refresh-token rejected — user must re-auth in app."""
    await send_admin_alert(
        title="ML OAuth refresh failed",
        detail=f"User {user_id} needs re-auth.\n{detail[:500]}",
        severity="error",
        service="ml_oauth.refresh",
        deduplicate_key=f"ml_oauth_refresh_failed:{user_id}",
    )


async def alert_migration_failure(table: str, detail: str) -> None:
    """ensure_schema raised — DB might be in inconsistent state."""
    await send_admin_alert(
        title=f"DB migration failed — {table}",
        detail=detail[:1000],
        severity="critical",
        service=f"ensure_schema:{table}",
        deduplicate_key=f"migration:{table}",
    )
