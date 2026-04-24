"""Send formatted ML notice to a Telegram chat via Bot API."""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from . import translate as translate_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"

# MarkdownV2 requires escaping these: _ * [ ] ( ) ~ ` > # + - = | { } . !
_MD_ESCAPE = str.maketrans({
    c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"
})


def _escape(text: str) -> str:
    return (text or "").translate(_MD_ESCAPE)


def _format_message(label: str, description: str, actions: Any, notice_id: str) -> str:
    parts: list[str] = []
    if label:
        parts.append(f"*{_escape(label)}*")
    if description:
        parts.append(_escape(description))

    # actions MAY be: list[dict], list[str], a JSON-string, or None.
    # Normalize to a list before iterating, then skip any item that isn't a dict.
    if isinstance(actions, str):
        try:
            import json as _json
            actions = _json.loads(actions)
        except Exception:  # noqa: BLE001
            actions = []
    if not isinstance(actions, list):
        actions = []

    action_lines: list[str] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        url = a.get("url") or a.get("link")
        if not url:
            continue
        lbl = a.get("label") or "Abrir no ML"
        action_lines.append(f"[{_escape(str(lbl))}]({url})")
    if action_lines:
        parts.append("\n".join(action_lines))

    parts.append(f"_ID: {_escape(notice_id)}_")
    return "\n\n".join(parts)


async def send_notice(
    chat_id: str,
    notice: dict[str, Any],
    language: str,
    http: httpx.AsyncClient,
) -> bool:
    """Send a single notice. Returns True if delivered (200 from Telegram)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.warning("TELEGRAM_BOT_TOKEN not set — cannot send notice %s", notice.get("notice_id"))
        return False

    label = notice.get("label") or ""
    description = notice.get("description") or ""

    # language: 'pt' (no translation), 'ru', or 'en'. Anything else → treated as 'pt'.
    if language in ("ru", "en"):
        label = await translate_svc.translate(label, target=language, http=http)
        description = await translate_svc.translate(description, target=language, http=http)

    text = _format_message(
        label=label,
        description=description,
        actions=notice.get("actions") or [],
        notice_id=str(notice.get("notice_id") or ""),
    )

    # Telegram hard limit: 4096 chars. Truncate body (not label) if needed.
    if len(text) > 4000:
        text = text[:3990] + "…"

    url = f"{TG_API_BASE}/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": False,
    }

    for attempt in (0, 1):
        try:
            r = await http.post(url, json=payload, timeout=10.0)
            if r.status_code == 200:
                return True
            # Retry once on 429/5xx; give up on 4xx (bad chat_id, formatting, etc.)
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("TG %s (attempt %s): %s", r.status_code, attempt, r.text[:200])
                continue
            log.error("TG failed %s: %s", r.status_code, r.text[:200])
            return False
        except httpx.HTTPError as err:
            log.warning("TG network error (attempt %s): %s", attempt, err)
            continue
    return False
