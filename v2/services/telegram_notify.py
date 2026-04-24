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


# Emoji per ML topic/category for quick visual scanning in Telegram.
# Falls back to 🔔 when notice_id prefix or topic don't match anything known.
_TOPIC_EMOJI = {
    "orders_v2": "🛒",
    "orders": "🛒",
    "questions": "❓",
    "claims": "⚠️",
    "items": "📦",
    "messages": "💬",
    "invoices": "💰",
    "payments": "💳",
    "stock-locations": "📍",
    "shipments": "🚚",
    "fbm_stock_operations": "📦",
    "post_purchase": "📬",
}


def _topic_from_notice_id(notice_id: str) -> Optional[str]:
    """notice_id format is often '<topic>:<ml_id>' (e.g. 'invoices:5929454404')."""
    if not notice_id or ":" not in notice_id:
        return None
    return notice_id.split(":", 1)[0].strip() or None


def _derive_emoji(topic: Optional[str], notice_id: str) -> str:
    key = (topic or _topic_from_notice_id(notice_id) or "").lower()
    return _TOPIC_EMOJI.get(key, "🔔")


def _extract_ml_url(notice_id: str, topic: Optional[str]) -> Optional[str]:
    """Best-effort deep-link into ML Seller Center by topic + resource id."""
    resource_id = notice_id.split(":", 1)[1] if ":" in notice_id else notice_id
    key = (topic or _topic_from_notice_id(notice_id) or "").lower()
    if key == "orders_v2" or key == "orders":
        return f"https://myaccount.mercadolivre.com.br/sales/{resource_id}/detail"
    if key == "questions":
        return f"https://myaccount.mercadolivre.com.br/sales/questions"
    if key == "claims":
        return f"https://myaccount.mercadolivre.com.br/post-purchase/cases/{resource_id}"
    if key == "invoices":
        return f"https://myaccount.mercadolivre.com.br/invoices"
    if key == "items":
        return f"https://produto.mercadolivre.com.br/{resource_id}"
    return None


def _enrich_from_raw(raw: Any, topic: Optional[str]) -> list[str]:
    """Pull human-readable details from the full ML payload — amount, buyer, title, status."""
    if not isinstance(raw, dict):
        return []
    lines: list[str] = []
    key = (topic or "").lower()
    # Orders — show total + buyer nickname + status
    if key in ("orders_v2", "orders"):
        total = raw.get("total_amount") or raw.get("paid_amount")
        currency = raw.get("currency_id") or "BRL"
        buyer = (raw.get("buyer") or {}).get("nickname") if isinstance(raw.get("buyer"), dict) else None
        status = raw.get("status")
        if total is not None:
            lines.append(f"Valor: {currency} {total}")
        if buyer:
            lines.append(f"Comprador: @{buyer}")
        if status:
            lines.append(f"Status: {status}")
    # Items — show title + status
    elif key == "items":
        title = raw.get("title")
        status = raw.get("status")
        sub = raw.get("sub_status")
        if title:
            lines.append(str(title))
        if status:
            lines.append(f"Status: {status}")
        if sub:
            lines.append(f"Sub-status: {sub if isinstance(sub, str) else ', '.join(sub) if isinstance(sub, list) else sub}")
    # Questions — show question text + item title
    elif key == "questions":
        text_q = raw.get("text")
        if text_q:
            lines.append(str(text_q))
    # Claims — show reason + stage
    elif key == "claims":
        reason = raw.get("reason_id") or (raw.get("reason") or {}).get("id")
        stage = raw.get("stage")
        if reason:
            lines.append(f"Motivo: {reason}")
        if stage:
            lines.append(f"Etapa: {stage}")
    return lines


def _format_message(
    label: str,
    description: str,
    actions: Any,
    notice_id: str,
    tags: Any = None,
    topic: Optional[str] = None,
    raw: Any = None,
) -> str:
    emoji = _derive_emoji(topic, notice_id)
    parts: list[str] = []

    header = label or (topic or "").replace("_", " ").title() or "Notificação"
    parts.append(f"{emoji} *{_escape(header)}*")

    if description and description.strip() and description.strip() != header.strip():
        parts.append(_escape(description))

    # Enrich from raw payload (amount/buyer/status/title/…)
    extra = _enrich_from_raw(raw, topic)
    if extra:
        parts.append("\n".join(_escape(line) for line in extra))

    # Actions: normalize and render as markdown links. Defensive vs strings/mixed.
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

    # Fallback link if no action URLs — derive from topic + notice_id
    if not action_lines:
        derived = _extract_ml_url(notice_id, topic)
        if derived:
            action_lines.append(f"[{_escape('Abrir no Mercado Livre')}]({derived})")

    if action_lines:
        parts.append("\n".join(action_lines))

    # Tags footer (small, for context: SALE/REFUND/PAUSED/…)
    if isinstance(tags, list) and tags:
        tag_line = " · ".join(f"\\#{_escape(str(t))}" for t in tags if t)
        if tag_line:
            parts.append(f"_{tag_line}_")

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

    # Skip translation for short key-like labels ("invoices", "stock-locations")
    # — translation renders them as confusing literals ("счета", "остатки-местоположения").
    def _is_key_like(s: str) -> bool:
        t = s.strip()
        return 0 < len(t) < 30 and " " not in t

    if language in ("ru", "en"):
        if label and not _is_key_like(label):
            label = await translate_svc.translate(label, target=language, http=http)
        if description and not _is_key_like(description):
            description = await translate_svc.translate(description, target=language, http=http)

    text = _format_message(
        label=label,
        description=description,
        actions=notice.get("actions") or [],
        notice_id=str(notice.get("notice_id") or ""),
        tags=notice.get("tags") or [],
        topic=notice.get("topic"),
        raw=notice.get("raw") or {},
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
