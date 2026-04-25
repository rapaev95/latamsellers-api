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


def _fmt_date(iso: Any) -> Optional[str]:
    if not iso or not isinstance(iso, str):
        return None
    # '2026-04-24T13:46:27.484Z' → '24/04 13:46'
    try:
        from datetime import datetime
        s = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d/%m %H:%M")
    except Exception:  # noqa: BLE001
        return iso[:16]


def _fmt_money(amount: Any, currency: Any = "BRL") -> Optional[str]:
    if amount is None:
        return None
    try:
        f = float(amount)
    except (TypeError, ValueError):
        return None
    cur = str(currency or "BRL")
    if cur == "BRL":
        return f"R$ {f:.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{cur} {f:.2f}"


def _enrich_from_raw(raw: Any, topic: Optional[str]) -> list[str]:
    """Pull human-readable details from the full ML payload so the TG message
    is self-contained (user doesn't need to click the link for basic info).
    """
    if not isinstance(raw, dict):
        return []
    lines: list[str] = []
    key = (topic or "").lower()

    # ── ORDERS / SALES ────────────────────────────────────────────────────
    if key in ("orders_v2", "orders"):
        total = raw.get("total_amount") or raw.get("paid_amount")
        currency = raw.get("currency_id") or "BRL"
        money = _fmt_money(total, currency)
        if money:
            lines.append(f"💵 {money}")

        items = raw.get("order_items") or []
        if items and isinstance(items, list):
            parts: list[str] = []
            for it in items[:3]:  # max 3 items
                if not isinstance(it, dict):
                    continue
                inner = it.get("item") or {}
                if not isinstance(inner, dict):
                    continue
                title = inner.get("title")
                qty = it.get("quantity")
                if title:
                    suffix = f" × {qty}" if qty and qty != 1 else ""
                    parts.append(f"{title}{suffix}")
            if parts:
                lines.append("📦 " + "\n📦 ".join(parts))

        buyer = raw.get("buyer")
        if isinstance(buyer, dict):
            nick = buyer.get("nickname") or buyer.get("first_name")
            if nick:
                lines.append(f"👤 @{nick}")

        status = raw.get("status")
        if status:
            status_emoji = {"paid": "✅", "cancelled": "❌", "pending": "⏳"}.get(status, "")
            lines.append(f"{status_emoji} {status}".strip())

        when = _fmt_date(raw.get("date_created") or raw.get("date_closed"))
        if when:
            lines.append(f"🕐 {when}")

    # ── ITEMS (status changes) ────────────────────────────────────────────
    elif key == "items":
        title = raw.get("title")
        if title:
            lines.append(f"📦 {title}")
        status = raw.get("status")
        if status:
            status_emoji = {"active": "🟢", "paused": "⏸️", "closed": "🔴", "under_review": "🟡"}.get(status, "")
            lines.append(f"{status_emoji} Status: {status}".strip())
        sub = raw.get("sub_status")
        if sub:
            sub_str = sub if isinstance(sub, str) else (", ".join(str(s) for s in sub) if isinstance(sub, list) else str(sub))
            if sub_str:
                lines.append(f"⚠️ {sub_str}")
        price = raw.get("price")
        if price is not None:
            lines.append(f"💵 {_fmt_money(price, raw.get('currency_id') or 'BRL')}")

    # ── QUESTIONS ─────────────────────────────────────────────────────────
    elif key == "questions":
        text_q = raw.get("text")
        if text_q:
            # Show up to 300 chars of the question
            q = str(text_q)
            lines.append(f"💬 {q[:300]}{'…' if len(q) > 300 else ''}")
        item = raw.get("item") or {}
        if isinstance(item, dict):
            title = item.get("title")
            if title:
                lines.append(f"📦 {title}")
        from_user = raw.get("from") or {}
        if isinstance(from_user, dict):
            nick = from_user.get("nickname")
            if nick:
                lines.append(f"👤 @{nick}")
        when = _fmt_date(raw.get("date_created"))
        if when:
            lines.append(f"🕐 {when}")

    # ── CLAIMS ────────────────────────────────────────────────────────────
    elif key == "claims":
        claim_type = raw.get("type")
        if claim_type:
            lines.append(f"📄 Tipo: {claim_type}")
        reason = raw.get("reason_id") or (raw.get("reason") or {}).get("id")
        if reason:
            lines.append(f"❗ Motivo: {reason}")
        stage = raw.get("stage")
        if stage:
            lines.append(f"🔄 Etapa: {stage}")
        status = raw.get("status")
        if status:
            lines.append(f"📊 Status: {status}")
        resource = raw.get("resource") or raw.get("resource_id")
        if resource:
            lines.append(f"🔗 Pedido: {resource}")
        when = _fmt_date(raw.get("date_created"))
        if when:
            lines.append(f"🕐 {when}")

    # ── MESSAGES (post-sale chat) ─────────────────────────────────────────
    elif key == "messages":
        text = raw.get("text") or raw.get("message") or (raw.get("body") or {}).get("text") if isinstance(raw.get("body"), dict) else raw.get("text")
        if text:
            t = str(text)
            lines.append(f"💬 {t[:300]}{'…' if len(t) > 300 else ''}")
        from_user = raw.get("from") or {}
        if isinstance(from_user, dict):
            nick = from_user.get("nickname") or from_user.get("user_id")
            if nick:
                lines.append(f"👤 {nick}")
        when = _fmt_date(raw.get("message_date") or raw.get("date_created"))
        if when:
            lines.append(f"🕐 {when}")

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

    def _fix_currency(s: str) -> str:
        # The translator sometimes localizes "R$" into the target locale's
        # currency symbol (₽ for ru, $ for en, € for es). All ML notices carry
        # BRL only, so force the canonical R$ back.
        return (s
                .replace("₽", "R$")
                .replace("€", "R$")
                .replace(" $ ", " R$ "))

    if language in ("ru", "en"):
        if label and not _is_key_like(label):
            label = _fix_currency(await translate_svc.translate(label, target=language, http=http))
        if description and not _is_key_like(description):
            description = _fix_currency(await translate_svc.translate(description, target=language, http=http))

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
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": False,
    }

    # Interactive buttons for paused-item notices: "Ignorar SKU" snoozes the
    # MLB id from /escalar/products listing without leaving Telegram. The
    # callback handler in app/api/telegram-webhook recognizes the `is:` prefix.
    topic = notice.get("topic")
    raw = notice.get("raw") or {}
    actions = notice.get("actions") or []
    if topic == "items":
        item_id = ""
        # Notice_id is "items:MLB6155302096" — extract the MLB part.
        nid = str(notice.get("notice_id") or "")
        if ":" in nid:
            item_id = nid.split(":", 1)[1].strip()
        if not item_id:
            item_id = str(raw.get("resource_id") or raw.get("id") or "")
        if item_id:
            ignore_label = {
                "ru": "🚫 Игнорировать SKU",
                "en": "🚫 Ignore SKU",
                "pt": "🚫 Ignorar SKU",
            }.get(language, "🚫 Ignorar SKU")
            open_label = {
                "ru": "🔍 Открыть в ML",
                "en": "🔍 Open in ML",
                "pt": "🔍 Abrir no ML",
            }.get(language, "🔍 Abrir no ML")
            buttons = [{"text": ignore_label, "callback_data": f"is:{item_id}"}]
            # Stick the ML link into the same row if we have a permalink — keeps
            # it close to the action so user doesn't have to scroll up.
            permalink = next(
                (a.get("url") for a in actions if a and a.get("url")),
                None,
            )
            if permalink:
                buttons.append({"text": open_label, "url": permalink})
            payload["reply_markup"] = {"inline_keyboard": [buttons]}

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
