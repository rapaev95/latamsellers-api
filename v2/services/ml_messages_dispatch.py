"""Send buyer→seller order messages to Telegram with translation + buttons.

Cron-driven (every CLAIMS_TG_INTERVAL_MIN, default 5 min). Mirrors
ml_questions_dispatch / ml_claims_dispatch:

1. For each user with telegram_chat_id set:
   a. Find ml_notices rows with topic='messages' and
      messages_tg_dispatched_at IS NULL.
   b. Format card: buyer text + AI translation/summary in seller's
      language + deep-links.
   c. Send TG MarkdownV2 (with parse_mode=None fallback on 400).
   d. Mark messages_tg_dispatched_at + messages_tg_message_id.

Why not the legacy ml_notices path: that one renders a single-line
"Nova mensagem do comprador" + raw text, no translation, no rich
context. We bulk-skip topic='messages' in ml_notices._dispatch_to_telegram
so the rich path here owns delivery (mirrors questions/claims).

ML's public messaging API is restricted (10/10 candidates 404 in our
probes), so webhook-driven payload is the only data source we have.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"
TG_BATCH_CAP = 10
TG_THROTTLE = 1.1

_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
_MD_CODE_ESCAPE = str.maketrans({"`": "\\`", "\\": "\\\\"})
_MD2_UNESCAPE_RE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _esc(text: Any) -> str:
    return (str(text or "")).translate(_MD_ESCAPE)


def _esc_code(text: Any) -> str:
    return (str(text or "")).translate(_MD_CODE_ESCAPE)


def _strip_md2(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"(?<!\\)[*_~`]", "", text)
    return _MD2_UNESCAPE_RE.sub(r"\1", out)


# ── Schema ──────────────────────────────────────────────────────────

ALTER_SQL = """
ALTER TABLE ml_notices
  ADD COLUMN IF NOT EXISTS messages_tg_dispatched_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS messages_tg_message_id TEXT,
  ADD COLUMN IF NOT EXISTS messages_tg_translation TEXT,
  ADD COLUMN IF NOT EXISTS messages_tg_translation_lang TEXT;
CREATE INDEX IF NOT EXISTS idx_ml_notices_messages_pending
  ON ml_notices(user_id)
  WHERE topic = 'messages' AND messages_tg_dispatched_at IS NULL;
"""

# One-shot backfill: messages already delivered via the legacy translate
# pipeline are marked dispatched in the new system too, so the rich-format
# cron doesn't re-send them as duplicates. Idempotent — once a row's
# messages_tg_dispatched_at is set it stays set; new incoming messages
# (with telegram_sent_at IS NULL because we now skip them in legacy)
# remain NULL and get picked up by the new dispatch.
BACKFILL_SQL = """
UPDATE ml_notices
   SET messages_tg_dispatched_at = telegram_sent_at
 WHERE topic = 'messages'
   AND telegram_sent_at IS NOT NULL
   AND messages_tg_dispatched_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(ALTER_SQL)
        await conn.execute(BACKFILL_SQL)


# ── AI translation (same OpenRouter setup as questions/claims) ──────

TRANSLATE_SYSTEM_PROMPT_RU = """Ты помогаешь продавцу Mercado Livre понять что пишет покупатель.

ПРАВИЛА:
1. Переведи сообщение покупателя на русский — точно, кратко, не дословно.
2. Если сообщение длинное (>200 символов), сожми до 2-3 ключевых пунктов с «• ».
3. Если короткое (вопрос, приветствие) — просто переведи в одну строку.
4. НЕ цитируй покупателя дословно. НЕ пиши «покупатель пишет что...».
5. Без преамбулы, без заголовков. Только перевод/тезисы.
6. Максимум 300 символов всего."""

TRANSLATE_SYSTEM_PROMPT_EN = """You help an MLB seller understand what the buyer is writing.

RULES:
1. Translate the buyer's message to English — accurate, concise, not literal.
2. If long (>200 chars), compress to 2-3 bullets with "• ".
3. If short (question, greeting) — one-line translation.
4. Don't quote verbatim. No "The buyer says...".
5. No preamble, no headers. Just translation/bullets.
6. Max 300 chars total."""


async def _translate(
    http: httpx.AsyncClient,
    text: str,
    target_lang: str = "ru",
) -> Optional[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or not text or not text.strip():
        return None

    target = (target_lang or "ru").lower()
    if target == "en":
        system_prompt = TRANSLATE_SYSTEM_PROMPT_EN
    else:
        system_prompt = TRANSLATE_SYSTEM_PROMPT_RU

    try:
        r = await http.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://app.lsprofit.app",
                "X-Title": "LS Profit App",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:2000]},
                ],
            },
            timeout=15.0,
        )
        if r.status_code != 200:
            log.warning("OpenRouter translate %s: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return content.strip() if isinstance(content, str) else None
    except Exception as err:  # noqa: BLE001
        log.exception("message translate failed: %s", err)
        return None


# ── Card builder ────────────────────────────────────────────────────

def _format_age(date_iso: Optional[str]) -> str:
    if not date_iso:
        return ""
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(str(date_iso).replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 3600:
            return f"há {max(1, secs // 60)} min"
        if secs < 86400:
            return f"há {secs // 3600}h"
        return f"há {secs // 86400}d"
    except Exception:  # noqa: BLE001
        return ""


def _extract_message_text(notice: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull buyer text + metadata from notice's raw payload."""
    raw = notice.get("raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    text = str(raw.get("text") or raw.get("message") or notice.get("description") or "")
    text = _HTML_TAG_RE.sub("", text).strip()
    return text, raw


def _build_message_card(
    notice: dict[str, Any],
    translation: Optional[str] = None,
    translation_lang: str = "ru",
) -> str:
    text, raw = _extract_message_text(notice)
    pack_id = raw.get("pack_id") or raw.get("resource_id") or ""
    from_user = (raw.get("from") or {}).get("user_id") if isinstance(raw.get("from"), dict) else ""
    from_user = from_user or raw.get("from_user_id") or ""
    nickname = (raw.get("from") or {}).get("nickname") if isinstance(raw.get("from"), dict) else ""
    age = _format_age(raw.get("date_created") or notice.get("from_date"))

    lines: list[str] = ["📩 *Nova mensagem do comprador*"]
    lines.append("")
    if pack_id:
        lines.append(f"📦 *Venda:* `{_esc_code(pack_id)}`")
    if nickname:
        lines.append(f"👤 @{_esc(nickname)}")
    elif from_user:
        lines.append(f"👤 ID `{_esc_code(from_user)}`")
    if age:
        lines.append(f"🕒 {_esc(age)}")

    if text:
        lines.append("")
        lines.append("💬 *Mensagem original:*")
        clip = text[:400].rstrip()
        if len(text) > 400:
            clip += "…"
        lines.append(_esc(clip))

    if translation and translation.strip():
        header = (
            "🌐 *Перевод:*" if translation_lang == "ru"
            else ("🌐 *Translation:*" if translation_lang == "en" else "🌐 *Tradução:*")
        )
        lines.append("")
        lines.append(header)
        lines.append(_esc(translation.strip()))

    return "\n".join(lines)


def _build_keyboard(pack_id: Any, app_base_url: str) -> dict[str, Any]:
    pack = str(pack_id or "").strip()
    app_link = f"{app_base_url.rstrip('/')}/escalar"  # no dedicated /messages page yet
    ml_link = f"https://www.mercadolivre.com.br/mensagens/{pack}" if pack else (
        "https://www.mercadolivre.com.br/vendas/novo/mensagens"
    )
    return {
        "inline_keyboard": [
            [
                {"text": "💬 Responder no ML", "url": ml_link},
                {"text": "⚡ Abrir no app", "url": app_link},
            ],
        ],
    }


# ── TG send (with MD2 fallback) ─────────────────────────────────────

async def _tg_post(http: httpx.AsyncClient, bot_token: str, payload: dict[str, Any]) -> tuple[int, str, Optional[str]]:
    r = await http.post(
        f"{TG_API_BASE}/bot{bot_token}/sendMessage",
        json=payload, timeout=10.0,
    )
    body_preview = r.text[:200] if r.status_code != 200 else ""
    msg_id: Optional[str] = None
    if r.status_code == 200:
        mid = (r.json() or {}).get("result", {}).get("message_id")
        msg_id = str(mid) if mid else None
    return r.status_code, body_preview, msg_id


async def _send_message(
    http: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    notice: dict[str, Any],
    app_base_url: str,
    translation: Optional[str] = None,
    translation_lang: str = "ru",
) -> Optional[str]:
    text = _build_message_card(notice, translation=translation, translation_lang=translation_lang)
    if len(text) > 4000:
        text = text[:3990] + "…"

    raw = notice.get("raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            raw = {}
    pack_id = (raw or {}).get("pack_id") or (raw or {}).get("resource_id") or ""
    keyboard = _build_keyboard(pack_id, app_base_url)

    try:
        status, body_preview, msg_id = await _tg_post(http, bot_token, {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "reply_markup": keyboard,
            "disable_web_page_preview": True,
        })
        if status == 200 and msg_id:
            return msg_id
        if status == 400 and "parse" in body_preview.lower():
            log.warning("TG MD2 parse failed for message notice %s: %s", notice.get("notice_id"), body_preview)
            plain = _strip_md2(text)
            status2, body2, msg_id2 = await _tg_post(http, bot_token, {
                "chat_id": chat_id,
                "text": plain,
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            })
            if status2 == 200 and msg_id2:
                return msg_id2
            log.warning("TG plain fallback also failed: %s %s", status2, body2)
        log.warning("TG send_message notice=%s status=%s body=%s", notice.get("notice_id"), status, body_preview)
    except Exception as err:  # noqa: BLE001
        log.exception("TG send message %s failed: %s", notice.get("notice_id"), err)
    return None


# ── Batched per-user dispatch ────────────────────────────────────────
# ML's messaging content API is closed to our app (10/13 endpoints 404,
# 1 returned 403 "Invalid caller.id" on /marketplace/messages — endpoint
# exists but we lack permissions). The webhook only delivers metadata
# (topic, resource_id, sent), no text/pack_id/from. So per-message rich
# cards would all be empty — useless noise.
#
# Strategy: ONE batched card per user per cron tick saying "you have N
# new buyer messages". Click → ML inbox. Mark ALL pending rows as
# dispatched together.

def _build_batched_card(count: int, lang: str) -> str:
    if lang == "ru":
        title = "📩 *Новые сообщения от покупателей*"
        if count == 1:
            body = "У вас *1* новое непрочитанное сообщение\\."
        else:
            body = f"У вас *{count}* новых непрочитанных сообщения\\."
        cta = "_Откройте центр сообщений Mercado Livre, чтобы прочитать и ответить\\._"
    elif lang == "en":
        title = "📩 *New buyer messages*"
        body = f"You have *{count}* unread message{'s' if count > 1 else ''}\\."
        cta = "_Open the Mercado Livre inbox to read and reply\\._"
    else:
        title = "📩 *Novas mensagens de compradores*"
        plural = "ns" if count > 1 else "m"
        body = f"Você tem *{count}* mensag{('e' + plural)} não lida{'s' if count > 1 else ''}\\."
        cta = "_Abra a central de mensagens do Mercado Livre para ler e responder\\._"

    return "\n\n".join([title, body, cta])


def _build_batched_keyboard(app_base_url: str, lang: str) -> dict[str, Any]:
    if lang == "ru":
        ml_label = "💬 Открыть в ML"
        app_label = "⚡ Перейти в app"
    elif lang == "en":
        ml_label = "💬 Open ML inbox"
        app_label = "⚡ Open in app"
    else:
        ml_label = "💬 Abrir no ML"
        app_label = "⚡ Abrir no app"
    return {
        "inline_keyboard": [
            [
                {"text": ml_label, "url": "https://www.mercadolivre.com.br/vendas/novo/mensagens"},
                {"text": app_label, "url": f"{app_base_url.rstrip('/')}/escalar"},
            ],
        ],
    }


async def _dispatch_for_user(pool: asyncpg.Pool, user_id: int, app_base_url: str) -> dict[str, int]:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"sent": 0}

    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
        if not settings or not settings["telegram_chat_id"]:
            return {"sent": 0}
        chat_id = str(settings["telegram_chat_id"])
        target_lang = (settings["language"] or "pt").lower()

        # Pull all pending messages-topic notices for this user. We don't
        # cap (TG_BATCH_CAP not used) because we send ONE batched card no
        # matter how many — and we must mark all of them dispatched in the
        # same transaction to avoid re-sending.
        rows = await conn.fetch(
            """
            SELECT notice_id
              FROM ml_notices
             WHERE user_id = $1
               AND topic = 'messages'
               AND messages_tg_dispatched_at IS NULL
            """,
            user_id,
        )

    if not rows:
        return {"sent": 0}

    count = len(rows)
    notice_ids = [r["notice_id"] for r in rows]

    text = _build_batched_card(count, target_lang)
    keyboard = _build_batched_keyboard(app_base_url, target_lang)

    msg_id: Optional[str] = None
    async with httpx.AsyncClient() as http:
        try:
            status, body_preview, msg_id = await _tg_post(http, bot_token, {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "MarkdownV2",
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            })
            if not (status == 200 and msg_id) and status == 400 and "parse" in body_preview.lower():
                plain = _strip_md2(text)
                status2, body2, msg_id = await _tg_post(http, bot_token, {
                    "chat_id": chat_id,
                    "text": plain,
                    "reply_markup": keyboard,
                    "disable_web_page_preview": True,
                })
                if status2 != 200:
                    log.warning("TG plain fallback also failed: %s %s", status2, body2)
            elif status != 200:
                log.warning("TG batched messages send failed status=%s body=%s", status, body_preview)
        except Exception as err:  # noqa: BLE001
            log.exception("TG batched messages send exception: %s", err)

    if not msg_id:
        return {"sent": 0, "pending": count}

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ml_notices
               SET messages_tg_dispatched_at = NOW(),
                   messages_tg_message_id = $1
             WHERE user_id = $2
               AND notice_id = ANY($3::text[])
            """,
            msg_id, user_id, notice_ids,
        )
    return {"sent": 1, "messages": count}


# ── Cron entry point ─────────────────────────────────────────────────

async def dispatch_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    if pool is None:
        return {"users": 0, "sent": 0}
    await ensure_schema(pool)

    app_base_url = os.environ.get("APP_BASE_URL", "https://app.lsprofit.app")

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
    user_ids = [r["user_id"] for r in rows]
    totals = {"users": 0, "sent": 0}
    for uid in user_ids:
        try:
            res = await _dispatch_for_user(pool, uid, app_base_url)
            totals["users"] += 1
            totals["sent"] += res["sent"]
        except Exception as err:  # noqa: BLE001
            log.exception("messages dispatch user %s failed: %s", uid, err)
        await asyncio.sleep(0.3)
    return totals
