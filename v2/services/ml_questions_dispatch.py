"""Send unanswered questions to seller's Telegram with AI-suggested reply.

Cron-driven (every NOTICES_SYNC_INTERVAL_MIN, default 5 min):
1. For each user with telegram_chat_id set:
   a. Refresh ml_user_questions cache (so latest UNANSWERED are present)
   b. Find UNANSWERED questions where tg_dispatched_at IS NULL
   c. Generate AI suggestion via OpenRouter (gpt-4o-mini, pt-BR)
   d. Send TG message with inline keyboard:
      ✅ Aprovar  /  ✏️ Editar  /  🔄 Outra sugestão
   e. Mark tg_dispatched_at + tg_message_id

Schema migration is applied lazily via ensure_schema() (idempotent ALTER TABLE).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_user_questions as questions_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"
TG_BATCH_CAP = 10  # max questions per user per tick (avoid TG rate limit)
TG_THROTTLE = 1.1  # 1 sec between TG sends per chat

# MarkdownV2 escape per Telegram spec
_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})


def _esc(text: str) -> str:
    return (text or "").translate(_MD_ESCAPE)


SUGGEST_SYSTEM_PROMPT = """Você é um vendedor profissional do Mercado Livre Brasil.

REGRAS:
1. Resposta em português brasileiro, tom amigável e profissional
2. 1-3 frases (máx 280 caracteres)
3. Se a pergunta é sobre dimensões/cor/material e a info NÃO está no título — peça desculpas e diga que verificará
4. NUNCA invente specs (dimensões, peso, materiais)
5. Termine positivamente
6. Máx 1 emoji
7. SEM links externos

Output APENAS o texto da resposta — sem aspas, sem prefácio."""


# ── Schema bootstrap (migration) ──────────────────────────────────────────────

ALTER_SQL = """
ALTER TABLE ml_user_questions
  ADD COLUMN IF NOT EXISTS tg_dispatched_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS tg_message_id TEXT,
  ADD COLUMN IF NOT EXISTS tg_suggestion TEXT;
CREATE INDEX IF NOT EXISTS idx_ml_questions_tg_pending
  ON ml_user_questions(user_id) WHERE status = 'UNANSWERED' AND tg_dispatched_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(ALTER_SQL)


# ── OpenRouter AI suggestion ──────────────────────────────────────────────────

async def _ai_suggest(http: httpx.AsyncClient, question_text: str, item_title: str) -> Optional[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — skipping AI suggestion")
        return None
    user_msg = (
        f'Anúncio: "{item_title or "sem título"}"\n'
        f'Pergunta do comprador: "{question_text}"\n\n'
        f"Escreva a resposta agora."
    )
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
                    {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=15.0,
        )
        if r.status_code != 200:
            log.warning("OpenRouter %s: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return content.strip() if isinstance(content, str) else None
    except Exception as err:  # noqa: BLE001
        log.exception("AI suggest failed: %s", err)
        return None


# ── Telegram send with inline keyboard ────────────────────────────────────────

async def _tg_send_question(
    http: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    question_id: int,
    item_title: str,
    item_thumbnail: Optional[str],
    item_permalink: Optional[str],
    buyer_nickname: Optional[str],
    question_text: str,
    suggestion: str,
) -> Optional[str]:
    """Send a question card with action buttons. Returns tg message_id on success."""
    body_lines: list[str] = []
    body_lines.append(f"❓ *Nova pergunta*")
    if item_title:
        body_lines.append(f"📦 {_esc(item_title)}")
    if buyer_nickname:
        body_lines.append(f"👤 @{_esc(buyer_nickname)}")
    body_lines.append("")
    body_lines.append(f"💬 _{_esc(question_text)}_")
    body_lines.append("")
    body_lines.append(f"🪄 *Sugestão de resposta:*")
    body_lines.append(_esc(suggestion))
    text = "\n".join(body_lines)
    if len(text) > 4000:
        text = text[:3990] + "…"

    keyboard = {
        "inline_keyboard": [
            [
                {"text": "✅ Aprovar", "callback_data": f"qa:{question_id}"},
                {"text": "✏️ Editar", "callback_data": f"qe:{question_id}"},
            ],
            [
                {"text": "🔄 Outra sugestão", "callback_data": f"qr:{question_id}"},
                *([{"text": "🔗 Ver no ML", "url": item_permalink}] if item_permalink else []),
            ],
        ],
    }

    try:
        r = await http.post(
            f"{TG_API_BASE}/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "MarkdownV2",
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            log.warning("TG send_message %s: %s", r.status_code, r.text[:200])
            return None
        msg_id = (r.json() or {}).get("result", {}).get("message_id")
        return str(msg_id) if msg_id else None
    except Exception as err:  # noqa: BLE001
        log.exception("TG send failed: %s", err)
        return None


# ── Per-user dispatch ─────────────────────────────────────────────────────────

async def _dispatch_for_user(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    """Find pending UNANSWERED questions for this user, generate AI replies, send to TG."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"sent": 0, "skipped": 0}

    async with pool.acquire() as conn:
        # Get user's TG settings — must have chat_id and notify_daily_sales/etc enabled
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language,
                   COALESCE(notify_daily_sales, TRUE) AS notify_sales
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
        if not settings or not settings["telegram_chat_id"]:
            return {"sent": 0, "skipped": 0}
        chat_id = str(settings["telegram_chat_id"])

        # Refresh questions cache first so we have latest UNANSWERED
    await questions_svc.refresh_user_questions(pool, user_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT question_id, item_id, text, raw, from_nickname
              FROM ml_user_questions
             WHERE user_id = $1
               AND status = 'UNANSWERED'
               AND tg_dispatched_at IS NULL
             ORDER BY date_created DESC
             LIMIT $2
            """,
            user_id, TG_BATCH_CAP,
        )

    sent = 0
    async with httpx.AsyncClient() as http:
        for row in rows:
            qid = int(row["question_id"])
            text = row["text"] or ""
            # Pull item_title + permalink + thumbnail from raw payload
            raw = row["raw"]
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:  # noqa: BLE001
                    raw = {}
            elif raw is None:
                raw = {}
            item = (raw or {}).get("item") or {}
            item_title = item.get("title") or ""
            item_thumbnail = item.get("thumbnail") or item.get("secure_thumbnail")
            item_permalink = item.get("permalink")
            buyer_nick = row["from_nickname"]

            suggestion = await _ai_suggest(http, text, item_title) or "(falha ao gerar sugestão; responda manualmente)"

            msg_id = await _tg_send_question(
                http, bot_token, chat_id, qid,
                item_title, item_thumbnail, item_permalink, buyer_nick,
                text, suggestion,
            )

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE ml_user_questions
                       SET tg_dispatched_at = NOW(),
                           tg_message_id = $1,
                           tg_suggestion = $2
                     WHERE user_id = $3 AND question_id = $4
                    """,
                    msg_id, suggestion, user_id, qid,
                )
            if msg_id:
                sent += 1
            await asyncio.sleep(TG_THROTTLE)

    return {"sent": sent, "skipped": 0}


# ── Cron entry point (called from main.py APScheduler) ────────────────────────

async def dispatch_all_users(pool: asyncpg.Pool) -> dict[str, int]:
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
            """
        )
    user_ids = [r["user_id"] for r in rows]
    totals = {"users": 0, "sent": 0}
    for uid in user_ids:
        try:
            res = await _dispatch_for_user(pool, uid)
            totals["users"] += 1
            totals["sent"] += res["sent"]
        except Exception as err:  # noqa: BLE001
            log.exception("questions dispatch user %s failed: %s", uid, err)
        await asyncio.sleep(0.5)
    return totals
