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
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_user_questions as questions_svc
from . import ml_oauth as ml_oauth_svc
from . import ml_item_context as item_ctx_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "anthropic/claude-sonnet-4.5"  # match Next.js ai-suggest, supports vision
IMAGE_DETAIL = "high"  # ~$0.01/image; 6 images ≈ $0.06 per suggestion
TG_BATCH_CAP = 10  # max questions per user per tick (avoid TG rate limit)
TG_THROTTLE = 1.1  # 1 sec between TG sends per chat

# Reminder cadence for questions still UNANSWERED after the first dispatch.
# All three are env-tunable so cadence can change without a redeploy.
#
# Defaults pick a balance between "ML penalizes unanswered questions hard"
# (so we WANT to keep reminding) and "don't spam the seller with one TG
# message per stale question every hour".
#   - first reminder 12h after the initial dispatch
#   - subsequent every 48h (every 2 days)
#   - up to 30 reminders per question — that's ~60 days of nagging.
#     After that the seller has clearly chosen to ignore; either
#     answer it or pause the item. The cap also keeps the BATCH safe
#     when a user has dozens of stale questions on inactive listings.
REMINDER_FIRST_HOURS = float(os.environ.get("QUESTIONS_REMINDER_FIRST_HOURS", "12"))
REMINDER_INTERVAL_HOURS = float(os.environ.get("QUESTIONS_REMINDER_INTERVAL_HOURS", "48"))
REMINDER_MAX_COUNT = int(os.environ.get("QUESTIONS_REMINDER_MAX_COUNT", "30"))

# MarkdownV2 escape per Telegram spec
_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
# Inside inline code (`...`) only ` and \ need escaping (per TG MarkdownV2 spec)
_MD_CODE_ESCAPE = str.maketrans({"`": "\\`", "\\": "\\\\"})


def _esc(text: str) -> str:
    return (text or "").translate(_MD_ESCAPE)


def _esc_code(text: str) -> str:
    """Escape for use INSIDE `...` inline code blocks."""
    return (text or "").translate(_MD_CODE_ESCAPE)


# Strip MarkdownV2 escape backslashes for the plain-text fallback path.
# Keeps real backslashes from doubling — only removes the escape preceding
# one of the MD2 special characters.
import re as _re  # noqa: E402

_MD2_UNESCAPE_RE = _re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


def _strip_md2_escapes(text: str) -> str:
    """Convert MarkdownV2-formatted text to plain text.

    Removes formatting markers (* _ ` ~) without consuming the surrounding
    content so "*Nova pergunta*" → "Nova pergunta". Then de-escapes any
    \\X back to X. Used when TG rejects our MarkdownV2 — better to send
    ugly plain text than nothing.
    """
    if not text:
        return ""
    # Drop formatting markers ONLY when they look like balanced toggles —
    # i.e., not preceded by a backslash. We use a regex with negative
    # look-behind so escaped `\*` survives this step.
    out = _re.sub(r"(?<!\\)[*_~`]", "", text)
    # Now safely de-escape \X → X for all special chars.
    return _MD2_UNESCAPE_RE.sub(r"\1", out)


SUGGEST_SYSTEM_PROMPT = """Você é um vendedor profissional do Mercado Livre Brasil respondendo perguntas de compradores.

═══════════════════════════════════════════════════
REGRA #1 — NUNCA INVENTE
═══════════════════════════════════════════════════
Se a resposta NÃO está literalmente no CONTEXTO ou nas FOTOS — responda:
"Vou verificar essa informação e respondo em breve. 😊"

⚠️ MAS: NÃO use "vou verificar" como resposta preguiçosa. Antes de dizer:
  1. RELEIA todo o CONTEXTO (atributos + descrição completa).
  2. INSPECIONE TODAS as fotos uma por uma — fotos 2-6 normalmente mostram interior/ângulos/medidas.
  3. Se a resposta está visível em alguma foto OU em qualquer parte do contexto → RESPONDA.

═══════════════════════════════════════════════════
COMO RESPONDER POR TIPO DE PERGUNTA
═══════════════════════════════════════════════════

▸ MARCA / FABRICANTE
  Só responda se atributo "Marca" tem valor explícito. NÃO confunda título com marca.

▸ MATERIAL
  Use a PALAVRA EXATA do contexto. "couro" sem qualificador → "couro" (não "legítimo").

▸ COR / APARÊNCIA VISUAL
  Você TEM as fotos. OLHE PARA ELAS AGORA. Descreva o que VÊ.

▸ CONTAGEM / NÚMERO DE ELEMENTOS (quantas divisórias, bolsos, compartimentos, alças)
  - INSPECIONE TODAS AS FOTOS antes de responder. Conte os elementos visíveis.
  - Fotos de bolsas/mochilas/organizadores tipicamente mostram o INTERIOR aberto em fotos 2-6.
  - Resposta tipo: "Pelo que mostram as fotos, esta bolsa tem 3 divisórias principais (compartimento central + 2 bolsos com zíper)."
  - Não diga "Vou verificar" se a foto mostra a resposta — é desperdício de tempo do comprador.

▸ DIMENSÕES / PESO / CAPACIDADE
  - Atributo primeiro, depois DESCRIÇÃO (ex: "18 cm ALTURA, 24 cm LARGURA"), depois fotos com etiqueta/régua.
  - Só "vou verificar" se NEM contexto, NEM descrição, NEM fotos têm a info.

▸ COMPATIBILIDADE
  Só afirme se contexto lista X explicitamente. Senão "Vou confirmar a compatibilidade com [X]."

═══════════════════════════════════════════════════
ABORDAGEM DE VENDAS
═══════════════════════════════════════════════════
Atributo "Gênero: Masculino" não é proibição — diga unissex.
NUNCA termine com "Não" seco — sempre alternativa OU pergunta qualificadora.
Cliente honesto > venda forçada.

═══════════════════════════════════════════════════
FORMATO
═══════════════════════════════════════════════════
- Português brasileiro, tom amigável mas direto
- 1-3 frases (máx 350 caracteres)
- Termine cordial: "Qualquer dúvida, estou à disposição!" ou similar
- Máx 1 emoji
- NÃO mencione SKU (é interno)
- NÃO inclua links externos
- Output APENAS o texto da resposta — sem aspas, sem prefácio."""


# ── Product context loader (ML attrs + description + custom docs + SKU) ──────

def _normalize_item_id(raw: str | None) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().upper()
    if not s:
        return None
    numeric = s[3:] if s.startswith("MLB") else s
    clean = numeric.split(".", 1)[0]
    if not clean.isdigit():
        return None
    return f"MLB{clean}"


async def _fetch_product_context(
    http: httpx.AsyncClient,
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
) -> dict[str, Any]:
    """Aggregate cached ML context + custom docs + SKU.

    Cache-first via ml_item_context (TTL=24h). Falls back to live ML fetch
    on miss/stale, persists result to DB for next time. DB-only data
    (escalar_item_docs, ml_user_items.sku) is always read fresh.
    """
    mlb = _normalize_item_id(item_id)
    out: dict[str, Any] = {"itemId": mlb, "title": "", "sku": None, "permalink": None,
                          "attributes": [], "description": "", "customDocs": []}
    if not mlb:
        return out

    # Three parallel reads: cached ML context, DB docs, DB sku
    async def _ml_ctx():
        try:
            return await item_ctx_svc.get_or_refresh(pool, http, user_id, mlb)
        except Exception as err:  # noqa: BLE001
            log.warning("get_or_refresh failed user=%s item=%s: %s", user_id, mlb, err)
            return None

    async def _db_docs():
        try:
            async with pool.acquire() as conn:
                # Table is created by Next.js (ensureSchema); harmless if not yet
                rows = await conn.fetch(
                    """
                    SELECT kind, title, content
                      FROM escalar_item_docs
                     WHERE user_id = $1 AND item_id = $2
                     ORDER BY updated_at DESC LIMIT 20
                    """,
                    user_id, mlb,
                )
                return [{"kind": r["kind"], "title": r["title"], "content": r["content"]} for r in rows]
        except Exception:  # noqa: BLE001
            return []

    async def _db_sku():
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT sku FROM ml_user_items WHERE user_id = $1 AND item_id = $2 LIMIT 1",
                    user_id, mlb,
                )
                return row["sku"] if row else None
        except Exception:  # noqa: BLE001
            return None

    ml_ctx, docs, sku = await asyncio.gather(
        _ml_ctx(), _db_docs(), _db_sku(), return_exceptions=False,
    )

    if ml_ctx:
        out["title"] = ml_ctx.get("title") or ""
        out["permalink"] = ml_ctx.get("permalink")
        out["condition"] = ml_ctx.get("condition")
        out["price"] = ml_ctx.get("price")
        out["currency"] = ml_ctx.get("currency") or "BRL"
        out["availableQuantity"] = ml_ctx.get("available_quantity")
        out["warranty"] = ml_ctx.get("warranty")
        out["shippingFree"] = ml_ctx.get("shipping_free")
        out["attributes"] = ml_ctx.get("attributes") or []
        out["description"] = ml_ctx.get("description") or ""
        out["pictures"] = ml_ctx.get("pictures") or []
        out["fetchedAt"] = ml_ctx.get("fetched_at")
    out["customDocs"] = docs or []
    out["sku"] = sku
    return out


def _build_context_block(ctx: dict[str, Any]) -> str:
    lines: list[str] = []
    if ctx.get("title"):
        lines.append(f"Título do anúncio: {ctx['title']}")
    if ctx.get("sku"):
        lines.append(f"SKU interno: {ctx['sku']}")
    if ctx.get("condition"):
        lines.append(f"Condição: {ctx['condition']}")
    if ctx.get("price") is not None:
        lines.append(f"Preço: {ctx['price']} {ctx.get('currency', 'BRL')}")
    if ctx.get("availableQuantity") is not None:
        lines.append(f"Quantidade disponível: {ctx['availableQuantity']}")
    if ctx.get("warranty"):
        lines.append(f"Garantia: {ctx['warranty']}")
    if ctx.get("shippingFree") is not None:
        lines.append(f"Frete grátis: {'sim' if ctx['shippingFree'] else 'não'}")

    attrs = ctx.get("attributes") or []
    if attrs:
        lines.append("")
        lines.append("Características técnicas (do anúncio ML):")
        for a in attrs:
            lines.append(f"  • {a['name']}: {a['value']}")
    desc = (ctx.get("description") or "").strip()
    if desc:
        lines.append("")
        lines.append("Descrição do anúncio:")
        lines.append(desc[:2000])
    docs = ctx.get("customDocs") or []
    if docs:
        lines.append("")
        lines.append("Documentos adicionais do vendedor (manuais, FAQs, notas):")
        for d in docs:
            head = f"[{d.get('kind', 'note')}]"
            if d.get("title"):
                head += f" {d['title']}"
            lines.append(f"--- {head} ---")
            lines.append((d.get("content") or "")[:1500])
    return "\n".join(lines)


# ── Schema bootstrap (migration) ──────────────────────────────────────────────

ALTER_SQL = """
ALTER TABLE ml_user_questions
  ADD COLUMN IF NOT EXISTS tg_dispatched_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS tg_message_id TEXT,
  ADD COLUMN IF NOT EXISTS tg_suggestion TEXT,
  ADD COLUMN IF NOT EXISTS tg_reminder_count INT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS tg_last_reminder_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_ml_questions_tg_pending
  ON ml_user_questions(user_id) WHERE status = 'UNANSWERED' AND tg_dispatched_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_ml_questions_tg_reminders
  ON ml_user_questions(user_id, tg_dispatched_at)
  WHERE status = 'UNANSWERED' AND tg_dispatched_at IS NOT NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(ALTER_SQL)


# ── OpenRouter AI suggestion ──────────────────────────────────────────────────

async def _ai_suggest(
    http: httpx.AsyncClient,
    question_text: str,
    item_title: str,
    context_block: Optional[str] = None,
    picture_urls: Optional[list[str]] = None,
    photo_descriptions_block: Optional[str] = None,
    pool: Optional[asyncpg.Pool] = None,
    user_id: Optional[int] = None,
) -> Optional[str]:
    """Generate AI suggestion. Multimodal: passes up to 6 product photos to
    Claude as image_url blocks alongside the textual product context. Without
    photos Claude cannot answer count/dimension/visual questions and falls
    back to "Vou verificar" — exactly the bug user reported.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — skipping AI suggestion")
        return None
    pic_urls = [u for u in (picture_urls or []) if u][:6]
    photo_hint = ""
    if pic_urls:
        photo_hint = (
            f"\n\n[{len(pic_urls)} foto(s) anexada(s) — INSPECIONE cada uma; "
            "fotos 2-6 normalmente mostram interior/ângulos/medidas]"
        )
    desc_section = ""
    if photo_descriptions_block:
        desc_section = f"\n\n{photo_descriptions_block}"

    if context_block:
        user_text = (
            "CONTEXTO DO PRODUTO (use SOMENTE estas informações para responder):\n"
            f"{context_block}{desc_section}{photo_hint}\n\n"
            "---\n\n"
            f'Pergunta do comprador: "{question_text}"\n\n'
            "Escreva a resposta agora, baseada APENAS no contexto, descrições e fotos acima."
        )
    else:
        user_text = (
            f'Anúncio: "{item_title or "sem título"}"{photo_hint}\n'
            f'Pergunta do comprador: "{question_text}"\n\n'
            "⚠️ Sem informações detalhadas do produto. Se a pergunta exige specs e nem as fotos respondem, peça desculpas e diga que vai verificar."
        )

    if pic_urls:
        user_content: Any = [
            {"type": "text", "text": user_text},
            *[
                {"type": "image_url", "image_url": {"url": u, "detail": IMAGE_DETAIL}}
                for u in pic_urls
            ],
        ]
    else:
        user_content = user_text

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
                "max_tokens": 250,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            log.warning("OpenRouter %s: %s", r.status_code, r.text[:200])
            try:
                from . import tg_admin_alerts as _alerts
                await _alerts.alert_openrouter_failure(
                    r.status_code, r.text, service="questions/ai-suggest",
                )
            except Exception as err:  # noqa: BLE001
                log.debug("admin alert failed: %s", err)
            try:
                from . import ai_usage_tracker as _tracker
                await _tracker.log_call(
                    pool, user_id=user_id, service="questions/ai-suggest",
                    model=LLM_MODEL, response_data=None, status_code=r.status_code,
                )
            except Exception:  # noqa: BLE001
                pass
            return None
        data = r.json()
        try:
            from . import ai_usage_tracker as _tracker
            await _tracker.log_call(
                pool, user_id=user_id, service="questions/ai-suggest",
                model=LLM_MODEL, response_data=data, status_code=200,
            )
        except Exception:  # noqa: BLE001
            pass
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
    sku: Optional[str] = None,
    item_id: Optional[str] = None,
    context_used: bool = False,
) -> Optional[str]:
    """Send a question card with action buttons. Returns tg message_id on success."""
    body_lines: list[str] = []
    body_lines.append(f"❓ \\[PERGUNTA\\] *Nova pergunta*")
    if item_title:
        body_lines.append(f"📦 {_esc(item_title)}")
    # SKU + item_id prominently — seller needs them to identify the product
    sku_line_parts: list[str] = []
    if sku:
        sku_line_parts.append(f"🏷 *SKU:* `{_esc_code(sku)}`")
    if item_id:
        sku_line_parts.append(f"🆔 `{_esc_code(item_id)}`")
    if sku_line_parts:
        body_lines.append(" / ".join(sku_line_parts))
    if buyer_nickname:
        body_lines.append(f"👤 @{_esc(buyer_nickname)}")
    body_lines.append("")
    body_lines.append(f"💬 _{_esc(question_text)}_")
    body_lines.append("")
    suggestion_label = "🪄 *Sugestão \\(com contexto do produto\\):*" if context_used else "🪄 *Sugestão \\(sem contexto detalhado\\):*"
    body_lines.append(suggestion_label)
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

    async def _post(payload: dict[str, Any]) -> tuple[int, str, Optional[str]]:
        r = await http.post(
            f"{TG_API_BASE}/bot{bot_token}/sendMessage",
            json=payload,
            timeout=10.0,
        )
        body_preview = r.text[:200] if r.status_code != 200 else ""
        msg_id: Optional[str] = None
        if r.status_code == 200:
            mid = (r.json() or {}).get("result", {}).get("message_id")
            msg_id = str(mid) if mid else None
        return r.status_code, body_preview, msg_id

    try:
        # First attempt: MarkdownV2 with rich formatting.
        status, body_preview, msg_id = await _post({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "reply_markup": keyboard,
            "disable_web_page_preview": True,
        })
        if status == 200 and msg_id:
            return msg_id

        # MarkdownV2 parser rejected something (TG returns 400 + "can't parse
        # entities"). Strip ALL Markdown escapes and resend as plain text so
        # the seller still gets the question + suggestion + buttons. Better
        # ugly than silent.
        if status == 400 and "parse" in body_preview.lower():
            log.warning("TG MD2 parse failed (q=%s): %s — retrying plain", question_id, body_preview)
            plain_text = _strip_md2_escapes(text)
            status2, body2, msg_id2 = await _post({
                "chat_id": chat_id,
                "text": plain_text,
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            })
            if status2 == 200 and msg_id2:
                return msg_id2
            log.warning("TG plain fallback also failed (q=%s): %s %s", question_id, status2, body2)
            return None

        log.warning("TG send_message %s: %s", status, body_preview)
        return None
    except Exception as err:  # noqa: BLE001
        log.exception("TG send failed: %s", err)
        return None


async def _tg_send_reminder(
    http: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    question_id: int,
    item_title: str,
    item_permalink: Optional[str],
    buyer_nickname: Optional[str],
    question_text: str,
    suggestion: Optional[str],
    hours_pending: int,
    reply_to_message_id: Optional[str],
) -> Optional[str]:
    """Send a compact reminder for an unanswered question. Threads as a reply
    to the original card when possible; falls back to a standalone message
    if reply_to fails (original message deleted)."""
    # Format the pending duration humanely — ">3000h pending" reads ugly
    # for questions that have been stale for months.
    if hours_pending < 48:
        pending_label = f"~{hours_pending}h"
    elif hours_pending < 24 * 60:  # < 60 days
        pending_label = f"~{hours_pending // 24}d"
    else:
        pending_label = f"~{hours_pending // (24 * 30)}mo"
    body_lines = [f"🔔 \\[PERGUNTA\\] *LEMBRETE \\(sem resposta há {pending_label}\\)*"]
    if item_title:
        body_lines.append(f"📦 {_esc(item_title)}")
    if buyer_nickname:
        body_lines.append(f"👤 @{_esc(buyer_nickname)}")
    body_lines.append("")
    body_lines.append(f"💬 _{_esc(question_text)}_")
    if suggestion:
        body_lines.append("")
        body_lines.append("🪄 *Sugestão anterior:*")
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

    async def _post(payload: dict[str, Any]) -> tuple[int, Optional[str]]:
        r = await http.post(
            f"{TG_API_BASE}/bot{bot_token}/sendMessage",
            json=payload,
            timeout=10.0,
        )
        msg_id: Optional[str] = None
        if r.status_code == 200:
            mid = (r.json() or {}).get("result", {}).get("message_id")
            msg_id = str(mid) if mid else None
        return r.status_code, msg_id

    try:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "reply_markup": keyboard,
            "disable_web_page_preview": True,
        }
        if reply_to_message_id:
            # allow_sending_without_reply: still send the reminder if the
            # original card was deleted from the chat (TG would otherwise 400).
            payload["reply_parameters"] = {
                "message_id": int(reply_to_message_id),
                "allow_sending_without_reply": True,
            }
        status, msg_id = await _post(payload)
        if status == 200 and msg_id:
            return msg_id

        plain_text = _strip_md2_escapes(text)
        payload["text"] = plain_text
        payload.pop("parse_mode", None)
        status2, msg_id2 = await _post(payload)
        if status2 == 200 and msg_id2:
            return msg_id2
        log.warning("TG reminder send failed (q=%s): status=%s", question_id, status2)
        return None
    except Exception as err:  # noqa: BLE001
        log.exception("TG reminder send failed: %s", err)
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
            SELECT q.question_id, q.item_id, q.text, q.raw, q.from_nickname
              FROM ml_user_questions q
              LEFT JOIN ml_user_items i
                ON i.user_id = q.user_id AND i.item_id = q.item_id
             WHERE q.user_id = $1
               AND q.status = 'UNANSWERED'
               AND q.tg_dispatched_at IS NULL
               AND COALESCE(i.status, 'active') = 'active'
             ORDER BY q.date_created DESC
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
            item_id = row["item_id"] or item.get("id")
            item_title = item.get("title") or ""
            item_thumbnail = item.get("thumbnail") or item.get("secure_thumbnail")
            item_permalink = item.get("permalink")
            buyer_nick = row["from_nickname"]

            # Pull product context (ML attrs + description + custom docs + sku)
            ctx_block = ""
            sku: Optional[str] = None
            context_used = False
            picture_urls: list[str] = []
            if item_id:
                try:
                    ctx = await _fetch_product_context(http, pool, user_id, item_id)
                    if ctx.get("title") and not item_title:
                        item_title = ctx["title"]
                    if ctx.get("permalink") and not item_permalink:
                        item_permalink = ctx["permalink"]
                    sku = ctx.get("sku")
                    block = _build_context_block(ctx)
                    # require at least attributes OR description OR docs to count as "context used"
                    if ctx.get("attributes") or ctx.get("description") or ctx.get("customDocs"):
                        ctx_block = block
                        context_used = True
                    # Extract up to 6 picture URLs for vision (secure_url first).
                    for p in (ctx.get("pictures") or [])[:6]:
                        if isinstance(p, dict):
                            url = p.get("secure_url") or p.get("url")
                            if url:
                                picture_urls.append(str(url))
                except Exception as err:  # noqa: BLE001
                    log.warning("context fetch failed for q=%s item=%s: %s", qid, item_id, err)

            # Pre-generated photo descriptions — RAG block. Helps Claude
            # find counts/measurements faster and reduces vision-miss rate.
            photo_descs_block: Optional[str] = None
            if item_id:
                try:
                    from . import ml_photo_descriptions as photo_svc
                    descs = await photo_svc.get_descriptions_for_item(
                        pool, user_id, item_id,
                    )
                    if descs:
                        photo_descs_block = photo_svc.descriptions_to_prompt_block(descs)
                except Exception as err:  # noqa: BLE001
                    log.debug("photo desc fetch failed for q=%s item=%s: %s", qid, item_id, err)

            suggestion = await _ai_suggest(
                http, text, item_title, ctx_block,
                picture_urls=picture_urls,
                photo_descriptions_block=photo_descs_block,
                pool=pool, user_id=user_id,
            ) or "(falha ao gerar sugestão; responda manualmente)"

            msg_id = await _tg_send_question(
                http, bot_token, chat_id, qid,
                item_title, item_thumbnail, item_permalink, buyer_nick,
                text, suggestion,
                sku=sku, item_id=item_id, context_used=context_used,
            )

            # Only mark dispatched if TG accepted the message — otherwise
            # the next tick should retry. We still persist the suggestion so
            # we don't burn another OpenRouter call if it was generated.
            if msg_id:
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
                sent += 1
            else:
                # Persist the suggestion so retry doesn't waste another LLM call.
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE ml_user_questions
                           SET tg_suggestion = COALESCE(tg_suggestion, $1)
                         WHERE user_id = $2 AND question_id = $3
                        """,
                        suggestion, user_id, qid,
                    )
            await asyncio.sleep(TG_THROTTLE)

    reminded = await _send_reminders_for_user(pool, user_id, bot_token, chat_id)

    return {"sent": sent, "skipped": 0, "reminded": reminded}


async def _send_reminders_for_user(
    pool: asyncpg.Pool,
    user_id: int,
    bot_token: str,
    chat_id: str,
) -> int:
    """Re-ping questions still UNANSWERED past the reminder threshold.

    Cadence:
      - first reminder: REMINDER_FIRST_HOURS after initial dispatch
      - subsequent:     REMINDER_INTERVAL_HOURS after the previous reminder
      - capped at REMINDER_MAX_COUNT total reminders per question
    """
    if REMINDER_MAX_COUNT <= 0:
        return 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT q.question_id, q.item_id, q.text, q.raw, q.from_nickname,
                   q.tg_message_id, q.tg_suggestion, q.tg_dispatched_at,
                   q.tg_reminder_count
              FROM ml_user_questions q
              LEFT JOIN ml_user_items i
                ON i.user_id = q.user_id AND i.item_id = q.item_id
             WHERE q.user_id = $1
               AND q.status = 'UNANSWERED'
               AND q.tg_dispatched_at IS NOT NULL
               AND q.tg_reminder_count < $2
               AND COALESCE(i.status, 'active') = 'active'
               AND (
                 (q.tg_last_reminder_at IS NULL
                   AND q.tg_dispatched_at + ($3 * INTERVAL '1 hour') < NOW())
                 OR (q.tg_last_reminder_at IS NOT NULL
                   AND q.tg_last_reminder_at + ($4 * INTERVAL '1 hour') < NOW())
               )
             ORDER BY q.tg_dispatched_at ASC
             LIMIT $5
            """,
            user_id, REMINDER_MAX_COUNT,
            REMINDER_FIRST_HOURS, REMINDER_INTERVAL_HOURS,
            TG_BATCH_CAP,
        )

    if not rows:
        return 0

    reminded = 0
    async with httpx.AsyncClient() as http:
        for row in rows:
            qid = int(row["question_id"])
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
            item_permalink = item.get("permalink")

            dispatched_at = row["tg_dispatched_at"]
            hours_pending = max(
                1,
                int((datetime.now(timezone.utc) - dispatched_at).total_seconds() // 3600),
            )

            new_msg_id = await _tg_send_reminder(
                http, bot_token, chat_id, qid,
                item_title, item_permalink, row["from_nickname"],
                row["text"] or "", row["tg_suggestion"],
                hours_pending, row["tg_message_id"],
            )

            if new_msg_id:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE ml_user_questions
                           SET tg_reminder_count = tg_reminder_count + 1,
                               tg_last_reminder_at = NOW()
                         WHERE user_id = $1 AND question_id = $2
                        """,
                        user_id, qid,
                    )
                reminded += 1
            await asyncio.sleep(TG_THROTTLE)

    return reminded


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

    # Honour onboarding alert_prefs.questions: users who explicitly disabled
    # the new-questions push are skipped here (they can still answer from UI).
    from v2.services import onboarding as _onboarding_svc
    await _onboarding_svc.ensure_schema(pool)
    prefs_map = await _onboarding_svc.get_alert_prefs_for_users(pool, user_ids)

    totals = {"users": 0, "sent": 0, "reminded": 0, "skipped_pref": 0}
    for uid in user_ids:
        user_prefs = prefs_map.get(uid)
        if user_prefs is not None and user_prefs.get("questions") is False:
            totals["skipped_pref"] += 1
            continue
        try:
            res = await _dispatch_for_user(pool, uid)
            totals["users"] += 1
            totals["sent"] += res.get("sent", 0)
            totals["reminded"] += res.get("reminded", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("questions dispatch user %s failed: %s", uid, err)
        await asyncio.sleep(0.5)
    return totals
