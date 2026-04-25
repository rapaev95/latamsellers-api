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
from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"
TG_BATCH_CAP = 10  # max questions per user per tick (avoid TG rate limit)
TG_THROTTLE = 1.1  # 1 sec between TG sends per chat

# MarkdownV2 escape per Telegram spec
_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
# Inside inline code (`...`) only ` and \ need escaping (per TG MarkdownV2 spec)
_MD_CODE_ESCAPE = str.maketrans({"`": "\\`", "\\": "\\\\"})


def _esc(text: str) -> str:
    return (text or "").translate(_MD_ESCAPE)


def _esc_code(text: str) -> str:
    """Escape for use INSIDE `...` inline code blocks."""
    return (text or "").translate(_MD_CODE_ESCAPE)


SUGGEST_SYSTEM_PROMPT = """Você é um vendedor profissional do Mercado Livre Brasil.

REGRAS CRÍTICAS:
1. Resposta SEMPRE em português brasileiro, tom amigável e profissional
2. 1-3 frases (máx 350 caracteres)
3. Use APENAS as informações fornecidas em "CONTEXTO DO PRODUTO"
4. Se a resposta NÃO está no contexto fornecido — diga claramente que vai verificar e responde em breve
5. NUNCA invente specs (dimensões, peso, materiais, compatibilidades)
6. Se há SKU no contexto, NÃO mencione SKU para o cliente (é interno do vendedor)
7. Termine de forma cordial
8. Máx 1 emoji
9. SEM links externos

Output APENAS o texto da resposta — sem aspas, sem prefácio."""


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
    """Aggregate ML core + description + custom docs + SKU. Returns dict
    suitable for prompt-context. Best-effort: missing pieces are simply absent."""
    mlb = _normalize_item_id(item_id)
    out: dict[str, Any] = {"itemId": mlb, "title": "", "sku": None, "permalink": None,
                          "attributes": [], "description": "", "customDocs": []}
    if not mlb:
        return out

    # Get a valid ML access token (auto-refresh)
    try:
        access_token = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except Exception as err:  # noqa: BLE001
        log.warning("oauth fetch failed for user %s: %s", user_id, err)
        access_token = None

    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

    # Parallel: ML item, ML description, DB docs, DB sku
    async def _ml_item():
        if not access_token:
            return None
        try:
            attrs_q = "id,title,condition,price,currency_id,permalink,available_quantity,attributes,shipping,warranty"
            r = await http.get(
                f"https://api.mercadolibre.com/items/{mlb}?attributes={attrs_q}",
                headers=headers, timeout=10.0,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        return None

    async def _ml_desc():
        if not access_token:
            return ""
        try:
            r = await http.get(
                f"https://api.mercadolibre.com/items/{mlb}/description",
                headers=headers, timeout=10.0,
            )
            if r.status_code == 200:
                d = r.json() or {}
                return (d.get("plain_text") or d.get("text") or "")[:3000]
        except Exception:  # noqa: BLE001
            pass
        return ""

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

    item, desc, docs, sku = await asyncio.gather(
        _ml_item(), _ml_desc(), _db_docs(), _db_sku(), return_exceptions=False,
    )

    if item:
        out["title"] = item.get("title") or ""
        out["permalink"] = item.get("permalink")
        out["condition"] = item.get("condition")
        out["price"] = item.get("price")
        out["currency"] = item.get("currency_id") or "BRL"
        out["availableQuantity"] = item.get("available_quantity")
        out["warranty"] = item.get("warranty")
        sh = item.get("shipping") or {}
        out["shippingFree"] = sh.get("free_shipping")
        attrs = []
        for a in (item.get("attributes") or [])[:30]:
            name = a.get("name") or a.get("id") or ""
            val = a.get("value_name")
            if not val and a.get("values"):
                vv = a["values"][0] if a["values"] else None
                val = (vv or {}).get("name")
            if name and val:
                attrs.append({"name": str(name), "value": str(val)})
        out["attributes"] = attrs

    out["description"] = desc or ""
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
  ADD COLUMN IF NOT EXISTS tg_suggestion TEXT;
CREATE INDEX IF NOT EXISTS idx_ml_questions_tg_pending
  ON ml_user_questions(user_id) WHERE status = 'UNANSWERED' AND tg_dispatched_at IS NULL;
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
) -> Optional[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — skipping AI suggestion")
        return None
    if context_block:
        user_msg = (
            "CONTEXTO DO PRODUTO (use SOMENTE estas informações para responder):\n"
            f"{context_block}\n\n"
            "---\n\n"
            f'Pergunta do comprador: "{question_text}"\n\n'
            "Escreva a resposta agora, baseada APENAS no contexto acima."
        )
    else:
        user_msg = (
            f'Anúncio: "{item_title or "sem título"}"\n'
            f'Pergunta do comprador: "{question_text}"\n\n'
            "⚠️ Sem informações detalhadas do produto. Se a pergunta exige specs, peça desculpas e diga que vai verificar."
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
    sku: Optional[str] = None,
    item_id: Optional[str] = None,
    context_used: bool = False,
) -> Optional[str]:
    """Send a question card with action buttons. Returns tg message_id on success."""
    body_lines: list[str] = []
    body_lines.append(f"❓ *Nova pergunta*")
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
            item_id = row["item_id"] or item.get("id")
            item_title = item.get("title") or ""
            item_thumbnail = item.get("thumbnail") or item.get("secure_thumbnail")
            item_permalink = item.get("permalink")
            buyer_nick = row["from_nickname"]

            # Pull product context (ML attrs + description + custom docs + sku)
            ctx_block = ""
            sku: Optional[str] = None
            context_used = False
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
                except Exception as err:  # noqa: BLE001
                    log.warning("context fetch failed for q=%s item=%s: %s", qid, item_id, err)

            suggestion = await _ai_suggest(http, text, item_title, ctx_block) or "(falha ao gerar sugestão; responda manualmente)"

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
