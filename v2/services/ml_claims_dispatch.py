"""Send actionable claims to seller's Telegram with action buttons.

Cron-driven (every CLAIMS_TG_INTERVAL_MIN, default 5 min). Mirrors
ml_questions_dispatch — independent of the legacy ml_notices path:

1. For each user with telegram_chat_id set:
   a. Refresh ml_user_claims cache (so latest opened+enriched are present
      and stale-opened rows are reconciled with ML).
   b. Find rows where status='opened', _compute_needs_action=True,
      tg_dispatched_at IS NULL.
   c. Build a card with claim_id, age, reason, order id, return state.
   d. Send TG with inline keyboard: Atender no app / Abrir no ML.
   e. On 200 → mark tg_dispatched_at + tg_message_id.

Schema migration is applied lazily via ensure_schema() (idempotent ALTER).

Why a separate dispatch (not just notices): the legacy ml_notices path
runs translations through the LLM and depends on the ML webhook firing
correctly + /communications/notices including the claim. In production
the user had 12 disputes with no TG message because at least one of
those gates failed. This path reads ml_user_claims directly — same
source of truth as the /escalar/claims UI — so as long as the cache is
fresh, TG gets the message.
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

from . import ml_user_claims as claims_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"
TG_BATCH_CAP = 10  # max claims per user per tick
TG_THROTTLE = 1.1  # 1 sec between TG sends per chat


SUMMARY_SYSTEM_PROMPT_RU = """Ты — ассистент продавца Mercado Livre. Твоя задача — кратко и тезисно объяснить продавцу суть рекламации покупателя.

ПРАВИЛА:
1. Перевод на русский, но НЕ дословный — выдели только важное.
2. Формат: 2-4 коротких тезиса, каждый с новой строки, начинается с «• ».
3. Что включать:
   • Что не так с товаром (брак, не соответствует описанию, не работает и т.д.)
   • Что покупатель уже сделал/сообщил (прислал фото, попробовал инструкции и т.д.)
   • Какое решение предлагает ML или просит покупатель (если упомянуто)
4. НЕ цитируй покупателя дословно. НЕ пиши «покупатель пишет что...». Сразу к сути.
5. Без преамбулы, без заголовка, без «Тезисно:» — только сами тезисы.
6. Максимум 350 символов всего."""

SUMMARY_SYSTEM_PROMPT_EN = """You are a Mercado Livre seller's assistant. Summarize the buyer's claim message into bullet points so the seller can triage instantly.

RULES:
1. Translate to English, but NOT literally — keep only what matters.
2. Format: 2-4 short bullets, each on its own line, starting with "• ".
3. Include:
   • What is wrong with the product (defect, mismatch, doesn't work, etc.)
   • What the buyer has already done/reported (sent photos, tried instructions, etc.)
   • What resolution the ML mediator or buyer is suggesting (if mentioned)
4. Don't quote the buyer verbatim. No "The buyer says...". Get to the point.
5. No preamble, no headers, no "Summary:" — just the bullets.
6. Max 350 characters total."""

# MarkdownV2 escape per Telegram spec
_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
_MD_CODE_ESCAPE = str.maketrans({"`": "\\`", "\\": "\\\\"})
_MD2_UNESCAPE_RE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


def _esc(text: Any) -> str:
    return (str(text or "")).translate(_MD_ESCAPE)


def _esc_code(text: Any) -> str:
    return (str(text or "")).translate(_MD_CODE_ESCAPE)


def _strip_md2_escapes(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"(?<!\\)[*_~`]", "", text)
    return _MD2_UNESCAPE_RE.sub(r"\1", out)


# ── Schema ──────────────────────────────────────────────────────────

ALTER_SQL = """
ALTER TABLE ml_user_claims
  ADD COLUMN IF NOT EXISTS tg_dispatched_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS tg_message_id TEXT,
  ADD COLUMN IF NOT EXISTS tg_summary TEXT,
  ADD COLUMN IF NOT EXISTS tg_summary_lang TEXT;
CREATE INDEX IF NOT EXISTS idx_ml_user_claims_tg_pending
  ON ml_user_claims(user_id)
  WHERE status = 'opened' AND tg_dispatched_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(ALTER_SQL)


# ── TG send (with MD2 → plain fallback) ─────────────────────────────

def _format_age(date_iso: Optional[str]) -> str:
    """'há 4 dias' / 'há 12 horas' from ISO date."""
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


def _summarize_return(claim: dict[str, Any]) -> Optional[str]:
    """One-line summary of the return state, if any."""
    returns = claim.get("returns")
    if not returns or not isinstance(returns, list) or not returns:
        return None
    head = returns[0] if isinstance(returns[0], dict) else {}
    status = head.get("status") or "—"
    subtype = head.get("subtype") or ""
    parts = [str(status)]
    if subtype:
        parts.append(str(subtype))
    refund = head.get("refund_at")
    if refund:
        parts.append(f"refund→{refund}")
    return " · ".join(parts)


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_buyer_complaint(claim: dict[str, Any]) -> Optional[str]:
    """Pull the most recent buyer-relevant message from the claim thread.

    ML's mediator/buyer messages live in claim.messages (added by enrich
    via /claims/{id}/messages). Each message has shape roughly:
      { id, sender_role: 'complainant'|'respondent'|'mediator',
        message: 'text', date_created: ISO, ... }
    We prefer the latest mediator/complainant message because that's where
    "O comprador relatou que..." text shows up. Strip basic HTML tags
    that ML occasionally embeds (e.g. <strong>...</strong>).
    """
    messages = claim.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    # Look for messages from mediator/complainant; prefer mediator (richer
    # summary including "O comprador relatou ...").
    candidates: list[tuple[int, str]] = []  # (priority, text)
    for m in messages:
        if not isinstance(m, dict):
            continue
        sender = m.get("sender_role") or ""
        if isinstance(m.get("sender"), dict):
            sender = m["sender"].get("role") or sender
        if sender not in ("mediator", "complainant", "buyer"):
            continue
        text = (m.get("message") or m.get("text") or "").strip()
        if not text:
            continue
        text = _HTML_TAG_RE.sub("", text)
        # Replace the bullet-list markdown chars ML sometimes uses
        text = text.replace("\n- ", "\n• ").replace("\n  - ", "\n  • ")
        priority = 1 if sender == "mediator" else 2
        candidates.append((priority, text))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])  # mediator first
    return candidates[0][1]


def _extract_photo_urls(claim: dict[str, Any]) -> list[dict[str, str]]:
    """Pull buyer-uploaded photo URLs from claim messages + returns.

    Two sources we've seen in ML's payload:
      - claim.messages[*].attachments[*]   (when buyer attaches to chat msg)
      - claim.returns[*].evidences[*]      (when buyer uploads on return flow)

    Returns list of {url, name?, source} dicts (deduped). Caller sends each
    via TG sendPhoto. Empty list = nothing to send.
    """
    seen_urls: set[str] = set()
    out: list[dict[str, str]] = []

    # 1. Message-level attachments
    messages = claim.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            attachments = m.get("attachments") or []
            if not isinstance(attachments, list):
                continue
            for a in attachments:
                if not isinstance(a, dict):
                    continue
                url = a.get("url") or a.get("source") or a.get("file_url")
                if not url or not isinstance(url, str):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                out.append({
                    "url": url,
                    "name": str(a.get("name") or a.get("filename") or ""),
                    "source": "message",
                })

    # 2. Return-level evidences
    returns = claim.get("returns")
    if isinstance(returns, list):
        for ret in returns:
            if not isinstance(ret, dict):
                continue
            evidences = ret.get("evidences") or ret.get("attachments") or []
            if not isinstance(evidences, list):
                continue
            for e in evidences:
                if not isinstance(e, dict):
                    continue
                url = e.get("url") or e.get("source") or e.get("file_url")
                if not url or not isinstance(url, str):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                out.append({
                    "url": url,
                    "name": str(e.get("name") or e.get("filename") or ""),
                    "source": "return",
                })

    return out


async def _send_photos(
    http: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    claim_id: int,
    photos: list[dict[str, str]],
    access_token: Optional[str] = None,
) -> int:
    """Send buyer-uploaded photos to TG via sendPhoto. Cap at 5 to avoid
    spam. Tries URL-direct first; if TG fails (auth-protected URL), we
    fetch via ML token and re-upload as multipart.
    Returns count actually sent.
    """
    sent = 0
    for i, p in enumerate(photos[:5]):
        url = p["url"]
        caption = f"📎 Evidência #{i + 1} · claim `{claim_id}`"
        # Attempt 1: TG fetches the URL directly
        try:
            r = await http.post(
                f"{TG_API_BASE}/bot{bot_token}/sendPhoto",
                json={
                    "chat_id": chat_id,
                    "photo": url,
                    "caption": caption,
                    "parse_mode": "MarkdownV2",
                },
                timeout=15.0,
            )
            if r.status_code == 200:
                sent += 1
                continue
            log.info("TG sendPhoto direct claim=%s photo=%s status=%s body=%s",
                     claim_id, i, r.status_code, r.text[:200])
        except Exception as err:  # noqa: BLE001
            log.warning("TG sendPhoto direct exception: %s", err)

        # Attempt 2: download via ML, re-upload to TG as multipart
        if access_token:
            try:
                pr = await http.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=20.0,
                )
                if pr.status_code != 200:
                    continue
                files = {"photo": (p.get("name") or f"evidence_{i}.jpg", pr.content)}
                data = {"chat_id": chat_id, "caption": caption.replace("`", "")}
                tr = await http.post(
                    f"{TG_API_BASE}/bot{bot_token}/sendPhoto",
                    data=data, files=files, timeout=30.0,
                )
                if tr.status_code == 200:
                    sent += 1
            except Exception as err:  # noqa: BLE001
                log.warning("TG sendPhoto multipart exception: %s", err)
        await asyncio.sleep(0.5)
    return sent


async def _summarize_complaint(
    http: httpx.AsyncClient,
    complaint: str,
    target_lang: str = "ru",
) -> Optional[str]:
    """Use OpenRouter (same model as ml_questions_dispatch) to produce a 2-4
    bullet summary of the buyer's claim message in the seller's language.

    Returns None if the API key is missing, the call fails, or the response
    is empty. Caller should fall back to raw complaint text.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — claim summary skipped")
        return None
    if not complaint or not complaint.strip():
        return None

    target = (target_lang or "ru").lower()
    if target == "en":
        system_prompt = SUMMARY_SYSTEM_PROMPT_EN
    else:
        system_prompt = SUMMARY_SYSTEM_PROMPT_RU

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
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": complaint[:3000]},
                ],
            },
            timeout=15.0,
        )
        if r.status_code != 200:
            log.warning("OpenRouter summary %s: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        text = content.strip() if isinstance(content, str) else None
        if text and not text.startswith("•"):
            # Some models prepend a header — strip empty lines, keep only
            # the bullet block.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            text = "\n".join(ln for ln in lines if ln.startswith("•") or ln.startswith("-"))
            text = text.replace("- ", "• ") or None
        return text
    except Exception as err:  # noqa: BLE001
        log.exception("claim summary failed: %s", err)
        return None


def _build_claim_card(
    claim: dict[str, Any],
    summary: Optional[str] = None,
    summary_lang: str = "ru",
) -> str:
    """MarkdownV2-formatted TG card for one actionable claim."""
    cid = claim.get("id") or "?"
    reason = claim.get("reason_id") or "?"
    stage = claim.get("stage") or "?"
    order_id = claim.get("resource_id") or "—"
    age = _format_age(claim.get("date_created"))
    ret_summary = _summarize_return(claim)

    # Check the structured field directly — _summarize_return may include
    # "delivered" inside refund_at or other sub-tokens which would falsely
    # trigger the "needs review" branch.
    returns = claim.get("returns")
    head_status = ""
    if isinstance(returns, list) and returns and isinstance(returns[0], dict):
        head_status = (returns[0].get("status") or "").lower()
    needs_review = head_status == "delivered"
    if needs_review:
        title = "📦 *Devolução chegou — inspecionar*"
    else:
        title = "⚠️ *Reclamação aberta — atender agora*"

    lines: list[str] = [title]
    lines.append("")
    # Product title + variation hint — what the buyer is complaining ABOUT.
    # Without it the seller sees only an order_id and has to open ML to
    # know what product is involved.
    order_item = claim.get("order_item") or {}
    product_title = order_item.get("title") if isinstance(order_item, dict) else None
    qty = order_item.get("quantity") if isinstance(order_item, dict) else None
    if product_title:
        title_line = f"🛍 *{_esc(product_title[:120])}*"
        if qty and isinstance(qty, int) and qty > 1:
            title_line += f" · {qty}×"
        lines.append(title_line)
    lines.append(f"🆔 `{_esc_code(cid)}` · {_esc(reason)} · `{_esc_code(stage)}`")
    if age:
        lines.append(f"🕒 {_esc(age)}")
    lines.append(f"📦 *Pedido:* `{_esc_code(order_id)}`")

    # Buyer nickname when ML's order endpoint surfaces it.
    buyer_block = claim.get("order_buyer") or {}
    nickname = buyer_block.get("nickname") if isinstance(buyer_block, dict) else None
    if nickname:
        lines.append(f"👤 @{_esc(nickname)}")

    if ret_summary:
        lines.append(f"↩️ *Devolução:* `{_esc_code(ret_summary)}`")

    # Buyer's complaint / mediator summary — the "why" of the claim.
    # Prefer the AI-generated bullet summary in the seller's language; fall
    # back to the raw mediator/buyer message if the summary failed (no API
    # key, OpenRouter down, etc.).
    if summary and summary.strip():
        header = "💬 *Суть жалобы:*" if summary_lang == "ru" else (
            "💬 *Buyer's claim summary:*" if summary_lang == "en"
            else "💬 *Resumo da reclamação:*"
        )
        lines.append("")
        lines.append(header)
        lines.append(_esc(summary.strip()))
    else:
        complaint = _extract_buyer_complaint(claim)
        if complaint:
            clip = complaint[:600].rstrip()
            if len(complaint) > 600:
                clip += "…"
            lines.append("")
            lines.append("💬 *Motivo do comprador:*")
            lines.append(_esc(clip))

    lines.append("")
    if needs_review:
        lines.append("_Devolução em mãos\\. Decida\\: aceitar reembolso ou contestar\\._")
    else:
        lines.append("_Escolha uma solução abaixo antes do prazo expirar\\._")
    return "\n".join(lines)


def _build_keyboard(claim_id: int, app_base_url: str) -> dict[str, Any]:
    """Inline keyboard with one-click resolution actions.

    callback_data prefixes (handled by the Next.js telegram-webhook route):
      cl_rf:{id}  → refund (full)
      cl_rt:{id}  → return_product (accept return; buyer ships back)
      cl_ex:{id}  → change_product (exchange / accept return for replacement)
      cl_pa:{id}  → partial-refund — opens the app (needs amount input)

    Both "Atender no app" and "Ver no ML" stay as URL fallbacks for the
    cases the seller wants to handle in a richer UI (orientation, evidences,
    custom percentage).
    """
    app_link = f"{app_base_url.rstrip('/')}/escalar/claims"
    ml_link = f"https://myaccount.mercadolivre.com.br/post-purchase/cases/{claim_id}"
    return {
        "inline_keyboard": [
            [
                {"text": "💵 Reembolsar", "callback_data": f"cl_rf:{claim_id}"},
                {"text": "↩️ Aceitar devolução", "callback_data": f"cl_rt:{claim_id}"},
            ],
            [
                {"text": "🔄 Trocar", "callback_data": f"cl_ex:{claim_id}"},
                {"text": "💸 Reembolso parcial", "callback_data": f"cl_pa:{claim_id}"},
            ],
            [
                {"text": "⚡ Atender no app", "url": app_link},
                {"text": "🔗 Ver no ML", "url": ml_link},
            ],
        ],
    }


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


async def _send_claim(
    http: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    claim: dict[str, Any],
    app_base_url: str,
    summary: Optional[str] = None,
    summary_lang: str = "ru",
) -> Optional[str]:
    """Send one claim card. Returns TG message_id on success."""
    cid_raw = claim.get("id")
    try:
        cid = int(cid_raw)
    except (TypeError, ValueError):
        return None

    text = _build_claim_card(claim, summary=summary, summary_lang=summary_lang)
    if len(text) > 4000:
        text = text[:3990] + "…"
    keyboard = _build_keyboard(cid, app_base_url)

    try:
        # First attempt: MarkdownV2
        status, body_preview, msg_id = await _tg_post(http, bot_token, {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "reply_markup": keyboard,
            "disable_web_page_preview": True,
        })
        if status == 200 and msg_id:
            return msg_id

        # MD2 parse failure → plain text fallback
        if status == 400 and "parse" in body_preview.lower():
            log.warning("TG MD2 parse failed for claim %s: %s", cid, body_preview)
            plain = _strip_md2_escapes(text)
            status2, body2, msg_id2 = await _tg_post(http, bot_token, {
                "chat_id": chat_id,
                "text": plain,
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            })
            if status2 == 200 and msg_id2:
                return msg_id2
            log.warning("TG plain fallback also failed for claim %s: %s %s", cid, status2, body2)

        log.warning("TG send_message claim=%s status=%s body=%s", cid, status, body_preview)
    except Exception as err:  # noqa: BLE001
        log.exception("TG send claim %s failed: %s", cid, err)
    return None


# ── Per-user dispatch ────────────────────────────────────────────────

async def _dispatch_for_user(pool: asyncpg.Pool, user_id: int, app_base_url: str) -> dict[str, int]:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"sent": 0, "skipped": 0}

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
            return {"sent": 0, "skipped": 0}
        chat_id = str(settings["telegram_chat_id"])

    # Refresh cache so we have the latest claim state + run reconciliation.
    # This keeps the TG path self-sufficient — no external job has to run
    # before us.
    try:
        await claims_svc.refresh_user_claims(pool, user_id)
    except Exception as err:  # noqa: BLE001
        log.warning("claims refresh user=%s failed: %s", user_id, err)

    # The seller's preferred language drives both the AI summary language
    # and the in-card section header.
    summary_lang = (settings["language"] or "pt").lower()

    # Pull pending actionable claims (over-fetch — many will fail
    # needs_action filter). Also pull cached summary so we don't pay for
    # OpenRouter on every retry.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT claim_id, enriched, tg_summary, tg_summary_lang
              FROM ml_user_claims
             WHERE user_id = $1
               AND status = 'opened'
               AND tg_dispatched_at IS NULL
             ORDER BY date_created DESC NULLS LAST
             LIMIT $2
            """,
            user_id, TG_BATCH_CAP * 4,
        )

    sent = 0
    skipped = 0
    async with httpx.AsyncClient() as http:
        for row in rows:
            if sent >= TG_BATCH_CAP:
                break
            enriched = row["enriched"]
            if isinstance(enriched, str):
                try:
                    enriched = json.loads(enriched)
                except Exception:  # noqa: BLE001
                    enriched = {}
            if not isinstance(enriched, dict):
                continue
            if not claims_svc._compute_needs_action(enriched):
                skipped += 1
                continue

            cid = int(row["claim_id"])

            # Generate (or reuse cached) bullet summary in the seller's
            # language. Skip for pt — sellers comfortable with pt-BR see the
            # original mediator/buyer text, no AI roundtrip needed.
            summary: Optional[str] = None
            if summary_lang in ("ru", "en"):
                cached = row["tg_summary"]
                cached_lang = row["tg_summary_lang"]
                if cached and cached_lang == summary_lang:
                    summary = cached
                else:
                    complaint = _extract_buyer_complaint(enriched)
                    if complaint:
                        summary = await _summarize_complaint(http, complaint, summary_lang)
                        if summary:
                            async with pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE ml_user_claims
                                       SET tg_summary = $1,
                                           tg_summary_lang = $2
                                     WHERE user_id = $3 AND claim_id = $4
                                    """,
                                    summary, summary_lang, user_id, cid,
                                )

            msg_id = await _send_claim(
                http, bot_token, chat_id, enriched, app_base_url,
                summary=summary, summary_lang=summary_lang,
            )
            if msg_id:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE ml_user_claims
                           SET tg_dispatched_at = NOW(),
                               tg_message_id = $1
                         WHERE user_id = $2 AND claim_id = $3
                        """,
                        msg_id, user_id, cid,
                    )
                sent += 1

                # After main card lands, send any buyer-uploaded photos so
                # the seller can inspect evidence inline. Empty list = no-op.
                photos = _extract_photo_urls(enriched)
                if photos:
                    try:
                        from . import ml_oauth as _oauth
                        photo_token, *_ = await _oauth.get_valid_access_token(pool, user_id)
                    except Exception:  # noqa: BLE001
                        photo_token = None
                    photo_count = await _send_photos(
                        http, bot_token, chat_id, cid, photos, access_token=photo_token,
                    )
                    if photo_count:
                        log.info("claims dispatch: sent %s photos for claim %s", photo_count, cid)
            await asyncio.sleep(TG_THROTTLE)

    return {"sent": sent, "skipped": skipped}


# ── Cron entry point (called from main.py) ────────────────────────────

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
    totals = {"users": 0, "sent": 0, "skipped": 0}
    for uid in user_ids:
        try:
            res = await _dispatch_for_user(pool, uid, app_base_url)
            totals["users"] += 1
            totals["sent"] += res["sent"]
            totals["skipped"] += res.get("skipped", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("claims dispatch user %s failed: %s", uid, err)
        await asyncio.sleep(0.5)
    return totals
