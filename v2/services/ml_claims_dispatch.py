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
TG_BATCH_CAP = 10  # max claims per user per tick
TG_THROTTLE = 1.1  # 1 sec between TG sends per chat

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
  ADD COLUMN IF NOT EXISTS tg_message_id TEXT;
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


def _build_claim_card(claim: dict[str, Any]) -> str:
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
    lines.append(f"🆔 `{_esc_code(cid)}` · {_esc(reason)} · `{_esc_code(stage)}`")
    if age:
        lines.append(f"🕒 {_esc(age)}")
    lines.append(f"📦 *Pedido:* `{_esc_code(order_id)}`")
    if ret_summary:
        lines.append(f"↩️ *Devolução:* `{_esc_code(ret_summary)}`")
    if needs_review:
        lines.append("")
        lines.append("_O comprador devolveu o produto\\. Inspecione e decida\\: reembolsar, trocar ou contestar\\._")
    else:
        lines.append("")
        lines.append("_Escolha uma solução para o comprador antes do prazo expirar\\._")
    return "\n".join(lines)


def _build_keyboard(claim_id: int, app_base_url: str) -> dict[str, Any]:
    """Inline keyboard. For now: deep-links to our app + ML. Action buttons
    via callback_query can be added later — they need a webhook handler in
    the Next.js TG webhook route to call /api/escalar/claims/{id}/resolution.
    """
    app_link = f"{app_base_url.rstrip('/')}/escalar/claims"
    ml_link = f"https://myaccount.mercadolivre.com.br/post-purchase/cases/{claim_id}"
    return {
        "inline_keyboard": [
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
) -> Optional[str]:
    """Send one claim card. Returns TG message_id on success."""
    cid_raw = claim.get("id")
    try:
        cid = int(cid_raw)
    except (TypeError, ValueError):
        return None

    text = _build_claim_card(claim)
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

    # Pull pending actionable claims
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT claim_id, enriched
              FROM ml_user_claims
             WHERE user_id = $1
               AND status = 'opened'
               AND tg_dispatched_at IS NULL
             ORDER BY date_created DESC NULLS LAST
             LIMIT $2
            """,
            user_id, TG_BATCH_CAP * 4,  # over-fetch — many will fail needs_action filter
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
                # Not actionable right now — skip this tick. We DO mark it
                # dispatched so we don't re-evaluate it forever; if its state
                # changes later (e.g. return arrives → delivered) we'll get
                # a fresh row when refresh re-upserts it... but the upsert
                # doesn't reset tg_dispatched_at. So leave the flag NULL for
                # now and rely on per-tick filtering.
                skipped += 1
                continue

            cid = int(row["claim_id"])
            msg_id = await _send_claim(http, bot_token, chat_id, enriched, app_base_url)
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
