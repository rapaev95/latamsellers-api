"""Background verification that ML actually kept the price we PUT.

Problem this solves: when the seller hits "+10%" on a sale-card / promo
card, we PUT /items/{id} {price} and ML returns 200. But Mercado Livre's
pricing-automation (Precificação Dinâmica) can silently rewrite the
price back to a competitor-driven value within minutes — visible only
when the next sale arrives at the OLD price. User feedback 2026-05-16:
clicked +10%, got success toast, an hour later a new order arrived at
the original price.

What this module does:
  1. Every 5 min, scan escalar_audit for `price_raised` / `price_lowered`
     events between 5 and 60 min old whose `verified_at` is NULL.
  2. For each, GET /items/{target_id} and compare `price` to
     `metadata.new_price`.
  3. If diverged by > 1% — send a TG follow-up to the seller:
        ⚠️ ML cancelou seu ajuste de preço
        MLB6143605452 · Preço definido: R$ 21,34 · Atual: R$ 19,40
        Provavelmente Precificação Dinâmica voltou a operar.
        Tente: ↩️ Habilitar Dyn / Reaplique manualmente
  4. Mark the audit row verified (with status flag in metadata) so we
     don't re-alert on the same event.

Out of scope: the auto-disable+retry path already runs at PUT time and
covers the synchronous case. This module is the asynchronous safety net.
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

from v2.services import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)
TG_API_BASE = "https://api.telegram.org"

# Tunables.
SCAN_WINDOW_MIN_AGE_SEC = 5 * 60     # don't check too early — ML can lag on Read-after-Write
SCAN_WINDOW_MAX_AGE_SEC = 60 * 60    # don't keep re-checking after 1h
DIVERGENCE_THRESHOLD = 0.01          # 1% — accommodates rounding in centavos
BATCH_LIMIT = 30                     # per tick


ALTER_SQL = """
ALTER TABLE escalar_audit
  ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_audit_price_unverified
  ON escalar_audit(occurred_at)
  WHERE action IN ('price_raised', 'price_lowered') AND verified_at IS NULL;
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(ALTER_SQL)


async def _fetch_pending_events(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, user_id, target_id, metadata,
                   occurred_at AT TIME ZONE 'UTC' AS occurred_at_utc
              FROM escalar_audit
             WHERE action IN ('price_raised', 'price_lowered')
               AND verified_at IS NULL
               AND occurred_at < NOW() - INTERVAL '{SCAN_WINDOW_MIN_AGE_SEC} seconds'
               AND occurred_at > NOW() - INTERVAL '{SCAN_WINDOW_MAX_AGE_SEC} seconds'
             ORDER BY occurred_at ASC
             LIMIT {BATCH_LIMIT}
            """
        )
    return [dict(r) for r in rows]


async def _mark_verified(
    pool: asyncpg.Pool, audit_id: int, status: str, extra: dict[str, Any],
) -> None:
    """Stamp verified_at + merge verification result into metadata.

    `status` is one of:
      - kept      — current price within tolerance of our target
      - reverted  — current price diverged
      - missing   — ML returned no current price (paused/closed item)
      - error     — ML API failed
    """
    blob = json.dumps({"verify_status": status, **extra}, default=str)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE escalar_audit
               SET verified_at = NOW(),
                   metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
             WHERE id = $1
            """,
            audit_id, blob,
        )


async def _notify_revert(
    http: httpx.AsyncClient,
    pool: asyncpg.Pool,
    user_id: int,
    mlb: str,
    expected: float,
    actual: float,
    audit_id: int,
) -> None:
    """Send a "your price was reverted" follow-up to the seller's TG chat."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        log.warning("[price-revert] TELEGRAM_BOT_TOKEN missing — cannot alert user=%s", user_id)
        return
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language
              FROM notification_settings WHERE user_id = $1
            """,
            user_id,
        )
    if not row or not row["telegram_chat_id"]:
        log.info("[price-revert] no TG chat for user=%s — skip alert", user_id)
        return

    chat_id = str(row["telegram_chat_id"])
    lang = (row["language"] or "pt").lower()
    expected_fmt = f"R$ {expected:.2f}".replace(".", ",")
    actual_fmt = f"R$ {actual:.2f}".replace(".", ",")
    diff_pct = abs((actual - expected) / expected * 100) if expected else 0

    if lang == "ru":
        text = (
            "⚠️ ML откатил твоё изменение цены\n\n"
            f"Товар: {mlb}\n"
            f"Установлено: {expected_fmt}\n"
            f"Текущее: {actual_fmt} (-{diff_pct:.1f}%)\n\n"
            "Скорее всего ML повторно включила Precificação Dinâmica или "
            "автоматический ценовой механизм всё ещё работает. Попробуй "
            "нажать «↩️ Habilitar Dyn Pricing» в исходном сообщении или "
            "переустановить цену вручную."
        )
    elif lang == "en":
        text = (
            "⚠️ ML reverted your price change\n\n"
            f"Item: {mlb}\n"
            f"You set: {expected_fmt}\n"
            f"Current: {actual_fmt} (-{diff_pct:.1f}%)\n\n"
            "ML's Precificação Dinâmica likely re-engaged. Try the "
            "↩️ Habilitar Dyn Pricing button on the original message or "
            "re-apply the price manually."
        )
    else:
        text = (
            "⚠️ ML cancelou seu ajuste de preço\n\n"
            f"Anúncio: {mlb}\n"
            f"Você definiu: {expected_fmt}\n"
            f"Atual: {actual_fmt} (-{diff_pct:.1f}%)\n\n"
            "Precificação Dinâmica provavelmente voltou a operar. "
            "Tente o botão ↩️ Habilitar Dyn Pricing na mensagem original "
            "ou reaplique o preço manualmente."
        )

    try:
        r = await http.post(
            f"{TG_API_BASE}/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            log.warning(
                "[price-revert] TG sendMessage failed audit_id=%s status=%s body=%s",
                audit_id, r.status_code, r.text[:200],
            )
    except Exception as err:  # noqa: BLE001
        log.warning("[price-revert] TG send exception audit_id=%s: %s", audit_id, err)


async def _verify_one(
    http: httpx.AsyncClient, pool: asyncpg.Pool, event: dict[str, Any],
) -> str:
    """Verify a single audit event. Returns the verification status string."""
    audit_id = event["id"]
    user_id = event["user_id"]
    mlb = (event.get("target_id") or "").strip().upper()
    meta = event.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    expected = meta.get("new_price")
    try:
        expected_f = float(expected) if expected is not None else None
    except (TypeError, ValueError):
        expected_f = None

    if not mlb or expected_f is None or expected_f <= 0:
        await _mark_verified(
            pool, audit_id, "skip",
            {"reason": "missing_mlb_or_new_price"},
        )
        return "skip"

    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except Exception as err:  # noqa: BLE001
        await _mark_verified(
            pool, audit_id, "error",
            {"reason": "oauth_failed", "err": str(err)[:200]},
        )
        return "error"

    try:
        r = await http.get(
            f"https://api.mercadolibre.com/items/{mlb}?attributes=id,price,status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("[price-revert] GET item failed audit_id=%s mlb=%s: %s", audit_id, mlb, err)
        # don't mark — retry on next tick
        return "transient"

    if r.status_code == 404:
        await _mark_verified(
            pool, audit_id, "missing",
            {"reason": "item_404"},
        )
        return "missing"
    if r.status_code >= 400:
        log.warning(
            "[price-revert] GET item rejected audit_id=%s mlb=%s status=%s",
            audit_id, mlb, r.status_code,
        )
        return "transient"

    try:
        actual = float((r.json() or {}).get("price") or 0.0)
    except (TypeError, ValueError):
        actual = 0.0
    if actual <= 0:
        await _mark_verified(
            pool, audit_id, "missing",
            {"reason": "no_current_price"},
        )
        return "missing"

    diff_ratio = abs(actual - expected_f) / expected_f
    if diff_ratio <= DIVERGENCE_THRESHOLD:
        await _mark_verified(
            pool, audit_id, "kept",
            {"expected": expected_f, "actual": actual},
        )
        return "kept"

    # Reverted. Send TG alert + mark.
    await _notify_revert(http, pool, user_id, mlb, expected_f, actual, audit_id)
    await _mark_verified(
        pool, audit_id, "reverted",
        {"expected": expected_f, "actual": actual, "diff_ratio": round(diff_ratio, 4)},
    )
    return "reverted"


async def run_check_tick(pool: asyncpg.Pool) -> dict[str, int]:
    """Single scheduler-tick invocation. Logs aggregate counts."""
    if pool is None:
        return {"checked": 0}
    await ensure_schema(pool)
    events = await _fetch_pending_events(pool)
    if not events:
        return {"checked": 0}
    stats = {"checked": 0, "kept": 0, "reverted": 0, "missing": 0, "skip": 0,
             "error": 0, "transient": 0}
    async with httpx.AsyncClient() as http:
        for ev in events:
            status = await _verify_one(http, pool, ev)
            stats["checked"] += 1
            stats[status] = stats.get(status, 0) + 1
            await asyncio.sleep(0.05)  # gentle rate-limit
    log.info("[price-revert] tick %s", stats)
    return stats
