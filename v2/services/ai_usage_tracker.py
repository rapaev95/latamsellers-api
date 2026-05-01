"""OpenRouter usage tracking — per-call logging into ai_usage table.

User has hit OpenRouter `402: requires more credits` once already. Without
spend tracking we fly blind — can't tell if cost is from buyer-question
suggestions / news digests / photo descriptions / daily narratives.

Each AI call:
  1. logs usage row into ai_usage with tokens + computed cost_brl
  2. on completion, optional admin TG alert при daily threshold breach

Cost computation — model-pricing table compiled from OpenRouter
public rates (snapshot 2026-04). USD-to-BRL rate is env-tunable
(default 5.0) — coarse but good enough for budget visibility.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ai_usage (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER,
  service TEXT NOT NULL,
  model TEXT NOT NULL,
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  cost_usd NUMERIC DEFAULT 0,
  cost_brl NUMERIC DEFAULT 0,
  status_code INTEGER,
  occurred_at TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_ai_usage_user_day
  ON ai_usage(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_usage_service_day
  ON ai_usage(service, occurred_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# Per-1M-tokens USD rates — OpenRouter snapshot 2026-04.
# Update by hand if needed; env override via AI_RATES_OVERRIDE possible later.
MODEL_RATES_USD = {
    # Claude family
    "anthropic/claude-sonnet-4.5":  (3.00, 15.00),    # in / out per 1M
    "anthropic/claude-haiku-4.5":   (1.00,  5.00),
    "anthropic/claude-opus-4":      (15.00, 75.00),
    # OpenAI
    "openai/gpt-4o-mini":           (0.15,  0.60),
    "openai/gpt-4o":                (2.50, 10.00),
    "openai/gpt-4.1":               (2.00,  8.00),
}
DEFAULT_RATE_USD = (3.00, 15.00)  # conservative — assume Sonnet-class


def _usd_to_brl_rate() -> float:
    try:
        return float(os.environ.get("USD_BRL_RATE_FOR_AI_COST", "5.0"))
    except (TypeError, ValueError):
        return 5.0


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> tuple[float, float]:
    """Returns (cost_usd, cost_brl)."""
    rate_in, rate_out = MODEL_RATES_USD.get(model, DEFAULT_RATE_USD)
    cost_usd = (tokens_in / 1_000_000) * rate_in + (tokens_out / 1_000_000) * rate_out
    cost_brl = cost_usd * _usd_to_brl_rate()
    return round(cost_usd, 6), round(cost_brl, 4)


async def log_call(
    pool: asyncpg.Pool,
    *,
    user_id: Optional[int],
    service: str,
    model: str,
    response_data: Optional[dict[str, Any]] = None,
    status_code: int = 200,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Fire-and-forget: extract usage from OpenRouter response, log row.

    Never raises — caller must keep happy path intact.
    OpenRouter response shape: {"usage": {"prompt_tokens": N,
    "completion_tokens": M, "total_tokens": K}, "choices": [...]}.
    On 4xx/5xx response_data may be None — we log a 0-token row for
    error-rate visibility.
    """
    if pool is None:
        return
    try:
        await ensure_schema(pool)
        usage = (response_data or {}).get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        cost_usd, cost_brl = compute_cost(model, tokens_in, tokens_out)
        async with pool.acquire() as conn:
            import json as _json
            await conn.execute(
                """
                INSERT INTO ai_usage
                  (user_id, service, model, tokens_in, tokens_out,
                   cost_usd, cost_brl, status_code, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                """,
                user_id, service, model, tokens_in, tokens_out,
                cost_usd, cost_brl, status_code,
                _json.dumps(metadata) if metadata else None,
            )
    except Exception as err:  # noqa: BLE001
        log.debug("ai_usage log failed service=%s: %s", service, err)


async def get_usage_summary(
    pool: asyncpg.Pool, user_id: Optional[int], days: int = 7,
) -> dict[str, Any]:
    """Returns usage breakdown за последние N дней."""
    if pool is None:
        return {"error": "no_db"}
    try:
        await ensure_schema(pool)
    except Exception:  # noqa: BLE001
        pass
    since = datetime.now(timezone.utc) - timedelta(days=days)

    user_filter_sql = "AND user_id = $2" if user_id is not None else ""
    args = [since] + ([user_id] if user_id is not None else [])

    async with pool.acquire() as conn:
        totals_row = await conn.fetchrow(
            f"""
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(tokens_in), 0) AS in_total,
                   COALESCE(SUM(tokens_out), 0) AS out_total,
                   COALESCE(SUM(cost_usd), 0) AS usd_total,
                   COALESCE(SUM(cost_brl), 0) AS brl_total,
                   COUNT(*) FILTER (WHERE status_code >= 400) AS errors
              FROM ai_usage
             WHERE occurred_at >= $1 {user_filter_sql}
            """,
            *args,
        )
        by_service = await conn.fetch(
            f"""
            SELECT service, COUNT(*) AS calls,
                   COALESCE(SUM(cost_brl), 0) AS brl_total,
                   COALESCE(SUM(cost_usd), 0) AS usd_total,
                   COALESCE(SUM(tokens_in + tokens_out), 0) AS tokens
              FROM ai_usage
             WHERE occurred_at >= $1 {user_filter_sql}
             GROUP BY service
             ORDER BY brl_total DESC
            """,
            *args,
        )
        by_model = await conn.fetch(
            f"""
            SELECT model, COUNT(*) AS calls,
                   COALESCE(SUM(cost_brl), 0) AS brl_total
              FROM ai_usage
             WHERE occurred_at >= $1 {user_filter_sql}
             GROUP BY model
             ORDER BY brl_total DESC
            """,
            *args,
        )
        by_day = await conn.fetch(
            f"""
            SELECT DATE_TRUNC('day', occurred_at) AS day,
                   COUNT(*) AS calls,
                   COALESCE(SUM(cost_brl), 0) AS brl_total
              FROM ai_usage
             WHERE occurred_at >= $1 {user_filter_sql}
             GROUP BY day
             ORDER BY day ASC
            """,
            *args,
        )

    return {
        "period_days": days,
        "user_id": user_id,
        "calls": int(totals_row["calls"] or 0),
        "tokens_in": int(totals_row["in_total"] or 0),
        "tokens_out": int(totals_row["out_total"] or 0),
        "cost_usd": round(float(totals_row["usd_total"] or 0), 4),
        "cost_brl": round(float(totals_row["brl_total"] or 0), 2),
        "errors_count": int(totals_row["errors"] or 0),
        "by_service": [
            {
                "service": r["service"],
                "calls": int(r["calls"] or 0),
                "tokens": int(r["tokens"] or 0),
                "cost_brl": round(float(r["brl_total"] or 0), 2),
                "cost_usd": round(float(r["usd_total"] or 0), 4),
            }
            for r in by_service
        ],
        "by_model": [
            {
                "model": r["model"],
                "calls": int(r["calls"] or 0),
                "cost_brl": round(float(r["brl_total"] or 0), 2),
            }
            for r in by_model
        ],
        "by_day": [
            {
                "day": r["day"].date().isoformat() if r["day"] else None,
                "calls": int(r["calls"] or 0),
                "cost_brl": round(float(r["brl_total"] or 0), 2),
            }
            for r in by_day
        ],
    }


async def dispatch_daily_admin_summary(pool: asyncpg.Pool) -> None:
    """Fire one TG message to super-admins with yesterday's totals."""
    if pool is None:
        return
    summary = await get_usage_summary(pool, user_id=None, days=1)
    calls = summary.get("calls", 0)
    if calls == 0:
        return
    cost_brl = summary.get("cost_brl", 0)
    cost_usd = summary.get("cost_usd", 0)
    by_service = summary.get("by_service") or []
    lines = [
        "📊 *AI usage за 24h*",
        "",
        f"Total: {calls} calls · *R$ {cost_brl:.2f}* (US$ {cost_usd:.2f})",
        f"Errors: {summary.get('errors_count', 0)}",
        "",
    ]
    if by_service:
        lines.append("По сервисам:")
        for s in by_service[:8]:
            lines.append(
                f"  • {s['service']}: {s['calls']} calls · R$ {s['cost_brl']:.2f}"
            )
    text = "\n".join(lines)
    try:
        from . import tg_admin_alerts as alerts_svc
        await alerts_svc.send_admin_alert(
            title="AI usage daily",
            detail=text,
            severity="warn" if cost_brl > 50 else "warn",  # always warn-level
            service="ai_usage_tracker",
            deduplicate_key=f"ai_usage_daily:{datetime.now(timezone.utc).date().isoformat()}",
        )
    except Exception as err:  # noqa: BLE001
        log.debug("admin daily summary failed: %s", err)
