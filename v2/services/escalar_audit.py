"""Audit trail of AI actions and seller decisions.

Seller asks: «не понимаю что AI сам сделал». This service answers.

Each AI suggestion (questions / claims / promos / price shifts) and each
seller action on top of it is logged into escalar_audit. Daily summary
shows: «AI suggested 47 answers, seller approve rate 89%, edit rate 11%».

Action types:
  - ai_suggest_question  — AI generated a Q&A suggestion
  - q_approved          — seller hit ✅ Aprovar
  - q_edited            — seller hit ✏️ Editar (then sent custom)
  - q_regenerated       — seller hit 🔄 Outra sugestão
  - claim_resolved      — seller chose refund/return/exchange/etc
  - promo_accepted      — accept / accept_with_raise
  - promo_rejected      — reject / exit
  - price_raised        — raise_only / sale price shift
  - dyn_pricing_off     — raise_with_disable_dyn / sale shift с disable
  - dyn_pricing_on      — reenable_dynamic_pricing
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS escalar_audit (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  target_type TEXT,
  target_id TEXT,
  ai_response TEXT,
  user_action TEXT,
  metadata JSONB,
  occurred_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_user_day
  ON escalar_audit(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action
  ON escalar_audit(action, occurred_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


async def log_event(
    pool: asyncpg.Pool,
    *,
    user_id: int,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    ai_response: Optional[str] = None,
    user_action: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Fire-and-forget — never raises."""
    if pool is None:
        return
    try:
        await ensure_schema(pool)
        import json as _json
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO escalar_audit
                  (user_id, action, target_type, target_id,
                   ai_response, user_action, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                user_id, action, target_type, target_id,
                (ai_response or "")[:5000] if ai_response else None,
                user_action,
                _json.dumps(metadata) if metadata else None,
            )
    except Exception as err:  # noqa: BLE001
        log.debug("audit log failed action=%s: %s", action, err)


async def get_summary(
    pool: asyncpg.Pool, user_id: int, days: int = 7,
) -> dict[str, Any]:
    """Returns approve/edit/regen rates, action counts, recent events."""
    if pool is None:
        return {"error": "no_db"}
    await ensure_schema(pool)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with pool.acquire() as conn:
        action_counts = await conn.fetch(
            """
            SELECT action, COUNT(*) AS n
              FROM escalar_audit
             WHERE user_id = $1 AND occurred_at >= $2
             GROUP BY action ORDER BY n DESC
            """,
            user_id, since,
        )
        # Q&A funnel — pair ai_suggest_question with q_* outcomes by target_id.
        suggested = await conn.fetchval(
            """
            SELECT COUNT(*) FROM escalar_audit
             WHERE user_id = $1 AND occurred_at >= $2
               AND action = 'ai_suggest_question'
            """,
            user_id, since,
        ) or 0
        approved = await conn.fetchval(
            "SELECT COUNT(*) FROM escalar_audit WHERE user_id = $1 "
            "AND occurred_at >= $2 AND action = 'q_approved'",
            user_id, since,
        ) or 0
        edited = await conn.fetchval(
            "SELECT COUNT(*) FROM escalar_audit WHERE user_id = $1 "
            "AND occurred_at >= $2 AND action = 'q_edited'",
            user_id, since,
        ) or 0
        regenerated = await conn.fetchval(
            "SELECT COUNT(*) FROM escalar_audit WHERE user_id = $1 "
            "AND occurred_at >= $2 AND action = 'q_regenerated'",
            user_id, since,
        ) or 0
        recent = await conn.fetch(
            """
            SELECT action, target_id, user_action, occurred_at
              FROM escalar_audit
             WHERE user_id = $1 AND occurred_at >= $2
             ORDER BY occurred_at DESC LIMIT 30
            """,
            user_id, since,
        )

    qa_total = max(int(approved) + int(edited) + int(regenerated), 1)
    return {
        "period_days": days,
        "action_counts": [{"action": r["action"], "count": int(r["n"])} for r in action_counts],
        "qa_funnel": {
            "suggested": int(suggested),
            "approved": int(approved),
            "edited": int(edited),
            "regenerated": int(regenerated),
            "approve_rate_pct": round(int(approved) * 100 / qa_total, 1),
            "edit_rate_pct": round(int(edited) * 100 / qa_total, 1),
            "regen_rate_pct": round(int(regenerated) * 100 / qa_total, 1),
        },
        "recent_events": [
            {
                "action": r["action"],
                "target_id": r["target_id"],
                "user_action": r["user_action"],
                "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] else None,
            }
            for r in recent
        ],
    }
