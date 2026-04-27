"""User onboarding state — collects business context that drives Escalar UX.

The seller answers a short questionnaire on first entry; their answers persist
in user_onboarding and inform downstream modules:

- business_model:  'private_label' / 'reseller' / 'mixed' / 'unknown'
                    drives catalog flow (PL skips it; Reseller pushes it)
- alert_prefs:     {subStatus, stockRunout, liderTier, anomaly, promocoes,
                    questions} → which TG pushes the user wants
- completed_at:    NULL until the wizard is fully done; auto-redirect logic
                    on the frontend uses this to surface the wizard once.

A user may re-run the wizard via the Settings page (resets completed_at to
NULL but keeps prior answers as defaults).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS user_onboarding (
  user_id INTEGER PRIMARY KEY,
  business_model TEXT,
  has_own_warehouse BOOLEAN,
  alert_prefs JSONB DEFAULT '{}'::jsonb,
  current_step INTEGER DEFAULT 1,
  completed_at TIMESTAMPTZ,
  skipped_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

VALID_BUSINESS_MODELS = ("private_label", "reseller", "mixed", "unknown")

DEFAULT_ALERT_PREFS = {
    "subStatus": True,
    "stockRunout": True,
    "liderTier": True,
    "anomaly": False,
    "promocoes": True,
    "questions": True,
}


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


async def get_state(pool: asyncpg.Pool, user_id: int) -> dict[str, Any]:
    """Return current onboarding state. If row missing, returns sane defaults."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT business_model, has_own_warehouse, alert_prefs,
                   current_step, completed_at, skipped_at
              FROM user_onboarding
             WHERE user_id = $1
            """,
            user_id,
        )
    if not row:
        return {
            "businessModel": None,
            "hasOwnWarehouse": None,
            "alertPrefs": DEFAULT_ALERT_PREFS,
            "currentStep": 1,
            "completedAt": None,
            "skippedAt": None,
            "exists": False,
        }
    prefs = row["alert_prefs"]
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except Exception:  # noqa: BLE001
            prefs = {}
    if not isinstance(prefs, dict):
        prefs = {}
    return {
        "businessModel": row["business_model"],
        "hasOwnWarehouse": row["has_own_warehouse"],
        "alertPrefs": {**DEFAULT_ALERT_PREFS, **prefs},
        "currentStep": int(row["current_step"] or 1),
        "completedAt": row["completed_at"].isoformat() if row["completed_at"] else None,
        "skippedAt": row["skipped_at"].isoformat() if row["skipped_at"] else None,
        "exists": True,
    }


async def upsert_step(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    current_step: Optional[int] = None,
    business_model: Optional[str] = None,
    has_own_warehouse: Optional[bool] = None,
    alert_prefs: Optional[dict[str, Any]] = None,
) -> None:
    """Save a step's answers without marking complete."""
    if business_model is not None and business_model not in VALID_BUSINESS_MODELS:
        raise ValueError(f"invalid_business_model: {business_model}")
    async with pool.acquire() as conn:
        # Upsert the row, merging alert_prefs jsonb
        await conn.execute(
            """
            INSERT INTO user_onboarding (user_id, current_step, business_model,
                                         has_own_warehouse, alert_prefs, updated_at)
            VALUES ($1, COALESCE($2, 1), $3, $4, COALESCE($5::jsonb, '{}'::jsonb), NOW())
            ON CONFLICT (user_id) DO UPDATE SET
              current_step      = COALESCE($2, user_onboarding.current_step),
              business_model    = COALESCE($3, user_onboarding.business_model),
              has_own_warehouse = COALESCE($4, user_onboarding.has_own_warehouse),
              alert_prefs       = CASE
                                    WHEN $5::jsonb IS NULL THEN user_onboarding.alert_prefs
                                    ELSE user_onboarding.alert_prefs || $5::jsonb
                                  END,
              updated_at        = NOW()
            """,
            user_id, current_step, business_model, has_own_warehouse,
            json.dumps(alert_prefs) if alert_prefs is not None else None,
        )


async def complete(pool: asyncpg.Pool, user_id: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_onboarding (user_id, completed_at, current_step, updated_at)
            VALUES ($1, NOW(), 999, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
              completed_at = NOW(),
              current_step = 999,
              skipped_at = NULL,
              updated_at = NOW()
            """,
            user_id,
        )


async def skip(pool: asyncpg.Pool, user_id: int) -> None:
    """User dismissed the wizard for now. Won't auto-prompt again until they
    explicitly open it from settings."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_onboarding (user_id, skipped_at, updated_at)
            VALUES ($1, NOW(), NOW())
            ON CONFLICT (user_id) DO UPDATE SET
              skipped_at = NOW(),
              updated_at = NOW()
            """,
            user_id,
        )


async def reset(pool: asyncpg.Pool, user_id: int) -> None:
    """Reopen the wizard — keeps prior answers as defaults, clears completed/skipped."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE user_onboarding
               SET completed_at = NULL,
                   skipped_at = NULL,
                   current_step = 1,
                   updated_at = NOW()
             WHERE user_id = $1
            """,
            user_id,
        )
