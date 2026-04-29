"""Bank-balance anchors for the Classification page.

Seller manually sets a known balance for a bank (e.g. "R$ 5000 em
01/04/2026"). Everything after that anchor is reconciled against the
sum of transactions in our parsed statements:

    expected_now = recorded_balance + Σ(value of every tx with date > recorded_date)

If the next statement closes at `expected_now`, the books are clean.
If not, something is missing or duplicated.

One active anchor per `(user, source_key)`. Older ones are kept (soft
superseded) so we can audit when the user re-anchored and why.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

CURRENCY_BY_BANK = {
    "extrato_mp": "BRL",
    "extrato_nubank": "BRL",
    "extrato_c6_brl": "BRL",
    "extrato_c6_usd": "USD",
}

# Mapping from the bank `id` used in legacy onboarding (`projects[*].bank_accounts`)
# to our normalized `source_key`. Several banks reuse extrato_nubank as the
# storage source — see `_admin/onboarding.py:BANK_LIST`.
LEGACY_BANK_TO_SOURCE = {
    "nubank":       "extrato_nubank",
    "itau":         "extrato_nubank",
    "bradesco":     "extrato_nubank",
    "bb":           "extrato_nubank",
    "caixa":        "extrato_nubank",
    "santander":    "extrato_nubank",
    "c6":           "extrato_c6_brl",
    "c6_brl":       "extrato_c6_brl",
    "c6_usd":       "extrato_c6_usd",
    "mercado_pago": "extrato_mp",
    "mp":           "extrato_mp",
}


_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS escalar_bank_balances (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL,
      source_key TEXT NOT NULL,
      balance NUMERIC NOT NULL,
      currency TEXT NOT NULL DEFAULT 'BRL',
      balance_date DATE NOT NULL,
      notes TEXT,
      active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      superseded_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bank_balances_user_source_active ON escalar_bank_balances(user_id, source_key, active)",
    "CREATE INDEX IF NOT EXISTS idx_bank_balances_user_source_date ON escalar_bank_balances(user_id, source_key, balance_date DESC)",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    for stmt in _CREATE_STATEMENTS:
        async with pool.acquire() as conn:
            await conn.execute(stmt)


def _legacy_anchor_from_user_data(user_id: int, source_key: str) -> Optional[dict[str, Any]]:
    """Mine `user_data.projects[*].bank_accounts` for an onboarding-saved
    balance that maps to this `source_key`. Returns the entry with the most
    recent balance_date, or None if nothing matches.

    The legacy onboarding (Streamlit step 9) duplicates the same bank_accounts
    list across every project — we just deduplicate by (bank_id, balance_date,
    balance) before picking the latest.
    """
    try:
        from v2.legacy import db_storage as _legacy_db
        # Try f2_-prefixed key first (current convention), fall back to legacy
        projects = _legacy_db.db_load("f2_projects", user_id=user_id)
        if not projects:
            projects = _legacy_db.db_load("projects", user_id=user_id)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(projects, dict):
        return None

    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for pdata in projects.values():
        if not isinstance(pdata, dict):
            continue
        for entry in (pdata.get("bank_accounts") or []):
            if not isinstance(entry, dict):
                continue
            mapped = LEGACY_BANK_TO_SOURCE.get(str(entry.get("bank") or "").lower())
            if mapped != source_key:
                continue
            try:
                bal = float(entry.get("balance"))
            except (TypeError, ValueError):
                continue
            bdate = str(entry.get("balance_date") or "")[:10]
            key = (bdate, round(bal, 2))
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"balance": bal, "balance_date": bdate})

    if not candidates:
        return None
    # Latest balance_date wins; ISO strings sort correctly
    candidates.sort(key=lambda x: x["balance_date"] or "", reverse=True)
    pick = candidates[0]
    return {
        "balance": pick["balance"],
        "balance_date": pick["balance_date"],
        "currency": CURRENCY_BY_BANK.get(source_key, "BRL"),
    }


async def get_active(
    pool: asyncpg.Pool, user_id: int, source_key: str,
) -> Optional[dict[str, Any]]:
    """Return the active anchor for one bank.

    Looks in `escalar_bank_balances` first; if empty, tries to migrate any
    onboarding-saved balance from `user_data.projects[*].bank_accounts`.
    The migration is idempotent (one-shot upsert) so subsequent reads hit
    the table directly.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, source_key, balance, currency,
                   balance_date, notes, active,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
              FROM escalar_bank_balances
             WHERE user_id = $1 AND source_key = $2 AND active = TRUE
             ORDER BY balance_date DESC, id DESC
             LIMIT 1
            """,
            user_id, source_key,
        )

    if not row:
        # Migrate from legacy onboarding storage on first access
        legacy = _legacy_anchor_from_user_data(user_id, source_key)
        if legacy and legacy.get("balance_date"):
            try:
                bdate = date.fromisoformat(legacy["balance_date"])
                migrated = await upsert_balance(
                    pool, user_id,
                    source_key=source_key,
                    balance=legacy["balance"],
                    balance_date=bdate,
                    currency=legacy["currency"],
                    notes="Importado do onboarding",
                )
                migrated["migrated_from_legacy"] = True
                log.info("bank_balances: migrated legacy onboarding anchor user=%s source=%s balance=%s date=%s",
                         user_id, source_key, legacy["balance"], legacy["balance_date"])
                return migrated
            except Exception as err:  # noqa: BLE001
                log.warning("bank_balances legacy migration failed: %s", err)
        return None

    return {
        "id": row["id"],
        "source_key": row["source_key"],
        "balance": float(row["balance"]),
        "currency": row["currency"],
        "balance_date": row["balance_date"].isoformat() if row["balance_date"] else None,
        "notes": row["notes"] or "",
        "active": row["active"],
        "created_at": row["created_at"],
    }


async def list_history(
    pool: asyncpg.Pool, user_id: int, source_key: str, *, limit: int = 20,
) -> list[dict[str, Any]]:
    """All anchors (active + superseded) for one bank, newest first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, balance, currency, balance_date, notes, active,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
                   to_char(superseded_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS superseded_at
              FROM escalar_bank_balances
             WHERE user_id = $1 AND source_key = $2
             ORDER BY balance_date DESC, id DESC
             LIMIT $3
            """,
            user_id, source_key, limit,
        )
    return [
        {
            "id": r["id"],
            "balance": float(r["balance"]),
            "currency": r["currency"],
            "balance_date": r["balance_date"].isoformat() if r["balance_date"] else None,
            "notes": r["notes"] or "",
            "active": r["active"],
            "created_at": r["created_at"],
            "superseded_at": r["superseded_at"],
        }
        for r in rows
    ]


async def upsert_balance(
    pool: asyncpg.Pool, user_id: int, *,
    source_key: str,
    balance: float,
    balance_date: date,
    currency: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Insert a new anchor and supersede the previous active one (if any).

    Returns the new active anchor.
    """
    if source_key not in CURRENCY_BY_BANK:
        return {"error": "unsupported_source", "source_key": source_key}
    if currency is None:
        currency = CURRENCY_BY_BANK[source_key]

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE escalar_bank_balances
                   SET active = FALSE, superseded_at = NOW()
                 WHERE user_id = $1 AND source_key = $2 AND active = TRUE
                """,
                user_id, source_key,
            )
            row = await conn.fetchrow(
                """
                INSERT INTO escalar_bank_balances
                  (user_id, source_key, balance, currency, balance_date, notes, active)
                VALUES ($1, $2, $3, $4, $5, $6, TRUE)
                RETURNING id, balance_date,
                          to_char(created_at AT TIME ZONE 'UTC',
                                  'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
                """,
                user_id, source_key, balance, currency, balance_date, notes,
            )
    return {
        "id": row["id"],
        "source_key": source_key,
        "balance": float(balance),
        "currency": currency,
        "balance_date": row["balance_date"].isoformat() if row["balance_date"] else None,
        "notes": notes or "",
        "active": True,
        "created_at": row["created_at"],
    }


async def delete_balance(
    pool: asyncpg.Pool, user_id: int, source_key: str,
) -> dict[str, Any]:
    """Remove the active anchor for a bank — leaves history intact."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE escalar_bank_balances
               SET active = FALSE, superseded_at = NOW()
             WHERE user_id = $1 AND source_key = $2 AND active = TRUE
            """,
            user_id, source_key,
        )
    return {"cleared": result.endswith(" 1") or result.endswith(" 0") is False, "source_key": source_key}


# ── Reconciliation ──────────────────────────────────────────────────────────


def reconcile(
    *,
    recorded_balance: float,
    recorded_date: date,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum every parsed transaction whose date is STRICTLY AFTER the anchor
    date and project the expected current balance.

    Returns a small summary dict the UI can render directly:
      {recorded_balance, recorded_date,
       txn_count_after, txn_sum_after,
       expected_balance, latest_tx_date}
    """
    txn_sum = 0.0
    txn_count = 0
    latest = None
    for r in rows:
        d_str = (r.get("date") or "")[:10]
        if not d_str:
            continue
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        if d <= recorded_date:
            continue
        try:
            txn_sum += float(r.get("value_brl") or 0)
        except (TypeError, ValueError):
            continue
        txn_count += 1
        if latest is None or d > latest:
            latest = d

    return {
        "recorded_balance": float(recorded_balance),
        "recorded_date": recorded_date.isoformat(),
        "txn_count_after": txn_count,
        "txn_sum_after": round(txn_sum, 2),
        "expected_balance": round(float(recorded_balance) + txn_sum, 2),
        "latest_tx_date": latest.isoformat() if latest else None,
    }
