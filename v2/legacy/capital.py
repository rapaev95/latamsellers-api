"""Capital & Obligations storage — loans, dividends, initial equity.

Per-user JSONB storage via `db_storage.db_load/db_save`, same pattern as
`planning.py` (planned_payments). No new SQL tables.

Keys in `user_data`:
    - f2_loans       → list[dict]  (active + closed loans)
    - f2_dividends   → list[dict]  (owner payouts)

`initial_equity_brl` lives on the project itself (projects_db JSON / config
editable keys), not here — it's a single scalar per project.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

LOANS_KEY = "f2_loans"
DIVIDENDS_KEY = "f2_dividends"


# ── Shared helpers ──────────────────────────────────────────────────────────

def _next_id(items: list[dict]) -> int:
    existing = [int(it.get("id") or 0) for it in items if isinstance(it.get("id"), (int, float))]
    return (max(existing) + 1) if existing else 1


def _filter_by_project(items: list[dict], project: Optional[str]) -> list[dict]:
    if not project:
        return items
    pj = project.upper()
    return [it for it in items if str(it.get("project") or "").upper() == pj]


def _safe_float(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


# ── Loans ────────────────────────────────────────────────────────────────────

def load_loans(project: Optional[str] = None) -> list[dict]:
    from .db_storage import db_load
    data = db_load(LOANS_KEY)
    if not isinstance(data, list):
        return []
    return _filter_by_project(data, project)


def _save_loans(items: list[dict]) -> None:
    from .db_storage import db_save
    db_save(LOANS_KEY, items)


def add_loan(entry: dict) -> dict:
    items = load_loans()
    row = {
        "id": _next_id(items),
        "project": str(entry.get("project") or "").upper(),
        "name": str(entry.get("name") or "").strip(),
        "principal_brl": _safe_float(entry.get("principal_brl")),
        "outstanding_brl": _safe_float(entry.get("outstanding_brl") or entry.get("principal_brl")),
        "monthly_payment_brl": _safe_float(entry.get("monthly_payment_brl")),
        "rate_pct": _safe_float(entry.get("rate_pct")),
        "start_date": str(entry.get("start_date") or ""),
        "closed_at": entry.get("closed_at"),  # None = active
        "note": str(entry.get("note") or ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    items.append(row)
    _save_loans(items)
    return row


def update_loan(loan_id: int, updates: dict) -> bool:
    allowed = {"name", "principal_brl", "outstanding_brl", "monthly_payment_brl",
               "rate_pct", "start_date", "closed_at", "note", "project"}
    items = load_loans()
    found = False
    for it in items:
        if int(it.get("id") or -1) == int(loan_id):
            for k, v in updates.items():
                if k in allowed:
                    it[k] = v
            found = True
            break
    if not found:
        return False
    _save_loans(items)
    return True


def delete_loan(loan_id: int) -> bool:
    items = load_loans()
    new_list = [it for it in items if int(it.get("id") or -1) != int(loan_id)]
    if len(new_list) == len(items):
        return False
    _save_loans(new_list)
    return True


def loans_balance(project: str, as_of: Optional[date] = None) -> float:
    """Sum of `outstanding_brl` across active loans for the project.

    `closed_at > as_of` → treated as active (loan closed after as_of date).
    """
    loans = load_loans(project)
    total = 0.0
    for it in loans:
        closed = it.get("closed_at")
        if closed and as_of:
            try:
                d = datetime.strptime(str(closed)[:10], "%Y-%m-%d").date()
                if d <= as_of:
                    continue  # closed before as_of
            except (ValueError, TypeError):
                pass
        elif closed:
            continue
        total += _safe_float(it.get("outstanding_brl"))
    return round(total, 2)


# ── Dividends ────────────────────────────────────────────────────────────────

def load_dividends(project: Optional[str] = None) -> list[dict]:
    from .db_storage import db_load
    data = db_load(DIVIDENDS_KEY)
    if not isinstance(data, list):
        return []
    return _filter_by_project(data, project)


def _save_dividends(items: list[dict]) -> None:
    from .db_storage import db_save
    db_save(DIVIDENDS_KEY, items)


def add_dividend(entry: dict) -> dict:
    items = load_dividends()
    row = {
        "id": _next_id(items),
        "project": str(entry.get("project") or "").upper(),
        "date": str(entry.get("date") or date.today().isoformat()),
        "amount_brl": _safe_float(entry.get("amount_brl")),
        "note": str(entry.get("note") or ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    items.append(row)
    _save_dividends(items)
    return row


def delete_dividend(dividend_id: int) -> bool:
    items = load_dividends()
    new_list = [it for it in items if int(it.get("id") or -1) != int(dividend_id)]
    if len(new_list) == len(items):
        return False
    _save_dividends(new_list)
    return True


def dividends_total(project: str, as_of: Optional[date] = None) -> float:
    """Sum dividend payouts up to `as_of` (inclusive)."""
    items = load_dividends(project)
    total = 0.0
    for it in items:
        ds = str(it.get("date") or "")[:10]
        if as_of:
            try:
                d = datetime.strptime(ds, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if d > as_of:
                continue
        total += _safe_float(it.get("amount_brl"))
    return round(total, 2)


# ── Initial equity (lives on the project dict) ───────────────────────────────

def initial_equity(project_meta: dict) -> float:
    return _safe_float(project_meta.get("initial_equity_brl"))
