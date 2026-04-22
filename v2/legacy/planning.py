"""Planned-payments / DDS planning — per-user port of `_admin/dds_planning.py`.

Storage: `user_data.f2_planned_payments` (JSONB list). Each payment:
    { id, date, amount, direction, recurrence, contragent, category, note,
      project, created_at }
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

PLANNED_KEY = "f2_planned_payments"

RECURRENCE_OPTS = ("once", "monthly", "quarterly", "yearly")
DIRECTION_OPTS = ("expense", "income")


# ── Storage ─────────────────────────────────────────────────────────────────

def load_payments() -> list[dict]:
    from .db_storage import db_load
    data = db_load(PLANNED_KEY)
    return list(data) if isinstance(data, list) else []


def _save_payments(payments: list[dict]) -> None:
    from .db_storage import db_save
    db_save(PLANNED_KEY, payments)


def _next_id(payments: list[dict]) -> int:
    existing = [int(p.get("id") or 0) for p in payments if isinstance(p.get("id"), (int, float))]
    return (max(existing) + 1) if existing else 1


def add_payment(entry: dict) -> dict:
    """Append a payment with auto-id and created_at. Returns the stored row.

    `paid_at` (ISO timestamp) — null when the payment is still pending.
    Overdue unpaid expenses feed into the Balance sheet as Accounts Payable.
    """
    payments = load_payments()
    row = {
        "id": _next_id(payments),
        "date": str(entry.get("date") or date.today().isoformat()),
        "amount": float(entry.get("amount", 0)),
        "direction": entry.get("direction", "expense"),
        "recurrence": entry.get("recurrence", "once"),
        "contragent": str(entry.get("contragent", "")),
        "category": str(entry.get("category", "")),
        "note": str(entry.get("note", "")),
        "project": entry.get("project"),
        "paid_at": entry.get("paid_at") or None,
        "created_at": datetime.now().isoformat(),
    }
    if row["direction"] not in DIRECTION_OPTS:
        row["direction"] = "expense"
    if row["recurrence"] not in RECURRENCE_OPTS:
        row["recurrence"] = "once"
    payments.append(row)
    _save_payments(payments)
    return row


def delete_payment(payment_id: int) -> bool:
    payments = load_payments()
    new_list = [p for p in payments if int(p.get("id") or -1) != int(payment_id)]
    if len(new_list) == len(payments):
        return False
    _save_payments(new_list)
    return True


def update_payment(payment_id: int, updates: dict) -> bool:
    allowed = {"date", "amount", "direction", "recurrence", "contragent",
               "category", "note", "project", "paid_at"}
    payments = load_payments()
    found = False
    for p in payments:
        if int(p.get("id") or -1) == int(payment_id):
            for k, v in updates.items():
                if k in allowed:
                    p[k] = v
            found = True
            break
    if not found:
        return False
    _save_payments(payments)
    return True


def mark_paid(payment_id: int, paid_at: Any = None) -> bool:
    """Toggle the paid state: set `paid_at` to ISO timestamp (or clear it).

    Pass `paid_at=None` to mark as unpaid again. Omit to stamp with now().
    """
    if paid_at is None:
        stamp = datetime.now().isoformat(timespec="seconds")
    elif paid_at is False or paid_at == "":
        stamp = None  # clear — caller wants to mark pending
    else:
        stamp = str(paid_at)
    return update_payment(payment_id, {"paid_at": stamp})


def list_unpaid_ap(project: str, as_of: date) -> list[dict]:
    """AP feed for the Balance sheet: expense-direction payments with
    `date <= as_of` and `paid_at IS NULL`. Returns rows sorted by date asc.

    Recurring payments are expanded elsewhere (monthly plan view). For AP we
    only care about the concrete dated rows the user put in — no expansion.
    """
    pj = (project or "").upper()
    rows: list[dict] = []
    for p in load_payments():
        if str(p.get("direction") or "").lower() != "expense":
            continue
        if pj and str(p.get("project") or "").upper() != pj:
            continue
        if p.get("paid_at"):
            continue  # already paid — not an obligation
        try:
            d = _parse_date(p.get("date"))
        except Exception:
            continue
        if d > as_of:
            continue  # future — not yet overdue
        rows.append(p)
    rows.sort(key=lambda x: str(x.get("date") or ""))
    return rows


def unpaid_ap_total(project: str, as_of: date) -> float:
    """Sum of |amount| for unpaid expenses with `date <= as_of`."""
    return round(sum(abs(float(p.get("amount") or 0))
                     for p in list_unpaid_ap(project, as_of)), 2)


# ── Monthly expansion ──────────────────────────────────────────────────────

def _parse_date(s: Any) -> date:
    if isinstance(s, date):
        return s
    s = str(s or "")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return date.today()


def expand_payment_to_months(payment: dict, start: date, months: int = 12) -> list[dict]:
    """Return [{month, amount, direction, ...}] — one entry per month the payment hits."""
    recurrence = payment.get("recurrence", "once")
    dt = _parse_date(payment.get("date"))
    amount = float(payment.get("amount", 0))
    out: list[dict] = []

    for i in range(months):
        m = start.month + i
        y = start.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        month_key = f"{y:04d}-{m:02d}"

        hit = False
        if recurrence == "once":
            hit = (dt.year == y and dt.month == m)
        elif recurrence == "monthly":
            hit = (dt <= date(y, m, 28))
        elif recurrence == "quarterly":
            if dt <= date(y, m, 28):
                diff = (y - dt.year) * 12 + (m - dt.month)
                hit = (diff % 3 == 0)
        elif recurrence == "yearly":
            if dt <= date(y, m, 28):
                hit = (m == dt.month)

        if hit:
            out.append({
                "month": month_key,
                "amount": amount,
                "direction": payment.get("direction", "expense"),
                "contragent": payment.get("contragent", ""),
                "category": payment.get("category", ""),
                "note": payment.get("note", ""),
                "project": payment.get("project"),
                "payment_id": payment.get("id"),
                "recurrence": recurrence,
            })
    return out


def build_monthly_plan(months: int = 12, start: date | None = None) -> dict:
    """Aggregate all payments into monthly buckets for the next `months`."""
    start = start or date.today().replace(day=1)
    plan: dict[str, dict] = {}
    for i in range(months):
        m = start.month + i
        y = start.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        plan[f"{y:04d}-{m:02d}"] = {
            "income": [], "expense": [], "total_in": 0.0, "total_out": 0.0, "net": 0.0,
        }

    for payment in load_payments():
        for item in expand_payment_to_months(payment, start, months):
            b = plan.get(item["month"])
            if not b:
                continue
            if item["direction"] == "income":
                b["income"].append(item)
                b["total_in"] += item["amount"]
            else:
                b["expense"].append(item)
                b["total_out"] += item["amount"]
    for b in plan.values():
        b["net"] = b["total_in"] - b["total_out"]
    return plan


# ── Recurring detection from bank uploads ──────────────────────────────────

_BANK_SOURCES = {"extrato_mp", "extrato_nubank", "extrato_c6_brl", "extrato_c6_usd"}


def detect_recurring_from_bank_sync(user_id: int, min_occurrences: int = 3) -> list[dict]:
    """Scan every bank upload of the user, apply classification, and flag
    labels that occur in >= min_occurrences distinct months.

    Synchronous version — reads file_bytes directly via psycopg2 (we use this
    inside FastAPI route thread-pool; no asyncpg dependency).
    """
    import os
    import psycopg2
    import pandas as pd  # noqa: F401
    from .bank_tx import parse_bank_tx_bytes

    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")
    if not dsn:
        return []
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            """SELECT id, source_key, file_bytes
               FROM uploads
               WHERE user_id = %s AND source_key = ANY(%s) AND file_bytes IS NOT NULL""",
            (user_id, list(_BANK_SOURCES)),
        )
        uploads = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        return []

    # Collect all classified transactions across all bank files
    all_rows: list[dict] = []
    for _uid, source_key, file_bytes in uploads:
        try:
            rows = parse_bank_tx_bytes(source_key, bytes(file_bytes))
        except Exception:
            continue
        all_rows.extend(rows)

    if not all_rows:
        return []

    # Extract month from date string — best effort (supports DD/MM/YYYY + YYYY-MM-DD)
    def _month_of(date_str: str) -> str:
        s = (date_str or "").strip()
        if not s:
            return ""
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(s[:10], fmt).strftime("%Y-%m")
            except ValueError:
                continue
        # ML-like "24 de março de 2026"
        import re
        m = re.search(r"(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})", s)
        if m:
            pt_months = {
                "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
                "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
                "outubro": 10, "novembro": 11, "dezembro": 12,
            }
            mn = pt_months.get(m.group(2).lower())
            if mn:
                return f"{m.group(3)}-{mn:02d}"
        return ""

    # Group by label (rule label or description prefix)
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for tx in all_rows:
        cat = tx.get("category", "uncategorized")
        if cat in ("internal_transfer",):
            continue
        label = tx.get("label") or str(tx.get("description", ""))[:40]
        direction = "income" if cat == "income" or (tx.get("value_brl") or 0) > 0 and cat != "uncategorized" else "expense"
        if cat == "uncategorized":
            continue  # skip unclassified for now — noisy
        key = f"{direction}||{label}"
        mo = _month_of(str(tx.get("date", "")))
        amt = abs(float(tx.get("value_brl") or 0))
        if mo and amt > 0:
            groups[key].append({"month": mo, "amount": amt, "category": cat})

    suggestions: list[dict] = []
    for key, rows in groups.items():
        unique_months = {r["month"] for r in rows}
        if len(unique_months) < min_occurrences:
            continue
        # Sum per-month (collapse multiple txs in same month)
        per_month: dict[str, float] = {}
        for r in rows:
            per_month[r["month"]] = per_month.get(r["month"], 0) + r["amount"]
        vals = list(per_month.values())
        direction, label = key.split("||", 1)
        suggestions.append({
            "contragent": label,
            "category": rows[0]["category"] if rows else "",
            "direction": direction,
            "avg_amount": sum(vals) / len(vals),
            "min_amount": min(vals),
            "max_amount": max(vals),
            "months_count": len(unique_months),
            "total_txs": len(rows),
        })
    suggestions.sort(key=lambda x: (-1 if x["direction"] == "expense" else 1, -x["avg_amount"]))
    return suggestions
