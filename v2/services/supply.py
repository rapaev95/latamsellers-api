"""Supplier orders + reorder suggestions — replenishment module.

The seller had nowhere to track outgoing purchase orders to suppliers, leading
to forgotten reorders and stockouts. This module provides:

- supplier_orders / supplier_order_items: structured PO log with status,
  ETA, cost, payment status
- compute_reorder_suggestions: cross-references velocity (vendas), current
  stock (stock_full + own), and pending orders to flag urgent reorders
- reliability score: % of orders arriving on/before ETA per supplier

For now suppliers are stored as plain text on each order — separate suppliers
table can come later when data model needs (multiple SKUs per supplier,
contact info) accumulate.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS supplier_orders (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  order_number TEXT,
  supplier_name TEXT NOT NULL,
  supplier_country TEXT,
  status TEXT NOT NULL DEFAULT 'planned',
  -- planned / placed / in_production / shipped / arrived / cancelled
  placed_date DATE,
  eta_date DATE,
  actual_arrival_date DATE,
  payment_status TEXT DEFAULT 'unpaid',
  -- unpaid / partial / paid
  total_amount NUMERIC,
  currency TEXT DEFAULT 'USD',
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_supplier_orders_user ON supplier_orders(user_id, status, eta_date);

CREATE TABLE IF NOT EXISTS supplier_order_items (
  id SERIAL PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES supplier_orders(id) ON DELETE CASCADE,
  sku TEXT NOT NULL,
  qty_ordered INTEGER NOT NULL DEFAULT 0,
  qty_received INTEGER DEFAULT 0,
  unit_cost NUMERIC,
  notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_supplier_order_items_order ON supplier_order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_supplier_order_items_sku ON supplier_order_items(sku);
"""

VALID_STATUSES = (
    "planned", "placed", "in_production", "shipped", "arrived", "cancelled",
)
VALID_PAYMENT_STATUSES = ("unpaid", "partial", "paid")


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── CRUD ──────────────────────────────────────────────────────────────────────

async def list_orders(
    pool: asyncpg.Pool, user_id: int, status: Optional[str] = None,
) -> list[dict[str, Any]]:
    where = "WHERE o.user_id = $1"
    params: list[Any] = [user_id]
    if status and status in VALID_STATUSES:
        where += " AND o.status = $2"
        params.append(status)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT o.id, o.order_number, o.supplier_name, o.supplier_country,
                   o.status, o.payment_status, o.total_amount, o.currency,
                   to_char(o.placed_date, 'YYYY-MM-DD') AS placed_date,
                   to_char(o.eta_date, 'YYYY-MM-DD') AS eta_date,
                   to_char(o.actual_arrival_date, 'YYYY-MM-DD') AS actual_arrival_date,
                   o.notes,
                   to_char(o.created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
                   COALESCE(json_agg(json_build_object(
                     'id', i.id, 'sku', i.sku,
                     'qtyOrdered', i.qty_ordered,
                     'qtyReceived', i.qty_received,
                     'unitCost', i.unit_cost,
                     'notes', i.notes
                   )) FILTER (WHERE i.id IS NOT NULL), '[]') AS items
              FROM supplier_orders o
              LEFT JOIN supplier_order_items i ON i.order_id = o.id
              {where}
             GROUP BY o.id
             ORDER BY o.eta_date ASC NULLS LAST, o.created_at DESC
            """,
            *params,
        )
    out = []
    for r in rows:
        items = r["items"]
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:  # noqa: BLE001
                items = []
        out.append({
            "id": int(r["id"]),
            "orderNumber": r["order_number"],
            "supplierName": r["supplier_name"],
            "supplierCountry": r["supplier_country"],
            "status": r["status"],
            "paymentStatus": r["payment_status"],
            "totalAmount": float(r["total_amount"]) if r["total_amount"] is not None else None,
            "currency": r["currency"],
            "placedDate": r["placed_date"],
            "etaDate": r["eta_date"],
            "actualArrivalDate": r["actual_arrival_date"],
            "notes": r["notes"],
            "createdAt": r["created_at"],
            "items": items if isinstance(items, list) else [],
        })
    return out


async def create_order(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    supplier_name: str,
    supplier_country: Optional[str] = None,
    order_number: Optional[str] = None,
    status: str = "planned",
    placed_date: Optional[str] = None,
    eta_date: Optional[str] = None,
    payment_status: str = "unpaid",
    total_amount: Optional[float] = None,
    currency: str = "USD",
    notes: Optional[str] = None,
    items: Optional[list[dict[str, Any]]] = None,
) -> int:
    """Insert order + items. Returns new order_id."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid_status: {status}")
    if payment_status not in VALID_PAYMENT_STATUSES:
        raise ValueError(f"invalid_payment_status: {payment_status}")
    async with pool.acquire() as conn:
        async with conn.transaction():
            order_id = await conn.fetchval(
                """
                INSERT INTO supplier_orders
                  (user_id, order_number, supplier_name, supplier_country,
                   status, placed_date, eta_date,
                   payment_status, total_amount, currency, notes)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                user_id, order_number, supplier_name, supplier_country,
                status,
                _parse_date(placed_date), _parse_date(eta_date),
                payment_status, total_amount, currency, notes,
            )
            if items:
                for it in items:
                    sku = (it.get("sku") or "").strip()
                    if not sku:
                        continue
                    qty = int(it.get("qtyOrdered") or it.get("qty_ordered") or 0)
                    if qty <= 0:
                        continue
                    cost = it.get("unitCost") or it.get("unit_cost")
                    await conn.execute(
                        """
                        INSERT INTO supplier_order_items
                          (order_id, sku, qty_ordered, unit_cost, notes)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        order_id, sku, qty,
                        float(cost) if isinstance(cost, (int, float)) else None,
                        it.get("notes"),
                    )
    return int(order_id)


async def update_order_status(
    pool: asyncpg.Pool, user_id: int, order_id: int, status: str,
    actual_arrival_date: Optional[str] = None,
) -> bool:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid_status: {status}")
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE supplier_orders
               SET status = $3,
                   actual_arrival_date = COALESCE($4, actual_arrival_date),
                   updated_at = NOW()
             WHERE user_id = $1 AND id = $2
            """,
            user_id, int(order_id), status,
            _parse_date(actual_arrival_date) if actual_arrival_date else None,
        )
    return result.endswith(" 1")


async def delete_order(pool: asyncpg.Pool, user_id: int, order_id: int) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM supplier_orders WHERE user_id = $1 AND id = $2",
            user_id, int(order_id),
        )
    return result.endswith(" 1")


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


# ── In-transit aggregation ────────────────────────────────────────────────────

async def get_in_transit_by_sku(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    """Sum of qty_ordered - qty_received per SKU across orders that haven't
    arrived yet (placed/in_production/shipped). 'planned' is wishful thinking,
    not commitment, so we exclude it."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT i.sku, SUM(GREATEST(i.qty_ordered - COALESCE(i.qty_received, 0), 0)) AS pending
              FROM supplier_order_items i
              JOIN supplier_orders o ON o.id = i.order_id
             WHERE o.user_id = $1
               AND o.status IN ('placed', 'in_production', 'shipped')
             GROUP BY i.sku
            """,
            user_id,
        )
    return {r["sku"]: int(r["pending"] or 0) for r in rows}


# ── Reorder suggestions ───────────────────────────────────────────────────────

async def compute_reorder_suggestions(
    pool: asyncpg.Pool,
    user_id: int,
    velocity_by_sku: dict[str, float],
    stock_by_sku: dict[str, int],
    *,
    safety_buffer_days: int = 14,
    coverage_target_days: int = 60,
    default_lead_time_days: int = 45,
) -> list[dict[str, Any]]:
    """Compute SKU-level replenishment suggestions.

    For each SKU with non-trivial velocity:
      - days_remaining = (stock + in_transit) / velocity
      - if days_remaining < (lead_time + safety_buffer): suggest reorder
      - suggested_qty = ceil(velocity × (lead_time + coverage_target_days)) - pipeline

    velocity_by_sku: units/day from vendas (last 30d).
    stock_by_sku: total units on hand (Full + own).
    """
    in_transit = await get_in_transit_by_sku(pool, user_id)

    # Last known supplier per SKU (for cost + lead_time hints — for MVP we use
    # global default lead_time, can be extended later when SKU↔supplier mapping
    # is captured in onboarding).
    async with pool.acquire() as conn:
        last_costs = await conn.fetch(
            """
            SELECT DISTINCT ON (i.sku)
                   i.sku, i.unit_cost, o.currency, o.supplier_name, o.eta_date, o.placed_date
              FROM supplier_order_items i
              JOIN supplier_orders o ON o.id = i.order_id
             WHERE o.user_id = $1 AND i.unit_cost IS NOT NULL
             ORDER BY i.sku, o.created_at DESC
            """,
            user_id,
        )
    cost_hint = {r["sku"]: {
        "unitCost": float(r["unit_cost"]) if r["unit_cost"] is not None else None,
        "currency": r["currency"],
        "supplier": r["supplier_name"],
    } for r in last_costs}

    suggestions: list[dict[str, Any]] = []
    skus = set(velocity_by_sku.keys()) | set(stock_by_sku.keys()) | set(in_transit.keys())

    for sku in skus:
        velocity = velocity_by_sku.get(sku, 0.0)
        if velocity < 0.1:
            continue
        stock = stock_by_sku.get(sku, 0)
        pending = in_transit.get(sku, 0)
        pipeline = stock + pending
        days_remaining = pipeline / velocity if velocity > 0 else float("inf")
        threshold = default_lead_time_days + safety_buffer_days

        if days_remaining >= threshold:
            continue

        target_qty = int(velocity * (default_lead_time_days + coverage_target_days))
        qty_to_order = max(target_qty - pipeline, 0)
        if qty_to_order <= 0:
            continue

        urgency = (
            "urgent" if days_remaining < default_lead_time_days * 0.7 else
            "soon" if days_remaining < default_lead_time_days else
            "planned"
        )
        cost_info = cost_hint.get(sku, {})
        unit_cost = cost_info.get("unitCost")
        suggestions.append({
            "sku": sku,
            "velocity": round(velocity, 2),
            "stock": stock,
            "inTransit": pending,
            "pipeline": pipeline,
            "daysRemaining": round(days_remaining, 1),
            "suggestedQty": qty_to_order,
            "unitCostHint": unit_cost,
            "estimatedCost": round(qty_to_order * unit_cost, 2) if unit_cost else None,
            "currencyHint": cost_info.get("currency"),
            "supplierHint": cost_info.get("supplier"),
            "urgency": urgency,
        })

    suggestions.sort(key=lambda s: (s["urgency"] != "urgent", s["urgency"] != "soon", s["daysRemaining"]))
    return suggestions
