"""Per-item margin using the same opex formulas as the OPiU dashboard.

Reuses v2/legacy.compute_pnl + load_vendas_ml_report so margins shown in
TG promotion cards reconcile with the project-level PnL the seller sees
in the dashboard.

Allocation rules for project-level overhead (compute_pnl operating_expenses):
  - Publicidade        → by units_sku / units_project
  - Armazenagem (Full) → by units_sku / units_project
  - DAS (taxes)        → by revenue_sku / revenue_project
  - Aluguel            → by revenue_sku / revenue_project
  - Fulfillment        → by units_sku / units_project (per-shipment)
  - tarifa_venda (ML)  → already in vendas rows, summed per item
  - COGS               → unit_cost_brl × qty from sku_catalog (None if missing)

The legacy code is sync (pandas + psycopg2) and reads the user from a
contextvar (set_current_user_id). We wrap calls in asyncio.to_thread so
the cron's event loop isn't blocked on CSV parsing or PnL computation.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any, Optional

log = logging.getLogger(__name__)

# Tiny in-process TTL cache so dispatching N promotions for the same item
# doesn't recompute margin N times. compute_pnl already caches the heavy
# DataFrame & matrix internally; this just dedups the final assembly.
_MARGIN_CACHE: dict[tuple[int, str, int, Optional[float]], tuple[float, dict]] = {}
_MARGIN_TTL_SEC = 60.0


async def get_item_margin(
    user_id: int,
    item_id: str,
    months: int = 3,
    hypothetical_price: Optional[float] = None,
) -> dict[str, Any]:
    """Return margin breakdown for `item_id` over the last `months`.

    If `hypothetical_price` is given, revenue and ml_fees are scaled to
    simulate "what would margin look like if every unit sold at this price".
    Used to preview post-promotion margin in the TG card.
    """
    key = (user_id, item_id.upper(), months, hypothetical_price)
    now = time.monotonic()
    cached = _MARGIN_CACHE.get(key)
    if cached and now - cached[0] < _MARGIN_TTL_SEC:
        return cached[1]
    result = await asyncio.to_thread(
        _compute_margin_sync, user_id, item_id, months, hypothetical_price,
    )
    _MARGIN_CACHE[key] = (now, result)
    return result


def _period_for_months(months: int) -> tuple[date, date]:
    today = date.today()
    return (today - timedelta(days=30 * months), today)


def _compute_margin_sync(
    user_id: int,
    item_id: str,
    months: int,
    hypothetical_price: Optional[float],
) -> dict[str, Any]:
    from ..legacy.db_storage import set_current_user_id, db_load
    from ..legacy.reports import load_vendas_ml_report
    from ..legacy.finance import compute_pnl

    set_current_user_id(user_id)
    period = _period_for_months(months)

    try:
        df = load_vendas_ml_report()
    except Exception as err:  # noqa: BLE001
        log.warning("load_vendas_ml_report(user=%s) failed: %s", user_id, err)
        return {"ok": False, "error": "vendas_load_failed"}
    if df is None or len(df) == 0:
        return {"ok": False, "error": "no_vendas_data"}

    catalog = db_load("sku_catalog", user_id=user_id) or {}
    cat_items = catalog.get("items") if isinstance(catalog, dict) else []
    if not isinstance(cat_items, list):
        cat_items = []

    sku, project, unit_cost = _lookup_sku_project_cost(item_id, df, cat_items)
    if not project:
        return {
            "ok": False, "error": "no_project_match",
            "item_id": item_id, "sku": sku,
        }

    item_df = _filter_vendas_for_item(df, item_id, period)
    if item_df is None or len(item_df) == 0:
        return {
            "ok": False, "error": "no_sales_in_period",
            "project": project, "sku": sku, "period_months": months,
        }

    revenue, ml_fees, units = _sum_item_basics(item_df)
    if revenue <= 0 or units <= 0:
        return {
            "ok": False, "error": "zero_revenue_or_units",
            "project": project, "sku": sku,
        }

    try:
        pnl = compute_pnl(project, period)
    except Exception as err:  # noqa: BLE001
        log.warning("compute_pnl(%s) failed: %s", project, err)
        return {"ok": False, "error": "pnl_failed", "project": project}

    total_units = max(int(getattr(pnl, "vendas_delivered_count", 0) or 0), 1)
    total_revenue = max(
        float(getattr(pnl, "revenue_net", 0) or getattr(pnl, "revenue_gross", 0) or 0),
        1.0,
    )
    opex_lines = list(getattr(pnl, "operating_expenses", []) or [])
    opex_by_label = {str(getattr(line, "label", "")): float(getattr(line, "value", 0) or 0) for line in opex_lines}

    publicidade = _opex_pick(opex_by_label, ["publicidade"])
    armazenagem = _opex_pick(opex_by_label, ["armazenagem"])
    aluguel = _opex_pick(opex_by_label, ["aluguel"])
    das = _opex_pick(opex_by_label, ["das", "simples", "lucro"])
    fulfillment = _opex_pick(opex_by_label, ["fulfillment", "доставка"])

    units_share = units / total_units
    revenue_share = revenue / total_revenue

    if hypothetical_price is not None and units > 0:
        avg_unit_price = revenue / units
        if avg_unit_price > 0:
            scale = float(hypothetical_price) / avg_unit_price
            revenue = revenue * scale
            ml_fees = ml_fees * scale

    publicidade_share = publicidade * units_share
    armazenagem_share = armazenagem * units_share
    aluguel_share = aluguel * revenue_share
    das_share = das * revenue_share
    fulfillment_share = fulfillment * units_share

    cogs = unit_cost * units if unit_cost is not None else None
    missing_cost = unit_cost is None

    overhead_total = (
        publicidade_share + armazenagem_share + aluguel_share
        + das_share + fulfillment_share
    )
    if cogs is not None:
        net_profit = revenue - ml_fees - cogs - overhead_total
        margin_pct = round(net_profit / revenue * 100, 1) if revenue else None
    else:
        net_profit = None
        margin_pct = None

    return {
        "ok": True,
        "project": project,
        "sku": sku,
        "period_months": months,
        "period": {"from": period[0].isoformat(), "to": period[1].isoformat()},
        "units_sold": units,
        "revenue": round(revenue, 2),
        "ml_fees": round(ml_fees, 2),
        "cogs": round(cogs, 2) if cogs is not None else None,
        "publicidade_share": round(publicidade_share, 2),
        "armazenagem_share": round(armazenagem_share, 2),
        "aluguel_share": round(aluguel_share, 2),
        "das_share": round(das_share, 2),
        "fulfillment_share": round(fulfillment_share, 2),
        "overhead_total": round(overhead_total, 2),
        "net_profit": round(net_profit, 2) if net_profit is not None else None,
        "margin_pct": margin_pct,
        "unit_cost_brl": unit_cost,
        "missing_cost": missing_cost,
        "hypothetical_price": hypothetical_price,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

_ANUNCIO_COLS = ("#anúncio", "# anúncio", "anuncio_id", "MLB", "mlb", "ID")
_DATE_COLS = ("Data da venda", "Data de venda", "date_created", "Date")
_REVENUE_COLS = ("Receita por produtos (BRL)", "Receita por produtos", "Receita")
_FEE_COLS = ("Tarifa de venda e impostos (BRL)", "Tarifa de venda e impostos", "Tarifa")
_QTY_COLS = ("Unidades", "Quantidade", "Qty")
_SKU_COLS = ("SKU", "sku")


def _first_col(df, candidates: tuple[str, ...]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _lookup_sku_project_cost(
    item_id: str,
    df,
    cat_items: list[dict],
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """Returns (sku, project, unit_cost_brl). Tries sku_catalog by mlb first,
    then derives from vendas DataFrame (item_id → SKU → catalog)."""
    target = item_id.upper()

    for entry in cat_items:
        if str(entry.get("mlb") or "").upper() == target:
            cost = entry.get("unit_cost_brl") or entry.get("custo_brl")
            try:
                cost_f = float(cost) if cost is not None else None
            except (TypeError, ValueError):
                cost_f = None
            return (entry.get("sku"), entry.get("project"), cost_f)

    anuncio_col = _first_col(df, _ANUNCIO_COLS)
    sku_col = _first_col(df, _SKU_COLS)
    if not anuncio_col or not sku_col:
        return (None, None, None)

    matching = df[df[anuncio_col].astype(str).str.upper() == target]
    if len(matching) == 0:
        return (None, None, None)
    sku_val = str(matching[sku_col].iloc[0] or "").strip()
    if not sku_val:
        return (None, None, None)

    for entry in cat_items:
        if str(entry.get("sku") or "") == sku_val:
            cost = entry.get("unit_cost_brl") or entry.get("custo_brl")
            try:
                cost_f = float(cost) if cost is not None else None
            except (TypeError, ValueError):
                cost_f = None
            return (sku_val, entry.get("project"), cost_f)

    return (sku_val, None, None)


def _filter_vendas_for_item(df, item_id: str, period: tuple[date, date]):
    import pandas as pd
    anuncio_col = _first_col(df, _ANUNCIO_COLS)
    if not anuncio_col:
        return None
    sub = df[df[anuncio_col].astype(str).str.upper() == item_id.upper()]
    date_col = _first_col(df, _DATE_COLS)
    if date_col and len(sub) > 0:
        dt = pd.to_datetime(sub[date_col], errors="coerce", dayfirst=True)
        mask = (dt >= pd.Timestamp(period[0])) & (dt <= pd.Timestamp(period[1]))
        sub = sub[mask]
    return sub


def _sum_item_basics(item_df) -> tuple[float, float, int]:
    rev_col = _first_col(item_df, _REVENUE_COLS)
    fee_col = _first_col(item_df, _FEE_COLS)
    qty_col = _first_col(item_df, _QTY_COLS)
    revenue = float(item_df[rev_col].sum()) if rev_col else 0.0
    ml_fees = abs(float(item_df[fee_col].sum())) if fee_col else 0.0
    units = int(item_df[qty_col].sum()) if qty_col else len(item_df)
    return (revenue, ml_fees, units)


def _opex_pick(opex_by_label: dict, keywords: list[str]) -> float:
    for label, value in opex_by_label.items():
        lower = str(label).lower()
        for kw in keywords:
            if kw in lower:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
    return 0.0
