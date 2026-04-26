"""Per-item margin cache.

Pattern: TEST → DB → CACHE.
  - Heavy compute (compute_pnl + per-item allocation) runs in a nightly batch
    job that materialises every active item's margin into ml_item_margin_cache.
  - The TG promotion dispatcher and any UI endpoint read from the cache only —
    a single SELECT, microseconds of latency, zero CPU on the hot path.
  - Hypothetical-price scenarios (after-promo margin) are derived on the fly
    from the cached components without re-running compute_pnl.

Allocation rules (consistent with the OPiU dashboard):
  - Publicidade / Armazenagem / Fulfillment → by units_sku / units_project
  - DAS / Aluguel                            → by revenue_sku / revenue_project
  - tarifa_venda (ML fees)                   → per-row from vendas, summed
  - COGS (unit_cost_brl × qty)               → from sku_catalog (Dados Fiscais)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_item_margin_cache (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  period_months INT NOT NULL DEFAULT 3,
  payload JSONB NOT NULL,
  computed_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, item_id, period_months)
);
CREATE INDEX IF NOT EXISTS idx_ml_margin_user ON ml_item_margin_cache(user_id);
CREATE INDEX IF NOT EXISTS idx_ml_margin_stale ON ml_item_margin_cache(computed_at);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── Read path (hot — used by promo dispatcher) ────────────────────────────────

async def get_cached_margin(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    period_months: int = 3,
) -> Optional[dict]:
    """Single SELECT — returns the cached payload (with computed_at injected)
    or None when the item hasn't been computed yet."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT payload, computed_at
              FROM ml_item_margin_cache
             WHERE user_id = $1 AND item_id = $2 AND period_months = $3
            """,
            user_id, item_id.upper(), period_months,
        )
    if not row:
        return None
    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(payload, dict):
        return None
    payload["computed_at"] = row["computed_at"].isoformat() if row["computed_at"] else None
    return payload


def apply_hypothetical_price(
    cached: dict,
    hypothetical_price: float,
) -> dict:
    """Pure function: re-derive margin_pct / net_profit from cached components
    when revenue is scaled to a hypothetical unit price.

    Cogs and overhead shares stay the same (same units sold). Revenue and
    ml_fees scale linearly with price. This is good enough for a "what if"
    preview — exact ML fee structure is non-linear at the rate-card level
    but the linear approximation is within ~1pp for typical discount ranges.
    """
    if not cached or not cached.get("ok"):
        return cached or {}
    units = cached.get("units_sold") or 0
    revenue = float(cached.get("revenue") or 0)
    if units <= 0 or revenue <= 0:
        return dict(cached)
    avg_unit_price = revenue / units
    if avg_unit_price <= 0:
        return dict(cached)
    scale = float(hypothetical_price) / avg_unit_price
    new_revenue = revenue * scale
    new_ml_fees = float(cached.get("ml_fees") or 0) * scale
    cogs = cached.get("cogs")
    overhead = float(cached.get("overhead_total") or 0)
    if cogs is not None:
        new_profit = new_revenue - new_ml_fees - float(cogs) - overhead
        new_margin = round(new_profit / new_revenue * 100, 1) if new_revenue else None
    else:
        new_profit = None
        new_margin = None
    out = dict(cached)
    out["revenue"] = round(new_revenue, 2)
    out["ml_fees"] = round(new_ml_fees, 2)
    out["net_profit"] = round(new_profit, 2) if new_profit is not None else None
    out["margin_pct"] = new_margin
    out["hypothetical_price"] = float(hypothetical_price)
    return out


# ── Write path (cold — runs in nightly cron) ──────────────────────────────────

async def refresh_user_item_margins(
    pool: asyncpg.Pool,
    user_id: int,
    period_months: int = 3,
) -> dict:
    """Batch-compute and upsert margin for every active item of one user.

    Calls compute_pnl exactly once per project (cached internally inside
    legacy.reports), then per-item math is just DataFrame filter + scalar
    arithmetic. For ~200 items across 5 projects this is ~5-10s of CPU,
    fully amortised across all items.
    """
    return await asyncio.to_thread(_batch_refresh_sync, user_id, period_months)


def _period_for_months(months: int) -> tuple[date, date]:
    today = date.today()
    return (today - timedelta(days=30 * months), today)


def _batch_refresh_sync(user_id: int, period_months: int) -> dict:
    from ..legacy.db_storage import set_current_user_id, db_load
    from ..legacy.reports import load_vendas_ml_report
    from ..legacy.finance import compute_pnl
    import psycopg2

    set_current_user_id(user_id)
    period = _period_for_months(period_months)

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

    dsn = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")
    if not dsn:
        return {"ok": False, "error": "no_db_url"}

    item_ids: list[str] = []
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            "SELECT item_id FROM ml_user_items WHERE user_id = %s",
            (user_id,),
        )
        item_ids = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
    except Exception as err:  # noqa: BLE001
        log.warning("ml_user_items query failed for user=%s: %s", user_id, err)
        return {"ok": False, "error": "items_query_failed"}

    if not item_ids:
        return {"ok": True, "computed": 0, "items_total": 0, "projects": 0}

    pnl_cache: dict[str, Any] = {}
    payloads: list[tuple[str, str]] = []  # (item_id, payload_json)
    skipped_no_project = 0
    skipped_no_pnl = 0
    skipped_no_sales = 0

    for item_id in item_ids:
        try:
            sku, project, unit_cost = _lookup_sku_project_cost(item_id, df, cat_items)
            if not project:
                skipped_no_project += 1
                continue
            if project not in pnl_cache:
                try:
                    pnl_cache[project] = compute_pnl(project, period)
                except Exception as err:  # noqa: BLE001
                    log.warning("compute_pnl(%s) failed: %s", project, err)
                    pnl_cache[project] = None
            pnl = pnl_cache[project]
            if pnl is None:
                skipped_no_pnl += 1
                continue

            item_df = _filter_vendas_for_item(df, item_id, period)
            if item_df is None or len(item_df) == 0:
                skipped_no_sales += 1
                # Still write a "no sales" row so the dispatcher can show a
                # meaningful message instead of "calculando".
                payload = {
                    "ok": False, "error": "no_sales_in_period",
                    "project": project, "sku": sku, "period_months": period_months,
                }
                payloads.append((item_id.upper(), json.dumps(payload)))
                continue

            payload = _build_item_payload(
                pnl, item_df, project, sku, unit_cost, period_months,
            )
            payloads.append((item_id.upper(), json.dumps(payload)))
        except Exception as err:  # noqa: BLE001
            log.exception("margin compute %s failed: %s", item_id, err)

    if payloads:
        try:
            conn = psycopg2.connect(dsn)
            cur = conn.cursor()
            for item_id_norm, payload_json in payloads:
                cur.execute(
                    """
                    INSERT INTO ml_item_margin_cache
                      (user_id, item_id, period_months, payload, computed_at)
                    VALUES (%s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (user_id, item_id, period_months) DO UPDATE
                      SET payload = EXCLUDED.payload, computed_at = NOW()
                    """,
                    (user_id, item_id_norm, period_months, payload_json),
                )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as err:  # noqa: BLE001
            log.exception("margin upsert failed for user=%s: %s", user_id, err)
            return {"ok": False, "error": "upsert_failed"}

    return {
        "ok": True,
        "computed": len(payloads),
        "items_total": len(item_ids),
        "projects": len(pnl_cache),
        "skipped_no_project": skipped_no_project,
        "skipped_no_pnl": skipped_no_pnl,
        "skipped_no_sales": skipped_no_sales,
    }


# ── Building blocks ───────────────────────────────────────────────────────────

def _build_item_payload(
    pnl,
    item_df,
    project: str,
    sku: Optional[str],
    unit_cost: Optional[float],
    period_months: int,
) -> dict:
    """Project pnl + filtered item rows → payload dict shaped like the
    historical get_item_margin response so the renderer doesn't need to change."""
    revenue, ml_fees, units = _sum_item_basics(item_df)

    total_units = max(int(getattr(pnl, "vendas_delivered_count", 0) or 0), 1)
    total_revenue = max(
        float(getattr(pnl, "revenue_net", 0) or getattr(pnl, "revenue_gross", 0) or 0),
        1.0,
    )
    opex_lines = list(getattr(pnl, "operating_expenses", []) or [])
    opex_by_label = {
        str(getattr(line, "label", "")): float(getattr(line, "value", 0) or 0)
        for line in opex_lines
    }

    publicidade = _opex_pick(opex_by_label, ["publicidade"])
    armazenagem = _opex_pick(opex_by_label, ["armazenagem"])
    aluguel = _opex_pick(opex_by_label, ["aluguel"])
    das = _opex_pick(opex_by_label, ["das", "simples", "lucro"])
    fulfillment = _opex_pick(opex_by_label, ["fulfillment", "доставка"])

    units_share = units / total_units
    revenue_share = revenue / total_revenue

    publicidade_share = publicidade * units_share
    armazenagem_share = armazenagem * units_share
    aluguel_share = aluguel * revenue_share
    das_share = das * revenue_share
    fulfillment_share = fulfillment * units_share

    cogs = unit_cost * units if unit_cost is not None else None
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
        "period_months": period_months,
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
        "missing_cost": unit_cost is None,
    }


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
