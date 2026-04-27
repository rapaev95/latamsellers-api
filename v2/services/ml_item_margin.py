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

    # ── PnL margin recompute (overhead allocation stays fixed in BRL terms) ──
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

    # ── Unit economics recompute (only variable costs scale with price) ──
    # Use stored RATES directly so the formula is independent of "scale" —
    # ml_fee_rate × hypothetical_price gives the right per-sale fee at any
    # price point (matching ML's actual % billing).
    unit_in = cached.get("unit") or {}
    new_unit: dict[str, Any] | None = None
    if unit_in:
        new_price = float(hypothetical_price)
        ml_fee_rate = float(unit_in.get("ml_fee_rate") or 0.167)
        new_ml_fee_pu = new_price * ml_fee_rate
        ful_pu = float(unit_in.get("fulfillment_per_sale") or 0)
        cogs_pu = unit_in.get("cogs_per_unit")
        armaz_pu = float(unit_in.get("armaz_per_unit") or 0)
        das_rate = float(unit_in.get("das_rate") or 0.045)
        new_das_pu = new_price * das_rate
        if cogs_pu is not None:
            new_var_cost = new_ml_fee_pu + ful_pu + float(cogs_pu) + new_das_pu + armaz_pu
            new_unit_profit = new_price - new_var_cost
            new_unit_margin = round(new_unit_profit / new_price * 100, 1) if new_price else None
        else:
            new_var_cost = None
            new_unit_profit = None
            new_unit_margin = None
        new_unit = dict(unit_in)
        new_unit.update({
            "current_price": round(new_price, 2),
            "avg_price": round(new_price, 2),  # legacy alias
            "ml_fee_per_unit": round(new_ml_fee_pu, 2),
            "das_per_unit": round(new_das_pu, 2),
            "variable_cost": round(new_var_cost, 2) if new_var_cost is not None else None,
            "profit_per_unit": round(new_unit_profit, 2) if new_unit_profit is not None else None,
            "margin_pct": new_unit_margin,
        })

    out = dict(cached)
    out["revenue"] = round(new_revenue, 2)
    out["ml_fees"] = round(new_ml_fees, 2)
    out["net_profit"] = round(new_profit, 2) if new_profit is not None else None
    out["margin_pct"] = new_margin
    if new_unit is not None:
        out["unit"] = new_unit
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
    current_prices: dict[str, float] = {}
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(
            "SELECT item_id, price FROM ml_user_items WHERE user_id = %s",
            (user_id,),
        )
        for row in cur.fetchall():
            item_ids.append(row[0])
            if row[1] is not None:
                current_prices[row[0]] = float(row[1])
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
                current_price=current_prices.get(item_id),
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
    current_price: Optional[float] = None,
) -> dict:
    """Project pnl + filtered item rows → payload dict shaped like the
    historical get_item_margin response so the renderer doesn't need to change.

    `current_price` is the live ml_user_items.price — used as the basis for
    unit economics so margin reflects today's listing price, not the
    historical average over the period (which can be lower if the item ran
    a discount mid-period)."""
    revenue, ml_fees, units = _sum_item_basics(item_df)

    # PnLReport exposes vendas_count (delivered+returned in period). The
    # field name `vendas_delivered_count` doesn't exist — using getattr with
    # default=0 silently set total_units=1 and over-allocated every share by
    # factor=item_units. ORGANIZADOR's 18.50 BRL armaz showed up as 4k.
    total_units = max(int(getattr(pnl, "vendas_count", 0) or 0), 1)
    total_revenue = max(
        float(getattr(pnl, "revenue_net", 0) or getattr(pnl, "revenue_gross", 0) or 0),
        1.0,
    )
    opex_lines = list(getattr(pnl, "operating_expenses", []) or [])
    # PnLLine field is `amount_brl`, not `value` — using the wrong attribute
    # silently zeroed every overhead line and inflated the cached margin.
    opex_by_label = {
        str(getattr(line, "label", "")): float(getattr(line, "amount_brl", 0) or 0)
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

    # ── Unit economics (variable costs only) ───────────────────────────────────
    # PnL margin includes fixed costs (Aluguel, etc.) — useful for accounting
    # but inflates the price floor for promo decisions. Unit economics treats
    # only per-sale / per-unit costs:
    #   - ML fee (% of price, scales with discount)
    #   - Fulfillment (per shipment, fixed amount)
    #   - COGS (per unit)
    #   - DAS (% of revenue, scales with discount)
    #   - Armazenagem (per unit, daily — small)
    # Aluguel / Publicidade-as-fixed are excluded.
    historical_avg_price = revenue / units if units > 0 else 0.0
    # Prefer the live listing price as the basis. Fall back to historical
    # average when ml_user_items has no row (rare — e.g. paused listing
    # before our cache caught up).
    price_basis = float(current_price) if current_price else historical_avg_price

    # ml_fee/unit is derived as a RATE (% of revenue) so it scales with the
    # current price — matching how ML actually charges per sale.
    ml_fee_rate = ml_fees / revenue if revenue > 0 else 0.167
    ml_fee_per_unit = price_basis * ml_fee_rate

    fulfillment_per_sale = fulfillment / total_units if total_units > 0 else 0.0
    armaz_per_unit = armazenagem / total_units if total_units > 0 else 0.0
    # DAS rate must come from tax_info.effective_pct (already considers RBT12
    # bracket + Anexo I/II/III). Dividing total DAS by revenue_net would
    # over-state the rate by ~1.5x because net excludes tarifa_venda /
    # cancelamentos while DAS is computed on bruto.
    tax_info = getattr(pnl, "tax_info", None) or {}
    das_effective_pct = tax_info.get("effective_pct")
    if das_effective_pct is not None:
        das_rate = float(das_effective_pct) / 100.0
    else:
        revenue_gross = float(getattr(pnl, "revenue_gross", 0) or 0)
        das_rate = (das / revenue_gross) if revenue_gross > 0 else 0.045
    das_per_unit = price_basis * das_rate
    cogs_per_unit = float(unit_cost) if unit_cost is not None else None

    if cogs_per_unit is not None:
        unit_variable_cost = (
            ml_fee_per_unit + fulfillment_per_sale + cogs_per_unit
            + das_per_unit + armaz_per_unit
        )
        unit_profit = price_basis - unit_variable_cost
        unit_margin_pct = round(unit_profit / price_basis * 100, 1) if price_basis else None
    else:
        unit_variable_cost = None
        unit_profit = None
        unit_margin_pct = None

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
        # Unit-economics block — variable per-sale / per-unit costs only.
        # `current_price` is what we're computing margin AT (live listing
        # price). `historical_avg_price` is the period's actual average —
        # kept for transparency but not used in the margin calculation.
        "unit": {
            "current_price": round(price_basis, 2),
            "historical_avg_price": round(historical_avg_price, 2),
            # Legacy alias (some readers may still look for `avg_price`).
            "avg_price": round(price_basis, 2),
            "ml_fee_per_unit": round(ml_fee_per_unit, 2),
            "ml_fee_rate": round(ml_fee_rate, 4),
            "fulfillment_per_sale": round(fulfillment_per_sale, 2),
            "cogs_per_unit": round(cogs_per_unit, 2) if cogs_per_unit is not None else None,
            "das_per_unit": round(das_per_unit, 2),
            "armaz_per_unit": round(armaz_per_unit, 2),
            "variable_cost": round(unit_variable_cost, 2) if unit_variable_cost is not None else None,
            "profit_per_unit": round(unit_profit, 2) if unit_profit is not None else None,
            "margin_pct": unit_margin_pct,
            "das_rate": round(das_rate, 4),
        },
    }


_ANUNCIO_COLS = (
    "# de anúncio", "# de anuncio", "#anúncio", "# anúncio",
    "anuncio_id", "MLB", "mlb", "ID",
)
_DATE_COLS = ("Data da venda", "Data de venda", "date_created", "Date")

# Portuguese natural-language date format used by Vendas ML CSVs:
#   "17 de abril de 2026 13:13 hs."
_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}
import re as _re_pt
_PT_DATE_RE = _re_pt.compile(r"(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})")


def _parse_pt_date(s):
    if s is None:
        return None
    m = _PT_DATE_RE.search(str(s))
    if not m:
        return None
    mn = _PT_MONTHS.get(m.group(2).lower())
    if not mn:
        return None
    try:
        return date(int(m.group(3)), mn, int(m.group(1)))
    except (ValueError, TypeError):
        return None
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
    anuncio_col = _first_col(df, _ANUNCIO_COLS)
    if not anuncio_col:
        return None
    sub = df[df[anuncio_col].astype(str).str.upper() == item_id.upper()]
    date_col = _first_col(df, _DATE_COLS)
    if date_col and len(sub) > 0:
        parsed = sub[date_col].map(_parse_pt_date)
        period_start, period_end = period
        mask = parsed.map(
            lambda d: d is not None and period_start <= d <= period_end
        )
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
