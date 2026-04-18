"""ABC analysis aggregator.

Combines:
  - vendas rows (units, revenue, commission, shipping, refunds, ad-flag)
  - armazenagem (currentStock, days_in_stock → salesPerDay)
  - project resolver (SKU → project)
  - snoozed SKUs (excluded from totals)

Mirrors super-calculator-app/lib/escalar/vendas-loader.ts:loadVendasAndAggregate().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from v2.parsers.vendas_ml import VendasRow, list_vendas_files, load_all_vendas, shipping_mode_key
from v2.parsers.armazenagem import StorageData, load_all_armazenagem
from v2.parsers.stock_full import StockFullSku, load_all_stock_full
from v2.services.projects import ProjectResolver

LEAD_TIME = 30
BATCH_SIZE = 100
LIBERACAO = 5


@dataclass
class _Acc:
    sku: str
    title: str
    mlb: str
    units: int = 0
    revenue: float = 0.0
    commission: float = 0.0
    shipping: float = 0.0
    return_shipping: float = 0.0
    ad_orders: int = 0
    total_orders: int = 0
    cancelled_orders: int = 0
    shipping_modes: dict[str, int] = field(default_factory=dict)


def _is_cancelled(status: str) -> bool:
    s = (status or "").lower()
    return "cancel" in s or "reembols" in s


def aggregate(
    *,
    days: int | str = 30,
    project: str = "",
    snoozed_skus: set[str] | None = None,
    resolver: ProjectResolver | None = None,
    vendas_rows: Iterable[VendasRow] | None = None,
    storage_map: dict[str, StorageData] | None = None,
    stock_full_map: dict[str, StockFullSku] | None = None,
    vendas_filenames: list[str] | None = None,
) -> dict[str, Any]:
    """Build full ABC summary. Returns dict with `products` and `meta`.

    Data sources:
      - `vendas_rows`, `vendas_filenames` — per-user vendas (DB mode).
      - `storage_map` — armazenagem (storage-cost CSV, fallback stock source).
      - `stock_full_map` — authoritative Stock Full inventory; overrides
        armazenagem for `currentStock` when present for a SKU.
    Omitted args fall back to the shared filesystem loaders (legacy `fs` mode).
    """
    snoozed_skus = snoozed_skus or set()
    project_lc = project.strip().lower()

    cutoff_ms = 0
    if days != "all":
        try:
            n = int(days)
            cutoff_ms = int((datetime.now() - timedelta(days=n)).timestamp() * 1000)
        except (ValueError, TypeError):
            cutoff_ms = 0

    if storage_map is None:
        storage_map = load_all_armazenagem()
    if stock_full_map is None:
        stock_full_map = load_all_stock_full()

    source_rows: Iterable[VendasRow] = vendas_rows if vendas_rows is not None else load_all_vendas()
    filenames: list[str] = (
        vendas_filenames
        if vendas_filenames is not None
        else [f.name for f in list_vendas_files()]
    )

    rows: list[VendasRow] = []
    min_date = float("inf")
    max_date = 0
    total_seen = 0
    for row in source_rows:
        total_seen += 1
        if cutoff_ms > 0 and row.date_ms > 0 and row.date_ms < cutoff_ms:
            continue
        if project_lc and resolver:
            proj = resolver.resolve(row.sku, row.mlb).lower()
            if proj != project_lc:
                continue
        if row.sku in snoozed_skus:
            continue
        if row.date_ms > 0:
            min_date = min(min_date, row.date_ms)
            max_date = max(max_date, row.date_ms)
        rows.append(row)

    by_sku: dict[str, _Acc] = {}
    for r in rows:
        acc = by_sku.get(r.sku)
        if acc is None:
            acc = _Acc(sku=r.sku, title=r.title or r.sku, mlb=r.mlb)
            by_sku[r.sku] = acc
        if not acc.title and r.title:
            acc.title = r.title
        if not acc.mlb and r.mlb:
            acc.mlb = r.mlb
        if _is_cancelled(r.status):
            acc.cancelled_orders += 1
        else:
            acc.units += r.units
            acc.revenue += r.receita
            acc.commission += r.tarifa_venda
            acc.shipping += r.tarifa_envio
            acc.total_orders += 1
            if r.ads:
                acc.ad_orders += 1
        acc.return_shipping += r.custo_troca
        sm = shipping_mode_key(r.ship_mode)
        if sm:
            acc.shipping_modes[sm] = acc.shipping_modes.get(sm, 0) + 1

    products: list[dict[str, Any]] = []
    accs = sorted([a for a in by_sku.values() if a.units > 0], key=lambda x: -x.revenue)
    for a in accs:
        avg_price = a.revenue / a.units if a.units > 0 else 0.0
        comm_pu = a.commission / a.units if a.units > 0 else 0.0
        ship_pu = a.shipping / a.units if a.units > 0 else 0.0
        refund_pu = a.return_shipping / a.units if a.units > 0 else 0.0  # only return shipping (cash-out)

        st = storage_map.get(a.sku)
        storage_pu = (st.custos_acumulados / a.units) if (st and a.units > 0) else 0.0
        # Stock source priority:
        #   1. stock_full.xlsx  — authoritative ML inventory (weekly export)
        #   2. armazenagem CSV  — last non-empty daily cell (approximation)
        sf = stock_full_map.get(a.sku) if stock_full_map else None
        if sf is not None:
            current_stock = sf.total
        else:
            current_stock = st.current_stock if st else 0
        sales_per_day = (a.units / st.days_in_stock) if (st and st.days_in_stock > 0) else 0.0
        days_of_stock = (current_stock / sales_per_day) if sales_per_day > 0 else 0.0

        ad_pu = 0.0
        unit_cost = 0.0
        margin = avg_price - comm_pu - ship_pu - refund_pu - ad_pu - storage_pu - unit_cost
        margin_pct = (margin / avg_price * 100) if avg_price > 0 else 0.0

        dominant_ship = max(a.shipping_modes.items(), key=lambda x: x[1])[0] if a.shipping_modes else None
        ad_pct = (a.ad_orders / a.total_orders * 100) if a.total_orders > 0 else 0.0
        denom = a.total_orders + a.cancelled_orders
        return_pct = (a.cancelled_orders / denom * 100) if denom > 0 else 0.0

        if current_stock <= 0:
            reorder_status = "crit"
            buffer = 0.0
        else:
            buffer = days_of_stock - (LEAD_TIME + LIBERACAO) if sales_per_day > 0 else 999.0
            if buffer <= 0:
                reorder_status = "crit"
            elif buffer <= 7:
                reorder_status = "warn"
            else:
                reorder_status = "ok"

        order_by_date = None
        if sales_per_day > 0 and buffer < 999:
            order_by_date = (datetime.now() + timedelta(days=max(0, buffer))).strftime("%Y-%m-%d")

        full_cycle = LEAD_TIME + (BATCH_SIZE / sales_per_day if sales_per_day > 0 else 0) + LIBERACAO
        turns_per_year = (365 / full_cycle) if full_cycle > 0 else 0.0

        products.append({
            "sku": a.sku,
            "title": a.title,
            "itemId": a.mlb or None,
            "units": a.units,
            "revenue": a.revenue,
            "avgPrice": avg_price,
            "commPerUnit": comm_pu,
            "shipPerUnit": ship_pu,
            "adPerUnit": ad_pu,
            "storagePerUnit": storage_pu,
            "refundPerUnit": refund_pu,
            "unitCost": unit_cost,
            "margin": margin,
            "marginPct": margin_pct,
            "roi": 0.0,
            "abcGrade": "C",
            "revenuePct": 0.0,
            "cumulativePct": 0.0,
            "currentStock": current_stock,
            "salesPerDay": sales_per_day,
            "daysOfStock": days_of_stock,
            "leadTime": LEAD_TIME,
            "batchSize": BATCH_SIZE,
            "liberacao": LIBERACAO,
            "fullCycle": full_cycle,
            "turnsPerYear": turns_per_year,
            "annualRoi": 0.0,
            "reorderStatus": reorder_status,
            "orderByDate": order_by_date,
            "shippingMode": dominant_ship,
            "project": (resolver.resolve(a.sku, a.mlb) if resolver else "") or None,
            "returnPct": return_pct,
            "adPct": ad_pct,
        })

    # Include SKUs that sit in the warehouse but didn't sell in the period.
    # Without this the on-screen "ESTOQUE TOTAL" undercounts real inventory
    # (the UI sums currentStock across the products list). These show as
    # category C with zero revenue — exactly what "dead stock" should look like.
    sold_skus = set(by_sku.keys())
    for sku, sf in (stock_full_map or {}).items():
        if sku in sold_skus or sku in snoozed_skus or sf.total <= 0:
            continue
        proj_resolved = resolver.resolve(sku, sf.mlb) if resolver else ""
        if project_lc and proj_resolved.lower() != project_lc:
            continue
        st = storage_map.get(sku)
        dominant_ship = None
        products.append({
            "sku": sku,
            "title": sf.title or (st.produto if st else sku),
            "itemId": sf.mlb or (st.mlb if st else None) or None,
            "units": 0,
            "revenue": 0.0,
            "avgPrice": 0.0,
            "commPerUnit": 0.0,
            "shipPerUnit": 0.0,
            "adPerUnit": 0.0,
            "storagePerUnit": 0.0,
            "refundPerUnit": 0.0,
            "unitCost": 0.0,
            "margin": 0.0,
            "marginPct": 0.0,
            "roi": 0.0,
            "abcGrade": "C",
            "revenuePct": 0.0,
            "cumulativePct": 0.0,
            "currentStock": sf.total,
            "salesPerDay": 0.0,
            "daysOfStock": 0.0,
            "leadTime": LEAD_TIME,
            "batchSize": BATCH_SIZE,
            "liberacao": LIBERACAO,
            "fullCycle": 0.0,
            "turnsPerYear": 0.0,
            "annualRoi": 0.0,
            "reorderStatus": "crit",  # dead stock → liquidate candidate
            "orderByDate": None,
            "shippingMode": dominant_ship,
            "project": proj_resolved or None,
            "returnPct": 0.0,
            "adPct": 0.0,
        })

    # ABC classification (Pareto 80/95)
    total_revenue = sum(p["revenue"] for p in products)
    cum = 0.0
    for p in products:
        p["revenuePct"] = (p["revenue"] / total_revenue * 100) if total_revenue > 0 else 0.0
        cum += p["revenuePct"]
        p["cumulativePct"] = cum
        p["abcGrade"] = "A" if cum <= 80 else "B" if cum <= 95 else "C"

    project_set: set[str] = set(resolver.project_names) if resolver else set()
    if not project_set:
        for a in by_sku.values():
            proj = resolver.resolve(a.sku, a.mlb) if resolver else ""
            if proj:
                project_set.add(proj)

    period_from = datetime.fromtimestamp(min_date / 1000).strftime("%Y-%m-%d") if max_date > 0 else None
    period_to = datetime.fromtimestamp(max_date / 1000).strftime("%Y-%m-%d") if max_date > 0 else None

    return {
        "products": products,
        "meta": {
            "files": filenames,
            "totalRows": len(rows),
            "uniqueSales": len(rows),
            "totalRevenue": total_revenue,
            "totalUnits": sum(a.units for a in by_sku.values()),
            "storageSkus": len(stock_full_map) if stock_full_map else len(storage_map),
            "stockSource": "stock_full" if stock_full_map else ("armazenagem" if storage_map else "none"),
            "stockFullSkus": len(stock_full_map) if stock_full_map else 0,
            "armazenagemSkus": len(storage_map) if storage_map else 0,
            "storageFiles": 0,  # TODO: surface from armazenagem loader
            "projects": sorted(project_set),
            "days": days,
            "periodFrom": period_from,
            "periodTo": period_to,
            "snoozedSkus": sorted(snoozed_skus),
        },
    }
