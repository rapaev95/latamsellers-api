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
from v2.parsers.publicidade import PublicidadeRow
from v2.parsers.stock_full import StockFullSku, load_all_stock_full
from v2.services.projects import ProjectResolver
from v2.parsers.text_cells import excel_scalar_to_clean_str


def _normalize_mlb(raw: str | None) -> str:
    """Produce 'MLB{digits}' form for join keys. Mirrors ml_quality.normalize_item_id
    but inlined to avoid cross-service import chain."""
    s = str(raw or "").strip().upper()
    if not s:
        return ""
    numeric = s[3:] if s.startswith("MLB") else s
    if "." in numeric:
        head, _, tail = numeric.partition(".")
        if tail.strip("0") == "":
            numeric = head
        else:
            return ""
    if not numeric.isdigit():
        return ""
    return f"MLB{numeric}"


# Поля из Dados Fiscais, пробрасываемые в каждую запись продукта для UI.
# Значения берутся из sku_catalog (загружается один раз в начале aggregate).
_DADOS_FISCAIS_PRODUCT_FIELDS = (
    "ncm", "origem_type", "peso_liquido_kg", "peso_bruto_kg",
    "ean", "csosn_venda", "descricao_nfe",
)


def _empty_dados_fiscais_fields() -> dict[str, Any]:
    return {k: None for k in _DADOS_FISCAIS_PRODUCT_FIELDS}


def _item_id_json(mlb: str | None) -> str | None:
    """Strip Excel float tail from MLB for JSON / UI."""
    s = excel_scalar_to_clean_str(mlb or "")
    return s or None

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
    publicidade_rows: Iterable[PublicidadeRow] | None = None,
    current_prices_map: dict[str, float] | None = None,
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

    # Sku_catalog index — подтягивает NCM/Origem/Peso/EAN/CSOSN из Dados Fiscais.
    # Безопасно относительно отсутствия каталога (возвращает {}).
    try:
        from v2.legacy.sku_catalog import load_catalog, normalize_sku as _nsku
        _catalog_by_sku: dict[str, dict[str, Any]] = {}
        for _item in load_catalog():
            _sk = _nsku(str(_item.get("sku") or ""))
            if _sk:
                _catalog_by_sku[_sk] = _item
    except Exception:
        _catalog_by_sku = {}

    def _fiscal_fields_for(sku: str) -> dict[str, Any]:
        item = _catalog_by_sku.get((sku or "").upper())
        if not item:
            return _empty_dados_fiscais_fields()
        return {k: item.get(k) for k in _DADOS_FISCAIS_PRODUCT_FIELDS}

    def _unit_cost_for(sku: str) -> float:
        """COGS from sku_catalog. Populated by Dados Fiscais sync (custo_brl) OR
        manual entry in /finance/sku-mapping. 0.0 if neither — UI flags this."""
        item = _catalog_by_sku.get((sku or "").upper())
        if not item:
            return 0.0
        raw = item.get("unit_cost_brl")
        if raw is None:
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _extra_fixed_cost_for(sku: str) -> float:
        """Manual fixed cost per unit (embalagem, mão de obra, custos extras)
        entered by the seller in the UI. Stored alongside unit_cost_brl in
        sku_catalog. Default 0 — only subtracts when explicitly set."""
        item = _catalog_by_sku.get((sku or "").upper())
        if not item:
            return 0.0
        raw = item.get("extra_fixed_cost_brl")
        if raw is None:
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    # Project-level fixed costs (Aluguel, Fulfillment from PnL) allocated per
    # unit by total project deliveries. Cached per call so we hit compute_pnl
    # at most once per project.
    from datetime import date
    if isinstance(days, int):
        _pnl_period = (date.today() - timedelta(days=int(days)), date.today())
    elif isinstance(days, str) and days.isdigit():
        _pnl_period = (date.today() - timedelta(days=int(days)), date.today())
    else:
        _pnl_period = (date(2000, 1, 1), date.today())
    _project_alloc_cache: dict[str, tuple[float, float, float]] = {}

    def _project_fixed_per_unit(project_name: str) -> tuple[float, float, float]:
        """Returns (aluguel_per_unit, fulfillment_per_unit, das_rate) for project.

        das_rate is the effective Simples Nacional rate (Anexo I/II/III with
        RBT12 bracket applied), as a decimal (e.g. 0.047 for 4.7%). Used to
        compute per-sale DAS at the current selling price for unit econ.
        """
        key = (project_name or "").strip()
        if not key:
            return (0.0, 0.0, 0.045)
        cached = _project_alloc_cache.get(key)
        if cached is not None:
            return cached
        try:
            from v2.legacy.finance import compute_pnl
            pnl = compute_pnl(key, _pnl_period)
        except Exception:
            _project_alloc_cache[key] = (0.0, 0.0, 0.045)
            return _project_alloc_cache[key]
        aluguel_total = 0.0
        fulfillment_total = 0.0
        for line in (getattr(pnl, "operating_expenses", None) or []):
            label_lower = (getattr(line, "label", "") or "").lower()
            amount = float(getattr(line, "amount_brl", 0) or 0)
            if "aluguel" in label_lower:
                aluguel_total += amount
            elif "fulfillment" in label_lower or "доставка" in label_lower:
                fulfillment_total += amount
        units_total = max(int(getattr(pnl, "vendas_count", 0) or 0), 1)
        tax_info = getattr(pnl, "tax_info", None) or {}
        try:
            das_rate = float(tax_info.get("effective_pct", 4.5)) / 100.0
        except (TypeError, ValueError):
            das_rate = 0.045
        result = (
            aluguel_total / units_total,
            fulfillment_total / units_total,
            das_rate,
        )
        _project_alloc_cache[key] = result
        return result

    # Aggregate ads investment per MLB over the selected period. Publicidade
    # export rows have `desde`/`ate` (period they cover) and `investimento` (R$).
    # We sum all rows whose `ate` falls after the cutoff.
    ads_by_mlb: dict[str, float] = {}
    if publicidade_rows is not None:
        for pr in publicidade_rows:
            if cutoff_ms > 0:
                pr_ms = int(datetime(pr.ate.year, pr.ate.month, pr.ate.day).timestamp() * 1000)
                if pr_ms < cutoff_ms:
                    continue
            mlb_key = _normalize_mlb(pr.mlb)
            if not mlb_key:
                continue
            ads_by_mlb[mlb_key] = ads_by_mlb.get(mlb_key, 0.0) + pr.investimento

    source_rows: Iterable[VendasRow] = vendas_rows if vendas_rows is not None else load_all_vendas()
    filenames: list[str] = (
        vendas_filenames
        if vendas_filenames is not None
        else [f.name for f in list_vendas_files()]
    )

    # Snooze list accepts either a SKU or an MLB id — TG "Ignorar SKU" button
    # passes item_id (MLB) since paused-item notices don't carry the SKU.
    # Pre-normalize once so the per-row check is O(1) for both forms.
    snoozed_mlbs: set[str] = set()
    for s in snoozed_skus:
        norm = _normalize_mlb(s)
        if norm:
            snoozed_mlbs.add(norm)

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
        # Also skip if the row's MLB is in the snooze list (TG "Ignorar SKU" path).
        if snoozed_mlbs and _normalize_mlb(row.mlb) in snoozed_mlbs:
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

        # Ads cost per-unit — sum of investimento across ALL campaigns for this
        # MLB in the selected period, divided by units sold.
        mlb_key = _normalize_mlb(a.mlb)
        ads_total = ads_by_mlb.get(mlb_key, 0.0) if mlb_key else 0.0
        ad_pu = (ads_total / a.units) if a.units > 0 else 0.0
        # COGS — from Dados Fiscais sync (custo_brl) or manual /finance/sku-mapping.
        unit_cost = _unit_cost_for(a.sku)
        # Manual extra fixed cost per unit (embalagem, mão de obra, etc.) —
        # entered inline in the products dashboard.
        extra_fixed = _extra_fixed_cost_for(a.sku)
        # Project-level fixed costs allocated per unit (Aluguel + Fulfillment
        # from PnL) + DAS rate. Same per-unit value across SKUs in the same
        # project. Cache hit on repeat calls.
        project_for_pnl = (resolver.resolve(a.sku, a.mlb) if resolver else "") or ""
        aluguel_pu, fulfillment_pu, das_rate = _project_fixed_per_unit(project_for_pnl)
        margin = (
            avg_price - comm_pu - ship_pu - refund_pu - ad_pu - storage_pu
            - unit_cost - extra_fixed - aluguel_pu - fulfillment_pu
        )
        margin_pct = (margin / avg_price * 100) if avg_price > 0 else 0.0

        # ── Unit economics: contribution margin at CURRENT listing price ──────
        # Source of truth = ml_user_items.price (live), passed in via
        # current_prices_map. Falls back to historical avg_price when missing.
        # Variable costs only — Aluguel + ad_pu (overhead allocation) are
        # explicitly excluded so the seller can decide on a single sale
        # without fixed-cost-per-unit volatility.
        current_price = avg_price
        if current_prices_map:
            cp = current_prices_map.get(a.mlb)
            if cp is not None:
                try:
                    cp_f = float(cp)
                    if cp_f > 0:
                        current_price = cp_f
                except (TypeError, ValueError):
                    pass
        commission_rate = (comm_pu / avg_price) if avg_price > 0 else 0.167
        ml_fee_now = current_price * commission_rate
        das_now = current_price * das_rate
        # Variable per-sale / per-unit costs only. Aluguel and ad_pu
        # (advertising allocation) are FIXED — excluded from CM.
        variable_cost = (
            unit_cost
            + ml_fee_now
            + ship_pu          # envios subsídio (per shipment, fixed amount)
            + storage_pu       # armazenagem per unit per day
            + refund_pu        # return shipping per refund
            + fulfillment_pu   # fulfillment cost per shipment
            + das_now          # DAS scales with current price
            + extra_fixed      # manual extra cost per unit
        )
        contribution = current_price - variable_cost
        contribution_pct = (contribution / current_price * 100) if current_price > 0 else 0.0
        # ROI = profit_per_unit / cost_per_unit. Cost = COGS + ad spend (what
        # the seller actually puts at risk). Returns 0 if no cost basis yet.
        cost_basis = unit_cost + ad_pu
        roi = (margin / cost_basis * 100) if cost_basis > 0 else 0.0

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
            "itemId": _item_id_json(a.mlb),
            "units": a.units,
            "revenue": a.revenue,
            "avgPrice": avg_price,
            "commPerUnit": comm_pu,
            "shipPerUnit": ship_pu,
            "adPerUnit": ad_pu,
            "storagePerUnit": storage_pu,
            "refundPerUnit": refund_pu,
            "unitCost": unit_cost,
            "extraFixedCost": extra_fixed,
            "aluguelPerUnit": aluguel_pu,
            "fulfillmentPerUnit": fulfillment_pu,
            # Live listing price + CM at current price (not historical avg).
            # Used by /escalar/promotions Unit Econ panel for promo decisions.
            "currentPrice": current_price,
            "commissionRate": round(commission_rate, 4),
            "dasRate": round(das_rate, 4),
            "contributionMargin": round(contribution, 2),
            "contributionMarginPct": round(contribution_pct, 2),
            "margin": margin,
            "marginPct": margin_pct,
            "roi": roi,
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
            **_fiscal_fields_for(a.sku),
        })

    # Include SKUs that sit in the warehouse but didn't sell in the period.
    # Without this the on-screen "ESTOQUE TOTAL" undercounts real inventory
    # (the UI sums currentStock across the products list). These show as
    # category C with zero revenue — exactly what "dead stock" should look like.
    sold_skus = set(by_sku.keys())
    for sku, sf in (stock_full_map or {}).items():
        if sku in sold_skus or sku in snoozed_skus or sf.total <= 0:
            continue
        if snoozed_mlbs and _normalize_mlb(sf.mlb) in snoozed_mlbs:
            continue
        proj_resolved = resolver.resolve(sku, sf.mlb) if resolver else ""
        if project_lc and proj_resolved.lower() != project_lc:
            continue
        st = storage_map.get(sku)
        dominant_ship = None
        idle_unit_cost = _unit_cost_for(sku)
        products.append({
            "sku": sku,
            "title": sf.title or (st.produto if st else sku),
            "itemId": _item_id_json(sf.mlb or (st.mlb if st else "") or ""),
            "units": 0,
            "revenue": 0.0,
            "avgPrice": 0.0,
            "commPerUnit": 0.0,
            "shipPerUnit": 0.0,
            "adPerUnit": 0.0,
            "storagePerUnit": 0.0,
            "refundPerUnit": 0.0,
            "unitCost": idle_unit_cost,
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
            **_fiscal_fields_for(sku),
        })

    # ABC classification (Pareto 80/95)
    total_revenue = sum(p["revenue"] for p in products)
    cum = 0.0
    for p in products:
        p["revenuePct"] = (p["revenue"] / total_revenue * 100) if total_revenue > 0 else 0.0
        cum += p["revenuePct"]
        p["cumulativePct"] = cum
        p["abcGrade"] = "A" if cum <= 80 else "B" if cum <= 95 else "C"

    # Dropdown «проект»: не только ключи из `projects`, но и фактические resolve()
    # по продажам / складу / каталогу — иначе при пустом `projects` в БД список пустой.
    project_set: set[str] = set()
    if resolver:
        project_set.update(resolver.project_names)
        for a in by_sku.values():
            proj = resolver.resolve(a.sku, a.mlb)
            if proj and str(proj).upper() != "NAO_CLASSIFICADO":
                project_set.add(proj)
        for sku, sf in (stock_full_map or {}).items():
            if sku in snoozed_skus:
                continue
            proj = resolver.resolve(sku, (sf.mlb or "").strip())
            if proj and str(proj).upper() != "NAO_CLASSIFICADO":
                project_set.add(proj)
        for it in resolver.catalog.values():
            if not isinstance(it, dict):
                continue
            proj = (it.get("project") or "").strip()
            if proj and proj.upper() != "NAO_CLASSIFICADO":
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
