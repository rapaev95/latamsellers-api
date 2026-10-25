"""Escalar home: what a project sold over a period, from live ML data.

The page the seller opens first. It answers "how is my project doing" with
the numbers ML itself shows — revenue, orders, units, cancellations,
returns, visits, conversion — and compares them with the period before.

Sources, all caches of the ML API under the cabinet owner:
  - ml_user_orders  — orders (webhook + periodic pulls), items as JSONB
  - ml_user_claims  — claims; type 'returns' is a return
  - ml_item_visits  — daily visits per listing, last 30 days only
  - ml_user_items   — title / thumbnail / status for the top list

Scope is the caller's `ml_access` scope: an order counts for a project only
through its listings, so a mixed order contributes just the project's lines.

A number we cannot know is None, never 0: visits for a previous period
older than the 30 days ML keeps, conversion without visits.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))
VISITS_HISTORY_DAYS = 30
# Orders older than this are pulled again from ML in the background.
ORDERS_STALE = timedelta(minutes=20)


def _as_list(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except ValueError:
            return []
    return []


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except ValueError:
            return {}
    return {}


def _norm(item_id: Any) -> str:
    s = str(item_id or "").strip().upper()
    if s and s.isdigit():
        s = "MLB" + s
    return s


@dataclass
class _Window:
    start: date   # inclusive, BRT
    end: date     # inclusive, BRT

    def has(self, d: date) -> bool:
        return self.start <= d <= self.end

    def days(self) -> list[date]:
        n = (self.end - self.start).days + 1
        return [self.start + timedelta(days=i) for i in range(n)]


def _brt_day(dt: Any) -> Optional[date]:
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(BRT).date()
    if isinstance(dt, str) and dt:
        try:
            return _brt_day(datetime.fromisoformat(dt.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _pct(part: Optional[float], whole: Optional[float]) -> Optional[float]:
    if part is None or not whole:
        return None
    return round(part / whole * 100, 2)


def _delta(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    """Relative change in %, None when there is nothing to compare with."""
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / prev * 100, 1)


def build_overview(
    *,
    orders: Iterable[dict],
    claims: Iterable[dict],
    visits: dict[str, list[dict]],
    items_meta: dict[str, dict],
    allowed: Optional[frozenset[str]],
    today: date,
    days: int,
    top_n: int = 10,
    visits_today: Optional[date] = None,
) -> dict[str, Any]:
    """Pure aggregation, so it can be tested without a database.

    `allowed` is the scope's item set; None means the whole cabinet.
    `visits` maps item_id → [{date: 'YYYY-MM-DD', total: n}, ...].
    `today` is the last day of the period; `visits_today` the real today,
    which is where ML's 30 days of visits end (defaults to `today`).
    """
    cur = _Window(today - timedelta(days=days - 1), today)
    prev = _Window(cur.start - timedelta(days=days), cur.start - timedelta(days=1))

    def ok(item_id: str) -> bool:
        return allowed is None or item_id in allowed

    # Сначала заказы проекта (только его строки), потом возвраты по ним.
    # Возврат относится к периоду своего заказа, а не к дате обращения —
    # иначе «заказы − отмены − возвраты = продано» не сходится в одном окне.
    scoped: list[dict[str, Any]] = []
    for o in orders:
        day = _brt_day(o.get("date_created"))
        if day is None:
            continue
        bucket = "cur" if cur.has(day) else "prev" if prev.has(day) else None
        if bucket is None:
            continue
        lines = []
        for it in _as_list(o.get("items")):
            iid = _norm(it.get("mlb"))
            if not iid or not ok(iid):
                continue
            qty = int(float(it.get("quantity") or 0))
            lines.append({
                "iid": iid, "qty": qty, "rev": float(it.get("unit_price") or 0) * qty,
                "title": it.get("title"), "sku": it.get("seller_sku"),
            })
        if lines:
            scoped.append({
                "id": str(o.get("order_id")), "day": day, "bucket": bucket,
                "cancelled": (o.get("status") or "") == "cancelled", "lines": lines,
            })

    # order_id → returned units per listing (None = whole order, ML gave no detail)
    returned: dict[str, Optional[dict[str, int]]] = {}
    for c in claims:
        if (c.get("type") or "") != "returns":
            continue
        enriched = _as_dict(c.get("enriched"))
        detail: dict[str, int] = {}
        order_ids: set[str] = set()
        for r in _as_list(enriched.get("returns")):
            for ro in _as_list(_as_dict(r).get("orders")):
                iid = _norm(ro.get("item_id"))
                if ro.get("order_id"):
                    order_ids.add(str(ro.get("order_id")))
                if iid:
                    q = ro.get("return_quantity") or ro.get("total_quantity")
                    detail[iid] = detail.get(iid, 0) + int(float(q or 1))
        if c.get("resource_id"):
            order_ids.add(str(c.get("resource_id")))
        for oid in order_ids:
            returned[oid] = detail or None

    zero = lambda: {  # noqa: E731
        "ordersGross": 0, "unitsGross": 0, "revenueGross": 0.0,
        "cancelled": 0, "returns": 0, "returnedUnits": 0, "returnedRevenue": 0.0,
        "orders": 0, "units": 0, "revenue": 0.0,
    }
    tot = {"cur": zero(), "prev": zero()}
    daily: dict[date, dict[str, float]] = {
        d: {"revenue": 0.0, "orders": 0, "units": 0} for d in cur.days()
    }
    per_item: dict[str, dict[str, Any]] = {}

    for o in scoped:
        t = tot[o["bucket"]]
        t["ordersGross"] += 1
        for ln in o["lines"]:
            t["unitsGross"] += ln["qty"]
            t["revenueGross"] += ln["rev"]
        if o["cancelled"]:
            t["cancelled"] += 1
            continue
        ret = returned.get(o["id"], False)
        if ret is not False:
            t["returns"] += 1
        sold_any = False
        for ln in o["lines"]:
            qty, rev = ln["qty"], ln["rev"]
            if ret is not False:
                back = qty if ret is None else min(qty, ret.get(ln["iid"], 0))
                share = back / qty if qty else 0
                t["returnedUnits"] += back
                t["returnedRevenue"] += rev * share
                qty, rev = qty - back, rev * (1 - share)
                if o["bucket"] == "cur" and back:
                    pi = per_item.setdefault(ln["iid"], {
                        "itemId": ln["iid"], "title": ln["title"], "sku": ln["sku"],
                        "revenue": 0.0, "units": 0, "orders": 0,
                    })
                    pi["returns"] = pi.get("returns", 0) + 1
            if qty <= 0:
                continue
            sold_any = True
            t["units"] += qty
            t["revenue"] += rev
            if o["bucket"] == "cur":
                daily[o["day"]]["units"] += qty
                daily[o["day"]]["revenue"] += rev
                pi = per_item.setdefault(ln["iid"], {
                    "itemId": ln["iid"], "title": ln["title"], "sku": ln["sku"],
                    "revenue": 0.0, "units": 0, "orders": 0,
                })
                pi["revenue"] += rev
                pi["units"] += qty
                pi["orders"] += 1
        if sold_any and ret is False:
            t["orders"] += 1
            if o["bucket"] == "cur":
                daily[o["day"]]["orders"] += 1

    # Visits: ML keeps 30 days. A window that reaches further back is unknown.
    history_start = (visits_today or today) - timedelta(days=VISITS_HISTORY_DAYS - 1)
    have_visits = any(ok(i) for i in visits)

    def visits_in(w: _Window) -> Optional[int]:
        if not have_visits or w.start < history_start:
            return None
        n = 0
        for iid, rows in visits.items():
            if not ok(iid):
                continue
            for r in rows:
                d = _brt_day(str(r.get("date") or "") + "T12:00:00+00:00")
                if d and w.has(d):
                    n += int(r.get("total") or 0)
        return n

    v_cur, v_prev = visits_in(cur), visits_in(prev)
    daily_visits: dict[date, int] = {}
    if have_visits:
        for iid, rows in visits.items():
            if not ok(iid):
                continue
            for r in rows:
                d = _brt_day(str(r.get("date") or "") + "T12:00:00+00:00")
                if d and cur.has(d):
                    daily_visits[d] = daily_visits.get(d, 0) + int(r.get("total") or 0)
                    if iid in per_item:
                        per_item[iid]["visits"] = per_item[iid].get("visits", 0) + int(r.get("total") or 0)

    def kpis(t: dict, v: Optional[int]) -> dict[str, Any]:
        """orders = ordersGross − cancelled − returns; units and revenue are
        what stayed sold. Conversion is the buyer's act: gross orders ÷ visits."""
        return {
            "ordersGross": t["ordersGross"],
            "unitsGross": t["unitsGross"],
            "revenueGross": round(t["revenueGross"], 2),
            "cancelled": t["cancelled"],
            "cancelledPct": _pct(t["cancelled"], t["ordersGross"]),
            "returns": t["returns"],
            "returnsPct": _pct(t["returns"], t["ordersGross"]),
            "returnedUnits": t["returnedUnits"],
            "returnedRevenue": round(t["returnedRevenue"], 2),
            "orders": t["orders"],
            "units": t["units"],
            "revenue": round(t["revenue"], 2),
            "avgTicket": round(t["revenue"] / t["orders"], 2) if t["orders"] else None,
            "visits": v,
            "conversionPct": _pct(t["ordersGross"], v),
        }

    k_cur, k_prev = kpis(tot["cur"], v_cur), kpis(tot["prev"], v_prev)
    change = {
        k: _delta(k_cur[k], k_prev[k])
        for k in ("revenue", "orders", "units", "ordersGross", "visits", "conversionPct", "avgTicket")
    }

    top = sorted(per_item.values(), key=lambda p: p["revenue"], reverse=True)[:top_n]
    for p in top:
        meta = items_meta.get(p["itemId"]) or {}
        p["title"] = meta.get("title") or p.get("title")
        p["thumbnail"] = meta.get("thumbnail")
        p["status"] = meta.get("status")
        p["revenue"] = round(p["revenue"], 2)
        p.setdefault("returns", 0)
        p["visits"] = p.get("visits") if have_visits else None
        p["conversionPct"] = _pct(p["orders"], p["visits"])
        p["units"] = int(p["units"])

    return {
        "period": {"from": cur.start.isoformat(), "to": cur.end.isoformat(), "days": days},
        "previousPeriod": {"from": prev.start.isoformat(), "to": prev.end.isoformat()},
        "kpis": k_cur,
        "previous": k_prev,
        "change": change,
        "daily": [
            {
                "date": d.isoformat(),
                "revenue": round(daily[d]["revenue"], 2),
                "orders": daily[d]["orders"],
                "units": daily[d]["units"],
                "visits": daily_visits.get(d) if have_visits else None,
            }
            for d in cur.days()
        ],
        "top": top,
    }


# ── loading ──────────────────────────────────────────────────────────────────

_refreshing: set[int] = set()
# The loop keeps only weak references to tasks; hold them until they finish.
_tasks: set[asyncio.Task] = set()


async def _refresh_orders_bg(pool, owner_id: int, days_back: int) -> None:
    from v2.services import ml_orders as ml_orders_svc

    try:
        await ml_orders_svc.refresh_for_period(pool, owner_id, days_back=days_back)
    except Exception as err:  # noqa: BLE001 — a background pull must not crash
        log.warning("overview orders refresh owner=%s failed: %s", owner_id, err)
    finally:
        _refreshing.discard(owner_id)


async def load_overview(
    pool, scope, *, days: int, only_items: Optional[Iterable[str]] = None,
    end: Optional[date] = None,
) -> dict[str, Any]:
    """`only_items` narrows the scope further — the promotions screen asks for
    the numbers of the listings that are in a promotion."""
    from v2.services import ml_account_health as health_svc
    from v2.services import ml_orders as ml_orders_svc
    from v2.services import ml_user_claims as claims_svc
    from v2.services import ml_visits as visits_svc

    owner = scope.owner_id
    real_today = datetime.now(BRT).date()
    # `end` — последний день выбранного периода; по умолчанию сегодня.
    today = min(end or real_today, real_today)
    since = datetime.combine(today - timedelta(days=2 * days), datetime.min.time(), tzinfo=BRT)

    await ml_orders_svc.ensure_schema(pool)
    await claims_svc.ensure_schema(pool)
    await visits_svc.ensure_schema(pool)
    async with pool.acquire() as conn:
        orders = await conn.fetch(
            """
            SELECT order_id, date_created, status, items
              FROM ml_user_orders
             WHERE user_id = $1 AND date_created >= $2
            """,
            owner, since,
        )
        last_order_fetch = await conn.fetchval(
            "SELECT max(fetched_at) FROM ml_user_orders WHERE user_id = $1", owner,
        )
        orders_from = await conn.fetchval(
            "SELECT min(date_created) FROM ml_user_orders WHERE user_id = $1", owner,
        )
        claims = await conn.fetch(
            """
            SELECT claim_id, resource_id, type, date_created, enriched
              FROM ml_user_claims
             WHERE user_id = $1 AND type = 'returns' AND date_created >= $2
            """,
            owner, since,
        )
        visits_map = await visits_svc.get_cached_map(conn, owner)
        items = await conn.fetch(
            "SELECT item_id, title, thumbnail, status FROM ml_user_items WHERE user_id = $1",
            owner,
        )

    # Orders arrive by webhook, which can miss some; pull the period from ML
    # in the background when the cache is old. This response uses the cache.
    stale = last_order_fetch is None or (
        datetime.now(timezone.utc) - last_order_fetch > ORDERS_STALE
    )
    if stale and owner not in _refreshing:
        _refreshing.add(owner)
        reach = (real_today - today).days + 2 * days
        task = asyncio.create_task(_refresh_orders_bg(pool, owner, min(60, reach)))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)

    out = build_overview(
        orders=[dict(r) for r in orders],
        claims=[dict(r) for r in claims],
        visits={k: v.get("daily") or [] for k, v in visits_map.items()},
        items_meta={_norm(r["item_id"]): dict(r) for r in items},
        allowed=(
            scope.item_ids if only_items is None
            else frozenset(_norm(i) for i in only_items if scope.allows_read(_norm(i)))
        ),
        today=today,
        days=days,
        visits_today=real_today,
    )
    # Живой кэш заказов держит ~60 дней. Если период начинается раньше —
    # заказы за него есть только в отчёте Vendas: считаем цепочку по нему.
    period_start = date.fromisoformat(out["period"]["from"])
    covered_from = _brt_day(orders_from) if orders_from else None
    out["funnel"] = None
    if covered_from is None or period_start < covered_from:
        projects = None if scope.item_ids is None else {p for p in scope.projects}
        out["funnel"] = await asyncio.to_thread(
            vendas_funnel_sync, owner, projects, period_start, today,
        )
        if out["funnel"] is not None:
            out["funnel"]["source"] = "vendas"
    health = await health_svc.get_cached(pool, owner)
    out["reputation"] = (health or {}).get("reputation")
    out["projects"] = sorted(scope.projects)
    out["sources"] = {
        "ordersFetchedAt": last_order_fetch.isoformat() if last_order_fetch else None,
        "ordersRefreshing": owner in _refreshing,
        "visitsFetchedAt": max(
            (v.get("fetchedAt") for v in visits_map.values() if v.get("fetchedAt")),
            default=None,
        ),
        "reputationFetchedAt": (health or {}).get("fetchedAt"),
        "ordersCoveredFrom": covered_from.isoformat() if covered_from else None,
    }
    return out


# ── funnel from the Vendas report (periods older than the live cache) ──────

_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _vendas_date(s: Any) -> Optional[date]:
    import re

    g = re.search(r"(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})", str(s))
    if not g or g.group(2).lower() not in _PT_MONTHS:
        return None
    try:
        return date(int(g.group(3)), _PT_MONTHS[g.group(2).lower()], int(g.group(1)))
    except ValueError:
        return None


def vendas_funnel_sync(owner_id: int, projects: Optional[set[str]], start: date, end: date) -> Optional[dict]:
    """Ordered − cancelled − returned − in transit = delivered, from the same
    Vendas rows the P&L reads, so «Продано» here equals P&L revenue_delivered.

    The live orders cache keeps about 60 days; for older periods this is the
    only source of orders. `projects` None = the whole cabinet.
    """
    import pandas as pd

    from v2.legacy import db_storage as legacy_db
    from v2.legacy.reports import load_vendas_ml_report

    before = legacy_db._current_user_id()
    try:
        legacy_db.set_current_user_id(owner_id)
        df = load_vendas_ml_report()
    finally:
        legacy_db.set_current_user_id(before)
    if df is None or df.empty:
        return None

    def num(v: Any) -> float:
        x = pd.to_numeric(v, errors="coerce")
        return 0.0 if pd.isna(x) else float(x)

    out = {k: {"count": 0, "units": 0, "revenue": 0.0}
           for k in ("ordered", "cancelled", "returned", "inTransit", "delivered")}
    states: dict[str, int] = {}
    for _, row in df.iterrows():
        if projects is not None and row.get("__project") not in projects:
            continue
        d = _vendas_date(row.get("Data da venda"))
        if d is None or d < start or d > end:
            continue
        rev = num(row.get("Receita por produtos (BRL)"))
        units = int(num(row.get("Unidades")))
        # Строка без выручки — не продажа: заголовок пакета (товары идут
        # строками под ним) или обмен («Troca…» — ML шлёт замену за R$ 0).
        # Иначе она висит «в пути» с R$ 0 и сбивает счёт заказов.
        if rev == 0:
            continue
        bucket = row.get("__bucket")
        estado = str(row.get("Estado") or "").lower()
        if bucket == "delivered":
            key = "delivered"
        elif bucket == "returned":
            key = "returned"
        elif "cancel" in estado:
            key = "cancelled"
        else:
            key = "inTransit"
        for k in ("ordered", key):
            out[k]["count"] += 1
            out[k]["units"] += units
            out[k]["revenue"] += rev
        if key == "inTransit":
            st = str(row.get("Estado") or "").strip()[:80]
            states[st] = states.get(st, 0) + 1
    for v in out.values():
        v["revenue"] = round(v["revenue"], 2)
    # Статусы ML у незавершённых — чтобы было видно, что именно «в пути».
    out["inTransit"]["states"] = sorted(states.items(), key=lambda kv: -kv[1])[:5]
    return out
