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
) -> dict[str, Any]:
    """Pure aggregation, so it can be tested without a database.

    `allowed` is the scope's item set; None means the whole cabinet.
    `visits` maps item_id → [{date: 'YYYY-MM-DD', total: n}, ...].
    """
    cur = _Window(today - timedelta(days=days - 1), today)
    prev = _Window(cur.start - timedelta(days=days), cur.start - timedelta(days=1))

    def ok(item_id: str) -> bool:
        return allowed is None or item_id in allowed

    zero = lambda: {"revenue": 0.0, "orders": 0, "units": 0, "cancelled": 0, "returns": 0}  # noqa: E731
    tot = {"cur": zero(), "prev": zero()}
    daily: dict[date, dict[str, float]] = {
        d: {"revenue": 0.0, "orders": 0, "units": 0} for d in cur.days()
    }
    per_item: dict[str, dict[str, Any]] = {}
    order_items: dict[str, set[str]] = {}

    for o in orders:
        day = _brt_day(o.get("date_created"))
        if day is None:
            continue
        bucket = "cur" if cur.has(day) else "prev" if prev.has(day) else None
        lines = [
            (_norm(it.get("mlb")), it) for it in _as_list(o.get("items"))
        ]
        lines = [(iid, it) for iid, it in lines if iid and ok(iid)]
        if not lines:
            continue
        order_items[str(o.get("order_id"))] = {iid for iid, _ in lines}
        if bucket is None:
            continue
        t = tot[bucket]
        if (o.get("status") or "") == "cancelled":
            t["cancelled"] += 1
            continue
        t["orders"] += 1
        for iid, it in lines:
            qty = int(float(it.get("quantity") or 0))
            rev = float(it.get("unit_price") or 0) * qty
            t["units"] += qty
            t["revenue"] += rev
            if bucket == "cur":
                daily[day]["units"] += qty
                daily[day]["revenue"] += rev
                pi = per_item.setdefault(iid, {
                    "itemId": iid, "title": it.get("title"), "sku": it.get("seller_sku"),
                    "revenue": 0.0, "units": 0, "orders": 0,
                })
                pi["revenue"] += rev
                pi["units"] += qty
                pi["orders"] += 1
        if bucket == "cur":
            daily[day]["orders"] += 1

    for c in claims:
        if (c.get("type") or "") != "returns":
            continue
        day = _brt_day(c.get("date_created"))
        if day is None:
            continue
        bucket = "cur" if cur.has(day) else "prev" if prev.has(day) else None
        if bucket is None:
            continue
        enriched = _as_dict(c.get("enriched"))
        ids = {
            _norm(o.get("item_id"))
            for r in _as_list(enriched.get("returns"))
            for o in _as_list(_as_dict(r).get("orders"))
        }
        ids.discard("")
        if not ids:
            ids = order_items.get(str(c.get("resource_id") or ""), set())
        if any(ok(i) for i in ids) and ids:
            tot[bucket]["returns"] += 1
            if bucket == "cur":
                for i in ids:
                    if i in per_item:
                        per_item[i]["returns"] = per_item[i].get("returns", 0) + 1

    # Visits: ML keeps 30 days. A window that reaches further back is unknown.
    history_start = today - timedelta(days=VISITS_HISTORY_DAYS - 1)
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
        gross = t["orders"] + t["cancelled"]
        return {
            "revenue": round(t["revenue"], 2),
            "orders": t["orders"],
            "units": t["units"],
            "avgTicket": round(t["revenue"] / t["orders"], 2) if t["orders"] else None,
            "cancelled": t["cancelled"],
            "cancelledPct": _pct(t["cancelled"], gross),
            "returns": t["returns"],
            "returnsPct": _pct(t["returns"], t["orders"]),
            "visits": v,
            "conversionPct": _pct(t["orders"], v),
        }

    k_cur, k_prev = kpis(tot["cur"], v_cur), kpis(tot["prev"], v_prev)
    change = {
        k: _delta(k_cur[k], k_prev[k])
        for k in ("revenue", "orders", "units", "visits", "conversionPct", "avgTicket")
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


async def load_overview(pool, scope, *, days: int) -> dict[str, Any]:
    from v2.services import ml_account_health as health_svc
    from v2.services import ml_orders as ml_orders_svc
    from v2.services import ml_user_claims as claims_svc
    from v2.services import ml_visits as visits_svc

    owner = scope.owner_id
    today = datetime.now(BRT).date()
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
        task = asyncio.create_task(_refresh_orders_bg(pool, owner, min(60, 2 * days)))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)

    out = build_overview(
        orders=[dict(r) for r in orders],
        claims=[dict(r) for r in claims],
        visits={k: v.get("daily") or [] for k, v in visits_map.items()},
        items_meta={_norm(r["item_id"]): dict(r) for r in items},
        allowed=scope.item_ids,
        today=today,
        days=days,
    )
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
    }
    return out
