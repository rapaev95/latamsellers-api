"""Finance screen of one project: money in, money out, money on the way.

Built per project on purpose. The Mercado Pago account is the company's and
holds every partner's money (docs/BUSINESS-MODEL.md: turnover is not
income), so its balance is never shown as a project's balance. What a
project sees is its own flow:

  - sales: its order lines (cancelled orders excluded);
  - ML sale fees on those lines;
  - Product Ads spend: each campaign's daily cost times the share of that
    campaign's spend that went to the project's listings;
  - fixed costs: the project's monthly fixed costs and rent, prorated to
    the period;
  - released / to receive: Mercado Pago payments of the project's orders,
    by money_release_date. Mixed orders count by the project's share.

When Mercado Pago does not answer, released / to receive are None, never 0.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from v2.services.escalar_overview import BRT, _as_list, _brt_day, _norm

log = logging.getLogger(__name__)

MP = "https://api.mercadopago.com"
_MP_CACHE: dict[int, tuple[float, Optional[list[dict]]]] = {}
MP_TTL = 600.0
FIXED_LABELS = {
    "salaries": ("Equipe / Operacionais", "Команда"),
    "utilities": ("Contas e serviços", "Коммунальные и сервисы"),
    "software": ("Software / Ferramentas", "Софт и сервисы"),
    "outros": ("Outros", "Прочее"),
    "aluguel": ("Aluguel da operação", "Аренда"),
}


async def _mp_payments(token: str, owner_id: int, since: datetime) -> Optional[list[dict]]:
    """Approved payments of the account since `since`, cached for 10 minutes."""
    hit = _MP_CACHE.get(owner_id)
    if hit and time.monotonic() - hit[0] < MP_TTL:
        return hit[1]
    out: list[dict] = []
    params = {
        "sort": "date_created", "criteria": "desc", "range": "date_created",
        "begin_date": since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000-00:00"),
        "end_date": "NOW", "status": "approved", "limit": 100,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            for offset in range(0, 2000, 100):
                r = await http.get(
                    f"{MP}/v1/payments/search", params={**params, "offset": offset},
                    headers={"Authorization": f"Bearer {token}"},
                )
                if r.status_code != 200:
                    log.warning("mp payments search owner=%s status=%s body=%s",
                                owner_id, r.status_code, r.text[:200])
                    _MP_CACHE[owner_id] = (time.monotonic(), None)
                    return None
                results = (r.json() or {}).get("results") or []
                out.extend(results)
                if len(results) < 100:
                    break
    except Exception as err:  # noqa: BLE001
        log.warning("mp payments search owner=%s failed: %s", owner_id, err)
        return None
    _MP_CACHE[owner_id] = (time.monotonic(), out)
    return out


def _fixed_costs(owner_id: int, projects: list[str]) -> dict[str, float]:
    """Monthly fixed costs (+ rent in BRL) summed over the given projects."""
    from v2.legacy import config as legacy_config
    from v2.legacy import db_storage as legacy_db

    before = legacy_db._current_user_id()
    try:
        legacy_db.set_current_user_id(owner_id)
        all_projects = legacy_config.load_projects() or {}
    finally:
        legacy_db.set_current_user_id(before)
    total: dict[str, float] = {}
    for pid in projects:
        p = all_projects.get(pid) or {}
        saved = p.get("fixed_costs_monthly") or {}
        for cat in legacy_config.FIXED_COST_CATEGORIES:
            try:
                v = max(0.0, float((saved or {}).get(cat) or 0))
            except (TypeError, ValueError):
                v = 0.0
            if v:
                total[cat] = total.get(cat, 0.0) + v
        try:
            rent = float(p.get("aluguel_mensal") or 0)
        except (TypeError, ValueError):
            rent = 0.0
        if rent > 0:
            total["aluguel"] = total.get("aluguel", 0.0) + rent
    return total


async def load_project_finance(pool, scope, *, days: int = 30) -> dict[str, Any]:
    from v2.services import ml_oauth as ml_oauth_svc
    from v2.storage import ads_storage

    owner = scope.owner_id
    today = datetime.now(BRT).date()
    cur_start = today - timedelta(days=days - 1)
    prev_start = cur_start - timedelta(days=days)
    since_dt = datetime.combine(prev_start, datetime.min.time(), tzinfo=BRT)
    allowed = scope.item_ids

    def ok(iid: str) -> bool:
        return allowed is None or iid in allowed

    await ads_storage.ensure_schema(pool)
    async with pool.acquire() as conn:
        orders = await conn.fetch(
            """
            SELECT order_id, date_created, status, items, total_amount
              FROM ml_user_orders WHERE user_id = $1 AND date_created >= $2
            """,
            owner, since_dt - timedelta(days=45),
        )
        ad_rows = await conn.fetch(
            "SELECT campaign_id, upper(item_id) AS item_id, metrics FROM ml_ad_ads WHERE user_id = $1",
            owner,
        )
        daily_rows = await conn.fetch(
            """
            SELECT campaign_id, date, metrics FROM ml_ad_campaign_metrics_daily
             WHERE user_id = $1 AND date >= $2
            """,
            owner, prev_start,
        )
        thumbs = await conn.fetch(
            "SELECT upper(item_id) AS item_id, thumbnail FROM ml_user_items WHERE user_id = $1", owner,
        )
    thumb = {r["item_id"]: r["thumbnail"] for r in thumbs}

    # ── orders: the project's share of each ──
    share_by_order: dict[str, float] = {}
    entries: list[dict[str, Any]] = []
    sales = {"cur": 0.0, "prev": 0.0}
    fees = {"cur": 0.0, "prev": 0.0}
    daily: dict[date, dict[str, float]] = {
        cur_start + timedelta(days=i): {"sales": 0.0, "ads": 0.0, "released": 0.0}
        for i in range(days)
    }
    for o in orders:
        lines = _as_list(o["items"])
        mine = [li for li in lines if ok(_norm(li.get("mlb")))]
        if not mine:
            continue
        rev_all = sum(float(li.get("unit_price") or 0) * float(li.get("quantity") or 0) for li in lines)
        rev = sum(float(li.get("unit_price") or 0) * float(li.get("quantity") or 0) for li in mine)
        fee = sum(float(li.get("sale_fee") or 0) * float(li.get("quantity") or 0) for li in mine)
        oid = str(o["order_id"])
        share_by_order[oid] = (rev / rev_all) if rev_all else 1.0
        day = _brt_day(o["date_created"])
        if day is None or day < prev_start:
            continue
        bucket = "cur" if day >= cur_start else "prev"
        cancelled = (o["status"] or "") == "cancelled"
        if not cancelled:
            sales[bucket] += rev
            fees[bucket] += fee
            if bucket == "cur":
                daily[day]["sales"] += rev
        if bucket == "cur":
            first = mine[0]
            entries.append({
                "date": day.isoformat(), "kind": "sale",
                "title": f"Venda #{oid}", "subtitle": first.get("title"),
                "thumbnail": thumb.get(_norm(first.get("mlb"))),
                "amount": round(rev, 2), "status": "cancelled" if cancelled else "paid",
            })
            if fee and not cancelled:
                entries.append({
                    "date": day.isoformat(), "kind": "fee",
                    "title": "Tarifa de venda", "subtitle": f"Pedido #{oid}",
                    "amount": -round(fee, 2), "status": "done",
                })

    # ── Product Ads: campaign daily cost × the project's share of it ──
    camp_total: dict[int, float] = {}
    camp_mine: dict[int, float] = {}
    for a in ad_rows:
        m = a["metrics"]
        m = json.loads(m) if isinstance(m, str) else (m or {})
        cost = float(m.get("cost") or 0)
        cid = int(a["campaign_id"] or 0)
        camp_total[cid] = camp_total.get(cid, 0.0) + cost
        if ok(a["item_id"]):
            camp_mine[cid] = camp_mine.get(cid, 0.0) + cost
    ads = {"cur": 0.0, "prev": 0.0}
    ads_by_day: dict[date, float] = {}
    for r in daily_rows:
        cid = int(r["campaign_id"])
        share = 1.0 if allowed is None else (
            camp_mine.get(cid, 0.0) / camp_total[cid] if camp_total.get(cid) else 0.0
        )
        if not share:
            continue
        m = r["metrics"]
        m = json.loads(m) if isinstance(m, str) else (m or {})
        cost = float(m.get("cost") or 0) * share
        d = r["date"]
        if d > today:
            continue
        bucket = "cur" if d >= cur_start else "prev"
        ads[bucket] += cost
        if bucket == "cur":
            ads_by_day[d] = ads_by_day.get(d, 0.0) + cost
            if d in daily:
                daily[d]["ads"] += cost
    for d, cost in sorted(ads_by_day.items()):
        if cost >= 0.01:
            entries.append({
                "date": d.isoformat(), "kind": "ads", "title": "Investimento em anúncios",
                "subtitle": "Product Ads", "amount": -round(cost, 2), "status": "done",
            })

    # ── fixed costs, prorated to the period ──
    projects = sorted(p for p in scope.projects if p != "NAO_CLASSIFICADO")
    monthly = _fixed_costs(owner, projects) if projects else {}
    factor = days / 30.0
    fixed = [
        {"key": k, "label": FIXED_LABELS.get(k, (k, k)), "monthly": round(v, 2),
         "period": round(v * factor, 2)}
        for k, v in sorted(monthly.items(), key=lambda kv: -kv[1])
    ]
    for f in fixed:
        entries.append({
            "date": today.isoformat(), "kind": "fixed", "title": "Custo fixo",
            "subtitle": f["label"][0], "amount": -f["period"], "status": "done",
        })

    # ── Mercado Pago: released and to receive ──
    released: Optional[dict[str, float]] = None
    receivable: Optional[dict[str, float]] = None
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, owner)
        payments = await _mp_payments(token, owner, since_dt - timedelta(days=45))
    except ml_oauth_svc.MLRefreshError:
        payments = None
    if payments is not None:
        released = {"cur": 0.0, "prev": 0.0}
        receivable = {"today": 0.0, "week": 0.0, "later": 0.0}
        for p in payments:
            oid = str(((p.get("order") or {}).get("id")) or "")
            share = share_by_order.get(oid)
            if share is None:
                continue
            net = float(((p.get("transaction_details") or {}).get("net_received_amount")) or 0) * share
            rel = _brt_day(p.get("money_release_date"))
            if not rel:
                continue
            if (p.get("money_release_status") or "") == "released":
                if rel >= cur_start:
                    released["cur"] += net
                    if rel in daily:
                        daily[rel]["released"] += net
                    entries.append({
                        "date": rel.isoformat(), "kind": "release", "title": "Repasse liberado",
                        "subtitle": f"Pedido #{oid}", "amount": round(net, 2), "status": "released",
                    })
                elif rel >= prev_start:
                    released["prev"] += net
            else:
                key = "today" if rel <= today else "week" if rel <= today + timedelta(days=7) else "later"
                receivable[key] += net

    def chg(c: float, p: float) -> Optional[float]:
        return round((c - p) / p * 100, 1) if p else None

    entries.sort(key=lambda e: (e["date"], e["kind"] != "sale"), reverse=True)
    r2 = lambda v: round(v, 2)  # noqa: E731
    return {
        "period": {"from": cur_start.isoformat(), "to": today.isoformat(), "days": days},
        "projects": projects,
        "sales": {"total": r2(sales["cur"]), "change": chg(sales["cur"], sales["prev"])},
        "fees": {"total": r2(fees["cur"]), "change": chg(fees["cur"], fees["prev"])},
        "ads": {
            "total": r2(ads["cur"]), "change": chg(ads["cur"], ads["prev"]),
            "daily": [{"date": d.isoformat(), "value": r2(daily[d]["ads"])} for d in sorted(daily)],
        },
        "fixedCosts": {
            "items": fixed,
            "totalPeriod": r2(sum(f["period"] for f in fixed)),
            "totalMonthly": r2(sum(f["monthly"] for f in fixed)),
        },
        "released": None if released is None else {
            "total": r2(released["cur"]), "change": chg(released["cur"], released["prev"]),
        },
        "receivable": None if receivable is None else {
            **{k: r2(v) for k, v in receivable.items()},
            "total": r2(sum(receivable.values())),
        },
        # What stays with the project before cost of goods: sales − fees − ads − fixed.
        "result": r2(sales["cur"] - fees["cur"] - ads["cur"] - sum(f["period"] for f in fixed)),
        "daily": [
            {"date": d.isoformat(), "sales": r2(v["sales"]), "ads": r2(v["ads"]),
             "released": r2(v["released"]) if released is not None else None}
            for d, v in sorted(daily.items())
        ],
        "entries": entries[:500],
    }
