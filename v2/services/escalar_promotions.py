"""Escalar "Promoções": the seller's promotions screen, shaped like ML's own.

Three blocks, one read:
  - performance of the listings that are in a promotion (visits, sales,
    conversion, units, gross sales vs the previous period) — the overview
    narrowed to those listings;
  - tasks: how many listings can join a lightning deal, a Full-stock
    promotion, a co-funded one;
  - the table: per listing, the promotions it is in or can join, with the
    final price and what the seller keeps after ML's sale fee.

Offers come from `ml_user_promotions` (ML /seller-promotions, refreshed by
the promotions job). Offers whose window already ended are dropped: ML keeps
returning old DEAL candidates long after they closed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from v2.services import escalar_overview as overview_svc
from v2.services import ml_access

BRT = timezone(timedelta(hours=-3))
CALENDAR_DAYS = 5
LIGHTNING = {"LIGHTNING", "SMART_LIGHTNING"}
# Promotions ML co-funds: the seller's discount is partly paid by ML.
CO_FUNDED = {"SMART", "SMART_LIGHTNING", "PRICE_MATCHING", "MARKETPLACE_CAMPAIGN"}
ACTIVE = {"started", "pending"}


def _parse_dt(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _offer(o: dict, fee_ratio: Optional[float]) -> dict[str, Any]:
    original = _f(o.get("original_price"))
    # LIGHTNING/DEAL candidates carry a range; ML suggests a price inside it.
    final = _f(o.get("deal_price")) or _f(o.get("suggested_discounted_price")) or _f(o.get("price"))
    if final == 0:
        final = None
    discount = round(original - final, 2) if original and final else None
    stock = o.get("stock") if isinstance(o.get("stock"), dict) else None
    return {
        "id": o.get("id"),
        "type": o.get("type"),
        "subType": o.get("sub_type"),
        "name": o.get("name"),
        "status": o.get("status"),
        "offerId": o.get("offer_id") or o.get("ref_id"),
        "originalPrice": original,
        "finalPrice": final,
        "discount": discount,
        "discountPct": round(_f(o.get("discount_percentage")) or 0, 1) or None,
        "minPrice": _f(o.get("min_discounted_price")),
        "maxPrice": _f(o.get("max_discounted_price")),
        "suggestedPrice": _f(o.get("suggested_discounted_price")),
        "meliPct": _f(o.get("meli_percentage")),
        "sellerPct": _f(o.get("seller_percentage")),
        "stock": stock,
        "start": o.get("start_date"),
        "finish": o.get("finish_date"),
        # What stays after ML's sale fee. Shipping is not in it — ML charges
        # it per shipment, not per listing price.
        "youReceive": round(final * (1 - fee_ratio), 2) if final and fee_ratio is not None else None,
        "coFunded": (o.get("type") in CO_FUNDED) or bool(_f(o.get("meli_percentage"))),
    }


async def load_promotions(pool, scope, *, days: int = 30) -> dict[str, Any]:
    from v2.services import ml_user_promotions as promo_svc

    owner = scope.owner_id
    now = datetime.now(timezone.utc)
    await promo_svc.ensure_schema(pool)
    cached = await promo_svc.get_cached(pool, owner)

    async with pool.acquire() as conn:
        items = await conn.fetch(
            """
            SELECT item_id, title, thumbnail, permalink, status, price,
                   available_quantity, listing_type_id,
                   (shipping_json->>'free_shipping')::boolean AS free_shipping
              FROM ml_user_items
             WHERE user_id = $1
            """,
            owner,
        )
        fees = await conn.fetch(
            """
            SELECT upper(li->>'mlb') AS item_id,
                   sum((li->>'sale_fee')::numeric * (li->>'quantity')::numeric)
                     / nullif(sum((li->>'unit_price')::numeric * (li->>'quantity')::numeric), 0) AS ratio
              FROM ml_user_orders o, jsonb_array_elements(o.items) li
             WHERE o.user_id = $1 AND o.date_created >= now() - interval '90 days'
               AND li->>'sale_fee' IS NOT NULL
             GROUP BY 1
            """,
            owner,
        )
    meta = {str(r["item_id"]).upper(): dict(r) for r in items}
    fee_by_item = {r["item_id"]: float(r["ratio"]) for r in fees if r["ratio"] is not None}

    rows: list[dict[str, Any]] = []
    promoted: set[str] = set()
    tasks = {"lightning": set(), "fullStock": set(), "coFunded": set(), "deals": set()}
    calendar: dict[str, dict[str, dict]] = {}
    days_ahead = [(now.astimezone(BRT) + timedelta(days=i)).date() for i in range(CALENDAR_DAYS)]

    for raw_id, offers in (cached.get("items") or {}).items():
        iid = overview_svc._norm(raw_id)
        if not scope.allows_read(iid):
            continue
        m = meta.get(iid) or {}
        if m.get("status") == "closed":
            continue
        live = []
        for o in offers:
            if o.get("status") in ("finished", "expired"):
                continue
            finish = _parse_dt(o.get("finish_date"))
            if finish and finish < now:
                continue
            live.append(_offer(o, fee_by_item.get(iid)))
        if not live:
            continue

        live.sort(key=lambda x: (x["status"] not in ACTIVE, -(x["finalPrice"] or 0)))
        active = [x for x in live if x["status"] in ACTIVE]
        candidates = [x for x in live if x["status"] == "candidate"]
        if active:
            promoted.add(iid)
        for c in candidates:
            if c["type"] in LIGHTNING:
                tasks["lightning"].add(iid)
            if c["type"] == "UNHEALTHY_STOCK":
                tasks["fullStock"].add(iid)
            if c["coFunded"]:
                tasks["coFunded"].add(iid)
            if c["type"] in ("DEAL", "SELLER_CAMPAIGN"):
                tasks["deals"].add(iid)

        for x in live:
            start, finish = _parse_dt(x["start"]), _parse_dt(x["finish"])
            if not start and not finish:
                continue
            for d in days_ahead:
                s_ok = start is None or start.astimezone(BRT).date() <= d
                f_ok = finish is None or finish.astimezone(BRT).date() >= d
                if s_ok and f_ok:
                    key = x["name"] or x["type"]
                    slot = calendar.setdefault(d.isoformat(), {}).setdefault(
                        key, {"name": key, "type": x["type"], "active": 0, "candidates": 0},
                    )
                    slot["active" if x["status"] in ACTIVE else "candidates"] += 1

        rows.append({
            "itemId": iid,
            "title": m.get("title"),
            "thumbnail": m.get("thumbnail"),
            "permalink": m.get("permalink"),
            "status": m.get("status"),
            "price": _f(m.get("price")),
            "stock": m.get("available_quantity"),
            "listingType": m.get("listing_type_id"),
            "freeShipping": m.get("free_shipping"),
            "feeRatio": fee_by_item.get(iid),
            "active": active,
            "candidates": candidates,
            "canEdit": scope.allows_write(iid),
        })

    rows.sort(key=lambda r: (not r["active"], -len(r["candidates"]), r["title"] or ""))
    performance = await overview_svc.load_overview(pool, scope, days=days, only_items=promoted)
    eligible = len(rows)
    return {
        "performance": {
            "period": performance["period"],
            "kpis": performance["kpis"],
            "change": performance["change"],
            "listings": len(promoted),
        },
        "tasks": {
            "participatingPct": round(len(promoted) / eligible * 100) if eligible else None,
            "eligible": eligible,
            "lightning": len(tasks["lightning"]),
            "fullStock": len(tasks["fullStock"]),
            "coFunded": len(tasks["coFunded"]),
            "deals": len(tasks["deals"]),
        },
        "calendar": [
            {"date": d.isoformat(), "promotions": sorted(
                calendar.get(d.isoformat(), {}).values(),
                key=lambda s: (-s["active"], -s["candidates"]),
            )}
            for d in days_ahead
        ],
        "listings": rows,
        "fetchedAt": cached.get("fetchedAt"),
    }
