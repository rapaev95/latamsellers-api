"""Escalar "Meus anúncios": every listing of the caller's project in one row.

One aggregating read per screen (docs/PERFORMANCE.md): listing, live stock,
sales of the last 30 days, visits, quality score, Product Ads state and
promotions — each from its own ML cache under the cabinet owner, joined here
instead of the page firing one request per listing.

A listing without a cached value carries None for it (no visits fetched yet,
quality never computed, not in any campaign), never 0.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from v2.services import ml_access

BRT = timezone(timedelta(hours=-3))
SALES_DAYS = 30


def _num(v: Any) -> Optional[float]:
    return float(v) if v is not None else None


async def load_listings(pool, scope) -> dict[str, Any]:
    from v2.services import ml_orders as ml_orders_svc
    from v2.services import ml_quality as quality_svc
    from v2.services import ml_user_promotions as promo_svc
    from v2.services import ml_visits as visits_svc
    from v2.storage import ads_storage

    owner = scope.owner_id
    since = datetime.now(BRT) - timedelta(days=SALES_DAYS)

    await ml_orders_svc.ensure_schema(pool)
    await quality_svc.ensure_schema(pool)
    await visits_svc.ensure_schema(pool)
    await promo_svc.ensure_schema(pool)
    await ads_storage.ensure_schema(pool)

    async with pool.acquire() as conn:
        items = await conn.fetch(
            """
            SELECT item_id, title, thumbnail, permalink, status, price,
                   available_quantity, sold_quantity, listing_type_id, category_id,
                   raw->>'sub_status' AS sub_status_raw,
                   to_char(fetched_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_items
             WHERE user_id = $1
            """,
            owner,
        )
        sales = await conn.fetch(
            """
            SELECT upper(li->>'mlb') AS item_id,
                   count(DISTINCT o.order_id) AS orders,
                   sum((li->>'quantity')::numeric) AS units,
                   sum((li->>'quantity')::numeric * (li->>'unit_price')::numeric) AS revenue
              FROM ml_user_orders o, jsonb_array_elements(o.items) li
             WHERE o.user_id = $1 AND o.date_created >= $2
               AND coalesce(o.status, '') <> 'cancelled'
             GROUP BY 1
            """,
            owner, since,
        )
        quality = await conn.fetch(
            "SELECT item_id, score, level FROM ml_item_quality WHERE user_id = $1", owner,
        )
        visits = await conn.fetch(
            "SELECT item_id, visits_30d FROM ml_item_visits WHERE user_id = $1", owner,
        )
        ads = await conn.fetch(
            """
            SELECT DISTINCT ON (item_id) item_id, campaign_id, status
              FROM ml_ad_ads
             WHERE user_id = $1
             ORDER BY item_id, synced_at DESC
            """,
            owner,
        )
        promos = await conn.fetch(
            """
            SELECT item_id,
                   count(*) FILTER (WHERE status = 'started') AS active,
                   count(*) FILTER (WHERE status = 'candidate') AS available
              FROM ml_user_promotions
             WHERE user_id = $1
             GROUP BY item_id
            """,
            owner,
        )

    by = lambda rows: {str(r["item_id"]).upper(): r for r in rows}  # noqa: E731
    sales_m, quality_m, visits_m = by(sales), by(quality), by(visits)
    ads_m, promos_m = by(ads), by(promos)

    attribution = ml_access.attribution(owner)
    skus = ml_access.item_skus(owner)

    out = []
    for r in items:
        iid = str(r["item_id"]).upper()
        if not scope.allows_read(iid):
            continue
        s, q, v, a, p = (m.get(iid) for m in (sales_m, quality_m, visits_m, ads_m, promos_m))
        orders = int(s["orders"]) if s else 0
        v30 = int(v["visits_30d"]) if v and v["visits_30d"] is not None else None
        sub_status = r["sub_status_raw"]
        if sub_status and sub_status.startswith("["):
            try:
                sub_status = ", ".join(json.loads(sub_status)) or None
            except ValueError:
                pass
        out.append({
            "itemId": iid,
            "title": r["title"],
            "sku": skus.get(iid) or skus.get(r["item_id"]),
            "project": attribution.get(iid) or attribution.get(r["item_id"]),
            "thumbnail": r["thumbnail"],
            "permalink": r["permalink"],
            "status": r["status"],
            "subStatus": sub_status or None,
            "price": _num(r["price"]),
            "stock": r["available_quantity"],
            "soldTotal": r["sold_quantity"],
            "listingType": r["listing_type_id"],
            "categoryId": r["category_id"],
            "orders30d": orders,
            "units30d": int(s["units"]) if s else 0,
            "revenue30d": round(float(s["revenue"]), 2) if s and s["revenue"] is not None else 0.0,
            "visits30d": v30,
            "conversionPct": round(orders / v30 * 100, 2) if v30 else None,
            "qualityScore": _num(q["score"]) if q else None,
            "qualityLevel": q["level"] if q else None,
            "ads": (
                {"campaignId": a["campaign_id"], "status": a["status"]} if a else None
            ),
            "promotions": (
                {"active": int(p["active"]), "available": int(p["available"])} if p else None
            ),
            "canEdit": scope.allows_write(iid),
        })

    out.sort(key=lambda x: (x["status"] != "active", -x["revenue30d"], x["title"] or ""))
    return {
        "listings": out,
        "total": len(out),
        "projects": sorted(scope.projects),
        "itemsFetchedAt": max((r["fetched_at"] for r in items if r["fetched_at"]), default=None),
    }
