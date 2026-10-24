import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Escalar stock comes from the live listing, not a months-old upload.

On 2026-10-24 the newest Stock Full xlsx predated the sneakers, so they all
showed 0 while ML had 34 and 47 units on Full.
"""
from v2.services import abc


def _p(sku, item_id, stock=0, spd=0.0):
    return {"sku": sku, "itemId": item_id, "currentStock": stock, "salesPerDay": spd,
            "daysOfStock": 0.0, "reorderStatus": "crit", "orderByDate": None}


def test_live_quantity_replaces_upload():
    ps = [_p("1021-1", "MLB4702288939", stock=0, spd=1.0)]
    assert abc.apply_live_stock(ps, {"MLB4702288939": 13}) == 1
    assert ps[0]["currentStock"] == 13
    assert ps[0]["daysOfStock"] == 13.0
    assert ps[0]["reorderStatus"] == "crit"  # 13 days < lead time + liberação


def test_unknown_listing_keeps_upload_value():
    ps = [_p("X", "MLB1", stock=7)]
    assert abc.apply_live_stock(ps, {"MLB2": 3}) == 0
    assert ps[0]["currentStock"] == 7


def test_listing_shared_by_several_skus_is_not_multiplied():
    ps = [_p("sumka8-1", "5649425304", stock=2), _p("sumka8-2", "5649425304", stock=3)]
    assert abc.apply_live_stock(ps, {"MLB5649425304": 40}) == 0
    assert [p["currentStock"] for p in ps] == [2, 3]


def test_item_id_forms_match():
    ps = [_p("A", "6824606366")]
    abc.apply_live_stock(ps, {"MLB6824606366": 4})
    assert ps[0]["currentStock"] == 4
