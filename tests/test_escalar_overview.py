import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Escalar home numbers: scoped to the project, unknown is None, not 0.

The previous home read sales through an endpoint that did not exist and so
showed R$ 0 for every project. These pin what the replacement counts.
"""
from datetime import date

from v2.services.escalar_overview import build_overview

TODAY = date(2026, 10, 24)


def _order(oid, day, items, status="paid"):
    return {"order_id": oid, "date_created": f"{day}T15:00:00Z", "status": status,
            "items": items}


def _line(mlb, qty, price, sku="S"):
    return {"mlb": mlb, "quantity": qty, "unit_price": price, "seller_sku": sku,
            "title": mlb}


ORDERS = [
    _order(1, "2026-10-24", [_line("MLB1", 2, 50.0)]),
    _order(2, "2026-10-20", [_line("MLB1", 1, 50.0), _line("MLB9", 1, 999.0)]),  # mixed
    _order(3, "2026-10-21", [_line("MLB1", 1, 50.0)], status="cancelled"),
    _order(4, "2026-10-10", [_line("MLB1", 1, 40.0)]),   # previous 7-day window? no: 14 days back
    _order(5, "2026-10-15", [_line("MLB1", 1, 40.0)]),   # previous window 10-11..10-17
    _order(6, "2026-10-22", [_line("MLB9", 3, 999.0)]),  # other project only
]
CLAIMS = [
    {"type": "returns", "date_created": "2026-10-23T10:00:00Z", "resource_id": "1",
     "enriched": {"returns": [{"orders": [{"item_id": "MLB1"}]}]}},
    {"type": "returns", "date_created": "2026-10-23T10:00:00Z", "resource_id": "6",
     "enriched": {}},  # falls back to the order's items → MLB9, out of scope
]
VISITS = {
    "MLB1": [{"date": "2026-10-24", "total": 30}, {"date": "2026-10-19", "total": 20},
             {"date": "2026-10-12", "total": 25}],
    "MLB9": [{"date": "2026-10-24", "total": 1000}],
}


def _run(allowed, days=7):
    return build_overview(
        orders=ORDERS, claims=CLAIMS, visits=VISITS, items_meta={},
        allowed=allowed, today=TODAY, days=days,
    )


def test_sold_is_orders_minus_cancelled_and_returns():
    o = _run(frozenset({"MLB1"}))
    k = o["kpis"]
    assert k["ordersGross"] == 3       # orders 1, 2 (its MLB1 line) and cancelled 3
    assert k["cancelled"] == 1
    assert k["returns"] == 1           # order 1; the claim on order 6 is MLB9's
    assert k["orders"] == 1            # 3 − 1 − 1
    assert k["orders"] == k["ordersGross"] - k["cancelled"] - k["returns"]
    assert k["returnedUnits"] == 1     # one of the two units of order 1 came back
    assert k["units"] == 2             # order 2's unit + the kept unit of order 1
    assert k["revenue"] == 100.0
    assert k["revenueGross"] == 200.0  # 100 + 50 + cancelled 50
    assert k["visits"] == 50
    assert k["conversionPct"] == 6.0   # gross orders ÷ visits
    assert o["previous"]["orders"] == 1
    assert o["previous"]["visits"] == 25


def test_whole_cabinet_when_unscoped():
    k = _run(None)["kpis"]
    assert k["ordersGross"] == 4
    assert k["returns"] == 2
    assert k["orders"] == 1


def test_visits_older_than_ml_keeps_are_unknown():
    o = _run(frozenset({"MLB1"}), days=30)
    assert o["kpis"]["visits"] == 75
    assert o["previous"]["visits"] is None
    assert o["change"]["visits"] is None


def test_no_visits_means_no_conversion_not_zero():
    o = build_overview(orders=ORDERS, claims=[], visits={}, items_meta={},
                       allowed=frozenset({"MLB1"}), today=TODAY, days=7)
    assert o["kpis"]["visits"] is None
    assert o["kpis"]["conversionPct"] is None
    assert o["daily"][-1]["visits"] is None


def test_daily_series_and_top():
    o = _run(frozenset({"MLB1"}))
    assert len(o["daily"]) == 7
    assert o["daily"][-1] == {"date": "2026-10-24", "revenue": 50.0, "orders": 0,
                              "units": 1, "visits": 30}
    assert [t["itemId"] for t in o["top"]] == ["MLB1"]
    assert o["top"][0]["returns"] == 1


def test_past_period_keeps_visits_inside_mls_30_days():
    # A 7-day period ending 2026-10-21, seen on 2026-10-24: still inside the
    # 30 days ML keeps, so visits are known, counted only inside the period.
    o = build_overview(orders=ORDERS, claims=CLAIMS, visits=VISITS, items_meta={},
                       allowed=frozenset({"MLB1"}), today=date(2026, 10, 21), days=7,
                       visits_today=TODAY)
    assert o["period"] == {"from": "2026-10-15", "to": "2026-10-21", "days": 7}
    assert o["kpis"]["visits"] == 20        # only 2026-10-19 falls inside
    assert o["kpis"]["ordersGross"] == 3    # orders 2 and 3 (cancelled), 5


def test_vendas_funnel_adds_up_and_splits_cancellations(monkeypatch):
    import pandas as pd
    from v2.services import escalar_overview as ov
    from v2.legacy import reports

    rows = [
        # (date, estado, bucket, project, revenue, units)
        ("5 de janeiro de 2026 10:00 hs.", "Entregue", "delivered", "ARTHUR", 100, 1),
        ("6 de janeiro de 2026 10:00 hs.", "Venda cancelada", "in_progress", "ARTHUR", 50, 1),
        ("7 de janeiro de 2026 10:00 hs.", "Devolução finalizada", "returned", "ARTHUR", 80, 1),
        ("8 de janeiro de 2026 10:00 hs.", "A caminho", "in_progress", "ARTHUR", 30, 2),
        ("9 de janeiro de 2026 10:00 hs.", "Entregue", "delivered", "GANZA", 999, 1),
        ("9 de fevereiro de 2026 10:00 hs.", "Entregue", "delivered", "ARTHUR", 70, 1),
    ]
    df = pd.DataFrame(
        [{"Data da venda": d, "Estado": e, "__bucket": b, "__project": p,
          "Receita por produtos (BRL)": r, "Unidades": u} for d, e, b, p, r, u in rows]
    )
    monkeypatch.setattr(reports, "load_vendas_ml_report", lambda: df)
    f = ov.vendas_funnel_sync(6, {"ARTHUR"}, date(2026, 1, 1), date(2026, 1, 31))
    assert f["ordered"] == {"count": 4, "units": 5, "revenue": 260.0}
    assert f["cancelled"]["revenue"] == 50.0
    assert f["returned"]["revenue"] == 80.0
    assert f["inTransit"] == {"count": 1, "units": 2, "revenue": 30.0}
    assert f["delivered"]["revenue"] == 100.0
    o = f["ordered"]["revenue"] - f["cancelled"]["revenue"] - f["returned"]["revenue"] - f["inTransit"]["revenue"]
    assert o == f["delivered"]["revenue"]
