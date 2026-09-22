import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Company roll-up: cache hits, computed misses, pending, forbidden, and the
rule that a project without numbers is never summed as zero."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from v2.routers import finance
from v2.deps import CurrentUser, current_user
from v2.db import get_pool
from v2.legacy import config as legacy_config
from v2.services import finance_cache

PROJECTS = {
    "ARTHUR": {"type": "ecom"},
    "JOOM":   {"type": "ecom"},
    "ESTONIA": {"type": "services"},
}
legacy_config.load_projects = lambda: PROJECTS
finance._bind_user_id = lambda uid: None
finance._bind_user = lambda u: None

MATRIX = {"months": ["2026-08", "2026-09"], "rows": [
    {"label": "pnl_rev_gross",       "total": 350.4, "values": {"2026-08": 100.4, "2026-09": 250}},
    {"label": "pnl_op_profit",       "total": -20.6, "values": {"2026-09": -20.6}},
    {"label": "pnl_orders_delivered","total": 12,    "values": {"2026-09": 12}},
]}

finance_cache.compute_fingerprint = lambda uid, extra_deps=None: ("fp1", {})
# ARTHUR cached; JOOM absent → must be computed.
finance_cache.read_many_cached = lambda uid, keys, fp: {"matrix:ARTHUR": MATRIX}

computed = []
def fake_matrix(uid, project, force=False, timeout=90):
    computed.append(project)
    return MATRIX, "miss"
finance._pnl_matrix_cached = fake_matrix

app = FastAPI()
app.include_router(finance.router, prefix="/api/v2")
app.dependency_overrides[get_pool] = lambda: None

def run(user, **params):
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as c:
        r = c.get("/api/v2/finance/company/revenue-by-month", params=params)
        assert r.status_code == 200, r.text
        return r.json()

# ── regular admin: services project is not readable, and must be REPORTED ──
body = run(CurrentUser(id=7, email="a@b.c", name="T", role="admin"))
by = {p["project"]: p for p in body["projects"]}
print("statuses:", {k: v["status"] for k, v in by.items()})
assert by["ARTHUR"]["status"] == "cached"
assert by["JOOM"]["status"] == "computed" and computed == ["JOOM"], computed
assert by["ESTONIA"]["status"] == "forbidden"
assert by["ESTONIA"]["by_month"] == {}, by["ESTONIA"]
assert body["complete"] is False and body["error_count"] == 1, body
assert body["months"] == ["2026-08", "2026-09"], body
assert by["ARTHUR"]["by_month"] == {"2026-08": 100.0, "2026-09": 250.0}, by["ARTHUR"]
assert by["ARTHUR"]["segment"] == "partner" and by["ESTONIA"]["segment"] == "own"
print("forbidden project reported, not dropped ✓")

# ── budget exhausted → pending, never a silent zero ──
finance._COMPANY_BUDGET_SECONDS = -100
computed.clear()
body = run(CurrentUser(id=7, email="a@b.c", name="T", role="admin"))
by = {p["project"]: p for p in body["projects"]}
print("zero-budget statuses:", {k: v["status"] for k, v in by.items()})
assert by["JOOM"]["status"] == "pending" and computed == [], computed
assert by["JOOM"]["by_month"] == {}, by["JOOM"]
assert body["pending_count"] == 1 and body["complete"] is False, body
assert by["ARTHUR"]["status"] == "cached"    # cached ones are unaffected
print("pending carries no numbers ✓")

# ── the /finance dashboard needs totals, not just a revenue series ─────────
finance._COMPANY_BUDGET_SECONDS = 45
computed.clear()
body = run(CurrentUser(id=7, email="a@b.c", name="T", role="admin"))
arthur = {p["project"]: p for p in body["projects"]}["ARTHUR"]
assert arthur["rows"]["pnl_rev_gross"]["total"] == 350.0, arthur["rows"]
assert arthur["rows"]["pnl_op_profit"]["total"] == -21.0, arthur["rows"]
# A month absent from a row reads as 0 for that row, matching what the UI did
# when it held the whole matrix itself.
assert arthur["rows"]["pnl_op_profit"]["by_month"] == {"2026-08": 0.0, "2026-09": -21.0}, arthur["rows"]
# A row the matrix doesn't have must still be present and zeroed, not missing.
assert arthur["rows"]["pnl_net_revenue"]["total"] == 0.0, arthur["rows"]
# Back-compat mirror for the previously deployed UI build.
assert arthur["by_month"] == arthur["rows"]["pnl_rev_gross"]["by_month"]
print("totals + monthly series per row \u2713")

# Subset selection.
body = run(CurrentUser(id=7, email="a@b.c", name="T", role="admin"), rows="pnl_op_profit")
arthur = {p["project"]: p for p in body["projects"]}["ARTHUR"]
assert set(arthur["rows"]) == {"pnl_op_profit"}, arthur["rows"]
print("rows= selects a subset \u2713")

# services=matrix puts services projects on the ecom footing, which is what the
# dashboard has always done — and it must not need superadmin for that.
computed.clear()
body = run(CurrentUser(id=7, email="a@b.c", name="T", role="admin"), services="matrix")
est = {p["project"]: p for p in body["projects"]}["ESTONIA"]
assert est["status"] == "computed", est          # no longer "forbidden"
assert est["segment"] == "own", est              # still «прямые»
assert est["rows"]["pnl_rev_gross"]["total"] == 350.0, est["rows"]
assert "ESTONIA" in computed, computed
print("services=matrix reads the ecom matrix \u2713")

print("ALL COMPANY TESTS PASSED")
