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

MATRIX = {"months": ["2026-08", "2026-09"],
          "rows": [{"label": "pnl_rev_gross", "values": {"2026-08": 100.4, "2026-09": 250}}]}

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

def run(user):
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as c:
        r = c.get("/api/v2/finance/company/revenue-by-month")
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

print("ALL COMPANY TESTS PASSED")
