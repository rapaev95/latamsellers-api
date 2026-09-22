import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Reports roll-up: cached vs background job, and no silent zeros.

The Escalar P&L waterfall used to fan out one /finance/reports per project and
swallow failures with `catch → null`, so a project that didn't compute simply
left the aggregate smaller. Here every project is accounted for.
"""
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from v2.routers import finance
from v2.deps import CurrentUser, current_user
from v2.db import get_pool
from v2.legacy import config as legacy_config
from v2.services import finance_cache

PROJECTS = {"ARTHUR": {}, "AZAT": {}, "BROKEN": {}}
legacy_config.load_projects = lambda: PROJECTS
finance._bind_user_id = lambda uid: None
finance._bind_user = lambda u: None
finance_cache.compute_fingerprint = lambda uid, extra_deps=None: ("fp1", {})

def pnl_for(project: str) -> dict:
    # `project` is the only required field on PnLReportOut; the rest mirror what
    # the Escalar waterfall actually reads off a report.
    return {"project": project, "revenue_gross": 1000.0, "revenue_net": 800.0,
            "operating_profit": 120.0,
            "operating_expenses": [
                {"label": "Publicidade (Mercado Ads)", "amount_brl": -50.0}]}

KEY = lambda p: f"reports:{p}:2026-06-24:2026-09-22:accrual"
# ARTHUR warm; BROKEN cached but errored; AZAT missing → must become a job.
CACHE = {KEY("ARTHUR"): {"pnl": pnl_for("ARTHUR")},
         KEY("BROKEN"): {"pnl": None, "pnl_error": "no vendas"}}
finance_cache.read_many_cached = lambda uid, keys, fp: {k: CACHE[k] for k in keys if k in CACHE}

computed = []
def fake_bundle(uid, project, projects, pf, pt, basis, force=False):
    computed.append(project)
    time.sleep(0.2)
    CACHE[KEY(project)] = {"pnl": pnl_for(project)}
    return {"pnl": pnl_for(project)}, "miss"
finance._reports_bundle_cached = fake_bundle

app = FastAPI()
app.include_router(finance.router, prefix="/api/v2")
app.dependency_overrides[get_pool] = lambda: None
app.dependency_overrides[current_user] = lambda: CurrentUser(
    id=7, email="a@b.c", name="T", role="admin")
CLIENT = TestClient(app)
CLIENT.__enter__()

P = {"from": "2026-06-24", "to": "2026-09-22"}

def run():
    r = CLIENT.get("/api/v2/finance/reports-rollup", params=P)
    assert r.status_code == 200, r.text
    return r.json()

body = run()
by = {p["project"]: p for p in body["projects"]}
print("statuses:", {k: v["status"] for k, v in by.items()})

assert by["ARTHUR"]["status"] == "cached", by["ARTHUR"]
assert by["ARTHUR"]["pnl"]["revenue_gross"] == 1000.0, by["ARTHUR"]

# A cached-but-errored bundle is an error, not a silently empty report.
assert by["BROKEN"]["status"] == "error" and by["BROKEN"]["pnl"] is None, by["BROKEN"]
assert "no vendas" in by["BROKEN"]["error"], by["BROKEN"]
print("cached pnl served; cached error surfaced as error ✓")

# A miss is a job, not a wait, and carries no numbers.
assert by["AZAT"]["status"] == "pending" and by["AZAT"]["pnl"] is None, by["AZAT"]
assert body["running"] is True and body["pending_count"] == 1, body
assert body["complete"] is False and body["error_count"] == 1, body
print("miss returns pending with no pnl ✓")

run(); run()
assert computed.count("AZAT") == 1, computed
print("polling doesn't restart the job ✓")

deadline = time.time() + 10
while time.time() < deadline:
    body = run()
    if not body["running"]:
        break
    time.sleep(0.1)
by = {p["project"]: p for p in body["projects"]}
print("after job:", {k: v["status"] for k, v in by.items()})
assert by["AZAT"]["status"] == "cached", by["AZAT"]
assert by["AZAT"]["pnl"]["revenue_gross"] == 1000.0, by["AZAT"]
assert body["running"] is False and body["pending_count"] == 0, body
print("background job fills it in ✓")

# Cache key must match what GET /reports writes, or the two never share.
assert KEY("ARTHUR") == finance.reports_cache_key(
    "ARTHUR", __import__("datetime").date(2026, 6, 24),
    __import__("datetime").date(2026, 9, 22), "accrual")
print("cache key matches /reports ✓")

print("ALL REPORTS ROLLUP TESTS PASSED")
