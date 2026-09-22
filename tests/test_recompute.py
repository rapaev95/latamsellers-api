import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Recompute endpoints: job dedup, key correctness, status transitions."""
import asyncio, time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from v2.routers import finance
from v2.deps import CurrentUser, current_user
from v2.db import get_pool
from v2.legacy import config as legacy_config
from v2.services import finance_jobs

USER = CurrentUser(id=7, email="a@b.c", name="T", role="admin")
PROJECTS = {"ARTHUR": {"type": "ecom"}, "ESTONIA": {"type": "services"}}

compute_calls = {"reports": 0, "matrix": 0}

def fake_reports(uid, project, projects, pf, pt, basis, force=False):
    compute_calls["reports"] += 1
    time.sleep(0.3)                       # stand-in for the real cold compute
    return {"pnl": {}}, "force"

def fake_matrix(uid, project, force=False, timeout=90):
    compute_calls["matrix"] += 1
    time.sleep(0.3)
    return {"months": ["2026-09"], "rows": []}, "force"

async def fake_resolve(pool, caller, project):
    return caller.id

finance._reports_bundle_cached = fake_reports
finance._pnl_matrix_cached = fake_matrix
finance._resolve_effective_user_for_project = fake_resolve
legacy_config.load_projects = lambda: PROJECTS

app = FastAPI()
app.include_router(finance.router, prefix="/api/v2")
app.dependency_overrides[current_user] = lambda: USER
app.dependency_overrides[get_pool] = lambda: None

P = {"project": "ARTHUR", "from": "2026-06-24", "to": "2026-09-22"}

with TestClient(app) as c:
    r = c.post("/api/v2/finance/recompute", params=P)
    assert r.status_code == 200, r.text
    body = r.json()
    keys = {j["kind"]: j["key"] for j in body["jobs"]}
    print("jobs:", body["jobs"])
    assert keys["matrix"] == "matrix:ARTHUR", keys
    assert keys["reports"] == "reports:ARTHUR:2026-06-24:2026-09-22:accrual", keys
    assert body["running"] is True, body

    # Second click while the first is in flight must NOT start a second compute.
    c.post("/api/v2/finance/recompute", params=P)
    c.post("/api/v2/finance/recompute", params=P)

    deadline = time.time() + 15
    while time.time() < deadline:
        st = c.get("/api/v2/finance/recompute/status", params=P).json()
        if not st["running"]:
            break
        time.sleep(0.2)
    print("final:", st["jobs"])
    assert st["running"] is False, st
    assert all(j["status"] == "done" for j in st["jobs"]), st
    assert compute_calls == {"reports": 1, "matrix": 1}, compute_calls
    print("dedup held:", compute_calls)

    # Unknown project → 404, not a silently started job.
    r = c.post("/api/v2/finance/recompute", params={"project": "NOPE"})
    assert r.status_code == 404, r.status_code

    # Services project needs superadmin.
    r = c.post("/api/v2/finance/recompute", params={"project": "ESTONIA"})
    assert r.status_code == 403, r.status_code

print("ALL RECOMPUTE TESTS PASSED")
