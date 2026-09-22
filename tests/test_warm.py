import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Warm pass: iterates projects, honours the budget, survives per-project errors."""
import asyncio, time
from v2.routers import finance
from v2.legacy import config as legacy_config
from v2.services import finance_warm

calls = {"matrix": [], "reports": [], "services": [], "abc": []}

def fake_matrix(uid, project, force=False, timeout=90):
    calls["matrix"].append((uid, project, force))
    if project == "BROKEN":
        raise RuntimeError("compute exploded")
    return {"months": []}, "hit"

def fake_reports(uid, project, projects, pf, pt, basis, force=False):
    calls["reports"].append((uid, project, basis, force))
    return {}, "hit"

async def fake_services(pool, uid, project, pf, pt, fingerprint=None, deps=None):
    calls["services"].append((uid, project))
    return {}

async def fake_abc(pool, uid, days_v, project="", *, fresh=False, step=None):
    calls["abc"].append((uid, days_v, project, fresh))
    return {"products": []}, "miss"

finance._pnl_matrix_cached = fake_matrix
finance._reports_bundle_cached = fake_reports
finance._services_bundle_computed = fake_services
import v2.routers.escalar as escalar_router
escalar_router.abc_summary_cached = fake_abc
finance._bind_user_id = lambda uid: None
legacy_config.load_projects = lambda: {
    "ARTHUR": {"type": "ecom"}, "BROKEN": {"type": "ecom"}, "ESTONIA": {"type": "services"},
}

async def fake_users(pool):
    return [7, 9]
finance_warm._users_with_projects = fake_users

stats = asyncio.run(finance_warm.warm_all(object()))
print("stats:", stats)
assert stats["users"] == 2, stats
# Counters increment only after a project actually succeeds, so BROKEN adds
# nothing to "matrix" and lands in "errors" instead.
assert stats["matrix"] == 2, stats
assert stats["reports"] == 2, stats
assert stats["services"] == 2, stats
# ABC is warmed per USER per window (key is abc:all:<days>), not per project.
assert stats["abc"] == 4, stats                       # 2 users x {30, 90}
assert sorted({d for _, d, _, _ in calls["abc"]}) == [30, 90], calls["abc"]
assert all(project == "" for _, _, project, _ in calls["abc"]), calls["abc"]
assert all(fresh is False for *_, fresh in calls["abc"]), calls["abc"]
assert stats["errors"] == 2, stats            # BROKEN failed for each user, pass continued
assert not stats["stopped_early"], stats

# Warming must NOT force: that is what makes an unchanged project nearly free.
assert all(force is False for *_, force in calls["matrix"]), calls["matrix"]
assert all(force is False for *_, force in calls["reports"]), calls["reports"]
print("no forced recomputes:", set(c[-1] for c in calls["matrix"]))

# Budget: a zero budget must stop the pass immediately instead of running long.
finance_warm._BUDGET_S = -1
calls["matrix"].clear()
stats2 = asyncio.run(finance_warm.warm_all(object()))
print("zero-budget pass:", stats2)
assert stats2["stopped_early"] is True and not calls["matrix"], stats2

print("ALL WARM TESTS PASSED")
