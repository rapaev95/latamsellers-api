import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""abc_summary_cached: the function is CALLED, not just imported.

The regression this guards against was a NameError inside the body — the
extraction left a reference to `_asyncio`, an alias that only existed inside
the endpoint. Importing the module proved nothing, and a symtable pass over
the source reported every name as resolvable. Only executing it fails.

It also pins the ordering rule: on a cache hit the heavy inputs must never be
loaded. That ordering is the whole point of a read-through cache and is
invisible in the response — a wrong order returns identical data, just slowly.
"""
import asyncio
import v2.routers.escalar as esc
from v2.services import finance_cache

loaded: list[str] = []

async def _track(name, result):
    loaded.append(name)
    return result

esc.legacy_db.set_current_user_id = lambda uid: None
esc.user_storage.get = lambda pool, uid, key: asyncio.sleep(0, result=[])
esc.projects.load_resolver = lambda pool, uid: asyncio.sleep(0, result=object())
esc.get_settings = lambda: type("S", (), {"storage_mode": "db"})()

class _Conn:
    async def fetch(self, *a): return []
    async def fetchrow(self, *a): return None
class _Acq:
    async def __aenter__(self): return _Conn()
    async def __aexit__(self, *a): return False
class _Pool:
    def acquire(self): return _Acq()
POOL = _Pool()
esc.db_loader.load_user_vendas = lambda p, u: _track("vendas", [])
esc.db_loader.load_user_armazenagem = lambda p, u: _track("armazenagem", {})
esc.db_loader.load_user_stock_full = lambda p, u: _track("stock_full", {})
esc.db_loader.list_user_vendas_filenames = lambda p, u: _track("filenames", [])
esc.db_loader.load_user_publicidade = lambda p, u: _track("publicidade", [])
esc.abc.aggregate = lambda **kw: {"products": ["computed"]}

finance_cache.compute_fingerprint = lambda uid, extra=None: ("fp1", {})
finance_cache.cached_compute = lambda uid, key, fn, **kw: (fn(), "miss")

# ── hit: answers without touching the loaders ─────────────────────────────
finance_cache._read_cached = lambda uid, k, fp: {"products": ["cached"]}
loaded.clear()
summary, status = asyncio.run(esc.abc_summary_cached(POOL, 7, 30))
print("hit :", status, summary["products"], "| загружено:", loaded)
assert status == "hit" and summary["products"] == ["cached"], (status, summary)
assert loaded == [], f"на попадании в кэш грузить входные данные нельзя: {loaded}"
print("попадание не грузит входные данные ✓")

# ── miss: loads them, then computes ───────────────────────────────────────
finance_cache._read_cached = lambda uid, k, fp: None
loaded.clear()
summary, status = asyncio.run(esc.abc_summary_cached(POOL, 7, 30))
print("miss:", status, summary["products"], "| загружено:", loaded)
assert status == "miss" and summary["products"] == ["computed"], (status, summary)
assert set(loaded) == {"vendas", "armazenagem", "stock_full", "filenames", "publicidade"}, loaded
print("промах грузит всё и считает ✓")

# ── fresh: skips the read even when something is cached ───────────────────
finance_cache._read_cached = lambda uid, k, fp: {"products": ["cached"]}
loaded.clear()
summary, status = asyncio.run(esc.abc_summary_cached(POOL, 7, 30, fresh=True))
assert status == "miss" and summary["products"] == ["computed"], (status, summary)
assert loaded, "fresh обязан загрузить входные данные"
print("fresh игнорирует кэш ✓")

print("ALL ABC SUMMARY TESTS PASSED")
