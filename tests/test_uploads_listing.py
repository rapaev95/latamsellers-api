import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""The uploads LIST view must never pull file contents.

It used to build the listing from `fetch_files_by_source`, which selects
`file_bytes`, purely to call `len()` on them. Every page view therefore
streamed every stored file out of Postgres, and got slower with each upload
until /finance/uploads hit the proxy's 30s cap and returned 502.

This pins the shape of the query, because the regression is invisible in the
response: swapping the helper back would return byte-for-byte identical JSON
and just be slow again.
"""
import asyncio, re
from datetime import datetime
from v2.storage import uploads_storage as us

SQL = []

class _Conn:
    async def fetch(self, sql, *args):
        SQL.append(" ".join(sql.split()))
        return [{
            "id": 1, "user_id": 6, "filename": "vendas.xlsx", "source_key": "vendas_ml",
            "created_at": datetime(2026, 9, 1), "size_bytes": 1_572_864, "project_name": "ARTHUR",
        }]
    async def execute(self, *a, **k): return None

class _Acq:
    async def __aenter__(self): return _Conn()
    async def __aexit__(self, *a): return False

class _Pool:
    def acquire(self): return _Acq()

us.ensure_project_name_column = lambda pool: asyncio.sleep(0)

def projection(sql: str) -> str:
    return re.search(r"SELECT (.*?) FROM", sql, re.S).group(1)

rows = asyncio.run(us.list_files_meta(_Pool(), 6, "vendas_ml"))
assert rows[0].size_bytes == 1_572_864, rows
proj = projection(SQL[-1])
assert "octet_length(file_bytes)" in proj, proj
assert not re.search(r"(^|,)\s*file_bytes\s*(,|$)", proj), f"listing still selects raw bytes: {proj}"
print("list_files_meta: size via octet_length, no bytes in projection ✓")

rows = asyncio.run(us.list_files_meta_for_project(_Pool(), [6, 9], "vendas_ml", "ARTHUR"))
assert rows[0].project_name == "ARTHUR" and rows[0].size_bytes == 1_572_864, rows
proj = projection(SQL[-1])
assert "octet_length(file_bytes)" in proj, proj
assert not re.search(r"(^|,)\s*file_bytes\s*(,|$)", proj), f"listing still selects raw bytes: {proj}"
print("list_files_meta_for_project: same ✓")

# The parsing helpers must still return real bytes — they exist to be parsed,
# so this fix must not have "optimised" them too.
import inspect
assert "file_bytes" in projection(inspect.getsource(us.fetch_files_by_source))
assert "file_bytes" in projection(inspect.getsource(us.fetch_files_for_project))
print("byte-fetching helpers untouched (parsers depend on them) ✓")

# And the router's listing path must not call them. Comments are stripped first:
# the code there *names* those helpers while explaining why it stopped using them.
router = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "v2/routers/finance.py"), encoding="utf-8").read()
body = router[router.index("async def list_uploads("):router.index("_MAX_UPLOAD_BYTES")]
code = "\n".join(re.sub(r"#.*", "", ln) for ln in body.splitlines())
for helper in ("fetch_files_by_source", "fetch_files_for_project"):
    assert helper not in code, f"list_uploads calls {helper} again — that fetches every file"
assert "len(f.file_bytes)" not in code, "list_uploads is measuring bytes it had to download"
print("list_uploads uses metadata-only helpers ✓")

print("ALL UPLOADS LISTING TESTS PASSED")
