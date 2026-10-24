import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""MCP-сервер: рукопожатие, список инструментов и граница доступа.

Два разных риска проверяются здесь вместе, потому что в MCP они сходятся в
одной точке. Первый — протокол: клиент, которому не ответили на `initialize`
или `tools/list`, просто показывает «сервер недоступен», и отладить это со
стороны партнёра нечем. Второй — доступ: инструмент, забывший спросить
`resolve_scope`, отдаёт карточки всего кабинета, и снаружи такой ответ
выглядит абсолютно нормально.

Отдельно закреплено, что чужая карточка даёт `not_found`, а не `forbidden`:
«запрещено» подтверждает, что карточка в кабинете есть, и превращает перебор
MLB-id в способ узнать ассортимент соседа.
"""
import asyncio
import json

from v2.routers import mcp as mcp_router
from v2.services import ml_access

ATTRIBUTION = {
    "MLB1": "ARTHUR",
    "MLB2": "ARTEM-TENIS",
    "MLB3": "NAO_CLASSIFICADO",
}
ROWS = {
    "MLB1": {"item_id": "MLB1", "title": "кепка", "price": 100, "currency_id": "BRL",
             "status": "active", "available_quantity": 5, "sold_quantity": 30,
             "permalink": "u1", "site_id": "MLB", "category_id": "c", 
             "listing_type_id": "gold", "health": 0.9, "raw": {}},
    "MLB2": {"item_id": "MLB2", "title": "ракетка", "price": 200, "currency_id": "BRL",
             "status": "active", "available_quantity": 2, "sold_quantity": 10,
             "permalink": "u2", "site_id": "MLB", "category_id": "c",
             "listing_type_id": "gold", "health": 0.8, "raw": {}},
}

ml_access._attribute_items = lambda owner_id: dict(ATTRIBUTION)
ml_access._load_own_projects = lambda caller_id: (
    {"ARTHUR", "ARTEM-TENIS"} if caller_id == 6 else set()
)
MEMBERSHIPS = {
    7: [{"project_name": "ARTHUR", "owner_id": 6, "role": "admin", "synced": True}],
    8: [{"project_name": "ARTEM-TENIS", "owner_id": 6, "role": "viewer", "synced": True}],
}
ml_access._load_memberships = lambda pool, uid: asyncio.sleep(
    0, result=list(MEMBERSHIPS.get(uid, []))
)


class _Conn:
    """Подставная БД: возвращает строки из ROWS, уважая фильтр по item_id."""

    def __init__(self): self.last_sql = ""

    async def fetch(self, sql, *params):
        self.last_sql = sql
        allowed = None
        for p in params:
            if isinstance(p, list):
                allowed = set(p)
        return [
            r for r in ROWS.values()
            if allowed is None or r["item_id"] in allowed
        ]

    async def fetchrow(self, sql, *params):
        want = params[-1] if params else None
        return ROWS.get(want)

    async def execute(self, *a): return "UPDATE 1"


class _Acq:
    def __init__(self, conn): self.conn = conn
    async def __aenter__(self): return self.conn
    async def __aexit__(self, *a): return False


class _Pool:
    def __init__(self): self.conn = _Conn()
    def acquire(self): return _Acq(self.conn)


POOL = _Pool()
mcp_router.ml_user_items_svc.ensure_schema = lambda pool: asyncio.sleep(0)


class _U:
    def __init__(self, uid): self.id = uid


def run(coro): return asyncio.get_event_loop().run_until_complete(coro)


fails: list[str] = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
    print(("ok   " if got == want else "FAIL ") + label)


class _Req:
    """Запрос с заголовком ключа; сам ключ разрешается подменённым resolve."""

    def __init__(self, uid=None): self._uid = uid

    @property
    def headers(self):
        return {"authorization": "Bearer ls_mcp_test"} if self._uid else {}


mcp_router.mcp_tokens.resolve = lambda pool, token: asyncio.sleep(
    0, result=({"id": _Req.current_uid, "email": "p@x", "name": "p",
                "role": "user", "tiers": [], "token_id": 1}
               if _Req.current_uid else None)
)
_Req.current_uid = None


def rpc(method, params=None, uid=None):
    _Req.current_uid = uid
    return run(mcp_router._handle_rpc(
        _Req(uid), POOL,
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
    ))


def tool(name, args=None, uid=6):
    res = rpc("tools/call", {"name": name, "arguments": args or {}}, uid=uid)
    assert "result" in res, res
    return json.loads(res["result"]["content"][0]["text"])


# --- протокол --------------------------------------------------------------
init = rpc("initialize", {"protocolVersion": "2025-06-18"})
check("initialize отвечает", init["result"]["protocolVersion"], "2025-06-18")
check("серверу есть имя", init["result"]["serverInfo"]["name"], "latamsellers-escalar")
check("объявлены tools", "tools" in init["result"]["capabilities"], True)

old = rpc("initialize", {"protocolVersion": "2024-11-05"})
check("старая версия протокола принята", old["result"]["protocolVersion"], "2024-11-05")
unknown = rpc("initialize", {"protocolVersion": "1999-01-01"})
check("незнакомая версия → своя", unknown["result"]["protocolVersion"],
      mcp_router.PROTOCOL_VERSION)

check("ping", rpc("ping")["result"], {})
check("уведомление без ответа",
      run(mcp_router._handle_rpc(_Req(), POOL,
          {"jsonrpc": "2.0", "method": "notifications/initialized"})), None)
check("неизвестный метод → -32601",
      rpc("blah")["error"]["code"], -32601)

tools = rpc("tools/list")["result"]["tools"]
check("инструментов пять", len(tools), 5)
check("у каждого есть схема", all("inputSchema" in t for t in tools), True)
check("tools/list без ключа", "result" in rpc("tools/list"), True)

# --- ключ обязателен для вызова -------------------------------------------
no_key = rpc("tools/call", {"name": "list_my_listings"}, uid=None)
check("вызов без ключа отклонён", no_key["error"]["code"], -32001)
check("несуществующий инструмент",
      rpc("tools/call", {"name": "rm_rf"}, uid=6)["error"]["code"], -32602)

# --- граница доступа -------------------------------------------------------
owner = tool("list_my_listings", uid=6)
check("владелец видит оба", {l["item_id"] for l in owner["listings"]}, {"MLB1", "MLB2"})

a = tool("list_my_listings", uid=7)
check("партнёр A видит только своё", {l["item_id"] for l in a["listings"]}, {"MLB1"})
check("A получил проект в ответе", a["listings"][0]["project"], "ARTHUR")

b = tool("list_my_listings", uid=8)
check("партнёр B видит только своё", {l["item_id"] for l in b["listings"]}, {"MLB2"})

check("A не читает карточку B", tool("get_listing", {"item_id": "MLB2"}, uid=7),
      {"error": "not_found", "item_id": "MLB2"})
check("A читает свою", tool("get_listing", {"item_id": "MLB1"}, uid=7)["title"], "кепка")
check("A не видит непривязанную", tool("get_listing", {"item_id": "MLB3"}, uid=7)["error"],
      "not_found")

# --- запись ---------------------------------------------------------------
check("A не меняет цену B",
      tool("set_listing_price", {"item_id": "MLB2", "price": 1}, uid=7)["error"],
      "not_found")
check("B (viewer) не меняет свою",
      tool("set_listing_price", {"item_id": "MLB2", "price": 1}, uid=8)["error"],
      "read_only")
check("отрицательная цена отклонена",
      tool("set_listing_price", {"item_id": "MLB1", "price": -5}, uid=7)["error"],
      "bad_price")
check("мусорный статус отклонён",
      tool("set_listing_status", {"item_id": "MLB1", "status": "deleted"}, uid=7)["error"],
      "bad_status")

# --- проекты --------------------------------------------------------------
projects_a = tool("list_my_projects", uid=7)
check("A: один проект", [p["project"] for p in projects_a["projects"]], ["ARTHUR"])
check("A: не владелец кабинета", projects_a["cabinet_owner"], False)
check("A: может править", projects_a["projects"][0]["can_edit"], True)
projects_b = tool("list_my_projects", uid=8)
check("B: viewer не может править", projects_b["projects"][0]["can_edit"], False)

# --- фильтр уходит в SQL, а не применяется после LIMIT --------------------
tool("list_my_listings", {"limit": 1}, uid=7)
check("фильтр по item_id в SQL", "item_id = ANY(" in POOL.conn.last_sql, True)
check("LIMIT после фильтра",
      POOL.conn.last_sql.index("item_id = ANY(") < POOL.conn.last_sql.index("LIMIT"), True)

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all mcp checks passed")
