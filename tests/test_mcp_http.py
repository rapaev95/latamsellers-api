import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""MCP через настоящий HTTP-слой (ASGI, без сети и без прод-БД).

test_mcp.py вызывает `_handle_rpc` напрямую и поэтому не видит целый класс
поломок: не тот путь роутера, проглоченный заголовок Authorization,
несовпадение content-type, падение разбора тела. Клиент в таком случае просто
показывает «сервер недоступен», и причину с его стороны не видно.

Пул здесь подставной: тест проверяет транспорт, а не данные.
"""
import asyncio
import json

import httpx
from fastapi import FastAPI

from v2.app import router as v2_router
from v2.deps import get_pool
from v2.routers import mcp as mcp_router
from v2.services import ml_access

app = FastAPI()
app.include_router(v2_router)


class _Conn:
    async def fetch(self, *a): return []
    async def fetchrow(self, *a): return None
    async def execute(self, *a): return "UPDATE 0"


class _Acq:
    async def __aenter__(self): return _Conn()
    async def __aexit__(self, *a): return False


class _Pool:
    def acquire(self): return _Acq()


app.dependency_overrides[get_pool] = lambda: _Pool()

# Ключ ls_mcp_good принадлежит пользователю 7, остальные не существуют.
mcp_router.mcp_tokens.resolve = lambda pool, token: asyncio.sleep(
    0,
    result=({"id": 7, "email": "a@x", "name": "artur", "role": "user",
             "tiers": [], "token_id": 1} if token == "ls_mcp_good" else None),
)
mcp_router.ml_user_items_svc.ensure_schema = lambda pool: asyncio.sleep(0)
ml_access._load_own_projects = lambda uid: set()
ml_access._load_memberships = lambda pool, uid: asyncio.sleep(
    0,
    result=([{"project_name": "ARTHUR", "owner_id": 6, "role": "admin",
              "synced": True}] if uid == 7 else []),
)
ml_access._attribute_items = lambda owner_id: {"MLB1": "ARTHUR"}

fails: list[str] = []


def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r}, want {want!r}")
    print(("ok   " if got == want else "FAIL ") + label)


async def main():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        # рукопожатие без заголовков
        r = await c.post("/api/v2/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        })
        check("initialize: 200", r.status_code, 200)
        check("initialize: версия", r.json()["result"]["protocolVersion"], "2025-06-18")

        r = await c.post("/api/v2/mcp/", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        check("адрес со слэшем тоже работает", r.status_code, 200)
        check("пять инструментов", len(r.json()["result"]["tools"]), 5)

        # вызов без ключа
        r = await c.post("/api/v2/mcp", json={
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "list_my_projects", "arguments": {}},
        })
        check("без ключа: 200 с JSON-RPC ошибкой", r.status_code, 200)
        check("без ключа: код -32001", r.json()["error"]["code"], -32001)

        # вызов с негодным ключом
        r = await c.post("/api/v2/mcp", json={
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "list_my_projects", "arguments": {}},
        }, headers={"Authorization": "Bearer ls_mcp_wrong"})
        check("негодный ключ отклонён", r.json()["error"]["code"], -32001)

        # вызов с годным ключом — заголовок доехал до роутера
        r = await c.post("/api/v2/mcp", json={
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "list_my_projects", "arguments": {}},
        }, headers={"Authorization": "Bearer ls_mcp_good"})
        body = r.json()
        check("годный ключ принят", "result" in body, True)
        payload = json.loads(body["result"]["content"][0]["text"])
        check("области доступа применены",
              [p["project"] for p in payload["projects"]], ["ARTHUR"])
        check("не владелец кабинета", payload["cabinet_owner"], False)

        # битое тело
        r = await c.post(
            "/api/v2/mcp", content=b"{not json",
            headers={"content-type": "application/json"},
        )
        check("битый JSON → -32700", r.json()["error"]["code"], -32700)

        # батч
        r = await c.post("/api/v2/mcp", json=[
            {"jsonrpc": "2.0", "id": 6, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
        ])
        batch = r.json()
        check("батч: ответов два (уведомление без ответа)", len(batch), 2)
        check("батч: id сохранены", [x["id"] for x in batch], [6, 7])

        # управление ключами требует cookie-логина, а не Bearer
        r = await c.get("/api/v2/mcp/tokens")
        check("список ключей без cookie → 401", r.status_code, 401)
        r = await c.get("/api/v2/mcp/tokens",
                        headers={"Authorization": "Bearer ls_mcp_good"})
        check("Bearer не заменяет логин для выдачи ключей", r.status_code, 401)


asyncio.run(main())
print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all mcp http checks passed")
