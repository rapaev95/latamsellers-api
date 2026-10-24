"""MCP-сервер над карточками кабинета.

Партнёр подключает свой клиент (Claude и подобные) и спрашивает про товары
словами, вместо того чтобы открывать UI. Сервер отвечает ровно тем же, что
партнёр увидел бы на странице: ни больше, ни меньше.

Границу доступа сервер не определяет — он её спрашивает. Каждый инструмент
начинается с `ml_access.resolve_scope` и пропускает данные через `scope`.
Своя проверка здесь была бы вторым набором правил, который рано или поздно
разойдётся с первым; расхождение в правилах доступа — это утечка.

Транспорт — streamable HTTP: JSON-RPC 2.0 одним POST на /api/v2/mcp.
Аутентификация — `Authorization: Bearer ls_mcp_...`, ключи в
v2.services.mcp_tokens. Cookie здесь не используется: у MCP-клиента её нет.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request

from v2.deps import CurrentUser, current_user, get_pool
from v2.services import mcp_tokens, ml_access
from v2.services import ml_oauth as ml_oauth_svc
from v2.services import ml_user_items as ml_user_items_svc

router = APIRouter(prefix="/mcp", tags=["mcp"])
log = logging.getLogger("mcp")

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = (PROTOCOL_VERSION, "2025-03-26", "2024-11-05")
SERVER_NAME = "latamsellers-escalar"
SERVER_VERSION = "1.0.0"

# Сколько карточек отдаём за один вызов. Клиент получает их как текст, и
# полный кабинет в одном ответе бесполезен и для модели, и для партнёра.
_MAX_LIMIT = 100


class _TokenUser:
    """Пользователь, опознанный по MCP-ключу.

    Повторяет ту часть формы CurrentUser, на которую смотрит ml_access, —
    этого достаточно, и от cookie-зависимостей сервер остаётся свободен.
    """

    def __init__(self, data: dict[str, Any]):
        self.id = int(data["id"])
        self.email = data.get("email")
        self.name = data.get("name")
        self.role = data.get("role")
        self.tiers = data.get("tiers")


# ── управление ключами (через обычный логин в UI) ────────────────────────────

@router.post("/tokens")
async def create_token(
    body: dict[str, Any] = Body(default={}),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    """Выдать ключ для MCP-клиента. Открытое значение — только в этом ответе."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    return await mcp_tokens.create(pool, user.id, str(body.get("name") or ""))


@router.get("/tokens")
async def list_tokens(
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    return {"tokens": await mcp_tokens.list_for_user(pool, user.id)}


@router.delete("/tokens/{token_id}")
async def revoke_token(
    token_id: int,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    ok = await mcp_tokens.revoke(pool, user.id, token_id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error": "token_not_found"})
    return {"ok": True}


# ── инструменты ─────────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_my_projects",
        "description": (
            "Проекты, к которым у вызывающего есть доступ в кабинете Mercado "
            "Livre, с ролью и числом карточек в каждом."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_my_listings",
        "description": (
            "Карточки вызывающего: MLB-id, название, цена, остаток, статус и "
            "проект. Чужие карточки того же кабинета не возвращаются."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Сузить до одного проекта (необязательно).",
                },
                "status": {
                    "type": "string",
                    "description": "active | paused | closed (необязательно).",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT},
            },
        },
    },
    {
        "name": "get_listing",
        "description": (
            "Подробности одной карточки по MLB-id. Если карточка не "
            "принадлежит вызывающему, возвращается «не найдено»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}},
            "required": ["item_id"],
        },
    },
    {
        "name": "set_listing_price",
        "description": (
            "Изменить цену своей карточки в Mercado Livre. Требует роли с "
            "правом записи; у роли viewer вызов отклоняется."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "price": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["item_id", "price"],
        },
    },
    {
        "name": "set_listing_status",
        "description": (
            "Поставить свою карточку на паузу или снова активировать "
            "(active | paused)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "paused"]},
            },
            "required": ["item_id", "status"],
        },
    },
]


async def _listing_rows(
    pool, scope: ml_access.MlScope, status: str = "", limit: int = 50,
) -> list[dict[str, Any]]:
    """Карточки из кэша кабинета, отфильтрованные областью доступа.

    Фильтр накладывается в SQL по списку разрешённых item_id, а не после
    выборки: иначе `LIMIT` считался бы по всему кабинету и партнёр с десятью
    карточками получал бы пустые страницы.
    """
    params: list[Any] = [scope.owner_id]
    where = ["user_id = $1"]
    if scope.item_ids is not None:
        params.append(list(scope.item_ids))
        where.append(f"item_id = ANY(${len(params)})")
    if status:
        params.append(status)
        where.append(f"status = ${len(params)}")
    params.append(max(1, min(limit, _MAX_LIMIT)))
    sql = (
        "SELECT item_id, title, price, currency_id, status, available_quantity,"
        " sold_quantity, permalink, site_id"
        "  FROM ml_user_items"
        f" WHERE {' AND '.join(where)}"
        " ORDER BY sold_quantity DESC NULLS LAST"
        f" LIMIT ${len(params)}"
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    attribution = ml_access._attribute_items(scope.owner_id)
    return [
        {
            "item_id": r["item_id"],
            "title": r["title"],
            "price": float(r["price"]) if r["price"] is not None else None,
            "currency": r["currency_id"],
            "status": r["status"],
            "stock": r["available_quantity"],
            "sold": r["sold_quantity"],
            "url": r["permalink"],
            "site": r["site_id"],
            "project": attribution.get(r["item_id"]),
        }
        for r in rows
    ]


async def _ml_put_item(
    pool, scope: ml_access.MlScope, item_id: str, payload: dict[str, Any],
) -> dict[str, Any]:
    """Изменение карточки в ML токеном владельца кабинета.

    Право на запись проверяет вызывающий; здесь только обращение к ML и
    синхронизация локального кэша, чтобы следующий вызов не вернул старое
    значение.
    """
    try:
        token, *_ = await ml_oauth_svc.get_valid_access_token(pool, scope.owner_id)
    except ml_oauth_svc.MLRefreshError as err:
        return {"ok": False, "error": "ml_oauth_required", "detail": str(err)}

    async with httpx.AsyncClient() as http:
        try:
            resp = await http.put(
                f"https://api.mercadolibre.com/items/{item_id}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
        except Exception as err:  # noqa: BLE001
            return {"ok": False, "error": "ml_call_failed", "detail": str(err)}
    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": "ml_rejected",
            "status": resp.status_code,
            "detail": resp.text[:400],
        }

    body = resp.json() or {}
    sets, params = [], []
    if "price" in payload and body.get("price") is not None:
        params.append(float(body["price"]))
        sets.append(f"price = ${len(params)}")
    if "status" in payload and body.get("status"):
        params.append(str(body["status"]))
        sets.append(f"status = ${len(params)}")
    if sets:
        params.extend([scope.owner_id, item_id])
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE ml_user_items SET {', '.join(sets)}"
                f" WHERE user_id = ${len(params) - 1} AND item_id = ${len(params)}",
                *params,
            )
    return {
        "ok": True,
        "item_id": item_id,
        "price": body.get("price"),
        "status": body.get("status"),
    }


async def _call_tool(
    pool, user: Any, name: str, args: dict[str, Any],
) -> dict[str, Any]:
    if name == "list_my_projects":
        scope = await ml_access.resolve_scope(pool, user)
        out = []
        for project in sorted(scope.projects):
            one = await ml_access.resolve_scope(pool, user, project=project)
            out.append({
                "project": project,
                "is_owner": one.is_owner,
                "can_edit": one.may_write,
                "listings": (
                    len(one.item_ids) if one.item_ids is not None else None
                ),
            })
        return {"projects": out, "cabinet_owner": scope.is_owner}

    if name == "list_my_listings":
        project = str(args.get("project") or "").strip()
        scope = await ml_access.resolve_scope(pool, user, project=project)
        if not scope.is_owner and not scope.projects:
            return {"listings": [], "count": 0}
        rows = await _listing_rows(
            pool, scope,
            status=str(args.get("status") or "").strip(),
            limit=int(args.get("limit") or 50),
        )
        return {"listings": rows, "count": len(rows)}

    if name == "get_listing":
        item_id = str(args.get("item_id") or "").strip()
        scope = await ml_access.resolve_scope(pool, user)
        if not scope.allows_read(item_id):
            # Та же формулировка, что для несуществующей карточки: иначе
            # ответ подтверждает, что она есть в кабинете у кого-то другого.
            return {"error": "not_found", "item_id": item_id}
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT item_id, title, price, currency_id, status,"
                " available_quantity, sold_quantity, permalink, site_id,"
                " category_id, listing_type_id, health, raw"
                "  FROM ml_user_items WHERE user_id = $1 AND item_id = $2",
                scope.owner_id, item_id,
            )
        if row is None:
            return {"error": "not_found", "item_id": item_id}
        raw = row["raw"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:  # noqa: BLE001
                raw = {}
        raw = raw if isinstance(raw, dict) else {}
        attribution = ml_access._attribute_items(scope.owner_id)
        return {
            "item_id": row["item_id"],
            "title": row["title"],
            "price": float(row["price"]) if row["price"] is not None else None,
            "currency": row["currency_id"],
            "status": row["status"],
            "sub_status": raw.get("sub_status") or [],
            "stock": row["available_quantity"],
            "sold": row["sold_quantity"],
            "url": row["permalink"],
            "site": row["site_id"],
            "category_id": row["category_id"],
            "listing_type": row["listing_type_id"],
            "health": float(row["health"]) if row["health"] is not None else None,
            "project": attribution.get(row["item_id"]),
            "can_edit": (
                await ml_access.resolve_scope(pool, user)
            ).allows_write(item_id),
        }

    if name in ("set_listing_price", "set_listing_status"):
        item_id = str(args.get("item_id") or "").strip()
        scope = await ml_access.resolve_scope(pool, user)
        if not scope.allows_read(item_id):
            return {"error": "not_found", "item_id": item_id}
        if not scope.may_write:
            return {
                "error": "read_only",
                "detail": "у вашей роли в этом проекте нет права изменять карточки",
            }
        if name == "set_listing_price":
            try:
                price = float(args.get("price"))
            except (TypeError, ValueError):
                return {"error": "bad_price"}
            if price <= 0:
                return {"error": "bad_price"}
            return await _ml_put_item(pool, scope, item_id, {"price": price})
        status = str(args.get("status") or "").strip()
        if status not in ("active", "paused"):
            return {"error": "bad_status", "allowed": ["active", "paused"]}
        return await _ml_put_item(pool, scope, item_id, {"status": status})

    return {"error": "unknown_tool", "tool": name}


# ── транспорт JSON-RPC ──────────────────────────────────────────────────────

def _result(req_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": payload}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def _bearer_user(request: Request, pool) -> Optional[_TokenUser]:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return None
    data = await mcp_tokens.resolve(pool, header[7:].strip())
    return _TokenUser(data) if data else None


@router.post("")
@router.post("/")
async def mcp_endpoint(
    request: Request,
    pool=Depends(get_pool),
):
    """Единственная точка входа MCP: JSON-RPC 2.0 поверх POST."""
    if pool is None:
        raise HTTPException(status_code=503, detail="db_unavailable")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _error(None, -32700, "parse error")

    # Батч-запрос протокол допускает; обрабатываем по одному.
    if isinstance(body, list):
        out = []
        for item in body:
            one = await _handle_rpc(request, pool, item)
            if one is not None:
                out.append(one)
        return out
    return await _handle_rpc(request, pool, body) or {}


async def _handle_rpc(request: Request, pool, body: Any):
    if not isinstance(body, dict):
        return _error(None, -32600, "invalid request")
    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params") or {}

    # Уведомления (без id) ответа не требуют.
    if req_id is None and isinstance(method, str) and method.startswith("notifications/"):
        return None

    if method == "initialize":
        asked = str(params.get("protocolVersion") or "")
        version = asked if asked in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
        return _result(req_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Инструменты показывают только те карточки Mercado Livre, "
                "которые принадлежат проектам вызывающего. Пустой список "
                "значит «нет доступа или нет данных», а не ошибку."
            ),
        })

    if method == "ping":
        return _result(req_id, {})

    if method == "tools/list":
        return _result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        # Ключ проверяем здесь, а не на initialize: клиенты нередко
        # рукопожатие делают без заголовка, и отказ на нём выглядит как
        # неработающий сервер.
        user = await _bearer_user(request, pool)
        if user is None:
            return _error(req_id, -32001, "нужен действующий ключ: Authorization: Bearer ls_mcp_...")
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return _error(req_id, -32602, "arguments must be an object")
        if name not in {t["name"] for t in TOOLS}:
            return _error(req_id, -32602, f"unknown tool: {name}")
        try:
            await ml_user_items_svc.ensure_schema(pool)
            payload = await _call_tool(pool, user, name, args)
        except Exception as err:  # noqa: BLE001
            log.exception("mcp tool %s failed", name)
            return _result(req_id, {
                "content": [{"type": "text", "text": f"инструмент не выполнился: {err}"}],
                "isError": True,
            })
        return _result(req_id, {
            "content": [{
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            }],
            "isError": bool(payload.get("error")) if isinstance(payload, dict) else False,
        })

    return _error(req_id, -32601, f"method not found: {method}")
