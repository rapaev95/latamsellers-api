"""Bulk post-publication actions — Sprint 8.

Five primitive actions per `item_id`:
- `pause`    → PUT /items/{id}  body: {"status": "paused"}
- `activate` → PUT /items/{id}  body: {"status": "active"}
- `close`    → PUT /items/{id}  body: {"status": "closed"}
- `change_qty`   → PUT /items/{id} body: {"available_quantity": N}
- `change_price` → POST /items/{id}/prices/standard
                   body: {amount, currency_id, conditions: [{channel}]}
  Direct PUT of `price` is rejected from 18.03.2026 (`CLAUDE.md` →
  «работа с ценой ML.md» line 1). The /prices/standard endpoint is the
  going-forward way.

We throttle 0.5s between Items just like the publisher to stay polite.
The router is responsible for resolving the right ML token (managed or
self) and passing it in.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
THROTTLE_S = 0.5

ACTIONS = {"pause", "activate", "close", "change_qty", "change_price"}


async def _put_status(http: httpx.AsyncClient, token: str, item_id: str, status: str) -> tuple[int, Any]:
    r = await http.put(
        f"{ML_API_BASE}/items/{item_id}",
        json={"status": status},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=20.0,
    )
    return r.status_code, _safe_json(r)


async def _put_qty(http: httpx.AsyncClient, token: str, item_id: str, qty: int) -> tuple[int, Any]:
    r = await http.put(
        f"{ML_API_BASE}/items/{item_id}",
        json={"available_quantity": int(qty)},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=20.0,
    )
    return r.status_code, _safe_json(r)


async def _post_price_standard(
    http: httpx.AsyncClient, token: str, item_id: str,
    *, amount: float, currency_id: str = "BRL", channel: str = "marketplace",
) -> tuple[int, Any]:
    body = {
        "amount": float(amount),
        "currency_id": currency_id,
        "conditions": [{"channel": channel}],
    }
    r = await http.post(
        f"{ML_API_BASE}/items/{item_id}/prices/standard",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=20.0,
    )
    return r.status_code, _safe_json(r)


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {"raw": r.text[:600]}


async def run_bulk(
    *,
    token: str,
    item_ids: list[str],
    action: str,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run `action` on each `item_id` sequentially.

    Returns `{action, totals: {success, error}, results: [{item_id, status, http_status, body}]}`.

    For `change_price` and `change_qty`, the per-item payload can override
    the global `payload`:
      - bulk-uniform: `payload = {amount: 99.9, currency_id: "BRL"}`
      - per-item:    `payload = {by_item: {"MLB123": {"amount": 99.9}, ...}}`
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown_action: {action}")
    if not item_ids:
        return {"action": action, "totals": {"success": 0, "error": 0}, "results": []}

    payload = payload or {}
    by_item: dict[str, dict[str, Any]] = (payload.get("by_item") or {}) if isinstance(payload, dict) else {}

    results: list[dict[str, Any]] = []
    totals = {"success": 0, "error": 0}

    async with httpx.AsyncClient() as http:
        for idx, item_id in enumerate(item_ids):
            per = by_item.get(item_id, {})
            try:
                if action == "pause":
                    http_status, body = await _put_status(http, token, item_id, "paused")
                elif action == "activate":
                    http_status, body = await _put_status(http, token, item_id, "active")
                elif action == "close":
                    http_status, body = await _put_status(http, token, item_id, "closed")
                elif action == "change_qty":
                    qty = per.get("qty", payload.get("qty"))
                    if qty is None:
                        raise ValueError("change_qty: missing qty")
                    http_status, body = await _put_qty(http, token, item_id, int(qty))
                elif action == "change_price":
                    amount = per.get("amount", payload.get("amount"))
                    if amount is None:
                        raise ValueError("change_price: missing amount")
                    http_status, body = await _post_price_standard(
                        http, token, item_id,
                        amount=float(amount),
                        currency_id=per.get("currency_id") or payload.get("currency_id") or "BRL",
                        channel=per.get("channel") or payload.get("channel") or "marketplace",
                    )
                else:
                    raise ValueError(f"unhandled action {action}")

                status = "success" if http_status < 400 else "error"
                totals[status] += 1
                results.append({
                    "item_id": item_id,
                    "status": status,
                    "http_status": http_status,
                    "body": body,
                })
            except Exception as err:  # noqa: BLE001
                log.exception("bulk action %s on %s failed", action, item_id)
                totals["error"] += 1
                results.append({
                    "item_id": item_id,
                    "status": "error",
                    "http_status": 0,
                    "error": str(err),
                })
            await asyncio.sleep(THROTTLE_S)

    return {"action": action, "totals": totals, "results": results}
