"""Escalar "Atendimento": questions, returns and claims of the project.

Shaped like ML's own Atendimento screen. Everything comes from the caches
under the cabinet owner — `ml_user_questions` (pre-sale questions) and
`ml_user_claims` (claims, mediations, returns) — filtered to the caller's
scope through the listing each one is about.

Claims carry ML codes; this module turns them into what the seller reads:
the buyer's reason and how the case ended.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from v2.services import ml_access
from v2.services.escalar_overview import _norm

# reason_id prefixes ML uses on post-purchase claims.
REASONS = {
    "PDD": ("Produto diferente ou com defeito", "Товар не тот или с дефектом"),
    "PNR": ("Produto não recebido", "Товар не получен"),
    "CS": ("Cancelamento da compra", "Отмена покупки"),
}
# resolution.reason → what happened, in the seller's words.
OUTCOMES = {
    "item_returned": ("Produto devolvido", "Товар вернули"),
    "warehouse_decision": ("Decisão do armazém Full", "Решение склада Full"),
    "coverage_decision": ("Mercado Livre cobriu o valor", "ML покрыл сумму"),
    "no_bpp": ("Sem proteção ao comprador", "Без защиты покупателя"),
    "return_expired": ("Prazo de devolução expirou", "Срок возврата истёк"),
    "payment_refunded": ("Pagamento reembolsado", "Деньги возвращены"),
    "item_changed": ("Produto trocado", "Товар заменили"),
    "partial_refunded": ("Reembolso parcial", "Частичный возврат денег"),
    "opened_claim_by_mistake": ("Aberta por engano", "Открыта по ошибке"),
    "already_shipped": ("Já enviado", "Уже отправлен"),
}


def _j(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return {}
    return v or {}


def _dt(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str) and v:
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _iso(v: Any) -> Optional[str]:
    d = _dt(v)
    return d.isoformat() if d else None


def claim_view(c: dict) -> dict[str, Any]:
    """One claim as the screen shows it. Pure, for tests."""
    reason_id = str(c.get("reason_id") or "")
    prefix = next((p for p in REASONS if reason_id.startswith(p)), None)
    resolution = c.get("resolution") or {}
    outcome = OUTCOMES.get(resolution.get("reason") or "")
    benefited = resolution.get("benefited") or []
    kind = (
        "return" if c.get("type") == "returns" or "return" in (c.get("related_entities") or [])
        else "cancel" if str(c.get("type") or "").startswith("cancel")
        else "claim"
    )
    item = c.get("order_item") or {}
    returns = c.get("returns") or []
    ret = returns[0] if returns and isinstance(returns[0], dict) else {}
    return {
        "id": c.get("id"),
        "kind": kind,
        "type": c.get("type"),
        "stage": c.get("stage"),
        "status": c.get("status"),
        "orderId": c.get("resource_id"),
        "itemId": _norm(item.get("id")),
        "itemTitle": item.get("title"),
        "unitPrice": item.get("unit_price"),
        "quantity": item.get("quantity"),
        "reasonId": reason_id or None,
        "reason": REASONS[prefix] if prefix else None,
        "outcome": outcome or ((resolution.get("reason"),) * 2 if resolution.get("reason") else None),
        "closedBy": resolution.get("closed_by"),
        "sellerWon": "respondent" in benefited,
        "coverage": bool(resolution.get("applied_coverage")),
        "mediation": c.get("stage") == "dispute",
        "returnStatus": ret.get("status"),
        "needsAction": bool(c.get("needsAction")),
        "created": c.get("date_created"),
        "updated": c.get("last_updated"),
    }


async def load_atendimento(pool, scope, *, days: int = 30) -> dict[str, Any]:
    from v2.services import ml_user_claims as claims_svc
    from v2.services import ml_user_questions as questions_svc

    owner = scope.owner_id
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    prev_since = since - timedelta(days=days)
    await questions_svc.ensure_schema(pool)
    await claims_svc.ensure_schema(pool)

    async with pool.acquire() as conn:
        questions = await conn.fetch(
            """
            SELECT question_id, item_id, text, status, date_created,
                   answer_text, answer_date, from_nickname
              FROM ml_user_questions
             WHERE user_id = $1
               AND (status = 'UNANSWERED' OR date_created >= $2)
             ORDER BY date_created DESC
            """,
            owner, prev_since,
        )
        claims = await conn.fetch(
            """
            SELECT claim_id, type, status, stage, date_created, enriched
              FROM ml_user_claims
             WHERE user_id = $1 AND (status = 'opened' OR date_created >= $2)
             ORDER BY date_created DESC
            """,
            owner, prev_since,
        )
        items = await conn.fetch(
            """
            SELECT item_id, title, thumbnail, permalink, price, available_quantity,
                   status, (shipping_json->>'free_shipping')::boolean AS free_shipping,
                   shipping_json->>'logistic_type' AS logistic_type
              FROM ml_user_items WHERE user_id = $1
            """,
            owner,
        )
    meta = {str(r["item_id"]).upper(): dict(r) for r in items}
    skus = ml_access.item_skus(owner)

    def item_card(iid: str, fallback_title: Optional[str] = None) -> dict[str, Any]:
        m = meta.get(iid) or {}
        return {
            "itemId": iid,
            "title": m.get("title") or fallback_title,
            "thumbnail": m.get("thumbnail"),
            "permalink": m.get("permalink"),
            "price": float(m["price"]) if m.get("price") is not None else None,
            "stock": m.get("available_quantity"),
            "freeShipping": m.get("free_shipping"),
            "full": m.get("logistic_type") == "fulfillment",
            "sku": skus.get(iid),
        }

    # ── questions, grouped by listing ──
    groups: dict[str, dict[str, Any]] = {}
    answered_now: list[float] = []
    answered_cur = answered_prev = 0
    pending = 0
    for q in questions:
        iid = _norm(q["item_id"])
        if not scope.allows_read(iid):
            continue
        created, answered = _dt(q["date_created"]), _dt(q["answer_date"])
        if q["status"] == "UNANSWERED":
            pending += 1
        if answered and created:
            if answered >= since:
                answered_cur += 1
                answered_now.append((answered - created).total_seconds() / 3600)
            elif answered >= prev_since:
                answered_prev += 1
        if q["status"] != "UNANSWERED" and (not created or created < since):
            continue
        g = groups.setdefault(iid, {**item_card(iid), "questions": [], "canEdit": scope.allows_write(iid)})
        g["questions"].append({
            "id": q["question_id"],
            "text": q["text"],
            "status": q["status"],
            "created": _iso(q["date_created"]),
            "answer": q["answer_text"],
            "answered": _iso(q["answer_date"]),
            "from": q["from_nickname"],
            "waitingHours": round((now - created).total_seconds() / 3600, 1)
            if q["status"] == "UNANSWERED" and created else None,
        })
    # Listings with unanswered questions first, then the most recent.
    question_groups = sorted(
        groups.values(),
        key=lambda g: (
            sum(1 for x in g["questions"] if x["status"] == "UNANSWERED"),
            max((x["created"] or "") for x in g["questions"]),
        ),
        reverse=True,
    )

    # ── claims, returns, mediations ──
    out_claims: list[dict[str, Any]] = []
    closed_cur = closed_prev = 0
    for r in claims:
        c = _j(r["enriched"])
        v = claim_view({**c, "type": c.get("type") or r["type"], "status": c.get("status") or r["status"],
                        "stage": c.get("stage") or r["stage"]})
        if not v["itemId"] or not scope.allows_read(v["itemId"]):
            continue
        created = _dt(v["created"]) or _dt(r["date_created"])
        updated = _dt(v["updated"]) or created
        if v["status"] == "closed" and updated:
            if updated >= since:
                closed_cur += 1
            elif updated >= prev_since:
                closed_prev += 1
        if v["status"] != "opened" and (not created or created < since):
            continue
        v["item"] = item_card(v["itemId"], v["itemTitle"])
        out_claims.append(v)

    def change(cur: int, prev: int) -> Optional[float]:
        return round((cur - prev) / prev * 100, 1) if prev else None

    returns_open = [c for c in out_claims if c["kind"] == "return" and c["status"] == "opened"]
    return {
        "period": {"days": days, "from": since.date().isoformat(), "to": now.date().isoformat()},
        "summary": {
            "questionsPending": pending,
            "questionsAnswered": answered_cur,
            "questionsAnsweredChange": change(answered_cur, answered_prev),
            "responseMedianHours": round(statistics.median(answered_now), 1) if answered_now else None,
            "returnsOpen": len(returns_open),
            "claimsOpen": sum(1 for c in out_claims if c["kind"] != "return" and c["status"] == "opened"),
            "needsAction": sum(1 for c in out_claims if c["needsAction"]),
            "attended": closed_cur,
            "attendedChange": change(closed_cur, closed_prev),
        },
        "questions": question_groups,
        "claims": out_claims,
    }
