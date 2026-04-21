"""Python port of super-calculator-app/lib/escalar/notice-normalize.ts.

Takes a ML event (topic + resource + enriched payload) and produces a row
ready for upsert into ml_notices.
"""
from __future__ import annotations

import re
from typing import Any

_RESOURCE_ID_RE = re.compile(r"/([^/?]+)(?:\?|$)")


def _resource_id(resource: str | None) -> str:
    if not resource:
        return "unknown"
    matches = _RESOURCE_ID_RE.findall(resource)
    if not matches:
        return resource
    last = matches[-1]
    return last or resource


def _brl(value: Any) -> str:
    try:
        n = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return "—"
    if n != n:  # NaN
        return "—"
    # pt-BR format: R$ 1.234,56
    formatted = f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def normalize_event(topic: str, resource: str | None, enriched: dict[str, Any]) -> dict[str, Any]:
    enriched = enriched or {}
    rid = _resource_id(resource)
    notice_id = f"{topic}:{rid}"

    base = {
        "notice_id": notice_id,
        "topic": topic,
        "resource": resource,
        "raw": enriched,
    }

    if topic in ("orders_v2", "orders"):
        total = enriched.get("total_amount") or enriched.get("paid_amount") or enriched.get("transaction_amount")
        status = str(enriched.get("status") or "unknown")
        order_items = enriched.get("order_items") or enriched.get("items") or []
        item_title = ""
        if order_items:
            first = order_items[0] or {}
            item_title = (first.get("item") or {}).get("title") or first.get("title") or ""
        buyer_obj = enriched.get("buyer") or {}
        buyer = buyer_obj.get("nickname") or buyer_obj.get("first_name") or ""
        shipping = enriched.get("shipping") or {}
        pack_id = enriched.get("pack_id")
        desc_lines = [
            f"Produto: {item_title}" if item_title else "",
            f"Comprador: {buyer}" if buyer else "",
            f"Envio: {shipping.get('status')}" if shipping.get("status") else "",
            f"Pack #{pack_id}" if pack_id else "",
        ]
        permalink = (
            f"https://www.mercadolivre.com.br/vendas/{enriched.get('id')}/detalhe"
            if enriched.get("id")
            else ""
        )
        return {
            **base,
            "label": f"Nova venda {_brl(total)} — {status}",
            "description": "\n".join(x for x in desc_lines if x),
            "from_date": enriched.get("date_created") or enriched.get("last_updated"),
            "tags": [t for t in ["ORDERS", status.upper()] if t],
            "actions": [{"label": "Ver pedido", "url": permalink}] if permalink else [],
        }

    if topic in ("questions", "questions_v2"):
        text = str(enriched.get("text") or "")
        item_id = enriched.get("item_id") or (enriched.get("item") or {}).get("id") or ""
        status = str(enriched.get("status") or "UNANSWERED").upper()
        answer = enriched.get("answer") or {}
        answer_text = str(answer.get("text") or "") if isinstance(answer, dict) else ""
        answer_date = (answer.get("date_created") or "") if isinstance(answer, dict) else ""
        permalink = (
            f"https://www.mercadolivre.com.br/anuncios/{item_id}/perguntas" if item_id else ""
        )

        desc_lines: list[str] = []
        if text:
            desc_lines.append(f"❓ {text}")
        if answer_text:
            date_label = f" ({str(answer_date)[:10]})" if answer_date else ""
            desc_lines.append(f"✅ Resposta{date_label}: {answer_text}")

        action_label = "Ver pergunta" if status == "ANSWERED" else "Responder"
        label_suffix = " (respondida)" if status == "ANSWERED" else ""
        return {
            **base,
            "label": f"Pergunta sobre {item_id or 'anúncio'}{label_suffix}",
            "description": "\n\n".join(desc_lines),
            "from_date": enriched.get("date_created"),
            "tags": ["QUESTIONS", status],
            "actions": [{"label": action_label, "url": permalink}] if permalink else [],
        }

    if topic == "claims":
        type_ = str(enriched.get("type") or enriched.get("claim_type") or "")
        stage = str(enriched.get("stage") or enriched.get("status") or "")
        reason = enriched.get("reason_id") or (enriched.get("reason") or {}).get("name") or ""
        claim_id = enriched.get("resource_id") or enriched.get("id") or rid
        returns = enriched.get("returns") if isinstance(enriched.get("returns"), list) else []
        extra_tags: list[str] = []
        desc_lines = [
            f"Motivo: {reason}" if reason else "",
            f"Tipo: {type_ or 'reclamação'}",
            f"Estágio: {stage}" if stage else "",
        ]

        if returns:
            extra_tags.append("RETURN")
            ret = returns[0] or {}
            ret_parts = []
            if ret.get("status"): ret_parts.append(f"status: {ret['status']}")
            if ret.get("subtype"): ret_parts.append(f"tipo: {ret['subtype']}")
            if ret.get("status_money"): ret_parts.append(f"dinheiro: {ret['status_money']}")
            if ret.get("refund_at"): ret_parts.append(f"reembolso: {ret['refund_at']}")
            if ret_parts:
                desc_lines.append(f"Devolução — {' · '.join(ret_parts)}")
            extra_tags.append(str(ret.get("status") or "").upper())

            shipments = ret.get("shipments") if isinstance(ret.get("shipments"), list) else []
            shipment = shipments[0] if shipments else None
            if shipment:
                sparts = []
                if shipment.get("status"): sparts.append(f"envio: {shipment['status']}")
                if shipment.get("tracking_number"): sparts.append(f"rastreio: {shipment['tracking_number']}")
                dest = (shipment.get("destination") or {}).get("name")
                if dest: sparts.append(f"destino: {dest}")
                if sparts:
                    desc_lines.append(" · ".join(sparts))

            reviews = ret.get("reviews") if isinstance(ret.get("reviews"), list) else []
            review = reviews[0] if reviews else None
            rr = (review.get("resource_reviews") or [None])[0] if review else None
            if rr:
                rparts = []
                if rr.get("status"): rparts.append(f"triagem: {rr['status']}")
                if rr.get("product_condition"): rparts.append(f"condição: {rr['product_condition']}")
                if rr.get("reason_id"): rparts.append(f"motivo: {rr['reason_id']}")
                if rr.get("benefited"): rparts.append(f"beneficiário: {rr['benefited']}")
                if rr.get("seller_status"): rparts.append(f"vendedor: {rr['seller_status']}")
                if rparts:
                    desc_lines.append(f"Triagem ML — {' · '.join(rparts)}")
                extra_tags.extend(["TRIAGE", str(rr.get("status") or "").upper()])

        return {
            **base,
            "label": f"Reclamação: {stage or type_ or rid}",
            "description": "\n".join(x for x in desc_lines if x),
            "from_date": enriched.get("date_created") or enriched.get("last_updated"),
            "tags": [t for t in ["CLAIMS", type_.upper(), stage.upper(), *extra_tags] if t],
            "actions": [{
                "label": "Abrir reclamação",
                "url": f"https://www.mercadolivre.com.br/vendas/centro-de-reclamacoes/{claim_id}",
            }],
        }

    if topic == "items":
        status = str(enriched.get("status") or "")
        sub_status = enriched.get("sub_status") or []
        if not isinstance(sub_status, list):
            sub_status = []
        title = enriched.get("title") or rid
        permalink = enriched.get("permalink") or ""
        if status == "paused":
            label = f"Anúncio pausado: {title}"
        elif status == "closed":
            label = f"Anúncio finalizado: {title}"
        else:
            label = f"{title} — {status}"
        desc_lines: list[str] = []
        if sub_status:
            desc_lines.append(f"Motivos (sub_status): {', '.join(str(s) for s in sub_status)}")
        else:
            desc_lines.append(f"Status: {status}")
        if permalink:
            desc_lines.append(permalink)
        return {
            **base,
            "label": label,
            "description": "\n\n".join(desc_lines),
            "from_date": enriched.get("last_updated"),
            "tags": [
                t for t in ["ITEMS", status.upper(), *[str(s).upper() for s in sub_status]] if t
            ],
            "actions": [{"label": "Abrir no ML", "url": permalink}] if permalink else [],
        }

    if topic == "messages":
        text = str(enriched.get("text") or enriched.get("message") or "")
        from_user = (enriched.get("from") or {}).get("user_id") or enriched.get("from_user_id") or ""
        pack_id = enriched.get("pack_id")
        permalink = f"https://www.mercadolivre.com.br/mensagens/{pack_id}" if pack_id else ""
        desc = "\n\n".join(x for x in [f"De: {from_user}" if from_user else "", text] if x)
        return {
            **base,
            "label": "Nova mensagem do comprador",
            "description": desc,
            "from_date": enriched.get("date_created"),
            "tags": ["MESSAGES"],
            "actions": [{"label": "Abrir conversa", "url": permalink}] if permalink else [],
        }

    # Fallback
    return {
        **base,
        "label": str(enriched.get("label") or enriched.get("title") or topic),
        "description": str(enriched.get("description") or enriched.get("text") or ""),
        "from_date": enriched.get("date_created") or enriched.get("from_date"),
        "tags": ["RAW", topic.upper()],
        "actions": [],
    }
