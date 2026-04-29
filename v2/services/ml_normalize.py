"""Python port of super-calculator-app/lib/escalar/notice-normalize.ts.

Takes a ML event (topic + resource + enriched payload) and produces a row
ready for upsert into ml_notices.
"""
from __future__ import annotations

import re
from typing import Any, Optional

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

    if topic in ("public_offers", "public_candidates"):
        # ML emits these topics for promotion invitations / active offers.
        # Resource id format: "{TYPE}-{MLB_ID}-{NUMERIC_ID}"
        #   public_offers:     "OFFER-MLB6143708560-12805121122"
        #   public_candidates: "CANDIDATE-MLB6143760498-75745703624"
        # Extract MLB to surface a useful label and link to /escalar/promotions
        # so the seller can act. The numeric tail is ML's internal offer/candidate
        # id and isn't directly usable as `promotion_id` in seller-promotions API
        # — that one comes from /seller-promotions/items/{mlb}, which the cache
        # service refreshes asynchronously.
        item_id = ""
        m = re.search(r"MLB\d+", str(rid))
        if m:
            item_id = m.group(0).upper()

        is_candidate = topic == "public_candidates"
        if is_candidate:
            label = "Você foi convidado para uma promoção"
            tag = "CANDIDATE"
        else:
            label = "Nova oferta de promoção disponível"
            tag = "OFFER"

        desc_lines: list[str] = []
        if item_id:
            desc_lines.append(f"Item: {item_id}")
        desc_lines.append("Toque em \"Detalhes\" para ver desconto e aceitar.")

        item_url = (
            f"https://produto.mercadolivre.com.br/MLB-{item_id[3:]}" if item_id else ""
        )
        actions: list[dict] = []
        if item_url:
            actions.append({"label": "Abrir item", "url": item_url})

        return {
            **base,
            "label": label,
            "description": "\n".join(desc_lines),
            "from_date": enriched.get("date_created") or enriched.get("created_at"),
            "tags": [t for t in ["PROMOTIONS", tag] if t],
            "actions": actions,
        }

    if topic == "promotions":
        # Enriched payload here is the raw offer dict from
        # /seller-promotions/items/{mlb}?app_version=v2 plus item_id and
        # _item_{title,permalink,thumbnail} injected by _refresh_promotions_job
        # in main.py from the ml_user_items cache.
        promo_id = (
            str(enriched.get("id") or enriched.get("promotion_id") or rid).strip()
        )
        promo_type = str(enriched.get("type") or enriched.get("promotion_type") or "").upper()
        sub_type = str(enriched.get("sub_type") or "").upper()
        status = str(enriched.get("status") or "candidate").lower()
        item_id = str(enriched.get("item_id") or "").upper()
        item_title = str(enriched.get("_item_title") or "")
        item_permalink = str(enriched.get("_item_permalink") or "")
        original_price = enriched.get("original_price")
        # ML uses different field names per offer type:
        #   DEAL/SELLER_CAMPAIGN  → deal_price
        #   LIGHTNING/SMART/PRICE_MATCHING/UNHEALTHY_STOCK → price or
        #     suggested_discounted_price
        deal_price = (
            enriched.get("deal_price")
            or enriched.get("price")
            or enriched.get("suggested_discounted_price")
        )
        discount_pct = enriched.get("discount_percentage")
        if discount_pct is None:
            mp = enriched.get("meli_percentage")
            sp = enriched.get("seller_percentage")
            if mp is not None or sp is not None:
                try:
                    discount_pct = round((float(mp or 0) + float(sp or 0)), 1)
                except (TypeError, ValueError):
                    discount_pct = None
        if discount_pct is None and original_price and deal_price:
            try:
                discount_pct = round(
                    (1 - float(deal_price) / float(original_price)) * 100, 1,
                )
            except (ZeroDivisionError, TypeError, ValueError):
                discount_pct = None
        start_date = enriched.get("start_date")
        finish_date = enriched.get("finish_date")

        # type_friendly выбирается с учётом статуса:
        # - candidate: SMART/UNHEALTHY ещё НЕ активна, ML может опт-ин если ничего
        #   не делать. Текущее «ativada automaticamente» сбивает с толку (читается
        #   как «уже активна, без согласия»). Делаем явно: «pode ser ativada
        #   automaticamente se você não decidir».
        # - started: акция РЕАЛЬНО включена (если SMART — ML её auto-opt-in).
        #   Пользователю нужны кнопки «Выйти / Поднять цену», не «Принять».
        is_started = status == "started"
        if is_started:
            type_friendly = {
                "DEAL": "Oferta do dia (ativa)",
                "DOD": "Oferta do dia (ativa)",
                "LIGHTNING": "Promoção Relâmpago (ativa)",
                "SELLER_CAMPAIGN": "Campanha do vendedor (ativa)",
                "PRICE_DISCOUNT": "Desconto de preço (ativo)",
                "PRICE_MATCHING": "Pareamento de preço (ativo, reduz tarifas)",
                "MARKETPLACE_CAMPAIGN": "Campanha Mercado Livre (ativa)",
                "VOLUME": "Desconto por volume (ativo)",
                "SMART": "Promoção SMART ATIVADA automaticamente pela ML",
                "UNHEALTHY_STOCK": "Estoque parado no Full (promoção ativa)",
            }.get(promo_type, (promo_type or "Promoção") + " (ativa)")
        else:
            type_friendly = {
                "DEAL": "Oferta do dia",
                "DOD": "Oferta do dia",
                "LIGHTNING": "Promoção Relâmpago",
                "SELLER_CAMPAIGN": "Campanha do vendedor",
                "PRICE_DISCOUNT": "Desconto de preço",
                "PRICE_MATCHING": "Pareamento de preço (reduz tarifas)",
                "MARKETPLACE_CAMPAIGN": "Campanha Mercado Livre",
                "VOLUME": "Desconto por volume",
                # Для candidate явно говорим что ЕЩЁ НЕ активна — но если seller
                # не примет решение, ML может сама опт-ин: «pode ser ativada».
                "SMART": "Promoção SMART (pode ser ativada automaticamente pela ML)",
                "UNHEALTHY_STOCK": "Estoque parado no Full",
            }.get(promo_type, promo_type or "Promoção")
        promo_name = str(enriched.get("name") or "").strip()
        meli_pct = enriched.get("meli_percentage")
        seller_pct = enriched.get("seller_percentage")
        max_disc = enriched.get("max_discounted_price")
        min_disc = enriched.get("min_discounted_price")
        suggested = enriched.get("suggested_discounted_price")
        status_friendly = {
            "candidate": "candidato",
            "started": "em andamento",
            "finished": "finalizado",
            "pending": "pendente",
        }.get(status, status)

        if discount_pct:
            label = f"Nova promoção: {type_friendly} −{discount_pct}%"
        else:
            label = f"Nova promoção: {type_friendly} ({status_friendly})"

        def _money(v: Any) -> Optional[str]:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f"R$ {f:.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        def _short_date(v: Any) -> Optional[str]:
            if not v:
                return None
            s = str(v)[:10]
            try:
                from datetime import datetime as _dt
                return _dt.fromisoformat(s).strftime("%d/%m/%Y")
            except Exception:  # noqa: BLE001
                return s

        desc_lines: list[str] = []
        if item_title:
            t = item_title if len(item_title) <= 90 else item_title[:87] + "..."
            desc_lines.append(f"📦 {t}")
        desc_lines.append(f"🆔 {item_id}")
        desc_lines.append("")
        desc_lines.append(f"🏷 Tipo: {type_friendly}")
        if promo_name and promo_name.lower() != type_friendly.lower():
            desc_lines.append(f"📌 Promoção: {promo_name}")
        desc_lines.append(f"📊 Status: {status_friendly}")
        desc_lines.append("")

        orig_str = _money(original_price)
        sug_str = _money(suggested)
        max_str = _money(max_disc)  # max_discounted_price = MINIMUM discount
        min_str = _money(min_disc)  # min_discounted_price = MAXIMUM discount

        def _pct_off(price):
            if not price or not original_price:
                return None
            try:
                return round((1 - float(price) / float(original_price)) * 100, 1)
            except (ZeroDivisionError, TypeError, ValueError):
                return None

        if orig_str:
            desc_lines.append(f"💰 Preço atual: {orig_str}")

        # Range view — only when ML returns a flexible window. Three points:
        # max_discounted_price = entry ticket (smallest discount needed)
        # suggested_discounted_price = ML recommendation
        # min_discounted_price = deepest allowed discount
        if max_str or sug_str or min_str:
            desc_lines.append("")
            desc_lines.append("🎯 Faixa permitida pela ML:")
            if max_str:
                pct = _pct_off(max_disc)
                pct_str = f" (−{pct}%)" if pct else ""
                desc_lines.append(f"   • Entrada (mín. desc.): {max_str}{pct_str}")
            if sug_str:
                pct = _pct_off(suggested)
                pct_str = f" (−{pct}%)" if pct else ""
                desc_lines.append(f"   • Sugerido pela ML:    {sug_str}{pct_str}")
            if min_str:
                pct = _pct_off(min_disc)
                pct_str = f" (−{pct}%)" if pct else ""
                desc_lines.append(f"   • Máx. desconto:       {min_str}{pct_str}")

        # Quem-paga breakdown — most useful for SMART / UNHEALTHY_STOCK /
        # PRICE_MATCHING where ML and seller split the discount.
        try:
            mp = float(enriched.get("meli_percentage")) if enriched.get("meli_percentage") is not None else None
            sp = float(enriched.get("seller_percentage")) if enriched.get("seller_percentage") is not None else None
        except (TypeError, ValueError):
            mp, sp = None, None
        if (mp is not None and mp > 0) or (sp is not None and sp > 0):
            parts: list[str] = []
            if sp is not None and sp > 0:
                parts.append(f"você cobre {sp:g}%")
            if mp is not None and mp > 0:
                parts.append(f"ML cobre {mp:g}%")
            if parts:
                desc_lines.append("📐 Custo do desconto: " + " + ".join(parts))

        # Fee-reduction badge — PRICE_MATCHING and any promo whose name
        # mentions "reduza tarifas" carry a side benefit of lower ML fees
        # while the seller is participating.
        name_lower = promo_name.lower()
        if (
            promo_type == "PRICE_MATCHING"
            or "reduza tarifa" in name_lower
            or "reduza suas tarifa" in name_lower
        ):
            desc_lines.append("🎁 *Esta promoção reduz suas tarifas de venda*")

        s_str = _short_date(start_date)
        f_str = _short_date(finish_date)
        if s_str and f_str:
            desc_lines.append(f"📅 Período: {s_str} → {f_str}")
        elif f_str:
            desc_lines.append(f"📅 Termina: {f_str}")
        elif promo_type == "LIGHTNING":
            desc_lines.append("📅 Datas: a ML decide próximo ao início")

        # Margin block — read from ml_item_margin_cache (refreshed nightly).
        # Same opex formulas as the OPiU dashboard, allocated per-item.
        margin = enriched.get("_margin_3m") or None
        margin_at_min = enriched.get("_margin_at_min_discount") or {}
        margin_at_sug = enriched.get("_margin_at_suggested") or {}
        margin_at_max = enriched.get("_margin_at_max_discount") or {}
        if margin is None:
            desc_lines.append("")
            desc_lines.append("📊 Margem 3M: calculando (próxima atualização noturna)")
        elif margin.get("ok"):
            units = margin.get("units_sold")
            now_pct = margin.get("margin_pct")
            desc_lines.append("")
            if margin.get("missing_cost"):
                desc_lines.append(
                    f"📊 Margem 3M: indisponível — cadastre custo do produto"
                    f" ({units} un. vendidas)"
                )
            elif now_pct is not None:
                # Two views per ML product: full PnL margin (with all overhead
                # incl. fixed Aluguel) and unit economics (variable per-sale
                # costs only). Promo decisions usually need the unit number;
                # the PnL number is shown for accounting consistency.
                unit = margin.get("unit") or {}
                unit_pct = unit.get("margin_pct")
                unit_profit = unit.get("profit_per_unit")
                desc_lines.append(f"📊 *Margem (PnL): {now_pct}%* ({units} un. vendidas)")
                lucro = margin.get("net_profit")
                if lucro is not None:
                    desc_lines.append(f"💵 Lucro líquido total: {_money(lucro) or '—'}")
                if unit_pct is not None:
                    desc_lines.append("")
                    desc_lines.append(f"📐 *Margem unitária: {unit_pct}%* (apenas custos variáveis)")
                    if unit_profit is not None:
                        desc_lines.append(f"💰 Lucro por unidade: {_money(unit_profit) or '—'}")
                # Show unit margin at each price point in the range.
                pts: list[tuple[str, dict]] = []
                if margin_at_min.get("ok"):
                    pts.append(("entrada", margin_at_min))
                if margin_at_sug.get("ok"):
                    pts.append(("sugerido", margin_at_sug))
                if margin_at_max.get("ok"):
                    pts.append(("máx desc", margin_at_max))
                if pts:
                    desc_lines.append("")
                    desc_lines.append("📈 Margem unitária após promo:")
                    for label_pt, m in pts:
                        u = m.get("unit") or {}
                        u_pct = u.get("margin_pct")
                        u_profit = u.get("profit_per_unit")
                        if u_pct is None:
                            continue
                        line = f"   • {label_pt}: {u_pct}%"
                        if u_profit is not None:
                            line += f" (lucro/un. {_money(u_profit) or '—'})"
                        desc_lines.append(line)
        elif margin.get("error") == "no_sales_in_period":
            desc_lines.append("")
            desc_lines.append("📊 Sem vendas nos últimos 3 meses — margem indisponível")
        elif margin.get("error") in ("no_vendas_data", "vendas_load_failed"):
            desc_lines.append("")
            desc_lines.append("📊 Margem indisponível: carregue relatório Vendas ML")

        if status == "candidate":
            desc_lines.append("")
            # Для SMART/UNHEALTHY кандидата явно предупреждаем что если seller
            # не примет решение — ML может опт-ин активировать сама. Для
            # остальных типов это не происходит, обычная подсказка.
            if promo_type in ("SMART", "UNHEALTHY_STOCK"):
                desc_lines.append("⚠ Se você não decidir, ML pode ativar automaticamente")
                desc_lines.append("⏰ Aceite ou rejeite agora")
            else:
                desc_lines.append("⏰ Aceite agora para participar")
        elif status == "started":
            desc_lines.append("")
            # Для started SMART подчёркиваем что ML её саму активировала, и
            # предлагаем 2 действия: выйти из акции или поднять цену чтобы
            # покупатель видел тот же оригинальный price (даже после скидки ML).
            if promo_type in ("SMART", "UNHEALTHY_STOCK"):
                desc_lines.append("🤖 ML ativou automaticamente. Quer sair ou subir o preço?")
            else:
                desc_lines.append("✅ Promoção em andamento")

        return {
            **base,
            "label": label,
            "description": "\n".join(desc_lines),
            "from_date": enriched.get("created_at") or enriched.get("start_date"),
            "tags": [t for t in ["PROMOTIONS", promo_type, sub_type, status.upper()] if t],
            # Inline TG buttons are added by telegram_notify.send_notice when
            # topic == "promotions". Actions list stays empty so the message
            # body doesn't double the link.
            "actions": (
                [{"label": "Abrir produto", "url": item_permalink}]
                if item_permalink else []
            ),
        }

    if topic == "sales":
        # Per-sale TG-уведомление (см. ml_orders.dispatch_pending_sales).
        # enriched: {order_id, item_id, title, qty, sale_price, ml_fee,
        #            currency, total_amount, _margin (apply_hypothetical_price
        #            output), _permalink}
        order_id = str(enriched.get("order_id") or rid).strip()
        item_id = str(enriched.get("item_id") or "").upper()
        title = str(enriched.get("title") or "")
        qty = int(enriched.get("qty") or 1)
        try:
            sale_price = float(enriched.get("sale_price") or 0.0)
        except (TypeError, ValueError):
            sale_price = 0.0
        try:
            ml_fee = float(enriched.get("ml_fee") or 0.0)
        except (TypeError, ValueError):
            ml_fee = 0.0
        try:
            total_amount = float(enriched.get("total_amount") or 0.0)
        except (TypeError, ValueError):
            total_amount = sale_price * qty
        permalink = str(enriched.get("_permalink") or "")

        def _money(v: Any) -> str:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return "—"
            return f"R$ {f:.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        revenue = sale_price * qty if sale_price > 0 else total_amount

        # Margin block — берём re-derived от sale_price.
        margin = enriched.get("_margin") or {}
        unit = (margin.get("unit") or {}) if isinstance(margin, dict) else {}
        unit_profit = unit.get("profit_per_unit")
        unit_margin_pct = unit.get("margin_pct")
        cogs_pu = unit.get("cogs_per_unit")
        ml_fee_pu = unit.get("ml_fee_per_unit")
        envios_pu = unit.get("envios_per_sale")
        ful_pu = unit.get("fulfillment_per_sale")
        das_pu = unit.get("das_per_unit")
        armaz_pu = unit.get("armaz_per_unit")

        desc_lines: list[str] = []
        if title:
            t = title if len(title) <= 90 else title[:87] + "..."
            desc_lines.append(f"📦 {t}")
        if item_id:
            desc_lines.append(f"🆔 {item_id}")
        desc_lines.append(f"🧾 Pedido: {order_id}")
        desc_lines.append("")

        # Revenue line.
        if qty > 1:
            desc_lines.append(f"💵 Receita: {_money(revenue)}  ({qty} un. × {_money(sale_price)})")
        else:
            desc_lines.append(f"💵 Receita: {_money(revenue)}")

        # Profit block.
        if unit_profit is not None and qty > 0:
            total_profit = float(unit_profit) * qty
            margin_str = f"{unit_margin_pct}%" if unit_margin_pct is not None else "—"
            desc_lines.append("")
            desc_lines.append(f"📊 *Lucro líquido: {_money(total_profit)}* ({margin_str})")
            # Per-unit breakdown — показывает откуда пришёл profit.
            cost_lines: list[str] = []
            if cogs_pu is not None:
                cost_lines.append(f"   • Custo produto: {_money(float(cogs_pu) * qty)}")
            if ml_fee_pu is not None:
                cost_lines.append(f"   • Tarifa ML: {_money(float(ml_fee_pu) * qty)}")
            elif ml_fee > 0:
                cost_lines.append(f"   • Tarifa ML: {_money(ml_fee)}")
            if envios_pu is not None and float(envios_pu) > 0:
                cost_lines.append(f"   • Envios: {_money(float(envios_pu) * qty)}")
            if ful_pu is not None and float(ful_pu) > 0:
                cost_lines.append(f"   • Fulfillment: {_money(float(ful_pu) * qty)}")
            if armaz_pu is not None and float(armaz_pu) > 0:
                cost_lines.append(f"   • Armazenagem: {_money(float(armaz_pu) * qty)}")
            if das_pu is not None and float(das_pu) > 0:
                # das_rate в кэше — это уже EFFECTIVE ставка от compute_das
                # (учитывает RBT12 / faixa / Anexo), не номинальная 4.5%.
                # Показываем процент в подписи чтобы пользователь видел реальную
                # налоговую нагрузку Simples Nacional, а не «как-будто 4.5%».
                das_rate = unit.get("das_rate")
                if das_rate is not None and float(das_rate) > 0:
                    pct = float(das_rate) * 100
                    cost_lines.append(f"   • DAS ({pct:.2f}% efetivo): {_money(float(das_pu) * qty)}")
                else:
                    cost_lines.append(f"   • DAS: {_money(float(das_pu) * qty)}")
            desc_lines.extend(cost_lines)
        else:
            desc_lines.append("")
            desc_lines.append("📊 Margem indisponível — preencha custo do produto em /finance/sku-mapping")

        if unit_margin_pct is not None and float(unit_margin_pct) < 0:
            desc_lines.append("")
            desc_lines.append("⚠ *Esta venda foi negativa* — produto vendido abaixo do custo")

        actions: list[dict] = []
        if permalink:
            actions.append({"label": "Ver pedido", "url": permalink})

        if sale_price > 0:
            label = f"Nova venda: {_money(revenue)}"
            if unit_margin_pct is not None:
                label += f" ({unit_margin_pct}% margem)"
        else:
            label = f"Nova venda: pedido {order_id}"

        return {
            **base,
            "label": label,
            "description": "\n".join(desc_lines),
            "from_date": enriched.get("date_created"),
            "tags": [t for t in ["SALES", item_id] if t],
            "actions": actions,
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
