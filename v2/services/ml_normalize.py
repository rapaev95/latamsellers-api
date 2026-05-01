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
        # Enrichment failure: ML API вернул empty/partial payload и у нас
        # нет ни items, ни buyer, ни total. Backfill cron часто такой видит
        # для очень старых orders. Возвращаем placeholder со SKIP_TG тэгом
        # — TG не отправит, БД-запись остаётся для аналитики.
        items_check = enriched.get("order_items") or enriched.get("items") or []
        if (
            status.lower() == "unknown"
            and (not total or float(total or 0) == 0)
            and not items_check
        ):
            return {
                **base,
                "label": f"Pedido placeholder (sem dados)",
                "description": "Enrichment não retornou order details (ML API empty/stale).",
                "from_date": enriched.get("date_created") or enriched.get("last_updated"),
                "tags": ["ORDERS", "SKIP_TG", "ENRICHMENT_FAILED"],
                "actions": [],
            }
        # Skip cancelled/invalid — это не реальная продажа, бесполезный TG-noise.
        # ml_notices хранит запись для аналитики, но label/description короткий
        # без кнопок (telegram_notify не отправит keyboard если sale_price = 0).
        if status.lower() in ("cancelled", "invalid"):
            permalink_x = (
                f"https://www.mercadolivre.com.br/vendas/{enriched.get('id')}/detalhe"
                if enriched.get("id") else ""
            )
            return {
                **base,
                "label": f"Pedido cancelado {_brl(total)}",
                "description": "Status: cancelado / inválido — não conta como venda.",
                "from_date": enriched.get("date_created") or enriched.get("last_updated"),
                "tags": ["ORDERS", "CANCELLED"],
                "actions": [{"label": "Ver pedido", "url": permalink_x}] if permalink_x else [],
            }
        order_items = enriched.get("order_items") or enriched.get("items") or []
        first = (order_items[0] or {}) if order_items else {}
        # Поддерживаем оба shape: ml_backfill использует {item: {...}, quantity, unit_price};
        # ml_orders использует {mlb, title, quantity, unit_price} (slim).
        inner = first.get("item") if isinstance(first.get("item"), dict) else first
        item_title = (inner.get("title") if isinstance(inner, dict) else "") or first.get("title") or ""
        item_id = (inner.get("id") if isinstance(inner, dict) else "") or first.get("mlb") or ""
        try:
            qty = int(first.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1
        try:
            sale_price = float(first.get("unit_price") or 0.0)
        except (TypeError, ValueError):
            sale_price = 0.0
        buyer_obj = enriched.get("buyer") or {}
        buyer = buyer_obj.get("nickname") or buyer_obj.get("first_name") or ""
        shipping = enriched.get("shipping") or {}
        pack_id = enriched.get("pack_id")

        def _money(v: Any) -> str:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return "—"
            return f"R$ {f:.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # Margin block — если ml_backfill / ml_orders инжектил `_margin`
        # (apply_hypothetical_price от sale_price), показываем profit + breakdown.
        # Если нет — оставляем legacy формат (только Produto / Comprador / Pack).
        margin = enriched.get("_margin") or {}
        unit = (margin.get("unit") or {}) if isinstance(margin, dict) else {}
        # Сессия B/C: предпочитаем true variable margin вместо legacy net.
        # profit_variable = price - (cogs + ml_fee + envios + das)
        # margin_variable_pct — true per-sale economics («стоит ли продавать»).
        # legacy profit_per_unit / margin_pct включают fulfillment+armaz —
        # сохранены для back-compat но в TG показываем variable как primary.
        profit_variable_pu = unit.get("profit_variable")
        margin_variable_pct = unit.get("margin_variable_pct")
        # Net (после fixed overhead allocation) — second-tier metric.
        profit_net_pu = unit.get("profit_net_per_unit")
        margin_net_pct = unit.get("margin_net_pct")
        # Fallback на legacy если variable не доступен (старый cache).
        unit_profit = profit_variable_pu if profit_variable_pu is not None else unit.get("profit_per_unit")
        unit_margin_pct = margin_variable_pct if margin_variable_pct is not None else unit.get("margin_pct")

        desc_lines: list[str] = []
        if item_title:
            t_short = item_title if len(item_title) <= 90 else item_title[:87] + "..."
            desc_lines.append(f"Produto: {t_short}")
        if buyer:
            desc_lines.append(f"Comprador: {buyer}")
        if shipping.get("status"):
            desc_lines.append(f"Envio: {shipping.get('status')}")
        if pack_id:
            desc_lines.append(f"Pack #{pack_id}")

        # Attribution — organic vs ads. ML order payload иногда включает:
        #   - context.flow == 'mshop_advertising' → product ads
        #   - tags array contains 'advertising' / 'meli_advertising' / similar
        #   - mediations[].marketplace_promotional_offer для product ads
        # Если нашли — пишем «📣 De Ads», иначе «🌱 Orgânica». Не показываем
        # если данных недостаточно для вывода (NULL).
        ctx_flow = ""
        if isinstance(enriched.get("context"), dict):
            ctx_flow = str(enriched["context"].get("flow") or "").lower()
        order_tags = enriched.get("tags") or []
        if isinstance(order_tags, list):
            tags_lower = " ".join(str(t).lower() for t in order_tags if t)
        else:
            tags_lower = ""
        is_ads = (
            "advertising" in ctx_flow
            or "advertising" in tags_lower
            or "ads" in ctx_flow
            or "promotional_offer" in tags_lower
        )
        if is_ads:
            desc_lines.append("📣 *Vinda de Ads* (campanha pago)")
        elif ctx_flow or tags_lower:
            # У нас есть какой-то context => уверенно органика
            desc_lines.append("🌱 *Vinda orgânica*")

        # Profit / margin block (когда _margin доступен из enricher).
        # Показываем VARIABLE margin (true per-sale) как primary — это
        # «стоит ли продавать ЭТУ единицу». Net margin (после overhead)
        # отображается только в P&L reports, не в TG.
        if unit_profit is not None and qty > 0 and sale_price > 0:
            total_profit_var = float(unit_profit) * qty
            margin_str = f"{unit_margin_pct}%" if unit_margin_pct is not None else "—"
            desc_lines.append("")
            desc_lines.append(f"📊 *Margem variável: {margin_str} ({_money(total_profit_var)})*")
            # Per-unit breakdown — только variable costs (cogs+ml_fee+envios+das).
            cogs_pu = unit.get("cogs_per_unit")
            ml_fee_pu = unit.get("ml_fee_per_unit")
            envios_pu = unit.get("envios_per_sale")
            das_pu = unit.get("das_per_unit")
            das_rate = unit.get("das_rate")
            cost_lines: list[str] = []
            if cogs_pu is not None:
                cost_lines.append(f"   • Custo produto: {_money(float(cogs_pu) * qty)}")
            if ml_fee_pu is not None:
                cost_lines.append(f"   • Tarifa ML: {_money(float(ml_fee_pu) * qty)}")
            if envios_pu is not None and float(envios_pu) > 0:
                cost_lines.append(f"   • Envios: {_money(float(envios_pu) * qty)}")
            if das_pu is not None and float(das_pu) > 0:
                if das_rate is not None and float(das_rate) > 0:
                    pct = float(das_rate) * 100
                    cost_lines.append(f"   • DAS ({pct:.2f}% efetivo): {_money(float(das_pu) * qty)}")
                else:
                    cost_lines.append(f"   • DAS: {_money(float(das_pu) * qty)}")
            desc_lines.extend(cost_lines)
            if unit_margin_pct is not None and float(unit_margin_pct) < 0:
                desc_lines.append("")
                desc_lines.append("⚠ *Esta venda é negativa* — preço abaixo do custo variável")

            # ── Net margin block (после fixed overhead allocation) ──────
            # Net = variable - fixed_overhead_per_unit. Показывает реальную
            # прибыль после распределения publi/armaz/aluguel/fulfillment/
            # manual fixed costs на каждую единицу из месячного объёма.
            #
            # В TG показываем monthly project total (real BRL/мес) — per-unit
            # цифра ничего не говорит продавцу. Breakdown — только non-zero
            # статьи, чтобы видно что реально кушает в проекте.
            if profit_net_pu is not None and qty > 0:
                total_profit_net = float(profit_net_pu) * qty
                net_str = f"{margin_net_pct}%" if margin_net_pct is not None else "—"
                emoji = "💰" if (margin_net_pct or 0) >= 0 else "📉"
                desc_lines.append("")
                desc_lines.append(
                    f"{emoji} *Lucro líquido: {net_str} ({_money(total_profit_net)})*"
                )
                # Monthly fixed costs — только то что пользователь явно ввёл
                # в проекте (`fixed_costs_monthly` через UI). НЕ подмешиваем
                # computed shares (publi из ads, armaz из reports, и т.д.) —
                # юзер требовал чтобы блок отражал только его настройки.
                # Если ни одной категории не заполнено — блок не показываем.
                fc_user = unit.get("manual_fc_user_breakdown") or {}
                fixed_total_m = float(unit.get("manual_fc_user_total") or 0)

                if fixed_total_m > 0 and isinstance(fc_user, dict):
                    LABELS_PT = {
                        "aluguel": "Aluguel",
                        "armazenagem": "Armaz",
                        "salaries": "Salários",
                        "utilities": "Utilidades",
                        "software": "Software",
                        "outros": "Outros",
                    }
                    parts: list[str] = []
                    for key, label in LABELS_PT.items():
                        v = float(fc_user.get(key) or 0)
                        if v > 0:
                            parts.append(f"{label} {_money(v)}")
                    breakdown = " · ".join(parts)
                    if breakdown:
                        desc_lines.append(
                            f"   • Custos fixos /mês: {_money(fixed_total_m)} ({breakdown})"
                        )
                    else:
                        desc_lines.append(
                            f"   • Custos fixos /mês: {_money(fixed_total_m)}"
                        )
                if margin_net_pct is not None and float(margin_net_pct) < 0:
                    desc_lines.append(
                        "   ⚠ Sem cobrir overhead — considere subir preço ou cortar custos fixos"
                    )

            # ── Break-even progress block (Сессия C) ────────────────────
            # _breakeven инжектится ml_backfill._enrich_order_with_margin
            # после увеличения cumulative variable margin за этот месяц.
            # Показывает прогресс окупаемости fixed costs проекта.
            be = enriched.get("_breakeven") or {}
            if be:
                target = float(be.get("target_total") or 0.0)
                cumulative = float(be.get("cumulative") or 0.0)
                sales_n = int(be.get("sales_count") or 0)
                breakeven_at = be.get("breakeven_reached_at")
                just_reached = bool(be.get("just_reached"))
                net_after = float(be.get("net_profit_after_breakeven") or 0.0)
                project_id_be = str(be.get("project_id") or "")

                desc_lines.append("")
                if breakeven_at and not just_reached:
                    # Уже окупились ранее в этом месяце.
                    desc_lines.append(
                        f"🎉 *Já cobriu fixos do mês ({project_id_be})*: "
                        f"R$ {net_after:,.2f} de lucro líquido em {sales_n} vendas"
                        .replace(",", "X").replace(".", ",").replace("X", ".")
                    )
                elif just_reached:
                    # Эта продажа достигла break-even.
                    desc_lines.append(
                        f"🎉 *Break-even alcançado!* Custos fixos R$ {target:,.2f} cobertos."
                        .replace(",", "X").replace(".", ",").replace("X", ".")
                    )
                    if net_after > 0:
                        desc_lines.append(
                            f"💎 Excedente: R$ {net_after:,.2f} (lucro líquido depois de cobrir fixos)"
                            .replace(",", "X").replace(".", ",").replace("X", ".")
                        )
                elif target > 0:
                    # До break-even ещё далеко.
                    pct = (cumulative / target * 100) if target > 0 else 0
                    pct = max(0.0, min(100.0, pct))
                    # ASCII progress bar 10 chars.
                    filled = int(round(pct / 10))
                    bar = "█" * filled + "░" * (10 - filled)
                    remaining = max(0.0, target - cumulative)
                    avg_per_sale = cumulative / sales_n if sales_n > 0 else 0.0
                    # Estimate sales_to_breakeven based on running average.
                    if avg_per_sale > 0 and remaining > 0:
                        sales_left_est = max(1, int(round(remaining / avg_per_sale)))
                        eta_str = f" · ≈{sales_left_est} vendas"
                    else:
                        eta_str = ""
                    desc_lines.append(
                        f"⚙️ *Break-even {project_id_be}* {bar} {pct:.0f}%"
                    )
                    desc_lines.append(
                        (
                            f"   Cobertos R$ {cumulative:,.2f} / R$ {target:,.2f} · "
                            f"faltam R$ {remaining:,.2f}{eta_str}"
                        ).replace(",", "X").replace(".", ",").replace("X", ".")
                    )
                # Если target=0 — нет fixed costs, не показываем progress.

            # ── Inventory forecast block ──────────────────────────────
            # Stock + window-based velocity → days_left. memory:
            # project_inventory_forecast_in_tg.md (14d default, cancelled
            # excluded, both Full и MEnvios).
            inv = enriched.get("_inventory") or {}
            if inv:
                stock = int(inv.get("stock") or 0)
                sold_w = int(inv.get("sold_in_window") or 0)
                window_d = int(inv.get("window_days") or 14)
                avg_d = float(inv.get("avg_daily") or 0)
                days_left = inv.get("days_left")
                level = inv.get("level") or "ok"
                var_label = inv.get("variation_label") or ""
                # «Estoque (Preto · M):» когда stock считался per-variation.
                est_label = f"Estoque ({var_label})" if var_label else "Estoque"

                desc_lines.append("")
                # Stockout = особый случай: сообщать «хватит на ~0 дней» —
                # тавтология. Лучше показать скорость продаж + suggested
                # refill (на 30 дней по среднему темпу), чтобы продавец
                # сразу понимал сколько закупить.
                if stock <= 0 and avg_d > 0:
                    refill_30d = int(round(avg_d * 30))
                    desc_lines.append(
                        f"📦 🛑 {est_label} esgotado! {window_d}d: {sold_w} vendas "
                        f"(≈{avg_d:.1f}/dia) → reabasteça ≈{refill_30d} un. "
                        f"para 30 dias"
                    )
                elif stock <= 0:
                    # Stockout без истории продаж — просто маркер
                    desc_lines.append(
                        f"📦 🛑 {est_label} esgotado · sem vendas em {window_d}d "
                        f"(impossível estimar refill)"
                    )
                elif level == "no_history" or avg_d <= 0:
                    desc_lines.append(f"📦 {est_label}: {stock} un. · sem histórico em {window_d}d")
                elif level == "critical":
                    days_disp = days_left if days_left is not None else 0
                    desc_lines.append(
                        f"📦 ⚠️ {est_label}: {stock} un. · {window_d}d: {sold_w} vendas "
                        f"(≈{avg_d:.1f}/dia) · ⏰ apenas ~{days_disp} dias!"
                    )
                else:
                    days_disp = days_left if days_left is not None else "∞"
                    desc_lines.append(
                        f"📦 {est_label}: {stock} un. · {window_d}d: {sold_w} vendas "
                        f"(≈{avg_d:.1f}/dia) · ⏰ ~{days_disp} dias"
                    )
        elif sale_price > 0 and item_id:
            # Margin не подсчитан. Причины: либо unit_cost_brl пуст в каталоге
            # (тогда заполнить через sku-mapping), либо ml_item_margin_cache
            # пуст для этого item (refresh runs nightly — новые SKU могут
            # ждать до следующего refresh).
            desc_lines.append("")
            desc_lines.append(
                f"📊 Margem indisponível ({item_id}) — preencha custo "
                f"em /finance/sku-mapping ou aguarde refresh noturno"
            )

        # Paused-with-stock — top-3 объявления на паузе с остатком + ссылки
        # на активацию (signed link, кликабельны прямо из TG).
        # ml_backfill инжектит этот block в enriched["_paused_with_stock"].
        paused_pws = enriched.get("_paused_with_stock") or []
        if paused_pws:
            desc_lines.append("")
            desc_lines.append("⚠️ *Pausados mas com estoque:*")
            for p in paused_pws[:3]:
                pid = p.get("item_id") or ""
                pstock = int(p.get("stock") or 0)
                psold = int(p.get("sold") or 0)
                purl = p.get("activate_url") or ""
                line = f"   • `{pid}` ({pstock} un., vendido {psold} historicamente)"
                if purl:
                    line += f" → [Ativar]({purl})"
                desc_lines.append(line)

        permalink = (
            f"https://www.mercadolivre.com.br/vendas/{enriched.get('id')}/detalhe"
            if enriched.get("id")
            else ""
        )
        actions = []
        if permalink:
            actions.append({"label": "Ver pedido", "url": permalink})

        return {
            **base,
            "label": f"Nova venda {_brl(total)} — {status}",
            "description": "\n".join(x for x in desc_lines if x),
            "from_date": enriched.get("date_created") or enriched.get("last_updated"),
            "tags": [t for t in ["ORDERS", status.upper()] if t],
            "actions": actions,
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
