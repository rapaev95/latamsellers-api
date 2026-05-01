"""Per-campaign Ads summary for TG — top campaigns by spend/ROAS + buttons.

User feedback: «нужна сводка по рекламе — для каждой РК: spent / sold /
revenue / ROAS, кнопки повысить-понизить ROAS / budget / pause».

Phase 1 (this commit) — read-only summary с deep-link buttons на ML Ads.
Phase 2 (после получения ML Ads PUT API docs) — real budget/ROAS controls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone, date
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))
TG_API_BASE = "https://api.telegram.org"
DEFAULT_WINDOW_DAYS = 14
MAX_CAMPAIGNS_IN_TG = 10


async def aggregate_per_campaign(
    pool: asyncpg.Pool, user_id: int, days: int = DEFAULT_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Returns campaigns sorted by total_amount DESC for last N days,
    enriched with ads count + budget + ROAS_target. Each row contains
    BOTH window metrics (cost/clicks/etc) AND today_* metrics for quick
    today-vs-trend visualization.

    Metrics JSONB fields used (per ML Ads spec):
      cost, clicks, prints, total_amount, direct_amount, indirect_amount,
      direct_units_quantity / units_quantity (orders).
    """
    # ML metrics обновляются ~10 AM BRT за prev день — данные «сегодня»
    # обычно отсутствуют до этого момента. Используем yesterday как
    # «recent point in time» — это также то что seller видит в Mercado
    # Ads UI как «Ontem».
    target_to = date.today() - timedelta(days=1)
    target_from = target_to - timedelta(days=days - 1)
    # Previous window для trend сравнения
    prev_to = target_from - timedelta(days=1)
    prev_from = prev_to - timedelta(days=days - 1)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH agg AS (
                SELECT m.advertiser_id, m.campaign_id,
                       COALESCE(SUM((m.metrics->>'cost')::numeric), 0)            AS cost,
                       COALESCE(SUM((m.metrics->>'clicks')::int), 0)              AS clicks,
                       COALESCE(SUM((m.metrics->>'prints')::int), 0)              AS prints,
                       COALESCE(SUM((m.metrics->>'total_amount')::numeric), 0)    AS revenue,
                       COALESCE(SUM((m.metrics->>'direct_amount')::numeric), 0)   AS direct_revenue,
                       COALESCE(SUM(
                         COALESCE((m.metrics->>'units_quantity')::int,
                                  (m.metrics->>'direct_units_quantity')::int, 0)
                       ), 0) AS units,
                       -- Share of Voice family — average over the window
                       AVG(NULLIF((m.metrics->>'impression_share')::numeric, 0))            AS impression_share_avg,
                       AVG(NULLIF((m.metrics->>'top_impression_share')::numeric, 0))        AS top_impression_share_avg,
                       AVG(NULLIF((m.metrics->>'lost_impression_share_by_budget')::numeric, 0)) AS lost_budget_avg,
                       AVG(NULLIF((m.metrics->>'lost_impression_share_by_ad_rank')::numeric, 0)) AS lost_rank_avg
                  FROM ml_ad_campaign_metrics_daily m
                 WHERE m.user_id = $1 AND m.date BETWEEN $2 AND $3
                 GROUP BY m.advertiser_id, m.campaign_id
            ),
            yesterday_agg AS (
                SELECT m.advertiser_id, m.campaign_id,
                       COALESCE(SUM((m.metrics->>'cost')::numeric), 0)            AS cost,
                       COALESCE(SUM((m.metrics->>'clicks')::int), 0)              AS clicks,
                       COALESCE(SUM((m.metrics->>'prints')::int), 0)              AS prints,
                       COALESCE(SUM((m.metrics->>'total_amount')::numeric), 0)    AS revenue,
                       COALESCE(SUM(
                         COALESCE((m.metrics->>'units_quantity')::int,
                                  (m.metrics->>'direct_units_quantity')::int, 0)
                       ), 0) AS units
                  FROM ml_ad_campaign_metrics_daily m
                 WHERE m.user_id = $1 AND m.date = $3
                 GROUP BY m.advertiser_id, m.campaign_id
            ),
            prev_agg AS (
                SELECT m.advertiser_id, m.campaign_id,
                       COALESCE(SUM((m.metrics->>'cost')::numeric), 0)            AS cost,
                       COALESCE(SUM((m.metrics->>'total_amount')::numeric), 0)    AS revenue,
                       COALESCE(SUM(
                         COALESCE((m.metrics->>'units_quantity')::int,
                                  (m.metrics->>'direct_units_quantity')::int, 0)
                       ), 0) AS units
                  FROM ml_ad_campaign_metrics_daily m
                 WHERE m.user_id = $1 AND m.date BETWEEN $4 AND $5
                 GROUP BY m.advertiser_id, m.campaign_id
            )
            SELECT a.*,
                   COALESCE(t.cost, 0)    AS yest_cost,
                   COALESCE(t.clicks, 0)  AS yest_clicks,
                   COALESCE(t.prints, 0)  AS yest_prints,
                   COALESCE(t.revenue, 0) AS yest_revenue,
                   COALESCE(t.units, 0)   AS yest_units,
                   COALESCE(p.cost, 0)    AS prev_cost,
                   COALESCE(p.revenue, 0) AS prev_revenue,
                   COALESCE(p.units, 0)   AS prev_units,
                   c.name, c.status, c.strategy, c.budget,
                   c.roas_target, c.product_id, c.cpc,
                   (SELECT COUNT(*) FROM ml_ad_ads ad
                     WHERE ad.user_id = $1
                       AND ad.advertiser_id = a.advertiser_id
                       AND ad.campaign_id = a.campaign_id) AS ads_count
              FROM agg a
              LEFT JOIN yesterday_agg t
                ON t.advertiser_id = a.advertiser_id
               AND t.campaign_id = a.campaign_id
              LEFT JOIN prev_agg p
                ON p.advertiser_id = a.advertiser_id
               AND p.campaign_id = a.campaign_id
              LEFT JOIN ml_ad_campaigns c
                ON c.user_id = $1
               AND c.advertiser_id = a.advertiser_id
               AND c.campaign_id = a.campaign_id
             WHERE a.cost > 0 OR a.revenue > 0
             ORDER BY a.revenue DESC NULLS LAST, a.cost DESC NULLS LAST
            """,
            user_id, target_from, target_to, prev_from, prev_to,
        )

    out: list[dict[str, Any]] = []
    for r in rows:
        cost = float(r["cost"] or 0)
        revenue = float(r["revenue"] or 0)
        units = int(r["units"] or 0)
        clicks = int(r["clicks"] or 0)
        prints = int(r["prints"] or 0)
        roas = (revenue / cost) if cost > 0 else 0.0
        acos = (cost / revenue * 100) if revenue > 0 else 0.0
        romi = ((revenue - cost) / cost * 100) if cost > 0 else 0.0
        ctr = (clicks / prints * 100) if prints > 0 else 0.0
        cr = (units / clicks * 100) if clicks > 0 else 0.0
        cpc = (cost / clicks) if clicks > 0 else 0.0
        # Yesterday metrics (ML обновляет ~10 AM BRT — за prev день)
        yest_cost = float(r["yest_cost"] or 0)
        yest_revenue = float(r["yest_revenue"] or 0)
        yest_units = int(r["yest_units"] or 0)
        yest_roas = (yest_revenue / yest_cost) if yest_cost > 0 else 0.0
        yest_acos = (yest_cost / yest_revenue * 100) if yest_revenue > 0 else 0.0
        yest_romi = ((yest_revenue - yest_cost) / yest_cost * 100) if yest_cost > 0 else 0.0
        # Previous-window trend
        prev_cost = float(r["prev_cost"] or 0)
        prev_revenue = float(r["prev_revenue"] or 0)
        prev_units = int(r["prev_units"] or 0)
        delta_cost_pct = ((cost - prev_cost) / prev_cost * 100) if prev_cost > 0 else None
        delta_revenue_pct = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else None
        delta_units = units - prev_units
        out.append({
            "advertiser_id": int(r["advertiser_id"]),
            "campaign_id": int(r["campaign_id"]),
            "name": r["name"] or f"Campaign {r['campaign_id']}",
            "status": r["status"],
            "strategy": r["strategy"],
            "budget": float(r["budget"]) if r["budget"] is not None else None,
            "roas_target": float(r["roas_target"]) if r["roas_target"] is not None else None,
            "ads_count": int(r["ads_count"] or 0),
            "product_id": r["product_id"] or "PADS",
            # Window (default 14d)
            "cost_brl": round(cost, 2),
            "clicks": clicks,
            "prints": prints,
            "revenue_brl": round(revenue, 2),
            "units": units,
            "roas": round(roas, 2),
            "acos_pct": round(acos, 1),
            "romi_pct": round(romi, 1),
            "ctr_pct": round(ctr, 2),
            "cr_pct": round(cr, 1),
            "cpc_brl": round(cpc, 2),
            # Trend vs previous N days
            "delta_cost_pct": round(delta_cost_pct, 1) if delta_cost_pct is not None else None,
            "delta_revenue_pct": round(delta_revenue_pct, 1) if delta_revenue_pct is not None else None,
            "delta_units": delta_units,
            "prev_cost_brl": round(prev_cost, 2),
            "prev_revenue_brl": round(prev_revenue, 2),
            "prev_units": prev_units,
            # Today
            "today_cost_brl": round(today_cost, 2),
            "today_revenue_brl": round(today_revenue, 2),
            "today_units": today_units,
            "today_clicks": int(r["today_clicks"] or 0),
            "today_prints": int(r["today_prints"] or 0),
            "today_roas": round(today_roas, 2),
            "today_acos_pct": round(today_acos, 1),
            "today_romi_pct": round(today_romi, 1),
            # Share of Voice (avg over window, percent units)
            "impression_share_pct": (
                round(float(r["impression_share_avg"] or 0) * 100, 1)
                if r.get("impression_share_avg") is not None else None
            ),
            "top_impression_share_pct": (
                round(float(r["top_impression_share_avg"] or 0) * 100, 1)
                if r.get("top_impression_share_avg") is not None else None
            ),
            "lost_share_budget_pct": (
                round(float(r["lost_budget_avg"] or 0) * 100, 1)
                if r.get("lost_budget_avg") is not None else None
            ),
            "lost_share_rank_pct": (
                round(float(r["lost_rank_avg"] or 0) * 100, 1)
                if r.get("lost_rank_avg") is not None else None
            ),
        })
    return out


def _ml_campaign_url(advertiser_id: int, campaign_id: int, product_id: str = "PADS") -> str:
    """Deep-link на campaign settings в Mercado Ads UI."""
    return f"https://ads.mercadolivre.com.br/campaigns/{campaign_id}"


def _money(v: float) -> str:
    """R$ X,XX in pt-BR convention."""
    if v is None:
        return "—"
    return ("R$ " + f"{float(v):,.2f}").replace(",", "X").replace(".", ",").replace("X", ".")


def _esc_md(text: str) -> str:
    """Escape Markdown (legacy, not MD2) reserved chars: _ * [ ] ( )"""
    if not text:
        return ""
    out = text
    for ch in ["_", "*", "[", "]", "`"]:
        out = out.replace(ch, f"\\{ch}")
    return out


def _make_recommendation(c: dict[str, Any], lang: str) -> str:
    """Rule-based короткий совет по метрикам кампании.

    Логика:
      - ROAS значительно выше target и budget регулярно «съедается» → suggest +budget
      - ROMI > 100% AND lost-by-budget > 20% → ML могла бы показать больше но
        не хватает бюджета → +budget
      - ROMI < 0% AND ACOS > 50% → терять деньги → suggest снизить бюджет/раcs
      - lost-by-rank > 50% → классификация низкая → не помогает budget,
        нужен ad copy / лучшие ключевики
      - delta_revenue < -30% AND delta_cost > 0 → негативный тренд при
        стабильном бюджете → проверь акции конкурентов / category trends
    """
    romi = c.get("romi_pct") or 0
    roas = c.get("roas") or 0
    acos = c.get("acos_pct") or 0
    roas_target = c.get("roas_target")
    lost_budget = c.get("lost_share_budget_pct") or 0
    lost_rank = c.get("lost_share_rank_pct") or 0
    delta_rev = c.get("delta_revenue_pct")
    delta_cost = c.get("delta_cost_pct")

    if lang == "ru":
        prefix = "💡 *Совет:* "
    elif lang == "en":
        prefix = "💡 *Tip:* "
    else:
        prefix = "💡 *Dica:* "

    # 1. ROMI < 0 AND ACOS > 50 — кампания убыточная
    if romi < 0 and acos > 50:
        if lang == "ru":
            return prefix + "*убыточно* — снизь бюджет до R$1 (история сохранится) или повысь ROAS-таргет."
        if lang == "en":
            return prefix + "*losing money* — drop budget to R$1 (history preserved) or raise ROAS target."
        return prefix + "*perdendo dinheiro* — derrube o orçamento p/ R$1 (histórico preservado) ou suba ROAS target."

    # 2. ROMI > 100 + lost by budget > 20 — недоинвестирована
    if romi > 100 and lost_budget > 20:
        if lang == "ru":
            return prefix + f"высокая отдача *(ROMI {romi:.0f}%)*, но *{lost_budget:.0f}% показов терим из-за бюджета*. Подними бюджет на R$50."
        if lang == "en":
            return prefix + f"strong ROMI {romi:.0f}% but losing *{lost_budget:.0f}% impressions to budget*. Bump budget +R$50."
        return prefix + f"ROMI alto *({romi:.0f}%)* mas perde *{lost_budget:.0f}%* de impressões por orçamento. Aumente orçamento +R$50."

    # 3. lost_rank > 50 — слабая классификация, ad copy issue
    if lost_rank > 50:
        if lang == "ru":
            return prefix + f"*{lost_rank:.0f}% показов терим из-за низкого ранга*. Бюджет не поможет — улучши описание/фото/ключевые слова."
        if lang == "en":
            return prefix + f"*Losing {lost_rank:.0f}% to ad rank*. Budget won't help — improve title/photos/keywords."
        return prefix + f"*Perdendo {lost_rank:.0f}% de impressões por classificação*. Orçamento não vai ajudar — melhore título/fotos/palavras-chave."

    # 4. ROAS заметно выше target — таргет можно поднять
    if roas_target and roas > roas_target * 1.5 and roas_target > 0:
        if lang == "ru":
            return prefix + f"ROAS ({roas:.1f}x) сильно выше target ({roas_target:.1f}x). Подними target — будет больше показов и продаж."
        if lang == "en":
            return prefix + f"ROAS ({roas:.1f}x) far exceeds target ({roas_target:.1f}x). Raise target — more impressions/sales."
        return prefix + f"ROAS ({roas:.1f}x) muito acima do target ({roas_target:.1f}x). Suba target — mais impressões/vendas."

    # 5. Negative trend
    if delta_rev is not None and delta_rev < -30 and (delta_cost or 0) > -10:
        if lang == "ru":
            return prefix + f"выручка упала на *{abs(delta_rev):.0f}%* vs пред. 14d при стабильном бюджете. Проверь акции конкурентов / тренды категории."
        if lang == "en":
            return prefix + f"revenue dropped *{abs(delta_rev):.0f}%* vs prev 14d with stable spend. Check competitor promos / category trends."
        return prefix + f"receita caiu *{abs(delta_rev):.0f}%* vs 14d ant. com orçamento estável. Confira promos concorrentes / tendências da categoria."

    # 6. Default — green, no action
    return ""


def _format_card(c: dict[str, Any], lang: str = "pt") -> tuple[str, dict[str, Any]]:
    """Returns (text, keyboard) for one campaign — sections (Ontem | 14d
    + trend + ratios + SoV)."""
    name_safe = _esc_md((c["name"] or "")[:60])
    # 14d
    cost = c["cost_brl"]; revenue = c["revenue_brl"]; units = c["units"]
    roas = c["roas"]; acos = c["acos_pct"]; romi = c["romi_pct"]
    ads_count = c["ads_count"]; clicks = c["clicks"]; prints = c["prints"]
    ctr = c["ctr_pct"]; cr = c["cr_pct"]; cpc = c["cpc_brl"]
    budget = c["budget"]; roas_target = c["roas_target"]
    # yesterday
    y_cost = c["yest_cost_brl"]; y_rev = c["yest_revenue_brl"]
    y_units = c["yest_units"]; y_roas = c["yest_roas"]
    y_acos = c["yest_acos_pct"]; y_romi = c["yest_romi_pct"]
    y_clicks = c["yest_clicks"]; y_prints = c["yest_prints"]
    # trend
    delta_cost = c.get("delta_cost_pct")
    delta_rev = c.get("delta_revenue_pct")
    delta_units = c.get("delta_units")

    health_emoji = "🟢" if romi > 30 else ("🟡" if romi > 0 else "🔴")
    yest_health = "🟢" if y_romi > 30 else ("🟡" if y_romi > 0 else ("🔴" if y_cost > 0 else "⚪"))

    def _delta_arrow(v: Optional[float]) -> str:
        if v is None:
            return ""
        if v > 5:
            return f" ↗ +{v:.0f}%"
        if v < -5:
            return f" ↘ {v:.0f}%"
        return f" → {v:+.0f}%"

    if lang == "ru":
        title = f"📢 *РЕКЛАМА — {name_safe}*"
        h_yest = "📅 Вчера:"; h_14d = "📊 За 14 дней:"
        l_spent = "Потрачено"; l_rev = "Выручка"; l_units = "Продаж"
        l_ads = "Объявлений"; l_budget = "Бюджет/день"
        l_target = "ROAS target"
        l_eff = "Эффективность"; l_trend = "Тренд vs пред. 14d"
        l_cpc = "CPC"; l_ctr = "CTR"; l_cr = "Конверсия"
        no_yest = "_Вчера без активности._"
    elif lang == "en":
        title = f"📢 *ADS — {name_safe}*"
        h_yest = "📅 Yesterday:"; h_14d = "📊 Last 14d:"
        l_spent = "Spent"; l_rev = "Revenue"; l_units = "Orders"
        l_ads = "Ads"; l_budget = "Daily budget"
        l_target = "ROAS target"
        l_eff = "Efficiency"; l_trend = "Trend vs prev 14d"
        l_cpc = "CPC"; l_ctr = "CTR"; l_cr = "CR"
        no_yest = "_No activity yesterday._"
    else:
        title = f"📢 *ADS — {name_safe}*"
        h_yest = "📅 Ontem:"; h_14d = "📊 Últimos 14d:"
        l_spent = "Gasto"; l_rev = "Receita"; l_units = "Pedidos"
        l_ads = "Anúncios"; l_budget = "Orçamento/dia"
        l_target = "ROAS target"
        l_eff = "Eficiência"; l_trend = "Tendência vs 14d ant."
        l_cpc = "CPC"; l_ctr = "CTR"; l_cr = "CR"
        no_yest = "_Sem atividade ontem._"

    lines = [title, ""]

    # Yesterday block
    lines.append(h_yest)
    if y_cost > 0 or y_rev > 0:
        lines.append(f"  {l_spent}: {_money(y_cost)} · {l_rev}: {_money(y_rev)}")
        lines.append(f"  {l_units}: {y_units} · 👁 {y_prints} · 🖱 {y_clicks}")
        lines.append(f"  {yest_health} ROMI *{y_romi:+.1f}%* · ROAS *{y_roas:.2f}x* · ACOS *{y_acos:.1f}%*")
    else:
        lines.append(f"  {no_yest}")
    lines.append("")

    # 14d block
    lines.append(h_14d)
    lines.append(f"  {l_spent}: {_money(cost)} · {l_rev}: {_money(revenue)}")
    lines.append(f"  {l_units}: {units} · 👁 {prints} · 🖱 {clicks}")
    lines.append(f"  {health_emoji} ROMI *{romi:+.1f}%* · ROAS *{roas:.2f}x* · ACOS *{acos:.1f}%*")
    # Efficiency ratios
    lines.append(f"  ⚡ {l_eff}: CPC {_money(cpc)} · CTR *{ctr:.2f}%* · {l_cr} *{cr:.1f}%*")

    # Trend block — only if previous window had data
    if delta_cost is not None or delta_rev is not None:
        lines.append("")
        lines.append(f"📈 *{l_trend}:*")
        if delta_rev is not None:
            lines.append(f"  {l_rev}:{_delta_arrow(delta_rev)}")
        if delta_cost is not None:
            lines.append(f"  {l_spent}:{_delta_arrow(delta_cost)}")
        if delta_units is not None and (c.get("prev_units") or 0) > 0:
            sign = "+" if (delta_units or 0) >= 0 else ""
            lines.append(f"  {l_units}: {sign}{delta_units} ед.")
    lines.append("")

    # Settings
    lines.append(f"🎯 {l_ads}: {ads_count}")
    if budget is not None:
        lines.append(f"💼 {l_budget}: {_money(budget)}")
    if roas_target is not None:
        lines.append(f"🎯 {l_target}: {roas_target:.2f}x")

    # AI recommendation block — short tactical hint based on metrics.
    rec = _make_recommendation(c, lang)
    if rec:
        lines.append("")
        lines.append(rec)

    # Share of Voice — «exibido vs concorrência»
    sov_show = c.get("impression_share_pct")
    sov_lost_budget = c.get("lost_share_budget_pct")
    sov_lost_rank = c.get("lost_share_rank_pct")
    sov_top = c.get("top_impression_share_pct")
    if (sov_show is not None or sov_lost_budget is not None or sov_lost_rank is not None):
        lines.append("")
        if lang == "ru":
            lines.append("📡 *Доля показов vs конкуренты:*")
            if sov_show is not None:
                lines.append(f"  • Показано: *{sov_show:.1f}%*"
                             + (f" (топ-страница: {sov_top:.1f}%)" if sov_top is not None else ""))
            if sov_lost_budget is not None:
                lines.append(f"  • Не показано — бюджета не хватает: *{sov_lost_budget:.1f}%*")
            if sov_lost_rank is not None:
                lines.append(f"  • Не показано — низкий ранг: *{sov_lost_rank:.1f}%*")
        elif lang == "en":
            lines.append("📡 *Impression share vs competition:*")
            if sov_show is not None:
                lines.append(f"  • Shown: *{sov_show:.1f}%*"
                             + (f" (top spot: {sov_top:.1f}%)" if sov_top is not None else ""))
            if sov_lost_budget is not None:
                lines.append(f"  • Lost (budget): *{sov_lost_budget:.1f}%*")
            if sov_lost_rank is not None:
                lines.append(f"  • Lost (ad rank): *{sov_lost_rank:.1f}%*")
        else:
            lines.append("📡 *Share of Voice vs concorrência:*")
            if sov_show is not None:
                lines.append(f"  • Exibido: *{sov_show:.1f}%*"
                             + (f" (topo: {sov_top:.1f}%)" if sov_top is not None else ""))
            if sov_lost_budget is not None:
                lines.append(f"  • Não exibido — orçamento insuficiente: *{sov_lost_budget:.1f}%*")
            if sov_lost_rank is not None:
                lines.append(f"  • Não exibido — classificação: *{sov_lost_rank:.1f}%*")

    # Real-action callback buttons — wired to Mercado Ads PUT API через
    # FastAPI /escalar/ads/campaigns/{id}/update. callback payload:
    #   cbb:{up|dn|min}:{advId}:{campId}:{currentBudget}
    #   cbr:{up|dn}:{advId}:{campId}:{currentAcosTarget}
    #   cbp:{pause|resume}:{advId}:{campId}
    cid = c["campaign_id"]
    aid = c["advertiser_id"]
    base_url = f"https://ads.mercadolivre.com.br/campaigns/{cid}"
    cur_budget = float(budget or 0)
    cur_acos = float(roas_target or 0) if roas_target is not None else 0.0
    is_paused = (c.get("status") or "").lower() == "paused"

    if lang == "ru":
        btn_b_up = f"💼 +R$50"
        btn_b_dn = f"💼 -R$50"
        btn_b_min = "🛑 Бюджет=R$1"
        btn_r_up = "🎯 ROAS +0.5"
        btn_r_dn = "🎯 ROAS -0.5"
        btn_pause = "▶️ Возобновить" if is_paused else "🛑 Пауза"
        btn_open = "🔗 Открыть в ML"
    elif lang == "en":
        btn_b_up = f"💼 +R$50"
        btn_b_dn = f"💼 -R$50"
        btn_b_min = "🛑 Budget=R$1"
        btn_r_up = "🎯 ROAS +0.5"
        btn_r_dn = "🎯 ROAS -0.5"
        btn_pause = "▶️ Resume" if is_paused else "🛑 Pause"
        btn_open = "🔗 Open in ML"
    else:
        btn_b_up = f"💼 +R$50"
        btn_b_dn = f"💼 -R$50"
        btn_b_min = "🛑 Orçamento=R$1"
        btn_r_up = "🎯 ROAS +0.5"
        btn_r_dn = "🎯 ROAS -0.5"
        btn_pause = "▶️ Retomar" if is_paused else "🛑 Pausar"
        btn_open = "🔗 Abrir no ML"

    pause_action = "resume" if is_paused else "pause"
    rows: list[list[dict[str, Any]]] = []
    # Budget controls
    rows.append([
        {"text": btn_b_dn, "callback_data": f"cbb:dn:{aid}:{cid}:{cur_budget:.2f}"},
        {"text": btn_b_up, "callback_data": f"cbb:up:{aid}:{cid}:{cur_budget:.2f}"},
    ])
    rows.append([
        {"text": btn_b_min, "callback_data": f"cbb:min:{aid}:{cid}:{cur_budget:.2f}"},
    ])
    # ROAS controls (only if acos_target known — иначе skip)
    if cur_acos > 0:
        rows.append([
            {"text": btn_r_dn, "callback_data": f"cbr:dn:{aid}:{cid}:{cur_acos:.2f}"},
            {"text": btn_r_up, "callback_data": f"cbr:up:{aid}:{cid}:{cur_acos:.2f}"},
        ])
    # Pause + ML link
    rows.append([
        {"text": btn_pause, "callback_data": f"cbp:{pause_action}:{aid}:{cid}"},
        {"text": btn_open, "url": base_url},
    ])
    keyboard = {"inline_keyboard": rows}
    return "\n".join(lines), keyboard


async def dispatch_for_user(
    pool: asyncpg.Pool, user_id: int, days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, int]:
    """Sends one TG message per top campaign (max 10). Throttled."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"sent": 0, "reason": "no_bot_token"}
    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language,
                   COALESCE(notify_acos_change, TRUE) AS notify_on
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
    if not settings or not settings["telegram_chat_id"] or not settings["notify_on"]:
        return {"sent": 0, "reason": "disabled_or_no_chat"}
    chat_id = str(settings["telegram_chat_id"])
    lang = (settings["language"] or "pt").lower()

    campaigns = await aggregate_per_campaign(pool, user_id, days=days)
    if not campaigns:
        return {"sent": 0, "campaigns": 0}

    import asyncio as _asyncio
    sent = 0
    async with httpx.AsyncClient(timeout=15.0) as http:
        # Send overview header first
        header = {
            "ru": f"📊 *Сводка рекламы за {days}d* — топ {len(campaigns[:MAX_CAMPAIGNS_IN_TG])} РК",
            "en": f"📊 *Ads recap last {days}d* — top {len(campaigns[:MAX_CAMPAIGNS_IN_TG])}",
            "pt": f"📊 *Recap Ads {days}d* — top {len(campaigns[:MAX_CAMPAIGNS_IN_TG])}",
        }.get(lang, f"📊 *Recap Ads {days}d*")
        try:
            await http.post(
                f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": header, "parse_mode": "Markdown"},
            )
            await _asyncio.sleep(0.4)
        except Exception:  # noqa: BLE001
            pass

        for c in campaigns[:MAX_CAMPAIGNS_IN_TG]:
            text, keyboard = _format_card(c, lang)
            try:
                r = await http.post(
                    f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                        "reply_markup": keyboard,
                        "disable_web_page_preview": True,
                    },
                )
                if r.status_code == 200:
                    sent += 1
                else:
                    log.warning("ads-summary TG failed user=%s campaign=%s status=%s",
                                user_id, c["campaign_id"], r.status_code)
            except Exception as err:  # noqa: BLE001
                log.warning("ads-summary exception: %s", err)
            await _asyncio.sleep(0.6)
    return {"sent": sent, "campaigns": len(campaigns)}


async def dispatch_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Daily cron entrypoint."""
    if pool is None:
        return {"users": 0}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT t.user_id
              FROM ml_user_tokens t
              JOIN notification_settings n ON n.user_id = t.user_id
             WHERE t.access_token IS NOT NULL
               AND n.telegram_chat_id IS NOT NULL
            """,
        )
    totals = {"users": 0, "sent": 0, "campaigns": 0}
    for r in rows:
        try:
            res = await dispatch_for_user(pool, r["user_id"])
            totals["users"] += 1
            totals["sent"] += res.get("sent", 0)
            totals["campaigns"] += res.get("campaigns", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("ads-summary user=%s failed: %s", r["user_id"], err)
    return totals
