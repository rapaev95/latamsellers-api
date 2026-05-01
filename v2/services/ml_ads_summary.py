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
    target_to = date.today()
    target_from = target_to - timedelta(days=days)
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
            today_agg AS (
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
            )
            SELECT a.*,
                   COALESCE(t.cost, 0)    AS today_cost,
                   COALESCE(t.clicks, 0)  AS today_clicks,
                   COALESCE(t.prints, 0)  AS today_prints,
                   COALESCE(t.revenue, 0) AS today_revenue,
                   COALESCE(t.units, 0)   AS today_units,
                   c.name, c.status, c.strategy, c.budget,
                   c.roas_target, c.product_id, c.cpc,
                   (SELECT COUNT(*) FROM ml_ad_ads ad
                     WHERE ad.user_id = $1
                       AND ad.advertiser_id = a.advertiser_id
                       AND ad.campaign_id = a.campaign_id) AS ads_count
              FROM agg a
              LEFT JOIN today_agg t
                ON t.advertiser_id = a.advertiser_id
               AND t.campaign_id = a.campaign_id
              LEFT JOIN ml_ad_campaigns c
                ON c.user_id = $1
               AND c.advertiser_id = a.advertiser_id
               AND c.campaign_id = a.campaign_id
             WHERE a.cost > 0 OR a.revenue > 0
             ORDER BY a.revenue DESC NULLS LAST, a.cost DESC NULLS LAST
            """,
            user_id, target_from, target_to,
        )

    out: list[dict[str, Any]] = []
    for r in rows:
        cost = float(r["cost"] or 0)
        revenue = float(r["revenue"] or 0)
        units = int(r["units"] or 0)
        roas = (revenue / cost) if cost > 0 else 0.0
        acos = (cost / revenue * 100) if revenue > 0 else 0.0
        romi = ((revenue - cost) / cost * 100) if cost > 0 else 0.0
        # Today metrics
        today_cost = float(r["today_cost"] or 0)
        today_revenue = float(r["today_revenue"] or 0)
        today_units = int(r["today_units"] or 0)
        today_roas = (today_revenue / today_cost) if today_cost > 0 else 0.0
        today_acos = (today_cost / today_revenue * 100) if today_revenue > 0 else 0.0
        today_romi = ((today_revenue - today_cost) / today_cost * 100) if today_cost > 0 else 0.0
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
            "clicks": int(r["clicks"] or 0),
            "prints": int(r["prints"] or 0),
            "revenue_brl": round(revenue, 2),
            "units": units,
            "roas": round(roas, 2),
            "acos_pct": round(acos, 1),
            "romi_pct": round(romi, 1),
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


def _format_card(c: dict[str, Any], lang: str = "pt") -> tuple[str, dict[str, Any]]:
    """Returns (text, keyboard) for one campaign — 2 sections (Hoje | 14d)."""
    name_safe = _esc_md((c["name"] or "")[:60])
    # 14d
    cost = c["cost_brl"]; revenue = c["revenue_brl"]; units = c["units"]
    roas = c["roas"]; acos = c["acos_pct"]; romi = c["romi_pct"]
    ads_count = c["ads_count"]; clicks = c["clicks"]; prints = c["prints"]
    budget = c["budget"]; roas_target = c["roas_target"]
    # today
    t_cost = c["today_cost_brl"]; t_rev = c["today_revenue_brl"]
    t_units = c["today_units"]; t_roas = c["today_roas"]
    t_acos = c["today_acos_pct"]; t_romi = c["today_romi_pct"]
    t_clicks = c["today_clicks"]; t_prints = c["today_prints"]

    health_emoji = "🟢" if romi > 30 else ("🟡" if romi > 0 else "🔴")
    today_health = "🟢" if t_romi > 30 else ("🟡" if t_romi > 0 else ("🔴" if t_cost > 0 else "⚪"))

    if lang == "ru":
        title = f"📢 *РЕКЛАМА — {name_safe}*"
        h_today = "📅 Сегодня:"; h_14d = "📊 За 14 дней:"
        l_spent = "Потрачено"; l_rev = "Выручка"; l_units = "Продаж"
        l_imp = "Показы"; l_clk = "Клики"
        l_ads = "Объявлений"; l_budget = "Бюджет/день"
        l_target = "ROAS target"
        no_today = "_Сегодня без активности._"
    elif lang == "en":
        title = f"📢 *ADS — {name_safe}*"
        h_today = "📅 Today:"; h_14d = "📊 Last 14d:"
        l_spent = "Spent"; l_rev = "Revenue"; l_units = "Orders"
        l_imp = "Impressions"; l_clk = "Clicks"
        l_ads = "Ads"; l_budget = "Daily budget"
        l_target = "ROAS target"
        no_today = "_No activity today._"
    else:
        title = f"📢 *ADS — {name_safe}*"
        h_today = "📅 Hoje:"; h_14d = "📊 Últimos 14d:"
        l_spent = "Gasto"; l_rev = "Receita"; l_units = "Pedidos"
        l_imp = "Impressões"; l_clk = "Cliques"
        l_ads = "Anúncios"; l_budget = "Orçamento/dia"
        l_target = "ROAS target"
        no_today = "_Sem atividade hoje._"

    lines = [title, ""]

    # Today block
    lines.append(h_today)
    if t_cost > 0 or t_rev > 0:
        lines.append(f"  {l_spent}: {_money(t_cost)} · {l_rev}: {_money(t_rev)}")
        lines.append(f"  {l_units}: {t_units} · 👁 {t_prints} · 🖱 {t_clicks}")
        lines.append(f"  {today_health} ROMI *{t_romi:+.1f}%* · ROAS *{t_roas:.2f}x* · ACOS *{t_acos:.1f}%*")
    else:
        lines.append(f"  {no_today}")
    lines.append("")

    # 14d block
    lines.append(h_14d)
    lines.append(f"  {l_spent}: {_money(cost)} · {l_rev}: {_money(revenue)}")
    lines.append(f"  {l_units}: {units} · 👁 {prints} · 🖱 {clicks}")
    lines.append(f"  {health_emoji} ROMI *{romi:+.1f}%* · ROAS *{roas:.2f}x* · ACOS *{acos:.1f}%*")
    lines.append("")

    # Settings
    lines.append(f"🎯 {l_ads}: {ads_count}")
    if budget is not None:
        lines.append(f"💼 {l_budget}: {_money(budget)}")
    if roas_target is not None:
        lines.append(f"🎯 {l_target}: {roas_target:.2f}x")

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
