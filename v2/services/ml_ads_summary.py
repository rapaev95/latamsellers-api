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
    enriched with ads count + budget + ROAS_target.

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
                       ), 0) AS units
                  FROM ml_ad_campaign_metrics_daily m
                 WHERE m.user_id = $1 AND m.date BETWEEN $2 AND $3
                 GROUP BY m.advertiser_id, m.campaign_id
            )
            SELECT a.*,
                   c.name, c.status, c.strategy, c.budget,
                   c.roas_target, c.product_id, c.cpc,
                   (SELECT COUNT(*) FROM ml_ad_ads ad
                     WHERE ad.user_id = $1
                       AND ad.advertiser_id = a.advertiser_id
                       AND ad.campaign_id = a.campaign_id) AS ads_count
              FROM agg a
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
            "cost_brl": round(cost, 2),
            "clicks": int(r["clicks"] or 0),
            "prints": int(r["prints"] or 0),
            "revenue_brl": round(revenue, 2),
            "units": units,
            "roas": round(roas, 2),
            "acos_pct": round(acos, 1),
            "romi_pct": round(romi, 1),
        })
    return out


def _ml_campaign_url(advertiser_id: int, campaign_id: int, product_id: str = "PADS") -> str:
    """Deep-link на campaign settings в Mercado Ads UI."""
    # Both PADS (Product Ads) and BADS (Brand Ads) use the same URL
    # template; ML routes by product_id internally.
    return f"https://ads.mercadolivre.com.br/campaigns/{campaign_id}"


def _format_card(c: dict[str, Any], lang: str = "pt") -> tuple[str, dict[str, Any]]:
    """Returns (text, keyboard) for one campaign."""
    name = (c["name"] or "")[:60]
    cost = c["cost_brl"]
    revenue = c["revenue_brl"]
    units = c["units"]
    roas = c["roas"]
    acos = c["acos_pct"]
    romi = c["romi_pct"]
    ads_count = c["ads_count"]
    clicks = c["clicks"]
    prints = c["prints"]
    budget = c["budget"]
    roas_target = c["roas_target"]

    health_emoji = "🟢" if romi > 30 else ("🟡" if romi > 0 else "🔴")
    if lang == "ru":
        title = f"📢 \\[РЕКЛАМА\\] *{name}*"
        spent_lbl = "💸 Потрачено"
        sold_lbl = "📦 Продаж"
        rev_lbl = "💰 Выручка"
        ads_lbl = "🎯 Объявлений"
        budget_lbl = "💼 Бюджет/день"
    elif lang == "en":
        title = f"📢 \\[ADS\\] *{name}*"
        spent_lbl = "💸 Spent"
        sold_lbl = "📦 Orders"
        rev_lbl = "💰 Revenue"
        ads_lbl = "🎯 Ads"
        budget_lbl = "💼 Daily budget"
    else:
        title = f"📢 \\[ADS\\] *{name}*"
        spent_lbl = "💸 Gasto"
        sold_lbl = "📦 Pedidos"
        rev_lbl = "💰 Receita"
        ads_lbl = "🎯 Anúncios"
        budget_lbl = "💼 Orçamento/dia"

    lines = [
        title,
        "",
        f"{spent_lbl}: R$ {cost:.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        f"{rev_lbl}: R$ {revenue:.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        f"{sold_lbl}: {units} · 👁 {prints} · 🖱 {clicks}",
        f"{ads_lbl}: {ads_count}",
        "",
        f"{health_emoji} ROMI *{romi:+.1f}%* · ROAS *{roas:.2f}x* · ACOS *{acos:.1f}%*",
    ]
    if budget is not None:
        budget_str = f"R$ {budget:.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        lines.append(f"{budget_lbl}: {budget_str}")
    if roas_target is not None:
        lines.append(f"🎯 ROAS target: {roas_target:.2f}x")

    # Single-row keyboard with deep-link to ML Ads campaign settings.
    # Phase 2 после получения Mercado Ads PUT API docs — добавлю реальные
    # кнопки budget +50 / -50 / set to 1 / ROAS step etc.
    deep_link = _ml_campaign_url(c["advertiser_id"], c["campaign_id"], c["product_id"])
    keyboard = {
        "inline_keyboard": [
            [{"text": "🔗 Abrir campanha no ML", "url": deep_link}],
        ],
    }
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
