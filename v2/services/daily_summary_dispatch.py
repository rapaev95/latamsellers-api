"""Daily sales summary → Telegram cron.

Sends a per-seller morning/evening recap aggregating yesterday's:
  - orders count + revenue
  - visits (from ml_item_visits.daily JSONB)
  - ad spend / clicks / ACOS / ROAS (from ml_ad_campaign_metrics_daily)
  - conversion (orders / visits)
  - top 3 items by revenue + worst 2 by ROI
  - % delta vs day-before-yesterday for each metric

Triggered by the cron job _dispatch_daily_sales_summary_job in main.py
at 23:00 UTC = 20:00 BRT (end-of-day for the seller).

Schema: NO new tables here. Sources:
  - ml_user_orders via ml_orders.refresh_for_period + get_orders_for_day
    (replaced db_loader.load_user_vendas which was CSV-bound and stale
    when the seller stopped uploading files)
  - ml_item_visits.daily (visits per day)
  - ml_ad_campaign_metrics_daily (ads per day)
  - notification_settings (gating + language)

Activation: notification_settings.notify_daily_sales = TRUE AND
            telegram_chat_id IS NOT NULL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import httpx

from v2.services import ml_orders as ml_orders_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
TG_THROTTLE = 1.1  # seconds between TG sends per chat

# Brazil timezone (no DST since 2019). Yesterday's BRT day = UTC ms range.
BRT = timezone(timedelta(hours=-3))


# ── MarkdownV2 helpers (replicating ml_*_dispatch.py pattern) ──────

_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
_MD_CODE_ESCAPE = str.maketrans({"`": "\\`", "\\": "\\\\"})
_MD2_UNESCAPE_RE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


def _esc(text: Any) -> str:
    return (str(text or "")).translate(_MD_ESCAPE)


def _esc_code(text: Any) -> str:
    return (str(text or "")).translate(_MD_CODE_ESCAPE)


def _strip_md2_escapes(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"(?<!\\)[*_~`]", "", text)
    return _MD2_UNESCAPE_RE.sub(r"\1", out)


# ── Time helpers ───────────────────────────────────────────────────

def _brt_day_bounds_ms(target_date: date) -> tuple[int, int]:
    """Convert a BRT calendar date → [ms_start, ms_end] in UTC epoch ms."""
    start_brt = datetime.combine(target_date, datetime.min.time(), tzinfo=BRT)
    end_brt = start_brt + timedelta(days=1) - timedelta(milliseconds=1)
    return (
        int(start_brt.timestamp() * 1000),
        int(end_brt.timestamp() * 1000),
    )


def _brt_yesterday() -> date:
    return (datetime.now(BRT) - timedelta(days=1)).date()


# ── Aggregation ────────────────────────────────────────────────────

async def _aggregate_user_metrics(
    pool: asyncpg.Pool,
    user_id: int,
    target_date: date,
) -> dict[str, Any]:
    """Return all daily metrics for a single user/day in one shot.
    Returned shape is stable (same keys whether data is present or 0/None)
    so the diff/format layer can rely on it.
    """
    ms_start, ms_end = _brt_day_bounds_ms(target_date)
    target_iso = target_date.isoformat()

    # 1. Sales from ML /orders/search cache.
    # We pull a 2-day window (target_date and the day before) so we
    # cover late-arriving orders that ML backfilled after they were
    # initially created. Cache is then read filtered to the exact BRT
    # day. Day-before is also cached "for free" — useful for the diff
    # against yesterday in _build_card.
    refresh_end = max(target_date, _brt_yesterday())  # never future
    try:
        await ml_orders_svc.refresh_for_period(
            pool, user_id, days_back=2, end_date_brt=refresh_end,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("orders refresh user=%s date=%s failed (continuing with cache): %s",
                    user_id, target_date, err)

    day_agg = await ml_orders_svc.get_orders_for_day(pool, user_id, target_date)
    orders_count = day_agg["orders_count"]
    revenue = day_agg["revenue"]
    items_per_mlb = day_agg["items_per_mlb"]

    avg_ticket = (revenue / orders_count) if orders_count else 0.0

    # 2. Visits — sum across all this user's items for target_date
    async with pool.acquire() as conn:
        visits_row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM((d.elem->>'total')::int), 0) AS total_visits
              FROM ml_item_visits v
              CROSS JOIN LATERAL jsonb_array_elements(v.daily) AS d(elem)
             WHERE v.user_id = $1
               AND d.elem->>'date' LIKE $2
            """,
            user_id, f"{target_iso}%",
        )
    visits = int((visits_row and visits_row["total_visits"]) or 0)

    # 3. Ads metrics for the day — sum across all campaigns
    async with pool.acquire() as conn:
        try:
            ads_row = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM((metrics->>'cost')::numeric), 0)             AS cost,
                    COALESCE(SUM((metrics->>'clicks')::int), 0)               AS clicks,
                    COALESCE(SUM((metrics->>'prints')::int), 0)               AS prints,
                    COALESCE(SUM((metrics->>'total_amount')::numeric), 0)     AS total_amount,
                    COALESCE(SUM((metrics->>'direct_amount')::numeric), 0)    AS direct_amount,
                    COALESCE(SUM((metrics->>'indirect_amount')::numeric), 0)  AS indirect_amount
                  FROM ml_ad_campaign_metrics_daily
                 WHERE user_id = $1 AND date = $2
                """,
                user_id, target_date,
            )
        except Exception as err:  # noqa: BLE001
            log.warning("ads metrics query failed user=%s date=%s: %s", user_id, target_date, err)
            ads_row = None
    ad_cost = float((ads_row and ads_row["cost"]) or 0)
    ad_clicks = int((ads_row and ads_row["clicks"]) or 0)
    ad_prints = int((ads_row and ads_row["prints"]) or 0)
    ad_total_amount = float((ads_row and ads_row["total_amount"]) or 0)
    # ACOS = cost / total_amount * 100 (per ML docs convention)
    acos_pct = (ad_cost / ad_total_amount * 100) if ad_total_amount > 0 else 0.0
    roas = (ad_total_amount / ad_cost) if ad_cost > 0 else 0.0

    # 4. Top/worst items
    items_sorted = sorted(items_per_mlb.values(), key=lambda x: x["revenue"], reverse=True)
    top_items = items_sorted[:3]

    # Worst-by-ROI heuristic: items with ad_cost > 0 and 0 orders (ROI=0)
    # Without per-item ad metrics here we just surface "top low performers"
    # by revenue (bottom 2 with revenue > 0) — best-effort. Real per-item
    # ROI requires joining ml_ad_ads.metrics, which we keep out of MVP.
    worst_items = items_sorted[-2:] if len(items_sorted) > 5 else []

    # 4b. Enrich top/worst with SEO position + ads status. Two cheap
    # joins: position_history (latest per item across all tracked
    # keywords, INCLUDING found=false rows so we can say «item не
    # найден в топ-250 по X») + ml_ad_ads.
    enrich_mlbs = [it["mlb"] for it in (top_items + worst_items) if it.get("mlb")]
    if enrich_mlbs:
        async with pool.acquire() as conn:
            pos_rows = await conn.fetch(
                """
                SELECT DISTINCT ON (item_id)
                       item_id, position, keyword, found, total_results
                  FROM position_history
                 WHERE user_id = $1 AND item_id = ANY($2::text[])
                 ORDER BY item_id, checked_at DESC
                """,
                user_id, enrich_mlbs,
            )
            ad_rows = await conn.fetch(
                """
                SELECT item_id,
                       BOOL_OR(LOWER(COALESCE(status, '')) IN
                               ('active', 'enabled', 'on')) AS ads_active
                  FROM ml_ad_ads
                 WHERE user_id = $1 AND item_id = ANY($2::text[])
                 GROUP BY item_id
                """,
                user_id, enrich_mlbs,
            )
        pos_map = {
            r["item_id"]: {
                "position": r["position"],
                "keyword": r["keyword"],
                "found": r["found"],
                "total_results": r["total_results"],
            }
            for r in pos_rows
        }
        ads_map = {r["item_id"]: bool(r["ads_active"]) for r in ad_rows}
        for it in top_items + worst_items:
            mlb = it.get("mlb")
            if not mlb:
                continue
            p = pos_map.get(mlb)
            it["position"] = p["position"] if p else None
            it["position_keyword"] = p["keyword"] if p else None
            # `position_checked` = item HAS a position_history row, whether
            # found or not. Renderer differentiates 3 states:
            #   - never checked (no row): no SEO line
            #   - checked + found:   🔍 ОРГАНИКА #N (kw)
            #   - checked + not in scanned depth: 🔎 не найдена в топ-N
            it["position_checked"] = p is not None
            it["position_total"] = p.get("total_results") if p else None
            it["ads_active"] = ads_map.get(mlb, False)

    # 5. Conversion (orders / visits * 100)
    conversion_pct = (orders_count / visits * 100) if visits > 0 else 0.0

    return {
        "date": target_iso,
        "orders_count": orders_count,
        "revenue": revenue,
        "avg_ticket": avg_ticket,
        "visits": visits,
        "conversion_pct": conversion_pct,
        "ad_cost": ad_cost,
        "ad_clicks": ad_clicks,
        "ad_prints": ad_prints,
        "ad_total_amount": ad_total_amount,
        "acos_pct": acos_pct,
        "roas": roas,
        "top_items": top_items,
        "worst_items": worst_items,
        "items_total": len(items_sorted),
    }


def _diff_pct(today: float, yesterday: float) -> Optional[float]:
    """Percentage change today vs yesterday. None if yesterday=0 (undefined)."""
    if yesterday == 0:
        return None
    return (today - yesterday) / yesterday * 100


def _diff_pp(today: float, yesterday: float) -> float:
    """Percentage points change (for metrics already in %)."""
    return today - yesterday


# ── TG card builder ────────────────────────────────────────────────

def _fmt_brl(amount: float) -> str:
    """Format float as 'R$ 1.234,56' BRL convention."""
    if amount == 0:
        return "R\\$ 0"
    s = f"{amount:,.2f}"
    # ML/BR convention: thousand=',' → '.'; decimal='.' → ','. Two-step swap.
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R\\$ {s}"


def _fmt_int(n: int) -> str:
    """Format integer with thousand separators (BR style: 1.234)."""
    return f"{n:,}".replace(",", ".")


def _fmt_diff(pct: Optional[float]) -> str:
    """'(+18%)' / '(-3%)' / '(новый)' wrapped in MD2-safe escapes."""
    if pct is None:
        return "_no baseline_"
    sign = "+" if pct >= 0 else ""
    return f"\\({sign}{pct:.1f}%\\)"


def _fmt_diff_pp(pp: float) -> str:
    sign = "+" if pp >= 0 else ""
    return f"\\({sign}{pp:.2f}pp\\)"


_PT_WEEKDAY = {0: "segunda", 1: "terça", 2: "quarta", 3: "quinta", 4: "sexta", 5: "sábado", 6: "domingo"}
_PT_MONTH = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun", 7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}
_RU_WEEKDAY = {0: "понедельник", 1: "вторник", 2: "среда", 3: "четверг", 4: "пятница", 5: "суббота", 6: "воскресенье"}
_RU_MONTH = {1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "мая", 6: "июн", 7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек"}


def _date_label(d: date, lang: str) -> str:
    if lang == "ru":
        return f"{_RU_WEEKDAY[d.weekday()]}, {d.day} {_RU_MONTH[d.month]}"
    if lang == "en":
        return d.strftime("%a, %b %d")
    return f"{_PT_WEEKDAY[d.weekday()]}, {d.day} {_PT_MONTH[d.month]}"


def _build_card(
    today: dict[str, Any],
    yesterday: dict[str, Any],
    lang: str,
    app_url: str,
    extras: Optional[dict[str, int]] = None,
) -> str:
    target = date.fromisoformat(today["date"])
    label = _date_label(target, lang)

    if lang == "ru":
        title = f"📊 *Сводка за {_esc(label)}*"
        sales_lbl = "💰 Продажи"
        avg_lbl = "💼 Средний чек"
        vis_lbl = "👁 Визиты"
        conv_lbl = "📈 Конверсия"
        ads_lbl = "📢 Реклама"
        ads_clicks = "кликов"
        top_lbl = "🥇 Топ\\-3 по выручке"
        worst_lbl = "⚠️ Низкий ROI"
        cta = "📊 Подробнее в app"
    elif lang == "en":
        title = f"📊 *Daily summary — {_esc(label)}*"
        sales_lbl = "💰 Sales"
        avg_lbl = "💼 Avg ticket"
        vis_lbl = "👁 Visits"
        conv_lbl = "📈 Conversion"
        ads_lbl = "📢 Ads"
        ads_clicks = "clicks"
        top_lbl = "🥇 Top 3 revenue"
        worst_lbl = "⚠️ Low ROI"
        cta = "📊 More in app"
    else:
        title = f"📊 *Resumo de {_esc(label)}*"
        sales_lbl = "💰 Vendas"
        avg_lbl = "💼 Ticket médio"
        vis_lbl = "👁 Visitas"
        conv_lbl = "📈 Conversão"
        ads_lbl = "📢 Ads"
        ads_clicks = "cliques"
        top_lbl = "🥇 Top 3 receita"
        worst_lbl = "⚠️ ROI baixo"
        cta = "📊 Mais no app"

    revenue_diff = _diff_pct(today["revenue"], yesterday["revenue"])
    avg_diff = _diff_pct(today["avg_ticket"], yesterday["avg_ticket"])
    visits_diff = _diff_pct(today["visits"], yesterday["visits"])
    conv_diff = _diff_pp(today["conversion_pct"], yesterday["conversion_pct"])
    cost_diff = _diff_pct(today["ad_cost"], yesterday["ad_cost"])
    acos_diff = _diff_pp(today["acos_pct"], yesterday["acos_pct"])

    lines: list[str] = [title, ""]

    # Sales line — orders count + revenue + delta
    if today["orders_count"] > 0 or today["revenue"] > 0:
        units_word = (
            f"{today['orders_count']} {'заказов' if lang == 'ru' else 'orders' if lang == 'en' else 'pedidos'}"
        )
        lines.append(
            f"{sales_lbl}: {_esc(units_word)} · {_fmt_brl(today['revenue'])} {_fmt_diff(revenue_diff)}"
        )
        lines.append(f"{avg_lbl}: {_fmt_brl(today['avg_ticket'])} {_fmt_diff(avg_diff)}")
    else:
        # No sales — keep it short, don't pad with zeros
        empty = "Нет продаж" if lang == "ru" else "No sales" if lang == "en" else "Sem vendas"
        lines.append(f"{sales_lbl}: _{_esc(empty)}_")

    # Visits + conversion
    if today["visits"] > 0:
        lines.append(f"{vis_lbl}: {_esc(_fmt_int(today['visits']))} {_fmt_diff(visits_diff)}")
        lines.append(f"{conv_lbl}: {today['conversion_pct']:.2f}% {_fmt_diff_pp(conv_diff)}")

    # Ads block
    if today["ad_cost"] > 0:
        lines.append("")
        spent_w = "потрачено" if lang == "ru" else "spent" if lang == "en" else "gasto"
        lines.append(
            f"{ads_lbl}: {_fmt_brl(today['ad_cost'])} {_esc(spent_w)} · "
            f"{_esc(_fmt_int(today['ad_clicks']))} {ads_clicks} {_fmt_diff(cost_diff)}"
        )
        if today["acos_pct"] > 0:
            lines.append(
                f"   ACOS: {today['acos_pct']:.1f}% {_fmt_diff_pp(acos_diff)} · "
                f"ROAS: {today['roas']:.1f}x"
            )

    def _seo_ads_line(it: dict[str, Any]) -> Optional[str]:
        """Build an inline 'SEO + ads' annotation under each top/worst
        item. Three SEO states:
          - never checked (no position_history row): SEO part skipped
          - checked + found in organic: 🔍 ОРГАНИКА #N («keyword»)
          - checked + not in scanned depth: 🔎 не найдена в топ-N («keyword»)
        Plus ads ON/OFF on its own line (always shown when known).
        Whole annotation skipped only if BOTH SEO and ads are unknown.
        """
        position = it.get("position")
        position_checked = bool(it.get("position_checked"))
        ads_active = it.get("ads_active")
        if not position_checked and ads_active is None:
            return None
        parts: list[str] = []
        kw = (it.get("position_keyword") or "").strip()
        kw_short = (kw[:25] + "…") if len(kw) > 26 else kw
        if position is not None and position > 0:
            organic = "ОРГАНИКА" if lang == "ru" else "ORGÂNICO"
            place = "место" if lang == "ru" else "lugar"
            if kw_short:
                parts.append(f"🔍 {_esc(organic)} \\#{int(position)} \\({_esc(kw_short)}\\)")
            else:
                parts.append(f"🔍 {_esc(organic)} \\#{int(position)} {_esc(place)}")
        elif position_checked:
            # Item tracked but not in scanned depth — still surface a
            # signal so the seller knows their listing ranks low.
            depth_label = "топ\\-250" if lang == "ru" else "top\\-250"
            not_in = "не найдена в" if lang == "ru" else "fora do"
            if kw_short:
                parts.append(f"🔎 {_esc(not_in)} {depth_label} \\(«{_esc(kw_short)}»\\)")
            else:
                parts.append(f"🔎 {_esc(not_in)} {depth_label}")
        if ads_active is True:
            ads_on = "РЕКЛАМА = ON" if lang == "ru" else "ADS = ON"
            parts.append(f"📢 {_esc(ads_on)}")
        elif ads_active is False:
            ads_off = "РЕКЛАМА = OFF" if lang == "ru" else "ADS = OFF"
            parts.append(f"💤 {_esc(ads_off)}")
        if not parts:
            return None
        return "    " + " · ".join(parts)

    # Top items
    top = today.get("top_items") or []
    if top:
        lines.append("")
        lines.append(f"*{top_lbl}:*")
        for it in top:
            title_short = (it.get("title") or it.get("mlb") or "")[:55]
            units_lbl = (
                f"{int(it.get('units', 0))} шт"
                if lang == "ru"
                else f"{int(it.get('units', 0))} un"
            )
            lines.append(
                f"  • {_esc(title_short)} — {_fmt_brl(it['revenue'])} \\({_esc(units_lbl)}\\)"
            )
            seo_line = _seo_ads_line(it)
            if seo_line:
                lines.append(seo_line)

    # Worst (only if more than ~5 items in the day to make it meaningful)
    worst = today.get("worst_items") or []
    if worst:
        lines.append("")
        lines.append(f"*{worst_lbl}:*")
        for it in worst:
            title_short = (it.get("title") or it.get("mlb") or "")[:55]
            lines.append(f"  • {_esc(title_short)} — {_fmt_brl(it['revenue'])}")
            seo_line = _seo_ads_line(it)
            if seo_line:
                lines.append(seo_line)

    # Dynamic Pricing automation count — сколько объявлений под ML
    # автоматическим управлением цены (tag dynamic_standard_price).
    # Полезно seller'у: если много, и ACOS просел — скорее всего ML
    # снизил цены реагируя на конкурентов.
    if extras:
        n_dyn = int(extras.get("items_with_dyn_pricing") or 0)
        n_total = int(extras.get("active_items_total") or 0)
        if n_dyn > 0:
            lines.append("")
            if lang == "ru":
                pct_str = f" \\({n_dyn * 100 // max(n_total, 1)}%\\)" if n_total else ""
                lines.append(f"⚙️ *ML авто\\-цена:* {n_dyn} из {n_total}{pct_str}")
            elif lang == "en":
                pct_str = f" \\({n_dyn * 100 // max(n_total, 1)}%\\)" if n_total else ""
                lines.append(f"⚙️ *ML auto\\-price:* {n_dyn}/{n_total}{pct_str}")
            else:
                pct_str = f" \\({n_dyn * 100 // max(n_total, 1)}%\\)" if n_total else ""
                lines.append(f"⚙️ *Precificação ML:* {n_dyn}/{n_total}{pct_str}")

    return "\n".join(lines)


def _build_keyboard(app_base_url: str, lang: str) -> dict[str, Any]:
    label = "📊 Открыть дашборд" if lang == "ru" else "📊 Open dashboard" if lang == "en" else "📊 Abrir painel"
    return {
        "inline_keyboard": [
            [{"text": label, "url": f"{app_base_url.rstrip('/')}/dashboard"}],
        ],
    }


# ── TG send (with MD2 fallback to plain) ──────────────────────────

async def _tg_post(http: httpx.AsyncClient, bot_token: str, payload: dict[str, Any]) -> tuple[int, str, Optional[str]]:
    r = await http.post(f"{TG_API_BASE}/bot{bot_token}/sendMessage", json=payload, timeout=10.0)
    body_preview = r.text[:200] if r.status_code != 200 else ""
    msg_id: Optional[str] = None
    if r.status_code == 200:
        mid = (r.json() or {}).get("result", {}).get("message_id")
        msg_id = str(mid) if mid else None
    return r.status_code, body_preview, msg_id


async def _send_card(
    http: httpx.AsyncClient,
    bot_token: str,
    chat_id: str,
    text: str,
    keyboard: dict[str, Any],
) -> Optional[str]:
    try:
        status, body_preview, msg_id = await _tg_post(http, bot_token, {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "reply_markup": keyboard,
            "disable_web_page_preview": True,
        })
        if status == 200 and msg_id:
            return msg_id
        if status == 400 and "parse" in body_preview.lower():
            log.warning("daily-summary MD2 parse failed: %s", body_preview)
            plain = _strip_md2_escapes(text)
            status2, body2, msg_id2 = await _tg_post(http, bot_token, {
                "chat_id": chat_id,
                "text": plain,
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            })
            if status2 == 200 and msg_id2:
                return msg_id2
            log.warning("daily-summary plain fallback also failed: %s %s", status2, body2)
        log.warning("daily-summary send failed status=%s body=%s", status, body_preview)
    except Exception as err:  # noqa: BLE001
        log.exception("daily-summary send exception: %s", err)
    return None


# ── Per-user dispatch ─────────────────────────────────────────────

async def _dispatch_for_user(
    pool: asyncpg.Pool,
    user_id: int,
    *,
    target_date: Optional[date] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compute yesterday's metrics and send to user's Telegram if enabled.

    `target_date` defaults to yesterday in BRT.
    `force=True` ignores the notify_daily_sales flag — used by the preview
    endpoint so seller can demo the message on demand.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"sent": 0, "reason": "no_bot_token"}

    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language,
                   COALESCE(notify_daily_sales, TRUE) AS notify_daily_sales
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
    if not settings or not settings["telegram_chat_id"]:
        return {"sent": 0, "reason": "no_chat_id"}
    if not force and not settings["notify_daily_sales"]:
        return {"sent": 0, "reason": "disabled"}

    chat_id = str(settings["telegram_chat_id"])
    lang = (settings["language"] or "pt").lower()

    target = target_date or _brt_yesterday()
    yesterday_target = target - timedelta(days=1)

    today_metrics = await _aggregate_user_metrics(pool, user_id, target)
    yesterday_metrics = await _aggregate_user_metrics(pool, user_id, yesterday_target)

    # Skip if literally nothing happened — no point in noise.
    if (
        today_metrics["orders_count"] == 0
        and today_metrics["visits"] == 0
        and today_metrics["ad_cost"] == 0
        and not force
    ):
        return {"sent": 0, "reason": "no_activity"}

    app_base = os.environ.get("APP_BASE_URL", "https://app.lsprofit.app")
    # Aggregate extras first so card can include items_with_dyn_pricing
    # row alongside the metric breakdown.
    extras = await _aggregate_extras(pool, user_id, target_date)
    text = _build_card(today_metrics, yesterday_metrics, lang, app_base, extras=extras)
    if len(text) > 4000:
        text = text[:3990] + "…"
    keyboard = _build_keyboard(app_base, lang)

    # AI narrative briefing — short human-friendly summary BEFORE the
    # detailed metric card. Reuses extras already aggregated above.
    async with httpx.AsyncClient() as http:
        narrative = await _build_ai_narrative(
            http, today_metrics, yesterday_metrics, extras, lang,
        )
        if narrative:
            try:
                await http.post(
                    f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": narrative,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True,
                    },
                    timeout=10.0,
                )
                await asyncio.sleep(0.5)  # spacing before metric card
            except Exception as err:  # noqa: BLE001
                log.debug("AI narrative TG send failed: %s", err)
        msg_id = await _send_card(http, bot_token, chat_id, text, keyboard)

    return {
        "sent": 1 if msg_id else 0,
        "messageId": msg_id,
        "metrics": {
            "orders_count": today_metrics["orders_count"],
            "revenue": today_metrics["revenue"],
            "visits": today_metrics["visits"],
            "ad_cost": today_metrics["ad_cost"],
        },
    }


# ── AI narrative briefing ────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NARRATIVE_MODEL = "anthropic/claude-sonnet-4.5"


async def _aggregate_extras(
    pool: asyncpg.Pool, user_id: int, target_date: Optional[date],
) -> dict[str, int]:
    """Counts of new claims / unanswered questions / active promo candidates
    for the AI narrative. Failures fall back to zeros (narrative still useful)."""
    if target_date is None:
        target_date = _brt_yesterday()
    start_brt = datetime.combine(target_date, datetime.min.time(), tzinfo=BRT)
    end_brt = start_brt + timedelta(days=1)
    out = {
        "new_claims": 0, "open_claims_total": 0,
        "unanswered_questions": 0, "promo_candidates": 0,
        "active_items_total": 0, "items_with_dyn_pricing": 0,
    }
    try:
        async with pool.acquire() as conn:
            out["new_claims"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_claims WHERE user_id = $1 "
                "AND date_created BETWEEN $2 AND $3",
                user_id, start_brt, end_brt,
            ) or 0)
            out["open_claims_total"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_claims WHERE user_id = $1 "
                "AND status = 'opened'",
                user_id,
            ) or 0)
            out["unanswered_questions"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_questions WHERE user_id = $1 "
                "AND status = 'UNANSWERED'",
                user_id,
            ) or 0)
            out["promo_candidates"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_promotions WHERE user_id = $1 "
                "AND status = 'candidate' AND accepted_at IS NULL "
                "AND dismissed_at IS NULL",
                user_id,
            ) or 0)
            # Items with ML Dynamic Pricing tag — seller often не знает
            # сколько у него под автоматикой ML; показываем X из Y total.
            # Tag: 'dynamic_standard_price' в raw.tags JSONB.
            out["active_items_total"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM ml_user_items WHERE user_id = $1 "
                "AND COALESCE(status, 'active') = 'active'",
                user_id,
            ) or 0)
            out["items_with_dyn_pricing"] = int(await conn.fetchval(
                """
                SELECT COUNT(*) FROM ml_user_items
                 WHERE user_id = $1
                   AND COALESCE(status, 'active') = 'active'
                   AND raw -> 'tags' ? 'dynamic_standard_price'
                """,
                user_id,
            ) or 0)
    except Exception as err:  # noqa: BLE001
        log.debug("aggregate_extras failed user=%s: %s", user_id, err)
    return out


_NARRATIVE_LANG_LABELS = {
    "ru": "русский", "en": "English",
    "pt": "português brasileiro", "es": "español",
}

_NARRATIVE_PROMPT = """Você é um analista de vendas do Mercado Livre Brasil que escreve um BRIEFING DIÁRIO ULTRA-CURTO em {lang_label} para o seller, baseado nos números de hoje vs ontem + estado pendente.

Formato OBRIGATÓRIO:
*<emoji> Resumo do dia*

📊 *Hoje:* <1 frase com vendas + receita + delta vs ontem>
{extras_lines_format}

⚙️ *Precificação ML:* <X de Y anúncios sob automação ML — só se X > 0>

✅ *Foco para amanhã:*
• <ação 1 — verbo no imperativo>
• <ação 2 — apenas se realmente prioritária>

REGRAS:
- Total ≤ 600 caracteres.
- NÃO repita números que já estão no card detalhado (que vai logo depois).
- Linha "Precificação ML" — pular se 0 anúncios sob automação.
- Se ML autoprice é > 50% dos anúncios E ACOS hoje pior que ontem — sugira revisar.
- Se não houve atividade — diga isso direto, sem fluff.
- Use markdown simples (negrito *texto*) para Telegram.
- NÃO inventar — só usar números fornecidos."""


def _build_narrative_prompt_inputs(
    today: dict[str, Any], yesterday: dict[str, Any], extras: dict[str, int],
) -> str:
    delta_orders = (today.get("orders_count", 0) - yesterday.get("orders_count", 0))
    delta_revenue = (today.get("revenue", 0) - yesterday.get("revenue", 0))
    return (
        f"VENDAS HOJE: {today.get('orders_count', 0)} pedidos · "
        f"R$ {today.get('revenue', 0):.2f}\n"
        f"VENDAS ONTEM: {yesterday.get('orders_count', 0)} pedidos · "
        f"R$ {yesterday.get('revenue', 0):.2f}\n"
        f"DELTA: {delta_orders:+d} pedidos · R$ {delta_revenue:+.2f}\n"
        f"VISITAS HOJE: {today.get('visits', 0)} (ontem {yesterday.get('visits', 0)})\n"
        f"AD COST HOJE: R$ {today.get('ad_cost', 0):.2f} · "
        f"ACOS {today.get('acos_pct', 0):.1f}%\n"
        f"PENDENTES:\n"
        f"  • Reclamações novas hoje: {extras.get('new_claims', 0)}\n"
        f"  • Reclamações abertas total: {extras.get('open_claims_total', 0)}\n"
        f"  • Perguntas não respondidas: {extras.get('unanswered_questions', 0)}\n"
        f"  • Candidatos de promoção pendentes: {extras.get('promo_candidates', 0)}\n"
        f"AUTOMAÇÃO DE PREÇOS (Precificação Dinâmica do ML):\n"
        f"  • {extras.get('items_with_dyn_pricing', 0)} de "
        f"{extras.get('active_items_total', 0)} anúncios ativos sob ML autoprice"
    )


async def _build_ai_narrative(
    http: httpx.AsyncClient,
    today: dict[str, Any],
    yesterday: dict[str, Any],
    extras: dict[str, int],
    language: str,
) -> Optional[str]:
    """Returns formatted briefing text, or None if disabled / OpenRouter
    failure (caller falls back to silent — rule-based card still goes)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    lang_label = _NARRATIVE_LANG_LABELS.get((language or "pt").lower(), "português brasileiro")
    inputs = _build_narrative_prompt_inputs(today, yesterday, extras)
    extras_fmt = (
        "📌 *Pendentes:* <claims, perguntas e candidatos de promoção em 1 linha>"
    )
    prompt = _NARRATIVE_PROMPT.format(
        lang_label=lang_label, extras_lines_format=extras_fmt,
    )
    try:
        r = await http.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://app.lsprofit.app",
                "X-Title": "LS Profit App",
            },
            json={
                "model": NARRATIVE_MODEL,
                "max_tokens": 350,
                "temperature": 0.4,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": inputs},
                ],
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            log.warning("daily narrative %s: %s", r.status_code, r.text[:200])
            try:
                from . import tg_admin_alerts as _alerts
                await _alerts.alert_openrouter_failure(
                    r.status_code, r.text, service="daily-summary/narrative",
                )
            except Exception:  # noqa: BLE001
                pass
            return None
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return content.strip() if isinstance(content, str) else None
    except Exception as err:  # noqa: BLE001
        log.debug("narrative exception: %s", err)
        return None


# ── Cron entry point ─────────────────────────────────────────────

async def dispatch_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Walk all users with notify_daily_sales=TRUE + telegram_chat_id, send."""
    if pool is None:
        return {"users": 0, "sent": 0}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT n.user_id
              FROM notification_settings n
              JOIN ml_user_tokens t ON t.user_id = n.user_id
             WHERE n.telegram_chat_id IS NOT NULL
               AND COALESCE(n.notify_daily_sales, TRUE) = TRUE
               AND t.access_token IS NOT NULL
            """
        )
    user_ids = [r["user_id"] for r in rows]

    totals = {"users": 0, "sent": 0, "skipped": 0}
    for uid in user_ids:
        try:
            res = await _dispatch_for_user(pool, uid)
            totals["users"] += 1
            totals["sent"] += res["sent"]
            if res["sent"] == 0:
                totals["skipped"] += 1
        except Exception as err:  # noqa: BLE001
            log.exception("daily-summary dispatch user %s failed: %s", uid, err)
        await asyncio.sleep(TG_THROTTLE)
    return totals
