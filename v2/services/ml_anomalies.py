"""Anomaly detection on top of the live caches we already maintain.

Four detectors that run against yesterday's BRT day vs the 7-day median:
  1. acos_spike     — today_acos > max(2× median_acos, settings.acos_threshold)
  2. sales_drop     — today_orders < 0.5× median_orders AND median_orders >= 4
  3. visits_drop    — today_visits < 0.5× median_visits AND median_visits >= 100
  4. position_drop  — current position dropped >= 5 ranks AND now > 30
                      (or fell off the page entirely from a top-30 spot)

Detector deliberately uses the same per-day building blocks the Daily
Summary uses (ml_user_orders, ml_item_visits.daily, ml_ad_campaign_metrics_daily)
so a seller seeing a TG anomaly card sees numbers consistent with the
recap that arrives the same evening.

Storage: `escalar_anomalies` (one row per user/date/type/item). Dedup
on (user_id, date, anomaly_type, item_id) means cron can re-run without
double-pinging.

TG dispatch: a single summary card per user listing all today's
anomalies, MarkdownV2. Uses notify_acos_change toggle from
notification_settings — if disabled, store rows in DB anyway so the
in-app dashboard works.
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

from . import ml_orders as ml_orders_svc

log = logging.getLogger(__name__)

TG_API_BASE = "https://api.telegram.org"
BRT = timezone(timedelta(hours=-3))

# Detection thresholds. Conservative defaults — better to miss a few
# than to flood the seller's TG with false positives. All env-tunable.
ACOS_SPIKE_RATIO = float(os.environ.get("ANOMALIES_ACOS_RATIO", "2.0"))
SALES_DROP_RATIO = float(os.environ.get("ANOMALIES_SALES_RATIO", "0.5"))
SALES_DROP_MIN_BASELINE = int(os.environ.get("ANOMALIES_SALES_MIN_BASE", "4"))
VISITS_DROP_RATIO = float(os.environ.get("ANOMALIES_VISITS_RATIO", "0.5"))
VISITS_DROP_MIN_BASELINE = int(os.environ.get("ANOMALIES_VISITS_MIN_BASE", "100"))
# Position drop: trigger if rank dropped by at least N places AND new
# rank is past the first ~3 pages (50 results per page). Falling from
# top-30 to «not found» also fires.
POSITION_DROP_MIN_DELTA = int(os.environ.get("ANOMALIES_POSITION_MIN_DELTA", "5"))
POSITION_DROP_THRESHOLD = int(os.environ.get("ANOMALIES_POSITION_THRESHOLD", "30"))
HISTORY_DAYS = 7  # baseline window


# Each statement runs separately so a single syntax error doesn't take
# down the whole bootstrap. Postgres doesn't allow expressions like
# COALESCE(item_id, '') inside a table-level UNIQUE constraint, so we
# enforce the dedup key via a functional UNIQUE INDEX instead — which
# the ON CONFLICT clause in upsert_for_user matches by inference.
_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS escalar_anomalies (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL,
      date DATE NOT NULL,
      anomaly_type TEXT NOT NULL,
      severity TEXT NOT NULL DEFAULT 'warn',
      item_id TEXT,
      metric_value NUMERIC,
      baseline_value NUMERIC,
      delta_pct NUMERIC,
      message TEXT,
      tg_dispatched_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_escalar_anomalies_dedup
      ON escalar_anomalies (user_id, date, anomaly_type, (COALESCE(item_id, '')))
    """,
    "CREATE INDEX IF NOT EXISTS idx_escalar_anomalies_user_date ON escalar_anomalies(user_id, date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_escalar_anomalies_pending ON escalar_anomalies(user_id) WHERE tg_dispatched_at IS NULL",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    for stmt in _CREATE_STATEMENTS:
        async with pool.acquire() as conn:
            await conn.execute(stmt)


# ── MarkdownV2 helpers (same pattern as ml_*_dispatch.py) ─────────────────────

_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"_*[]()~`>#+-=|{}.!"})
_MD2_UNESCAPE_RE = re.compile(r"\\([_*\[\]()~`>#+\-=|{}.!\\])")


def _esc(text: Any) -> str:
    return (str(text or "")).translate(_MD_ESCAPE)


def _strip_md2(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"(?<!\\)[*_~`]", "", text)
    return _MD2_UNESCAPE_RE.sub(r"\1", out)


# ── BRT helpers ────────────────────────────────────────────────────────────

def _brt_yesterday() -> date:
    return (datetime.now(BRT) - timedelta(days=1)).date()


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


# ── Per-day metric collectors ──────────────────────────────────────────────

async def _orders_count_for_day(
    pool: asyncpg.Pool, user_id: int, target_date: date,
) -> int:
    """Total non-cancelled orders for one BRT day. Reuses ml_orders cache."""
    agg = await ml_orders_svc.get_orders_for_day(pool, user_id, target_date)
    return int(agg.get("orders_count") or 0)


async def _visits_for_day(
    pool: asyncpg.Pool, user_id: int, target_date: date,
) -> int:
    target_iso = target_date.isoformat()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM((d.elem->>'total')::int), 0) AS total_visits
              FROM ml_item_visits v
              CROSS JOIN LATERAL jsonb_array_elements(v.daily) AS d(elem)
             WHERE v.user_id = $1
               AND d.elem->>'date' LIKE $2
            """,
            user_id, f"{target_iso}%",
        )
    return int((row and row["total_visits"]) or 0)


async def _acos_for_day(
    pool: asyncpg.Pool, user_id: int, target_date: date,
) -> Optional[float]:
    """ACOS = ad_cost / total_amount * 100. None when total_amount is 0
    (division would be meaningless)."""
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM((metrics->>'cost')::numeric), 0) AS cost,
                    COALESCE(SUM((metrics->>'total_amount')::numeric), 0) AS total_amount
                  FROM ml_ad_campaign_metrics_daily
                 WHERE user_id = $1 AND date = $2
                """,
                user_id, target_date,
            )
        except Exception as err:  # noqa: BLE001
            log.warning("ACOS query user=%s date=%s failed: %s", user_id, target_date, err)
            return None
    cost = float((row and row["cost"]) or 0)
    total = float((row and row["total_amount"]) or 0)
    if total <= 0:
        return None
    return cost / total * 100


# ── Detection ─────────────────────────────────────────────────────────────


async def detect_for_user(
    pool: asyncpg.Pool, user_id: int, *,
    target_date: Optional[date] = None,
) -> list[dict[str, Any]]:
    """Run all detectors against `target_date` (default yesterday BRT)
    and return the list of anomaly dicts. Caller persists + dispatches."""
    target = target_date or _brt_yesterday()

    # Build the 7-day history per metric
    history_dates = [target - timedelta(days=i) for i in range(1, HISTORY_DAYS + 1)]

    today_orders, *hist_orders = await asyncio.gather(
        _orders_count_for_day(pool, user_id, target),
        *[_orders_count_for_day(pool, user_id, d) for d in history_dates],
    )
    today_visits, *hist_visits = await asyncio.gather(
        _visits_for_day(pool, user_id, target),
        *[_visits_for_day(pool, user_id, d) for d in history_dates],
    )
    today_acos, *hist_acos = await asyncio.gather(
        _acos_for_day(pool, user_id, target),
        *[_acos_for_day(pool, user_id, d) for d in history_dates],
    )

    median_orders = _median([float(x) for x in hist_orders])
    median_visits = _median([float(x) for x in hist_visits])
    median_acos = _median([x for x in hist_acos if x is not None])

    # Pull threshold from notification_settings
    async with pool.acquire() as conn:
        settings_row = await conn.fetchrow(
            "SELECT acos_threshold FROM notification_settings WHERE user_id = $1",
            user_id,
        )
    acos_threshold_setting: Optional[float] = None
    if settings_row and settings_row["acos_threshold"] is not None:
        try:
            acos_threshold_setting = float(settings_row["acos_threshold"])
        except (TypeError, ValueError):
            acos_threshold_setting = None

    anomalies: list[dict[str, Any]] = []

    # 1. ACOS spike
    if today_acos is not None and median_acos is not None and median_acos > 0:
        ratio_floor = median_acos * ACOS_SPIKE_RATIO
        threshold = max(ratio_floor, acos_threshold_setting or 0)
        if today_acos > threshold:
            delta = (today_acos - median_acos) / median_acos * 100
            anomalies.append({
                "type": "acos_spike",
                "severity": "critical" if today_acos > median_acos * 3 else "warn",
                "item_id": None,
                "metric_value": round(today_acos, 2),
                "baseline_value": round(median_acos, 2),
                "delta_pct": round(delta, 1),
                "message": (
                    f"ACOS hoje {today_acos:.1f}% — média 7d era {median_acos:.1f}%. "
                    f"Revise budgets ou suspenda campanhas com pior conversão."
                ),
            })

    # 2. Sales drop
    if median_orders is not None and median_orders >= SALES_DROP_MIN_BASELINE:
        if today_orders < median_orders * SALES_DROP_RATIO:
            delta = (today_orders - median_orders) / median_orders * 100
            anomalies.append({
                "type": "sales_drop",
                "severity": "critical" if today_orders == 0 else "warn",
                "item_id": None,
                "metric_value": float(today_orders),
                "baseline_value": round(median_orders, 1),
                "delta_pct": round(delta, 1),
                "message": (
                    f"Apenas {today_orders} pedidos — média 7d é {median_orders:.0f}. "
                    f"Verifique items pausados, ACOS, posição na busca."
                ),
            })

    # 3. Visits drop
    if median_visits is not None and median_visits >= VISITS_DROP_MIN_BASELINE:
        if today_visits < median_visits * VISITS_DROP_RATIO:
            delta = (today_visits - median_visits) / median_visits * 100
            anomalies.append({
                "type": "visits_drop",
                "severity": "warn",
                "item_id": None,
                "metric_value": float(today_visits),
                "baseline_value": round(median_visits, 0),
                "delta_pct": round(delta, 1),
                "message": (
                    f"Visitas hoje {today_visits} — média 7d é {median_visits:.0f}. "
                    f"Categoria, sazonalidade, ou items pausados? Veja dashboard."
                ),
            })

    # 4. Position drop — per (item_id, keyword) tracked by the seller.
    # We compare the latest position vs the prior 7-day median.
    position_anomalies = await _detect_position_drops(pool, user_id)
    anomalies.extend(position_anomalies)

    # 5. ads_no_sales — per item: ad_cost > threshold за last 7d AND
    # 0 заказов attributed → seller тратит на рекламу впустую.
    ads_no_sales = await _detect_ads_no_sales(pool, user_id)
    anomalies.extend(ads_no_sales)

    return anomalies


# Threshold for «реклама без продаж» alert — per-item ad spend over
# the last 7 days that fires the alert when matched with 0 orders.
ADS_NO_SALES_MIN_SPEND_BRL = float(os.environ.get("ADS_NO_SALES_MIN_SPEND_BRL", "30"))
ADS_NO_SALES_WINDOW_DAYS = int(os.environ.get("ADS_NO_SALES_WINDOW_DAYS", "7"))


async def _detect_ads_no_sales(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict[str, Any]]:
    """Per-item ads-without-sales detector. For each item where ad spend
    over the last N days exceeded a minimum threshold (default R$30) but
    zero orders were attributed → fire a `ads_no_sales` anomaly with the
    item_id so the TG card can offer a Pause-campaign button.
    """
    out: list[dict[str, Any]] = []
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH spend AS (
                    SELECT item_id,
                           SUM(COALESCE(cost_brl, 0)) AS total_cost
                      FROM ml_ad_campaign_metrics_daily
                     WHERE user_id = $1
                       AND day >= CURRENT_DATE - ($2::int * INTERVAL '1 day')
                     GROUP BY item_id
                    HAVING SUM(COALESCE(cost_brl, 0)) >= $3
                ),
                orders_count AS (
                    SELECT items_unnest->>'mlb' AS item_id,
                           COUNT(*) AS n_orders
                      FROM ml_user_orders,
                           jsonb_array_elements(items) AS items_unnest
                     WHERE user_id = $1
                       AND date_created >= NOW() - ($2::int * INTERVAL '1 day')
                       AND status NOT IN ('cancelled', 'invalid')
                     GROUP BY items_unnest->>'mlb'
                )
                SELECT s.item_id, s.total_cost, COALESCE(o.n_orders, 0) AS n_orders
                  FROM spend s
                  LEFT JOIN orders_count o ON o.item_id = s.item_id
                 WHERE COALESCE(o.n_orders, 0) = 0
                """,
                user_id, ADS_NO_SALES_WINDOW_DAYS, ADS_NO_SALES_MIN_SPEND_BRL,
            )
    except Exception as err:  # noqa: BLE001
        log.warning("ads_no_sales query user=%s failed (skipping): %s", user_id, err)
        return out

    for r in rows:
        cost = float(r["total_cost"] or 0)
        out.append({
            "type": "ads_no_sales",
            "severity": "critical" if cost > 100 else "warn",
            "item_id": r["item_id"],
            "metric_value": round(cost, 2),
            "baseline_value": 0,
            "delta_pct": None,
            "message": (
                f"Item {r['item_id']}: R$ {cost:.2f} de Ads em "
                f"{ADS_NO_SALES_WINDOW_DAYS}d, 0 vendas attributed. "
                f"Considere pausar campanha ou revisar palavras-chave."
            ),
        })
    return out


async def _detect_position_drops(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict[str, Any]]:
    """For each tracked keyword × item, compare the most recent
    position against the 7-day median. Fire if dropped by
    POSITION_DROP_MIN_DELTA places AND new rank is past
    POSITION_DROP_THRESHOLD (or vanished from top-N entirely).
    """
    out: list[dict[str, Any]] = []
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH latest AS (
                    SELECT DISTINCT ON (user_id, item_id, keyword)
                           user_id, item_id, keyword,
                           position AS today_position,
                           found AS today_found,
                           checked_at AS today_at
                      FROM position_history
                     WHERE user_id = $1
                       AND checked_at >= NOW() - INTERVAL '36 hours'
                     ORDER BY user_id, item_id, keyword, checked_at DESC
                ),
                baseline AS (
                    SELECT user_id, item_id, keyword,
                           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY position) AS med_pos,
                           COUNT(*) AS samples
                      FROM position_history
                     WHERE user_id = $1
                       AND found = TRUE
                       AND position IS NOT NULL
                       AND checked_at BETWEEN NOW() - INTERVAL '8 days'
                                          AND NOW() - INTERVAL '36 hours'
                     GROUP BY user_id, item_id, keyword
                    HAVING COUNT(*) >= 3
                )
                SELECT l.item_id, l.keyword, l.today_position, l.today_found,
                       b.med_pos, b.samples
                  FROM latest l
                  JOIN baseline b
                    ON b.user_id = l.user_id
                   AND b.item_id = l.item_id
                   AND b.keyword = l.keyword
                """,
                user_id,
            )
    except Exception as err:  # noqa: BLE001
        log.warning("position_drop query user=%s failed (skipping): %s", user_id, err)
        return out

    for r in rows:
        med = float(r["med_pos"]) if r["med_pos"] is not None else None
        if med is None:
            continue
        # Vanished entirely (was in top-N, now not_found)
        if not r["today_found"] and med <= POSITION_DROP_THRESHOLD:
            out.append({
                "type": "position_drop",
                "severity": "critical",
                "item_id": r["item_id"],
                "metric_value": None,
                "baseline_value": round(med, 1),
                "delta_pct": None,
                "message": (
                    f"\"{r['keyword']}\" — каіу da busca \\(antes ~{med:.0f}\\). "
                    f"Verifique status do anúncio + reputação."
                ),
            })
            continue
        if r["today_position"] is None:
            continue
        today = int(r["today_position"])
        delta_places = today - med  # positive = worse rank
        if delta_places >= POSITION_DROP_MIN_DELTA and today > POSITION_DROP_THRESHOLD:
            out.append({
                "type": "position_drop",
                "severity": "warn",
                "item_id": r["item_id"],
                "metric_value": float(today),
                "baseline_value": round(med, 1),
                "delta_pct": None,  # places moved, not %
                "message": (
                    f"\"{r['keyword']}\" passou de #{med:.0f} para #{today} "
                    f"\\({int(delta_places)} posições para baixo\\)."
                ),
            })
    return out


# ── Storage ───────────────────────────────────────────────────────────────


async def upsert_for_user(
    pool: asyncpg.Pool, user_id: int, target_date: date,
    anomalies: list[dict[str, Any]],
) -> int:
    """Persist detected anomalies. Returns count of NEW rows (existing
    rows for the same key are touched but not re-dispatched)."""
    if not anomalies:
        return 0
    new_count = 0
    async with pool.acquire() as conn:
        for a in anomalies:
            row = await conn.fetchrow(
                """
                INSERT INTO escalar_anomalies
                  (user_id, date, anomaly_type, severity, item_id,
                   metric_value, baseline_value, delta_pct, message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (user_id, date, anomaly_type, COALESCE(item_id, ''))
                DO NOTHING
                RETURNING id
                """,
                user_id, target_date, a["type"], a["severity"], a.get("item_id"),
                a.get("metric_value"), a.get("baseline_value"),
                a.get("delta_pct"), a.get("message"),
            )
            if row:
                new_count += 1
    return new_count


async def list_recent(
    pool: asyncpg.Pool, user_id: int, *, days: int = 14,
) -> list[dict[str, Any]]:
    cutoff = date.today() - timedelta(days=days)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, anomaly_type, severity, item_id,
                   to_char(date, 'YYYY-MM-DD') AS date,
                   metric_value, baseline_value, delta_pct, message,
                   tg_dispatched_at,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
              FROM escalar_anomalies
             WHERE user_id = $1 AND date >= $2
             ORDER BY date DESC, severity DESC, id DESC
            """,
            user_id, cutoff,
        )
    return [dict(r) for r in rows]


# ── TG dispatch ───────────────────────────────────────────────────────────


_SEVERITY_EMOJI = {"info": "ℹ️", "warn": "⚠️", "critical": "🚨"}
_TYPE_TITLE_RU = {
    "acos_spike": "Скачок ACOS",
    "sales_drop": "Просадка продаж",
    "visits_drop": "Просадка визитов",
    "position_drop": "Падение позиции",
    "ads_no_sales": "Реклама без продаж",
}
_TYPE_TITLE_PT = {
    "acos_spike": "Pico de ACOS",
    "sales_drop": "Queda de vendas",
    "visits_drop": "Queda de visitas",
    "position_drop": "Queda de posição",
    "ads_no_sales": "Ads sem vendas",
}


def _build_card(anomalies: list[dict[str, Any]], target_date: date, lang: str) -> str:
    titles = _TYPE_TITLE_RU if lang == "ru" else _TYPE_TITLE_PT
    header_ru = f"🚨 *Анналии за {_esc(target_date.strftime('%d.%m.%Y'))}*"
    header_pt = f"🚨 *Anomalias detectadas {_esc(target_date.strftime('%d/%m/%Y'))}*"
    lines = [header_ru if lang == "ru" else header_pt, ""]
    for a in anomalies:
        emoji = _SEVERITY_EMOJI.get(a.get("severity") or "warn", "⚠️")
        # Detector dicts use `type`; rows pulled from list_recent use
        # `anomaly_type` (DB column name). Accept both.
        atype = a.get("anomaly_type") or a.get("type") or ""
        title = titles.get(atype, atype)
        delta = a.get("delta_pct")
        delta_str = ""
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            delta_str = f" \\({_esc(sign + str(round(delta, 0)))}%\\)"
        metric = a.get("metric_value")
        baseline = a.get("baseline_value")
        body = ""
        if metric is not None and baseline is not None:
            body = (
                f"  Hoje: *{_esc(round(metric, 1))}* · 7d med: *{_esc(round(baseline, 1))}*{delta_str}"
            )
        msg = _esc(a.get("message") or "")
        lines.append(f"{emoji} *{_esc(title)}*")
        if body:
            lines.append(body)
        if msg:
            lines.append(f"  {msg}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_anomalies_keyboard(anomalies: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """For ads_no_sales anomalies — link straight into ML Ads dashboard for
    that item (one click = pause/edit campaign). Не делаем серверный pause
    callback потому что:
      1. ML pause endpoint требует campaign_id, не item_id (item может
         входить в несколько campaigns).
      2. Outage risk — лучше seller сам решит на ML interface, видя ROAS.
      3. Кнопка-URL = простой UX.
    """
    rows: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for a in anomalies:
        if (a.get("anomaly_type") or a.get("type")) != "ads_no_sales":
            continue
        item_id = (a.get("item_id") or "").upper()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        rows.append([{
            "text": f"🔗 Abrir Ads no ML — {item_id}",
            "url": f"https://www.mercadolivre.com.br/anuncios/{item_id}/posicionamento",
        }])
    # Snooze button — seller признал alert и не хочет повторов сегодня.
    if rows:
        rows.append([{
            "text": "🔕 Silenciar por 7d",
            "callback_data": "ads_snooze:7",
        }])
    return {"inline_keyboard": rows} if rows else None


async def _send_card(
    http: httpx.AsyncClient, bot_token: str, chat_id: str, text: str,
    keyboard: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    url = f"{TG_API_BASE}/bot{bot_token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }
    if keyboard:
        payload["reply_markup"] = keyboard
    try:
        r = await http.post(url, json=payload, timeout=15.0)
        if r.status_code == 200:
            data = r.json() or {}
            return str((data.get("result") or {}).get("message_id") or "")
        log.warning("TG anomalies MD2 failed: %s %s", r.status_code, r.text[:200])
        # Plain text fallback
        plain = _strip_md2(text)
        plain_payload = {
            "chat_id": chat_id, "text": plain,
            "disable_web_page_preview": True,
        }
        if keyboard:
            plain_payload["reply_markup"] = keyboard
        r2 = await http.post(url, json=plain_payload, timeout=15.0)
        if r2.status_code == 200:
            data = r2.json() or {}
            return str((data.get("result") or {}).get("message_id") or "")
        log.warning("TG anomalies plain failed: %s %s", r2.status_code, r2.text[:200])
    except Exception as err:  # noqa: BLE001
        log.exception("TG anomalies send failed: %s", err)
    return None


async def _dispatch_for_user(
    pool: asyncpg.Pool, user_id: int, target_date: Optional[date] = None,
) -> dict[str, int]:
    """Detect → upsert → if new rows, send TG card → mark dispatched."""
    await ensure_schema(pool)
    target = target_date or _brt_yesterday()

    detected = await detect_for_user(pool, user_id, target_date=target)
    new_count = await upsert_for_user(pool, user_id, target, detected)

    if new_count == 0 or not detected:
        return {"detected": len(detected), "new": new_count, "sent": 0}

    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language,
                   COALESCE(notify_acos_change, TRUE) AS notify_anomalies
              FROM notification_settings WHERE user_id = $1
            """,
            user_id,
        )
    if not settings or not settings["telegram_chat_id"]:
        return {"detected": len(detected), "new": new_count, "sent": 0, "reason": "no_chat"}
    if not settings["notify_anomalies"]:
        return {"detected": len(detected), "new": new_count, "sent": 0, "reason": "muted"}

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"detected": len(detected), "new": new_count, "sent": 0, "reason": "no_token"}

    text = _build_card(detected, target, (settings["language"] or "pt").lower())
    keyboard = _build_anomalies_keyboard(detected)
    async with httpx.AsyncClient() as http:
        msg_id = await _send_card(
            http, bot_token, str(settings["telegram_chat_id"]), text, keyboard,
        )

    if msg_id:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE escalar_anomalies
                   SET tg_dispatched_at = NOW()
                 WHERE user_id = $1 AND date = $2 AND tg_dispatched_at IS NULL
                """,
                user_id, target,
            )
        return {"detected": len(detected), "new": new_count, "sent": 1}
    return {"detected": len(detected), "new": new_count, "sent": 0, "reason": "tg_failed"}


async def dispatch_all_users(pool: asyncpg.Pool) -> dict[str, int]:
    """Cron entry. Walks all users with notify_acos_change=TRUE +
    telegram_chat_id, runs detect+dispatch."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT user_id FROM notification_settings
             WHERE telegram_chat_id IS NOT NULL
               AND COALESCE(notify_acos_change, TRUE) = TRUE
            """,
        )
    user_ids = [int(r["user_id"]) for r in rows]
    total_detected = 0
    total_new = 0
    sent = 0
    for uid in user_ids:
        try:
            res = await _dispatch_for_user(pool, uid)
            total_detected += res.get("detected", 0)
            total_new += res.get("new", 0)
            sent += res.get("sent", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("anomalies user %s: %s", uid, err)
    return {"users": len(user_ids), "detected": total_detected, "new": total_new, "sent": sent}
