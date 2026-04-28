"""Photo A/B test for ML listings.

Flow:
1. Seller publishes a new main photo via the Pictures UI (manual upload
   OR AI generation). UI offers a "Run A/B test" CTA.
2. We snapshot CURRENT visits + orders for the item over the previous
   `duration_days` (3/7/14) and store them as baseline_*.
3. After `ends_at` passes (cron fires hourly), we compute treatment_*
   over the SAME number of days starting at experiment start.
4. Diff is rendered as a TG card and sent to seller. Status flips to
   'completed'.

Schema: ONE new table `escalar_photo_experiments`. Reuses ml_item_visits
.daily JSONB for visits and ml_user_orders (via ml_orders service) for
orders — same sources daily_summary_dispatch uses, so baseline matches
what the seller sees in the Daily Summary card.

Triggered by _dispatch_photo_experiments_results_job in main.py at
hourly cadence (ends_at granularity is fine to within an hour).
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
TG_THROTTLE = 1.1
BRT = timezone(timedelta(hours=-3))


# ── Schema ─────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS escalar_photo_experiments (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  ai_asset_id INTEGER,
  old_picture_id TEXT,
  new_picture_id TEXT,
  duration_days INTEGER NOT NULL DEFAULT 7,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  ends_at TIMESTAMPTZ NOT NULL,
  baseline_visits INTEGER,
  baseline_orders INTEGER,
  baseline_period_days INTEGER,
  treatment_visits INTEGER,
  treatment_orders INTEGER,
  status TEXT DEFAULT 'testing',
  completed_at TIMESTAMPTZ,
  tg_dispatched_at TIMESTAMPTZ,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_photo_exp_user_item
  ON escalar_photo_experiments(user_id, item_id);
CREATE INDEX IF NOT EXISTS idx_photo_exp_pending
  ON escalar_photo_experiments(status) WHERE status = 'testing';
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── MarkdownV2 helpers (shared pattern) ────────────────────────────

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


# ── Metrics aggregation (reuses same data sources as daily-summary) ─

def _brt_day_bounds_ms(target_date: date) -> tuple[int, int]:
    start_brt = datetime.combine(target_date, datetime.min.time(), tzinfo=BRT)
    end_brt = start_brt + timedelta(days=1) - timedelta(milliseconds=1)
    return (
        int(start_brt.timestamp() * 1000),
        int(end_brt.timestamp() * 1000),
    )


async def _aggregate_window_for_item(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    """Sum visits + orders for a single item over the [start, end] BRT
    window. Used identically for baseline (before swap) and treatment
    (after swap) so the diff is apples-to-apples.

    Orders come from ml_user_orders (live ML cache). Caller is expected
    to call ml_orders_svc.refresh_for_period before this if freshness
    matters — start_experiment and _close_one_experiment do.
    """
    start_brt = start.astimezone(BRT)
    end_brt = end.astimezone(BRT)
    start_date = start_brt.date()
    end_date = end_brt.date()

    # 1. Visits — from ml_item_visits.daily JSONB.
    visits = 0
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.elem AS daily_entry
              FROM ml_item_visits v
              CROSS JOIN LATERAL jsonb_array_elements(v.daily) AS d(elem)
             WHERE v.user_id = $1
               AND v.item_id = $2
            """,
            user_id, item_id,
        )
    for row in rows:
        entry = row["daily_entry"]
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:  # noqa: BLE001
                continue
        if not isinstance(entry, dict):
            continue
        date_str = (entry.get("date") or "")[:10]
        if not date_str:
            continue
        try:
            entry_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        if start_date <= entry_date <= end_date:
            try:
                visits += int(entry.get("total") or 0)
            except (TypeError, ValueError):
                continue

    # 2. Orders — from ml_user_orders cache (filled by ml_orders_svc).
    orders_agg = await ml_orders_svc.get_orders_for_window(
        pool, user_id, item_id, start, end,
    )
    orders = orders_agg["orders"]

    return {"visits": visits, "orders": orders}


# ── Public API: start an experiment ───────────────────────────────

async def start_experiment(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    duration_days: int,
    *,
    new_picture_id: Optional[str] = None,
    old_picture_id: Optional[str] = None,
    ai_asset_id: Optional[int] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Snapshot baseline metrics over the previous `duration_days`,
    insert the experiment row with ends_at = now + duration_days.
    Returns the inserted row as a dict.
    """
    if duration_days not in (3, 7, 14):
        return {"error": "duration_days_must_be_3_7_or_14"}

    await ensure_schema(pool)

    now_utc = datetime.now(timezone.utc)
    baseline_start = now_utc - timedelta(days=duration_days)
    # Refresh ML orders cache for the baseline window before aggregating
    # so we don't snapshot stale numbers. +2 day buffer for late-arriving
    # orders ML may backfill.
    try:
        await ml_orders_svc.refresh_for_period(
            pool, user_id, days_back=duration_days + 2,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("photo-ab start: orders refresh user=%s failed (using cache): %s",
                    user_id, err)
    metrics = await _aggregate_window_for_item(
        pool, user_id, item_id, baseline_start, now_utc,
    )

    ends_at = now_utc + timedelta(days=duration_days)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO escalar_photo_experiments
              (user_id, item_id, ai_asset_id, old_picture_id, new_picture_id,
               duration_days, started_at, ends_at,
               baseline_visits, baseline_orders, baseline_period_days,
               status, notes)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7, $8, $9, $10, 'testing', $11)
            RETURNING id, user_id, item_id, ai_asset_id, old_picture_id,
                      new_picture_id, duration_days, started_at, ends_at,
                      baseline_visits, baseline_orders, status
            """,
            user_id, item_id, ai_asset_id, old_picture_id, new_picture_id,
            duration_days, ends_at,
            metrics["visits"], metrics["orders"], duration_days,
            notes,
        )
    return dict(row) if row else {"error": "insert_failed"}


async def cancel_experiment(
    pool: asyncpg.Pool,
    user_id: int,
    experiment_id: int,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE escalar_photo_experiments
               SET status = 'cancelled',
                   completed_at = NOW(),
                   updated_at = NOW()
             WHERE id = $1 AND user_id = $2 AND status = 'testing'
            RETURNING id, status
            """,
            experiment_id, user_id,
        )
    return dict(row) if row else {"error": "not_found_or_not_testing"}


async def list_experiments(
    pool: asyncpg.Pool,
    user_id: int,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    where = "WHERE user_id = $1"
    params: list[Any] = [user_id]
    if status:
        where += " AND status = $2"
        params.append(status)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, user_id, item_id, ai_asset_id, old_picture_id,
                   new_picture_id, duration_days,
                   to_char(started_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
                   to_char(ends_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS ends_at,
                   baseline_visits, baseline_orders, baseline_period_days,
                   treatment_visits, treatment_orders,
                   status,
                   to_char(completed_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS completed_at,
                   notes
              FROM escalar_photo_experiments
              {where}
             ORDER BY started_at DESC
            """,
            *params,
        )
    return [dict(r) for r in rows]


# ── Close + dispatch ──────────────────────────────────────────────

def _diff_pct(after: int, before: int) -> Optional[float]:
    if before == 0:
        return None
    return (after - before) / before * 100


def _build_result_card(
    experiment: dict[str, Any],
    item_title: Optional[str],
    lang: str,
) -> str:
    item_id = experiment["item_id"]
    duration = experiment["duration_days"]
    baseline_v = int(experiment["baseline_visits"] or 0)
    baseline_o = int(experiment["baseline_orders"] or 0)
    treatment_v = int(experiment["treatment_visits"] or 0)
    treatment_o = int(experiment["treatment_orders"] or 0)

    visits_diff = _diff_pct(treatment_v, baseline_v)
    orders_diff = _diff_pct(treatment_o, baseline_o)

    # Conversion (orders / visits * 100)
    base_conv = (baseline_o / baseline_v * 100) if baseline_v > 0 else 0.0
    trt_conv = (treatment_o / treatment_v * 100) if treatment_v > 0 else 0.0
    conv_diff_pp = trt_conv - base_conv

    def _fmt_diff(pct: Optional[float]) -> str:
        if pct is None:
            return "_no baseline_"
        sign = "+" if pct >= 0 else ""
        return f"\\({sign}{pct:.1f}%\\)"

    if lang == "ru":
        title = f"🧪 *A/B тест ЗАКРЫТ — фото*"
        subtitle = (
            f"📊 *{duration} дней:* до vs после смены фото"
            if duration != 1 else
            f"📊 *{duration} день:* до vs после"
        )
        v_lbl, o_lbl = "👁 Визиты", "🛒 Заказы"
        c_lbl = "📈 Конверсия"
        verdict_better = "✅ Новое фото эффективнее"
        verdict_worse = "❌ Новое фото хуже — рассмотри откат"
        verdict_neutral = "➖ Без значимого изменения"
        no_data = "Недостаточно данных для оценки"
        units = "до"
        units2 = "после"
    elif lang == "en":
        title = "🧪 *A/B test CLOSED — photo*"
        subtitle = f"📊 *{duration} days:* before vs after photo swap"
        v_lbl, o_lbl = "👁 Visits", "🛒 Orders"
        c_lbl = "📈 Conversion"
        verdict_better = "✅ New photo wins"
        verdict_worse = "❌ New photo lost — consider reverting"
        verdict_neutral = "➖ No meaningful change"
        no_data = "Not enough data to evaluate"
        units = "before"
        units2 = "after"
    else:
        title = "🧪 *Teste A/B FECHADO — foto*"
        subtitle = f"📊 *{duration} dias:* antes vs depois da troca"
        v_lbl, o_lbl = "👁 Visitas", "🛒 Pedidos"
        c_lbl = "📈 Conversão"
        verdict_better = "✅ Foto nova venceu"
        verdict_worse = "❌ Foto nova perdeu — considere reverter"
        verdict_neutral = "➖ Sem mudança significativa"
        no_data = "Dados insuficientes para avaliar"
        units = "antes"
        units2 = "depois"

    lines: list[str] = [title]
    if item_title:
        lines.append(f"🛍 *{_esc(item_title[:80])}*")
    lines.append(f"🆔 `{_esc_code(item_id)}`")
    lines.append("")
    lines.append(subtitle)
    lines.append("")

    # If we have absolutely no baseline data, surface that and stop.
    if baseline_v == 0 and baseline_o == 0:
        lines.append(f"_{_esc(no_data)}_")
        return "\n".join(lines)

    lines.append(f"{v_lbl}: {baseline_v} → {treatment_v} {_fmt_diff(visits_diff)}")
    lines.append(f"{o_lbl}: {baseline_o} → {treatment_o} {_fmt_diff(orders_diff)}")
    if baseline_v > 0 or treatment_v > 0:
        diff_sign = "+" if conv_diff_pp >= 0 else ""
        lines.append(
            f"{c_lbl}: {base_conv:.2f}% → {trt_conv:.2f}% \\({diff_sign}{conv_diff_pp:.2f}pp\\)"
        )

    # Verdict — prioritize conversion, fallback to orders/visits
    lines.append("")
    if treatment_v == 0 and treatment_o == 0:
        lines.append(f"_{_esc(no_data)}_")
    elif conv_diff_pp >= 0.20 or (orders_diff is not None and orders_diff >= 15):
        lines.append(verdict_better)
    elif conv_diff_pp <= -0.20 or (orders_diff is not None and orders_diff <= -15):
        lines.append(verdict_worse)
    else:
        lines.append(verdict_neutral)

    return "\n".join(lines)


def _build_keyboard(app_base_url: str, item_id: str, lang: str) -> dict[str, Any]:
    label = (
        "🔍 Подробнее" if lang == "ru"
        else "🔍 Details" if lang == "en"
        else "🔍 Detalhes"
    )
    pictures_label = (
        "🖼 Фото товара" if lang == "ru"
        else "🖼 Item pictures" if lang == "en"
        else "🖼 Fotos do item"
    )
    return {
        "inline_keyboard": [
            [
                {"text": label, "url": f"{app_base_url.rstrip('/')}/escalar/pictures"},
                {"text": pictures_label, "url": f"{app_base_url.rstrip('/')}/escalar/products"},
            ],
        ],
    }


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
            log.warning("photo-ab MD2 parse failed: %s", body_preview)
            plain = _strip_md2_escapes(text)
            status2, body2, msg_id2 = await _tg_post(http, bot_token, {
                "chat_id": chat_id,
                "text": plain,
                "reply_markup": keyboard,
                "disable_web_page_preview": True,
            })
            if status2 == 200 and msg_id2:
                return msg_id2
            log.warning("photo-ab plain fallback failed: %s %s", status2, body2)
        log.warning("photo-ab send failed status=%s body=%s", status, body_preview)
    except Exception as err:  # noqa: BLE001
        log.exception("photo-ab send exception: %s", err)
    return None


async def _close_one_experiment(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    experiment: dict[str, Any],
) -> bool:
    """Compute treatment metrics for a single experiment, dispatch TG,
    mark completed. Returns True if TG message landed."""
    user_id = int(experiment["user_id"])
    item_id = str(experiment["item_id"])
    started_at = experiment["started_at"]
    ends_at = experiment["ends_at"]
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    if isinstance(ends_at, str):
        ends_at = datetime.fromisoformat(ends_at.replace("Z", "+00:00"))

    # Refresh ML orders cache for the treatment window before aggregating.
    # The window may extend back duration_days, plus a small buffer for
    # late-arriving orders.
    duration_days = int(experiment.get("duration_days") or 7)
    try:
        await ml_orders_svc.refresh_for_period(
            pool, user_id, days_back=duration_days + 2,
        )
    except Exception as err:  # noqa: BLE001
        log.warning("photo-ab close: orders refresh user=%s exp=%s failed (using cache): %s",
                    user_id, experiment.get("id"), err)

    # Compute treatment metrics over [started_at, ends_at] same window length
    treatment = await _aggregate_window_for_item(
        pool, user_id, item_id, started_at, ends_at,
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE escalar_photo_experiments
               SET treatment_visits = $1,
                   treatment_orders = $2,
                   status = 'completed',
                   completed_at = NOW(),
                   updated_at = NOW()
             WHERE id = $3
            RETURNING id, user_id, item_id, duration_days,
                      baseline_visits, baseline_orders,
                      treatment_visits, treatment_orders
            """,
            treatment["visits"], treatment["orders"], int(experiment["id"]),
        )
        if not row:
            return False
        # Get user's TG settings + item title (best effort)
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
        item_title = await conn.fetchval(
            "SELECT title FROM ml_user_items WHERE user_id = $1 AND item_id = $2 LIMIT 1",
            user_id, item_id,
        )
        if not item_title:
            item_title = await conn.fetchval(
                "SELECT title FROM ml_item_context WHERE user_id = $1 AND item_id = $2 LIMIT 1",
                user_id, item_id,
            )

    if not settings or not settings["telegram_chat_id"]:
        return False

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return False

    lang = (settings["language"] or "pt").lower()
    app_base = os.environ.get("APP_BASE_URL", "https://app.lsprofit.app")
    text = _build_result_card(dict(row), item_title, lang)
    keyboard = _build_keyboard(app_base, item_id, lang)

    msg_id = await _send_card(http, bot_token, str(settings["telegram_chat_id"]), text, keyboard)
    if msg_id:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE escalar_photo_experiments
                   SET tg_dispatched_at = NOW()
                 WHERE id = $1
                """,
                int(experiment["id"]),
            )
        return True
    return False


async def dispatch_pending_results(pool: asyncpg.Pool) -> dict[str, int]:
    """Cron entry: find experiments where ends_at <= NOW() and status =
    'testing', compute treatment metrics, send TG, mark completed."""
    if pool is None:
        return {"experiments": 0, "sent": 0}
    await ensure_schema(pool)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, item_id, ai_asset_id, old_picture_id,
                   new_picture_id, duration_days,
                   started_at, ends_at,
                   baseline_visits, baseline_orders
              FROM escalar_photo_experiments
             WHERE status = 'testing'
               AND ends_at <= NOW()
             ORDER BY ends_at ASC
             LIMIT 50
            """
        )

    totals = {"experiments": 0, "sent": 0}
    if not rows:
        return totals

    async with httpx.AsyncClient() as http:
        for r in rows:
            try:
                ok = await _close_one_experiment(pool, http, dict(r))
                totals["experiments"] += 1
                if ok:
                    totals["sent"] += 1
            except Exception as err:  # noqa: BLE001
                log.exception("photo-ab close experiment %s failed: %s", r["id"], err)
            await asyncio.sleep(TG_THROTTLE)
    return totals
