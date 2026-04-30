"""Per-project per-month break-even tracker.

При каждой продаже из orders_v2 обновляем cumulative variable margin
по проекту в текущем месяце (BRT). Когда cumulative >= target_total →
ставим breakeven_reached_at и в TG приходит "🎉 Окупились!".

target_total за месяц = sum factual fixed costs (armazenagem +
aluguel + publicidade + das + fulfillment_share) + manual fixed config
(salaries + utilities + software + outros).

Reset: записи прошлых месяцев остаются для истории. State текущего
месяца получается через UNIQUE (user_id, project_id, year_month).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

# Brazil timezone — break-even считается по календарному месяцу BRT.
BRT = timezone(timedelta(hours=-3))


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS project_breakeven_state (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  project_id TEXT NOT NULL,
  year_month TEXT NOT NULL,                  -- 'YYYY-MM' BRT
  target_total_brl NUMERIC DEFAULT 0,
  target_computed_at TIMESTAMPTZ,
  target_breakdown JSONB,                    -- detail для UI: armaz/aluguel/etc
  cumulative_variable_margin_brl NUMERIC DEFAULT 0,
  sales_count INTEGER DEFAULT 0,
  breakeven_reached_at TIMESTAMPTZ,
  net_profit_after_breakeven_brl NUMERIC DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, project_id, year_month)
);
CREATE INDEX IF NOT EXISTS idx_breakeven_user_month
  ON project_breakeven_state(user_id, year_month DESC);

-- Idempotency log: один order должен инкрементировать cumulative ровно
-- один раз. Без этого replay-notice / backfill cron / live webhook все
-- перекрываются и cumulative завышается. order_id уникален per user
-- across все months (ML order ids глобально уникальны).
CREATE TABLE IF NOT EXISTS breakeven_sale_log (
  user_id INTEGER NOT NULL,
  order_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  year_month TEXT NOT NULL,
  profit_variable_brl NUMERIC NOT NULL,
  sale_date TIMESTAMPTZ,
  processed_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, order_id)
);
CREATE INDEX IF NOT EXISTS idx_breakeven_log_month
  ON breakeven_sale_log(user_id, project_id, year_month);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _ym_brt(dt: Optional[datetime] = None) -> str:
    """YYYY-MM в BRT timezone."""
    if dt is None:
        dt = datetime.now(BRT)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(BRT).strftime("%Y-%m")


def _compute_target_for_month_sync(user_id: int, project_id: str, year_month: str) -> tuple[float, dict]:
    """Считает target_total для проекта за месяц.

    sum factual fixed (armazenagem + aluguel + publicidade + das +
    fulfillment_share) — все из P&L matrix этого месяца — plus manual
    fixed costs из project.fixed_costs_monthly (salaries/utilities/etc).

    Sync — вызывается через asyncio.to_thread потому что compute_pnl
    использует pandas + JSON-cache (sync IO).
    """
    from ..legacy.db_storage import set_current_user_id
    from ..legacy.config import load_projects
    from ..legacy.reports import (
        get_publicidade_by_period, get_armazenagem_by_period,
        get_fulfillment_by_period,
    )
    from ..legacy.tax_brazil import compute_das

    set_current_user_id(user_id)

    # Year-month → period (1-st .. last day BRT).
    try:
        y_str, m_str = year_month.split("-")
        y, m = int(y_str), int(m_str)
    except (ValueError, AttributeError):
        return 0.0, {"error": "bad_year_month"}
    import calendar
    last_day = calendar.monthrange(y, m)[1]
    from datetime import date as _date
    pf = _date(y, m, 1)
    pt = _date(y, m, last_day)

    pid_upper = (project_id or "").upper()
    proj_meta = (load_projects() or {}).get(pid_upper) or {}

    # Factual costs из reports (per-month, BRL). Все могут вернуть 0 если
    # данных нет — это нормально (пример: ML не списал armazenagem в месяце).
    try:
        publi = float(get_publicidade_by_period(project_id, pf, pt).get("total", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        publi = 0.0
    try:
        armaz = float(get_armazenagem_by_period(project_id, pf, pt).get("total", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        armaz = 0.0
    try:
        fulf = float(get_fulfillment_by_period(project_id, pf, pt) or 0.0)
    except Exception:  # noqa: BLE001
        fulf = 0.0

    # DAS считаем приближённо: эффективная ставка × bruto (нужно vendas
    # данных за месяц). compute_das требует RBT12 — для фокусной задачи
    # break-even tracker'а берём 4.5% от revenue_gross проекта за месяц.
    # Если нужна точность — потом переключим на compute_pnl.tax_info.
    das = 0.0
    try:
        from ..legacy.reports import build_monthly_pnl_matrix
        m_matrix = build_monthly_pnl_matrix(project_id) or {}
        for r in (m_matrix.get("rows") or []):
            if r.get("key") == "das":
                vals = r.get("values") or {}
                das = abs(float(vals.get(year_month, 0.0) or 0.0))
                break
    except Exception:  # noqa: BLE001
        das = 0.0

    # Aluguel — пропорционально дням этого месяца. project.aluguel_mensal
    # это monthly amount (sometimes 0); если launch_date в этом месяце —
    # урезаем по факту.
    aluguel_mensal = float(proj_meta.get("aluguel_mensal") or 0.0)
    aluguel = 0.0
    if aluguel_mensal > 0:
        # Default: full monthly. Если launch_date в этом месяце, urewazaem
        # пропорционально дням (как в build_monthly_pnl_matrix).
        launch_date_str = proj_meta.get("launch_date")
        launch_date_obj = None
        if launch_date_str:
            try:
                launch_date_obj = datetime.strptime(str(launch_date_str)[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                launch_date_obj = None
        if launch_date_obj and launch_date_obj > pt:
            aluguel = 0.0
        elif launch_date_obj and launch_date_obj > pf:
            accrued_days = (pt - launch_date_obj).days + 1
            aluguel = round(aluguel_mensal * accrued_days / last_day, 2)
        else:
            aluguel = aluguel_mensal

    # Manual fixed costs — 4 категории из конфига проекта.
    manual_fc = proj_meta.get("fixed_costs_monthly") or {}
    if not isinstance(manual_fc, dict):
        manual_fc = {}
    manual_total = sum(
        max(0.0, float(manual_fc.get(k, 0) or 0))
        for k in ("salaries", "utilities", "software", "outros")
    )

    breakdown = {
        "armazenagem": round(armaz, 2),
        "aluguel": round(aluguel, 2),
        "publicidade": round(publi, 2),
        "das": round(das, 2),
        "fulfillment": round(fulf, 2),
        "manual_total": round(manual_total, 2),
    }
    target_total = round(sum(breakdown.values()), 2)
    breakdown["total"] = target_total
    return target_total, breakdown


async def get_or_init_state(
    pool: asyncpg.Pool, user_id: int, project_id: str, dt: Optional[datetime] = None,
) -> dict:
    """Возвращает state row для текущего месяца. Создаёт если нет.

    target_total компьюится при первом INSERT. Если запись старше 6 часов —
    target тоже обновляется (factual costs могут поменяться когда uploads
    приходят).
    """
    import asyncio
    ym = _ym_brt(dt)
    pid_upper = (project_id or "").upper().strip()
    if not pid_upper:
        return {}

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_id, project_id, year_month, target_total_brl,
                   target_computed_at, target_breakdown,
                   cumulative_variable_margin_brl, sales_count,
                   breakeven_reached_at, net_profit_after_breakeven_brl,
                   updated_at
              FROM project_breakeven_state
             WHERE user_id = $1 AND project_id = $2 AND year_month = $3
            """,
            user_id, pid_upper, ym,
        )

    needs_target_compute = False
    if row is None:
        needs_target_compute = True
    else:
        # Если target stale (>6h) — пересчитываем.
        tca = row["target_computed_at"]
        if tca is None or (datetime.now(timezone.utc) - tca.astimezone(timezone.utc)).total_seconds() > 21600:
            needs_target_compute = True

    if needs_target_compute:
        target_total, breakdown = await asyncio.to_thread(
            _compute_target_for_month_sync, user_id, pid_upper, ym,
        )
        import json as _json
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO project_breakeven_state
                  (user_id, project_id, year_month, target_total_brl,
                   target_computed_at, target_breakdown, updated_at)
                VALUES ($1, $2, $3, $4, NOW(), $5::jsonb, NOW())
                ON CONFLICT (user_id, project_id, year_month) DO UPDATE SET
                  target_total_brl = EXCLUDED.target_total_brl,
                  target_computed_at = NOW(),
                  target_breakdown = EXCLUDED.target_breakdown,
                  updated_at = NOW()
                RETURNING user_id, project_id, year_month, target_total_brl,
                          target_computed_at, target_breakdown,
                          cumulative_variable_margin_brl, sales_count,
                          breakeven_reached_at, net_profit_after_breakeven_brl,
                          updated_at
                """,
                user_id, pid_upper, ym, target_total,
                _json.dumps(breakdown, default=str),
            )

    out = dict(row) if row else {}
    for k, v in list(out.items()):
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


async def add_sale_and_check_breakeven(
    pool: asyncpg.Pool,
    user_id: int,
    project_id: str,
    profit_variable_brl: float,
    sale_date: Optional[datetime] = None,
    order_id: Optional[str] = None,
) -> dict:
    """Добавляет одну продажу: increment cumulative + sales_count.

    **Idempotent by order_id** — replay-notice, backfill cron, и live webhook
    могут вызвать эту функцию для одного и того же order'а. Без dedup'а
    cumulative завышался. С order_id мы пишем в breakeven_sale_log с
    ON CONFLICT DO NOTHING; если RETURNING пусто → already processed →
    возвращаем current state без changes.

    Если cumulative впервые достиг target — set breakeven_reached_at.
    После break-even каждая variable_margin идёт в net_profit_after_breakeven.

    Returns: {target_total, cumulative, sales_count, breakeven_reached_at,
             net_profit_after_breakeven, breakdown, just_reached, deduplicated}
    """
    pid_upper = (project_id or "").upper().strip()
    if not pid_upper or profit_variable_brl is None:
        return {}

    state = await get_or_init_state(pool, user_id, pid_upper, sale_date)
    if not state:
        return {}

    target = float(state.get("target_total_brl") or 0.0)
    cumulative_before = float(state.get("cumulative_variable_margin_brl") or 0.0)
    breakeven_was = state.get("breakeven_reached_at")
    sales_count_before = int(state.get("sales_count") or 0)
    net_profit_before = float(state.get("net_profit_after_breakeven_brl") or 0.0)
    margin = float(profit_variable_brl)
    ym = _ym_brt(sale_date)

    # Idempotency check via breakeven_sale_log. Без order_id скипаем
    # dedup (в этом случае всё равно инкрементируем — для совместимости
    # с callers которые ещё не передают order_id).
    if order_id:
        oid = str(order_id).strip()
        async with pool.acquire() as conn:
            inserted = await conn.fetchval(
                """
                INSERT INTO breakeven_sale_log
                  (user_id, order_id, project_id, year_month,
                   profit_variable_brl, sale_date)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, order_id) DO NOTHING
                RETURNING 1
                """,
                user_id, oid, pid_upper, ym, round(margin, 2), sale_date,
            )
        if not inserted:
            # Уже обработали этот order — возвращаем текущее состояние.
            return {
                "target_total": round(target, 2),
                "cumulative": round(cumulative_before, 2),
                "sales_count": sales_count_before,
                "breakeven_reached_at": (
                    breakeven_was.isoformat() if hasattr(breakeven_was, "isoformat")
                    else breakeven_was
                ),
                "net_profit_after_breakeven": round(net_profit_before, 2),
                "breakdown": state.get("target_breakdown") or {},
                "just_reached": False,
                "year_month": ym,
                "project_id": pid_upper,
                "deduplicated": True,
            }

    cumulative_after = cumulative_before + margin
    sales_count = sales_count_before + 1
    just_reached = False
    new_breakeven_at = breakeven_was
    net_profit_after = net_profit_before
    if cumulative_after >= target and target > 0 and not breakeven_was:
        new_breakeven_at = datetime.now(timezone.utc).isoformat()
        just_reached = True
        # Какая часть margin'а пошла после break-even
        net_profit_after += max(0.0, cumulative_after - target)
    elif breakeven_was:
        # Уже окупились — все следующие margin идут в net profit.
        net_profit_after += margin

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE project_breakeven_state
               SET cumulative_variable_margin_brl = $4,
                   sales_count = $5,
                   breakeven_reached_at = COALESCE($6::timestamptz, breakeven_reached_at),
                   net_profit_after_breakeven_brl = $7,
                   updated_at = NOW()
             WHERE user_id = $1 AND project_id = $2 AND year_month = $3
            """,
            user_id, pid_upper, ym,
            round(cumulative_after, 2), sales_count,
            new_breakeven_at, round(net_profit_after, 2),
        )

    return {
        "target_total": round(target, 2),
        "cumulative": round(cumulative_after, 2),
        "sales_count": sales_count,
        "breakeven_reached_at": new_breakeven_at,
        "net_profit_after_breakeven": round(net_profit_after, 2),
        "breakdown": state.get("target_breakdown") or {},
        "just_reached": just_reached,
        "year_month": ym,
        "project_id": pid_upper,
        "deduplicated": False,
    }


async def backfill_log_from_orders(
    pool: asyncpg.Pool, user_id: int, project_id: str, year_month: str,
) -> dict:
    """Recovery: populate breakeven_sale_log из ml_user_orders за месяц.

    Идёт по orders за month BRT, для каждого order находит margin в
    ml_item_margin_cache (по item_id), вызывает apply_hypothetical_price
    с unit_price → получает profit_variable. INSERT в log с
    ON CONFLICT DO NOTHING — идемпотентно.

    Filter по project_id: пропускаем orders где margin.project ≠ project_id.

    После этого recompute_state_from_log даст правильный cumulative.
    """
    import json as _json
    from . import ml_item_margin as ml_margin_svc

    pid_upper = (project_id or "").upper().strip()
    try:
        y_str, m_str = year_month.split("-")
        y, m = int(y_str), int(m_str)
    except (ValueError, AttributeError):
        return {"error": "bad_year_month"}
    import calendar
    last_day = calendar.monthrange(y, m)[1]
    start_brt = datetime(y, m, 1, tzinfo=BRT)
    end_brt = datetime(y, m, last_day, 23, 59, 59, tzinfo=BRT)

    inserted = 0
    skipped_wrong_project = 0
    skipped_no_margin = 0
    skipped_no_item = 0
    processed = 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT order_id, status, items, date_created
              FROM ml_user_orders
             WHERE user_id = $1
               AND date_created BETWEEN $2 AND $3
               AND status NOT IN ('cancelled', 'invalid')
            """,
            user_id, start_brt, end_brt,
        )

    for row in rows:
        processed += 1
        order_id = str(row["order_id"]) if row["order_id"] else None
        if not order_id:
            continue
        items_raw = row["items"]
        if isinstance(items_raw, str):
            try:
                items_raw = _json.loads(items_raw)
            except Exception:  # noqa: BLE001
                continue
        if not isinstance(items_raw, list) or not items_raw:
            skipped_no_item += 1
            continue
        first = items_raw[0]
        item_id = str(first.get("mlb") or "").strip().upper()
        try:
            sale_price = float(first.get("unit_price") or 0)
        except (TypeError, ValueError):
            sale_price = 0.0
        try:
            qty = int(first.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1
        if not item_id or sale_price <= 0:
            skipped_no_item += 1
            continue

        async with pool.acquire() as conn:
            mrow = await conn.fetchrow(
                """
                SELECT payload FROM ml_item_margin_cache
                 WHERE user_id = $1 AND item_id = $2 AND period_months = 3
                """,
                user_id, item_id,
            )
        if not mrow:
            skipped_no_margin += 1
            continue
        payload = mrow["payload"]
        if isinstance(payload, str):
            try:
                payload = _json.loads(payload)
            except Exception:  # noqa: BLE001
                skipped_no_margin += 1
                continue
        if not isinstance(payload, dict):
            skipped_no_margin += 1
            continue

        cache_project = (payload.get("project") or "").upper().strip()
        if cache_project != pid_upper:
            skipped_wrong_project += 1
            continue

        try:
            recomputed = ml_margin_svc.apply_hypothetical_price(payload, sale_price)
            unit_pv = (recomputed.get("unit") or {}).get("profit_variable")
            if unit_pv is None:
                skipped_no_margin += 1
                continue
            total_pv = float(unit_pv) * qty
        except Exception:  # noqa: BLE001
            skipped_no_margin += 1
            continue

        # Insert into log idempotently.
        async with pool.acquire() as conn:
            ok = await conn.fetchval(
                """
                INSERT INTO breakeven_sale_log
                  (user_id, order_id, project_id, year_month,
                   profit_variable_brl, sale_date)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, order_id) DO NOTHING
                RETURNING 1
                """,
                user_id, order_id, pid_upper, year_month,
                round(total_pv, 2), row["date_created"],
            )
        if ok:
            inserted += 1

    return {
        "user_id": user_id,
        "project_id": pid_upper,
        "year_month": year_month,
        "orders_processed": processed,
        "log_inserted": inserted,
        "skipped_wrong_project": skipped_wrong_project,
        "skipped_no_margin_cache": skipped_no_margin,
        "skipped_no_item": skipped_no_item,
    }


async def recompute_state_from_log(
    pool: asyncpg.Pool, user_id: int, project_id: str, year_month: str,
) -> dict:
    """Recovery: пересчитать project_breakeven_state.cumulative по факту из
    breakeven_sale_log за месяц. Используется когда cumulative завышено
    из-за исторических double-increments (до idempotency fix).

    target_total и breakdown — НЕ трогаем (они вычисляются из P&L matrix
    через get_or_init_state и по своему refresh-расписанию).
    """
    pid_upper = (project_id or "").upper().strip()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(profit_variable_brl), 0) AS sum_profit,
                   COUNT(*) AS n_sales,
                   MIN(sale_date) AS first_sale,
                   MAX(sale_date) AS last_sale
              FROM breakeven_sale_log
             WHERE user_id = $1 AND project_id = $2 AND year_month = $3
            """,
            user_id, pid_upper, year_month,
        )
        sum_profit = float(row["sum_profit"] or 0.0)
        n_sales = int(row["n_sales"] or 0)

        # Получаем target из state (создаём если нет).
        state = await get_or_init_state(pool, user_id, pid_upper)
        target = float(state.get("target_total_brl") or 0.0)

        breakeven_at = None
        net_after = 0.0
        if target > 0 and sum_profit >= target:
            # Approximate: если cumulative >= target, ставим breakeven_at = NOW
            # (точное время мы потеряли — без log-ordered scan).
            breakeven_at = datetime.now(timezone.utc).isoformat()
            net_after = max(0.0, sum_profit - target)

        await conn.execute(
            """
            UPDATE project_breakeven_state
               SET cumulative_variable_margin_brl = $4,
                   sales_count = $5,
                   breakeven_reached_at = $6::timestamptz,
                   net_profit_after_breakeven_brl = $7,
                   updated_at = NOW()
             WHERE user_id = $1 AND project_id = $2 AND year_month = $3
            """,
            user_id, pid_upper, year_month,
            round(sum_profit, 2), n_sales, breakeven_at,
            round(net_after, 2),
        )

    return {
        "user_id": user_id, "project_id": pid_upper, "year_month": year_month,
        "target_total": round(target, 2),
        "cumulative": round(sum_profit, 2),
        "sales_count": n_sales,
        "breakeven_reached_at": breakeven_at,
        "net_profit_after_breakeven": round(net_after, 2),
        "first_sale": row["first_sale"].isoformat() if row["first_sale"] else None,
        "last_sale": row["last_sale"].isoformat() if row["last_sale"] else None,
    }
