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
) -> dict:
    """Добавляет одну продажу: increment cumulative + sales_count.

    Если cumulative впервые достиг target — set breakeven_reached_at.
    После break-even каждая variable_margin идёт в net_profit_after_breakeven.

    Returns: {target_total, cumulative, sales_count, breakeven_reached_at,
             net_profit_after_breakeven, breakdown, just_reached}
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

    margin = float(profit_variable_brl)
    cumulative_after = cumulative_before + margin
    sales_count = int(state.get("sales_count") or 0) + 1

    just_reached = False
    new_breakeven_at = breakeven_was
    net_profit_after = float(state.get("net_profit_after_breakeven_brl") or 0.0)
    if cumulative_after >= target and target > 0 and not breakeven_was:
        new_breakeven_at = datetime.now(timezone.utc).isoformat()
        just_reached = True
        # Какая часть margin'а пошла после break-even
        net_profit_after += max(0.0, cumulative_after - target)
    elif breakeven_was:
        # Уже окупились — все следующие margin идут в net profit.
        net_profit_after += margin

    ym = _ym_brt(sale_date)
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
    }
