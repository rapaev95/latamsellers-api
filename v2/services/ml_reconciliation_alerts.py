"""Weekly reconciliation: ML fee report vs bank_tx + Full Express vs bank_tx.

Phase 1 — totals comparison per project per month. Если расхождение
превышает R$50 → admin TG card с деталями. Точная mapping (по строкам)
оставлена для Phase 2 — там нужно учитывать batch-списания и timing.

Sources:
  - ML fees: fatura_ml*.csv (parsed by legacy/reports → aggregated total)
  - Full Express: full_express*.{csv,pdf} (parsed by existing pipeline)
  - bank_tx: aggregate_classified_by_project с category in
    {ml_fee, mercadolivre, fulfillment}

Cron: Monday 14:00 UTC = 11:00 BRT (после Mon ads_summary 11 UTC).

Idempotency: log per (user, project, year_month, source) → не повторять
alert если уже отправили в этом цикле.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))
TG_API_BASE = "https://api.telegram.org"
DEFAULT_THRESHOLD_BRL = 50.0

CREATE_LOG_SQL = """
CREATE TABLE IF NOT EXISTS reconciliation_alert_log (
  user_id INTEGER NOT NULL,
  project TEXT NOT NULL,
  year_month TEXT NOT NULL,
  source TEXT NOT NULL,
  expected_brl NUMERIC,
  actual_brl NUMERIC,
  delta_brl NUMERIC,
  alerted_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (user_id, project, year_month, source)
);
CREATE INDEX IF NOT EXISTS idx_recon_alert_user
  ON reconciliation_alert_log(user_id, alerted_at DESC);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_LOG_SQL)


def _ym_brt(d: date) -> str:
    return d.strftime("%Y-%m")


def _month_bounds(target_ym: str) -> tuple[date, date]:
    y, m = map(int, target_ym.split("-"))
    import calendar
    last = calendar.monthrange(y, m)[1]
    return date(y, m, 1), date(y, m, last)


def _aggregate_ml_fees(project: str, period_from: date, period_to: date) -> float:
    """Total ML commission fees из fatura_ml за период. Использует существующий
    parser в legacy/reports."""
    try:
        from v2.legacy.reports import get_fatura_ml_by_period
    except ImportError:
        return 0.0
    try:
        result = get_fatura_ml_by_period(project, period_from, period_to)
        if isinstance(result, dict):
            return float(result.get("total") or result.get("comissoes") or 0.0)
        return float(result or 0.0)
    except Exception as err:  # noqa: BLE001
        log.debug("get_fatura_ml_by_period failed: %s", err)
        return 0.0


def _aggregate_full_express(project: str, period_from: date, period_to: date) -> float:
    """Total Full Express из CSV за период."""
    try:
        from v2.legacy.reports import get_full_express_by_period
    except ImportError:
        return 0.0
    try:
        result = get_full_express_by_period(project, period_from, period_to)
        if isinstance(result, dict):
            return float(result.get("total") or 0.0)
        return float(result or 0.0)
    except Exception as err:  # noqa: BLE001
        log.debug("get_full_express_by_period failed: %s", err)
        return 0.0


def _aggregate_bank_tx_by_category(project: str, category_keywords: list[str],
                                    period_from: date, period_to: date) -> float:
    """Sum bank_tx where Категория lowercase contains any keyword."""
    total = 0.0
    try:
        from v2.legacy.reports import aggregate_classified_by_project
        live = aggregate_classified_by_project(project) or {}
    except Exception as err:  # noqa: BLE001
        log.debug("aggregate_classified failed: %s", err)
        return 0.0
    keywords_l = {k.lower() for k in category_keywords}
    for tx in (live.get("transactions") or []):
        cat = str(tx.get("Категория", "") or "").lower()
        if not any(k in cat for k in keywords_l):
            continue
        ds = str(tx.get("Data", ""))
        try:
            import pandas as _pd
            td = _pd.to_datetime(ds, dayfirst=True).date()
        except Exception:  # noqa: BLE001
            continue
        if not (period_from <= td <= period_to):
            continue
        try:
            total += abs(float(tx.get("Valor", 0) or 0))
        except (ValueError, TypeError):
            pass
    return round(total, 2)


async def _was_alerted_this_month(
    pool: asyncpg.Pool, user_id: int, project: str, year_month: str, source: str,
) -> bool:
    async with pool.acquire() as conn:
        v = await conn.fetchval(
            """
            SELECT 1 FROM reconciliation_alert_log
             WHERE user_id = $1 AND project = $2 AND year_month = $3 AND source = $4
            """,
            user_id, project, year_month, source,
        )
    return v is not None


async def _record_alert(
    pool: asyncpg.Pool, user_id: int, project: str, year_month: str,
    source: str, expected: float, actual: float, delta: float,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO reconciliation_alert_log
              (user_id, project, year_month, source, expected_brl, actual_brl, delta_brl)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id, project, year_month, source) DO UPDATE
              SET expected_brl = EXCLUDED.expected_brl,
                  actual_brl = EXCLUDED.actual_brl,
                  delta_brl = EXCLUDED.delta_brl,
                  alerted_at = NOW()
            """,
            user_id, project, year_month, source,
            float(expected), float(actual), float(delta),
        )


async def reconcile_for_user(
    pool: asyncpg.Pool, user_id: int,
    threshold_brl: float = DEFAULT_THRESHOLD_BRL,
) -> dict[str, int]:
    """Per-user reconciliation за last completed month + current month-to-date.
    Iterates user projects, compares ML fees vs bank, Full Express vs bank.
    Returns counts of alerts fired."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {"alerts": 0, "reason": "no_bot_token"}
    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, COALESCE(language, 'pt') AS language
              FROM notification_settings
             WHERE user_id = $1 AND telegram_chat_id IS NOT NULL
            """,
            user_id,
        )
    if not settings:
        return {"alerts": 0, "reason": "no_chat_id"}

    # Bind legacy db_storage user-context.
    try:
        from v2.legacy import db_storage as _legacy_db
        from v2.legacy.config import load_projects
        _legacy_db.set_current_user_id(user_id)
        projects = list((load_projects() or {}).keys())
    except Exception as err:  # noqa: BLE001
        log.warning("legacy projects load failed user=%s: %s", user_id, err)
        return {"alerts": 0, "reason": "legacy_unavailable"}

    today = datetime.now(BRT).date()
    # Check last completed month + current MTD.
    months_to_check: list[str] = []
    last_month = today.replace(day=1) - timedelta(days=1)
    months_to_check.append(_ym_brt(last_month))
    months_to_check.append(_ym_brt(today))

    issues: list[dict[str, Any]] = []
    for project in projects:
        for ym in months_to_check:
            pf, pt_ = _month_bounds(ym)
            # Cap to today для current month
            pt_ = min(pt_, today)

            # ML fees
            ml_total = _aggregate_ml_fees(project, pf, pt_)
            bank_ml = _aggregate_bank_tx_by_category(
                project, ["mercadolivre", "ml_fee", "ml_comissao"], pf, pt_,
            )
            delta_ml = ml_total - bank_ml
            if (ml_total > 0 or bank_ml > 0) and abs(delta_ml) > threshold_brl:
                issues.append({
                    "project": project, "year_month": ym, "source": "ml_fees",
                    "expected": ml_total, "actual": bank_ml, "delta": delta_ml,
                })

            # Full Express
            fe_total = _aggregate_full_express(project, pf, pt_)
            bank_fe = _aggregate_bank_tx_by_category(
                project, ["fulfillment", "full_express", "ml_full"], pf, pt_,
            )
            delta_fe = fe_total - bank_fe
            if (fe_total > 0 or bank_fe > 0) and abs(delta_fe) > threshold_brl:
                issues.append({
                    "project": project, "year_month": ym, "source": "full_express",
                    "expected": fe_total, "actual": bank_fe, "delta": delta_fe,
                })

    if not issues:
        return {"alerts": 0, "checked_combinations": len(months_to_check) * len(projects) * 2}

    # Filter out already-alerted combinations (per-month dedup).
    new_issues: list[dict[str, Any]] = []
    for iss in issues:
        if not await _was_alerted_this_month(
            pool, user_id, iss["project"], iss["year_month"], iss["source"],
        ):
            new_issues.append(iss)
    if not new_issues:
        return {"alerts": 0, "all_already_alerted": True}

    # Build TG card.
    chat_id = str(settings["telegram_chat_id"])
    lang = (settings["language"] or "pt").lower()
    if lang == "ru":
        header = "🚨 *\\[АУДИТ\\] Расхождения учёт vs банк*"
        col_lbl = "Расхождение"
        underwithheld_lbl = "ML списал больше чем в банке"
        overwithheld_lbl = "ML начислил меньше чем списали в банке"
        action = "_Проверь fatura ML / Full Express vs банковскую выписку — могут быть пропущенные или двойные списания._"
    elif lang == "en":
        header = "🚨 *\\[AUDIT\\] Books vs bank discrepancies*"
        col_lbl = "Delta"
        underwithheld_lbl = "ML billed more than bank shows"
        overwithheld_lbl = "ML billed less than bank shows"
        action = "_Compare fatura ML / Full Express CSV vs bank statement — possible missed or duplicate charges._"
    else:
        header = "🚨 *\\[AUDITORIA\\] Discrepâncias livro vs banco*"
        col_lbl = "Delta"
        underwithheld_lbl = "ML cobrou mais do que aparece no banco"
        overwithheld_lbl = "ML cobrou menos do que aparece no banco"
        action = "_Confronte fatura ML / Full Express CSV vs extrato bancário — pode haver lançamento perdido ou duplicado._"

    lines = [header, ""]
    for iss in new_issues:
        delta = iss["delta"]
        sign = "+" if delta > 0 else ""
        explanation = underwithheld_lbl if delta > 0 else overwithheld_lbl
        lines.append(
            f"• *{iss['project']}* {iss['year_month']} · "
            f"{iss['source']}: livro R$ {iss['expected']:.2f} vs banco "
            f"R$ {iss['actual']:.2f} → {sign}R$ {delta:.2f}"
        )
        lines.append(f"  _{explanation}_")
    lines.append("")
    lines.append(action)

    text = "\n".join(lines).replace("$", "\\$")  # MD2 keeps $ literal но safer

    import httpx
    async with httpx.AsyncClient(timeout=15.0) as http:
        try:
            r = await http.post(
                f"{TG_API_BASE}/bot{bot_token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "MarkdownV2",
                    "disable_web_page_preview": True,
                },
            )
            if r.status_code == 200:
                for iss in new_issues:
                    await _record_alert(
                        pool, user_id, iss["project"], iss["year_month"],
                        iss["source"], iss["expected"], iss["actual"], iss["delta"],
                    )
                return {"alerts": len(new_issues)}
            else:
                log.warning("reconciliation TG send failed status=%s body=%s",
                            r.status_code, r.text[:200])
        except Exception as err:  # noqa: BLE001
            log.exception("reconciliation TG send failed: %s", err)
    return {"alerts": 0, "send_failed": True}


async def reconcile_all_users(
    pool: asyncpg.Pool, threshold_brl: float = DEFAULT_THRESHOLD_BRL,
) -> dict[str, int]:
    """Cron entrypoint."""
    if pool is None:
        return {"users": 0}
    await ensure_schema(pool)
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
    totals = {"users": 0, "alerts": 0}
    for r in rows:
        try:
            res = await reconcile_for_user(pool, r["user_id"], threshold_brl)
            totals["users"] += 1
            totals["alerts"] += res.get("alerts", 0)
        except Exception as err:  # noqa: BLE001
            log.exception("reconciliation user=%s failed: %s", r["user_id"], err)
    return totals
