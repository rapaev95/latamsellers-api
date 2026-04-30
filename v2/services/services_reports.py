"""Services-projects ОПиУ / ДДС / Balance generator.

Wraps legacy `generate_opiu_estonia / generate_dds_estonia / generate_balance_estonia
/ calculate_trafficstars_fifo` from `v2/legacy/reports.py` with a `project_id`-aware
public API. Today's reality:

  * Legacy functions are HARDCODED on Estonia (SHPS tomador, hardcoded transfers
    list, bracket rates 15.50/16.75/18.75/19.75%, hardcoded saldo_inicial /
    total_enviado from a curated CSV).
  * Project record now CARRIES the same data as fields (tomador_cnpj,
    commission_brackets, services_opening, das_apportionment) — see
    `v2/schemas/finance.py` and `v2/legacy/config.py:_PROJECT_EDITABLE_KEYS`.

This module is the bridge: caller passes `project_id`, we read the record,
and either
  (a) for project_id='ESTONIA' (the curated case) — delegate to the legacy
      hardcoded functions and return their output shape;
  (b) for other services-projects — refuse with a `needs_config` placeholder
      until the legacy functions are refactored to accept project record as
      input (Phase 6 — out of scope for this session).

This keeps endpoints + UI generic now, defers the heavy refactor.

TODO(phase-6): replace `generate_opiu_estonia()` etc with parametric
`generate_invoice_pnl(project, ...)` reading from `project["commission_brackets"]`,
`load_all_nfse(tomador_cnpj=project["tomador_cnpj"])`,
`manual_cashflow_entries[kind='approved_transfer'][project_id]` etc.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


# Project IDs that the *current* hardcoded legacy generators treat as their
# subject. Streamlit hardcoded everything on "Estonia"; if user names the
# project differently, we still want the hardcoded path to fire. Match on
# upper-cased project_id with substring detection.
_LEGACY_HARDCODED_ALIASES: tuple[str, ...] = ("ESTONIA",)


def _is_legacy_hardcoded(project_id: str) -> bool:
    pid = (project_id or "").upper()
    return any(alias in pid for alias in _LEGACY_HARDCODED_ALIASES)


def _empty_pnl(reason: str) -> dict[str, Any]:
    """Placeholder shape when we can't actually compute (project not configured
    or unsupported). Same keys as the real generator so the UI doesn't have
    to special-case structure."""
    return {
        "total_gross": 0.0,
        "total_tax_retained": 0.0,
        "total_net_client": 0.0,
        "invoice_count": 0,
        "saldo_inicial": 0.0,
        "total_enviado": 0.0,
        "debito_estonia": 0.0,
        "our_commission": 0.0,
        "our_rental_paid_usd": 0.0,
        "our_rental_pending_usd": 0.0,
        "our_revenue_brl": 0.0,
        "our_das_paid": 0.0,
        "our_das_estimated": 0.0,
        "our_das": 0.0,
        "total_company_das": 0.0,
        "total_trade_das": 0.0,
        "trade_das_rate": 0.0,
        "our_profit_brl": 0.0,
        "our_profit_brl_paid_only": 0.0,
        "das_payments": [],
        "das_pending": [],
        "pnl_by_month": [],
        "current_bracket": "—",
        "next_bracket": "—",
        "cumulative_gross": 0.0,
        "by_month": {},
        "_needs_config": True,
        "_reason": reason,
    }


def _empty_cashflow(reason: str) -> dict[str, Any]:
    return {
        "inflows": {"saldo_inicial": 0.0, "invoices_gross": 0.0,
                    "invoices_tax": 0.0, "invoices_net": 0.0},
        "outflows": {},
        "transfers": [],
        "total_outflows": 0.0,
        "debito_estonia": 0.0,
        "by_month": {},
        "_needs_config": True,
        "_reason": reason,
    }


def _empty_balance(reason: str) -> dict[str, Any]:
    return {
        "approved_date": None,
        "saldo_inicial": 0.0,
        "our_commission": 0.0,
        "our_rental_paid_usd": 0.0,
        "our_das": 0.0,
        "our_profit_brl": 0.0,
        "caliza_brl": 0.0,
        "bybit_brl": 0.0,
        "ts_fifo_brl": 0.0,
        "usd_in_stock": 0.0,
        "brl_value_in_stock": 0.0,
        "debito_real": 0.0,
        "_needs_config": True,
        "_reason": reason,
    }


def _validate_project(project_id: str) -> tuple[bool, Optional[str]]:
    """Returns (ok, error_reason). Loads project record, checks type='services'."""
    try:
        from v2.legacy import config as legacy_config
        projects = legacy_config.load_projects()
    except Exception as err:  # noqa: BLE001
        return False, f"projects_unavailable: {err}"
    proj = projects.get(project_id)
    if not proj:
        return False, "project_not_found"
    if proj.get("type") != "services":
        return False, f"not_services_project (type={proj.get('type')!r})"
    return True, None


# ── Public API ──────────────────────────────────────────────────────────────


def generate_services_pnl_sync(project_id: str) -> dict[str, Any]:
    """Sync — the legacy generator under the hood is fully sync. Caller from
    async context wraps in `asyncio.to_thread`."""
    ok, reason = _validate_project(project_id)
    if not ok:
        return _empty_pnl(reason or "unknown")

    if _is_legacy_hardcoded(project_id):
        from v2.legacy.reports import generate_opiu_estonia
        result = generate_opiu_estonia()
        result["_needs_config"] = False
        return result

    # Project is type='services' but not Estonia — Phase 6 will read brackets +
    # tomador from project record. For now, return placeholder.
    return _empty_pnl(
        "not_yet_supported: only ESTONIA is wired to legacy generators; "
        "configure tomador_cnpj/commission_brackets and wait for Phase 6 refactor"
    )


def generate_services_cashflow_sync(project_id: str) -> dict[str, Any]:
    ok, reason = _validate_project(project_id)
    if not ok:
        return _empty_cashflow(reason or "unknown")

    if _is_legacy_hardcoded(project_id):
        from v2.legacy.reports import generate_dds_estonia
        result = generate_dds_estonia()
        result["_needs_config"] = False
        return result

    return _empty_cashflow(
        "not_yet_supported: only ESTONIA is wired (Phase 6 generic refactor pending)"
    )


def generate_services_balance_sync(project_id: str) -> dict[str, Any]:
    ok, reason = _validate_project(project_id)
    if not ok:
        return _empty_balance(reason or "unknown")

    if _is_legacy_hardcoded(project_id):
        from v2.legacy.reports import generate_balance_estonia
        result = generate_balance_estonia()
        result["_needs_config"] = False
        return result

    return _empty_balance(
        "not_yet_supported: only ESTONIA is wired (Phase 6 generic refactor pending)"
    )


# ── Async wrappers — endpoint calls these via asyncio.to_thread ─────────────


async def compute_for_user(
    pool: asyncpg.Pool,
    user_id: int,
    project_id: str,
    period_from: Optional[date] = None,
    period_to: Optional[date] = None,
) -> dict[str, Any]:
    """Build the full services-reports bundle for one project.

    Returns `{project, period, pnl, cashflow, balance, errors}` shape matching
    `ServicesReportsBundleOut` Pydantic model. Errors on individual sub-reports
    are reported per-tab (same pattern as ecom `/finance/reports`) so a partial
    failure doesn't blank the whole page.

    `pool` and `user_id` carried for forward-compat — Phase 6 generic refactor
    will use them to query project-scoped tables (manual_cashflow_entries,
    manual_usd_inflows, etc.).
    """
    out: dict[str, Any] = {
        "project": project_id,
        "period": {
            "from": period_from.isoformat() if period_from else None,
            "to": period_to.isoformat() if period_to else None,
        },
    }

    # Run three sync legacy computes off the event loop. Each one is
    # independent — if one raises, the others still produce.
    async def _safe(label: str, fn):
        try:
            return await asyncio.to_thread(fn)
        except Exception as err:  # noqa: BLE001
            log.warning("services_reports.%s failed for %s: %s", label, project_id, err)
            return None, err

    pnl_task = asyncio.create_task(asyncio.to_thread(generate_services_pnl_sync, project_id))
    cf_task = asyncio.create_task(asyncio.to_thread(generate_services_cashflow_sync, project_id))
    bal_task = asyncio.create_task(asyncio.to_thread(generate_services_balance_sync, project_id))

    try:
        out["pnl"] = await pnl_task
    except Exception as err:  # noqa: BLE001
        out["pnl_error"] = f"{type(err).__name__}: {err}"
    try:
        out["cashflow"] = await cf_task
    except Exception as err:  # noqa: BLE001
        out["cashflow_error"] = f"{type(err).__name__}: {err}"
    try:
        out["balance"] = await bal_task
    except Exception as err:  # noqa: BLE001
        out["balance_error"] = f"{type(err).__name__}: {err}"

    return out
