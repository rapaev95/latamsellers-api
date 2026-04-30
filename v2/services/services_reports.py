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
from datetime import date, datetime
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)


# ── Dynamic rate computation (Phase F2) ──────────────────────────────────────
#
# Two models the project record can carry, both checked here so caller doesn't
# need to know the difference:
#
#   1. `commission_formula` — Simples Nacional progressive table (Anexo III/I/II)
#      with the user's flat margin on top. Used when rate grows continuously
#      with RBT12 within a faixa (GANZA: 20.85% → 22.71% inside faixa 4).
#
#   2. `commission_brackets` — hand-curated [{ceiling_brl, rate_pct}] table.
#      Used when rate is locked per-faixa (LATAMSELLERS: 15.68 / 17.32 /
#      18.75 / 19.75 with rate-cap effect on faixa 4).
#
# `compute_progressive_rate` and `compute_bracket_rate` return the same shape
# so the per-month iterator can pick either without branching downstream.


def compute_progressive_rate(rbt12: float, anexo: str, margin_pct: float) -> dict[str, Any]:
    """`total_rate = effective_simples(rbt12, anexo) + margin_pct`.

    Reuses the canonical Simples Nacional tables in `v2/legacy/tax_brazil.py`
    (Anexo I/II/III) so faixa boundaries + parcela values stay in sync with
    the rest of the app's tax math.

    Returns a dict with `total_rate_pct` (FRACTION, not percentage points) plus
    diagnostic fields the UI shows in the per-month breakdown.
    """
    from v2.legacy.tax_brazil import compute_simples_effective
    res = compute_simples_effective(rbt12, anexo)
    # tax_brazil returns effective_pct as a percentage (e.g. 11.97 for 11.97%);
    # we work in fractions internally so margin_pct (0.09) adds cleanly.
    effective_frac = float(res["effective_pct"]) / 100.0
    total = effective_frac + float(margin_pct)
    return {
        "rbt12": rbt12,
        "faixa": int(res["faixa"]),
        "anexo": anexo,
        "effective_pct": effective_frac,           # 0.1197 = 11.97%
        "margin_pct": float(margin_pct),
        "total_rate_pct": total,                   # 0.2097 = 20.97%
        "nominal_pct": float(res["aliquota_nominal"]) / 100.0,
        "parcela_deduzir": float(res["parcela_deduzir"]),
        "exceed_limit": bool(res["exceed_limit"]),
        "source": "formula",
    }


def compute_bracket_rate(rbt12: float, brackets: list[dict[str, Any]]) -> dict[str, Any]:
    """First bracket with `rbt12 <= ceiling_brl` wins; fallback to highest bracket.

    Brackets stored as `[{ceiling_brl, rate_pct}]` with `rate_pct` already a
    fraction. No effective/margin split — these are pre-curated final rates
    (Estonia / LATAMSELLERS style).
    """
    sorted_b = sorted(brackets, key=lambda b: float(b.get("ceiling_brl") or 0))
    if not sorted_b:
        return {"rbt12": rbt12, "total_rate_pct": 0.0, "source": "brackets_empty"}
    for b in sorted_b:
        ceiling = float(b.get("ceiling_brl") or 0)
        if rbt12 <= ceiling:
            return {
                "rbt12": rbt12,
                "ceiling_brl": ceiling,
                "total_rate_pct": float(b.get("rate_pct") or 0),
                "effective_pct": None,             # not split for hand-curated tables
                "margin_pct": None,
                "source": "brackets",
            }
    last = sorted_b[-1]
    return {
        "rbt12": rbt12,
        "ceiling_brl": float(last.get("ceiling_brl") or 0),
        "total_rate_pct": float(last.get("rate_pct") or 0),
        "effective_pct": None,
        "margin_pct": None,
        "source": "brackets_above_cap",
    }


def _rolling_rbt12(history: list[tuple[str, float]], current_iso: str, baseline: float = 0.0) -> float:
    """Sliding 12-month sum: revenue from `history` whose date falls in
    [current − 12 months, current], inclusive of current_iso's month. Adds
    `baseline` (used when project record carries pre-loaded RBT12 from a
    period before our local history starts — e.g. fresh services-project
    migrating from a paper-trail past).

    `history` items are `(date_iso, gross_brl)` already accumulated by caller.
    `current_iso` is the date of the row we're computing the rate for.
    """
    try:
        cur = datetime.strptime(current_iso[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return baseline
    # Window opens 12 months back, inclusive — i.e. for May 2026 it's
    # [Jun 2025, May 2026]. Approximation: 365 days back works for MVP;
    # exact "same day previous year" would need calendar arithmetic.
    cutoff = date.fromordinal(max(1, cur.toordinal() - 365))
    s = baseline
    for ds, g in history:
        try:
            d = datetime.strptime(ds[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if cutoff <= d <= cur:
            s += float(g or 0)
    return s


def compute_pnl_dynamic(project: dict[str, Any], invoices: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the ОПиУ for a services-project from a list of invoices.

    `invoices`: `[{date: 'YYYY-MM-DD', gross: float, ...}]` — caller pulls
    these from NFS-e uploads (`load_all_nfse` filtered by tomador_cnpj) or
    `manual_cashflow_entries[approved_invoice]` for the curated past.

    For each invoice:
      1. Update the rolling 12-month sum (`rbt12`) including all prior
         invoices in the same projection (+ optional baseline).
      2. Compute rate from project config — formula or brackets.
      3. tax = gross × rate.

    Returns a dict aligned with `ServicesPnlOut`'s pnl_by_month + totals.
    `pnl_by_month` rows include `rbt12 / faixa / effective_pct / margin_pct
    / total_rate_pct` so the UI can render the per-row breakdown the user
    asked for.
    """
    formula = project.get("commission_formula") or None
    brackets = project.get("commission_brackets") or []
    baseline = float(formula.get("baseline_rbt12", 0.0)) if formula else 0.0

    invoices_sorted = sorted(invoices, key=lambda inv: str(inv.get("date") or "")[:10])
    history: list[tuple[str, float]] = []
    pnl_by_month: list[dict[str, Any]] = []
    by_month: dict[str, dict[str, Any]] = {}

    total_gross = 0.0
    total_tax = 0.0

    for inv in invoices_sorted:
        d_iso = str(inv.get("date") or "")[:10]
        gross = float(inv.get("gross") or 0)
        if gross <= 0 or not d_iso:
            continue

        # Different "RBT12 reference point" conventions per mode:
        #
        # - formula (GANZA-style): rate uses RBT12 AFTER current invoice is
        #   added — matches user's spreadsheet column "Окно до" = baseline +
        #   sum-including-current.
        # - brackets (LATAMSELLERS-style): rate uses RBT12 BEFORE current —
        #   matches Receita Federal's strict definition (12 prior months only)
        #   and matches user's spreadsheet "Окно 0 / 210k / 420k" pattern.
        #
        # Both are valid; the user picks via mode. Receita's definition is the
        # legal one but the user's GANZA invoices use the "after" convention,
        # so we honour both. (Future: expose `rbt12_includes_current: bool`
        # for explicit override if needed.)
        is_formula = bool(formula and formula.get("mode") == "simples_progressive")

        if is_formula:
            history.append((d_iso, gross))
            rbt12 = _rolling_rbt12(history, d_iso, baseline=baseline)
        else:
            # bracket mode — RBT12 from history BEFORE adding this invoice
            rbt12 = _rolling_rbt12(history, d_iso, baseline=baseline)
            history.append((d_iso, gross))

        # Pick rate from whichever config the project has set.
        if is_formula:
            rate_info = compute_progressive_rate(
                rbt12,
                str(formula.get("simples_anexo") or "III"),
                float(formula.get("margin_pct") or 0.09),
            )
        elif brackets:
            rate_info = compute_bracket_rate(rbt12, brackets)
        else:
            rate_info = {"rbt12": rbt12, "total_rate_pct": 0.0, "source": "none"}

        rate_frac = float(rate_info.get("total_rate_pct") or 0)
        tax = gross * rate_frac
        total_gross += gross
        total_tax += tax

        m_key = d_iso[:7]
        if m_key not in by_month:
            by_month[m_key] = {"gross": 0.0, "tax": 0.0, "count": 0}
        by_month[m_key]["gross"] += gross
        by_month[m_key]["tax"] += tax
        by_month[m_key]["count"] += 1

        # Per-row payload for UI table (matches the user's spreadsheet
        # columns: Месяц / Поступление / Окно до / % ставка / Налог).
        pnl_by_month.append({
            "month": m_key,
            "date": d_iso,
            "invoice_count": 1,
            "invoice_gross": gross,
            "rbt12": round(rbt12, 2),
            "faixa": rate_info.get("faixa"),
            "effective_pct": rate_info.get("effective_pct"),
            "margin_pct": rate_info.get("margin_pct"),
            "total_rate_pct": round(rate_frac, 6),
            "commission": round(tax, 2),
            "tax": round(tax, 2),
            "rate_source": rate_info.get("source"),
        })

    # Top-level summary mirrors `generate_opiu_estonia` shape so the same
    # ServicesPnlOut Pydantic model accepts both legacy and dynamic outputs.
    cum_gross = baseline + sum(g for _, g in history)
    return {
        "total_gross": round(total_gross, 2),
        "total_tax_retained": round(total_tax, 2),
        "total_net_client": round(total_gross - total_tax, 2),
        "invoice_count": len(history),
        "saldo_inicial": 0.0,
        "total_enviado": 0.0,
        "debito_estonia": 0.0,
        "our_commission": round(total_tax, 2),
        "our_rental_paid_usd": 0.0,
        "our_rental_pending_usd": 0.0,
        "our_revenue_brl": round(total_tax, 2),
        "our_das_paid": 0.0,
        "our_das_estimated": 0.0,
        "our_das": 0.0,
        "total_company_das": 0.0,
        "total_trade_das": 0.0,
        "trade_das_rate": 0.0,
        "our_profit_brl": round(total_tax, 2),  # without DAS subtraction (separate concern)
        "our_profit_brl_paid_only": round(total_tax, 2),
        "das_payments": [],
        "das_pending": [],
        "pnl_by_month": pnl_by_month,
        "current_bracket": (pnl_by_month[-1].get("rate_source") if pnl_by_month else "none"),
        "next_bracket": "",
        "cumulative_gross": round(cum_gross, 2),
        "by_month": by_month,
    }


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
