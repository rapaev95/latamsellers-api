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


# Per-rbt12 commission step-function for Estonia (services project).
# Each tuple = (rbt12_ceiling_brl, commission_pct as fraction). Above the
# last ceiling — commission stays at the last value (5,61% cap confirmed
# by user). Derived from accountant's invoice rates with cumulative
# effective subtraction; see chat history Sprint 18 for derivation.
_ESTONIA_COMMISSION_TABLE: list[tuple[float, float]] = [
    (210_000,    0.1058),   # 0–210k
    (420_000,    0.0874),
    (630_000,    0.0776),
    (840_000,    0.0702),
    (1_100_000,  0.0682),
    (1_310_000,  0.0673),
    (1_520_000,  0.0669),
    (1_730_000,  0.0669),
    (1_940_000,  0.0644),
    (2_150_000,  0.0608),
    (2_360_000,  0.0581),
    (2_570_000,  0.0561),
]
_ESTONIA_COMMISSION_CAP = 0.0561


def commission_lookup_estonia(rbt12: float) -> float:
    """Step-function lookup — first ceiling >= rbt12 wins. Above cap → 5,61%.

    Returns commission as fraction (0.0682 = 6,82%). The accountant's table
    isn't a smooth formula, so a step-function with explicit ceilings is
    the only way to reproduce the exact rates they invoiced.
    """
    if rbt12 <= 0:
        return _ESTONIA_COMMISSION_TABLE[0][1]
    for ceiling, comm in _ESTONIA_COMMISSION_TABLE:
        if rbt12 <= ceiling:
            return comm
    return _ESTONIA_COMMISSION_CAP


def split_invoice_tax(gross_brl: float, rbt12_after: float, anexo: str = "III") -> dict[str, float]:
    """Compute DAS (Simples) + commission split for one invoice.

    `rbt12_after` is the cumulative RBT12 immediately AFTER this invoice
    settles (callers track cum_gross and pass it in). We apply both rates
    at that point — no bracket-split here because user's commission table
    is the same shape as a Simples faixa table and effective rate at
    rbt12_after already accounts for parcela_deduzir. Symmetric with how
    Receita Federal documents DAS: rate × gross at month of competência.

    Returns BRL amounts:
      • das        — for Receita Federal (Estonia pays this)
      • commission — our margin (Estonia keeps this)
      • tax_total  — what the client withholds at source (das + commission)
      • net        — what we forward to SHPP (gross − tax_total)
    """
    from v2.legacy.tax_brazil import compute_simples_effective
    eff_res = compute_simples_effective(rbt12_after, anexo)
    eff_frac = float(eff_res["effective_pct"]) / 100.0
    comm_frac = commission_lookup_estonia(rbt12_after)
    das = gross_brl * eff_frac
    commission = gross_brl * comm_frac
    tax_total = das + commission
    net = gross_brl - tax_total
    return {
        "das": round(das, 2),
        "commission": round(commission, 2),
        "tax_total": round(tax_total, 2),
        "net": round(net, 2),
        "eff_pct": round(eff_frac, 6),
        "commission_pct": round(comm_frac, 6),
        "total_rate_pct": round(eff_frac + comm_frac, 6),
        "rbt12": rbt12_after,
        "faixa": int(eff_res["faixa"]),
    }


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


def _rolling_rbt12(
    history: list[tuple[str, float]],
    current_iso: str,
    baseline: float = 0.0,
    baseline_as_of: Optional[str] = None,
) -> float:
    """Sliding 12-month sum: revenue from `history` whose date falls in
    [current − 12 months, current], inclusive of current_iso's month. Adds
    `baseline` (pre-loaded RBT12 from a period before local history starts —
    e.g. project migrating with paper-trail past).

    When `baseline_as_of` is set (ISO YYYY-MM-DD), the baseline ALSO obeys
    the 365-day decay: if `current_iso` is more than a year past
    `baseline_as_of`, baseline contributes 0 — modelling the legal Receita
    Federal definition where RBT12 is strictly the last 12 months. This is
    what makes GANZA's "12-month pause → fresh start" scenario work: by
    Jun 2027 (13+ months after baseline as_of=2026-04-30), the 676 699
    baseline drops out and only the still-in-window invoices count.

    `history` items are `(date_iso, gross_brl)` already accumulated by caller.
    """
    try:
        cur = datetime.strptime(current_iso[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return baseline

    # Window opens 12 months back, inclusive — i.e. for May 2026 it's
    # [Jun 2025, May 2026]. 365 days approximation works for MVP; exact
    # "same day previous year" would need calendar arithmetic.
    cutoff = date.fromordinal(max(1, cur.toordinal() - 365))

    # Baseline decay — only applies when baseline_as_of given (opt-in).
    base_contrib = baseline
    if baseline_as_of:
        try:
            ba = datetime.strptime(baseline_as_of[:10], "%Y-%m-%d").date()
            if (cur - ba).days > 365:
                base_contrib = 0.0
        except (ValueError, TypeError):
            pass  # malformed date → behave as if no decay (safer)

    s = base_contrib
    for ds, g in history:
        try:
            d = datetime.strptime(ds[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        # Strict-exclusive cutoff matches user's spreadsheet: Jun 2027 row
        # gives RBT12=630k = 3 invoices (Jul/Aug/Sep 2026), with the May/Jun
        # 2026 ones already out of the 365-day window. Current invoice isn't
        # in history yet (added after this fn) so the upper bound's
        # inclusiveness doesn't matter — kept open for clarity.
        if cutoff < d < cur:
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
    baseline_as_of = (formula or {}).get("baseline_rbt12_as_of") if formula else None

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

        # RBT12 by Receita Federal definition: sum of revenue of the 12 prior
        # months — STRICTLY EXCLUDING current month. Same convention for both
        # formula and brackets modes. User's 21-row scenario confirms it:
        #   - LATAMSELLERS Окт 2026 (RBT12=0, fresh): no past months ✓
        #   - GANZA Июн 2027 (RBT12=630k = Июл/Авг/Сен 2026 only):
        #     baseline already decayed (>365d after as_of), and 4-mo-prior
        #     invoice (Май 2026) just out of window ✓
        # Add current invoice to history AFTER computing rbt12.
        is_formula = bool(formula and formula.get("mode") == "simples_progressive")
        rbt12 = _rolling_rbt12(history, d_iso, baseline=baseline, baseline_as_of=baseline_as_of)
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
        "operating_expenses": {"total_brl": 0.0, "transactions": []},
        "_needs_config": True,
        "_reason": reason,
    }


def _load_approved_transfers(project_id: str) -> list[dict[str, Any]]:
    """Pull user-added approved transfers to the client for a services project.

    The hardcoded list inside generate_dds_estonia covers transfers up to the
    approved-report cutoff (19/03/2026). Anything after that — or any post-Phase-3
    backfill — is entered through ApprovedDataCard and lives on the project
    record under `approved_transfer`. We surface them here so the services DDS
    formula (saldo_inicial + invoices_net − total_outflows) reflects reality.

    Returns rows in the same shape as the hardcoded list:
      {date "DD/MM/YYYY", canal, usd|None, vet|None, brl, _user_added=True}.
    Currency normalisation:
      - BRL entries → brl = valor, usd/vet = None
      - USD/USDT entries → brl = valor × rate_brl, usd = valor, vet = rate_brl
    """
    try:
        from v2.legacy import config as legacy_config
        from v2.legacy.finance import _entry_valor_brl
        projects = legacy_config.load_projects()
    except Exception as err:  # noqa: BLE001
        log.warning("services_reports.approved_transfers: load failed for %s: %s", project_id, err)
        return []

    proj_meta = projects.get(project_id, {}) or {}
    raw = proj_meta.get("approved_transfer") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        try:
            brl = abs(_entry_valor_brl(item))
        except Exception:  # noqa: BLE001
            continue
        if brl <= 0:
            continue
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        # ApprovedDataCard stores ISO YYYY-MM-DD; the hardcoded list uses
        # DD/MM/YY. Convert so the UI table renders consistently.
        iso = str(item.get("date") or "")[:10]
        try:
            from datetime import datetime as _dt
            d = _dt.strptime(iso, "%Y-%m-%d").date()
            date_str = d.strftime("%d/%m/%y")
        except (ValueError, TypeError):
            date_str = iso
        out.append({
            "date": date_str,
            "canal": str(item.get("source") or item.get("note") or ""),
            "usd": float(item.get("valor") or 0) if cur in ("USD", "USDT") else None,
            "vet": float(item.get("rate_brl") or 0) if cur != "BRL" and item.get("rate_brl") else None,
            "brl": round(brl, 2),
            "_user_added": True,
        })
    return out


def make_invoice_key(numero: Any, date: Any, gross: Any) -> str:
    """Stable identifier for an invoice row in services-reports OPiU.

    NFS-e invoices always have a `numero` from the prefecture portal → use
    it verbatim (uniqueness guaranteed by source). Baseline hardcoded rows
    without numero → SHA1(`date|gross`) first 8 hex chars. Keys must stay
    stable across compute_for_user invocations so per-invoice overrides
    written by /edit-invoice-rate keep targeting the same row.
    """
    num = str(numero or "").strip()
    if num:
        return f"n:{num}"
    import hashlib
    d = str(date or "")[:10]
    g = f"{float(gross or 0):.2f}"
    h = hashlib.sha1(f"{d}|{g}".encode("utf-8")).hexdigest()[:8]
    return f"h:{h}"


def _load_invoice_rate_overrides(project_id: str) -> dict[str, dict[str, float]]:
    """Per-invoice rate overrides for a services project.

    Shape: `{invoice_key: {rate_eff?, rate_commission?}}`. Either field may
    be absent — partial overrides are valid (user edits DAS% only or
    Commission% only). Stored under `f2_services_invoice_rate_overrides_{project}`,
    same pattern as `_load_transfer_edits`. The UI triggers a fresh fetch
    after edit so the cache bypass is automatic.
    """
    try:
        from v2.legacy.db_storage import db_load
        raw = db_load(f"f2_services_invoice_rate_overrides_{project_id}")
    except Exception as err:  # noqa: BLE001
        log.warning("invoice_rate_overrides load failed for %s: %s", project_id, err)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        entry: dict[str, float] = {}
        for field in ("rate_eff", "rate_commission"):
            if field in v and v[field] is not None:
                try:
                    entry[field] = float(v[field])
                except (TypeError, ValueError):
                    continue
        if entry:
            out[k] = entry
    return out


def _load_operating_expenses(project_id: str) -> dict[str, Any]:
    """Pull manual operating expenses for a services project from projects_db.

    Services-cashflow generators (generate_dds_estonia) only handle invoice/transfer
    flows — they don't surface the company's running costs (ads, salaries, rent,
    accounting, software, etc.). Those live as `manual_expenses` on the project
    record, same shape that ecom compute_cashflow already consumes. We read them
    here so the services DDS tab can show them too.
    """
    try:
        from v2.legacy import config as legacy_config
        from v2.legacy.finance import _entry_valor_brl
        projects = legacy_config.load_projects()
    except Exception as err:  # noqa: BLE001
        log.warning("services_reports.operating_expenses: load failed for %s: %s", project_id, err)
        return {"total_brl": 0.0, "transactions": []}

    proj_meta = projects.get(project_id, {}) or {}
    raw = proj_meta.get("manual_expenses") or []
    txs: list[dict[str, Any]] = []
    total = 0.0
    for item in raw:
        try:
            v_brl = abs(_entry_valor_brl(item))
        except Exception:  # noqa: BLE001
            continue
        total += v_brl
        cur = str(item.get("currency", "BRL") or "BRL").upper()
        txs.append({
            "date": str(item.get("date") or "")[:10],
            "valor_brl": -v_brl,
            "category": item.get("category", "expense"),
            "note": item.get("note", ""),
            "currency": cur,
            "valor_orig": float(item.get("valor", 0) or 0),
            "rate": float(item.get("rate_brl", 0) or 0) if cur != "BRL" else None,
        })
    txs.sort(key=lambda r: r.get("date") or "", reverse=True)
    return {"total_brl": round(total, 2), "transactions": txs}


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


def generate_services_pnl_sync(
    project_id: str,
    *,
    loaded_nfs: list[dict[str, Any]] | None = None,
    rate_overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Sync — the legacy generator under the hood is fully sync. Caller from
    async context wraps in `asyncio.to_thread`.

    `loaded_nfs` lets compute_for_user pre-load NFS-e uploads from the DB
    (Railway disk is ephemeral, so `load_all_nfse()` alone finds nothing in
    prod). When None, legacy generator falls back to disk-only scan.

    `rate_overrides` — per-invoice user-edited DAS / commission rates from
    the project record. Shape: `{invoice_key: {rate_eff?, rate_commission?}}`.
    When None, falls back to the project-record load so legacy callers
    (Streamlit) still pick up overrides.
    """
    ok, reason = _validate_project(project_id)
    if not ok:
        return _empty_pnl(reason or "unknown")

    if _is_legacy_hardcoded(project_id):
        from v2.legacy.reports import generate_opiu_estonia
        if rate_overrides is None:
            rate_overrides = _load_invoice_rate_overrides(project_id)
        result = generate_opiu_estonia(
            loaded_nfs=loaded_nfs, rate_overrides=rate_overrides,
            project_id=project_id,
        )
        result["_needs_config"] = False
        return result

    # Project is type='services' but not Estonia — Phase 6 will read brackets +
    # tomador from project record. For now, return placeholder.
    return _empty_pnl(
        "not_yet_supported: only ESTONIA is wired to legacy generators; "
        "configure tomador_cnpj/commission_brackets and wait for Phase 6 refactor"
    )


def generate_services_cashflow_sync(
    project_id: str, *, include_hardcoded_outflows: bool = True,
) -> dict[str, Any]:
    ok, reason = _validate_project(project_id)
    if not ok:
        return _empty_cashflow(reason or "unknown")

    if _is_legacy_hardcoded(project_id):
        from v2.legacy.reports import generate_dds_estonia
        result = generate_dds_estonia(
            include_hardcoded_outflows=include_hardcoded_outflows,
            project_id=project_id,
        )
        result["_needs_config"] = False
        result["operating_expenses"] = _load_operating_expenses(project_id)

        # Append user-curated approved_transfers entered via ApprovedDataCard.
        # generate_dds_estonia's hardcoded list ends at 19/03/2026; anything
        # after that is appended here so debito_estonia drops as the user
        # records new client transfers.
        extra_transfers = _load_approved_transfers(project_id)
        if extra_transfers:
            existing = result.get("transfers") or []
            # Stamp continuation indices so the UI table can key rows uniquely.
            base = (existing[-1]["n"] if existing and existing[-1].get("n") is not None else 0)
            for i, tr in enumerate(extra_transfers, start=1):
                tr.setdefault("n", base + i)
            result["transfers"] = list(existing) + extra_transfers

            extra_brl = sum(float(t.get("brl") or 0) for t in extra_transfers)
            new_total = float(result.get("total_outflows") or 0) + extra_brl
            result["total_outflows"] = round(new_total, 2)

            # Recompute debito_estonia = saldo_inicial + invoices_net
            #                             + bank_inflows_total − total_outflows.
            inflows = result.get("inflows") or {}
            saldo = float(inflows.get("saldo_inicial") or 0)
            net = float(inflows.get("invoices_net") or 0)
            bank_in = float(inflows.get("bank_inflows_total") or 0)
            result["debito_estonia"] = round(saldo + net + bank_in - new_total, 2)

        return result

    # Non-Estonia services project — DDS skeleton is still empty, but manual
    # operating expenses (ads, rent, salaries, …) entered via the project UI
    # are real and should surface even before Phase 6.
    result = _empty_cashflow(
        "not_yet_supported: only ESTONIA is wired (Phase 6 generic refactor pending)"
    )
    result["operating_expenses"] = _load_operating_expenses(project_id)
    return result


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


async def load_nfse_uploads_for_user(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict[str, Any]]:
    """Read all NFS-e PDFs uploaded by `user_id` via the uploads table.

    Production path for Railway (disk-based sidecars don't survive deploys).
    Prefers `parsed_meta` (filled at upload time by the `nfse_shps` branch in
    /finance/uploads). Falls back to parsing `file_bytes` on the fly so older
    uploads predating the parser still surface in OPiU.

    Dedup by `numero` (matches `load_all_nfse()` behaviour for legacy disk
    sidecars). Rows without a numero are kept separately so they still appear
    in the invoice list.
    """
    if pool is None:
        return []
    rows = await pool.fetch(
        """
        SELECT id, filename, file_bytes, parsed_meta
          FROM uploads
         WHERE user_id = $1 AND source_key = 'nfse_shps'
           AND file_bytes IS NOT NULL
        """,
        user_id,
    )
    if not rows:
        return []

    from v2.legacy.reports import parse_nfse_pdf_bytes

    by_numero: dict[str, dict[str, Any]] = {}
    no_numero: list[dict[str, Any]] = []
    for r in rows:
        rec: Optional[dict[str, Any]] = None
        meta = r["parsed_meta"]
        if isinstance(meta, dict) and meta.get("valor"):
            rec = dict(meta)
        elif isinstance(meta, str) and meta.strip():
            try:
                import json as _json
                parsed = _json.loads(meta)
                if isinstance(parsed, dict) and parsed.get("valor"):
                    rec = parsed
            except Exception:  # noqa: BLE001
                rec = None
        if rec is None:
            # Fallback — parse bytes inline so we still surface the invoice.
            rec = parse_nfse_pdf_bytes(bytes(r["file_bytes"]), r["filename"])
        if not rec:
            continue
        num = str(rec.get("numero") or "").strip()
        if num:
            by_numero[num] = rec
        else:
            no_numero.append(rec)
    return list(by_numero.values()) + no_numero


async def compute_for_user(
    pool: asyncpg.Pool,
    user_id: int,
    project_id: str,
    period_from: Optional[date] = None,
    period_to: Optional[date] = None,
    *,
    include_hardcoded_outflows: bool = True,
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

    # Pre-load NFS-e uploads from DB (production path) and merge with disk
    # sidecars before handing off to the sync PnL generator. Disk path stays
    # for local dev; DB path is the only one that works on Railway. Dedup by
    # numero so re-uploading the same invoice doesn't double-count.
    try:
        db_nfs = await load_nfse_uploads_for_user(pool, user_id)
    except Exception as err:  # noqa: BLE001
        log.warning("services_reports.load_nfse_uploads_for_user failed for user=%s: %s", user_id, err)
        db_nfs = []
    try:
        from v2.legacy.reports import load_all_nfse
        disk_nfs = await asyncio.to_thread(load_all_nfse)
    except Exception as err:  # noqa: BLE001
        log.warning("services_reports.load_all_nfse (disk) failed: %s", err)
        disk_nfs = []
    merged_by_num: dict[str, dict[str, Any]] = {}
    merged_no_num: list[dict[str, Any]] = []
    for rec in list(disk_nfs) + list(db_nfs):
        num = str((rec or {}).get("numero") or "").strip()
        if num:
            merged_by_num[num] = rec  # DB wins on collision (last-write)
        else:
            merged_no_num.append(rec)
    merged_nfs = list(merged_by_num.values()) + merged_no_num

    # Per-invoice rate overrides (user edits in OPiU). Loaded sync since
    # legacy_config.load_projects() is sync — cheap (single read of
    # f2_projects user_data row) and keeps the signature clean.
    rate_overrides = _load_invoice_rate_overrides(project_id)
    # Diagnostic — surface effective_user_id + loaded override map in the
    # response so we can verify save/load is on the same user namespace.
    try:
        from v2.legacy.db_storage import _current_user_id as _legacy_uid  # noqa: SLF001
        _diag_uid = _legacy_uid()
    except Exception:  # noqa: BLE001
        _diag_uid = None
    out["diag_overrides"] = {
        "user_id_bound": _diag_uid,
        "user_id_arg": user_id,
        "rate_overrides_loaded": rate_overrides,
    }

    pnl_task = asyncio.create_task(asyncio.to_thread(
        generate_services_pnl_sync, project_id,
        loaded_nfs=merged_nfs, rate_overrides=rate_overrides,
    ))
    cf_task = asyncio.create_task(asyncio.to_thread(
        generate_services_cashflow_sync, project_id,
        include_hardcoded_outflows=include_hardcoded_outflows,
    ))
    bal_task = asyncio.create_task(asyncio.to_thread(generate_services_balance_sync, project_id))

    try:
        out["pnl"] = await pnl_task
    except Exception as err:  # noqa: BLE001
        out["pnl_error"] = f"{type(err).__name__}: {err}"
    try:
        out["cashflow"] = await cf_task
    except Exception as err:  # noqa: BLE001
        out["cashflow_error"] = f"{type(err).__name__}: {err}"
    else:
        # Auto-import TrafficStars USD débitos from C6 USD as approved transfers.
        # We pay TrafficStars on the client's behalf, so every TS debit on the
        # services-project C6 USD account is effectively a transfer to the
        # client. BRL cost comes from the cambio FIFO engine (real BRL spent
        # to acquire the consumed USD lots), not the spot rate at the time of
        # the debit. Falls back silently if uploads / FIFO fail.
        if _is_legacy_hardcoded(project_id) and isinstance(out.get("cashflow"), dict):
            try:
                ts_transfers = await _build_trafficstars_auto_transfers(pool, user_id)
            except Exception as err:  # noqa: BLE001
                log.warning("services_reports.ts_auto failed for %s: %s", project_id, err)
                ts_transfers = []
            if ts_transfers:
                cf = out["cashflow"]
                existing = cf.get("transfers") or []
                cf["transfers"] = list(existing) + ts_transfers
                extra_brl = sum(float(t.get("brl") or 0) for t in ts_transfers)
                cf["total_outflows"] = round(float(cf.get("total_outflows") or 0) + extra_brl, 2)
                inflows = cf.get("inflows") or {}
                saldo = float(inflows.get("saldo_inicial") or 0)
                net = float(inflows.get("invoices_net") or 0)
                bank_in = float(inflows.get("bank_inflows_total") or 0)
                cf["debito_estonia"] = round(saldo + net + bank_in - cf["total_outflows"], 2)
        # Sort the full transfers list chronologically and renumber. Without
        # this the table renders hardcoded rows first (in their author's order),
        # then ApprovedDataCard rows, then auto-TS rows — which jumbles dates
        # (e.g. 18/03/26 hardcoded → 20/01/26 hardcoded → 02/05/26 auto-TS).
        # Done once at the end so it covers every source uniformly.
        if isinstance(out.get("cashflow"), dict):
            cf = out["cashflow"]
            transfers = cf.get("transfers") or []
            # Per-row "❌ Скрыть": filter out rows the user has dismissed.
            # Surface them through `hidden_transfers` so the UI can list and
            # un-hide them later; they don't contribute to total_outflows or
            # debito_estonia.
            hidden_keys = _load_hidden_transfer_keys(project_id)
            edits = _load_transfer_edits(project_id)
            visible: list[dict[str, Any]] = []
            hidden_rows: list[dict[str, Any]] = []
            for tr in transfers:
                orig_key = transfer_hide_key(tr)
                # Apply user edit if any — original_key is the pre-edit hash,
                # patch fields overwrite tr in place. _edit_key keeps the
                # original_key so the UI can look up + clear the patch later.
                patch = edits.get(orig_key)
                if isinstance(patch, dict):
                    for f in ("date", "canal", "brl", "usd", "vet"):
                        if f in patch:
                            tr[f] = patch[f]
                    tr["_edited"] = True
                    tr["_orig_key"] = orig_key
                if orig_key in hidden_keys:
                    hidden_rows.append({**tr, "_hide_key": orig_key})
                else:
                    visible.append(tr)
            visible.sort(key=_transfer_sort_key)
            for i, tr in enumerate(visible, start=1):
                tr["n"] = i
                # Hide-key uses CURRENT (post-edit) values so the ✕ button
                # targets what the user actually sees. _orig_key (above)
                # remains stable for un-edit.
                tr["_hide_key"] = transfer_hide_key(tr)
            cf["transfers"] = visible
            cf["hidden_transfers"] = hidden_rows
            if hidden_rows:
                cf["total_outflows"] = round(sum(float(t.get("brl") or 0) for t in visible), 2)
                inflows = cf.get("inflows") or {}
                saldo = float(inflows.get("saldo_inicial") or 0)
                net = float(inflows.get("invoices_net") or 0)
                bank_in = float(inflows.get("bank_inflows_total") or 0)
                cf["debito_estonia"] = round(saldo + net + bank_in - cf["total_outflows"], 2)
    try:
        out["balance"] = await bal_task
    except Exception as err:  # noqa: BLE001
        out["balance_error"] = f"{type(err).__name__}: {err}"

    return out


def transfer_hide_key(tr: dict[str, Any]) -> str:
    """Stable identifier for a single transfer row, used by the per-row
    "❌ Скрыть" feature. Same date+canal+brl always hashes to the same key
    regardless of source (hardcoded / approved / auto-TS), so hiding from the
    UI persists across recomputes.
    """
    date_s = str(tr.get("date") or "")
    canal_s = str(tr.get("canal") or "").strip()
    try:
        brl = round(float(tr.get("brl") or 0), 2)
    except (TypeError, ValueError):
        brl = 0.0
    return f"{date_s}|{canal_s}|{brl:.2f}"


def _load_hidden_transfer_keys(project_id: str) -> set[str]:
    """Per-project list of transfer hide-keys the user dismissed via the
    "❌ Скрыть" button. Stored as JSONB list under
    `f2_services_hidden_transfers_{project_id}`.
    """
    try:
        from v2.legacy.db_storage import db_load
        raw = db_load(f"f2_services_hidden_transfers_{project_id}")
    except Exception as err:  # noqa: BLE001
        log.warning("hidden_transfers load failed for %s: %s", project_id, err)
        return set()
    if not isinstance(raw, list):
        return set()
    return {str(x) for x in raw if isinstance(x, str) and x}


def _load_transfer_edits(project_id: str) -> dict[str, dict[str, Any]]:
    """Per-project map of {original_hide_key: patch} applied to transfers
    after collection. Patch fields override the original row: any of
    `date`, `canal`, `brl`, `usd`, `vet`. Stored as JSONB under
    `f2_services_transfer_edits_{project_id}`.
    """
    try:
        from v2.legacy.db_storage import db_load
        raw = db_load(f"f2_services_transfer_edits_{project_id}")
    except Exception as err:  # noqa: BLE001
        log.warning("transfer_edits load failed for %s: %s", project_id, err)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def _transfer_sort_key(tr: dict[str, Any]) -> date:
    """Order transfers chronologically regardless of source.

    Hardcoded list uses DD/MM/YY ("22/09/25"), approved_transfer uses the
    same format after normalisation in _load_approved_transfers, auto-TS
    uses it too. A few edge cases need DD/MM/YYYY. Anything unparseable
    drops to date.min so it sits at the top of the table where it's easy
    to spot.
    """
    raw = str(tr.get("date") or "")
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except (ValueError, TypeError):
            continue
    return date.min


async def _build_trafficstars_auto_transfers(
    pool: asyncpg.Pool, user_id: int,
) -> list[dict[str, Any]]:
    """Auto-map C6 USD TrafficStars débitos into approved-transfer rows.

    For services projects (Estonia / GANZA-USD) we settle TrafficStars from
    the client's USD float on C6. Each TS debit is therefore money that left
    the float on the client's behalf — exactly the semantic of an approved
    transfer. Convert with BRL cost taken from the cambio FIFO (real BRL
    spent for the consumed USD lots), not from a synthetic spot rate.

    Filter rule: description (case-insensitive) contains "trafficstars". Other
    USD debits (Saque, Pagamento with another counterparty, …) stay outside —
    user can still add them manually via ApprovedDataCard.

    Skips débitos that came back uncovered (no FIFO lots to consume) — we'd
    have no BRL value to add to debito_estonia for them and silently writing
    R$ 0 would understate the debit.

    Returns rows in the same shape as the hardcoded list:
      {date "DD/MM/YY", canal, usd, vet (rate), brl, _auto_imported=True,
       _auto_source="trafficstars"}
    """
    from v2.legacy.bank_tx import parse_bank_tx_bytes
    from v2.services import cambio
    from v2.storage import uploads_storage

    try:
        files = await uploads_storage.fetch_files_by_source(pool, user_id, "extrato_c6_usd")
    except Exception as err:  # noqa: BLE001
        log.warning("ts_auto: fetch_files_by_source failed: %s", err)
        return []
    if not files:
        return []

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in files:
        try:
            parsed = parse_bank_tx_bytes("extrato_c6_usd", f.file_bytes)
        except Exception as err:  # noqa: BLE001
            log.warning("ts_auto: parse failed for upload %s: %s", getattr(f, "id", "?"), err)
            continue
        for r in parsed:
            h = r.get("tx_hash") or ""
            if not h or h in seen:
                continue
            seen.add(h)
            rows.append(r)
    if not rows:
        return []

    # Second-pass dedup by (date, usd amount) — covers the case where two
    # uploaded PDFs cover overlapping periods AND the parser captured the
    # same debit under slightly different descriptions (so tx_hash dedup
    # above doesn't catch them). Keeps the earliest-seen row, drops repeats.
    by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for r in rows:
        val = float(r.get("value_brl") or 0)
        if val >= 0:
            by_key[(str(r.get("date") or ""), val, r.get("tx_hash") or "")] = r  # entradas keyed by hash so they all stay
            continue
        k = (str(r.get("date") or "")[:10], round(abs(val), 2))
        # Skip if a debit with the same (date, |usd|) already exists. Two
        # legitimately separate debits on the same day for the exact same
        # amount are rare enough to call out via a manual ApprovedDataCard
        # entry if they ever happen.
        if k not in by_key:
            by_key[k] = r
    rows = list(by_key.values())

    try:
        result = await cambio.compute_for_user(pool, user_id, rows)
        cambio.enrich_rows(rows, result)
    except Exception as err:  # noqa: BLE001
        log.warning("ts_auto: cambio compute_for_user failed: %s", err)
        return []

    # Fallback rate for débitos that FIFO couldn't fully cover (no USD lots
    # left at the time). Without this we'd either skip them — hiding real
    # spend from debito_estonia — or write R$ 0 and silently misstate the
    # debit. Using the period's average câmbio rate gives a reasonable
    # estimate; rows that fall back are flagged _brl_estimated so the UI can
    # warn the user to backfill manual USD inflows (Bybit / Nubank / etc.).
    summary = result.summary if hasattr(result, "summary") else {}
    avg_rate = float(summary.get("avg_rate") or 0)

    out_transfers: list[dict[str, Any]] = []
    for r in rows:
        val = float(r.get("value_brl") or 0)
        if val >= 0:
            continue  # entradas only contribute to FIFO inventory, not transfers
        haystack = (str(r.get("description") or "") + " " + str(r.get("title") or "")).lower()
        if "trafficstars" not in haystack:
            continue
        usd_abs = abs(val)
        brl_cost = float(r.get("fifo_brl_cost") or 0)
        uncovered = float(r.get("fifo_uncovered_usd") or 0)
        estimated = False
        if uncovered > 0 and avg_rate > 0:
            # Cover the missing portion at avg rate; flag the row.
            brl_cost = round(brl_cost + uncovered * avg_rate, 2)
            estimated = True
        if brl_cost <= 0:
            # No FIFO data AND no avg rate to fall back on — can't represent
            # this debit in BRL at all. Skip so we don't write R$ 0 into the
            # transfers table.
            continue
        date_iso = str(r.get("date") or "")[:10]
        try:
            from datetime import datetime as _dt
            d = _dt.strptime(date_iso, "%Y-%m-%d").date()
            date_str = d.strftime("%d/%m/%y")
        except (ValueError, TypeError):
            date_str = date_iso
        rate = round(brl_cost / usd_abs, 4) if usd_abs > 0 else None
        out_transfers.append({
            "date": date_str,
            "canal": "C6 TrafficStars (auto)" + (" ~est" if estimated else ""),
            "usd": round(usd_abs, 2),
            "vet": rate,
            "brl": round(brl_cost, 2),
            "_auto_imported": True,
            "_auto_source": "trafficstars",
            "_brl_estimated": estimated,
        })
    out_transfers.sort(key=lambda r: r["date"])
    return out_transfers
