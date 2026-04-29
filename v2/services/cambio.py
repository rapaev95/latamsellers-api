"""C6 BRL ↔ C6 USD câmbio reconciliation + FIFO inventory.

Why:
  C6 records one câmbio (BRL→USD) operation as TWO rows in two separate
  statements — `-R$ X "Câmbio C6 Conta Global"` in C6 BRL and
  `+US$ Y "Entrada Transf C6 Conta Global Líquido"` in C6 USD. The bank does
  not link them with a tx-id, so we have to pair them ourselves: same date
  ±1 day + brl/usd ratio in a sane FX range.

  Once paired, each pair becomes a FIFO "lot" of USD with a known BRL cost.
  Subsequent USD outflows (debits, pending charges, PIX out, etc.) consume
  the oldest lots first — same logic Streamlit uses in
  `_admin/reports.py:calculate_trafficstars_fifo`. The result is the *real*
  BRL cost of every USD spend, not just the average rate at the time.

Output (consumed by /bank-transactions/grouped for `extrato_c6_usd`):
  • CambioSummary — avg rate, totals, lots remaining, period.
  • per-row enrichment — `cambio_rate` for entradas, `fifo_brl_cost` for saídas.

Phase 3 will add `manual_inflows` (Bybit/CALIZA-Nubank/etc.) on top of this —
the same FIFO engine, just more sources feeding the inventory.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date as date_type, datetime, timedelta
from typing import Any, Optional

import asyncpg

from v2.legacy.bank_tx import parse_bank_tx_bytes
from v2.legacy.db_storage import db_load
from v2.storage import uploads_storage

log = logging.getLogger(__name__)


# Pairing tolerances — derived from real C6 user data.
# - Date: BRL debit and USD credit are usually same-day, sometimes off by one
#   when the conversion settles after midnight. ±1 day covers all observed cases.
# - Rate range: BR market BRL/USD has stayed in [4.5, 6.5] for years. We extend
#   to [3.5, 8.0] so the pairing isn't brittle to extreme moves and so a typo
#   (BRL row matched with the wrong USD row) shows as out-of-range and gets
#   rejected instead of silently producing a nonsense rate.
_DATE_TOLERANCE_DAYS = 1
_RATE_MIN = 3.5
_RATE_MAX = 8.0

_FX_CATEGORY = "fx"  # `_admin/config.py:bank_transaction_rules` auto-tags Câmbio rows with this


@dataclass
class CambioPair:
    """One BRL→USD conversion event, paired across two statements."""
    date_brl: str         # ISO YYYY-MM-DD
    date_usd: str
    brl_paid: float       # absolute value (always positive)
    usd_received: float   # absolute value
    rate: float           # brl_paid / usd_received
    source: str = "c6_pair"   # phase 3 will add "manual"
    note: str = ""
    # Hashes of the two source rows — lets the UI link a pair back to its rows.
    brl_tx_hash: str = ""
    usd_tx_hash: str = ""


@dataclass
class FifoLot:
    """A USD inventory entry. Pushed by Entrada / paired câmbio, consumed FIFO."""
    date: str             # ISO
    usd_remaining: float  # mutates as saídas consume it
    rate: float
    usd_initial: float    # for reporting
    brl_initial: float    # for reporting
    source: str           # "c6_pair" | "manual"
    note: str = ""


@dataclass
class FifoConsumption:
    """Records how a single saída was satisfied across one or more lots.
    Stored per-row so the UI can show «-US$ 6.185,88 ≈ R$ 32.611 (FIFO)»."""
    saida_tx_hash: str
    saida_date: str
    saida_usd: float       # absolute value
    brl_cost: float        # FIFO-summed BRL cost
    lots_used: list[dict[str, Any]] = field(default_factory=list)
    # When set: saída exhausted the inventory; partial cost computed from lots
    # available, the rest left uncovered (e.g. user has C6 USD spends but no
    # paired câmbio loaded yet). UI shows this with a warning hint.
    uncovered_usd: float = 0.0


@dataclass
class CambioResult:
    pairs: list[CambioPair]
    fifo_lots_remaining: list[FifoLot]
    fifo_consumptions_by_hash: dict[str, FifoConsumption]
    summary: dict[str, Any]


# ── Loading effective fx-debits from C6 BRL ───────────────────────────────────

async def _load_c6_brl_fx_debits(pool: asyncpg.Pool, user_id: int) -> list[dict[str, Any]]:
    """Pull every BRL outflow that's tagged (auto-rule or manual override) as FX.

    The category may come from two places:
      1. `bank_tx.classify_transaction` running the user's rules JSON during parse
         (auto rules include "câmbio", "c6 conta global" → category=fx).
      2. Per-source override in user_data (key `f2_grouped_overrides_extrato_c6_brl`)
         — user manually re-labeled a row in Classification page.

    Override wins over auto. We compute *effective* category and filter by `fx`.
    """
    files = await uploads_storage.fetch_files_by_source(pool, user_id, "extrato_c6_brl")
    if not files:
        return []

    overrides = db_load("f2_grouped_overrides_extrato_c6_brl") or {}
    if not isinstance(overrides, dict):
        overrides = {}

    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for f in files:
        try:
            rows = parse_bank_tx_bytes("extrato_c6_brl", f.file_bytes)
        except Exception as err:  # noqa: BLE001
            log.warning("cambio: parse_bank_tx_bytes failed for upload %s: %s", f.id, err)
            continue
        for r in rows:
            h = r.get("tx_hash") or ""
            if not h or h in seen_hashes:
                continue
            seen_hashes.add(h)
            # Effective category: override beats parser-assigned.
            ov = overrides.get(h)
            eff_cat = (ov or {}).get("category") if isinstance(ov, dict) else None
            eff_cat = eff_cat or r.get("category")
            if eff_cat != _FX_CATEGORY:
                continue
            val = float(r.get("value_brl") or 0)
            if val >= 0:
                # FX = OUTflow only. Positive value here would be a refund/return
                # of a câmbio, not a câmbio itself — skip.
                continue
            out.append({
                "tx_hash": h,
                "date": (r.get("date") or "")[:10],
                "brl": abs(val),
                "description": r.get("description") or "",
            })
    # Date asc — pairing walks both lists in chronological order.
    out.sort(key=lambda r: r["date"])
    return out


def _load_c6_usd_entradas(c6_usd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull every Entrada-type credit from already-parsed C6 USD rows.

    Caller passes the rows it already parsed for the grouped view, so we don't
    re-parse the PDFs. Recognises an Entrada by sign (>0) — the parser tags
    every USD-row with currency=USD, and saídas are negative.
    """
    out: list[dict[str, Any]] = []
    for r in c6_usd_rows:
        val = float(r.get("value_brl") or 0)
        if val <= 0:
            continue
        out.append({
            "tx_hash": r.get("tx_hash") or "",
            "date": (r.get("date") or "")[:10],
            "usd": val,
            "description": r.get("description") or "",
        })
    out.sort(key=lambda r: r["date"])
    return out


# ── Pairing ──────────────────────────────────────────────────────────────────

def _date_diff_days(a: str, b: str) -> int:
    """Absolute day-diff between two ISO dates. Bad input → large number to fail tolerance."""
    try:
        da = datetime.strptime(a, "%Y-%m-%d").date()
        db = datetime.strptime(b, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 10**9
    return abs((da - db).days)


def pair_brl_usd(
    brl_rows: list[dict[str, Any]],
    usd_rows: list[dict[str, Any]],
    *,
    date_tolerance_days: int = _DATE_TOLERANCE_DAYS,
    rate_min: float = _RATE_MIN,
    rate_max: float = _RATE_MAX,
) -> tuple[list[CambioPair], list[dict], list[dict]]:
    """Greedy chronological pairing.

    Walks BRL rows oldest-first; for each BRL row finds the first unmatched
    USD row within `date_tolerance_days` whose implied rate is in the sane
    band. Marks both consumed.

    Returns (pairs, unmatched_brl, unmatched_usd) — the unmatched lists let
    the UI show "câmbio without USD-side / vice versa" warnings.
    """
    pairs: list[CambioPair] = []
    used_usd_hashes: set[str] = set()

    for brl in brl_rows:
        match: Optional[dict[str, Any]] = None
        for usd in usd_rows:
            if usd["tx_hash"] in used_usd_hashes:
                continue
            if _date_diff_days(brl["date"], usd["date"]) > date_tolerance_days:
                continue
            if usd["usd"] <= 0:
                continue
            implied_rate = brl["brl"] / usd["usd"]
            if implied_rate < rate_min or implied_rate > rate_max:
                continue
            match = usd
            break
        if match is None:
            continue
        used_usd_hashes.add(match["tx_hash"])
        pairs.append(CambioPair(
            date_brl=brl["date"],
            date_usd=match["date"],
            brl_paid=brl["brl"],
            usd_received=match["usd"],
            rate=brl["brl"] / match["usd"],
            source="c6_pair",
            note="C6 BRL→USD",
            brl_tx_hash=brl["tx_hash"],
            usd_tx_hash=match["tx_hash"],
        ))

    matched_brl_hashes = {p.brl_tx_hash for p in pairs}
    matched_usd_hashes = {p.usd_tx_hash for p in pairs}
    unmatched_brl = [r for r in brl_rows if r["tx_hash"] not in matched_brl_hashes]
    unmatched_usd = [r for r in usd_rows if r["tx_hash"] not in matched_usd_hashes]
    return pairs, unmatched_brl, unmatched_usd


# ── FIFO inventory engine ────────────────────────────────────────────────────

def apply_fifo(
    pairs: list[CambioPair],
    saidas: list[dict[str, Any]],
) -> tuple[list[FifoLot], dict[str, FifoConsumption]]:
    """Push pairs into a FIFO queue, consume saídas oldest-first.

    `saidas` is a list of `{tx_hash, date, usd}` where usd > 0 (caller already
    abs()'d — sign convention is "positive amount, FIFO will subtract").

    Returns (lots_remaining, consumptions_by_saida_hash). `lots_remaining` is
    snapshot of inventory after all saídas are processed — non-zero means
    there's USD inventory still on hand (good).
    """
    inventory: deque[FifoLot] = deque()
    for p in pairs:
        inventory.append(FifoLot(
            date=p.date_usd,
            usd_remaining=p.usd_received,
            rate=p.rate,
            usd_initial=p.usd_received,
            brl_initial=p.brl_paid,
            source=p.source,
            note=p.note,
        ))

    # Sort saídas chronologically — the queue must be consumed in arrival order
    # of the *outflows*, not whatever order the caller passed.
    saidas_sorted = sorted(saidas, key=lambda s: s["date"])
    consumptions: dict[str, FifoConsumption] = {}

    for s in saidas_sorted:
        remaining = float(s["usd"])
        brl_cost = 0.0
        used_lots: list[dict[str, Any]] = []
        while remaining > 0 and inventory:
            lot = inventory[0]
            take = min(remaining, lot.usd_remaining)
            cost = take * lot.rate
            brl_cost += cost
            used_lots.append({
                "lot_date": lot.date,
                "rate": round(lot.rate, 4),
                "usd_taken": round(take, 2),
                "brl_cost_chunk": round(cost, 2),
                "source": lot.source,
            })
            lot.usd_remaining -= take
            remaining -= take
            if lot.usd_remaining < 1e-6:
                inventory.popleft()
        consumptions[s["tx_hash"]] = FifoConsumption(
            saida_tx_hash=s["tx_hash"],
            saida_date=s["date"],
            saida_usd=float(s["usd"]),
            brl_cost=round(brl_cost, 2),
            lots_used=used_lots,
            uncovered_usd=round(remaining, 2) if remaining > 0 else 0.0,
        )

    return list(inventory), consumptions


# ── Summary ─────────────────────────────────────────────────────────────────

def build_summary(
    pairs: list[CambioPair],
    lots_remaining: list[FifoLot],
    consumptions: dict[str, FifoConsumption],
    unmatched_brl: list[dict[str, Any]],
    unmatched_usd: list[dict[str, Any]],
) -> dict[str, Any]:
    total_brl = round(sum(p.brl_paid for p in pairs), 2)
    total_usd = round(sum(p.usd_received for p in pairs), 2)
    avg_rate = round(total_brl / total_usd, 4) if total_usd > 0 else None

    usd_inventory = round(sum(l.usd_remaining for l in lots_remaining), 2)
    # Inventory's BRL value is the *original* cost of the USD that's still on hand,
    # weighted by what fraction of each lot is left.
    brl_inventory_value = round(
        sum(l.usd_remaining * l.rate for l in lots_remaining), 2,
    )

    total_saida_usd = round(sum(c.saida_usd for c in consumptions.values()), 2)
    total_saida_brl_fifo = round(sum(c.brl_cost for c in consumptions.values()), 2)
    uncovered_usd = round(sum(c.uncovered_usd for c in consumptions.values()), 2)

    period_start = min((p.date_brl for p in pairs), default=None)
    period_end = max((p.date_usd for p in pairs), default=None)

    return {
        "pairs_count": len(pairs),
        "avg_rate": avg_rate,
        "total_brl_converted": total_brl,
        "total_usd_received": total_usd,
        "usd_inventory_remaining": usd_inventory,
        "brl_inventory_value": brl_inventory_value,
        "total_saida_usd": total_saida_usd,
        "total_saida_brl_fifo": total_saida_brl_fifo,
        "uncovered_saida_usd": uncovered_usd,
        "period_start": period_start,
        "period_end": period_end,
        "unmatched_brl_count": len(unmatched_brl),
        "unmatched_usd_count": len(unmatched_usd),
        # Compact preview for UI tooltip — first 5 each
        "pairs_preview": [
            {
                "date": p.date_brl, "brl": p.brl_paid, "usd": p.usd_received,
                "rate": round(p.rate, 4), "source": p.source,
            }
            for p in pairs[:5]
        ],
    }


# ── Public entry-point ──────────────────────────────────────────────────────

async def compute_for_user(
    pool: asyncpg.Pool,
    user_id: int,
    c6_usd_rows: list[dict[str, Any]],
) -> CambioResult:
    """Build the full câmbio picture for one user.

    Caller passes the C6 USD rows it already parsed (they're already in the
    grouped-view response); we re-load only the C6 BRL side here. Phase 3 will
    extend this to also pull manual external USD inflows from a new table and
    feed them into the same `pairs` list.
    """
    brl_fx = await _load_c6_brl_fx_debits(pool, user_id)
    usd_entradas = _load_c6_usd_entradas(c6_usd_rows)
    pairs, unm_brl, unm_usd = pair_brl_usd(brl_fx, usd_entradas)

    # Saídas = absolute value of every C6 USD row with value < 0.
    saidas = [
        {
            "tx_hash": r.get("tx_hash") or "",
            "date": (r.get("date") or "")[:10],
            "usd": abs(float(r.get("value_brl") or 0)),
        }
        for r in c6_usd_rows
        if float(r.get("value_brl") or 0) < 0
    ]
    lots_remaining, consumptions = apply_fifo(pairs, saidas)
    summary = build_summary(pairs, lots_remaining, consumptions, unm_brl, unm_usd)

    return CambioResult(
        pairs=pairs,
        fifo_lots_remaining=lots_remaining,
        fifo_consumptions_by_hash=consumptions,
        summary=summary,
    )


def enrich_rows(rows: list[dict[str, Any]], result: CambioResult) -> None:
    """Mutate `rows` in place — add cambio fields where applicable.

    For Entrada-rows (value > 0): `cambio_rate`, `cambio_brl_paid`, `cambio_source`.
    For Saída-rows (value < 0): `fifo_brl_cost`, `fifo_uncovered_usd`,
    `fifo_lots_used` (compact for tooltip).
    """
    pair_by_usd_hash = {p.usd_tx_hash: p for p in result.pairs}
    for r in rows:
        h = r.get("tx_hash") or ""
        if not h:
            continue
        val = float(r.get("value_brl") or 0)
        if val > 0:
            p = pair_by_usd_hash.get(h)
            if p is not None:
                r["cambio_rate"] = round(p.rate, 4)
                r["cambio_brl_paid"] = round(p.brl_paid, 2)
                r["cambio_source"] = p.source
        elif val < 0:
            c = result.fifo_consumptions_by_hash.get(h)
            if c is not None:
                r["fifo_brl_cost"] = c.brl_cost
                if c.uncovered_usd > 0:
                    r["fifo_uncovered_usd"] = c.uncovered_usd
                # Cap lots in payload — UI tooltip needs at most a few
                if c.lots_used:
                    r["fifo_lots_used"] = c.lots_used[:5]
