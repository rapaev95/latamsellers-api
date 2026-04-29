"""DB-backed bank-classification aggregator for finance reports.

Why this exists
---------------
`v2/legacy/reports.aggregate_classified_by_project` reads classifications
from local JSON files at `_data/{month}/{src}_classifications.json`. That
worked in the Streamlit world where the user manually edited those files
on disk, but in production (Railway) the FS is ephemeral and the new UI
saves classifications into the database (`f2_classifications_grouped_*`
per-bank, `f2_classifications_*` per-upload), so the disk path returns
nothing → bank txns silently miss from Cashflow / Balance.

The router `/finance/reports` (async) prefetches every bank statement
through `parse_bank_tx_bytes`, applies user overrides from the DB, and
stuffs the merged list into a `ContextVar`. The legacy synchronous
compute_* functions can read it without changing their thread-pool
plumbing, since `contextvars.copy_context().run()` carries the context
into worker threads.

Falls through to the legacy disk path when the contextvar is unset, so
local development with manual JSON edits keeps working.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any, Optional

import asyncpg

log = logging.getLogger(__name__)

_BANK_SOURCES = ("extrato_nubank", "extrato_c6_brl", "extrato_c6_usd", "extrato_mp")

# Per-request prefetched bank txs, set by the router before _run_parallel_with_timeout
# fans out to compute_pnl/cashflow/balance. Each tx is a dict with keys
# matching `parse_bank_tx_bytes` output (date, value_brl, description,
# category, project, label, tx_class, tx_hash, ...) plus `_source_key`
# and `_upload_id` for downstream provenance.
_classified_txs_var: contextvars.ContextVar[Optional[list[dict[str, Any]]]] = (
    contextvars.ContextVar("classified_bank_txs", default=None)
)


def set_prefetched(txs: Optional[list[dict[str, Any]]]) -> None:
    _classified_txs_var.set(txs)


def get_prefetched() -> Optional[list[dict[str, Any]]]:
    return _classified_txs_var.get()


def clear_prefetched() -> None:
    _classified_txs_var.set(None)


# ── Async pre-fetch (called from /finance/reports router) ──────────────────


async def prefetch_for_user(pool: asyncpg.Pool, user_id: int) -> list[dict[str, Any]]:
    """Parse every bank statement uploaded by `user_id`, apply rules + saved
    overrides, dedupe by `tx_hash`, return one merged list ready for
    aggregation by project.
    """
    from v2.legacy.bank_tx import parse_bank_tx_bytes
    from v2.legacy.db_storage import db_load
    from v2.storage import uploads_storage

    merged: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for src in _BANK_SOURCES:
        try:
            files = await uploads_storage.fetch_files_by_source(pool, user_id, src)
        except Exception as err:  # noqa: BLE001
            log.warning("prefetch fetch_files_by_source(%s) failed: %s", src, err)
            continue
        if not files:
            continue

        # Per-bank overrides keyed by tx_hash (new grouped flow)
        grouped_ov = db_load(f"f2_classifications_grouped_{src}", user_id=user_id) or {}
        if not isinstance(grouped_ov, dict):
            grouped_ov = {}

        for f in files:
            # Per-upload overrides keyed by row idx (legacy per-file flow,
            # still in use for users who classified before the grouped UI)
            per_upload_ov = db_load(f"f2_classifications_{f.id}", user_id=user_id) or {}
            if not isinstance(per_upload_ov, dict):
                per_upload_ov = {}

            try:
                rows = parse_bank_tx_bytes(src, f.file_bytes)
            except Exception as err:  # noqa: BLE001
                log.warning("prefetch parse failed src=%s upload=%s: %s", src, f.id, err)
                continue

            for r in rows:
                h = r.get("tx_hash") or ""
                if h and h in seen_hashes:
                    continue
                if h:
                    seen_hashes.add(h)

                # Apply per-upload override first (legacy)
                pov = per_upload_ov.get(str(r.get("idx", 0)))
                if isinstance(pov, dict):
                    if pov.get("category"):
                        r["category"] = pov["category"]
                    if pov.get("project") is not None:
                        r["project"] = pov["project"]
                    if pov.get("label"):
                        r["label"] = pov["label"]

                # Apply grouped override (new) — wins over legacy when both exist
                gov = grouped_ov.get(h)
                if isinstance(gov, dict):
                    if gov.get("category"):
                        r["category"] = gov["category"]
                    if gov.get("project") is not None:
                        r["project"] = gov["project"]
                    if gov.get("label"):
                        r["label"] = gov["label"]

                r["_source_key"] = src
                r["_upload_id"] = f.id
                merged.append(r)

    log.info("bank_classifications prefetch user=%s txs=%s", user_id, len(merged))
    return merged


# ── Sync aggregator (consumed by compute_cashflow / compute_balance) ───────


def aggregate_for_project(
    project: str,
    after_date: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Aggregate prefetched txs scoped to one project.

    Returns the same shape that the legacy `aggregate_classified_by_project`
    returns (so caller code is identical). Returns `None` when no prefetch
    has been set — the caller should fall back to the disk-based legacy
    aggregator in that case.
    """
    txs = get_prefetched()
    if txs is None:
        return None

    import pandas as pd

    cutoff = None
    if after_date:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                cutoff = pd.to_datetime(after_date, format=fmt)
                break
            except (ValueError, TypeError):
                continue

    result: dict[str, Any] = {
        "inflows": 0.0,
        "outflows": 0.0,
        "by_category": {},
        "by_source": {},
        "transactions": [],
    }

    for r in txs:
        if (r.get("project") or "") != project:
            continue
        try:
            val = float(r.get("value_brl") or 0)
        except (TypeError, ValueError):
            val = 0.0

        if cutoff is not None:
            d_str = str(r.get("date") or "")[:10]
            try:
                tx_date = pd.to_datetime(d_str, format="%Y-%m-%d")
                if tx_date <= cutoff:
                    continue
            except Exception:  # noqa: BLE001
                pass

        if val > 0:
            result["inflows"] += val
        else:
            result["outflows"] += abs(val)

        cat_key = r.get("category") or "uncategorized"
        result["by_category"][cat_key] = result["by_category"].get(cat_key, 0.0) + val

        src_key = r.get("_source_key") or "unknown"
        result["by_source"][src_key] = result["by_source"].get(src_key, 0.0) + val

        # Convert to legacy-shape tx so downstream code that iterates over
        # `transactions` (looking for "Категория", "Valor", "Data") keeps
        # working without changes.
        d_iso = str(r.get("date") or "")[:10]
        # legacy format expected `DD/MM/YYYY` for compute_cashflow's
        # `pd.to_datetime(ds, dayfirst=True)` — but that accepts ISO too.
        result["transactions"].append({
            "Data": d_iso,
            "Категория": r.get("category", ""),
            "Проект": project,
            "Класс.": r.get("label", "") or r.get("description", ""),
            "Описание": r.get("description", ""),
            "Valor": val,
            "_source_key": r.get("_source_key", ""),
            "_upload_id": r.get("_upload_id"),
            "_tx_hash": r.get("tx_hash", ""),
        })

    return result
