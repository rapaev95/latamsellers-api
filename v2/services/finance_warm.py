"""Scheduled pre-warm of the finance compute cache.

The point is to move cold computes off the user's critical path. A cold
/finance/reports or /finance/pnl-matrix for a large project runs for tens of
seconds; whoever opens the page first after a data change pays that, and on a
single-worker process everyone else queues behind them.

What makes this cheap enough to run often: warming goes through the ordinary
read-through `cached_compute`, WITHOUT force. When nothing a project depends on
has changed, the fingerprint still matches and the call costs one SELECT — no
compute at all. So the cost of a pass is proportional to what actually changed
since the last one, not to the number of projects.

Deliberately sequential. These computes are CPU-bound Python on one gunicorn
worker: running them concurrently doesn't finish them sooner (GIL), it only
starves live requests. A warm pass must be invisible to anyone using the app.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date
from typing import Any

log = logging.getLogger("finance-warm")

# Overall wall-clock cap for one pass. A pass that runs long is not an
# emergency — it just stops and the next one picks up the rest, because
# everything it did finish is now cached.
_BUDGET_S = int(os.environ.get("FINANCE_WARM_BUDGET_S", "900"))
# The window /finance/reports requests on load (see the page's 90-day default).
# Warming any other window would just fill the cache with keys nobody reads.
_REPORTS_WINDOW_DAYS = int(os.environ.get("FINANCE_WARM_REPORTS_DAYS", "90"))
_WARM_REPORTS = os.environ.get("FINANCE_WARM_REPORTS", "1") not in ("0", "false", "False")
# Escalar ABC windows to pre-compute. 30 feeds /escalar/products, 90 feeds
# /escalar/promotions — the two the UI opens with. Per-project drill-downs
# (`abc:<project>:<days>`) stay on demand: warming every project × window
# would multiply the nightly cost for keys most users never open.
_WARM_ABC_DAYS = tuple(
    int(d) for d in os.environ.get("FINANCE_WARM_ABC_DAYS", "30,90").split(",") if d.strip()
)


async def _users_with_projects(pool) -> list[int]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT DISTINCT user_id
                 FROM user_data
                WHERE data_key IN ('projects', 'f2_projects')
                  AND user_id IS NOT NULL
                ORDER BY user_id"""
        )
    return [int(r["user_id"]) for r in rows]


async def warm_all(pool) -> dict[str, Any]:
    """One warm pass over every user's projects. Returns a summary for the log."""
    if pool is None:
        return {"skipped": "no_pool"}

    # Imported here, not at module scope: the router pulls in the whole legacy
    # compute stack, and startup shouldn't pay for that until a pass runs.
    from v2.routers.finance import (
        _bind_user_id, _pnl_matrix_cached, _reports_bundle_cached,
        _services_bundle_computed, _services_default_period,
    )
    # Same function the endpoint calls — see its docstring for why warming
    # through a copy of the logic would be worse than not warming at all.
    from v2.routers.escalar import abc_summary_cached
    from v2.legacy import config as legacy_config

    started = time.monotonic()
    deadline = started + _BUDGET_S
    stats = {"users": 0, "matrix": 0, "reports": 0, "services": 0, "abc": 0,
             "errors": 0, "stopped_early": False}

    try:
        user_ids = await _users_with_projects(pool)
    except Exception as err:  # noqa: BLE001
        log.warning("warm: can't list users: %s", err)
        return {"error": str(err)}

    today = date.today()
    pf = date.fromordinal(today.toordinal() - _REPORTS_WINDOW_DAYS)

    for user_id in user_ids:
        if time.monotonic() > deadline:
            stats["stopped_early"] = True
            break
        stats["users"] += 1
        try:
            _bind_user_id(user_id)
            projects = legacy_config.load_projects() or {}
        except Exception as err:  # noqa: BLE001
            log.warning("warm: user %s projects failed: %s", user_id, err)
            stats["errors"] += 1
            continue

        # Escalar ABC — per user, not per project: the key is `abc:all:<days>`.
        for days in _WARM_ABC_DAYS:
            if time.monotonic() > deadline:
                stats["stopped_early"] = True
                break
            try:
                await abc_summary_cached(pool, user_id, days)
                stats["abc"] += 1
            except Exception as err:  # noqa: BLE001
                stats["errors"] += 1
                log.warning("warm: abc %s/%sd failed: %s", user_id, days, err)

        for name, meta in projects.items():
            if time.monotonic() > deadline:
                stats["stopped_early"] = True
                break
            ptype = (meta or {}).get("type") if isinstance(meta, dict) else None
            try:
                if ptype == "services":
                    s_pf, s_pt = _services_default_period()
                    await _services_bundle_computed(pool, user_id, name, s_pf, s_pt)
                    stats["services"] += 1
                    continue

                await asyncio.to_thread(_pnl_matrix_cached, user_id, name, False)
                stats["matrix"] += 1

                if _WARM_REPORTS:
                    # Same prefetch the endpoint does — without it the compute
                    # reads no bank classifications and caches wrong numbers.
                    # Warming runs with no caller, so the owner IS the user here.
                    try:
                        from v2.services import bank_classifications as _bank_cls
                        prefetched = await _bank_cls.prefetch_for_user(pool, user_id)
                        _bank_cls.set_prefetched(prefetched)
                    except Exception:  # noqa: BLE001
                        pass
                    await asyncio.to_thread(
                        _reports_bundle_cached,
                        user_id, name, projects, pf, today, "accrual", False,
                    )
                    stats["reports"] += 1
            except Exception as err:  # noqa: BLE001
                stats["errors"] += 1
                log.warning("warm: %s/%s failed: %s", user_id, name, err)

    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    log.info("warm pass: %s", stats)
    return stats
