"""Hourly APScheduler job: populate `ml_ad_*` cache tables from ML.

Fire path: `main.py._ml_scheduler.add_job(ml_ads_sync.sync_tick, 'interval', hours=1)`.
Each tick fetches the list of users with ML tokens and runs `sync_user_full`
for each, isolating failures so one user's expired refresh_token never
stalls the loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import asyncpg

from v2.services import ml_ads
from v2.storage import ads_storage

log = logging.getLogger("ml-ads-sync")

# Cold-UI fallback: if cache for (user, advertiser) is older than this, the
# router triggers a synchronous sync for that advertiser before serving. The
# hourly scheduler normally keeps things fresher than that.
STALE_THRESHOLD_SECONDS = 2 * 3600  # 2h


async def sync_tick(pool: Optional[asyncpg.Pool]) -> dict[str, Any]:
    """Iterate all users with ML tokens, sync each. Never raises."""
    if pool is None:
        log.warning("[ml-ads-sync] No DB pool — skipping")
        return {"ok": 0, "failed": 0, "skipped": 1}

    t0 = time.perf_counter()
    try:
        user_ids = await ads_storage.all_users_with_ml_tokens(pool)
    except Exception as err:  # noqa: BLE001
        log.exception(f"[ml-ads-sync] user list query failed: {err}")
        return {"ok": 0, "failed": 0, "error": str(err)}

    ok = 0
    failed = 0
    details: list[dict] = []
    for uid in user_ids:
        try:
            stats = await ml_ads.sync_user_full(pool, uid)
            ok += 1
            details.append({
                "user_id": uid,
                "advertisers": stats.advertisers,
                "campaigns": stats.campaigns,
                "ads": stats.ads,
            })
        except Exception as err:  # noqa: BLE001
            log.warning(f"[ml-ads-sync] user={uid} failed: {err}")
            failed += 1
            details.append({"user_id": uid, "error": str(err)[:200]})

    elapsed = time.perf_counter() - t0
    log.info(
        f"[ml-ads-sync] tick done ok={ok} failed={failed} users={len(user_ids)} "
        f"elapsed={elapsed:.1f}s"
    )
    return {"ok": ok, "failed": failed, "users": len(user_ids), "elapsed_s": elapsed, "details": details}


async def ensure_fresh_for_advertiser(
    pool: asyncpg.Pool,
    user_id: int,
    advertiser_id: int,
    site_id: str,
    *,
    max_age_seconds: int = STALE_THRESHOLD_SECONDS,
) -> bool:
    """Sync this advertiser synchronously if cache is stale or empty.

    Returns True if a fresh sync was actually performed. UI-facing routes call
    this on cold cache so the user doesn't see an empty grid."""
    synced_at = await ads_storage.campaign_staleness(pool, user_id, advertiser_id)
    if synced_at is not None:
        from datetime import datetime, timezone
        age = (datetime.now(timezone.utc) - synced_at).total_seconds()
        if age < max_age_seconds:
            return False
    try:
        await ml_ads.sync_advertiser_campaigns(pool, user_id, advertiser_id, site_id)
        await ml_ads.sync_advertiser_ads(pool, user_id, advertiser_id, site_id)
    except ml_ads.MLAdsError as err:
        log.warning(
            f"[ml-ads-sync] on-demand sync user={user_id} adv={advertiser_id} "
            f"failed: {err}"
        )
        raise
    return True
