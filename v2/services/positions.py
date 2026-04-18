"""Position-in-search service for the Escalar Marketing page.

ML has no dedicated "position of my item for keyword X" endpoint. We poll the
public `/sites/<site>/search` (no auth, no OAuth token) with pagination and
scan for our MLB id. ML caps search depth at `offset + limit <= 1000`, so an
item beyond page ~20 is effectively "not found" for tracking purposes.

Keep ML-facing calls here; persistence lives in `storage/positions_storage.py`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx

ML_API = "https://api.mercadolibre.com"
PAGE_SIZE = 50
MAX_DEPTH = 1000  # ML hard cap
REQUEST_TIMEOUT = 15.0
DELAY_BETWEEN_PAGES_S = 0.25  # polite pacing — reduces 429 risk


@dataclass
class PositionResult:
    item_id: str
    keyword: str
    site_id: str
    position: Optional[int]  # 1-based
    found: bool
    total_results: int
    pages_scanned: int


class PositionCheckError(RuntimeError):
    """Raised when ML rejects the call for auth/quota reasons — UI surfaces this."""


async def check_position(
    *,
    item_id: str,
    keyword: str,
    site_id: str = "MLB",
    category_id: Optional[str] = None,
    max_depth: int = MAX_DEPTH,
    bearer_token: Optional[str] = None,
) -> PositionResult:
    """Search ML for `keyword`, paginate until `item_id` is found or exhausted.

    ML tightened `/sites/<site>/search` in 2025 — anonymous calls now return 403.
    A Bearer token from the user's ML OAuth (scope `read`) is required.
    Returns position as 1-based rank across the concatenated result list.
    """
    if not bearer_token:
        raise PositionCheckError("ml_oauth_required")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    offset = 0
    pages = 0
    total_results = 0
    found_position: Optional[int] = None

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        while offset < max_depth:
            params: dict[str, str] = {
                "q": keyword,
                "offset": str(offset),
                "limit": str(PAGE_SIZE),
            }
            if category_id:
                params["category"] = category_id

            try:
                resp = await client.get(
                    f"{ML_API}/sites/{site_id}/search",
                    params=params,
                    headers=headers,
                )
            except httpx.RequestError as err:
                raise PositionCheckError(f"network_error: {err}") from err
            if resp.status_code == 401:
                raise PositionCheckError("ml_token_invalid")
            if resp.status_code == 403:
                raise PositionCheckError("ml_forbidden")
            if resp.status_code == 429:
                raise PositionCheckError("ml_rate_limited")
            if resp.status_code != 200:
                raise PositionCheckError(f"ml_http_{resp.status_code}")

            data = resp.json()
            pages += 1
            results = data.get("results", []) or []
            if not total_results:
                total_results = int((data.get("paging") or {}).get("total") or 0)

            for i, item in enumerate(results):
                if (item.get("id") or "").upper() == item_id.upper():
                    found_position = offset + i + 1
                    break
            if found_position is not None:
                break

            if len(results) < PAGE_SIZE:
                break  # last page
            offset += PAGE_SIZE
            await asyncio.sleep(DELAY_BETWEEN_PAGES_S)

    return PositionResult(
        item_id=item_id,
        keyword=keyword,
        site_id=site_id,
        position=found_position,
        found=found_position is not None,
        total_results=total_results,
        pages_scanned=pages,
    )
