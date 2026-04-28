"""Position-in-search service for the Escalar Marketing page.

Two paths, used in this order per call:
  1. ML public JSON API `/sites/{site}/search?q=...&limit=50&offset=N`
     — fast, stable, doesn't need Playwright. Uses the seller's OAuth
     bearer (header `Authorization`); ML returns paginated `results[]`
     with `id` we match against `item_id`. **Default primary path.**
  2. Logged-in Playwright scraper (fallback) — used when the JSON API
     returns 403 / unexpected shape, OR when the item isn't in the
     first 4000 organic results (ML JSON paging cap). Scraper now
     accepts `ML_SCRAPER_STORAGE_STATE_B64` to act as logged-in user
     (less bot-detection on result pages).

`PositionResult` + `PositionCheckError` + `check_position` signature
kept intact so `routers/positions.py` and `storage/positions_storage.py`
don't care which back-end produced the numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus

import httpx

from v2.services import ml_scraper

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
JSON_API_PAGE = 50
JSON_API_MAX_DEPTH = 1000  # 20 pages × 50; matches scraper cap


@dataclass
class PositionResult:
    item_id: str
    keyword: str
    site_id: str
    position: Optional[int]   # 1-based organic rank; None when item is only in ads
    found: bool
    total_results: int        # ML's own "N resultados" count
    pages_scanned: int
    ads_above: Optional[int] = None  # sponsored slots encountered before organic
    source: str = "scraper"   # "json_api" | "scraper"


class PositionCheckError(RuntimeError):
    """ML-side failure — propagated to the UI via 502 by the router."""


# ── JSON API path ─────────────────────────────────────────────────────────────


def _is_sponsored_item(item: dict) -> bool:
    """ML's search API marks ads in a few different ways depending on
    site / API version. Cover all the forms we've observed."""
    tags = item.get("tags") or []
    if "sponsored" in tags or "ad_highlighted" in tags:
        return True
    # Some sites stamp `differential_pricing` or include `ad_metadata`.
    if isinstance(item.get("ad_metadata"), dict):
        return True
    if "ad_id" in item:
        return True
    return False


async def _check_via_json_api(
    *,
    item_id: str,
    keyword: str,
    site_id: str,
    bearer_token: Optional[str],
    max_depth: int,
) -> Optional[PositionResult]:
    """Returns PositionResult on success (item found OR scanned to max_depth
    without finding), or None when the API itself is unusable (403 / bad
    shape) so the caller can fall back to Playwright."""
    target_id = item_id.upper()
    headers = {"Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    organic_rank = 0
    ads_above = 0
    total_hint = 0
    pages_scanned = 0

    async with httpx.AsyncClient() as http:
        offset = 0
        while offset < max_depth:
            url = (
                f"{ML_API_BASE}/sites/{site_id}/search"
                f"?q={quote_plus(keyword)}&limit={JSON_API_PAGE}&offset={offset}"
            )
            try:
                r = await http.get(url, headers=headers, timeout=15.0)
            except Exception as err:  # noqa: BLE001
                log.warning("positions json_api offset=%s exception: %s", offset, err)
                return None

            if r.status_code == 403:
                # Non-Developer-Partner apps used to be blocked here. With
                # bearer + seller scopes it's currently OK, but if ML
                # restores the gate we drop to scraper instead of failing
                # the whole route.
                log.info("positions json_api forbidden — fallback to scraper")
                return None
            if r.status_code == 404:
                # Some site_ids return 404 for unknown queries; treat as no-results.
                break
            if r.status_code != 200:
                log.warning(
                    "positions json_api offset=%s status=%s body=%s",
                    offset, r.status_code, r.text[:200],
                )
                return None

            try:
                data = r.json() or {}
            except Exception:  # noqa: BLE001
                log.warning("positions json_api offset=%s non-JSON body", offset)
                return None

            results = data.get("results") or []
            pages_scanned += 1
            if not total_hint:
                total_hint = int((data.get("paging") or {}).get("total") or 0)

            if not results:
                if offset == 0:
                    # Genuine 0 results from ML for this query
                    return PositionResult(
                        item_id=item_id, keyword=keyword, site_id=site_id,
                        position=None, found=False,
                        total_results=0, pages_scanned=pages_scanned,
                        ads_above=0, source="json_api",
                    )
                break

            for item in results:
                if _is_sponsored_item(item):
                    ads_above += 1
                    if (item.get("id") or "").upper() == target_id:
                        # Item shows only as sponsored — record found, no
                        # organic rank
                        return PositionResult(
                            item_id=item_id, keyword=keyword, site_id=site_id,
                            position=None, found=True,
                            total_results=total_hint or 0,
                            pages_scanned=pages_scanned,
                            ads_above=ads_above, source="json_api",
                        )
                    continue
                organic_rank += 1
                if (item.get("id") or "").upper() == target_id:
                    return PositionResult(
                        item_id=item_id, keyword=keyword, site_id=site_id,
                        position=organic_rank, found=True,
                        total_results=total_hint or 0,
                        pages_scanned=pages_scanned,
                        ads_above=ads_above, source="json_api",
                    )

            if len(results) < JSON_API_PAGE:
                break
            offset += JSON_API_PAGE

    # Not found within scanned depth — explicit «not_found», don't fall back
    return PositionResult(
        item_id=item_id, keyword=keyword, site_id=site_id,
        position=None, found=False,
        total_results=total_hint or 0,
        pages_scanned=pages_scanned, ads_above=ads_above,
        source="json_api",
    )


# ── Public entry ─────────────────────────────────────────────────────────────


async def check_position(
    *,
    item_id: str,
    keyword: str,
    site_id: str = "MLB",
    category_id: Optional[str] = None,
    max_depth: int = JSON_API_MAX_DEPTH,
    bearer_token: Optional[str] = None,
    seller_id: Optional[int] = None,
) -> PositionResult:
    """Find `item_id` rank for `keyword` in ML buyer-facing search.

    Path order:
      1. JSON API (fast, default). Falls back to scraper only when API
         itself is unusable (403 / bad shape).
      2. Playwright scraper (slow, may be flaky). Used as last resort.

    Position semantics: 1-based index into the ORGANIC (non-sponsored)
    result list, ordered by ML's default relevance sort.
    """
    # Try JSON API first
    api_result = await _check_via_json_api(
        item_id=item_id,
        keyword=keyword,
        site_id=site_id,
        bearer_token=bearer_token,
        max_depth=max_depth,
    )
    if api_result is not None:
        return api_result

    # Fall back to scraper. Scraper now supports storage_state for
    # logged-in browsing — see ml_scraper.fetch_and_rank.
    try:
        r = await ml_scraper.fetch_and_rank(
            item_id=item_id,
            keyword=keyword,
            site_id=site_id,
        )
    except ml_scraper.PositionScraperError as err:
        raise PositionCheckError(str(err)) from err

    return PositionResult(
        item_id=item_id,
        keyword=keyword,
        site_id=site_id,
        position=r.position,
        found=r.found,
        total_results=r.total_results,
        pages_scanned=r.pages_scanned,
        ads_above=r.ads_above,
        source="scraper",
    )
