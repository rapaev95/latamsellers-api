"""Position-in-search service for the Escalar Marketing page.

Implementation switched from ML API to headless-Chromium scraping of
`https://lista.mercadolivre.com.br/<slug>` — see `ml_scraper.py` for why
(non-Developer-Partner apps are blocked from `/sites/<site>/search?q=...`).

`PositionResult` + `PositionCheckError` + `check_position` signature kept
intact so `routers/positions.py` and `storage/positions_storage.py` don't
care which back-end produced the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from v2.services import ml_scraper


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


class PositionCheckError(RuntimeError):
    """ML-side failure — propagated to the UI via 502 by the router."""


async def check_position(
    *,
    item_id: str,
    keyword: str,
    site_id: str = "MLB",
    category_id: Optional[str] = None,  # no-op in scraper MVP (kept for future hybrid)
    max_depth: int = 1000,               # no-op — scraper pages are bounded via env/config
    bearer_token: Optional[str] = None,  # no-op — scraper needs no OAuth
    seller_id: Optional[int] = None,     # no-op
) -> PositionResult:
    """Scrape ML buyer-facing search for `keyword`, find `item_id`'s rank.

    Position semantics: 1-based index into the ORGANIC (non-sponsored)
    result list, ordered by ML's default relevance sort (what a buyer sees
    when opening https://lista.mercadolivre.com.br/<keyword>).
    """
    try:
        r = await ml_scraper.fetch_and_rank(
            item_id=item_id,
            keyword=keyword,
            site_id=site_id,
        )
    except ml_scraper.PositionScraperError as err:
        # Re-raise under the caller-facing exception type so the router
        # only needs to catch `PositionCheckError`.
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
    )
