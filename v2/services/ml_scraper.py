"""Headless-Chromium scraper for Mercado Livre public search results.

Why this exists: `/sites/MLB/search?q=...` is gated to Developer Partner apps
(non-Partner tokens get 403 PolicyAgent). `/users/{seller}/items/search?q=...`
does a flat title match so real buyer queries like `organizador de carro`
return total=0 even when the item actually ranks #2 in live ML search.

Solution: drive Chromium against `https://lista.mercadolivre.com.br/<slug>`
— the same URL a buyer opens — parse the result list, find the target MLB.

Why **sync** Playwright in a thread pool instead of async?
Uvicorn on Windows forces `WindowsSelectorEventLoopPolicy`, which doesn't
implement `subprocess_exec` → Playwright async init raises NotImplementedError.
Sync Playwright uses `subprocess.Popen` directly, bypassing the asyncio loop;
we just wrap each scrape in `asyncio.to_thread` so FastAPI's event loop stays
responsive. Trade-off: one thread per concurrent scrape (bounded via
`POSITION_SCRAPER_CONCURRENCY`). Works cross-platform.

Contract:
  - `init()` / `close()` called from FastAPI startup/shutdown.
  - `fetch_and_rank(item_id, keyword, ...)` is the public entry; raises
    `PositionScraperError(token[:detail])` on captcha / ip block / parse drift.
  - Position is 1-based into ORGANIC (non-sponsored) cards; `ads_above`
    counts sponsored slots seen before the hit.

HREF is ground truth — ML churns class names. Any anchor with
`/MLB-<digits>` inside a recognised card container defines rank.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Playwright,
    TimeoutError as PlaywrightTimeout,
    sync_playwright,
)

log = logging.getLogger("ml-scraper")


# ── Config (env-overridable) ─────────────────────────────────────────────
MAX_PAGES = int(os.environ.get("POSITION_SCRAPER_MAX_PAGES", "5"))
DELAY_MS = int(os.environ.get("POSITION_SCRAPER_DELAY_MS", "800"))
TIMEOUT_S = float(os.environ.get("POSITION_SCRAPER_TIMEOUT_S", "15"))
CONCURRENCY = int(os.environ.get("POSITION_SCRAPER_CONCURRENCY", "3"))
HEADLESS = os.environ.get("POSITION_SCRAPER_HEADLESS", "true").strip().lower() != "false"
PROXY = os.environ.get("POSITION_SCRAPER_PROXY", "").strip() or None
BLOCK_RESOURCES = set(
    t.strip() for t in os.environ.get(
        "POSITION_SCRAPER_BLOCK_RESOURCES", "image,font,media"
    ).split(",") if t.strip()
)
RELAUNCH_EVERY = int(os.environ.get("POSITION_SCRAPER_RELAUNCH_EVERY", "500"))

HOST_BY_SITE = {"MLB": "lista.mercadolivre.com.br"}

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
]


class PositionScraperError(RuntimeError):
    """Single token (optionally ':detail') surfaced to the caller."""


# ── Logged-in storage state (optional) ───────────────────────────────────
# Same env var as the chat scraper (ml_scraper_chat.py) so a single
# manually-exported `state.json` covers both flows. Logged-in browsing
# significantly reduces ML's bot-detection on search-result pages.

def _apply_stealth(context: BrowserContext) -> None:
    """Apply playwright-stealth patches to a context.

    Without stealth, navigator.webdriver === true is the single most
    obvious bot signal — ML can flag a session in one request. Stealth
    also patches navigator.plugins, languages, chrome.runtime, the
    permissions API and a dozen other minor leaks.

    Library is optional: if not installed, fall through silently —
    UA rotation + headers + storage_state still provide partial
    coverage. Logged via debug only.
    """
    try:
        from playwright_stealth import stealth_sync  # type: ignore
    except ImportError:
        log.debug("playwright-stealth not installed — context not patched")
        return
    try:
        stealth_sync(context)
    except Exception as err:  # noqa: BLE001
        log.warning("playwright-stealth apply failed: %s", err)


def _simulate_mouse_jitter(page, moves: int = 3) -> None:
    """Emit a few mousemove events with realistic curves + delays.

    Bot signal we close: real users generate 50+ mousemove events per
    page session (just by holding the mouse), bots emit 0. ML's fraud
    detection scripts watch for this. Cost is ~500-1500ms total across
    `moves` jumps; far cheaper than the value of looking human.

    Best-effort: any failure (page closed, viewport unknown) is logged
    at debug and we keep going.
    """
    try:
        vp = page.viewport_size or {"width": 1366, "height": 800}
        # Start somewhere reasonable, then make small relative jumps
        # with steps>1 so Playwright generates intermediate mousemove
        # events along the curve (single-point .move() emits just one
        # event; with steps it emits N events).
        x = random.randint(100, max(101, vp["width"] - 100))
        y = random.randint(100, max(101, vp["height"] - 100))
        page.mouse.move(x, y, steps=1)
        for _ in range(moves):
            dx = random.randint(-200, 200)
            dy = random.randint(-150, 150)
            x = max(20, min(vp["width"] - 20, x + dx))
            y = max(20, min(vp["height"] - 20, y + dy))
            page.mouse.move(x, y, steps=random.randint(8, 20))
            page.wait_for_timeout(random.randint(120, 380))
    except Exception as err:  # noqa: BLE001
        log.debug("mouse jitter skipped: %s", err)


def _warmup_navigation(context: BrowserContext) -> None:
    """Visit ML home before the actual search.

    Real users hit the homepage / a category page / a previous search
    before the deep search URL we care about. Direct navigation to a
    `/<slug>_Desde_<offset>` URL with no referer history is one of the
    cheap signals ML's fraud filter weights heavily. A 1-2s warm-up
    plus a few mouse movements fixes that.

    Best-effort: failure here doesn't fail the scrape.
    """
    try:
        page = context.new_page()
        try:
            page.goto(
                "https://www.mercadolivre.com.br/",
                wait_until="domcontentloaded",
                timeout=int(TIMEOUT_S * 1000),
            )
            # Tiny pause + small scroll to look more like a person
            # before opening a new tab. Sync API doesn't have
            # asyncio.sleep, use page.wait_for_timeout (ms).
            page.wait_for_timeout(random.randint(800, 1800))
            _simulate_mouse_jitter(page, moves=random.randint(2, 4))
            try:
                page.evaluate("window.scrollBy(0, 200 + Math.random()*300)")
            except Exception:  # noqa: BLE001
                pass
            # Another short pause + jitter so the homepage gets a real
            # mousemove footprint before we abandon it.
            page.wait_for_timeout(random.randint(300, 700))
            _simulate_mouse_jitter(page, moves=random.randint(1, 2))
        finally:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as err:  # noqa: BLE001
        log.debug("warmup nav skipped: %s", err)


def _storage_state() -> Optional[dict[str, Any]]:
    """Decode ML_SCRAPER_STORAGE_STATE_B64 → dict.

    Format: base64-encoded JSON of the file Playwright dumps via
    `context.storage_state(path="state.json")`. Generated locally by a
    seller, pasted into Railway env. Returns None when unset → scraper
    runs anonymously (legacy behaviour).
    """
    raw = os.environ.get("ML_SCRAPER_STORAGE_STATE_B64", "").strip()
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        return json.loads(decoded)
    except Exception as err:  # noqa: BLE001
        log.warning("storage_state decode failed (continuing anonymous): %s", err)
        return None


# ── Thread-local Playwright (sync API is thread-bound) ────────────────────

_executor: Optional[ThreadPoolExecutor] = None
_thread_local = threading.local()
_scrape_counter = 0
_counter_lock = threading.Lock()


def _thread_browser() -> Browser:
    """Lazy-init a Playwright + Chromium instance bound to this thread.

    Sync Playwright objects can't cross threads, so each worker in the pool
    owns its own Browser. Threads are reused by the executor so the cold-
    launch cost is paid once per thread, not per scrape.
    """
    global _scrape_counter
    with _counter_lock:
        counter = _scrape_counter
    if counter >= RELAUNCH_EVERY and getattr(_thread_local, "browser", None):
        log.info("relaunch Chromium after %d scrapes (thread=%s)",
                 counter, threading.current_thread().name)
        try:
            _thread_local.browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            _thread_local.pw.stop()
        except Exception:  # noqa: BLE001
            pass
        _thread_local.browser = None
        _thread_local.pw = None
        with _counter_lock:
            _scrape_counter = 0

    if getattr(_thread_local, "browser", None) is None:
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch(headless=HEADLESS)
        except PlaywrightError as err:
            try:
                pw.stop()
            except Exception:  # noqa: BLE001
                pass
            raise PositionScraperError(f"browser_launch_failed: {err}") from err
        _thread_local.pw = pw
        _thread_local.browser = browser
    return _thread_local.browser


async def init() -> None:
    """Create a bounded thread pool so we can't DoS ML from one IP."""
    global _executor
    _executor = ThreadPoolExecutor(
        max_workers=CONCURRENCY,
        thread_name_prefix="ml-scraper",
    )
    log.info(
        "ml_scraper init OK (headless=%s, concurrency=%d, max_pages=%d)",
        HEADLESS, CONCURRENCY, MAX_PAGES,
    )


async def close() -> None:
    """Shut down the pool. Browsers owned by worker threads are closed in
    their own thread (Playwright requires same-thread cleanup); the pool
    shutdown will block until those finish."""
    global _executor
    pool = _executor
    _executor = None
    if pool is None:
        return

    # Tell each live worker to release its own browser.
    def _close_local() -> None:
        try:
            if getattr(_thread_local, "browser", None) is not None:
                _thread_local.browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(_thread_local, "pw", None) is not None:
                _thread_local.pw.stop()
        except Exception:  # noqa: BLE001
            pass

    try:
        futures = [pool.submit(_close_local) for _ in range(CONCURRENCY)]
        for f in futures:
            try:
                f.result(timeout=5)
            except Exception:  # noqa: BLE001
                pass
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ── URL building ─────────────────────────────────────────────────────────

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def slugify(keyword: str) -> str:
    """ML URL slug: NFKD → drop combining marks → lower → non-alnum→'-'."""
    if not keyword:
        return ""
    nfkd = unicodedata.normalize("NFKD", keyword)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = ascii_only.lower()
    slug = _SLUG_STRIP_RE.sub("-", lowered).strip("-")
    return slug[:100]


def build_search_url(keyword: str, site_id: str, page: int) -> str:
    host = HOST_BY_SITE.get(site_id.upper())
    if not host:
        raise PositionScraperError("site_not_supported")
    slug = slugify(keyword)
    if not slug:
        raise PositionScraperError("invalid_keyword")
    base = f"https://{host}/{slug}"
    if page <= 1:
        return base
    # Pagination: step 48, _Desde_<1-based offset>
    offset = 1 + 48 * (page - 1)
    return f"{base}_Desde_{offset}"


# ── Captcha / bot-wall detection ─────────────────────────────────────────

_CAPTCHA_URL_HINTS = ("abuse_interstitial", "captcha", "challenge", "/jms/")
_CAPTCHA_HTML_HINTS = (
    "verifique que você é humano",
    "verify you are human",
    "acesso negado",
    "access denied",
    "px-captcha",
    "cf-challenge",
)


def detect_captcha(html: str, final_url: str) -> bool:
    url_l = (final_url or "").lower()
    if any(h in url_l for h in _CAPTCHA_URL_HINTS):
        return True
    html_l = (html or "").lower()
    return any(h in html_l for h in _CAPTCHA_HTML_HINTS)


# ── HTML parsing helpers (run inside browser via page.evaluate) ──────────

_CARD_SELECTORS = [
    "li.ui-search-layout__item",
    "div.poly-card",
    "li[class*='search-layout']",
]
_AD_SELECTOR = (
    ".poly-component__ads-promotions, "
    "[data-testid*='ad' i], "
    "[class*='sponsored' i], "
    "[class*='ad-label' i]"
)
_MLB_HREF_RE = re.compile(r"MLB[-]?(\d{7,12})", re.IGNORECASE)
_TOTAL_RE = re.compile(r"([\d.,]+)\s+resultado", re.IGNORECASE)


@dataclass
class ResultItem:
    mlb_id: str
    is_ad: bool


# ── Sync scrape (runs in worker thread) ──────────────────────────────────

@dataclass
class RankedResult:
    position: Optional[int]
    found: bool
    total_results: int
    pages_scanned: int
    ads_above: int


def _build_proxy_arg() -> Optional[dict]:
    if not PROXY:
        return None
    parsed = urlparse(PROXY)
    arg: dict = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        arg["username"] = parsed.username
    if parsed.password:
        arg["password"] = parsed.password
    return arg


def _fetch_and_parse_one(context: BrowserContext, url: str) -> tuple[list[ResultItem], int]:
    page = context.new_page()
    try:
        if BLOCK_RESOURCES:
            def _route(route):
                if route.request.resource_type in BLOCK_RESOURCES:
                    route.abort()
                else:
                    route.continue_()
            page.route("**/*", _route)

        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=int(TIMEOUT_S * 1000))
        except PlaywrightTimeout as err:
            raise PositionScraperError(f"network_timeout: {err}") from err
        except PlaywrightError as err:
            raise PositionScraperError(f"network_error: {err}") from err

        status = resp.status if resp else 0
        final_url = page.url

        try:
            early_html = page.content()
        except PlaywrightError:
            early_html = ""

        if status == 403 and "abuse_interstitial" not in final_url.lower():
            raise PositionScraperError("ml_blocked_ip")
        if detect_captcha(early_html, final_url):
            raise PositionScraperError("ml_captcha")

        for sel in _CARD_SELECTORS:
            try:
                page.wait_for_selector(sel, timeout=5000)
                break
            except PlaywrightTimeout:
                continue

        # Look human while the cards "load": brief pause + mouse jitter
        # + small scroll. ML's fraud scripts watch for mousemove count
        # per page session — without these we'd emit 0 events and look
        # automated even though the DOM-side checks already passed.
        page.wait_for_timeout(random.randint(400, 1100))
        _simulate_mouse_jitter(page, moves=random.randint(2, 4))
        try:
            page.evaluate(
                "window.scrollBy(0, 250 + Math.random()*400)"
            )
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(random.randint(200, 500))

        # Pull card list + total via single page.evaluate for speed.
        try:
            payload = page.evaluate(
                """({cardSelectors, adSelector}) => {
                    let cards = [];
                    for (const sel of cardSelectors) {
                        cards = Array.from(document.querySelectorAll(sel));
                        if (cards.length) break;
                    }
                    const items = cards.map(card => {
                        const link = card.querySelector("a[href*='/MLB-'], a[href*='MLB']");
                        const href = link ? link.href : '';
                        const isAdClass = !!card.querySelector(adSelector);
                        const hasPatrocinado = (card.innerText || '').toLowerCase().includes('patrocinado');
                        return { href, isAd: isAdClass || hasPatrocinado };
                    });
                    const bodyText = document.body ? document.body.innerText.slice(0, 5000) : '';
                    return { items, bodyText };
                }""",
                {"cardSelectors": _CARD_SELECTORS, "adSelector": _AD_SELECTOR},
            )
        except PlaywrightError as err:
            raise PositionScraperError(f"parse_failed: {err}") from err

        results: list[ResultItem] = []
        for c in payload.get("items", []):
            href = c.get("href") or ""
            m = _MLB_HREF_RE.search(href)
            if not m:
                continue
            results.append(ResultItem(mlb_id=f"MLB{m.group(1)}", is_ad=bool(c.get("isAd"))))

        total = 0
        m = _TOTAL_RE.search(payload.get("bodyText") or "")
        if m:
            try:
                total = int(m.group(1).replace(".", "").replace(",", ""))
            except ValueError:
                total = 0
        return results, total
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass


def _scrape_sync(item_id: str, keyword: str, site_id: str, max_pages: int) -> RankedResult:
    global _scrape_counter
    target_id = item_id.upper().strip()
    browser = _thread_browser()

    ua = random.choice(_UA_POOL)
    viewport = {
        "width": random.choice([1280, 1366, 1440]),
        "height": random.choice([720, 800, 900]),
    }
    proxy_arg = _build_proxy_arg()
    storage_state = _storage_state()

    # Sec-Ch-Ua client hints — real Chrome sends these on every request,
    # bots usually omit them and ML's fraud detection notices.
    chrome_major = "129"
    if "Chrome/128" in ua:
        chrome_major = "128"
    sec_ch_ua = (
        f'"Chromium";v="{chrome_major}", "Not=A?Brand";v="24", '
        f'"Google Chrome";v="{chrome_major}"'
    )
    sec_ch_ua_platform = '"Windows"' if "Windows" in ua else (
        '"macOS"' if "Mac OS" in ua else '"Linux"'
    )

    ctx_kwargs: dict[str, Any] = {
        "user_agent": ua,
        "viewport": viewport,
        "locale": "pt-BR",
        "timezone_id": "America/Sao_Paulo",  # consistent with locale
        "extra_http_headers": {
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Sec-Ch-Ua": sec_ch_ua,
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": sec_ch_ua_platform,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        },
        "proxy": proxy_arg,
    }
    if storage_state is not None:
        # Logged-in browsing — ML returns higher-quality pages and
        # rarely shows the bot-wall to authenticated sellers.
        ctx_kwargs["storage_state"] = storage_state

    context = browser.new_context(**ctx_kwargs)
    # Stealth patches: navigator.webdriver=undefined, plugins.length=3,
    # chrome.runtime, languages, permissions API quirks. Library is
    # optional — if not installed, log and continue without (still
    # better than nothing thanks to UA + headers + storage_state).
    _apply_stealth(context)
    # Brief warm-up: visit ML home first so referer chain looks like a
    # real user landing page → search. ML's fraud signal weight on
    # «direct nav to deep search URL» is non-trivial.
    _warmup_navigation(context)
    try:
        organic_rank = 0
        ads_above_total = 0
        total_results_hint = 0
        pages = 0

        for page_idx in range(1, max_pages + 1):
            url = build_search_url(keyword, site_id, page_idx)
            log.info("scrape p%d %s", page_idx, url)
            results, total = _fetch_and_parse_one(context, url)
            pages += 1
            with _counter_lock:
                _scrape_counter += 1
            if total and not total_results_hint:
                total_results_hint = total

            if not results:
                if page_idx == 1:
                    raise PositionScraperError("ml_no_results")
                break

            for item in results:
                if item.is_ad:
                    ads_above_total += 1
                    if item.mlb_id.upper() == target_id:
                        # Appears only as a sponsored slot — record found but
                        # no organic position.
                        return RankedResult(
                            position=None,
                            found=True,
                            total_results=total_results_hint or 0,
                            pages_scanned=pages,
                            ads_above=ads_above_total,
                        )
                    continue
                organic_rank += 1
                if item.mlb_id.upper() == target_id:
                    return RankedResult(
                        position=organic_rank,
                        found=True,
                        total_results=total_results_hint or 0,
                        pages_scanned=pages,
                        ads_above=ads_above_total,
                    )

            # polite jitter between pages
            import time
            time.sleep(random.uniform(DELAY_MS * 0.5, DELAY_MS * 1.5) / 1000.0)

        return RankedResult(
            position=None,
            found=False,
            total_results=total_results_hint or 0,
            pages_scanned=pages,
            ads_above=ads_above_total,
        )
    finally:
        try:
            context.close()
        except Exception:  # noqa: BLE001
            pass


# ── Public async entry-point ──────────────────────────────────────────────

async def fetch_and_rank(
    *,
    item_id: str,
    keyword: str,
    site_id: str = "MLB",
    max_pages: int = MAX_PAGES,
) -> RankedResult:
    """Run the sync scraper in the bounded thread pool and await the result."""
    if _executor is None:
        # init() wasn't called — lazy-create a minimal pool (dev ergonomics)
        await init()
    assert _executor is not None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _scrape_sync, item_id, keyword, site_id, max_pages
    )
