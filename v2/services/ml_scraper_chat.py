"""Playwright-based scraper for ML's authenticated mediation/chat pages.

Why this exists: ML's public OAuth API doesn't expose POST endpoints for
sending messages on order chats or claim mediations. The seller-hub UI
uses a private API behind cookie auth that we can't replicate from a
backend service. Solution — drive Chromium against the actual UI URL
the seller would open in a browser, but using their stored login state.

Architecture:
- Sync Playwright in a thread pool (mirrors ml_scraper.py for positions).
- Storage state (cookies + localStorage of a logged-in ML session) is
  loaded from env ML_SCRAPER_STORAGE_STATE_B64 (one shared session for
  the seller until per-user storage gets wired). Base64 of the JSON
  Playwright produces via `context.storage_state(path=...)`.
- Two operations: read_chat() returns the latest buyer messages + AI
  reply seed text; send_chat() types and submits.

Trade-offs accepted (per user decision):
- Brittle: ML can change DOM at any time and break selectors.
- Single-tenant initial: storage_state is shared, not per-user.
- TOS-grey: we automate a real-user UI; ML may rate-limit or block.

Selectors are intentionally tolerant — text content fallbacks before
class-based ones, since ML churns class names.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeout,
    sync_playwright,
)

log = logging.getLogger("ml-scraper-chat")


# ── Config ──────────────────────────────────────────────────────────
HEADLESS = os.environ.get("ML_SCRAPER_CHAT_HEADLESS", "true").strip().lower() != "false"
CONCURRENCY = int(os.environ.get("ML_SCRAPER_CHAT_CONCURRENCY", "2"))
TIMEOUT_S = float(os.environ.get("ML_SCRAPER_CHAT_TIMEOUT_S", "20"))
RELAUNCH_EVERY = int(os.environ.get("ML_SCRAPER_CHAT_RELAUNCH_EVERY", "100"))

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


class ScraperChatError(RuntimeError):
    """One-token error surfaced to API caller."""


# ── Storage state loader ────────────────────────────────────────────

def _storage_state() -> Optional[dict[str, Any]]:
    """Decode ML_SCRAPER_STORAGE_STATE_B64 → dict.

    Format: base64-encoded JSON of the file Playwright dumps via:
      context.storage_state(path="state.json")
    Generated locally by the seller, pasted into Railway env.
    """
    raw = os.environ.get("ML_SCRAPER_STORAGE_STATE_B64", "").strip()
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        return json.loads(decoded)
    except Exception as err:  # noqa: BLE001
        log.error("storage_state decode failed: %s", err)
        return None


# ── Thread-local Playwright (sync API is thread-bound) ──────────────

_executor: Optional[ThreadPoolExecutor] = None
_thread_local = threading.local()
_scrape_counter = 0
_counter_lock = threading.Lock()


def _thread_browser() -> Browser:
    global _scrape_counter
    with _counter_lock:
        counter = _scrape_counter
    if counter >= RELAUNCH_EVERY and getattr(_thread_local, "browser", None):
        try:
            _thread_local.browser.close()
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
            raise ScraperChatError(f"browser_launch_failed: {err}") from err
        _thread_local.pw = pw
        _thread_local.browser = browser
    return _thread_local.browser


def _make_context(browser: Browser, state: Optional[dict[str, Any]]) -> BrowserContext:
    return browser.new_context(
        storage_state=state if state else None,
        user_agent=_UA,
        locale="pt-BR",
        viewport={"width": 1366, "height": 850},
    )


async def init() -> None:
    global _executor
    _executor = ThreadPoolExecutor(
        max_workers=CONCURRENCY,
        thread_name_prefix="ml-scraper-chat",
    )
    log.info("ml_scraper_chat init (headless=%s, concurrency=%d)", HEADLESS, CONCURRENCY)


async def close() -> None:
    global _executor
    pool = _executor
    _executor = None
    if pool is None:
        return

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


# ── URL builders ────────────────────────────────────────────────────

def _chat_url(pack_id: str | int, claim_id: Optional[str | int] = None) -> str:
    """Mediation URL for claim chat, or general chat for an order."""
    if claim_id:
        return (
            f"https://www.mercadolivre.com.br/vendas/novo/mensagens/{pack_id}"
            f"/mediacao/{claim_id}"
        )
    return f"https://www.mercadolivre.com.br/vendas/novo/mensagens/{pack_id}"


# ── Sync ops (run inside the worker thread) ─────────────────────────

def _bump_counter() -> None:
    global _scrape_counter
    with _counter_lock:
        _scrape_counter += 1


def _open_chat_page(page: Page, pack_id: str | int, claim_id: Optional[str | int]) -> None:
    page.set_default_timeout(int(TIMEOUT_S * 1000))
    url = _chat_url(pack_id, claim_id)
    page.goto(url, wait_until="domcontentloaded")
    # ML's chat UI lazy-loads. Wait for ANY text-bubble or input element.
    page.wait_for_load_state("networkidle", timeout=int(TIMEOUT_S * 1000))


def _scrape_messages_sync(
    pack_id: str | int,
    claim_id: Optional[str | int],
) -> list[dict[str, Any]]:
    """Open chat URL, return list of {sender_role, text, timestamp_str}."""
    state = _storage_state()
    if state is None:
        raise ScraperChatError("storage_state_missing")

    browser = _thread_browser()
    ctx = _make_context(browser, state)
    try:
        page = ctx.new_page()
        try:
            _open_chat_page(page, pack_id, claim_id)
            # Defensive selector: look for ANY message-like bubble. ML uses
            # different DOM per page type; fall back to text scraping.
            bubbles = page.locator(
                'div[class*="message"], div[class*="bubble"], li[class*="message"]'
            ).all()
            messages: list[dict[str, Any]] = []
            for b in bubbles[:30]:  # cap — chat can be long
                try:
                    text = b.inner_text().strip()
                except Exception:  # noqa: BLE001
                    continue
                if not text or len(text) < 3:
                    continue
                # Heuristic for sender — class / aria attributes vary.
                cls = (b.get_attribute("class") or "").lower()
                if "buyer" in cls or "complainant" in cls or "left" in cls:
                    role = "buyer"
                elif "seller" in cls or "respondent" in cls or "right" in cls:
                    role = "seller"
                else:
                    role = "unknown"
                messages.append({"sender_role": role, "text": text[:1000]})
            return messages
        finally:
            page.close()
    finally:
        ctx.close()
        _bump_counter()


def _send_message_sync(
    pack_id: str | int,
    claim_id: Optional[str | int],
    text: str,
) -> dict[str, Any]:
    """Open chat URL, type the text, click Enviar. Returns {ok, detail?}."""
    state = _storage_state()
    if state is None:
        raise ScraperChatError("storage_state_missing")
    if not text or not text.strip():
        return {"ok": False, "detail": "empty_text"}

    browser = _thread_browser()
    ctx = _make_context(browser, state)
    try:
        page = ctx.new_page()
        try:
            _open_chat_page(page, pack_id, claim_id)
            # Find textarea / contenteditable input. ML's chat usually has
            # one of these:
            input_selectors = [
                'textarea[placeholder*="Escreva"]',
                'textarea[placeholder*="mensagem"]',
                'textarea[placeholder*="Mensagem"]',
                'div[contenteditable="true"]',
                'textarea',
            ]
            target = None
            for sel in input_selectors:
                try:
                    loc = page.locator(sel).first
                    loc.wait_for(state="visible", timeout=4000)
                    target = loc
                    break
                except Exception:  # noqa: BLE001
                    continue
            if target is None:
                return {"ok": False, "detail": "input_not_found"}
            target.click()
            target.fill(text)
            page.wait_for_timeout(500)
            # Click Enviar button — try several labels
            for sel in [
                'button:has-text("Enviar")',
                'button:has-text("Send")',
                'button[type="submit"]',
                'button[aria-label*="Enviar"]',
            ]:
                try:
                    btn = page.locator(sel).first
                    btn.wait_for(state="visible", timeout=2000)
                    btn.click()
                    page.wait_for_timeout(1500)
                    return {"ok": True}
                except Exception:  # noqa: BLE001
                    continue
            # Fallback: press Enter
            try:
                target.press("Enter")
                page.wait_for_timeout(1500)
                return {"ok": True, "detail": "sent_via_enter"}
            except Exception as err:  # noqa: BLE001
                return {"ok": False, "detail": f"submit_failed: {err}"}
        finally:
            page.close()
    finally:
        ctx.close()
        _bump_counter()


# ── Public async entry points ───────────────────────────────────────

async def read_chat(pack_id: str | int, claim_id: Optional[str | int] = None) -> list[dict[str, Any]]:
    if _executor is None:
        raise ScraperChatError("scraper_not_initialized")
    return await asyncio.to_thread(_scrape_messages_sync, pack_id, claim_id)


async def send_chat(
    pack_id: str | int,
    text: str,
    claim_id: Optional[str | int] = None,
) -> dict[str, Any]:
    if _executor is None:
        raise ScraperChatError("scraper_not_initialized")
    return await asyncio.to_thread(_send_message_sync, pack_id, claim_id, text)
