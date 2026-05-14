"""FBM (Full) warehouse space-purchase capacity monitor.

ML's seller-portal page «Comprar mais espaço no Full» (URL:
mercadolivre.com.br/metricas/stock-full/capacity-buy/purchase) shows the
available units a seller can buy this month per size category:
  - «Pequenos e médios» — small/medium items
  - «Grandes e extragrandes» — large/extra-large items

These quotas are limited per seller per month. ML often shows «Não há
espaço disponível» for pequenos, then later opens slots. Seller wants to
know IMMEDIATELY when slots open so he can buy before they're gone.

Public ML API doesn't expose this — page is web-UI only. We scrape via
the existing Playwright session in v2/services/ml_scraper (uses
ML_SCRAPER_STORAGE_STATE_B64 for auth — same user account).

Strategy: poll every N hours, store latest snapshot. If category state
changes from «no space» → «space available», push TG alert with details
+ deep-link to the buy page. NO auto-buy — financial action requires
explicit user click.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

from . import ml_scraper as _scraper

log = logging.getLogger(__name__)

CAPACITY_URL = "https://www.mercadolivre.com.br/metricas/stock-full/capacity-buy/purchase"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_fbm_capacity (
  user_id INTEGER NOT NULL,
  category TEXT NOT NULL,  -- 'pequenos_e_medios' | 'grandes_e_extragrandes'
  available_units INTEGER NOT NULL DEFAULT 0,
  price_per_unit NUMERIC,
  raw_label TEXT,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_alerted_at TIMESTAMPTZ,
  PRIMARY KEY (user_id, category)
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


# ── Scraping ──────────────────────────────────────────────────────────────────

def _parse_capacity_text(body_text: str) -> dict[str, dict[str, Any]]:
    """Parse the bodyText scraped from the capacity-buy page → per-category dict.

    Looking for patterns like:
      «Pequenos e médios» followed by either:
        - «Não há espaço disponível»          → available=0
        - «Seu espaço para junho: 500 un.»    → available=500
      «Grandes e extragrandes» same shape.
    Plus price «R$ 9,40 por unidade».

    Returns:
      {
        "pequenos_e_medios": {"available": 0|N, "price_per_unit": float|None,
                              "raw_label": "Não há espaço disponível" | "Seu espaço para junho: 500 un."},
        "grandes_e_extragrandes": {...}
      }
    Empty {} if page didn't render expected content.
    """
    out: dict[str, dict[str, Any]] = {}
    if not body_text:
        return out

    # Normalize whitespace for line-based matching.
    text = re.sub(r"[ \t]+", " ", body_text)

    # Patterns are case-insensitive and tolerant of formatting variants.
    cat_patterns = [
        ("pequenos_e_medios", r"pequenos\s+e\s+m[eé]dios"),
        ("grandes_e_extragrandes", r"grandes\s+e\s+extragrandes"),
    ]
    no_space_re = re.compile(r"n[aã]o\s+h[aá]\s+espa[cç]o\s+dispon[ií]vel", re.IGNORECASE)
    avail_re = re.compile(
        r"seu\s+espa[cç]o\s+(?:para\s+\w+\s*:?\s*)?([\d.,]+)\s*un",
        re.IGNORECASE,
    )
    price_re = re.compile(r"R\$\s*([\d.,]+)\s*por\s+unidade", re.IGNORECASE)

    # Walk each category — slice from its match to the start of next category
    # (or end of text) to scope the searches.
    matches = []
    for key, pat in cat_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            matches.append((key, m.start(), m.end()))
    matches.sort(key=lambda x: x[1])

    for i, (key, start, end) in enumerate(matches):
        next_start = matches[i + 1][1] if i + 1 < len(matches) else len(text)
        chunk = text[end:next_start]
        # Defaults
        available = 0
        price_pu: Optional[float] = None
        raw_label = ""
        if no_space_re.search(chunk):
            raw_label = "Não há espaço disponível"
            available = 0
        else:
            am = avail_re.search(chunk)
            if am:
                num_str = am.group(1).replace(".", "").replace(",", ".")
                try:
                    available = int(float(num_str))
                except ValueError:
                    available = 0
                raw_label = am.group(0).strip()
        pm = price_re.search(chunk)
        if pm:
            num_str = pm.group(1).replace(".", "").replace(",", ".")
            try:
                price_pu = float(num_str)
            except ValueError:
                price_pu = None
        out[key] = {
            "available": available,
            "price_per_unit": price_pu,
            "raw_label": raw_label,
        }
    return out


def _scrape_sync() -> dict[str, dict[str, Any]]:
    """Synchronous Playwright scrape — runs in worker thread.

    Re-uses _thread_browser() from ml_scraper for auth + warm context.
    """
    import threading
    log.info("FBM capacity scrape starting on thread %s", threading.current_thread().name)
    browser = _scraper._thread_browser()
    storage = _scraper._storage_state()
    if not storage:
        log.warning("no ML_SCRAPER_STORAGE_STATE_B64 — capacity scrape будет анонимная (вероятно redirect на login)")
    context = browser.new_context(
        storage_state=storage,
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        ),
        locale="pt-BR",
        viewport={"width": 1366, "height": 800},
    )
    _scraper._apply_stealth(context)
    page = context.new_page()
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        try:
            resp = page.goto(CAPACITY_URL, wait_until="domcontentloaded", timeout=20000)
        except PlaywrightTimeout as err:
            log.warning("capacity goto timeout: %s", err)
            return {}
        status = resp.status if resp else 0
        if status >= 400:
            log.warning("capacity goto status=%s url=%s", status, page.url)
            return {}
        # Wait a moment for client-rendered content. The page is React-based;
        # «Pequenos e médios» / «Grandes e extragrandes» labels render after
        # initial paint.
        try:
            page.wait_for_selector("text=Pequenos e médios", timeout=8000)
        except PlaywrightTimeout:
            log.warning("capacity page didn't show expected labels — auth or layout change")
        page.wait_for_timeout(800)
        body_text = ""
        try:
            body_text = page.evaluate("() => document.body ? document.body.innerText : ''") or ""
        except Exception as err:  # noqa: BLE001
            log.warning("body innerText extract failed: %s", err)
        return _parse_capacity_text(body_text)
    finally:
        try:
            page.close()
            context.close()
        except Exception:  # noqa: BLE001
            pass


async def scrape_capacity() -> dict[str, dict[str, Any]]:
    """Async wrapper — runs _scrape_sync in a worker thread."""
    return await asyncio.to_thread(_scrape_sync)


# ── Persistence ────────────────────────────────────────────────────────────────

async def refresh_user_capacity(pool: asyncpg.Pool, user_id: int) -> dict[str, Any]:
    """Scrape + upsert. Returns {old, new, changes: ['pequenos_e_medios opened: 0→500', ...]}."""
    await ensure_schema(pool)
    new_state = await scrape_capacity()
    if not new_state:
        return {"ok": False, "reason": "scrape_empty"}
    async with pool.acquire() as conn:
        old_rows = await conn.fetch(
            "SELECT category, available_units, price_per_unit FROM ml_fbm_capacity WHERE user_id = $1",
            user_id,
        )
    old_state = {r["category"]: {
        "available": int(r["available_units"]),
        "price_per_unit": float(r["price_per_unit"]) if r["price_per_unit"] is not None else None,
    } for r in old_rows}

    changes: list[str] = []
    for cat, ns in new_state.items():
        os_ = old_state.get(cat, {"available": 0, "price_per_unit": None})
        old_avail = int(os_.get("available") or 0)
        new_avail = int(ns.get("available") or 0)
        if old_avail == 0 and new_avail > 0:
            changes.append(f"{cat}: opened ({new_avail} un.)")
        elif new_avail > old_avail:
            changes.append(f"{cat}: increased ({old_avail}→{new_avail} un.)")
        elif new_avail == 0 and old_avail > 0:
            changes.append(f"{cat}: closed ({old_avail} un. lost)")

    async with pool.acquire() as conn:
        for cat, ns in new_state.items():
            await conn.execute(
                """
                INSERT INTO ml_fbm_capacity (user_id, category, available_units, price_per_unit, raw_label, fetched_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (user_id, category) DO UPDATE SET
                  available_units = EXCLUDED.available_units,
                  price_per_unit  = EXCLUDED.price_per_unit,
                  raw_label       = EXCLUDED.raw_label,
                  fetched_at      = NOW()
                """,
                user_id, cat,
                int(ns.get("available") or 0),
                ns.get("price_per_unit"),
                str(ns.get("raw_label") or ""),
            )

    return {"ok": True, "new_state": new_state, "old_state": old_state, "changes": changes}


# ── Alerts ─────────────────────────────────────────────────────────────────────

CAT_LABELS = {
    "pequenos_e_medios": "Pequenos e médios",
    "grandes_e_extragrandes": "Grandes e extragrandes",
}


async def _push_capacity_alert(
    pool: asyncpg.Pool, user_id: int,
    category: str, new_avail: int, price_pu: Optional[float],
) -> None:
    """Insert ml_notices row → next TG dispatch tick sends it to seller."""
    from . import ml_notices as _ml_notices_svc
    label_cat = CAT_LABELS.get(category, category)
    price_str = ""
    if price_pu is not None and price_pu > 0:
        # BRL format
        price_str = f" · R$ {price_pu:.2f}/un".replace(".", ",")
    desc_lines = [
        f"🏭 *{label_cat}*",
        f"",
        f"✅ Espaço disponível: *{new_avail} un.*{price_str}",
        f"",
        f"ML acabou de abrir espaço — compre logo, vagas estão limitadas.",
        f"",
        f"🔗 [Comprar espaço Full]({CAPACITY_URL})",
    ]
    notice = {
        "notice_id": f"fbm_capacity:{category}:{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "label": f"FBM Full: espaço aberto em {label_cat}",
        "description": "\n".join(desc_lines),
        "tags": ["FBM_CAPACITY", category.upper()],
        "actions": [],
        "from_date": datetime.now(timezone.utc),
        "topic": "fbm_capacity",
        "resource": "/fbm/capacity-buy",
        "raw": {
            "category": category,
            "available_units": new_avail,
            "price_per_unit": price_pu,
        },
    }
    await _ml_notices_svc.upsert_normalized(pool, user_id, notice)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE ml_fbm_capacity SET last_alerted_at = NOW() WHERE user_id = $1 AND category = $2",
            user_id, category,
        )


async def check_and_alert_user(pool: asyncpg.Pool, user_id: int) -> dict[str, Any]:
    """Refresh capacity, detect transitions «0 → N > 0», push TG alert."""
    res = await refresh_user_capacity(pool, user_id)
    if not res.get("ok"):
        return res
    new_state = res["new_state"]
    old_state = res["old_state"]
    alerted: list[str] = []
    for cat, ns in new_state.items():
        old_avail = int((old_state.get(cat) or {}).get("available") or 0)
        new_avail = int(ns.get("available") or 0)
        # Alert: previously zero (or empty), now > 0. We do NOT alert на
        # каждое изменение — только moment of opening. Closure (N→0) — silent.
        if old_avail == 0 and new_avail > 0:
            await _push_capacity_alert(pool, user_id, cat, new_avail, ns.get("price_per_unit"))
            alerted.append(cat)
    return {"ok": True, "new_state": new_state, "alerted": alerted, "changes": res.get("changes", [])}


async def check_all_users(pool: asyncpg.Pool) -> dict[str, Any]:
    """Scheduler entry — currently only one user has scraper storage_state.
    Iterates over all ML-connected users for forward-compat."""
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
        )
    totals = {"users_checked": 0, "alerts": 0, "errors": 0}
    for r in rows:
        uid = r["user_id"]
        try:
            res = await check_and_alert_user(pool, uid)
            totals["users_checked"] += 1
            if res.get("alerted"):
                totals["alerts"] += len(res["alerted"])
        except Exception as err:  # noqa: BLE001
            log.warning("fbm capacity check user=%s failed: %s", uid, err)
            totals["errors"] += 1
    return totals
