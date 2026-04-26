"""
LatAm Sellers Finance — File Analysis API
Accepts uploaded files, auto-detects source type, returns summary.
Saves results to PostgreSQL.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from datetime import datetime

# Configure root logger before importing service modules — without an attached
# handler, module-level logger.info(...) calls are silently dropped.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Playwright (used by v2/services/ml_scraper.py) needs subprocess support
# from the asyncio loop. On Windows uvicorn defaults to a Selector loop
# which raises NotImplementedError for subprocess_exec. Switch policy
# BEFORE uvicorn picks up the loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import pandas as pd
import psycopg2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="LatAm Sellers Finance API")

# CORS — allowlist (was "*"). Required for v2 cookie auth (`allow_credentials=True`
# is incompatible with wildcard origin). v1 endpoints use the same allowlist.
# Add new origins via env: CORS_ORIGINS=https://example.com,https://other.com
import os as _os
_extra_origins = [o.strip() for o in _os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
_allowed_origins = list(dict.fromkeys([
    "http://localhost:3001",
    "http://localhost:3000",
    "https://lsprofit.app",
    "https://www.lsprofit.app",
    "https://app.lsprofit.app",
] + _extra_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount v2 router (Finance 2.0)
from v2.app import router as v2_router  # noqa: E402
from v2.db import close_pool, create_pool, get_pool  # noqa: E402
from v2.services import ml_backfill as ml_backfill_svc  # noqa: E402
from v2.services import ml_item_context as ml_item_context_svc  # noqa: E402
from v2.services import ml_notices as ml_notices_svc  # noqa: E402
from v2.services import ml_normalize as ml_normalize_svc  # noqa: E402
from v2.services import ml_oauth as ml_oauth_svc  # noqa: E402
from v2.services import ml_quality as ml_quality_svc  # noqa: E402
from v2.services import ml_questions_dispatch as ml_questions_dispatch_svc  # noqa: E402
from v2.services import ml_scraper  # noqa: E402
from v2.services import ml_user_claims as ml_user_claims_svc  # noqa: E402
from v2.services import ml_user_items as ml_user_items_svc  # noqa: E402
from v2.services import ml_user_promotions as ml_user_promotions_svc  # noqa: E402
from v2.services import ml_user_questions as ml_user_questions_svc  # noqa: E402
from v2.services import ml_account_health as ml_account_health_svc  # noqa: E402
from v2.services import ml_visits as ml_visits_svc  # noqa: E402
from v2.storage import positions_storage  # noqa: E402

app.include_router(v2_router)

# APScheduler for background ML token refresh (every 5h < 6h TTL)
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402
import asyncio as _asyncio  # noqa: E402
import logging as _logging  # noqa: E402

_ml_scheduler: AsyncIOScheduler | None = None
_ml_log = _logging.getLogger("ml-oauth-scheduler")


async def _refresh_ml_tokens_job() -> None:
    """Wrapper that resolves the pool lazily at fire time."""
    pool = await get_pool()
    result = await ml_oauth_svc.refresh_all_expiring_tokens(pool)
    _ml_log.info(
        "ML token refresh tick: refreshed=%s failed=%s",
        result.get("refreshed"), result.get("failed"),
    )


async def _dispatch_questions_to_tg_job() -> None:
    """Send new UNANSWERED questions to seller's Telegram with AI-suggested
    replies and inline action buttons. Cron-driven (default 5 min)."""
    try:
        pool = await get_pool()
        if pool is None:
            _ml_log.warning("Questions dispatch tick skipped: no DB pool")
            return
        result = await ml_questions_dispatch_svc.dispatch_all_users(pool)
        _ml_log.info(
            "Questions dispatch tick: users=%s sent=%s",
            result.get("users"), result.get("sent"),
        )
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("Questions dispatch job failed: %s", err)


async def _sync_ml_notices_job() -> None:
    """Pull /communications/notices for every user with a live ML token,
    upsert into Railway Postgres ml_notices, dispatch new ones to Telegram."""
    try:
        pool = await get_pool()
        if pool is None:
            _ml_log.warning("ML notices tick skipped: no DB pool")
            return
        result = await ml_notices_svc.sync_all_users_notices(pool)
        _ml_log.info(
            "ML notices tick: users=%s fetched=%s saved=%s tg_sent=%s",
            result.get("users"), result.get("fetched"),
            result.get("saved"), result.get("sent"),
        )
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("ML notices job failed: %s", err)


async def _backfill_all_users_job() -> None:
    """Daily catch-up: pull last 24h from orders/questions/claims/items/messages
    for every ML-connected user. Closes gaps if webhook delivery failed."""
    try:
        pool = await get_pool()
        if pool is None:
            _ml_log.warning("ML backfill tick skipped: no DB pool")
            return
        result = await ml_backfill_svc.backfill_all_users(pool, days=1)
        _ml_log.info(
            "ML backfill tick: users=%s fetched=%s saved=%s",
            result.get("users"), result.get("fetched"), result.get("saved"),
        )
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("ML backfill job failed: %s", err)


async def _dispatch_pending_telegram_job() -> None:
    """Drain Telegram queue for every user with pending notices.

    Independent of /communications/notices fetch (which runs every 5 min) —
    webhook writes orders/questions/claims/items in real time, this cron just
    empties the outbox without burning ML quota. Runs every 2 min so end-to-end
    latency from "ML event" → "TG message" is at most ~2 min, regardless of
    whether the user has the site open.
    """
    try:
        pool = await get_pool()
        if pool is None:
            _ml_log.warning("TG dispatch tick skipped: no DB pool")
            return
        result = await ml_notices_svc.dispatch_all_pending(pool)
        _ml_log.info(
            "TG dispatch tick: users=%s sent=%s",
            result.get("users"), result.get("sent"),
        )
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("TG dispatch job failed: %s", err)


async def _refresh_promotions_job() -> None:
    """Scan ML promo offers per user every 30 min. New `candidate` offers
    get pushed to ml_notices (topic='promotions') so the existing TG dispatch
    cron emits Aceitar/Rejeitar inline buttons.

    This is independent of the seller opening any UI — server-driven.
    """
    try:
        pool = await get_pool()
        if pool is None:
            _ml_log.warning("Promotions refresh tick skipped: no DB pool")
            return
        await ml_user_promotions_svc.ensure_schema(pool)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
            )
        user_ids = [r["user_id"] for r in rows]
        total_new = 0
        for uid in user_ids:
            try:
                result = await ml_user_promotions_svc.refresh_user_promotions(pool, uid)
            except Exception as err:  # noqa: BLE001
                _ml_log.exception("promotions refresh user %s failed: %s", uid, err)
                continue
            for offer in result.get("new_offers", []):
                enriched = dict(offer.get("raw") or {})
                enriched["item_id"] = offer["item_id"]
                notice = ml_normalize_svc.normalize_event("promotions", None, enriched)
                notice["notice_id"] = (
                    f"promotions:{offer['item_id']}:{offer['promotion_id']}"
                )
                if await ml_notices_svc.upsert_normalized(pool, uid, notice):
                    total_new += 1
                    await ml_user_promotions_svc.mark_notified(
                        pool, uid, offer["item_id"], offer["promotion_id"],
                    )
            await _asyncio.sleep(0.3)
        _ml_log.info(
            "Promotions refresh tick: users=%s new=%s", len(user_ids), total_new,
        )
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("Promotions refresh job failed: %s", err)


async def _nightly_refresh_all_users_job() -> None:
    """03:00 UTC = 00:00 BRT. Refresh every ML cache for every connected user
    so the morning's first dashboard load reads from DB instantly instead of
    waiting for ml_quality + ml_visits + ml_account_health refreshes (which
    can total 30-60s for a seller with 100+ items).

    Heavy job — runs once a day. Per-user budget: ~30s + ML rate-limit sleeps.
    """
    try:
        pool = await get_pool()
        if pool is None:
            _ml_log.warning("Nightly refresh skipped: no DB pool")
            return
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
            )
        user_ids = [r["user_id"] for r in rows]
        if not user_ids:
            return
        # Make sure all schemas exist (cheap, idempotent).
        await ml_account_health_svc.ensure_schema(pool)
        await ml_user_items_svc.ensure_schema(pool)
        await ml_quality_svc.ensure_schema(pool)
        await ml_visits_svc.ensure_schema(pool)
        await ml_user_questions_svc.ensure_schema(pool)
        await ml_user_claims_svc.ensure_schema(pool)
        await ml_user_promotions_svc.ensure_schema(pool)

        for uid in user_ids:
            try:
                await ml_account_health_svc.refresh_user_health(pool, uid)
            except Exception as err:  # noqa: BLE001
                _ml_log.warning("nightly health user %s: %s", uid, err)
            try:
                await ml_user_items_svc.refresh_user_items(pool, uid, status="active")
            except Exception as err:  # noqa: BLE001
                _ml_log.warning("nightly items user %s: %s", uid, err)
            # Get item ids for downstream caches.
            try:
                async with pool.acquire() as conn:
                    item_rows = await conn.fetch(
                        "SELECT item_id FROM ml_user_items WHERE user_id = $1 AND status = 'active'",
                        uid,
                    )
                ids = [r["item_id"] for r in item_rows]
            except Exception as err:  # noqa: BLE001
                _ml_log.warning("nightly items list user %s: %s", uid, err)
                ids = []
            if ids:
                try:
                    await ml_quality_svc.refresh_user_quality(pool, uid, ids, limit=500)
                except Exception as err:  # noqa: BLE001
                    _ml_log.warning("nightly quality user %s: %s", uid, err)
                try:
                    await ml_visits_svc.refresh_user_visits(pool, uid, ids, limit=500)
                except Exception as err:  # noqa: BLE001
                    _ml_log.warning("nightly visits user %s: %s", uid, err)
                try:
                    await ml_user_promotions_svc.refresh_user_promotions(
                        pool, uid, item_ids=ids,
                    )
                except Exception as err:  # noqa: BLE001
                    _ml_log.warning("nightly promotions user %s: %s", uid, err)
            try:
                await ml_user_questions_svc.refresh_user_questions(pool, uid)
            except Exception as err:  # noqa: BLE001
                _ml_log.warning("nightly questions user %s: %s", uid, err)
            try:
                await ml_user_claims_svc.refresh_user_claims(pool, uid)
            except Exception as err:  # noqa: BLE001
                _ml_log.warning("nightly claims user %s: %s", uid, err)
            await _asyncio.sleep(1.0)
        _ml_log.info("Nightly refresh tick complete: users=%s", len(user_ids))
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("Nightly refresh job failed: %s", err)


@app.on_event("startup")
async def _v2_startup() -> None:
    global _ml_scheduler
    pool = await create_pool()

    # Ensure ML OAuth tables exist on boot (idempotent).
    if pool is not None:
        try:
            await ml_oauth_svc.ensure_schema(pool)
        except Exception as err:  # noqa: BLE001
            _ml_log.exception("ML OAuth schema bootstrap failed: %s", err)
        try:
            await positions_storage.ensure_schema(pool)
        except Exception as err:  # noqa: BLE001
            _ml_log.exception("Positions schema bootstrap failed: %s", err)
        try:
            await ml_notices_svc.ensure_schema(pool)
        except Exception as err:  # noqa: BLE001
            _ml_log.exception("ML notices schema bootstrap failed: %s", err)
        try:
            await ml_item_context_svc.ensure_schema(pool)
        except Exception as err:  # noqa: BLE001
            _ml_log.exception("ML item context schema bootstrap failed: %s", err)

    # Spin up the headless Chromium used by /escalar/positions scraper.
    # Failure here is logged but non-fatal — the scraper self-heals on first use.
    try:
        await ml_scraper.init()
    except Exception as err:  # noqa: BLE001
        _ml_log.exception("ml_scraper init failed: %s", err)

    # Start APScheduler for periodic token refresh.
    _ml_scheduler = AsyncIOScheduler()
    _ml_scheduler.add_job(
        _refresh_ml_tokens_job,
        "interval",
        hours=5,
        id="ml_token_refresh",
        replace_existing=True,
    )
    # /communications/notices sync (Phase 1 — ML anouncements: billing, policies).
    # Usually empty for most sellers. Runs slowly as it's a secondary source.
    # ML «communications/notices» + Telegram. Default 5m matches product expectation;
    # set NOTICES_SYNC_INTERVAL_MIN=1440 (etc.) on Railway if you need less API traffic.
    _notices_interval = int(os.environ.get("NOTICES_SYNC_INTERVAL_MIN", "5"))
    _ml_scheduler.add_job(
        _sync_ml_notices_job,
        "interval",
        minutes=_notices_interval,
        id="ml_notices_sync",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    # Dispatch-only cron — drains TG queue every 2 min regardless of fetch
    # status. Webhook writes notices in real time, this empties the outbox so
    # the user gets pings within 2 min without any UI interaction.
    _dispatch_interval = int(os.environ.get("TG_DISPATCH_INTERVAL_MIN", "2"))
    _ml_scheduler.add_job(
        _dispatch_pending_telegram_job,
        "interval",
        minutes=_dispatch_interval,
        id="tg_dispatch_pending",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    # Questions auto-dispatch to Telegram with AI suggestion + inline buttons.
    # Default 5 min — gives seller near-realtime alerts on new buyer questions.
    _qa_interval = int(os.environ.get("QUESTIONS_TG_INTERVAL_MIN", "5"))
    _ml_scheduler.add_job(
        _dispatch_questions_to_tg_job,
        "interval",
        minutes=_qa_interval,
        id="questions_tg_dispatch",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    # Daily backfill from orders/questions/claims/items/messages (Phase 2 — catch-up
    # for webhook gaps). Webhook is the primary real-time source; this is the safety net.
    _backfill_interval_hours = float(os.environ.get("BACKFILL_INTERVAL_HOURS", "24"))
    _ml_scheduler.add_job(
        _backfill_all_users_job,
        "interval",
        hours=_backfill_interval_hours,
        id="ml_backfill_daily",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    # Promotions discovery — sweep ML for new offers per user, push new
    # candidates to ml_notices for TG dispatch. 30-min default. Independent
    # of seller opening UI — purely server-driven.
    _promo_interval = int(os.environ.get("PROMOTIONS_REFRESH_INTERVAL_MIN", "30"))
    _ml_scheduler.add_job(
        _refresh_promotions_job,
        "interval",
        minutes=_promo_interval,
        id="ml_promotions_refresh",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    # Nightly refresh — at 03:00 UTC (= 00:00 BRT), pre-warm every cache for
    # every connected user so the morning's first dashboard load reads from
    # DB instantly. Heavy but runs once a day.
    _ml_scheduler.add_job(
        _nightly_refresh_all_users_job,
        CronTrigger(hour=3, minute=0, timezone="UTC"),
        id="nightly_refresh_all_caches",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    _ml_scheduler.start()
    _ml_log.info(
        "ML schedulers started: token_refresh=5h, notices_sync=%dm, "
        "tg_dispatch=%dm, backfill=%.2fh, promotions_refresh=%dm, "
        "nightly_refresh=03:00 UTC",
        _notices_interval, _dispatch_interval, _backfill_interval_hours,
        _promo_interval,
    )

    # Kick once on boot so tokens that expired during downtime get refreshed immediately.
    _asyncio.create_task(_refresh_ml_tokens_job())
    # Also kick notices sync — catches anything missed while API was down.
    _asyncio.create_task(_sync_ml_notices_job())
    # And drain pending TG queue right away — no waiting for the first 2-min tick.
    _asyncio.create_task(_dispatch_pending_telegram_job())


@app.on_event("shutdown")
async def _v2_shutdown() -> None:
    global _ml_scheduler
    if _ml_scheduler is not None:
        try:
            _ml_scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
        _ml_scheduler = None
    try:
        await ml_scraper.close()
    except Exception:  # noqa: BLE001
        pass
    await close_pool()

# ── Database ──
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    if not DATABASE_URL:
        return
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            source_key TEXT,
            source_name TEXT,
            source_type TEXT,
            rows INTEGER,
            period TEXT,
            result JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_upload(filename: str, result: dict):
    if not DATABASE_URL:
        return
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO uploads (filename, source_key, source_name, source_type, rows, period, result)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                filename,
                result.get("source_key"),
                result.get("source_name"),
                result.get("source_type"),
                result.get("rows"),
                result.get("period"),
                json.dumps(result, ensure_ascii=False),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


@app.on_event("startup")
def on_startup():
    init_db()

# ── Source labels ──
SOURCE_LABELS = {
    "vendas_ml": {"name": "Vendas ML", "type": "ecom"},
    "collection_mp": {"name": "Collection MP", "type": "ecom"},
    "extrato_mp": {"name": "Extrato Mercado Pago", "type": "ecom"},
    "fatura_ml": {"name": "Fatura ML", "type": "ecom"},
    "ads_publicidade": {"name": "Anúncios Patrocinados", "type": "ecom"},
    "armazenagem_full": {"name": "Custos Armazenagem Full", "type": "ecom"},
    "stock_full": {"name": "Stock ML Full", "type": "ecom"},
    "after_collection": {"name": "Pós-vendas", "type": "ecom"},
    "extrato_nubank": {"name": "Extrato Nubank", "type": "bank"},
    "extrato_c6_brl": {"name": "Extrato C6 BRL", "type": "bank"},
    "extrato_c6_usd": {"name": "Extrato C6 USD", "type": "bank"},
    "trafficstars": {"name": "TrafficStars", "type": "expense"},
    "bybit_history": {"name": "Bybit History", "type": "crypto"},
    "das_simples": {"name": "DAS Simples Nacional", "type": "tax"},
    "nfse_shps": {"name": "NFS-e (SHPS)", "type": "invoice"},
    "full_express": {"name": "Full Express (3PL)", "type": "3pl"},
}


def try_read_csv(file_bytes: bytes, nrows: int | None = None) -> pd.DataFrame | None:
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        for sep in [";", ",", "\t"]:
            for skip in [0, 5, 1, 2, 3, 4]:
                try:
                    kw = {"nrows": nrows} if nrows else {}
                    df = pd.read_csv(
                        io.BytesIO(file_bytes), sep=sep, skiprows=skip,
                        encoding=enc, **kw,
                    )
                    if len(df.columns) > 2:
                        return df
                except Exception:
                    continue
    return None


def auto_detect_source(df: pd.DataFrame, filename: str) -> str | None:
    cols = set(c.strip().lower() for c in df.columns) if len(df.columns) > 0 else set()
    fname = filename.lower()

    def has_col(substring):
        return any(substring in c for c in cols)

    # Column-based detection
    if has_col("net_received_amount") or has_col("transaction_amount"):
        return "collection_mp"
    if has_col("# de anúncio") or has_col("# de anuncio"):
        return "vendas_ml"
    if has_col("investimento") and (has_col("acos") or has_col("roas")):
        return "ads_publicidade"
    if has_col("tarifa por unidade"):
        return "armazenagem_full"
    if has_col("amount_refunded") and has_col("shipment_status"):
        return "after_collection"
    if has_col("identificador") and has_col("descrição"):
        return "extrato_nubank"
    if "release_date" in cols or "transaction_net_amount" in cols or "partial_balance" in cols:
        return "extrato_mp"
    if "initial_balance" in cols and "final_balance" in cols:
        return "extrato_mp"
    if "data lançamento" in cols or "data lancamento" in cols:
        if any("r$" in c for c in cols):
            return "extrato_c6_brl"
        if any("us$" in c or "usd" in c for c in cols):
            return "extrato_c6_usd"
        return "extrato_c6_brl"

    # Filename-based detection
    if "collection" in fname:
        return "collection_mp"
    if "account_statement" in fname:
        return "extrato_mp"
    if "vendas" in fname and "mercado" in fname:
        return "vendas_ml"
    if "vendas" in fname:
        return "vendas_ml"
    if "anuncios" in fname or "patrocinados" in fname:
        return "ads_publicidade"
    if "armazenamento" in fname or "armazenagem" in fname:
        return "armazenagem_full"
    if "stock_general" in fname or "stock_full" in fname or fname.startswith("stock"):
        return "stock_full"
    if "after_collection" in fname or "pos" in fname:
        return "after_collection"
    if "fatura" in fname or "faturamento" in fname:
        return "fatura_ml"
    if "extrato" in fname and "nubank" in fname:
        return "extrato_nubank"
    if "c6" in fname:
        if "usd" in fname or "global_usd" in fname:
            return "extrato_c6_usd"
        return "extrato_c6_brl"
    if fname.startswith("01k"):
        return "extrato_c6_brl"
    if "trafficstars" in fname or "traffic" in fname:
        return "trafficstars"
    if "bybit" in fname:
        return "bybit_history"
    if "pgdasd" in fname or "das-" in fname:
        return "das_simples"
    if "nfs" in fname or "nfse" in fname:
        return "nfse_shps"
    return None


def detect_period(df: pd.DataFrame) -> str | None:
    """Try to detect date range from DataFrame."""
    date_cols = []
    for c in df.columns:
        cl = c.strip().lower()
        if any(k in cl for k in ["date", "data", "fecha", "release_date", "created_date"]):
            date_cols.append(c)

    if not date_cols:
        # Try first column if it looks like dates
        for c in df.columns:
            sample = df[c].dropna().head(5).astype(str)
            if sample.str.contains(r"\d{4}[-/]\d{2}", regex=True).any():
                date_cols.append(c)
                break
            if sample.str.contains(r"\d{2}/\d{2}/\d{4}", regex=True).any():
                date_cols.append(c)
                break

    for col in date_cols:
        try:
            dates = pd.to_datetime(df[col], dayfirst=True, errors="coerce").dropna()
            if len(dates) > 0:
                mn = dates.min()
                mx = dates.max()
                months_pt = [
                    "", "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
                ]
                if mn.year == mx.year and mn.month == mx.month:
                    return f"{months_pt[mn.month]} {mn.year}"
                return f"{months_pt[mn.month]}-{months_pt[mx.month]} {mx.year}"
        except Exception:
            continue
    return None


def compute_top_categories(df: pd.DataFrame, source: str) -> list[dict]:
    """Extract top expense/income categories depending on source type."""
    cats = []

    # For extrato_mp — group by 'description' or 'source_id'
    if source == "extrato_mp":
        for col_name in ["description", "source_id", "tipo"]:
            if col_name in [c.strip().lower() for c in df.columns]:
                real_col = [c for c in df.columns if c.strip().lower() == col_name][0]
                val_col = None
                for vc in df.columns:
                    vcl = vc.strip().lower()
                    if any(k in vcl for k in ["amount", "valor", "net", "gross"]):
                        val_col = vc
                        break
                if val_col:
                    try:
                        df[val_col] = pd.to_numeric(
                            df[val_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                            errors="coerce",
                        )
                        grouped = df.groupby(real_col)[val_col].sum().abs().sort_values(ascending=False).head(5)
                        for cat, val in grouped.items():
                            cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
                    except Exception:
                        pass
                break

    # For vendas_ml — by SKU or title
    if source == "vendas_ml":
        for col_name in ["sku", "título do anúncio", "titulo do anuncio"]:
            matched = [c for c in df.columns if c.strip().lower() == col_name]
            if matched:
                val_col = None
                for vc in df.columns:
                    vcl = vc.strip().lower()
                    if "receita" in vcl or "net" in vcl or "valor" in vcl:
                        val_col = vc
                        break
                if val_col:
                    try:
                        df[val_col] = pd.to_numeric(
                            df[val_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                            errors="coerce",
                        )
                        grouped = df.groupby(matched[0])[val_col].sum().abs().sort_values(ascending=False).head(5)
                        for cat, val in grouped.items():
                            cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
                    except Exception:
                        pass
                break

    # For fatura_ml — by tariff type
    if source == "fatura_ml":
        val_cols = [c for c in df.columns if any(
            k in c.strip().lower() for k in ["tarifa", "comiss", "valor"]
        )]
        for vc in val_cols[:3]:
            try:
                total = pd.to_numeric(
                    df[vc].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                ).sum()
                if abs(total) > 0:
                    cats.append({"category": vc.strip()[:40], "value": round(abs(float(total)), 2)})
            except Exception:
                pass
        cats.sort(key=lambda x: x["value"], reverse=True)

    # For bank extracts — group by description
    if source in ("extrato_nubank", "extrato_c6_brl", "extrato_c6_usd"):
        desc_col = None
        val_col = None
        for c in df.columns:
            cl = c.strip().lower()
            if any(k in cl for k in ["descri", "histórico", "historico", "descrição"]):
                desc_col = c
            if any(k in cl for k in ["valor", "saída", "saida", "entrada", "r$", "us$", "amount"]):
                if val_col is None:
                    val_col = c
        if desc_col and val_col:
            try:
                df[val_col] = pd.to_numeric(
                    df[val_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                )
                grouped = df.groupby(desc_col)[val_col].sum().abs().sort_values(ascending=False).head(5)
                for cat, val in grouped.items():
                    cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
            except Exception:
                pass

    # Generic fallback — find any numeric column and group by first text column
    if not cats:
        text_col = None
        num_col = None
        for c in df.columns:
            if text_col is None and df[c].dtype == "object":
                text_col = c
            if num_col is None:
                try:
                    vals = pd.to_numeric(
                        df[c].astype(str).str.replace(",", ".").str.replace(" ", ""),
                        errors="coerce",
                    )
                    if vals.notna().sum() > len(df) * 0.5:
                        num_col = c
                except Exception:
                    pass
        if text_col and num_col:
            try:
                df[num_col] = pd.to_numeric(
                    df[num_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                    errors="coerce",
                )
                grouped = df.groupby(text_col)[num_col].sum().abs().sort_values(ascending=False).head(5)
                for cat, val in grouped.items():
                    cats.append({"category": str(cat)[:40], "value": round(float(val), 2)})
            except Exception:
                pass

    return cats[:5]


@app.get("/health")
async def health():
    return {"status": "ok"}


def parse_brl(val) -> float:
    """Parse BRL value: '1.234,56' or '1234.56' → float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip()
    if not s or s in ("nan", "NaN", "None", ""):
        return 0.0
    s = s.replace("R$", "").replace("$", "").replace(" ", "")
    # BRL format: 1.234,56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def build_vendas_ml_opiu(df: pd.DataFrame) -> dict:
    """Build mini P&L (OPiU) from Vendas ML DataFrame."""
    receita_bruta = 0.0
    tarifa_venda = 0.0
    receita_envio = 0.0
    tarifa_envio = 0.0
    cancelamentos = 0.0
    total_net = 0.0
    vendas_count = 0
    ads_count = 0

    # Detect column names (may vary with accents)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "preço unitário" in cl or "preco unitario" in cl:
            col_map["preco"] = c
        if "receita por produtos" in cl:
            col_map["receita_prod"] = c
        if "tarifa de venda" in cl:
            col_map["tarifa_venda"] = c
        if "receita por envio" in cl:
            col_map["receita_envio"] = c
        if "tarifas de envio" in cl:
            col_map["tarifa_envio"] = c
        if "cancelamentos" in cl:
            col_map["cancelamentos"] = c
        if cl == "total (brl)":
            col_map["total"] = c
        if cl == "unidades":
            col_map["unidades"] = c
        if "venda por publicidade" in cl:
            col_map["ads"] = c

    for _, row in df.iterrows():
        # Receita bruta: Preço unitário × Unidades, or Receita por produtos
        if "preco" in col_map and "unidades" in col_map:
            preco = parse_brl(row.get(col_map["preco"], 0))
            try:
                u = row.get(col_map["unidades"], 0)
                if pd.isna(u) or u == "":
                    unidades = 0
                else:
                    unidades = int(float(str(u).strip()))
            except (ValueError, TypeError):
                unidades = 0
            receita_bruta += preco * unidades
        elif "receita_prod" in col_map:
            receita_bruta += parse_brl(row.get(col_map["receita_prod"], 0))
        tarifa_venda += parse_brl(row.get(col_map.get("tarifa_venda", "Tarifa de venda e impostos (BRL)"), 0))
        receita_envio += parse_brl(row.get(col_map.get("receita_envio", "Receita por envio (BRL)"), 0))
        tarifa_envio += parse_brl(row.get(col_map.get("tarifa_envio", "Tarifas de envio (BRL)"), 0))
        cancelamentos += parse_brl(row.get(col_map.get("cancelamentos", "Cancelamentos e reembolsos (BRL)"), 0))
        total_net += parse_brl(row.get(col_map.get("total", "Total (BRL)"), 0))
        vendas_count += 1
        ads_val = str(row.get(col_map.get("ads", "Venda por publicidade"), "")).strip().lower()
        if ads_val == "sim":
            ads_count += 1

    return {
        "receita_bruta": round(receita_bruta, 2),
        "tarifa_venda": round(abs(tarifa_venda), 2),
        "receita_envio": round(receita_envio, 2),
        "tarifa_envio": round(abs(tarifa_envio), 2),
        "cancelamentos": round(abs(cancelamentos), 2),
        "total_net": round(total_net, 2),
        "vendas_count": vendas_count,
        "ads_count": ads_count,
        "ads_pct": round(ads_count / vendas_count * 100, 1) if vendas_count > 0 else 0,
    }


def detect_vendas_period(df: pd.DataFrame) -> str | None:
    """Detect period from 'Data da venda' column (format: '24 de março de 2026 22:36 hs.')."""
    import re
    pt_months = {
        "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
        "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
        "agosto": "Ago", "setembro": "Set", "outubro": "Out",
        "novembro": "Nov", "dezembro": "Dez",
    }
    col = None
    for c in df.columns:
        if "data da venda" in c.strip().lower() or "data" in c.strip().lower():
            col = c
            break
    if col is None:
        return None

    months_found = set()
    year = None
    for val in df[col].dropna().astype(str):
        m = re.match(r'(\d{1,2}) de (\w+) de (\d{4})', val.strip())
        if m:
            mon_pt = m.group(2).lower()
            year = m.group(3)
            short = pt_months.get(mon_pt)
            if short:
                months_found.add(short)

    if not months_found:
        return None

    order = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    sorted_m = sorted(months_found, key=lambda x: order.index(x) if x in order else 99)
    if len(sorted_m) == 1:
        return f"{sorted_m[0]} {year}"
    return f"{sorted_m[0]}–{sorted_m[-1]} {year}"


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Accept a file upload, detect its type, and return analysis summary."""
    try:
        file_bytes = await file.read()
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Try to read as CSV (with multiple skiprows for vendas_ml format)
        df = None
        if ext in ("csv", "txt", "tsv", ""):
            df = try_read_csv(file_bytes, nrows=5)

        # Try xlsx
        if df is None and ext in ("xlsx", "xls"):
            try:
                df = pd.read_excel(io.BytesIO(file_bytes), nrows=5)
            except Exception:
                pass

        if df is None or df.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not parse file", "filename": filename},
            )

        # Detect source
        source = auto_detect_source(df, filename)

        # Read full file
        df_full = None
        if ext in ("csv", "txt", "tsv", ""):
            df_full = try_read_csv(file_bytes)
        elif ext in ("xlsx", "xls"):
            try:
                df_full = pd.read_excel(io.BytesIO(file_bytes))
            except Exception:
                df_full = df

        if df_full is None:
            df_full = df

        source_info = SOURCE_LABELS.get(source, {"name": source or "Desconhecido", "type": "unknown"})

        # Build response based on source type
        result = {
            "filename": filename,
            "source_key": source,
            "source_name": source_info["name"],
            "source_type": source_info["type"],
            "rows": len(df_full),
            "columns": len(df_full.columns),
        }

        # Special handling for vendas_ml — build OPiU
        if source == "vendas_ml":
            opiu = build_vendas_ml_opiu(df_full)
            result["opiu"] = opiu
            result["period"] = detect_vendas_period(df_full)
            result["top_categories"] = [
                {"category": "Receita Bruta", "value": opiu["receita_bruta"]},
                {"category": "Tarifa de Venda", "value": opiu["tarifa_venda"]},
                {"category": "Tarifa de Envio", "value": opiu["tarifa_envio"]},
                {"category": "Cancelamentos", "value": opiu["cancelamentos"]},
                {"category": "Total NET", "value": abs(opiu["total_net"])},
            ]
        else:
            result["period"] = detect_period(df_full)
            result["top_categories"] = compute_top_categories(df_full, source or "")

        # Save to DB
        save_upload(filename, result)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "filename": file.filename},
        )


@app.get("/uploads")
async def list_uploads(limit: int = 20):
    """Return recent upload history from DB."""
    if not DATABASE_URL:
        return []
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, filename, source_name, source_type, rows, period, created_at
               FROM uploads ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "id": r[0], "filename": r[1], "source_name": r[2],
                "source_type": r[3], "rows": r[4], "period": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/uploads/{upload_id}")
async def get_upload(upload_id: int):
    """Return full result of a specific upload."""
    if not DATABASE_URL:
        return JSONResponse(status_code=404, content={"error": "No DB"})
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT result, created_at FROM uploads WHERE id = %s", (upload_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Not found"})
        result = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        result["created_at"] = row[1].isoformat() if row[1] else None
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
