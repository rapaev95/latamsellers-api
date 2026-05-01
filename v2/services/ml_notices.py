"""ML notices sync job: pull /communications/notices per user → Railway Postgres → Telegram.

Reads and writes the SAME Railway Postgres that Next.js Escalar reads from:
- ml_user_tokens         (source of per-user OAuth tokens — owned by ml_oauth.py)
- ml_notices             (target: content + dedup + TG dispatch state)
- notification_settings  (per-user TG chat_id + language + notify_ml_news flag)

Token refresh is delegated to ml_oauth.get_valid_access_token() so we share
the same refresh-margin logic as the on-demand `/access-token` endpoint.

Idempotent by (user_id, notice_id): re-runs only touch mutable columns.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc
from . import telegram_notify as tg_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
ML_RATE_LIMIT_SLEEP = 0.05         # 50 req/sec app-wide → 50ms between user calls
TG_MESSAGE_THROTTLE = 1.1          # Telegram: 1 msg/sec per chat, 1.1 for safety
NOTICES_PAGE_LIMIT = 50
NOTICES_MAX_PAGES = 8              # up to 400 per run per user
TG_BATCH_CAP = 20                  # cap per-user TG sends per tick

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NEWS_DIGEST_MODEL = "anthropic/claude-sonnet-4.5"
# topics created by webhook ingest — rich notice with own dispatch path,
# no AI digest needed. Anything outside this set + with HTML content
# is treated as ML news / generic platform notice → run digest.
_WEBHOOK_TOPICS = frozenset({
    "orders", "orders_v2", "questions", "questions_v2", "claims",
    "items", "messages", "promotions", "public_offers", "public_candidates",
})


# ── Schema bootstrap ──────────────────────────────────────────────────────────

CREATE_NOTICES_SQL = """
CREATE TABLE IF NOT EXISTS ml_notices (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  notice_id TEXT NOT NULL,
  label TEXT,
  description TEXT,
  from_date TIMESTAMPTZ,
  tags JSONB DEFAULT '[]'::jsonb,
  actions JSONB DEFAULT '[]'::jsonb,
  raw JSONB,
  topic TEXT,
  resource TEXT,
  first_seen_at TIMESTAMPTZ DEFAULT NOW(),
  read_at TIMESTAMPTZ,
  telegram_sent_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, notice_id)
);
ALTER TABLE ml_notices ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE ml_notices ADD COLUMN IF NOT EXISTS resource TEXT;
ALTER TABLE ml_notices ADD COLUMN IF NOT EXISTS ai_digest_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_ml_notices_user_unread ON ml_notices(user_id, read_at);
CREATE INDEX IF NOT EXISTS idx_ml_notices_user_tg_pending ON ml_notices(user_id) WHERE telegram_sent_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_ml_notices_from_date ON ml_notices(user_id, from_date DESC);
CREATE INDEX IF NOT EXISTS idx_ml_notices_user_topic ON ml_notices(user_id, topic);

CREATE TABLE IF NOT EXISTS notification_settings (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL UNIQUE,
  telegram_chat_id TEXT,
  notify_daily_sales BOOLEAN DEFAULT TRUE,
  notify_acos_change BOOLEAN DEFAULT TRUE,
  notify_ml_news BOOLEAN DEFAULT TRUE,
  acos_threshold NUMERIC DEFAULT 5,
  language TEXT DEFAULT 'pt',
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_NOTICES_SQL)


# ── ML API ────────────────────────────────────────────────────────────────────

async def _fetch_notices(http: httpx.AsyncClient, access_token: str) -> list[dict[str, Any]]:
    """Page through /communications/notices until exhausted (cap NOTICES_MAX_PAGES)."""
    out: list[dict[str, Any]] = []
    offset = 0
    headers = {"Authorization": f"Bearer {access_token}"}
    for _ in range(NOTICES_MAX_PAGES):
        try:
            r = await http.get(
                f"{ML_API_BASE}/communications/notices",
                params={"limit": NOTICES_PAGE_LIMIT, "offset": offset},
                headers=headers,
                timeout=20.0,
            )
        except Exception as err:  # noqa: BLE001
            log.warning("notices fetch error: %s", err)
            break
        if r.status_code != 200:
            log.warning("notices %s: %s", r.status_code, r.text[:200])
            break
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            break
        out.extend(results)
        if len(results) < NOTICES_PAGE_LIMIT:
            break
        offset += NOTICES_PAGE_LIMIT
        await asyncio.sleep(ML_RATE_LIMIT_SLEEP)
    return out


# ── Sync one user ─────────────────────────────────────────────────────────────

import os as _os
import re as _re

_HTML_TAG_RE = _re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = _re.compile(r"&(nbsp|amp|lt|gt|quot|#\d+);")
_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
}


def _strip_html(text: str) -> str:
    """Remove HTML tags + decode common entities. ML news arrive with <p>,
    <strong>, <ul>/<li>, etc. that look ugly in TG."""
    if not text:
        return ""
    out = _HTML_TAG_RE.sub("", text)
    for ent, repl in _HTML_ENTITIES.items():
        out = out.replace(ent, repl)
    out = _HTML_ENTITY_RE.sub("", out)
    out = _re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_DIGEST_PROMPT = """Você é um analista de notificações de marketplace. Recebe um aviso oficial do Mercado Livre Brasil dirigido ao vendedor (em qualquer idioma) e produz um BRIEFING ULTRA-CURTO em {lang_label}.

Formato OBRIGATÓRIO:
*<emoji> <título-resumo em até 60 chars>*

📌 *Significado para o seller:*
<1-2 frases concretas — o que ESPECIFICAMENTE muda para este vendedor>

✅ *Ação recomendada:*
• <ação 1 — verbo no imperativo>
• <ação 2 — só se realmente necessário>
• (no máximo 3 ações; uma só já basta se uma resolve)

⏰ *Prazo / impacto:* <data ou "imediato" / "sem prazo">

REGRAS:
- Total ≤ 600 caracteres. Cortar tudo que for marketing fluff ("aproveite", "maximize", "destaque-se").
- Se o aviso é puramente promocional/educativo sem ação requerida — diga isso explicitamente em "Ação recomendada: nenhuma ação necessária".
- NÃO inventar números/datas que não estão no aviso.
- Output APENAS o briefing, sem prefixos tipo "Aqui está:" ou "Briefing:".
- Use markdown simples (negrito *texto*) para Telegram, sem code blocks."""

_DIGEST_LANG_LABELS = {
    "ru": "russo",
    "en": "inglês",
    "pt": "português brasileiro",
    "es": "espanhol",
}


async def _make_news_digest(
    http: httpx.AsyncClient,
    label: str,
    description: str,
    language: str = "pt",
    pool: asyncpg.Pool | None = None,
    user_id: int | None = None,
) -> str | None:
    """Run Claude Sonnet 4.5 over a raw ML notice → short actionable briefing.

    Returns None on failure (caller falls back to raw stripped HTML).
    """
    api_key = _os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    cleaned = _strip_html(description)
    if not cleaned and not label:
        return None
    lang_label = _DIGEST_LANG_LABELS.get((language or "pt").lower(), "português brasileiro")
    user_msg = (
        f"TÍTULO: {label or '(sem título)'}\n\n"
        f"CONTEÚDO:\n{cleaned[:2000]}"
    )
    try:
        r = await http.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://app.lsprofit.app",
                "X-Title": "LS Profit App",
            },
            json={
                "model": NEWS_DIGEST_MODEL,
                "max_tokens": 350,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": _DIGEST_PROMPT.format(lang_label=lang_label)},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=30.0,
        )
        if r.status_code != 200:
            log.warning("news digest %s: %s", r.status_code, r.text[:200])
            try:
                from . import tg_admin_alerts as _alerts
                await _alerts.alert_openrouter_failure(
                    r.status_code, r.text, service="news/digest",
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                from . import ai_usage_tracker as _tracker
                await _tracker.log_call(
                    pool, user_id=user_id, service="news/digest",
                    model=NEWS_DIGEST_MODEL, response_data=None, status_code=r.status_code,
                )
            except Exception:  # noqa: BLE001
                pass
            return None
        data = r.json()
        try:
            from . import ai_usage_tracker as _tracker
            await _tracker.log_call(
                pool, user_id=user_id, service="news/digest",
                model=NEWS_DIGEST_MODEL, response_data=data, status_code=200,
            )
        except Exception:  # noqa: BLE001
            pass
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return content.strip() if isinstance(content, str) else None
    except Exception as err:  # noqa: BLE001
        log.warning("news digest exception: %s", err)
        return None


def _needs_news_digest(topic: str | None, description: str | None) -> bool:
    """Skip digest for webhook-driven topics (their own dispatchers handle
    rich rendering). Run digest if any HTML or content suggests a marketing
    blast."""
    if topic and topic.lower() in _WEBHOOK_TOPICS:
        return False
    text = description or ""
    if not text or len(text) < 80:
        return False
    if "<p>" in text or "</p>" in text or "<strong>" in text or "<ul>" in text:
        return True
    return False


def _coerce_to_datetime(v: Any) -> Any:
    """asyncpg для TIMESTAMPTZ принимает только datetime instance — не string.
    normalize_event иногда возвращает from_date как ISO string (приходит из
    ML payload `date_created` без парсинга). Конвертируем здесь, чтобы
    upsert не падал «expected datetime instance, got 'str'».
    """
    if v is None or isinstance(v, datetime):
        return v
    if isinstance(v, str) and v.strip():
        s = v.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


async def upsert_normalized(
    pool: asyncpg.Pool,
    user_id: int,
    notice: dict[str, Any],
) -> bool:
    """Upsert a single normalize_event() output row into ml_notices.

    Used by cron jobs that synthesize notices from non-webhook sources (e.g.
    promotions discovered by ml_user_promotions.refresh). Returns True if a
    new row was inserted, False if it already existed (still updates content).
    """
    nid = str(notice.get("notice_id") or "")
    if not nid:
        return False
    async with pool.acquire() as conn:
        existed = await conn.fetchval(
            "SELECT 1 FROM ml_notices WHERE user_id = $1 AND notice_id = $2",
            user_id, nid,
        )
        await conn.execute(
            """
            INSERT INTO ml_notices
              (user_id, notice_id, label, description, from_date,
               tags, actions, raw, topic, resource, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, NOW())
            ON CONFLICT (user_id, notice_id) DO UPDATE SET
              label = EXCLUDED.label,
              description = EXCLUDED.description,
              from_date = EXCLUDED.from_date,
              tags = EXCLUDED.tags,
              actions = EXCLUDED.actions,
              raw = EXCLUDED.raw,
              topic = EXCLUDED.topic,
              resource = EXCLUDED.resource,
              updated_at = NOW()
            """,
            user_id,
            nid,
            notice.get("label"),
            notice.get("description"),
            _coerce_to_datetime(notice.get("from_date")),
            json.dumps(notice.get("tags") or [], default=str),
            json.dumps(notice.get("actions") or [], default=str),
            json.dumps(notice.get("raw") or {}, default=str),
            notice.get("topic"),
            notice.get("resource"),
        )
    return existed is None


async def _upsert_notices(conn: asyncpg.Connection, user_id: int, notices: list[dict[str, Any]]) -> int:
    saved = 0
    for n in notices:
        nid = n.get("id")
        if not nid:
            continue
        await conn.execute(
            """
            INSERT INTO ml_notices
              (user_id, notice_id, label, description, from_date, tags, actions, raw, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, NOW())
            ON CONFLICT (user_id, notice_id) DO UPDATE SET
              label = EXCLUDED.label,
              description = EXCLUDED.description,
              from_date = EXCLUDED.from_date,
              tags = EXCLUDED.tags,
              actions = EXCLUDED.actions,
              raw = EXCLUDED.raw,
              updated_at = NOW()
            """,
            user_id,
            str(nid),
            n.get("label"),
            n.get("description"),
            _coerce_to_datetime(n.get("from_date")),
            json.dumps(n.get("tags") or []),
            json.dumps(n.get("actions") or []),
            json.dumps(n),
        )
        saved += 1
    return saved


async def _dispatch_to_telegram(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
) -> int:
    """Send any not-yet-sent notices to this user's Telegram, if enabled."""
    async with pool.acquire() as conn:
        settings = await conn.fetchrow(
            """
            SELECT telegram_chat_id, notify_ml_news, COALESCE(language, 'pt') AS language
              FROM notification_settings
             WHERE user_id = $1
            """,
            user_id,
        )
        if not settings or not settings["notify_ml_news"] or not settings["telegram_chat_id"]:
            return 0
        chat_id = str(settings["telegram_chat_id"])
        language = settings["language"] or "pt"

        # Skip low-value platform notices (invoices, price_suggestion, etc.) —
        # they come from /communications/notices and flood the chat without
        # useful context. Only send real seller events (orders/questions/
        # claims/items/messages) + anything with a topic webhook topic attached.
        NOISE_PREFIXES = (
            "invoices:",
            "price_suggestion:",
            "payments:",
            "stock-locations:",
            "shipments:",
            "fbm_stock_operations:",
            "catalog_item_competition",
            "catalog_suggestions",
            # Promotions webhook events: useful for triggering refresh
            # (so ml_user_promotions cache + topic=promotions notice get
            # produced), but the events themselves carry only an internal
            # OFFER-/CANDIDATE- id, not the actionable promotion_id. Mark
            # silent — the seller gets ONE rich promotions: notice with
            # Aceitar/Rejeitar buttons instead of a stub here.
            "public_offers:",
            "public_candidates:",
        )
        # First pass: bulk-mark noisy notices as sent so they drop out of the
        # queue without consuming a TG send slot. Idempotent — once cleared,
        # future backfills won't duplicate them.
        noise_where = " OR ".join([f"notice_id LIKE '{p}%'" for p in NOISE_PREFIXES])
        if noise_where:
            await conn.execute(
                f"""
                UPDATE ml_notices
                   SET telegram_sent_at = NOW()
                 WHERE user_id = $1 AND telegram_sent_at IS NULL AND ({noise_where})
                """,
                user_id,
            )

        # Questions and messages are dispatched by their dedicated services
        # (ml_questions_dispatch, ml_messages_dispatch) — both produce rich
        # cards with AI translation/summary + action buttons. Bulk-mark them
        # sent here so the legacy translate-once dispatch doesn't double-
        # notify. The new dispatchers track their own state via separate
        # columns (questions: ml_user_questions.tg_dispatched_at; messages:
        # ml_notices.messages_tg_dispatched_at).
        await conn.execute(
            """
            UPDATE ml_notices
               SET telegram_sent_at = NOW()
             WHERE user_id = $1
               AND telegram_sent_at IS NULL
               AND topic IN ('questions', 'questions_v2', 'messages')
            """,
            user_id,
        )

        pending_rows = await conn.fetch(
            """
            SELECT notice_id, label, description, actions, tags, topic, raw,
                   ai_digest_at
              FROM ml_notices
             WHERE user_id = $1 AND telegram_sent_at IS NULL
             ORDER BY from_date ASC NULLS FIRST
             LIMIT $2
            """,
            user_id, TG_BATCH_CAP,
        )

    sent = 0
    for row in pending_rows:
        # AI digest для платформенных news (без webhook topic, с HTML).
        # Idempotent — уже digested rows имеют ai_digest_at IS NOT NULL.
        description_to_use = row["description"]
        label_to_use = row["label"]
        if (
            row["ai_digest_at"] is None
            and _needs_news_digest(row["topic"], row["description"])
        ):
            digest = await _make_news_digest(
                http, row["label"] or "", row["description"] or "", language,
                pool=pool, user_id=user_id,
            )
            if digest:
                description_to_use = digest
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE ml_notices
                           SET description = $1,
                               ai_digest_at = NOW()
                         WHERE user_id = $2 AND notice_id = $3
                        """,
                        digest, user_id, row["notice_id"],
                    )
            else:
                # Хотя бы strip HTML, чтобы </p> не торчало.
                description_to_use = _strip_html(row["description"] or "")

        notice = {
            "notice_id": row["notice_id"],
            "label": label_to_use,
            "description": description_to_use,
            "actions": row["actions"] or [],
            "tags": row["tags"] or [],
            "topic": row["topic"],
            "raw": row["raw"] or {},
        }
        ok = await tg_svc.send_notice(chat_id, notice, language, http)
        if ok:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ml_notices SET telegram_sent_at = NOW() WHERE user_id = $1 AND notice_id = $2",
                    user_id, row["notice_id"],
                )
            sent += 1
        await asyncio.sleep(TG_MESSAGE_THROTTLE)
    return sent


async def _sync_one_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
) -> dict[str, int]:
    """One tick:
      1. fetch /communications/notices (may be empty for active sellers — normal)
      2. upsert any new notices into ml_notices
      3. dispatch ALL pending rows to Telegram

    Step 3 must run on every tick regardless of step 1 — webhook is the
    primary source of seller events (orders/questions/items/...) and writes
    directly to ml_notices. If we early-returned on empty /communications,
    webhook-saved rows would never get dispatched (production bug 2026-04-25).
    """
    # Delegate refresh to ml_oauth — handles refresh-margin, invalid_grant, etc.
    try:
        access_token, _expires_at, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError as err:
        log.warning("user %s: token refresh failed: %s", user_id, err)
        return {"user_id": user_id, "fetched": 0, "saved": 0, "sent": 0}

    notices = await _fetch_notices(http, access_token)
    saved = 0
    if notices:
        async with pool.acquire() as conn:
            saved = await _upsert_notices(conn, user_id, notices)

    # Always dispatch — there may be pending rows from webhooks.
    sent = await _dispatch_to_telegram(pool, http, user_id)
    return {"user_id": user_id, "fetched": len(notices), "saved": saved, "sent": sent}


# ── Public helpers for schedulers ─────────────────────────────────────────────

async def dispatch_pending_for_user(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    user_id: int,
) -> int:
    """Public wrapper around `_dispatch_to_telegram` for the dispatch-only cron
    job (separate from the fetch loop in sync_all_users_notices)."""
    return await _dispatch_to_telegram(pool, http, user_id)


async def dispatch_all_pending(pool: asyncpg.Pool) -> dict[str, Any]:
    """Drain TG queue for every user with any pending notices.

    Independent of /communications/notices fetch — webhook keeps writing to
    ml_notices in real time, this cron just empties the outbox. Call every
    1-2 min to keep latency low for buyers' questions / new orders / paused
    items without burning ML rate-limits on /communications fetches.
    """
    if pool is None:
        return {"users": 0, "sent": 0}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT n.user_id
              FROM notification_settings n
              JOIN ml_notices m ON m.user_id = n.user_id
                                AND m.telegram_sent_at IS NULL
             WHERE n.notify_ml_news = TRUE
               AND n.telegram_chat_id IS NOT NULL
            """,
        )
    user_ids = [r["user_id"] for r in rows]
    if not user_ids:
        return {"users": 0, "sent": 0}

    total_sent = 0
    async with httpx.AsyncClient() as http:
        for uid in user_ids:
            try:
                sent = await _dispatch_to_telegram(pool, http, uid)
                total_sent += sent
            except Exception as err:  # noqa: BLE001
                log.exception("dispatch_all_pending: user %s failed: %s", uid, err)
            await asyncio.sleep(0.1)
    return {"users": len(user_ids), "sent": total_sent}


# ── Entry point for the APScheduler job ───────────────────────────────────────

async def sync_all_users_notices(pool: asyncpg.Pool) -> dict[str, Any]:
    """Pull notices for every user with stored ML tokens.
    Returns aggregate stats: {users, fetched, saved, sent}."""
    if pool is None:
        log.warning("ml_notices: no pool — skipping tick")
        return {"users": 0, "fetched": 0, "saved": 0, "sent": 0}

    await ensure_schema(pool)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id FROM ml_user_tokens WHERE access_token IS NOT NULL"
        )
    user_ids = [r["user_id"] for r in rows]

    totals = {"users": 0, "fetched": 0, "saved": 0, "sent": 0}
    if not user_ids:
        return totals

    async with httpx.AsyncClient() as http:
        for uid in user_ids:
            try:
                res = await _sync_one_user(pool, http, uid)
                totals["users"] += 1
                totals["fetched"] += res["fetched"]
                totals["saved"] += res["saved"]
                totals["sent"] += res["sent"]
            except Exception as err:  # noqa: BLE001
                log.exception("ml_notices: user %s failed: %s", uid, err)
            await asyncio.sleep(ML_RATE_LIMIT_SLEEP)
    return totals
