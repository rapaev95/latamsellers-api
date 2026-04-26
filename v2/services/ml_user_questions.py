"""Questions Q&A cache — ML `/my/received_questions/search`.

TTL=5 min (questions stream in rapid; user expects near-live view). Refresh
paginates both UNANSWERED and ANSWERED, upserts each row. POST /answers
flow (outside this module) should call `upsert_one_answered` after success
so the reply shows immediately without waiting for the next refresh.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"
RATE_SLEEP = 0.2


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_user_questions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  question_id BIGINT NOT NULL,
  item_id TEXT,
  text TEXT,
  status TEXT,
  date_created TIMESTAMPTZ,
  answer_text TEXT,
  answer_date TIMESTAMPTZ,
  from_nickname TEXT,
  raw JSONB,
  fetched_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, question_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_user_questions_user_status ON ml_user_questions(user_id, status);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


def _parse_dt(s: Any) -> Any:
    """asyncpg needs datetime, not ISO string. Pass-through if already datetime."""
    from datetime import datetime
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    try:
        raw = str(s).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


# ── ML API ────────────────────────────────────────────────────────────────────

async def _fetch_by_status(http: httpx.AsyncClient, token: str, status: str) -> list[dict]:
    """Scroll through /my/received_questions/search?status=X with offset paging."""
    out: list[dict] = []
    offset = 0
    limit = 50
    while True:
        url = (
            f"{ML_API_BASE}/my/received_questions/search?status={status}"
            f"&sort_fields=date_created&sort_types=DESC&api_version=4"
            f"&limit={limit}&offset={offset}"
        )
        try:
            r = await http.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20.0)
        except Exception as err:  # noqa: BLE001
            log.warning("questions page failed: %s", err)
            break
        if r.status_code != 200:
            break
        data = r.json() or {}
        questions = data.get("questions") or []
        if not questions:
            break
        out.extend(questions)
        total = data.get("total", 0)
        offset += limit
        if offset >= total or len(questions) < limit:
            break
        await asyncio.sleep(RATE_SLEEP)
    return out


# ── Refresh ───────────────────────────────────────────────────────────────────

async def refresh_user_questions(pool: asyncpg.Pool, user_id: int) -> dict[str, int]:
    try:
        token, _exp, _refreshed = await ml_oauth_svc.get_valid_access_token(pool, user_id)
    except ml_oauth_svc.MLRefreshError:
        return {"fetched": 0, "saved": 0}

    async with httpx.AsyncClient() as http:
        unanswered, answered = await asyncio.gather(
            _fetch_by_status(http, token, "UNANSWERED"),
            _fetch_by_status(http, token, "ANSWERED"),
        )

    all_q = unanswered + answered
    saved = 0
    for q in all_q:
        try:
            async with pool.acquire() as conn:
                await _upsert(conn, user_id, q)
            saved += 1
        except Exception as err:  # noqa: BLE001
            log.warning("upsert question %s failed: %s", q.get("id"), err)

    return {"fetched": len(all_q), "saved": saved}


async def _upsert(conn: asyncpg.Connection, user_id: int, q: dict) -> None:
    qid = q.get("id")
    if qid is None:
        return
    answer = q.get("answer") or {}
    from_blk = q.get("from") or {}
    await conn.execute(
        """
        INSERT INTO ml_user_questions
          (user_id, question_id, item_id, text, status, date_created,
           answer_text, answer_date, from_nickname, raw, fetched_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, NOW())
        ON CONFLICT (user_id, question_id) DO UPDATE SET
          item_id = EXCLUDED.item_id,
          text = EXCLUDED.text,
          status = EXCLUDED.status,
          date_created = EXCLUDED.date_created,
          answer_text = EXCLUDED.answer_text,
          answer_date = EXCLUDED.answer_date,
          from_nickname = EXCLUDED.from_nickname,
          raw = EXCLUDED.raw,
          fetched_at = NOW()
        """,
        user_id,
        int(qid),
        q.get("item_id"),
        q.get("text"),
        q.get("status"),
        _parse_dt(q.get("date_created")),
        answer.get("text") if isinstance(answer, dict) else None,
        _parse_dt(answer.get("date_created")) if isinstance(answer, dict) else None,
        from_blk.get("nickname") if isinstance(from_blk, dict) else None,
        json.dumps(q, default=str),
    )


async def upsert_one_answered(
    pool: asyncpg.Pool,
    user_id: int,
    question_id: int,
    answer_text: str,
) -> int:
    """Called after POST /answers succeeds — marks the row ANSWERED locally
    so UI reflects the reply without waiting for the next full refresh.
    Returns the number of rows updated (0 if no cached row matched)."""
    from datetime import datetime, timezone
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE ml_user_questions
               SET status = 'ANSWERED',
                   answer_text = $3,
                   answer_date = $4,
                   fetched_at = NOW()
             WHERE user_id = $1 AND question_id = $2
            """,
            user_id,
            int(question_id),
            answer_text,
            datetime.now(timezone.utc),
        )
    # asyncpg execute returns command tag like "UPDATE 1" / "UPDATE 0".
    try:
        return int(result.rsplit(" ", 1)[-1])
    except Exception:  # noqa: BLE001
        return 0


# ── Cache readback ────────────────────────────────────────────────────────────

async def get_cached(
    pool: asyncpg.Pool,
    user_id: int,
    status: str = "ALL",
) -> dict[str, Any]:
    where = "WHERE user_id = $1"
    params: list[Any] = [user_id]
    if status and status.upper() != "ALL":
        where += " AND status = $2"
        params.append(status.upper())

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT question_id, item_id, text, status,
                   to_char(date_created AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS date_created,
                   answer_text,
                   to_char(answer_date AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS answer_date,
                   from_nickname, raw,
                   to_char(fetched_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS fetched_at
              FROM ml_user_questions
              {where}
             ORDER BY date_created DESC NULLS LAST
            """,
            *params,
        )

    questions = []
    for r in rows:
        raw = r["raw"]
        if isinstance(raw, str):
            raw = json.loads(raw or "{}")
        questions.append({
            "id": int(r["question_id"]),
            "item_id": r["item_id"],
            "text": r["text"],
            "status": r["status"],
            "date_created": r["date_created"],
            "answer": {
                "text": r["answer_text"],
                "date_created": r["answer_date"],
            } if r["answer_text"] else None,
            "from": {"nickname": r["from_nickname"]} if r["from_nickname"] else None,
        })
    return {
        "total": len(questions),
        "questions": questions,
        "fetchedAt": rows[0]["fetched_at"] if rows else None,
    }


async def get_latest_fetched_at(pool: asyncpg.Pool, user_id: int) -> str | None:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT to_char(MAX(fetched_at) AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS"Z"')
              FROM ml_user_questions
             WHERE user_id = $1
            """,
            user_id,
        )
