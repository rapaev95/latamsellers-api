"""Synchronous psycopg2 facade for legacy Streamlit code.

Provides the same API as `_admin/db_storage.py` but without Streamlit:
  - `_get_db() / _put_db(conn)` — connection helpers (no pool yet, single conn)
  - `db_load(key) / db_save(key, value)` — JSONB read/write keyed by current user
  - `db_load_multi(keys)` — bulk read
  - `db_has_data(key)` — existence check
  - `db_is_available()` — DB+user_id readiness flag

Current user ID is read from a ContextVar set per-request by FastAPI middleware
(`v2.middleware.user_context`).
"""
from __future__ import annotations

import contextvars
import json
import os
from typing import Any

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_PUBLIC_URL")

_current_user_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "v2_legacy_current_user_id", default=None,
)


def set_current_user_id(uid: int | None) -> None:
    """Called by FastAPI middleware on each request."""
    _current_user_id_var.set(uid)


def _current_user_id() -> int | None:
    return _current_user_id_var.get()


def _get_db():
    if not DATABASE_URL:
        return None
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception:
        return None


def _put_db(conn) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def db_is_available() -> bool:
    if not DATABASE_URL:
        return False
    return _current_user_id() is not None


def db_save(data_key: str, data_value: Any, user_id: int | None = None) -> bool:
    uid = user_id or _current_user_id()
    if uid is None:
        return False
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_data (user_id, data_key, data_value, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (user_id, data_key)
            DO UPDATE SET data_value = EXCLUDED.data_value, updated_at = NOW()
            """,
            (uid, data_key, json.dumps(data_value, ensure_ascii=False, default=str)),
        )
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False
    finally:
        _put_db(conn)


def db_load(data_key: str, user_id: int | None = None) -> Any | None:
    uid = user_id or _current_user_id()
    if uid is None:
        return None
    conn = _get_db()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_value FROM user_data WHERE user_id = %s AND data_key = %s",
            (uid, data_key),
        )
        row = cur.fetchone()
        cur.close()
        if row is None or row[0] is None:
            return None
        val = row[0]
        if isinstance(val, (dict, list)):
            return val
        return json.loads(val)
    except Exception:
        return None
    finally:
        _put_db(conn)


def db_load_multi(data_keys: list[str], user_id: int | None = None) -> dict[str, Any]:
    uid = user_id or _current_user_id()
    if uid is None:
        return {}
    conn = _get_db()
    if conn is None:
        return {}
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_key, data_value FROM user_data WHERE user_id = %s AND data_key = ANY(%s)",
            (uid, data_keys),
        )
        rows = cur.fetchall()
        cur.close()
        out: dict[str, Any] = {}
        for key, val in rows:
            if isinstance(val, (dict, list)):
                out[key] = val
            else:
                out[key] = json.loads(val)
        return out
    except Exception:
        return {}
    finally:
        _put_db(conn)


def db_has_data(data_key: str, user_id: int | None = None) -> bool:
    uid = user_id or _current_user_id()
    if uid is None:
        return False
    conn = _get_db()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM user_data WHERE user_id = %s AND data_key = %s",
            (uid, data_key),
        )
        return cur.fetchone() is not None
    except Exception:
        return False
    finally:
        _put_db(conn)
