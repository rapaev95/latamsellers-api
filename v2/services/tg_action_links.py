"""HMAC-signed action links для TG-сообщений.

Юзер кликает ссылку из TG → попадает на наш GET-endpoint с подписью →
бэкенд верифицирует HMAC → выполняет действие (active item, и т.д.) →
возвращает HTML-страничку с результатом.

Зачем подпись (а не cookie auth):
  - В TG нет sticky session, ссылка должна работать standalone.
  - Кто получил TG-сообщение = тот и владелец юзер_id, secret гарантирует
    что link не подделан и не может быть перенаправлен на другого user'а.
  - TTL ограничивает время действия (default 7d).

Secret: env LS_LINK_SECRET. Если не выставлен — функции возвращают
пустую строку (link disabled), верификация всегда fails. Это намеренно:
без secret нельзя гарантировать целостность.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

_SECRET_FALLBACK_ENV_VARS = ("LS_LINK_SECRET", "ML_CLIENT_SECRET")


def _get_secret() -> str:
    for k in _SECRET_FALLBACK_ENV_VARS:
        v = os.environ.get(k)
        if v:
            return v
    return ""


def _app_base() -> str:
    return os.environ.get("APP_BASE_URL", "https://app.lsprofit.app").rstrip("/")


def _sign(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:24]


def make_signed_url(
    *,
    action: str,
    user_id: int,
    item_id: str,
    ttl_seconds: int = 7 * 24 * 3600,
) -> str:
    """Generate signed GET-link для action на конкретный item.
    Возвращает full URL или пустую строку если secret не выставлен.

    URL shape:
      {APP_BASE}/api/v2-proxy/escalar/items/{item_id}/{action}?uid=N&exp=N&sig=H
    """
    secret = _get_secret()
    if not secret:
        return ""
    exp = int(time.time()) + int(ttl_seconds)
    payload = f"{action}:{user_id}:{item_id}:{exp}"
    sig = _sign(payload, secret)
    return (
        f"{_app_base()}/api/v2-proxy/escalar/items/{item_id}/{action}"
        f"?uid={user_id}&exp={exp}&sig={sig}"
    )


def verify_signed(
    *,
    action: str,
    user_id: int,
    item_id: str,
    exp: int,
    sig: str,
) -> tuple[bool, str]:
    """Verify HMAC + TTL. Returns (ok, error_code)."""
    secret = _get_secret()
    if not secret:
        return False, "no_secret_configured"
    if int(time.time()) > int(exp):
        return False, "expired"
    payload = f"{action}:{user_id}:{item_id}:{exp}"
    expected = _sign(payload, secret)
    if not hmac.compare_digest(expected, sig):
        return False, "bad_signature"
    return True, ""
