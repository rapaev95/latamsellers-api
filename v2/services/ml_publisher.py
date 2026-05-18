"""ML publisher — Sprint 7.

Pipeline per Item:
  1. Strip wizard meta from body (`_lsp_meta`, `_keywords_hint`, `_description`).
  2. POST /items → expect 201 with `id` (MLB...) on success, or 400 with
     `cause: [...]` on validation failure.
  3. If description provided → POST /items/{id}/description with HTML body.
  4. Persist row in `nf_publishing_attempts`.

Cause-array parser is the «exception language»: ML returns granular field
references like `item.attributes[0]`, `item.variations[3].attribute_combinations[1]`.
We normalise to a structured list of `{department, cause_id, type, code, message,
references}` and surface back to the UI.

Throttling: 0.5s sleep between Items so a 30-Item plan doesn't burst-rate
ML. Per-plan publish is sequential — small N, easier to recover from
mid-batch failures.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import httpx

from . import ml_oauth as ml_oauth_svc

log = logging.getLogger(__name__)

ML_API_BASE = "https://api.mercadolibre.com"

# Per-item throttle. 0.5s = 2 RPS, fine for ML write limits.
ITEM_THROTTLE_S = 0.5

# Fields we add for UI/debug; never sent to ML.
_META_FIELDS = {"_lsp_meta", "_keywords_hint", "_description"}


def _strip_meta(body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k not in _META_FIELDS}


# ──────────────────────────────────────────────────────────────────────────────
# Validation cause parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_causes(body: Any) -> list[dict[str, Any]]:
    """ML returns `{message, error, status, cause: [...]}` on 400.

    Normalises the cause array into a list of `{department, cause_id, type,
    code, message, references}`. Robust to missing keys.
    """
    if not isinstance(body, dict):
        return []
    causes = body.get("cause") or []
    if not isinstance(causes, list):
        return []
    out: list[dict[str, Any]] = []
    for c in causes:
        if not isinstance(c, dict):
            continue
        out.append({
            "department": c.get("department"),
            "cause_id": c.get("cause_id"),
            "type": c.get("type"),       # 'error' | 'warning'
            "code": c.get("code"),
            "message": c.get("message"),
            "references": c.get("references") or [],
        })
    return out


def cause_severity(causes: list[dict[str, Any]]) -> str:
    """Returns 'error' if ANY cause has type=error, else 'warning' if any
    warning, else 'success'."""
    has_error = any(c.get("type") == "error" for c in causes)
    if has_error:
        return "error"
    has_warning = any(c.get("type") == "warning" for c in causes)
    if has_warning:
        return "warning"
    return "success"


# ──────────────────────────────────────────────────────────────────────────────
# Per-item publish
# ──────────────────────────────────────────────────────────────────────────────

async def _post_item(
    http: httpx.AsyncClient, token: str, body: dict[str, Any],
) -> tuple[int, Any]:
    r = await http.post(
        f"{ML_API_BASE}/items",
        json=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    try:
        parsed = r.json()
    except Exception:  # noqa: BLE001
        parsed = {"raw": r.text[:1000]}
    return r.status_code, parsed


async def _post_description(
    http: httpx.AsyncClient, token: str, item_id: str, plain_or_html: str,
) -> tuple[int, Any]:
    """POST /items/{id}/description with `{plain_text}` body. ML auto-converts."""
    r = await http.post(
        f"{ML_API_BASE}/items/{item_id}/description",
        json={"plain_text": plain_or_html},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=20.0,
    )
    try:
        parsed = r.json()
    except Exception:  # noqa: BLE001
        parsed = {"raw": r.text[:1000]}
    return r.status_code, parsed


async def _log_attempt(
    pool: asyncpg.Pool,
    *, plan_id: int, family_id: Optional[str], up_id: Optional[str],
    request_body: dict[str, Any], response_body: Any,
    causes: list[dict[str, Any]], status: str,
    http_status: int, ml_item_id: Optional[str],
) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO nf_publishing_attempts (
                plan_id, family_id, up_id, ml_item_id,
                request_body, response_body, cause_array,
                status, http_status
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8, $9)
            RETURNING id
            """,
            plan_id, family_id, up_id, ml_item_id,
            json.dumps(request_body), json.dumps(response_body),
            json.dumps(causes), status, http_status,
        )
    return int(row["id"])


async def publish_one_item(
    pool: asyncpg.Pool,
    http: httpx.AsyncClient,
    *,
    token: str,
    plan_id: int,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Publish one Item. Returns {status, ml_item_id?, causes, http_status, attempt_id}."""
    meta = (body.get("_lsp_meta") or {}) if isinstance(body, dict) else {}
    description = body.get("_description") if isinstance(body, dict) else None
    family_id = meta.get("family_id")
    up_id = meta.get("up_id")

    clean = _strip_meta(body)
    http_status, parsed = await _post_item(http, token, clean)
    causes = parse_causes(parsed)
    ml_item_id: Optional[str] = None
    status: str

    if http_status in (200, 201) and isinstance(parsed, dict):
        ml_item_id = parsed.get("id")
        sev = cause_severity(causes)
        status = "success" if sev != "warning" else "warning"
    else:
        status = "error"

    # Description (best-effort, non-blocking for success).
    description_result: Optional[dict[str, Any]] = None
    if status in ("success", "warning") and ml_item_id and description:
        try:
            dcode, dbody = await _post_description(http, token, ml_item_id, description)
            description_result = {"status": dcode, "body": dbody}
            if dcode >= 400:
                # Don't tank the publish — surface as warning.
                if status == "success":
                    status = "warning"
                causes.append({
                    "department": "description",
                    "cause_id": None,
                    "type": "warning",
                    "code": f"description_post_{dcode}",
                    "message": f"description POST returned {dcode}",
                    "references": [f"item:{ml_item_id}"],
                })
        except Exception as err:  # noqa: BLE001
            description_result = {"error": str(err)}
            if status == "success":
                status = "warning"
            causes.append({
                "department": "description",
                "cause_id": None, "type": "warning", "code": "description_exception",
                "message": str(err),
                "references": [f"item:{ml_item_id}"],
            })

    response_for_log: Any = parsed
    if description_result is not None:
        response_for_log = {"item": parsed, "description": description_result}

    attempt_id = await _log_attempt(
        pool,
        plan_id=plan_id, family_id=family_id, up_id=up_id,
        request_body=clean, response_body=response_for_log,
        causes=causes, status=status, http_status=http_status,
        ml_item_id=ml_item_id,
    )
    return {
        "attempt_id": attempt_id,
        "status": status,
        "http_status": http_status,
        "ml_item_id": ml_item_id,
        "causes": causes,
        "family_id": family_id,
        "up_id": up_id,
    }


async def publish_plan_items(
    pool: asyncpg.Pool,
    *,
    token: str,
    plan_id: int,
    items: list[dict[str, Any]],
    only_up_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Sequential bulk publish with throttling. Returns aggregate stats +
    per-Item result list.

    `token` is pre-resolved by the router via deps.resolve_ml_context() —
    so this service works for both self-service and managed accounts.
    """
    results: list[dict[str, Any]] = []
    counts = {"success": 0, "warning": 0, "error": 0, "skipped": 0}
    async with httpx.AsyncClient() as http:
        for idx, body in enumerate(items):
            meta = (body.get("_lsp_meta") or {}) if isinstance(body, dict) else {}
            up_id = meta.get("up_id")
            if only_up_ids is not None and up_id not in only_up_ids:
                counts["skipped"] += 1
                continue
            try:
                res = await publish_one_item(
                    pool, http, token=token, plan_id=plan_id, body=body,
                )
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                results.append(res)
            except Exception as err:  # noqa: BLE001
                log.exception("publish_one_item crashed at idx=%s", idx)
                counts["error"] += 1
                results.append({
                    "status": "error", "http_status": 0,
                    "ml_item_id": None, "causes": [],
                    "error": str(err),
                    "family_id": meta.get("family_id"),
                    "up_id": up_id,
                })
            await asyncio.sleep(ITEM_THROTTLE_S)
    return {
        "plan_id": plan_id,
        "totals": counts,
        "items": results,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
