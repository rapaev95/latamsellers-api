"""AI-generated descriptions per product photo, cached in DB.

Why: when a buyer asks 'quantas divisórias' / 'qual a textura' / etc, AI
needs to inspect the photos. Sending raw image URLs to Claude per buyer
question costs $0.06+ each (6 photos × $0.01 detail:high). For active
listings with 30+ questions/week that's wasteful — and Claude sometimes
misses obvious details under question-time pressure.

Solution: pre-generate one structured description per photo when item
context is refreshed (or on-demand via /escalar/items/{id}/photo-descriptions).
Descriptions live in ml_item_photo_descriptions and feed the AI-suggest
prompt as a text RAG block. Photos still attached as vision blocks too —
descriptions are belt-and-suspenders for cases where vision misses count
or text labels.

Schema:
  user_id, item_id, picture_id (ML-side stable id), picture_url,
  picture_index (0..5), description, generated_at, model

Memory: project_photo_descriptions_in_db.md
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PHOTO_DESC_MODEL = "anthropic/claude-sonnet-4.5"
IMAGE_DETAIL = "high"
MAX_PHOTOS = 6

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ml_item_photo_descriptions (
  user_id INTEGER NOT NULL,
  item_id TEXT NOT NULL,
  picture_id TEXT NOT NULL,
  picture_url TEXT NOT NULL,
  picture_index INTEGER NOT NULL,
  description TEXT NOT NULL,
  generated_at TIMESTAMPTZ DEFAULT NOW(),
  model TEXT,
  PRIMARY KEY (user_id, item_id, picture_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_item_photo_desc_item
  ON ml_item_photo_descriptions(user_id, item_id);
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)


_PHOTO_DESC_PROMPT = """Você inspeciona fotos de anúncios do Mercado Livre Brasil e gera UMA descrição estruturada por foto, em português brasileiro, factual e útil para responder perguntas de compradores.

Para CADA foto que você receber, descreva (4-7 linhas):
1. O QUE A FOTO MOSTRA — visão externa, interior aberto, detalhe de etiqueta/zíper, lifestyle, etc.
2. ELEMENTOS CONTÁVEIS — quantidade de divisões / bolsos / compartimentos / botões / alças / tirantes / etiquetas (só se forem visíveis e contáveis nesta foto específica).
3. MEDIDAS / TEXTOS visíveis — números, etiquetas, marcas, certificações.
4. MATERIAL aparente — couro / sintético / lona / metal / plástico / tecido (descreva o que VÊ, não invente origem).
5. CORES E ACABAMENTO — tom específico, brilho/fosco, costuras visíveis.

REGRAS:
- NÃO repita info entre fotos (foto 1 não precisa repetir o que está em foto 2).
- NÃO invente — se não dá para ver, omita.
- Output formato:
  Foto 1: <descrição>
  Foto 2: <descrição>
  ...

Se uma foto não acrescenta info nova, pode ser bem curta (1 linha)."""


async def generate_descriptions_for_item(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
    pictures: list[dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Generate one description per picture and persist.

    Returns: {generated: int, cached: int, skipped: int, model: str}.

    pictures shape: [{id, url|secure_url}, ...] — ML's standard structure.
    Idempotent: skips photos already in DB unless force=True.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"generated": 0, "cached": 0, "skipped": len(pictures), "model": None,
                "error": "no_api_key"}

    item_id = item_id.upper()
    pics_clean: list[tuple[int, str, str]] = []  # (idx, picture_id, url)
    for idx, p in enumerate(pictures[:MAX_PHOTOS]):
        if not isinstance(p, dict):
            continue
        url = p.get("secure_url") or p.get("url")
        pid = p.get("id") or url
        if not url or not pid:
            continue
        pics_clean.append((idx, str(pid), str(url)))

    if not pics_clean:
        return {"generated": 0, "cached": 0, "skipped": 0, "model": None}

    cached_ids: set[str] = set()
    if not force:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT picture_id FROM ml_item_photo_descriptions
                 WHERE user_id = $1 AND item_id = $2
                """,
                user_id, item_id,
            )
        cached_ids = {r["picture_id"] for r in rows}

    to_process = [(i, pid, u) for (i, pid, u) in pics_clean if pid not in cached_ids]
    if not to_process:
        return {"generated": 0, "cached": len(cached_ids), "skipped": 0, "model": PHOTO_DESC_MODEL}

    # Build multimodal user content — all pics in one Claude call (cheaper
    # than per-photo) — Claude returns one description per Foto N.
    user_content: list[dict[str, Any]] = [
        {"type": "text",
         "text": f"Inspecione cada uma das {len(to_process)} fotos abaixo e gere descrição por foto:"}
    ]
    for idx, _pid, url in to_process:
        user_content.append({"type": "text", "text": f"\nFoto {idx + 1}:"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": IMAGE_DETAIL},
        })

    try:
        async with httpx.AsyncClient() as http:
            r = await http.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://app.lsprofit.app",
                    "X-Title": "LS Profit App",
                },
                json={
                    "model": PHOTO_DESC_MODEL,
                    "max_tokens": 1500,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": _PHOTO_DESC_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                },
                timeout=60.0,
            )
        if r.status_code != 200:
            log.warning("photo desc OpenRouter %s: %s", r.status_code, r.text[:200])
            try:
                from . import tg_admin_alerts as _alerts
                await _alerts.alert_openrouter_failure(
                    r.status_code, r.text, service="photo-descriptions/generate",
                )
            except Exception:  # noqa: BLE001
                pass
            return {"generated": 0, "cached": len(cached_ids), "skipped": len(to_process),
                    "model": PHOTO_DESC_MODEL, "error": f"openrouter_{r.status_code}"}
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            return {"generated": 0, "cached": len(cached_ids), "skipped": len(to_process),
                    "model": PHOTO_DESC_MODEL, "error": "empty_response"}
    except Exception as err:  # noqa: BLE001
        log.warning("photo desc exception: %s", err)
        return {"generated": 0, "cached": len(cached_ids), "skipped": len(to_process),
                "model": PHOTO_DESC_MODEL, "error": str(err)}

    # Parse "Foto 1: ...\nFoto 2: ..." into per-photo strings.
    parsed = _parse_per_photo(content, max_count=len(to_process))
    if len(parsed) < len(to_process):
        # Filling missing slots with the entire response — better than nothing.
        for i in range(len(parsed), len(to_process)):
            parsed.append(content[:1500])

    saved = 0
    async with pool.acquire() as conn:
        for (idx, pid, url), desc in zip(to_process, parsed):
            try:
                await conn.execute(
                    """
                    INSERT INTO ml_item_photo_descriptions
                      (user_id, item_id, picture_id, picture_url, picture_index,
                       description, generated_at, model)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7)
                    ON CONFLICT (user_id, item_id, picture_id) DO UPDATE SET
                      picture_url = EXCLUDED.picture_url,
                      picture_index = EXCLUDED.picture_index,
                      description = EXCLUDED.description,
                      generated_at = NOW(),
                      model = EXCLUDED.model
                    """,
                    user_id, item_id, pid, url, idx, desc.strip(), PHOTO_DESC_MODEL,
                )
                saved += 1
            except Exception as err:  # noqa: BLE001
                log.warning("photo desc upsert failed item=%s pid=%s: %s", item_id, pid, err)

    return {
        "generated": saved,
        "cached": len(cached_ids),
        "skipped": len(to_process) - saved,
        "model": PHOTO_DESC_MODEL,
    }


def _parse_per_photo(content: str, max_count: int) -> list[str]:
    """Split 'Foto 1: ...\\nFoto 2: ...' into list of per-photo strings."""
    import re as _re
    parts = _re.split(r"(?im)^\s*Foto\s+\d+\s*[:.\-]?\s*", content)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:max_count]


async def get_descriptions_for_item(
    pool: asyncpg.Pool,
    user_id: int,
    item_id: str,
) -> list[dict[str, Any]]:
    """Returns list ordered by picture_index. Empty if no descriptions."""
    item_id = item_id.upper()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT picture_index, picture_url, description, generated_at
              FROM ml_item_photo_descriptions
             WHERE user_id = $1 AND item_id = $2
             ORDER BY picture_index ASC
            """,
            user_id, item_id,
        )
    return [
        {
            "picture_index": r["picture_index"],
            "picture_url": r["picture_url"],
            "description": r["description"],
            "generated_at": r["generated_at"].isoformat() if r["generated_at"] else None,
        }
        for r in rows
    ]


def descriptions_to_prompt_block(descriptions: list[dict[str, Any]]) -> str:
    """Format cached descriptions for inclusion in AI-suggest prompt."""
    if not descriptions:
        return ""
    lines = ["DESCRIÇÕES DAS FOTOS (geradas previamente por IA — use como contexto):"]
    for d in descriptions:
        idx = d.get("picture_index", 0) + 1
        desc = (d.get("description") or "").strip()
        if desc:
            lines.append(f"\nFoto {idx}: {desc}")
    return "\n".join(lines)
