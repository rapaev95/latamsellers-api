"""LLM description generator — Sprint 7.

Pipeline:
1. Receive family context (name, brand, model, cores, tamanhos, key attrs)
   + optional supplier doc text (extracted from PDF/DOCX upstream).
2. Ask Claude via OpenRouter to write a pt_BR description with:
   - opening 2-line hook
   - 3-5 bullet «destaques»
   - measurements / material if known
   - politeness/CTA close
3. Return plain text (ML's /items/{id}/description accepts plain_text and
   converts).

We reuse the OpenRouter + ai_usage_tracker pattern from
`ml_photo_descriptions.py` for consistency + billing audit.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import asyncpg
import httpx

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DESC_MODEL = "anthropic/claude-sonnet-4.5"

_PROMPT_SYSTEM = """Você é redator de produtos do Mercado Livre.
Escreva uma descrição em português brasileiro para um anúncio de moda.
Estrutura obrigatória:
- 1ª linha: gancho curto (≤ 70 caracteres)
- linha em branco
- 3-5 bullets "• ..." com destaques: material, conforto, ocasiões de uso, garantia
- linha em branco
- chamada para ação curta (1 linha)
Regras: sem emojis, sem ALL CAPS, sem promessas exageradas (não diga "100%"
sem fonte). Use ponto e vírgula com moderação. Máximo 1500 caracteres.
"""


def _build_context_block(
    *, family_name: str, brand: Optional[str], model: Optional[str],
    cores: list[str], tamanhos: list[str],
    keywords: list[str], supplier_text: Optional[str],
    extra_attrs: dict[str, str] | None = None,
) -> str:
    lines = [
        f"Nome do produto: {family_name}",
    ]
    if brand:
        lines.append(f"Marca: {brand}")
    if model:
        lines.append(f"Modelo: {model}")
    if cores:
        lines.append(f"Cores disponíveis: {', '.join(cores)}")
    if tamanhos:
        lines.append(f"Tamanhos: {', '.join(tamanhos)}")
    if keywords:
        lines.append(f"Palavras-chave: {', '.join(keywords)}")
    if extra_attrs:
        for k, v in extra_attrs.items():
            if v:
                lines.append(f"{k}: {v}")
    if supplier_text:
        lines.append("Ficha técnica do fornecedor:")
        lines.append(supplier_text[:6000])     # cap on supplier blob
    return "\n".join(lines)


async def generate_description(
    pool: asyncpg.Pool,
    *,
    ls_user_id: int,
    family_name: str,
    brand: Optional[str] = None,
    model: Optional[str] = None,
    cores: list[str] | None = None,
    tamanhos: list[str] | None = None,
    keywords: list[str] | None = None,
    supplier_text: Optional[str] = None,
    extra_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Returns {description, model, usage?}. On error: {error, status}."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "no_openrouter_api_key", "status": 500}

    context = _build_context_block(
        family_name=family_name, brand=brand, model=model,
        cores=cores or [], tamanhos=tamanhos or [],
        keywords=keywords or [], supplier_text=supplier_text,
        extra_attrs=extra_attrs,
    )
    user_msg = (
        "Gere a descrição com base no contexto abaixo. "
        "Foque no que pode ser comprovado pelo contexto, evite suposições.\n\n"
        f"{context}"
    )

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
                    "model": DESC_MODEL,
                    "max_tokens": 1500,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": _PROMPT_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                },
                timeout=60.0,
            )
    except Exception as err:  # noqa: BLE001
        return {"error": "network", "detail": str(err), "status": 0}

    if r.status_code != 200:
        log.warning("description LLM %s: %s", r.status_code, r.text[:200])
        return {"error": f"openrouter_{r.status_code}",
                "body_preview": r.text[:300], "status": r.status_code}

    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        return {"error": "non_json", "status": r.status_code}

    # Telemetry — best-effort, never blocks the response.
    try:
        from . import ai_usage_tracker as _tracker
        await _tracker.log_call(
            pool, user_id=ls_user_id,
            service="publish/description/generate",
            model=DESC_MODEL,
            response_data=data, status_code=r.status_code,
            metadata={"family_name": family_name[:80]},
        )
    except Exception:  # noqa: BLE001
        pass

    content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    if not content:
        return {"error": "empty_completion", "status": 500}
    return {
        "description": content.strip(),
        "model": DESC_MODEL,
        "usage": data.get("usage"),
    }
