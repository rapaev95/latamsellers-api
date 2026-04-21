"""Claude Haiku pt→ru translator for ML notices.

Zero-cost fast-path when the text is already Cyrillic or empty.
System prompt (glossary) is marked cache_control=ephemeral so repeated
calls within 5 minutes read from the prompt cache instead of re-billing tokens.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

SYSTEM_GLOSSARY = """Você é um tradutor profissional de português brasileiro para russo, especializado em e-commerce do Mercado Livre.

REGRAS:
1. Traduza o texto fornecido do pt-BR para o ru, preservando o sentido técnico exato dos termos oficiais do Mercado Livre.
2. Mantenha ligações, IDs, URLs e códigos alfanuméricos intactos.
3. Não adicione comentários, explicações ou markdown — retorne APENAS a tradução.
4. Se o texto contiver quebras de linha, preserve-as.

GLOSSÁRIO OBRIGATÓRIO (pt → ru):
- comissão → комиссия
- tarifa de venda → комиссия за продажу
- frete → доставка
- envio → отправление
- anúncio → объявление
- publicação → публикация
- produto → товар
- venda → продажа
- comprador → покупатель
- vendedor → продавец
- reclamação → жалоба (рекламация)
- mediação → медиация
- reputação → репутация
- conta → аккаунт
- moderação → модерация
- faturamento → биллинг
- fatura → счёт
- cancelamento → отмена
- reembolso → возврат средств
- estoque → склад (остаток)
- Full → Full (не переводить)
- MLB → MLB (не переводить)
- permalink → ссылка на товар
- política → правило (политика)
- suspensão → приостановка
- desativação → деактивация"""


async def translate_pt_ru(text: Optional[str], http: Optional[httpx.AsyncClient] = None) -> str:
    """Translate pt-BR → ru. Returns original on failure or fast-path match."""
    if not text or not text.strip():
        return text or ""
    if _CYRILLIC_RE.search(text):
        # Already (partly) Russian — skip.
        return text

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — returning original text")
        return text

    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": [
            {"type": "text", "text": SYSTEM_GLOSSARY, "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {"role": "user", "content": text},
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    client = http
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=20.0)
    try:
        r = await client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
        if r.status_code != 200:
            log.warning("Anthropic %s: %s", r.status_code, r.text[:200])
            return text
        data = r.json()
        content = data.get("content") or []
        if content and isinstance(content, list) and content[0].get("type") == "text":
            return content[0].get("text", text) or text
        return text
    except Exception as err:  # noqa: BLE001
        log.exception("translate_pt_ru failed: %s", err)
        return text
    finally:
        if owns_client and client is not None:
            await client.aclose()
