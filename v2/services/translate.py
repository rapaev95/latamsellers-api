"""ML notices translator — via OpenRouter (OpenAI-compatible chat API).

Supported targets: ru, en. Portuguese (pt) is source — for pt target we skip.

Fast-paths (no API call):
- Empty/whitespace → return as is.
- Target=ru and text already contains Cyrillic → return as is.
- Target=en and text has no Portuguese-specific letters (ã/ç/õ/etc.) and no
  Cyrillic → assume already English-ish, return as is.

Glossary is sent as system prompt; OpenRouter forwards `cache_control`
hints to providers that support them (Anthropic, etc.) so repeated calls
within the 5-minute window can reuse the cached prompt.

Env:
  OPENROUTER_API_KEY — required for live translation.
  TRANSLATE_MODEL — optional, default "anthropic/claude-haiku-4.5".
"""
from __future__ import annotations

import logging
import os
import re
from typing import Literal, Optional

import httpx

log = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# Target languages we actually translate TO. 'pt' is source — no-op.
TargetLang = Literal["ru", "en", "pt"]

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
# Portuguese-specific accented letters — if absent AND no cyrillic, a text in
# "en-speaker's e-commerce vocabulary" is probably already English-adjacent.
_PT_SPECIFIC_RE = re.compile(r"[ãâáàçéêíõôóúü]", re.IGNORECASE)

# Patterns that indicate the LLM refused to translate and replied with a
# meta-message instead of a translation. To avoid false-positives with legit
# buyer questions (e.g., a buyer politely asking "por favor, forneça mais
# detalhes"), each pattern requires a translation-context word: "перевести",
# "traduzir", "translate", "ortografia", "tradução" etc.
_REFUSAL_PATTERNS = [
    # Russian refusals — must mention translation
    re.compile(r"не\s+мог[уть]\s+перевести", re.IGNORECASE),
    re.compile(r"перевести\s+(этот|данн)", re.IGNORECASE),
    re.compile(r"для\s+перевода", re.IGNORECASE),
    # Portuguese refusals — must mention translating
    re.compile(r"não\s+(consigo|posso)\s+traduzir", re.IGNORECASE),
    re.compile(r"verificar\s+a\s+ortografia", re.IGNORECASE),
    re.compile(r"para\s+traduzir", re.IGNORECASE),
    # English refusals — must mention translation
    re.compile(r"can(?:'|no)t\s+translate", re.IGNORECASE),
    re.compile(r"unable\s+to\s+translate", re.IGNORECASE),
    re.compile(r"please\s+provide.*translat", re.IGNORECASE),
    # The specific "Não entendi sua solicitação" hallucination from Claude when
    # given short buyer text — it's a refusal in disguise.
    re.compile(r"não\s+entendi\s+sua\s+solicita", re.IGNORECASE),
]


def _looks_like_refusal(s: str) -> bool:
    sample = (s or "")[:300]
    return any(p.search(sample) for p in _REFUSAL_PATTERNS)


GLOSSARY_RU = """Você é um tradutor profissional de português brasileiro para russo, especializado em e-commerce do Mercado Livre.

REGRAS:
1. Traduza o texto fornecido do pt-BR para o ru, preservando o sentido técnico exato dos termos oficiais do Mercado Livre.
2. Mantenha ligações, IDs, URLs e códigos alfanuméricos intactos.
3. Não adicione comentários, explicações ou markdown — retorne APENAS a tradução.
4. Se o texto contiver quebras de linha, preserve-as.
5. SEMPRE traduza, mesmo se o texto for curto, ambíguo, contiver erros de digitação ou parecer sem sentido. NUNCA recuse traduzir. NUNCA peça esclarecimento. NUNCA pergunte de volta. Se o texto contém um erro óbvio (ex: "munda" → provavelmente "mudar"), traduza interpretando o erro. Se realmente não conseguir interpretar, retorne o texto original sem alterações.
6. NÃO prefacie a resposta. NÃO escreva "Перевод:" / "Translation:" / qualquer cabeçalho — apenas o texto traduzido.

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


GLOSSARY_EN = """You are a professional translator from Brazilian Portuguese to English, specialized in Mercado Livre e-commerce.

RULES:
1. Translate the provided text from pt-BR to en, preserving the exact technical sense of Mercado Livre's official terminology.
2. Keep links, IDs, URLs, and alphanumeric codes intact.
3. Do not add comments, explanations, or markdown — return ONLY the translation.
4. Preserve line breaks if present.
5. ALWAYS translate, even if the text is short, ambiguous, contains typos, or seems nonsensical. NEVER refuse. NEVER ask for clarification. NEVER ask the user back. If the text contains an obvious typo (e.g., "munda" likely "mudar"), translate interpreting the typo. If you truly cannot interpret it, return the original text unchanged.
6. DO NOT preface the response. DO NOT write "Translation:" / "Перевод:" / any header — only the translated text.

REQUIRED GLOSSARY (pt → en):
- comissão → commission
- tarifa de venda → sales fee
- frete → shipping
- envio → shipment
- anúncio → listing
- publicação → listing
- produto → product
- venda → sale
- comprador → buyer
- vendedor → seller
- reclamação → claim
- mediação → mediation
- reputação → reputation
- conta → account
- moderação → moderation
- faturamento → billing
- fatura → invoice
- cancelamento → cancellation
- reembolso → refund
- estoque → stock
- Full → Full (keep as is)
- MLB → MLB (keep as is)
- permalink → product link
- política → policy
- suspensão → suspension
- desativação → deactivation"""


def _fast_path(text: str, target: TargetLang) -> Optional[str]:
    """Return translated text if we can skip the API call, else None."""
    if not text or not text.strip():
        return text or ""
    if target == "pt":
        # No translation needed — pt is our source.
        return text
    if target == "ru" and _CYRILLIC_RE.search(text):
        return text
    if target == "en" and not _CYRILLIC_RE.search(text) and not _PT_SPECIFIC_RE.search(text):
        # Latin-only text without pt-specific diacritics — assume already English-ish.
        return text
    return None


async def translate(
    text: Optional[str],
    target: TargetLang = "ru",
    http: Optional[httpx.AsyncClient] = None,
) -> str:
    """Translate pt-BR text to `target`. Returns original on failure or fast-path hit.

    target='pt' is a no-op (returns text unchanged).
    """
    raw = text or ""
    if target == "pt":
        return raw

    skip = _fast_path(raw, target)
    if skip is not None:
        return skip

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — returning original text")
        return raw

    model = os.environ.get("TRANSLATE_MODEL", DEFAULT_MODEL)
    glossary = GLOSSARY_RU if target == "ru" else GLOSSARY_EN

    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            # OpenRouter accepts cache_control in `system` content parts for
            # providers that support it (Anthropic via OpenRouter).
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": glossary, "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "user", "content": raw},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional attribution headers recommended by OpenRouter (harmless).
        "HTTP-Referer": "https://app.lsprofit.app",
        "X-Title": "LS Profit App",
    }

    client = http
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=20.0)
    try:
        r = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
        if r.status_code != 200:
            log.warning("OpenRouter %s: %s", r.status_code, r.text[:200])
            return raw
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return raw
        content = choices[0].get("message", {}).get("content")
        result_text: Optional[str] = None
        if isinstance(content, str) and content.strip():
            result_text = content.strip()
        elif isinstance(content, list):
            # Some providers return content as list of parts.
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text")
                    if isinstance(txt, str) and txt.strip():
                        result_text = txt.strip()
                        break
        if not result_text:
            return raw
        # Guard: if the model refused to translate (asked back, said "I can't
        # translate this"), fall back to original text instead of shipping the
        # meta-message to the user's Telegram.
        if _looks_like_refusal(result_text):
            log.info("translate refusal detected — returning original. raw=%r reply=%r", raw[:80], result_text[:120])
            return raw
        return result_text
    except Exception as err:  # noqa: BLE001
        log.exception("translate failed (target=%s): %s", target, err)
        return raw
    finally:
        if owns_client and client is not None:
            await client.aclose()


# ── Backwards-compat shim ─────────────────────────────────────────────────────
# telegram_notify.py currently calls translate_pt_ru(). Keep it working until
# that caller migrates to the generic translate(target=...).

async def translate_pt_ru(text: Optional[str], http: Optional[httpx.AsyncClient] = None) -> str:
    return await translate(text, target="ru", http=http)
