"""Password-unlock helpers for protected Excel / PDF uploads.

Port of `_admin/unlocker.py` — tries a list of known passwords against the
encrypted file and returns the decrypted bytes, so the rest of the pipeline
treats them as regular uploads.

Known passwords live per-user in `user_data.f2_known_passwords` (JSONB list);
a hardcoded fallback covers the bootstrap case for new accounts.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

# Seeded with Streamlit's defaults (C6 Bank PDFs). Users add their own via
# the prompt on upload when a password isn't in their personal list.
DEFAULT_KNOWN_PASSWORDS: list[str] = [
    "716816",  # C6 Bank
    "456595",  # C6 Bank
]


def _user_passwords_key() -> str:
    return "f2_known_passwords"


def load_known_passwords() -> list[str]:
    """Per-user list + defaults (deduped, user-first)."""
    from .db_storage import db_load
    saved = db_load(_user_passwords_key())
    user_list = [str(p) for p in saved] if isinstance(saved, list) else []
    seen: set[str] = set()
    out: list[str] = []
    for p in user_list + DEFAULT_KNOWN_PASSWORDS:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def add_known_password(password: str) -> int:
    """Persist a password that just worked so next upload is 1-shot."""
    from .db_storage import db_load, db_save
    pw = (password or "").strip()
    if not pw:
        return 0
    saved = db_load(_user_passwords_key())
    user_list = [str(p) for p in saved] if isinstance(saved, list) else []
    if pw in user_list:
        return len(user_list)
    user_list.append(pw)
    db_save(_user_passwords_key(), user_list)
    return len(user_list)


# ── Excel ──────────────────────────────────────────────────────────────────

def unlock_excel(file_bytes: bytes, passwords: list[str]) -> tuple[Optional[bytes], Optional[str]]:
    try:
        import msoffcrypto
    except ImportError:
        return None, None
    for pwd in passwords:
        try:
            file_in = io.BytesIO(file_bytes)
            file_out = io.BytesIO()
            of = msoffcrypto.OfficeFile(file_in)
            of.load_key(password=pwd)
            of.decrypt(file_out)
            file_out.seek(0)
            return file_out.read(), pwd
        except Exception:
            continue
    return None, None


def is_excel_encrypted(file_bytes: bytes) -> bool:
    try:
        import msoffcrypto
        of = msoffcrypto.OfficeFile(io.BytesIO(file_bytes))
        return bool(of.is_encrypted())
    except Exception:
        return False


# ── PDF ────────────────────────────────────────────────────────────────────

def unlock_pdf(file_bytes: bytes, passwords: list[str]) -> tuple[Optional[bytes], Optional[str]]:
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        return None, None
    for pwd in passwords:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if not reader.is_encrypted:
                return file_bytes, "(sem senha)"
            if not reader.decrypt(pwd):
                continue
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            output = io.BytesIO()
            writer.write(output)
            output.seek(0)
            return output.read(), pwd
        except Exception:
            continue
    return None, None


def is_pdf_encrypted(file_bytes: bytes) -> bool:
    try:
        from PyPDF2 import PdfReader
        return bool(PdfReader(io.BytesIO(file_bytes)).is_encrypted)
    except Exception:
        return False


# ── Public interface ──────────────────────────────────────────────────────

def try_unlock(
    file_bytes: bytes,
    filename: str,
    extra_password: Optional[str] = None,
) -> tuple[Optional[bytes], Optional[str], str]:
    """Auto-detect file type and try to unlock.

    Priority: user-supplied `extra_password` (if given) → user's DB list → defaults.
    Returns (unlocked_bytes, password_used, status).
    status ∈ {"not_encrypted", "unlocked", "failed", "unsupported"}.
    """
    ext = Path(filename).suffix.lower()
    passwords = load_known_passwords()
    if extra_password:
        passwords = [extra_password.strip()] + passwords

    if ext in (".xlsx", ".xls"):
        if not is_excel_encrypted(file_bytes):
            return file_bytes, None, "not_encrypted"
        data, pwd = unlock_excel(file_bytes, passwords)
        return (data, pwd, "unlocked") if data else (None, None, "failed")

    if ext == ".pdf":
        if not is_pdf_encrypted(file_bytes):
            return file_bytes, None, "not_encrypted"
        data, pwd = unlock_pdf(file_bytes, passwords)
        return (data, pwd, "unlocked") if data else (None, None, "failed")

    return None, None, "unsupported"
