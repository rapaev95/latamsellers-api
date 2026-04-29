"""Brevo (ex-Sendinblue) transactional email helper.

Single-purpose: send a templated team invitation email. Future marketing
newsletters can reuse the same client by adding more `_send()` callers.

Configuration via env (Railway):
  BREVO_API_KEY   — required, starts with `xkeysib-...`
  BREVO_FROM      — sender email, must be on a verified domain (default team@lsprofit.app)
  BREVO_FROM_NAME — display name in From header (default "Latamsellers")

If BREVO_API_KEY is missing, send_invitation_email() returns
{ok: False, reason: "no_api_key"} so the caller can still create the
invitation row + return the link for copy-paste fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
DEFAULT_FROM = "team@lsprofit.app"
DEFAULT_FROM_NAME = "Latamsellers"


def _config() -> tuple[Optional[str], str, str]:
    api_key = os.environ.get("BREVO_API_KEY") or None
    sender = os.environ.get("BREVO_FROM") or DEFAULT_FROM
    sender_name = os.environ.get("BREVO_FROM_NAME") or DEFAULT_FROM_NAME
    return api_key, sender, sender_name


def is_configured() -> bool:
    return bool(os.environ.get("BREVO_API_KEY"))


async def _send(
    *,
    to_email: str,
    to_name: str,
    subject: str,
    html: str,
    text: str,
) -> dict[str, Any]:
    api_key, sender, sender_name = _config()
    if not api_key:
        return {"ok": False, "reason": "no_api_key"}

    payload = {
        "sender": {"email": sender, "name": sender_name},
        "to": [{"email": to_email, "name": to_name or to_email}],
        "subject": subject,
        "htmlContent": html,
        "textContent": text,
    }
    headers = {
        "api-key": api_key,
        "content-type": "application/json",
        "accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as cli:
            r = await cli.post(BREVO_ENDPOINT, json=payload, headers=headers)
        if r.status_code in (200, 201):
            data = r.json() if r.content else {}
            return {"ok": True, "message_id": data.get("messageId")}
        log.warning("brevo send failed status=%s body=%s", r.status_code, r.text[:300])
        return {"ok": False, "reason": "api_error", "status": r.status_code, "body": r.text[:300]}
    except httpx.HTTPError as err:
        log.warning("brevo send exception: %s", err)
        return {"ok": False, "reason": "http_error", "error": str(err)}


def _invitation_html(*, project_name: str, role: str, inviter_name: str,
                      inviter_email: str, accept_url: str) -> str:
    role_labels = {
        "admin": "Administrador",
        "analyst": "Analista",
        "viewer": "Visualizador",
    }
    role_label = role_labels.get(role, role)
    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0e1419;font-family:Nunito Sans,Arial,sans-serif;color:#e8ecef">
  <table width="100%" cellspacing="0" cellpadding="0" style="background:#0e1419;padding:40px 0">
    <tr><td align="center">
      <table width="540" cellspacing="0" cellpadding="0" style="background:#161c22;border:1px solid #232b32;border-radius:12px;overflow:hidden">
        <tr><td style="padding:32px 32px 24px 32px">
          <h1 style="margin:0 0 8px 0;font-size:22px;font-weight:800;color:#ffd500">Latamsellers</h1>
          <p style="margin:0 0 24px 0;color:#a8b0b8;font-size:13px">LS Profit App — Mercado Livre analytics</p>

          <h2 style="margin:0 0 16px 0;font-size:18px;font-weight:700;color:#e8ecef">Convite para projeto</h2>

          <p style="margin:0 0 12px 0;font-size:14px;line-height:1.55;color:#d0d6dc">
            <strong>{inviter_name or inviter_email}</strong> te convidou para o projeto
            <strong style="color:#ffd500">{project_name}</strong> como <strong>{role_label}</strong>.
          </p>

          <p style="margin:0 0 24px 0;font-size:14px;line-height:1.55;color:#d0d6dc">
            Aceite o convite e tenha acesso aos dados de vendas, anúncios, posições e qualidade desse projeto.
          </p>

          <table cellspacing="0" cellpadding="0" style="margin:24px 0">
            <tr><td style="background:#ffd500;border-radius:8px">
              <a href="{accept_url}" style="display:inline-block;padding:12px 28px;color:#0e1419;font-weight:700;font-size:14px;text-decoration:none">
                Aceitar convite
              </a>
            </td></tr>
          </table>

          <p style="margin:24px 0 0 0;font-size:11px;color:#7a818a;line-height:1.55">
            Esse link expira em 7 dias. Se você não esperava esse convite, ignore esse email.
          </p>
          <p style="margin:8px 0 0 0;font-size:11px;color:#7a818a;line-height:1.55;word-break:break-all">
            Link direto: {accept_url}
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _invitation_text(*, project_name: str, role: str, inviter_name: str,
                      inviter_email: str, accept_url: str) -> str:
    return (
        f"{inviter_name or inviter_email} convidou você para o projeto "
        f"\"{project_name}\" como {role}.\n\n"
        f"Aceite o convite: {accept_url}\n\n"
        f"O link expira em 7 dias.\n"
        f"-- LS Profit App"
    )


async def send_invitation_email(
    *,
    to_email: str,
    project_name: str,
    role: str,
    inviter_name: str,
    inviter_email: str,
    accept_url: str,
) -> dict[str, Any]:
    """Send the team-invite email. Returns
    `{ok: True, message_id}` on success or
    `{ok: False, reason}` on failure (caller falls back to copy-link UI)."""
    subject = f"Convite — projeto {project_name} no LS Profit App"
    html = _invitation_html(
        project_name=project_name, role=role,
        inviter_name=inviter_name, inviter_email=inviter_email,
        accept_url=accept_url,
    )
    text = _invitation_text(
        project_name=project_name, role=role,
        inviter_name=inviter_name, inviter_email=inviter_email,
        accept_url=accept_url,
    )
    return await _send(
        to_email=to_email,
        to_name="",
        subject=subject,
        html=html,
        text=text,
    )
