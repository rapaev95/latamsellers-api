"""Brazilian tax regime computation — Simples Nacional (Anexos I/II/III) + Lucro Presumido.

Pure functions — no IO, no date aggregation. Caller passes (rbt12, bruto_month).

Anexos I–III tiers mirror LC 123/2006 + LC 155/2016. Anexo I taken 1:1 from
super-calculator-app/components/simulators/das-calculator/DasCalculator.tsx:21-28.

Effective aliquot = max(((RBT12 * nominal/100) - parcela_deduzir) / RBT12 * 100, 0).
Edge cases:
  - rbt12 <= 0 → nominal of faixa 1 (fresh business, first month)
  - rbt12 > 4.8M → nominal of faixa 6, exceed_limit=True (Simples cap exceeded;
    caller may want to flag for migration to Lucro Presumido)
"""
from __future__ import annotations

from typing import Any, Optional


# ── Simples Nacional tables (LC 123/2006) ────────────────────────────────────

ANEXO_I_TIERS = [
    {"faixa": 1, "max_rbt12": 180_000,   "aliquota_nominal": 4.00,  "parcela_deduzir": 0},
    {"faixa": 2, "max_rbt12": 360_000,   "aliquota_nominal": 7.30,  "parcela_deduzir": 5_940},
    {"faixa": 3, "max_rbt12": 720_000,   "aliquota_nominal": 9.50,  "parcela_deduzir": 13_860},
    {"faixa": 4, "max_rbt12": 1_800_000, "aliquota_nominal": 10.70, "parcela_deduzir": 22_500},
    {"faixa": 5, "max_rbt12": 3_600_000, "aliquota_nominal": 14.30, "parcela_deduzir": 87_300},
    {"faixa": 6, "max_rbt12": 4_800_000, "aliquota_nominal": 19.00, "parcela_deduzir": 378_000},
]

ANEXO_II_TIERS = [
    {"faixa": 1, "max_rbt12": 180_000,   "aliquota_nominal": 4.50,  "parcela_deduzir": 0},
    {"faixa": 2, "max_rbt12": 360_000,   "aliquota_nominal": 7.80,  "parcela_deduzir": 5_940},
    {"faixa": 3, "max_rbt12": 720_000,   "aliquota_nominal": 10.00, "parcela_deduzir": 13_860},
    {"faixa": 4, "max_rbt12": 1_800_000, "aliquota_nominal": 11.20, "parcela_deduzir": 22_500},
    {"faixa": 5, "max_rbt12": 3_600_000, "aliquota_nominal": 14.70, "parcela_deduzir": 85_500},
    {"faixa": 6, "max_rbt12": 4_800_000, "aliquota_nominal": 30.00, "parcela_deduzir": 720_000},
]

ANEXO_III_TIERS = [
    {"faixa": 1, "max_rbt12": 180_000,   "aliquota_nominal": 6.00,  "parcela_deduzir": 0},
    {"faixa": 2, "max_rbt12": 360_000,   "aliquota_nominal": 11.20, "parcela_deduzir": 9_360},
    {"faixa": 3, "max_rbt12": 720_000,   "aliquota_nominal": 13.50, "parcela_deduzir": 17_640},
    {"faixa": 4, "max_rbt12": 1_800_000, "aliquota_nominal": 16.00, "parcela_deduzir": 35_640},
    {"faixa": 5, "max_rbt12": 3_600_000, "aliquota_nominal": 21.00, "parcela_deduzir": 125_640},
    {"faixa": 6, "max_rbt12": 4_800_000, "aliquota_nominal": 33.00, "parcela_deduzir": 648_000},
]

ANEXOS = {
    "I": ANEXO_I_TIERS,
    "II": ANEXO_II_TIERS,
    "III": ANEXO_III_TIERS,
}

# ── ICMS interstate rates (approximate 2025, alíquota interna) ───────────────
# Used for Lucro Presumido to reflect state-level sales tax.
ICMS_BY_STATE: dict[str, float] = {
    "AC": 19.0, "AL": 19.0, "AP": 18.0, "AM": 20.0, "BA": 19.0,
    "CE": 18.0, "DF": 18.0, "ES": 17.0, "GO": 17.0, "MA": 20.0,
    "MT": 17.0, "MS": 17.0, "MG": 18.0, "PA": 19.0, "PB": 18.0,
    "PR": 19.5, "PE": 18.0, "PI": 21.0, "RJ": 20.0, "RN": 18.0,
    "RS": 17.0, "RO": 17.5, "RR": 20.0, "SC": 17.0, "SP": 18.0,
    "SE": 19.0, "TO": 18.0,
}
ICMS_FALLBACK_PCT = 17.0

# Lucro Presumido federal combined: IRPJ 1.2 + CSLL 1.08 + PIS 0.65 + COFINS 3
LUCRO_PRESUMIDO_BASE_PCT = 5.93


# ── Core functions ──────────────────────────────────────────────────────────

def compute_simples_effective(rbt12: float, anexo: str = "I") -> dict[str, Any]:
    """Compute effective Simples Nacional rate for a given RBT12 and anexo.

    Returns a dict with faixa, aliquota_nominal, parcela_deduzir, effective_pct,
    exceed_limit. rbt12 is clamped to ≥0; >4.8M returns faixa 6 with exceed_limit.
    """
    tiers = ANEXOS.get(anexo, ANEXO_I_TIERS)
    rbt12 = max(0.0, float(rbt12 or 0))

    # Fresh business — no 12-month history yet → use faixa 1 nominal
    if rbt12 <= 0:
        t = tiers[0]
        return {
            "faixa": t["faixa"],
            "aliquota_nominal": t["aliquota_nominal"],
            "parcela_deduzir": t["parcela_deduzir"],
            "effective_pct": t["aliquota_nominal"],
            "exceed_limit": False,
        }

    # Simples cap — 4.8M. Above it the business must migrate to Lucro Presumido/Real.
    # Gracefully produce a numeric DAS at faixa 6 nominal (not effective).
    if rbt12 > tiers[-1]["max_rbt12"]:
        t = tiers[-1]
        return {
            "faixa": t["faixa"],
            "aliquota_nominal": t["aliquota_nominal"],
            "parcela_deduzir": t["parcela_deduzir"],
            "effective_pct": t["aliquota_nominal"],
            "exceed_limit": True,
        }

    for t in tiers:
        if rbt12 <= t["max_rbt12"]:
            eff = ((rbt12 * t["aliquota_nominal"] / 100.0) - t["parcela_deduzir"]) / rbt12 * 100.0
            return {
                "faixa": t["faixa"],
                "aliquota_nominal": t["aliquota_nominal"],
                "parcela_deduzir": t["parcela_deduzir"],
                "effective_pct": max(eff, 0.0),
                "exceed_limit": False,
            }

    # Unreachable (loop covers all faixas via max_rbt12 + >cap branch)
    return {
        "faixa": tiers[-1]["faixa"],
        "aliquota_nominal": tiers[-1]["aliquota_nominal"],
        "parcela_deduzir": tiers[-1]["parcela_deduzir"],
        "effective_pct": tiers[-1]["aliquota_nominal"],
        "exceed_limit": True,
    }


def compute_lucro_presumido_effective(state: Optional[str]) -> dict[str, Any]:
    """5.93% federal + ICMS по штату. Unknown/blank state → fallback 17%."""
    uf = (state or "").strip().upper()[:2]
    icms = ICMS_BY_STATE.get(uf, ICMS_FALLBACK_PCT)
    return {
        "base_pct": LUCRO_PRESUMIDO_BASE_PCT,
        "icms_pct": icms,
        "effective_pct": LUCRO_PRESUMIDO_BASE_PCT + icms,
        "state": uf or None,
    }


def resolve_tax_settings(
    project_meta: dict,
    all_projects: Optional[dict] = None,
) -> dict[str, Any]:
    """Determine effective tax_regime + anexo with company-level inheritance.

    Правило пользователя: проекты одной компании (same company_cnpj) делят
    налоговый режим и Anexo. Исключение — проекты с type == 'services':
    они всегда используют Anexo III независимо от наследования.

    Алгоритм:
    1. Если у проекта есть tax_regime — используется свой (override).
    2. Иначе — ищем любой другой проект с тем же company_cnpj,
       у которого tax_regime задан, и наследуем от него regime и anexo.
    3. Для simples_nacional: если type == 'services' — форсим Anexo III.

    Возвращает обогащённый meta-dict с ключами tax_regime, simples_anexo
    и inherited_from (для отладки и UI-подсказок).
    """
    cnpj = (project_meta.get("company_cnpj") or "").strip()
    own_regime = (project_meta.get("tax_regime") or "").strip().lower() or None
    own_anexo = (project_meta.get("simples_anexo") or "").strip().upper() or None
    ptype = (project_meta.get("type") or "").strip().lower()

    regime = own_regime
    anexo = own_anexo
    source_regime = "own" if own_regime else None
    source_anexo = "own" if own_anexo else None
    inherited_pid: Optional[str] = None

    # 1. Наследование от компании если своего режима нет
    if not regime and cnpj and all_projects:
        for pid, other in all_projects.items():
            if not isinstance(other, dict):
                continue
            if (other.get("company_cnpj") or "").strip() != cnpj:
                continue
            other_regime = (other.get("tax_regime") or "").strip().lower() or None
            if not other_regime:
                continue
            regime = other_regime
            source_regime = "inherited"
            inherited_pid = pid
            if not anexo:
                other_anexo = (other.get("simples_anexo") or "").strip().upper() or None
                if other_anexo:
                    anexo = other_anexo
                    source_anexo = "inherited"
            break

    # 2. Services проекты → Anexo III (даже если компания на I)
    if regime == "simples_nacional" and ptype == "services":
        if anexo != "III":
            anexo = "III"
            source_anexo = "forced_services"

    merged = dict(project_meta)
    merged["tax_regime"] = regime
    merged["simples_anexo"] = anexo
    merged["_tax_inheritance"] = {
        "source_regime": source_regime,
        "source_anexo": source_anexo,
        "inherited_from": inherited_pid,
    }
    return merged


def compute_das(project_meta: dict, bruto_month: float, rbt12: float) -> dict[str, Any]:
    """Single DAS entry point.

    Reads `tax_regime`, `simples_anexo`, `company_state` from project_meta.
    Returns unified dict consumed by compute_pnl, balance, matrix, и клиентом
    для рендера бейджа.

    Backward compatibility: if tax_regime is empty/None, falls back to legacy
    4.5% (matches historical behaviour for projects that predate this feature).
    """
    regime = (project_meta.get("tax_regime") or "").lower().strip()
    bruto_month = max(0.0, float(bruto_month or 0))
    rbt12 = max(0.0, float(rbt12 or 0))

    if regime == "simples_nacional":
        anexo = (project_meta.get("simples_anexo") or "I").strip().upper()
        if anexo not in ANEXOS:
            anexo = "I"
        simples = compute_simples_effective(rbt12, anexo)
        eff = simples["effective_pct"]
        das_brl = round(bruto_month * eff / 100.0, 2)
        faixa = simples["faixa"]
        display_pt = f"Simples Anexo {anexo} · Faixa {faixa} ({eff:.2f}%)"
        display_ru = f"Simples Annex {anexo} · Фаикса {faixa} ({eff:.2f}%)"
        return {
            "das_brl": das_brl,
            "effective_pct": eff,
            "regime": "simples_nacional",
            "anexo": anexo,
            "faixa": faixa,
            "aliquota_nominal": simples["aliquota_nominal"],
            "parcela_deduzir": simples["parcela_deduzir"],
            "icms_pct": None,
            "state": None,
            "rbt12": rbt12,
            "display_pt": display_pt,
            "display_ru": display_ru,
            "exceed_limit": simples["exceed_limit"],
            "inheritance": project_meta.get("_tax_inheritance"),
        }

    if regime == "lucro_presumido":
        lp = compute_lucro_presumido_effective(project_meta.get("company_state"))
        eff = lp["effective_pct"]
        das_brl = round(bruto_month * eff / 100.0, 2)
        state = lp["state"] or "—"
        display_pt = f"Lucro Presumido {lp['base_pct']}% + ICMS {lp['icms_pct']:.1f}% ({state})"
        display_ru = f"Lucro Presumido {lp['base_pct']}% + ICMS {lp['icms_pct']:.1f}% ({state})"
        return {
            "das_brl": das_brl,
            "effective_pct": eff,
            "regime": "lucro_presumido",
            "anexo": None,
            "faixa": None,
            "aliquota_nominal": None,
            "parcela_deduzir": None,
            "icms_pct": lp["icms_pct"],
            "state": lp["state"],
            "rbt12": rbt12,
            "display_pt": display_pt,
            "display_ru": display_ru,
            "exceed_limit": False,
            "inheritance": project_meta.get("_tax_inheritance"),
        }

    # Legacy fallback — behaviour identical to pre-feature code.
    eff = 4.5
    das_brl = round(bruto_month * eff / 100.0, 2)
    return {
        "das_brl": das_brl,
        "effective_pct": eff,
        "regime": "legacy",
        "anexo": None,
        "faixa": None,
        "aliquota_nominal": None,
        "parcela_deduzir": None,
        "icms_pct": None,
        "state": None,
        "rbt12": rbt12,
        "display_pt": "Legado Simples 4.5%",
        "display_ru": "Legacy Simples 4.5%",
        "exceed_limit": False,
        "inheritance": project_meta.get("_tax_inheritance"),
    }
