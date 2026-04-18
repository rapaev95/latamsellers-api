"""Project resolution: SKU → project name.

Mirrors the priority chain from `_admin/config.py:get_project_by_sku()`:
  1. Direct match in user's sku_catalog (DB key `f2_sku_catalog`, falls back to `sku_catalog`)
  2. project.sku_prefixes (DB key `f2_projects` → `projects`)
  3. project.mlb_fallback
  4. If exactly one project — return it
  5. '' (NAO_CLASSIFICADO)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from v2.storage import user_storage

# Repo-root-relative fallback paths (used when DB row missing — local dev convenience)
_BASE = Path(__file__).resolve().parents[3]  # _admin/api/v2/services → _admin
_FALLBACK_PROJECTS_JSON = _BASE / "projects_db.json"
_FALLBACK_CATALOG_JSON = _BASE.parent / "_data" / "sku_catalog.json"


@dataclass
class ProjectResolver:
    projects: dict[str, dict]
    catalog: dict[str, dict]  # SKU (uppercase) → catalog item

    def resolve(self, sku: str, mlb: str = "") -> str:
        sku_trim = (sku or "").strip()
        if not sku_trim:
            return ""
        sku_up = sku_trim.upper()

        # 1) Direct catalog match
        cat = self.catalog.get(sku_up)
        if cat:
            proj = (cat.get("project") or "").strip()
            if proj and proj != "NAO_CLASSIFICADO":
                return proj

        # 2) Prefix matching
        sku_lower = sku_trim.lower()
        mlb_trim = (mlb or "").strip()
        for pid, p in self.projects.items():
            for prefix in (p.get("sku_prefixes") or []):
                if prefix and sku_lower.startswith(prefix.lower()):
                    return pid
            for fb in (p.get("mlb_fallback") or []):
                if fb and mlb_trim == fb:
                    return pid

        # 3) Single-project fallback
        if len(self.projects) == 1:
            return next(iter(self.projects.keys()))
        return ""

    @property
    def project_names(self) -> list[str]:
        return sorted(n for n in self.projects.keys() if n and n != "NAO_CLASSIFICADO")


def _read_json_file(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def load_resolver(pool: asyncpg.Pool, user_id: int) -> ProjectResolver:
    """Build a resolver for one user.

    Reads f2_* first, then non-prefixed Streamlit keys, then JSON fallbacks.
    """
    # ── Projects ─────────────────────────────────────────────────
    projects_raw: Any | None = await user_storage.get(pool, user_id, "projects")
    if projects_raw is None:
        projects_raw = await user_storage.get_legacy(pool, user_id, "projects")
    if projects_raw is None:
        projects_raw = _read_json_file(_FALLBACK_PROJECTS_JSON) or {}
    projects = projects_raw if isinstance(projects_raw, dict) else {}

    # ── SKU catalog ──────────────────────────────────────────────
    catalog_raw: Any | None = await user_storage.get(pool, user_id, "sku_catalog")
    if catalog_raw is None:
        catalog_raw = await user_storage.get_legacy(pool, user_id, "sku_catalog")
    if catalog_raw is None:
        catalog_raw = _read_json_file(_FALLBACK_CATALOG_JSON) or {}

    if isinstance(catalog_raw, list):
        items = catalog_raw
    elif isinstance(catalog_raw, dict):
        items = catalog_raw.get("items") or []
    else:
        items = []

    catalog: dict[str, dict] = {}
    for it in items:
        if isinstance(it, dict) and it.get("sku"):
            catalog[str(it["sku"]).upper()] = it

    return ProjectResolver(projects=projects, catalog=catalog)
