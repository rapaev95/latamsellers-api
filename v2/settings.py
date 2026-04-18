"""v2 runtime settings — environment-driven."""
from __future__ import annotations

import os
from typing import List


class Settings:
    """Read-only settings. Plain class to avoid hard dep on pydantic-settings."""

    def __init__(self) -> None:
        self.database_url: str | None = (
            os.environ.get("DATABASE_URL")
            or os.environ.get("DATABASE_PUBLIC_URL")
        )
        # CORS origins permitted with credentials. Add new ones via env CORS_ORIGINS=a,b,c
        env_origins = os.environ.get("CORS_ORIGINS", "")
        defaults = [
            "http://localhost:3001",
            "http://localhost:3000",
            "https://lsprofit.app",
            "https://www.lsprofit.app",
            "https://app.lsprofit.app",
        ]
        if env_origins:
            extra = [o.strip() for o in env_origins.split(",") if o.strip()]
            self.cors_origins: List[str] = list(dict.fromkeys(defaults + extra))
        else:
            self.cors_origins = defaults
        self.environment: str = os.environ.get("ENVIRONMENT", "development")
        # Source of vendas/armazenagem data for per-user aggregators.
        #   "fs"  → legacy: shared filesystem dirs `vendas/` + `_data/armazenagem/`.
        #   "db"  → per-user: `uploads.file_bytes` keyed by user_id.
        # Default stays "fs" until the FS→DB backfill has been run in prod.
        self.storage_mode: str = os.environ.get("LS_STORAGE_MODE", "fs").strip().lower()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
