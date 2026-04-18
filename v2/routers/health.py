"""GET /api/v2/health — liveness probe (no auth, no DB)."""
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "ok": True,
        "version": "v2",
        "time": datetime.now(timezone.utc).isoformat(),
    }
