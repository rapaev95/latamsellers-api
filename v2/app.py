"""Aggregator router — all v2 endpoints mounted under `/api/v2`.

Usage in main.py:
    from v2.app import router as v2_router
    app.include_router(v2_router)
"""
from fastapi import APIRouter

from v2.routers import admin_users as admin_users_router
from v2.routers import ads as ads_router
from v2.routers import auth as auth_router
from v2.routers import escalar as escalar_router
from v2.routers import escalar_editor as escalar_editor_router
from v2.routers import finance as finance_router
from v2.routers import health as health_router
from v2.routers import mcp as mcp_router
from v2.routers import migrations as migrations_router
from v2.routers import ml_oauth as ml_oauth_router
from v2.routers import positions as positions_router
from v2.routers import team as team_router

router = APIRouter(prefix="/api/v2")
router.include_router(health_router.router)
router.include_router(auth_router.router)
router.include_router(admin_users_router.router)
router.include_router(migrations_router.router)
router.include_router(ads_router.router)
router.include_router(escalar_router.router)
router.include_router(escalar_editor_router.router)
router.include_router(finance_router.router)
router.include_router(ml_oauth_router.router)
# Positions router unregistered 2026-10-04 — the ML account behind the
# scraper is dead. Endpoints stay in the tree but are not reachable, so
# nothing can trigger a scrape by accident.
# router.include_router(positions_router.router)
router.include_router(team_router.router)
router.include_router(mcp_router.router)
