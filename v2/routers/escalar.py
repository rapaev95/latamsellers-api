"""ABC products + snooze endpoints for the Escalar (Promotion) section."""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, Query

from v2.deps import CurrentUser, current_user, get_pool
from v2.parsers import db_loader
from v2.schemas.escalar import EscalarProductsOut, SnoozeIn, SnoozeOut
from v2.services import abc, projects
from v2.settings import get_settings
from v2.storage import user_storage

router = APIRouter(prefix="/escalar", tags=["escalar"])

SNOOZE_KEY = "escalar_snoozed_skus"


def _parse_days(raw: Optional[str]) -> Union[int, str]:
    if raw is None or raw == "":
        return 30
    if raw == "all":
        return "all"
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return 30


@router.get("/products", response_model=EscalarProductsOut)
async def get_products(
    days: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    days_v = _parse_days(days)
    snoozed = set(await user_storage.get(pool, user.id, SNOOZE_KEY) or [])
    resolver = await projects.load_resolver(pool, user.id)

    vendas_rows = None
    storage_map = None
    stock_full_map = None
    vendas_filenames = None
    if get_settings().storage_mode == "db" and pool is not None:
        vendas_rows = await db_loader.load_user_vendas(pool, user.id)
        storage_map = await db_loader.load_user_armazenagem(pool, user.id)
        stock_full_map = await db_loader.load_user_stock_full(pool, user.id)
        vendas_filenames = await db_loader.list_user_vendas_filenames(pool, user.id)

    summary = abc.aggregate(
        days=days_v,
        project=project or "",
        snoozed_skus=snoozed,
        resolver=resolver,
        vendas_rows=vendas_rows,
        storage_map=storage_map,
        stock_full_map=stock_full_map,
        vendas_filenames=vendas_filenames,
    )
    return {
        "products": summary["products"],
        "hasData": len(summary["products"]) > 0,
        "meta": summary["meta"],
    }


@router.post("/snooze", response_model=SnoozeOut)
async def post_snooze(
    body: SnoozeIn,
    user: CurrentUser = Depends(current_user),
    pool=Depends(get_pool),
):
    current = set(await user_storage.get(pool, user.id, SNOOZE_KEY) or [])
    if body.snoozed:
        current.add(body.sku)
    else:
        current.discard(body.sku)
    new_list = sorted(current)
    await user_storage.put(pool, user.id, SNOOZE_KEY, new_list)
    return {"snoozedSkus": new_list}
