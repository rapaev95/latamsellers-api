"""Границы доступа к карточкам внутри одного кабинета Mercado Livre.

Один кабинет держит карточки нескольких партнёров — сегодня это 240 карточек
на восемь проектов. Токен ML, кэш `ml_user_items` и каталог лежат в
пространстве владельца кабинета, поэтому любой запрос партнёра приходится
читать из чужого пространства. Само по себе это не решение о доступе: если
только перепривязать партнёра к владельцу (как делают финансовые эндпоинты),
он получит весь кабинет.

Границей служит SKU:

    партнёр → project_members → проекты → sku_catalog → SKU
            → seller_sku в ml_user_items.raw → item_id

Карточка, SKU которой не разрешается ни в один разрешённый проект, не
принадлежит никому, кроме владельца кабинета. Это относится и к карточкам
вовсе без `seller_sku`, и к карточкам с SKU, которого нет в каталоге: из 240
живых карточек таких семь, и отдавать их «тому, кто спросил» нельзя.

Этот модуль — единственное место, где решается, какие `item_id` доступны
звонящему. HTTP-эндпоинты и MCP-сервер обязаны спрашивать его, а не повторять
логику у себя.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from v2.legacy import config as legacy_config
from v2.legacy import db_storage as legacy_db

# Роли, которым разрешена запись в свои карточки. `viewer` читает и только.
_WRITE_ROLES = frozenset({"owner", "admin", "editor"})


@dataclass(frozen=True)
class MlScope:
    """Что звонящему видно и что он может менять в одном кабинете.

    `item_ids is None` означает «без фильтра» и выставляется только владельцу
    кабинета, который не ограничил себя проектом. Любое другое значение —
    закрытый список: чего в нём нет, того для звонящего не существует.
    """

    caller_id: int
    owner_id: int
    is_owner: bool
    projects: frozenset[str]
    item_ids: frozenset[str] | None
    may_write: bool

    def allows_read(self, item_id: str) -> bool:
        if self.item_ids is None:
            return True
        return (item_id or "").strip() in self.item_ids

    def allows_write(self, item_id: str) -> bool:
        return self.may_write and self.allows_read(item_id)


# --- загрузчики, вынесенные ради тестируемости ------------------------------
# Подменяются в tests/test_ml_access.py; в проде читают БД.

def _bind(user_id: int) -> None:
    legacy_db.set_current_user_id(user_id)


def _attribute_items(owner_id: int) -> dict[str, str]:
    """item_id → проект, по правилам атрибуции владельца кабинета.

    Намеренно идёт через `get_project_by_sku` — ту же функцию, которой P&L
    решает, чья это выручка. Своя, упрощённая атрибуция здесь была бы
    расхождением: первая версия смотрела только в каталог и прятала от
    партнёра 12 его же карточек, привязанных через `mlb_fallback`. Правило —
    кому карточка приносит деньги, тот её и видит.

    Карточки, которые атрибуция не разрешает, получают `NAO_CLASSIFICADO` и
    остаются только у владельца.
    """
    caller_before = legacy_db._current_user_id()
    try:
        _bind(owner_id)
        item_sku = legacy_config.mlb_to_seller_sku()
        out: dict[str, str] = {}
        for item_id, sku in item_sku.items():
            iid = (item_id or "").strip()
            if not iid:
                continue
            out[iid] = legacy_config.get_project_by_sku(sku or "", iid)
        return out
    finally:
        legacy_db.set_current_user_id(caller_before)


def _load_own_projects(caller_id: int) -> set[str]:
    """Проекты в собственном пространстве звонящего."""
    caller_before = legacy_db._current_user_id()
    try:
        _bind(caller_id)
        return set(legacy_config.load_projects() or {})
    finally:
        legacy_db.set_current_user_id(caller_before)


async def _load_memberships(pool, caller_id: int) -> list[dict[str, Any]]:
    if pool is None:
        return []
    try:
        from v2.services import project_members as pm_svc

        return list(await pm_svc.list_inherited_projects(pool, caller_id))
    except Exception:
        return []


# --- разрешение области ----------------------------------------------------

def _items_for_projects(
    owner_id: int, allowed_projects: Iterable[str],
) -> frozenset[str]:
    """item_id карточек, атрибутированных в разрешённые проекты."""
    allowed = {p for p in allowed_projects if p}
    allowed.discard("NAO_CLASSIFICADO")
    if not allowed:
        return frozenset()
    return frozenset(
        item_id
        for item_id, project in _attribute_items(owner_id).items()
        if project in allowed
    )


async def resolve_scope(pool, user, *, project: str = "") -> MlScope:
    """Область доступа звонящего к карточкам кабинета.

    `project` сужает область до одного проекта — и для партнёра, и для
    владельца. Запрос проекта, к которому звонящий не имеет отношения, даёт
    пустую область, а не ошибку: пустой ответ нельзя спутать с утечкой.
    """
    caller_id = int(getattr(user, "id", 0) or 0)
    wanted = (project or "").strip()

    own_projects = _load_own_projects(caller_id)
    memberships = await _load_memberships(pool, caller_id)
    member_roles: dict[str, tuple[int, str]] = {}
    for m in memberships:
        name = (m.get("project_name") or "").strip()
        owner_id = m.get("owner_id")
        if not name or not owner_id or not m.get("synced", True):
            continue
        member_roles.setdefault(name, (int(owner_id), str(m.get("role") or "viewer")))

    # Один проект: владелец пространства определяется по этому проекту.
    if wanted:
        if wanted in own_projects:
            return MlScope(
                caller_id=caller_id,
                owner_id=caller_id,
                is_owner=True,
                projects=frozenset({wanted}),
                item_ids=_items_for_projects(caller_id, [wanted]),
                may_write=True,
            )
        if wanted in member_roles:
            owner_id, role = member_roles[wanted]
            return MlScope(
                caller_id=caller_id,
                owner_id=owner_id,
                is_owner=False,
                projects=frozenset({wanted}),
                item_ids=_items_for_projects(owner_id, [wanted]),
                may_write=role in _WRITE_ROLES,
            )
        return MlScope(
            caller_id=caller_id,
            owner_id=caller_id,
            is_owner=False,
            projects=frozenset(),
            item_ids=frozenset(),
            may_write=False,
        )

    # Без проекта: владелец кабинета видит его целиком, включая карточки без
    # привязки — они его и есть.
    if own_projects:
        return MlScope(
            caller_id=caller_id,
            owner_id=caller_id,
            is_owner=True,
            projects=frozenset(own_projects),
            item_ids=None,
            may_write=True,
        )

    if not member_roles:
        return MlScope(
            caller_id=caller_id,
            owner_id=caller_id,
            is_owner=False,
            projects=frozenset(),
            item_ids=frozenset(),
            may_write=False,
        )

    # Партнёр без своих проектов. Приглашения от нескольких владельцев — это
    # несколько кабинетов; берём наименьший owner_id, как и остальной код, и
    # не смешиваем кабинеты в одной области.
    owner_id = min(o for o, _ in member_roles.values())
    mine = {
        name: role for name, (o, role) in member_roles.items() if o == owner_id
    }
    return MlScope(
        caller_id=caller_id,
        owner_id=owner_id,
        is_owner=False,
        projects=frozenset(mine),
        item_ids=_items_for_projects(owner_id, mine),
        may_write=any(r in _WRITE_ROLES for r in mine.values()),
    )


def filter_rows(
    scope: MlScope, rows: Sequence[dict[str, Any]], key: str = "item_id",
) -> list[dict[str, Any]]:
    """Отбросить строки, которых звонящему видеть нельзя."""
    if scope.item_ids is None:
        return list(rows)
    return [r for r in rows if scope.allows_read(str(r.get(key) or ""))]
