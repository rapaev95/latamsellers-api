import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Listing editor: the technical sheet it shows, and who may write.

The editor writes to live listings with the cabinet owner's token, so the
access answers are pinned here: a listing outside the scope is 404, a
viewer gets 403, and only then does anything reach ML.
"""
import asyncio

import pytest
from fastapi import HTTPException

from v2.routers import escalar_editor as ed
from v2.services import ml_access


def test_merge_keeps_current_values_and_listing_only_attributes():
    cat = [
        {"id": "BRAND", "name": "Marca", "tags": {"required": True}},
        {"id": "COLOR", "name": "Cor", "tags": {"allow_variations": True}},
        {"id": "MATERIAL", "name": "Material", "tags": {}},
    ]
    item = [
        {"id": "BRAND", "value_id": "1", "value_name": "Genérica"},
        {"id": "ITEM_CONDITION", "name": "Condição", "value_name": "Novo"},
    ]
    merged = {a["id"]: a for a in ed._merge_attributes(cat, item)}
    assert merged["BRAND"]["current"] == {"value_id": "1", "value_name": "Genérica"}
    assert merged["MATERIAL"]["current"] is None
    assert merged["ITEM_CONDITION"]["tags"] == {"read_only": True}


def test_completeness_counts_what_the_seller_fills():
    attrs = ed._merge_attributes(
        [
            {"id": "BRAND", "tags": {"required": True}},
            {"id": "MODEL", "tags": {"required": True}},
            {"id": "MATERIAL", "tags": {}},
            {"id": "GTIN_HIDDEN", "tags": {"hidden": True}},
            {"id": "SIZE", "tags": {"variation_attribute": True}},
        ],
        [{"id": "BRAND", "value_name": "X"}, {"id": "MATERIAL", "value_name": "Couro"}],
    )
    c = ed._completeness(attrs)
    assert (c["filled"], c["total"], c["pct"]) == (2, 3, 67)
    assert c["requiredMissing"] == ["MODEL"]


def _scope(item_ids, may_write, write_ids=None):
    return ml_access.MlScope(
        caller_id=7, owner_id=6, is_owner=False, projects=frozenset({"ARTHUR"}),
        item_ids=frozenset(item_ids), may_write=may_write, write_item_ids=write_ids,
    )


def _resolve(scope):
    async def fake(pool, user, *, project=""):
        return scope
    return fake


class _User:
    id = 7


def test_foreign_listing_is_not_found(monkeypatch):
    monkeypatch.setattr(ml_access, "resolve_scope", _resolve(_scope({"MLB1"}, True)))
    with pytest.raises(HTTPException) as e:
        asyncio.run(ed._scope_for_item(object(), _User(), "MLB2", "", write=False))
    assert e.value.status_code == 404


def test_viewer_cannot_write(monkeypatch):
    monkeypatch.setattr(ml_access, "resolve_scope", _resolve(_scope({"MLB1"}, False)))
    scope, mlb = asyncio.run(ed._scope_for_item(object(), _User(), "1", "", write=False))
    assert mlb == "MLB1"
    with pytest.raises(HTTPException) as e:
        asyncio.run(ed._scope_for_item(object(), _User(), "MLB1", "", write=True))
    assert e.value.status_code == 403


def test_editor_in_one_project_viewer_in_another(monkeypatch):
    monkeypatch.setattr(
        ml_access, "resolve_scope",
        _resolve(_scope({"MLB1", "MLB3"}, True, write_ids=frozenset({"MLB1"}))),
    )
    asyncio.run(ed._scope_for_item(object(), _User(), "MLB1", "", write=True))
    with pytest.raises(HTTPException) as e:
        asyncio.run(ed._scope_for_item(object(), _User(), "MLB3", "", write=True))
    assert e.value.status_code == 403
