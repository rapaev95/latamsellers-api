import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Armazenagem uploads are parsed once per set of uploads, not once per call.

One cold /finance/reports calls load_armazenagem_report many times (pnl,
balance, every month of the matrix); re-parsing each xlsx every time pushed
the request past the 90s proxy cap.
"""
from v2.legacy import reports as r

STATE = {"fp": "1:100", "parses": 0}


def _setup(monkeypatch):
    r._ARMAZENAGEM_PARSE_CACHE.clear()
    STATE.update(fp="1:100", parses=0)
    monkeypatch.setattr(r, "_armazenagem_fingerprint_db", lambda uid: STATE["fp"])
    monkeypatch.setattr(r, "_load_files_from_db_sync", lambda uid, key: [("a.xlsx", b"x")])

    def _parse(_b):
        STATE["parses"] += 1
        return {"by_sku": {}, "daily_cols": []}
    monkeypatch.setattr(r, "_parse_armazenagem_bytes_daily", _parse)


def test_same_uploads_parse_once(monkeypatch):
    _setup(monkeypatch)
    r._armazenagem_parses_db(6)
    r._armazenagem_parses_db(6)
    assert STATE["parses"] == 1


def test_new_upload_reparses(monkeypatch):
    _setup(monkeypatch)
    r._armazenagem_parses_db(6)
    STATE["fp"] = "1:100,2:200"
    r._armazenagem_parses_db(6)
    assert STATE["parses"] == 2


def test_unknown_fingerprint_is_not_cached(monkeypatch):
    _setup(monkeypatch)
    STATE["fp"] = None  # DB probe failed — don't pin a result we can't validate
    r._armazenagem_parses_db(6)
    r._armazenagem_parses_db(6)
    assert STATE["parses"] == 2
