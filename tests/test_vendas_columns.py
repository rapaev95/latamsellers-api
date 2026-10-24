import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""ML renamed «Data da venda» → «Data de venda» in 2026-10. The newest export
wins the dedupe, so without the rename every sale in it lost its date and
dropped out of P&L and the matrix.
"""
import pandas as pd
from v2.legacy import reports as r


def _with(monkeypatch, df):
    monkeypatch.setattr(r, "_parse_one_vendas_file_raw", lambda fn, blob: df)
    return r._parse_one_vendas_file("x.xlsx", b"")


def test_new_spelling_is_renamed(monkeypatch):
    out = _with(monkeypatch, pd.DataFrame({"N.º de venda": [1], "Data de venda": ["24 de outubro de 2026"]}))
    assert "Data da venda" in out.columns and "Data de venda" not in out.columns


def test_old_spelling_untouched(monkeypatch):
    out = _with(monkeypatch, pd.DataFrame({"N.º de venda": [1], "Data da venda": ["25 de julho de 2026"]}))
    assert list(out.columns) == ["N.º de venda", "Data da venda"]


def test_old_and_new_exports_share_one_column(monkeypatch):
    new = _with(monkeypatch, pd.DataFrame({"N.º de venda": [1], "Data de venda": ["a"]}))
    old = _with(monkeypatch, pd.DataFrame({"N.º de venda": [2], "Data da venda": ["b"]}))
    both = pd.concat([new, old], ignore_index=True)
    assert both["Data da venda"].notna().all()
