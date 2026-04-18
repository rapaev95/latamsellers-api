"""Verbatim-copied Streamlit logic, adapted for FastAPI v2.

What lives here:
  - db_storage.py — sync psycopg2 wrapper (replaces the Streamlit version with
    a ContextVar-based current_user_id instead of st.session_state).
  - config.py — projects/rules/path constants. Copied from _admin/config.py
    with two changes: (a) BASE_DIR anchored to _admin/ via parents[3];
    (b) Streamlit session caching stripped (functools.lru_cache instead).
  - sku_catalog.py — copied as-is.
  - finance.py — copied as-is (compute_pnl/compute_cashflow/compute_balance).
  - reports.py — copied as-is (16 ML parsers + aggregators).

The originals in _admin/*.py keep working for the live Streamlit app.
This package is for FastAPI v2 endpoints only.
"""
