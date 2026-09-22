import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Fingerprint must react to the dynamic user_data key families.

Before prefix matching, editing an FX-rate override or hiding a transfer left
the fingerprint untouched, so the cache kept serving pre-edit numbers — which
is why the UI had to force `fresh=true` after every such edit.
"""
from datetime import datetime
from v2.services import finance_cache as fc

STATE = {"user_data": {}, "uploads": {}}
LAST_SQL = []


class _Cur:
    def __init__(self): self._rows = []
    def execute(self, sql, params):
        LAST_SQL.append(" ".join(sql.split()))
        if "FROM uploads" in sql:
            self._rows = list(STATE["uploads"].items())
        else:
            literal, patterns = params[1], params[2]
            self._rows = [
                (k, v) for k, v in STATE["user_data"].items()
                if k in literal or any(k.startswith(p[:-1]) for p in patterns)
            ]
    def fetchall(self): return self._rows
    def close(self): pass


class _Conn:
    def cursor(self): return _Cur()
    def close(self): pass


fc._connect = lambda: _Conn()
TS = datetime(2026, 9, 22, 10, 0, 0)

def fp():
    return fc.compute_fingerprint(7)[0]

# A dynamic key that the old literal list could never have named.
STATE["user_data"] = {"f2_projects": TS}
base = fp()

STATE["user_data"]["f2_services_hidden_transfers_ESTONIA"] = TS
after_add = fp()
assert after_add != base, "adding a hidden-transfers key must invalidate"

STATE["user_data"]["f2_services_hidden_transfers_ESTONIA"] = datetime(2026, 9, 22, 10, 5)
after_edit = fp()
assert after_edit != after_add, "editing it must invalidate"

del STATE["user_data"]["f2_services_hidden_transfers_ESTONIA"]
after_delete = fp()
assert after_delete == base, "deleting it must return to the pre-add fingerprint"
print("hidden_transfers: add/edit/delete all invalidate ✓")

# Every family from the audit, including the legacy per-upload classifications.
for key in (
    "f2_classifications_grouped_extrato_mp",
    "f2_classifications_48219",
    "f2_services_invoice_rate_overrides_ESTONIA",
    "f2_services_invoice_payment_overrides_ESTONIA",
    "f2_services_transfer_edits_ESTONIA",
):
    before = fp()
    STATE["user_data"][key] = TS
    assert fp() != before, f"{key} did not invalidate"
    del STATE["user_data"][key]
print("all five previously-invisible families invalidate ✓")

# Unrelated keys must NOT drag the cache down with them.
before = fp()
STATE["user_data"]["lms_progress_snapshot"] = TS
assert fp() == before, "unrelated key must not invalidate"
print("unrelated keys ignored ✓")

# Still one query for user_data, not one per family.
user_data_queries = [q for q in LAST_SQL if "FROM user_data" in q]
assert len(user_data_queries) == len([q for q in LAST_SQL if "FROM uploads" in q]), LAST_SQL[-3:]
assert "LIKE ANY" in user_data_queries[-1], user_data_queries[-1]
print("single user_data query with LIKE ANY ✓")

print("ALL FINGERPRINT TESTS PASSED")
