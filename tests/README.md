# tests

Plain scripts, no pytest — run them with the project's interpreter:

```bash
.venv/bin/python tests/test_recompute.py
.venv/bin/python tests/test_company.py
.venv/bin/python tests/test_warm.py
.venv/bin/python tests/test_fingerprint.py
.venv/bin/python tests/test_uploads_listing.py
```

They monkeypatch the compute layer, so nothing here touches Postgres or runs a
real report. What they pin down is the wiring that is easy to break silently:

- `test_recompute.py` — cache keys built by the recompute job match the ones the
  read endpoints use, two clicks produce one computation, statuses reach `done`,
  unknown project 404s and a services project needs superadmin.
- `test_company.py` — the roll-up reports `forbidden` / `pending` projects
  instead of dropping them, and a project without numbers is never summed as 0.
- `test_warm.py` — a warm pass never forces a recompute (that is what makes it
  cheap), keeps going after a per-project failure, and respects its budget.
- `test_fingerprint.py` — every dynamic `user_data` key family the UI edits
  invalidates the cache (add, edit AND delete), unrelated keys don't, and it
  still takes one query. This is what replaced the UI's forced `fresh=true`.
- `test_uploads_listing.py` — the uploads list view reads metadata only. The
  regression it guards is invisible in the response: fetching the bytes again
  returns identical JSON and is simply slow, so the query shape is pinned.

Run them after touching `finance_cache`, `finance_jobs`, the cache-key helpers,
or anything in `routers/finance.py` that those share.
