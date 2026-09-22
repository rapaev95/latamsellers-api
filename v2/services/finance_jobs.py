"""Background recompute jobs for the heavy finance endpoints.

Why this exists: `?fresh=1` deliberately bypasses the durable cache and
recomputes from scratch. For a large project that runs tens of seconds (see the
ARTHUR note in the Dockerfile), and the browser holds an HTTP connection open
the whole time — which every layer in between eventually gives up on: the
Next.js proxy at 90s, then Railway's edge. The user watches a spinner, gets a
502, and the work is discarded even when it actually finished.

So `fresh` becomes a job. Start it, answer immediately, let it write into
`finance_compute_cache` exactly like any other compute, and let the UI poll.
The page keeps showing the previous numbers meanwhile instead of a blank
spinner, and closing the tab no longer throws the computation away.

Jobs are asyncio tasks, not raw threads: the compute step still goes through
`asyncio.to_thread`, but the setup around it (the bank-classification prefetch)
is async, and a task inherits the caller's contextvars — which is how the
legacy `user_id` binding reaches the compute at all.

State lives in a process-local dict. That is sound ONLY because the API runs a
single gunicorn worker (`-w 1`, see Dockerfile): with more workers a client
could poll a different process than the one running its job and be told no such
job exists. If the worker count ever changes, this registry has to move into
Postgres. Note the *results* are unaffected either way — they go to the shared
durable cache — so the worst case is a lost progress indicator, not lost work.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

# How many recomputes may run at once. Deliberately small: these are CPU-bound
# pandas computes on a single-worker process, so more concurrency doesn't
# finish them sooner (GIL) — it just starves ordinary requests.
_MAX_CONCURRENT = max(1, int(os.environ.get("FINANCE_JOBS_MAX_CONCURRENT", "2")))
# Keep finished jobs around long enough for a polling UI to observe the
# terminal state, then forget them.
_RETAIN_SECONDS = int(os.environ.get("FINANCE_JOBS_RETAIN_S", "600"))

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass
class JobState:
    key: str
    status: str = QUEUED
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in (DONE, FAILED)

    def as_dict(self) -> dict[str, Any]:
        now = time.monotonic()
        ref = self.finished_at or now
        base = self.started_at or self.created_at
        return {
            "key": self.key,
            "status": self.status,
            "elapsed_s": round(max(0.0, ref - base), 1),
            "error": self.error,
        }


_JOBS: dict[str, JobState] = {}
_SEM: asyncio.Semaphore | None = None
_SEM_LOOP: asyncio.AbstractEventLoop | None = None


def _semaphore() -> asyncio.Semaphore:
    """Semaphore bound to the running loop. Recreated if the loop changed —
    matters for tests and for a worker restart, where a semaphore left over
    from a dead loop would deadlock every job."""
    global _SEM, _SEM_LOOP
    loop = asyncio.get_running_loop()
    if _SEM is None or _SEM_LOOP is not loop:
        _SEM = asyncio.Semaphore(_MAX_CONCURRENT)
        _SEM_LOOP = loop
    return _SEM


def _prune() -> None:
    now = time.monotonic()
    for key, state in list(_JOBS.items()):
        if state.terminal and state.finished_at and now - state.finished_at > _RETAIN_SECONDS:
            _JOBS.pop(key, None)


def get(key: str) -> JobState | None:
    _prune()
    return _JOBS.get(key)


def snapshot(keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    _prune()
    return {k: _JOBS[k].as_dict() for k in keys if k in _JOBS}


# asyncio only keeps weak references to running tasks; without this set a job
# can be garbage-collected mid-flight.
_TASK_REFS: set[asyncio.Task] = set()


def start(key: str, work: Callable[[], Awaitable[Any]]) -> JobState:
    """Start `work` under `key`, or return the in-flight job for that key.

    Deduplication is the point: two people hitting «Обновить» on the same
    project, or one person clicking twice, must produce one computation. `key`
    is the durable cache key, so it already carries the user namespace, the
    project and the period — the exact granularity at which a recompute is
    redundant.
    """
    _prune()
    existing = _JOBS.get(key)
    if existing is not None and not existing.terminal:
        return existing

    state = JobState(key=key)
    _JOBS[key] = state

    async def _runner() -> None:
        try:
            async with _semaphore():
                state.status = RUNNING
                state.started_at = time.monotonic()
                await work()
            state.status = DONE
        except asyncio.CancelledError:
            state.status = FAILED
            state.error = "cancelled"
            raise
        except Exception as err:  # noqa: BLE001
            state.status = FAILED
            state.error = str(err)
            log.warning("finance_jobs: %s failed: %s", key, err)
        finally:
            state.finished_at = time.monotonic()

    # create_task copies the current context, so the legacy user binding and
    # any prefetched classifications set by the caller travel into the job.
    task = asyncio.create_task(_runner(), name=f"finance-recompute:{key}")
    # Hold a reference so the task isn't garbage-collected mid-flight, and
    # surface the fact that we intentionally don't await it.
    _TASK_REFS.add(task)
    task.add_done_callback(_TASK_REFS.discard)
    return state

