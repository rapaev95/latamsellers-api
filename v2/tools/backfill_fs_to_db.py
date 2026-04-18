"""One-off backfill: import shared FS files into the `uploads` table for a user.

Existing `vendas/` and `_data/armazenagem/` directories were populated before
per-user isolation landed. This script picks every file that the FS loaders
would recognise and writes it into `uploads` under a target user_id, so the
DB-mode aggregator (`LS_STORAGE_MODE=db`) returns the same data the old FS
mode did for that user.

Usage (from repo root):
    DATABASE_URL=... python -m v2.tools.backfill_fs_to_db --user-id 1

Safe to re-run: the partial unique index `ux_uploads_user_hash(user_id, content_sha256)`
dedupes by content, so repeated runs only refresh `created_at`.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Resolve imports when run as `python -m v2.tools.backfill_fs_to_db`.
_ROOT = Path(__file__).resolve().parents[2]  # _admin/api
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.db import create_pool, close_pool  # noqa: E402
from v2.parsers.armazenagem import ARMAZENAGEM_DIRS, _is_armazenamento_file  # noqa: E402
from v2.parsers.stock_full import list_stock_full_files  # noqa: E402
from v2.parsers.vendas_ml import VENDAS_DIR, is_vendas_ml_file  # noqa: E402
from v2.storage import uploads_storage  # noqa: E402


async def _ingest_dir(pool, user_id: int, directory: Path, source_key: str, predicate) -> int:
    if not directory.exists():
        return 0
    n = 0
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if not predicate(entry.name):
            continue
        data = entry.read_bytes()
        await uploads_storage.put_file(
            pool,
            user_id=user_id,
            source_key=source_key,
            filename=entry.name,
            file_bytes=data,
        )
        n += 1
        print(f"  + {source_key}: {entry.name} ({len(data)} bytes)")
    return n


async def main(user_id: int) -> int:
    pool = await create_pool()
    if pool is None:
        print("DATABASE_URL not set — aborting.", file=sys.stderr)
        return 1
    try:
        print(f"Backfilling FS → uploads for user_id={user_id}")
        n_vendas = await _ingest_dir(pool, user_id, VENDAS_DIR, "vendas_ml", is_vendas_ml_file)
        print(f"Imported {n_vendas} vendas files from {VENDAS_DIR}")

        n_arm = 0
        for d in ARMAZENAGEM_DIRS:
            n_arm += await _ingest_dir(pool, user_id, d, "armazenagem_full", _is_armazenamento_file)
        print(f"Imported {n_arm} armazenagem files across {len(ARMAZENAGEM_DIRS)} dir(s)")

        # Stock Full — discovered via the dedicated helper (scans _data/<month>/
        # and vendas/). Import each file once, newest-first ordering handled by
        # the loader side.
        n_stock = 0
        for p in list_stock_full_files():
            data = p.read_bytes()
            await uploads_storage.put_file(
                pool,
                user_id=user_id,
                source_key="stock_full",
                filename=p.name,
                file_bytes=data,
            )
            n_stock += 1
            print(f"  + stock_full: {p.name} ({len(data)} bytes)")
        print(f"Imported {n_stock} stock_full files")
        return 0
    finally:
        await close_pool()


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user-id", type=int, required=True, help="Target user_id in users table")
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.user_id)))


if __name__ == "__main__":
    cli()
