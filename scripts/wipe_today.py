#!/usr/bin/env python3
"""Wipes TODAY's data only, so the next run of run_daily.py starts fresh
for today without touching any other day's history.

"Today's data" is two things, and both matter -- deleting the edition
file alone is not enough:
  1. data/editions/{today}.json -- today's published edition. Obvious to
     delete, and what past manual re-runs this session usually did.
  2. data/seen_events.json -- the persistent cross-run dedupe store.
     Today's entries are recorded in here too (added the moment they were
     first published, regardless of which run did it), so even after
     deleting (1), a fresh run recognizes them as "already known" and
     silently skips re-publishing them as duplicates, producing an
     incomplete second edition. Wiping this file's today-dated entries
     alongside the edition file is what actually makes the next run
     start clean.

Only entries whose date_found falls on today (UTC, matching today_str()
in models.py) are removed from the store -- every other day's history is
left untouched. docs/editions/{today}.html and docs/index.html are left
alone deliberately: build_site() unconditionally regenerates every
edition's HTML page from data/editions/ on the next real run, so there is
nothing to pre-clean there.

Usage:
    python scripts/wipe_today.py            # wipes today's data
    python scripts/wipe_today.py --dry-run  # shows what would be removed, changes nothing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import today_str
from src.store import DEFAULT_STORE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wipe_today")

ROOT = Path(__file__).resolve().parent.parent
EDITIONS_DATA_DIR = ROOT / "data" / "editions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would be removed without changing anything.")
    args = parser.parse_args()

    date_str = today_str()
    logger.info("Wiping data for %s (UTC)%s...", date_str, " [DRY RUN]" if args.dry_run else "")

    edition_path = EDITIONS_DATA_DIR / f"{date_str}.json"
    if edition_path.exists():
        logger.info("Removing %s", edition_path)
        if not args.dry_run:
            edition_path.unlink()
    else:
        logger.info("No edition file for %s to remove.", date_str)

    if DEFAULT_STORE_PATH.exists():
        with open(DEFAULT_STORE_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)
        kept = [e for e in entries if not (e.get("date_found") or "").startswith(date_str)]
        removed_count = len(entries) - len(kept)
        logger.info("Removing %d of %d event(s) from %s (date_found starting with %s).",
                     removed_count, len(entries), DEFAULT_STORE_PATH, date_str)
        if not args.dry_run and removed_count:
            with open(DEFAULT_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(kept, f, indent=2, ensure_ascii=False)
    else:
        logger.info("No event store at %s to prune.", DEFAULT_STORE_PATH)

    logger.info("Done.%s", " Nothing was actually changed (--dry-run)." if args.dry_run else "")


if __name__ == "__main__":
    main()
