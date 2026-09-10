#!/usr/bin/env python3
"""Ad-hoc test: does GDELT GKG actually catch a donation story we already
know happened? Fetches GKG files for an arbitrary past date range and runs
them through the same fetch -> trigger-filter -> cluster pipeline as the
real daily run, then just prints what came out — no AI calls unless you
pass --extract, and no interaction with data/gdelt_gkg_state.json or the
real site/data files at all, so this can never disturb the production
pipeline or its incremental tracking.

Usage:
    # See raw + filtered candidates only, no AI cost:
    python scripts/gdelt_backfill_test.py --start 2026-08-20 --end 2026-08-27

    # Also run the survivors through AI extraction (uses ai.provider from
    # settings.yaml, same as a real run — this DOES cost real AI calls):
    python scripts/gdelt_backfill_test.py --start 2026-08-20 --end 2026-08-27 --extract

Dates are UTC, "YYYY-MM-DD" (midnight) or "YYYY-MM-DDTHH:MM".
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_all
from src.gdelt_gkg import fetch_gdelt_gkg_articles_for_range
from src.clustering import filter_by_trigger_phrase, tag_country, cluster_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gdelt_backfill_test")


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="UTC start date/time, e.g. 2026-08-20 or 2026-08-20T06:00")
    parser.add_argument("--end", required=True, help="UTC end date/time (inclusive)")
    parser.add_argument("--extract", action="store_true",
                         help="Also run survivors through real AI extraction (costs real AI calls, "
                              "uses ai.provider from settings.yaml). Without this flag, just shows "
                              "candidates for free so you can eyeball them first.")
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    cfg = load_all()
    recipients = cfg["recipients"]
    countries = cfg["countries"]
    triggers = cfg["triggers"]
    settings = cfg["settings"]

    logger.info("=== GDELT backfill test: %s to %s UTC ===", start, end)
    articles, failures, stats = fetch_gdelt_gkg_articles_for_range(recipients, start, end)
    if failures:
        logger.warning("%d fetch failure(s) during backfill — coverage for this range may be incomplete:", len(failures))
        for f in failures:
            logger.warning("  %s: %s", f.source_label, f.error)

    if not articles:
        logger.info("No candidates found at all (no monitored recipient mentioned in this date range's "
                     "GDELT files) — either nothing relevant happened, or something's off with the "
                     "recipient-matching step itself.")
        return

    filtered = filter_by_trigger_phrase(articles, triggers)
    tag_country(filtered, countries)
    clusters = cluster_articles(filtered)

    logger.info("=== Summary: %d raw candidate(s) -> %d passed trigger-phrase filter -> %d cluster(s) ===",
                len(articles), len(filtered), len(clusters))

    if not args.extract:
        logger.info("Not running AI extraction (pass --extract to also do that). "
                     "Eyeball the clusters above — if a known real donation story isn't showing up "
                     "even as a raw candidate, the gap is upstream of the AI (recipient-matching or "
                     "trigger-phrase filter); if it shows up here but you'd expect it to have been "
                     "published, the gap is in the AI step or the recipient/donor backstops.")
        return

    logger.info("--extract passed: running %d cluster(s) through real AI extraction "
                "(ai.provider=%s) — this costs real AI calls.", len(clusters), settings["ai"]["provider"])
    from src.extraction import (
        extract_from_cluster, FatalExtractionError, is_generic_donor,
        is_generic_recipient, classify_recipient_type,
    )

    for i, cluster in enumerate(clusters, start=1):
        logger.info("Processing cluster %d/%d...", i, len(clusters))
        try:
            result = extract_from_cluster(cluster, ai_settings=settings["ai"])
        except FatalExtractionError as e:
            logger.error("Stopping early: %s", e)
            break
        if result is None:
            logger.info("  -> not judged relevant by the AI.")
            continue
        if is_generic_donor(result.donor):
            logger.info("  -> rejected: no specific donor named (%r).", result.donor)
            continue
        if is_generic_recipient(result.recipient):
            logger.info("  -> rejected: no specific recipient organization named (%r).", result.recipient)
            continue
        recipient_type = classify_recipient_type(result.recipient, recipients, result.recipient_type_guess)
        logger.info("  -> MATCH: %s donated %s to %s [%s] (%s)",
                    result.donor, result.amount_text or "(in-kind/unspecified)", result.recipient,
                    recipient_type, result.status)


if __name__ == "__main__":
    main()
