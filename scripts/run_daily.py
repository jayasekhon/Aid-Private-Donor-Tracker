#!/usr/bin/env python3
"""Runs the full daily pipeline:

  1. Load config
  2. Build search queries from recipients + trigger phrases
  3. Fetch Google News RSS + PR wire feeds
  4. Filter by trigger phrase (cheap, cuts volume before any AI calls)
  5. Tag country mentions
  6. Cluster near-duplicate articles into events (further cuts AI calls)
  7. Extract structured facts per cluster via Gemini (or --mock)
  8. Check each extracted entry against the rolling event store
     (skip exact duplicates, tag renewals, publish new ones)
  9. Save today's edition JSON + rebuild the static site

Usage:
    python scripts/run_daily.py                 # live run (needs GEMINI_API_KEY)
    python scripts/run_daily.py --mock           # no API calls, fake data throughout
    python scripts/run_daily.py --max-clusters 5 # cap AI calls for a cheap smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_all
from src.query_builder import build_recipient_trigger_queries, google_news_rss_url
from src.sources import fetch_all
from src.clustering import filter_by_recency, filter_by_trigger_phrase, tag_country, cluster_articles, rank_clusters_by_priority
from src.extraction import extract_from_cluster, build_donation_entry
from src.store import EventStore
from src.models import today_str, DonationEntry
from src.site_builder import save_edition_json, build_site

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_daily")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run with no external API calls (Gemini mocked, still fetches real RSS unless --mock-fetch too).")
    parser.add_argument("--mock-fetch", action="store_true", help="Also skip real RSS fetching and use built-in sample articles.")
    parser.add_argument("--max-clusters", type=int, default=None, help="Cap AI extraction calls for this run (overrides settings.yaml for a cheap test).")
    args = parser.parse_args()

    cfg = load_all()
    settings = cfg["settings"]
    recipients = cfg["recipients"]
    countries = cfg["countries"]
    triggers = cfg["triggers"]
    pr_wire_feeds = cfg["pr_wire_feeds"]

    date_str = today_str()
    logger.info("=== Daily run for %s ===", date_str)

    # --- Fetch ---
    if args.mock_fetch:
        from src.models import RawArticle, SourceTier
        logger.info("Using built-in sample articles (--mock-fetch).")
        raw_articles = [
            RawArticle("Acme Logistics donates $3 million to WFP for regional food response", "https://example.com/1", None, "Reuters", SourceTier.GENERAL_NEWS, "Acme Logistics announced a $3 million donation.", matched_recipient="World Food Programme"),
            RawArticle("Acme Logistics pledges $3M to World Food Programme", "https://example.com/2", None, "AP", SourceTier.GENERAL_NEWS, "", matched_recipient="World Food Programme"),
            RawArticle("Nordfield Pharma renews multi-year vaccine partnership with UNICEF", "https://example.com/3", None, "PR Newswire", SourceTier.WIRE, "Nordfield Pharma has renewed its partnership.", matched_recipient="UNICEF"),
            RawArticle("UNICEF marks World Children's Day", "https://example.com/4", None, "BBC", SourceTier.GENERAL_NEWS, "A routine editorial piece, not about a donation.", matched_recipient="UNICEF"),
        ]
        fetch_failures = []
    else:
        trigger_queries = build_recipient_trigger_queries(recipients, triggers, max_age_days=settings["search"]["max_article_age_days"])
        recipient_queries = {
            r.name: google_news_rss_url(q) for r, q in zip(recipients, trigger_queries)
        }
        raw_articles, failures = fetch_all(recipient_queries, pr_wire_feeds)
        fetch_failures = [f.__dict__ for f in failures]
        logger.info("Fetched %d raw articles across %d Google News queries + %d PR wire feeds (%d fetch failures).",
                     len(raw_articles), len(recipient_queries), len(pr_wire_feeds), len(fetch_failures))

    items_considered = len(raw_articles)

    # --- Recency filter (backstop behind the Google News "when:" restriction
    # above — catches PR wire items and anything that slips past it) ---
    max_article_age_days = settings["search"]["max_article_age_days"]
    raw_articles, items_filtered_as_stale = filter_by_recency(raw_articles, max_article_age_days)
    if items_filtered_as_stale:
        logger.info("Dropped %d article(s) older than %d days.", items_filtered_as_stale, max_article_age_days)

    # --- Filter ---
    filtered = filter_by_trigger_phrase(raw_articles, triggers)
    logger.info("%d/%d articles passed the trigger-phrase filter.", len(filtered), items_considered)

    tag_country(filtered, countries)

    # --- Cluster ---
    clusters = cluster_articles(filtered)
    logger.info("Grouped into %d event cluster(s).", len(clusters))

    max_calls = args.max_clusters or settings["ai"]["max_ai_calls_per_run"]
    if len(clusters) > max_calls:
        # Gemini's free tier can be as low as ~20 requests/day (verified
        # live — this changes without notice, see settings.yaml). When the
        # budget is tight, spend it on the most promising candidates first
        # rather than an arbitrary subset.
        clusters = rank_clusters_by_priority(clusters)
        logger.warning("Capping at %d clusters (of %d) to respect the AI call budget — "
                        "processing the highest-priority candidates first.", max_calls, len(clusters))
        skipped_due_to_budget = len(clusters) - max_calls
        clusters = clusters[:max_calls]
    else:
        skipped_due_to_budget = 0

    # --- Extract + dedupe ---
    store = EventStore()
    lookback = settings["ai"]["dedupe_lookback_days"]
    entries: list[DonationEntry] = []
    duplicates_skipped = 0

    from src.extraction import FatalExtractionError

    for i, cluster in enumerate(clusters, start=1):
        if i == 1 or i % 5 == 0 or i == len(clusters):
            logger.info("Processing cluster %d/%d...", i, len(clusters))
        try:
            result = extract_from_cluster(cluster, model=settings["ai"]["model"], mock=args.mock)
        except FatalExtractionError as e:
            logger.error("Stopping run early: %s", e)
            logger.error("No further clusters will be processed this run. Fix the issue above and "
                         "re-run manually from the Actions tab once resolved.")
            skipped_due_to_budget += (len(clusters) - i + 1)
            break
        if result is None:
            continue  # not relevant, or extraction failed after retries

        entry = build_donation_entry(cluster, result, settings["confidence"])

        if not settings["publishing"]["include_unspecified_scope"] and entry.country_scope == "Unspecified / global":
            continue

        prior, reason = store.find_possible_match(entry, lookback_days=lookback)
        if reason == "exact_duplicate":
            duplicates_skipped += 1
            logger.info("Skipping exact duplicate: %s -> %s", entry.donor, entry.recipient)
            continue
        elif reason == "likely_renewal" and prior is not None:
            entry.is_duplicate_of = prior.get("entry_id")
            # Leave entry.status as extracted by the model, UNLESS the model
            # said "unclear" — in that case, defer to the store's own signal.
            from src.models import EventStatus
            if entry.status == EventStatus.UNCLEAR:
                entry.status = EventStatus.RENEWAL

        store.add(entry)
        entries.append(entry)

    store.save()
    logger.info("Published %d entries today (%d duplicates skipped).", len(entries), duplicates_skipped)

    # --- Save + build site ---
    high_confidence_threshold = 8
    stats = {
        "feeds_checked": len(recipients) + len(pr_wire_feeds),
        "google_news_queries": len(recipients),
        "pr_wire_feeds": len(pr_wire_feeds),
        "items_considered": items_considered,
        "items_filtered_as_stale": items_filtered_as_stale,
        "max_article_age_days": max_article_age_days,
        "items_after_trigger_filter": len(filtered),
        "clusters_analysed": len(clusters),
        "skipped_due_to_ai_budget": skipped_due_to_budget,
        "duplicates_skipped": duplicates_skipped,
        "entries_published": len(entries),
        "high_confidence_count": sum(1 for e in entries if e.confidence_score >= high_confidence_threshold),
    }

    save_edition_json(date_str, entries, stats, fetch_failures)

    recipient_counts = {"UN": 0, "INGO": 0, "NGO": 0}
    for r in recipients:
        recipient_counts[r.org_type] += 1

    build_site(settings, recipient_counts, settings["confidence"]["low_confidence_threshold"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
