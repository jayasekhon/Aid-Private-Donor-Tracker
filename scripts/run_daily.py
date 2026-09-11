#!/usr/bin/env python3
"""Runs the full daily pipeline:

  1. Load config
  2. Build search queries from recipients + trigger phrases
  3. Fetch Google News RSS + PR wire feeds + GDELT GKG bulk files
  4. Filter by trigger phrase (cheap, cuts volume before any AI calls)
  5. Tag country mentions
  6. Cluster near-duplicate articles into events (further cuts AI calls)
  7. Extract structured facts per cluster via the configured AI provider
     (Bedrock or Gemini — see config/settings.yaml's ai.provider) (or --mock)
  8. Check each extracted entry against the rolling event store
     (skip exact duplicates, tag renewals, publish new ones)
  9. Save today's edition JSON + rebuild the static site

Usage:
    python scripts/run_daily.py                 # live run (needs AWS creds for Bedrock, or GEMINI_API_KEY if ai.provider is "gemini")
    python scripts/run_daily.py --mock           # no API calls, fake data throughout
    python scripts/run_daily.py --max-clusters 5 # cap AI calls for a cheap smoke test

    # TESTING ONLY: override GDELT's normal incremental fetch with a specific
    # UTC date/time range (Google News and PR wires are untouched -- they
    # always run their normal "today" behavior). Does NOT touch
    # data/gdelt_gkg_state.json, so the next normal scheduled run resumes
    # from wherever it actually left off, completely unaffected by this.
    # Unlike scripts/gdelt_backfill_test.py, this runs the REAL pipeline --
    # real AI extraction, real backstops against real history -- so it
    # checks whether a known real-world event would actually have been
    # caught, not just whether it shows up as a raw candidate. But it's
    # still fully isolated from the real site and event store: the edition
    # is saved as data/editions/TEST-<date>.json (not overwriting today's
    # real edition), nothing is written to data/seen_events.json, and the
    # live site (docs/) is not rebuilt at all.
    python scripts/run_daily.py --gdelt-start 2026-09-09 --gdelt-end 2026-09-10
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_all
from src.query_builder import build_recipient_trigger_queries, google_news_rss_url
from src.sources import fetch_all
from src.gdelt_gkg import fetch_gdelt_gkg_articles, fetch_gdelt_gkg_articles_for_range
from src.clustering import filter_by_recency, filter_by_trigger_phrase, tag_country, tag_official_sources, cluster_articles, rank_clusters_by_priority
from src.extraction import extract_from_cluster, build_donation_entry
from src.store import EventStore
from src.models import today_str, DonationEntry
from src.site_builder import save_edition_json, build_site, render_test_edition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_daily")


def _parse_gdelt_override_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run with no external API calls (Gemini mocked, still fetches real RSS unless --mock-fetch too).")
    parser.add_argument("--mock-fetch", action="store_true", help="Also skip real RSS fetching and use built-in sample articles.")
    parser.add_argument("--max-clusters", type=int, default=None, help="Cap AI extraction calls for this run (overrides settings.yaml for a cheap test).")
    parser.add_argument("--gdelt-start", type=str, default=None,
                         help="TESTING ONLY: UTC start date/time (YYYY-MM-DD or YYYY-MM-DDTHH:MM) to fetch GDELT "
                              "from, instead of its normal incremental window. Must be paired with --gdelt-end. "
                              "Does not affect Google News/PR wires, and does not touch "
                              "data/gdelt_gkg_state.json (the next normal run is unaffected).")
    parser.add_argument("--gdelt-end", type=str, default=None,
                         help="TESTING ONLY: UTC end date/time (inclusive), paired with --gdelt-start.")
    args = parser.parse_args()

    if bool(args.gdelt_start) != bool(args.gdelt_end):
        parser.error("--gdelt-start and --gdelt-end must be given together.")

    is_gdelt_test = bool(args.gdelt_start)
    gdelt_start = gdelt_end = None
    if is_gdelt_test:
        gdelt_start = _parse_gdelt_override_date(args.gdelt_start)
        gdelt_end = _parse_gdelt_override_date(args.gdelt_end)

    cfg = load_all()
    settings = cfg["settings"]
    recipients = cfg["recipients"]
    countries = cfg["countries"]
    triggers = cfg["triggers"]
    pr_wire_feeds = cfg["pr_wire_feeds"] if cfg["settings"]["search"].get("pr_wire_feeds_enabled", True) else []

    # A GDELT-override run gets its own TEST-<date> edition label rather than
    # today's real date -- today_str() would silently overwrite (not merge
    # with) whatever the real scheduled run already saved for today. See
    # the "Save + build site" section below for the rest of this run's
    # isolation from real site/event-store state.
    date_str = f"TEST-{gdelt_start:%Y-%m-%d}" if is_gdelt_test else today_str()
    logger.info("=== Daily run for %s ===", date_str)
    if is_gdelt_test:
        logger.warning("GDELT DATE OVERRIDE ACTIVE: fetching %s to %s instead of the normal incremental "
                        "window. This is a TESTING run -- data/gdelt_gkg_state.json, data/seen_events.json, "
                        "and the live site (docs/) are all left untouched; results are saved only to "
                        "data/editions/%s.json.", gdelt_start, gdelt_end, date_str)

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
        google_news_query_count = 0
        gdelt_files_processed = 0
        gdelt_candidates_found = 0
    else:
        recipient_queries = []
        if settings["search"].get("google_news_enabled", True):
            trigger_query_pairs = build_recipient_trigger_queries(recipients, triggers, max_age_days=settings["search"]["max_article_age_days"])
            recipient_queries = [(name, google_news_rss_url(q)) for name, q in trigger_query_pairs]

        raw_articles, failures = fetch_all(recipient_queries, pr_wire_feeds)
        fetch_failures = [f.__dict__ for f in failures]
        google_news_query_count = len(recipient_queries)

        gdelt_files_processed = 0
        gdelt_candidates_found = 0
        if settings["search"].get("gdelt_enabled", True):
            # A completely different fetch shape from Google News/PR wires:
            # bulk 15-minute GDELT GKG file downloads, not a per-query search
            # API (see gdelt_gkg.py's module docstring for why). It narrows
            # GDELT's global firehose down to "mentions a monitored
            # recipient" itself; the existing trigger-phrase filter below
            # then applies to these candidates exactly like any other
            # source's, with no GDELT-specific filtering code needed there.
            if is_gdelt_test:
                gdelt_articles, gdelt_failures, gdelt_stats = fetch_gdelt_gkg_articles_for_range(
                    recipients, gdelt_start, gdelt_end)
            else:
                gdelt_articles, gdelt_failures, gdelt_stats = fetch_gdelt_gkg_articles(recipients)
            raw_articles.extend(gdelt_articles)
            fetch_failures.extend(f.__dict__ for f in gdelt_failures)
            gdelt_files_processed = gdelt_stats.files_processed
            gdelt_candidates_found = gdelt_stats.candidates_found

        logger.info("Fetched %d raw articles: %d Google News queries (%d recipients, batched) + "
                     "%d GDELT GKG file(s) processed (%d candidates) + %d PR wire feeds (%d fetch failures).",
                     len(raw_articles), google_news_query_count, len(recipients),
                     gdelt_files_processed, gdelt_candidates_found, len(pr_wire_feeds), len(fetch_failures))

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
    tag_official_sources(filtered, recipients)

    # --- Cluster ---
    clusters = cluster_articles(filtered)
    logger.info("Grouped into %d event cluster(s).", len(clusters))

    max_calls = args.max_clusters or settings["ai"]["max_ai_calls_per_run"]
    if len(clusters) > max_calls:
        # max_ai_calls_per_run is a self-imposed cap — sized to stay under
        # Gemini's free-tier daily quota if ai.provider is "gemini", or to
        # bound worst-case Bedrock spend otherwise (see settings.yaml).
        # When the budget is tight, spend it on the most promising
        # candidates first rather than an arbitrary subset.
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
    rejected_unnamed_recipient = 0
    rejected_no_named_donor = 0

    from src.extraction import FatalExtractionError, is_generic_donor, is_generic_recipient

    def _cluster_sources(cluster) -> str:
        return ", ".join(sorted({a.fetch_source for a in cluster.articles}))

    for i, cluster in enumerate(clusters, start=1):
        if i == 1 or i % 5 == 0 or i == len(clusters):
            logger.info("Processing cluster %d/%d...", i, len(clusters))
        try:
            result = extract_from_cluster(cluster, ai_settings=settings["ai"], mock=args.mock)
        except FatalExtractionError as e:
            logger.error("Stopping run early: %s", e)
            logger.error("No further clusters will be processed this run. Fix the issue above and "
                         "re-run manually from the Actions tab once resolved.")
            skipped_due_to_budget += (len(clusters) - i + 1)
            break
        if result is None:
            # Previously silent -- no way to tell from the log whether the
            # AI judged this not relevant or extraction failed after
            # retries, let alone which fetch source(s) it came from. That
            # blind spot mattered in practice: GDELT/PR-wire candidates
            # were passing the trigger-phrase filter but zero of them ever
            # showed up in a published entry, and this was the one place
            # in the pipeline with no visibility into why.
            logger.info("Cluster %s (via %s) not judged relevant by the AI (or extraction failed).",
                         cluster.cluster_id, _cluster_sources(cluster))
            continue

        # Backstop behind the extraction prompt's own instructions — the
        # model is told to set is_relevant=false for these cases, but
        # doesn't always comply. A recipient no longer has to be on the
        # curated watchlist to be published (any named nonprofit counts —
        # see classify_recipient_type/build_donation_entry below, which tag
        # curated vs. off-list recipients rather than rejecting the latter);
        # what's still rejected is a recipient that was never actually
        # named at all, since that isn't a usable finding.
        if is_generic_donor(result.donor):
            rejected_no_named_donor += 1
            logger.info("Rejecting cluster %s (via %s): no specific donor named (%r).",
                         cluster.cluster_id, _cluster_sources(cluster), result.donor)
            continue
        if is_generic_recipient(result.recipient):
            rejected_unnamed_recipient += 1
            logger.info("Rejecting cluster %s (via %s): no specific recipient organization named (%r).",
                         cluster.cluster_id, _cluster_sources(cluster), result.recipient)
            continue

        entry = build_donation_entry(cluster, result, settings["confidence"], recipients)

        if not settings["publishing"]["include_unspecified_scope"] and entry.country_scope == "Unspecified / global":
            continue

        prior, reason = store.find_possible_match(entry, lookback_days=lookback)
        if reason == "exact_duplicate":
            duplicates_skipped += 1
            logger.info("Skipping exact duplicate (via %s): %s -> %s",
                         ", ".join(entry.source_channels), entry.donor, entry.recipient)
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

    # A GDELT-override test run still checks candidates against real
    # history via store.find_possible_match() above (so it correctly
    # recognizes an already-published donation as a duplicate), but must
    # NOT persist anything it finds back into the real store -- otherwise
    # a later real run would wrongly think this test-only entry was
    # already published for real, and silently skip actually publishing
    # it. store.add() above only mutates the in-memory object; store.save()
    # is what writes to disk, so skipping it here is sufficient.
    if not is_gdelt_test:
        store.save()
    logger.info("%s %d entries (%d duplicates skipped, %d rejected as no named recipient, %d rejected as no named donor).",
                 "Would publish" if is_gdelt_test else "Published", len(entries),
                 duplicates_skipped, rejected_unnamed_recipient, rejected_no_named_donor)

    # Highest confidence first -- the site should lead with its strongest,
    # best-evidenced findings rather than whatever order clusters happened
    # to be processed in (which is really just fetch/AI-call order, not a
    # meaningful ranking for a reader).
    entries.sort(key=lambda e: e.confidence_score, reverse=True)

    # --- Save + build site ---
    high_confidence_threshold = 8
    stats = {
        "feeds_checked": google_news_query_count + len(pr_wire_feeds),
        "google_news_queries": google_news_query_count,
        "gdelt_files_processed": gdelt_files_processed,
        "gdelt_candidates_found": gdelt_candidates_found,
        "recipients_watched": len(recipients),
        "pr_wire_feeds": len(pr_wire_feeds),
        "items_considered": items_considered,
        "items_filtered_as_stale": items_filtered_as_stale,
        "max_article_age_days": max_article_age_days,
        "items_after_trigger_filter": len(filtered),
        "clusters_analysed": len(clusters),
        "skipped_due_to_ai_budget": skipped_due_to_budget,
        "duplicates_skipped": duplicates_skipped,
        "rejected_unnamed_recipient": rejected_unnamed_recipient,
        "rejected_no_named_donor": rejected_no_named_donor,
        "entries_published": len(entries),
        "high_confidence_count": sum(1 for e in entries if e.confidence_score >= high_confidence_threshold),
    }

    save_edition_json(date_str, entries, stats, fetch_failures)

    if is_gdelt_test:
        # Deliberately does not call build_site(): the live site's homepage
        # is always "the latest edition" and its date-sorted archive/nav
        # expects real YYYY-MM-DD edition dates, neither of which a
        # TEST-<date> edition should ever become. Instead render a
        # standalone local HTML preview (same template/styling as a real
        # edition page) into test_output/ -- gitignored, entirely outside
        # docs/, so test/mock data can never end up reachable on the live
        # public site even unlinked.
        preview_path = render_test_edition(date_str, entries, stats, settings["site"])
        logger.info("GDELT date override run complete -- results saved to data/editions/%s.json, "
                     "preview page at %s. The live site was NOT rebuilt (this is a test run, "
                     "not a real edition).", date_str, preview_path)
    else:
        recipient_counts = {"UN": 0, "INGO": 0, "NGO": 0}
        for r in recipients:
            recipient_counts[r.org_type] += 1

        build_site(settings, recipient_counts, settings["confidence"]["low_confidence_threshold"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
