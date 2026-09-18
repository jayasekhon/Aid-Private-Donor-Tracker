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
from src.mediacloud_source import fetch_mediacloud_articles
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
    mediacloud_triggers = cfg["mediacloud_triggers"]
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
        mediacloud_queries_made = 0
        mediacloud_candidates_found = 0
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

        mediacloud_queries_made = 0
        mediacloud_candidates_found = 0
        if settings["search"].get("mediacloud_enabled", False):
            # A third, again completely different fetch shape: a real
            # structured search API (not an RSS hack like Google News, not
            # a global bulk-file firehose like GDELT) — see
            # mediacloud_source.py's module docstring for the collection
            # choice and why recipients are batched into few queries
            # rather than one per recipient. Raises immediately (not a
            # per-batch FetchFailure) if enabled without an API key, since
            # every batch would fail identically -- see that function's
            # docstring.
            mc_articles, mc_failures, mc_stats = fetch_mediacloud_articles(
                recipients, mediacloud_triggers,
                collection_id=settings["search"]["mediacloud_collection_id"],
                max_age_days=settings["search"]["max_article_age_days"],
                recipients_per_query=settings["search"].get("mediacloud_recipients_per_query", 8),
                triggers_per_query=settings["search"].get("mediacloud_triggers_per_query", 20),
            )
            raw_articles.extend(mc_articles)
            fetch_failures.extend(f.__dict__ for f in mc_failures)
            mediacloud_queries_made = mc_stats.queries_made
            mediacloud_candidates_found = mc_stats.candidates_found

        logger.info("Fetched %d raw articles: %d Google News queries (%d recipients, batched) + "
                     "%d GDELT GKG file(s) processed (%d candidates) + %d Media Cloud queries "
                     "(%d candidates) + %d PR wire feeds (%d fetch failures).",
                     len(raw_articles), google_news_query_count, len(recipients),
                     gdelt_files_processed, gdelt_candidates_found,
                     mediacloud_queries_made, mediacloud_candidates_found,
                     len(pr_wire_feeds), len(fetch_failures))

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
    rejected_out_of_scope_country = 0
    rejected_government_entity = 0
    rejected_unspecified_scope_excluded = 0
    rejected_donor_is_monitored_recipient = 0
    rejected_no_evidence = 0

    from src.extraction import (
        FatalExtractionError, is_generic_donor, is_generic_recipient, is_out_of_scope_country,
        assumptions_admit_missing_name, DONOR_NAME_CONTEXT_WORDS, RECIPIENT_NAME_CONTEXT_WORDS,
        UNSPECIFIED_SCOPE, is_government_entity, match_curated_recipient,
    )

    def _cluster_sources(cluster) -> str:
        return ", ".join(sorted({a.fetch_source for a in cluster.articles}))

    for i, cluster in enumerate(clusters, start=1):
        if i == 1 or i % 5 == 0 or i == len(clusters):
            logger.info("Processing cluster %d/%d...", i, len(clusters))
        try:
            results = extract_from_cluster(cluster, ai_settings=settings["ai"], mock=args.mock)
        except FatalExtractionError as e:
            logger.error("Stopping run early: %s", e)
            logger.error("No further clusters will be processed this run. Fix the issue above and "
                         "re-run manually from the Actions tab once resolved.")
            skipped_due_to_budget += (len(clusters) - i + 1)
            break
        if not results:
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
        if len(results) > 1:
            # A cluster is a keyword/title-similarity GUESS that a group of
            # articles describes one event; extract_from_cluster's prompt
            # now asks the model to pull out every distinct event it can
            # actually find rather than assuming there's exactly one, so a
            # cluster occasionally yields more than one real entry (a real
            # case: a dominant vague-recipient celebrity pledge that had a
            # second, specifically-named, unrelated donation mentioned in
            # just one of its ~20 syndicated articles). Logging each
            # event's donor/recipient here (not just the count) is what
            # made it possible to diagnose a real case where a multi-event
            # cluster ended up publishing only ONE of its events with no
            # rejection line logged for the other at all -- without this,
            # that kind of silent drop is invisible until someone notices
            # a real finding missing from the site days later.
            logger.info("Cluster %s (via %s) contained %d distinct events:",
                         cluster.cluster_id, _cluster_sources(cluster), len(results))
            for idx, r in enumerate(results, start=1):
                logger.info("  event %d/%d: %r -> %r", idx, len(results), r.donor, r.recipient)

        for result in results:
            # This tracker exists to track PRIVATE-SECTOR donations to
            # NONPROFITS -- neither side of that should be a government
            # body. A real run published two entries where both sides
            # were government: "Lagos State Government" -> "Nigerian
            # Railway Corporation" (a state-owned rail operator, not a
            # nonprofit), and "Lagos State Security Trust Fund" ->
            # "Lagos railway security agencies". The prompt already says
            # to reject a government donor, but had no equivalent
            # instruction for a government-run recipient, and evidently
            # isn't reliably complied with either way. Checked before the
            # donor/recipient-name checks below since it's a different
            # kind of gate -- not "is a name missing" but "is this even a
            # private-to-nonprofit donation at all".
            if is_government_entity(result.donor) or is_government_entity(result.recipient):
                rejected_government_entity += 1
                logger.info("Rejecting cluster %s (via %s): government entity on one side (donor %r, "
                             "recipient %r) -- not a private-sector-to-nonprofit donation.",
                             cluster.cluster_id, _cluster_sources(cluster), result.donor, result.recipient)
                continue
            # This tracker watches donations TO the curated 50 (UNICEF, WFP,
            # World Bank, etc.) -- one of them should never itself be the
            # DONOR. A real run published "World Bank" -> "Training program
            # in Niger" (the World Bank funding/sponsoring training, not
            # receiving a private donation) -- a direction mix-up, not a
            # private company giving to a recipient at all. The model's own
            # summary even said so ("this is not a private-sector donation")
            # but nothing acted on that free-text admission; this is the
            # same class of signal as is_government_entity above, checked
            # structurally instead of by pattern-matching prose.
            if match_curated_recipient(result.donor, recipients) is not None:
                rejected_donor_is_monitored_recipient += 1
                logger.info("Rejecting cluster %s (via %s): donor %r is itself one of the monitored "
                             "recipients -- likely a donor/recipient direction mix-up, not a private "
                             "donation TO it.", cluster.cluster_id, _cluster_sources(cluster), result.donor)
                continue
            # Backstop behind the extraction prompt's own instructions — the
            # model is told not to include an entry for these cases, but
            # doesn't always comply. A recipient no longer has to be on the
            # curated watchlist to be published (any named nonprofit counts —
            # see classify_recipient_type/build_donation_entry below, which tag
            # curated vs. off-list recipients rather than rejecting the latter);
            # what's still rejected is a recipient that was never actually
            # named at all, since that isn't a usable finding.
            if is_generic_donor(result.donor) or assumptions_admit_missing_name(result.assumptions, DONOR_NAME_CONTEXT_WORDS):
                rejected_no_named_donor += 1
                logger.info("Rejecting cluster %s (via %s): no specific donor named (%r).",
                             cluster.cluster_id, _cluster_sources(cluster), result.donor)
                continue
            # Recipient's own operating country/context must be one this
            # tracker is actually scoped to (GHO + Nepal, see countries.txt) —
            # the donor's own HQ/nationality is irrelevant and deliberately
            # unconstrained (query_builder.py's whole design anchors on the
            # recipient, not the donor, for exactly this reason). The AI
            # extracts country_scope as free text from the source article with
            # no constraint to the curated list at all, which is what let
            # entries about US university/hospital/community philanthropy
            # ("South Korea", "Singapore", a Cortland or Santa Barbara facility)
            # reach publication despite being outside this project's actual
            # geographic scope. Checked before the recipient-name check below
            # because it feeds that decision too.
            if is_out_of_scope_country(result.country_scope, countries):
                rejected_out_of_scope_country += 1
                logger.info("Rejecting cluster %s (via %s): recipient's country/context %r isn't on the "
                             "monitored list.", cluster.cluster_id, _cluster_sources(cluster), result.country_scope)
                continue
            # A named donor and a named recipient aren't enough on their own
            # -- there also has to be some actual evidence a gift happened
            # at all. A real run published "The Jockey Club" -> "PBC 2026
            # forum" (a PR-wire anniversary announcement for a conference,
            # no amount, not in-kind, status "unclear", and the model's own
            # assumptions admitted "the nature of the support ... is not
            # explicitly stated") and "Newport Healthcare" -> "TWLOHA" (a
            # ten-year partnership mention, same shape: no amount, not
            # in-kind) -- both real organization names, neither describing
            # an actual donation/gift the source text ever states. Requiring
            # at least one of a stated amount, a quoted figure, or an
            # in-kind description with what it was is a low bar any genuine
            # donation story clears, and catches exactly this "named
            # entities, zero substance" shape without penalizing real
            # donations that just don't state a dollar amount (those still
            # have an in-kind description, or a figure_quote like "an
            # undisclosed sum").
            if not (result.amount_text or result.figure_quote or (result.is_in_kind and result.in_kind_description)):
                rejected_no_evidence += 1
                logger.info("Rejecting cluster %s (via %s): no amount, figure, or in-kind description -- "
                             "no actual evidence a donation happened (donor %r, recipient %r).",
                             cluster.cluster_id, _cluster_sources(cluster), result.donor, result.recipient)
                continue
            recipient_is_vague = is_generic_recipient(result.recipient) or \
                assumptions_admit_missing_name(result.assumptions, RECIPIENT_NAME_CONTEXT_WORDS)
            if recipient_is_vague:
                # An unnamed recipient is still rejected outright when there's
                # no other anchor to make it a meaningful, trackable finding
                # (e.g. "children's group" with country_scope "Unspecified /
                # global" — no name AND no context). But when it's tied to a
                # specific, real, in-scope place, the donation is still worth
                # publishing even without the exact org name — a real case:
                # Macklemore pledging $1M "to Palestinian aid groups" without
                # ever naming which one, country_scope "Occupied Palestinian
                # Territory". score_confidence() scores this lower via
                # specific_recipient_named_points precisely because the name
                # is missing, rather than discarding a real, sourced, in-scope
                # donation just because press coverage didn't name a specific
                # group.
                if result.country_scope.strip().lower() == UNSPECIFIED_SCOPE:
                    rejected_unnamed_recipient += 1
                    logger.info("Rejecting cluster %s (via %s): no specific recipient organization named (%r).",
                                 cluster.cluster_id, _cluster_sources(cluster), result.recipient)
                    continue
                logger.info("Publishing cluster %s (via %s) despite an unnamed/vague recipient (%r) -- tied "
                             "to a specific in-scope place (%r), so still a meaningful finding; scored "
                             "lower for the missing name.", cluster.cluster_id, _cluster_sources(cluster),
                             result.recipient, result.country_scope)

            entry = build_donation_entry(cluster, result, settings["confidence"], recipients)

            if not settings["publishing"]["include_unspecified_scope"] and entry.country_scope == "Unspecified / global":
                # Currently a dormant path (include_unspecified_scope is
                # true by default) -- but it had no log line at all, which
                # would make a real drop here silently invisible the same
                # way a real one just was for a different reason (see the
                # per-event logging added above). Logged now on principle:
                # every path that can discard a built entry should say so.
                rejected_unspecified_scope_excluded += 1
                logger.info("Rejecting cluster %s (via %s): unspecified/global scope excluded by "
                             "publishing.include_unspecified_scope=false.",
                             cluster.cluster_id, _cluster_sources(cluster))
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
    logger.info("%s %d entries (%d duplicates skipped, %d rejected as no named recipient, %d rejected as no "
                 "named donor, %d rejected as out-of-scope country, %d rejected as a government entity, "
                 "%d rejected as unspecified scope, %d rejected as donor-is-monitored-recipient, %d rejected "
                 "for no evidence).",
                 "Would publish" if is_gdelt_test else "Published", len(entries),
                 duplicates_skipped, rejected_unnamed_recipient, rejected_no_named_donor,
                 rejected_out_of_scope_country, rejected_government_entity, rejected_unspecified_scope_excluded,
                 rejected_donor_is_monitored_recipient, rejected_no_evidence)

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
        "mediacloud_queries_made": mediacloud_queries_made,
        "mediacloud_candidates_found": mediacloud_candidates_found,
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
        "rejected_out_of_scope_country": rejected_out_of_scope_country,
        "rejected_government_entity": rejected_government_entity,
        "rejected_unspecified_scope_excluded": rejected_unspecified_scope_excluded,
        "rejected_donor_is_monitored_recipient": rejected_donor_is_monitored_recipient,
        "rejected_no_evidence": rejected_no_evidence,
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
