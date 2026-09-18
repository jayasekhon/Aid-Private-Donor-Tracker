"""Fetches candidate donation-story leads from Media Cloud
(mediacloud.org) — a real structured news search API/index, distinct
from Google News (an RSS-search-string hack, not a real API — see
query_builder.py's docstring on its reliability quirks) and GDELT (a
much broader, noisier, multilingual firehose scanned via bulk file
downloads rather than live search — see gdelt_gkg.py). Meant to catch
English-language stories neither of those two surface, particularly
smaller/regional business press that Google News' relevance ranking
tends to bury.

Collection choice: config/settings.yaml's search.mediacloud_collection_id
defaults to #9272347, "Global English Language Sources" (~1,600 curated
English-language sources) rather than #8876987, "Global Voices Cited
Sources" (11,000+ sources, heavily multilingual, weighted toward citizen/
alternative media) — this pipeline is English-only throughout (trigger
phrases, extraction prompt, alias matching), so the smaller curated
English collection is both more relevant and far lower-noise.

Query shape: BOTH recipients AND trigger phrases are batched into groups,
not ORed together all at once.

1. Media Cloud's own setup documentation states its API is rate-limited
   to "roughly 2 requests/minute" and explicitly recommends pacing calls
   31 seconds apart -- one query per recipient across 50 recipients would
   take 25+ minutes for this source alone even paced correctly, hence
   batching recipients. Pacing is done explicitly here (_pace_queries()
   below) rather than trusting a client library, since a real run showed
   nothing here self-throttles -- 7 batch queries fired within 0.79
   seconds of each other in that run.

2. A real run also showed EVERY batch query -- not just the first, and
   regardless of query size -- failing with "API Server Error 403",
   correctly paced 31s apart, ruling both rate-limiting AND query size
   out as the cause (an earlier attempt shrank the query size on that
   hypothesis; it made no difference -- still 403 on literally every
   batch). Account/key/collection permissions are fine (a simple query
   worked in Media Cloud's own web UI). What's different between "works
   in the browser" and "always 403 from here": the requesting User-Agent.
   This now talks to the API directly via `requests` (bypassing the
   `mediacloud` pip client entirely) for two reasons: (a) to set a
   browser-like User-Agent, since the pip client leaves this at
   `requests`' default ("python-requests/x.y.z") -- a well-known
   automated-client signature that a WAF in front of search.mediacloud.org
   could easily be blocking on, independent of IP or content; and (b)
   because the pip client's own error handling
   (`RuntimeError(f"API Server Error {status}. Params: {params}")`)
   discards the actual response body/headers on a non-200 -- if this
   User-Agent change turns out not to be the fix either, the next run's
   logs need that real response text to diagnose further, which the
   library was structurally throwing away. This is not yet confirmed
   against a live run.

Each batch query ORs together a group of recipients' searchable_names
(see config_loader.Recipient.searchable_names — excludes "WHO"/"CARE"-
style ambiguous aliases, the same fix already made for Google News and
GDELT) AND ORs together a batch of trigger phrases, mirroring the
boolean shape query_builder.py already uses for Google News. At the
default batch sizes (8 recipients/query, 15 triggers/query), one daily
run costs 7 x 2 = 14 requests, correctly paced ~31s apart, adding
roughly 7 minutes to the run -- against Media Cloud's default quota of
4,000 requests/week (~98/week at this size), still comfortable headroom.
Deliberately fetches only the first page of results per batch (no
pagination_token follow-up) to keep total request count -- and therefore
total runtime -- fully predictable given the 2/minute limit; if real
runs show this under-fetching, pagination can be added later against
that same rate-limit budget.

Unlike Google News (one query per recipient, so the match is by
construction) or GDELT (recipient mentions come from GKG's own structured
NER field), a batched Media Cloud query can't cheaply attribute a
returned story to one specific recipient in the batch. RawArticle.
matched_recipient is left None here, same as PR wire articles already
are — the AI extraction step is what actually identifies the recipient
from the article text, and that's what confidence scoring and the
recipient-type tag are keyed on (see extraction.py's build_donation_entry
and score_confidence), not this pre-tag.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests

from .config_loader import Recipient
from .models import RawArticle, SourceTier
from .sources import FetchFailure

logger = logging.getLogger(__name__)

MEDIACLOUD_API_BASE = "https://search.mediacloud.org/api/"
MEDIACLOUD_PLATFORM = "onlinenews-mediacloud"

# A generic, current-looking desktop Chrome UA -- see module docstring for
# why this is set explicitly rather than left at requests' default
# ("python-requests/x.y.z"), which is a well-known automated-client
# signature.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# A real run showed nothing between requests here self-throttles -- 7
# batch queries fired within 0.79 SECONDS of each other despite Media
# Cloud's own documented "roughly 2 requests/minute" limit. Paces calls
# explicitly, the same way extraction.py's _pace_ai_calls() already does
# for Bedrock/Gemini. 31s (not a bare 30s) leaves a small margin for
# clock/measurement slack rather than sitting exactly on the limit.
MIN_SECONDS_BETWEEN_QUERIES = 31.0
_last_query_time: float = 0.0


def _pace_queries() -> None:
    global _last_query_time
    elapsed = time.monotonic() - _last_query_time
    if elapsed < MIN_SECONDS_BETWEEN_QUERIES:
        time.sleep(MIN_SECONDS_BETWEEN_QUERIES - elapsed)
    _last_query_time = time.monotonic()

# How much of a story's full body text (only returned when expanded=True)
# to keep as the RawArticle summary. Uncapped, a Media Cloud story's
# `text` field is the full article body — potentially thousands of
# words, which would both bloat the AI extraction prompt (real token
# cost) and go well beyond what the trigger-phrase filter actually needs
# (it only checks for a handful of short phrases anywhere in the text).
# 1000 characters is a couple of paragraphs — comfortably enough to
# contain the donation-relevant sentence in the vast majority of news
# articles, which lead with their most newsworthy fact.
SUMMARY_CHARS_FROM_TEXT = 1000

# How much of a failed response's body to keep in a FetchFailure/log line
# -- enough to show a Cloudflare/WAF block page's telltale text (or a
# proper JSON auth-error message) without dumping an entire HTML page
# into the log.
FAILURE_BODY_CHARS = 500


@dataclass
class MediaCloudFetchStats:
    queries_made: int
    candidates_found: int


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _build_query(recipients: list[Recipient], triggers: list[str]) -> str:
    names = [n for r in recipients for n in r.searchable_names]
    name_clause = " OR ".join(f'"{n}"' for n in names)
    trigger_clause = " OR ".join(f'"{t}"' for t in triggers)
    return f"({name_clause}) AND ({trigger_clause})"


def _fetch_stories(api_key: str, query: str, start_date: date, end_date: date,
                    collection_id: int) -> list[dict]:
    """Calls Media Cloud's search/story-list endpoint directly with
    `requests` rather than through the `mediacloud` pip client -- see
    module docstring. Raises RuntimeError (with the real response status,
    headers, and a body excerpt) on anything other than 200.
    """
    params = {
        "q": query,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "platform": MEDIACLOUD_PLATFORM,
        "cs": str(collection_id),
        "expanded": 1,
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/json",
        "User-Agent": BROWSER_USER_AGENT,
    }
    r = requests.get(MEDIACLOUD_API_BASE + "search/story-list", params=params,
                      headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(
            f"API Server Error {r.status_code}. "
            f"Response headers: {dict(r.headers)!r}. "
            f"Body (first {FAILURE_BODY_CHARS} chars): {r.text[:FAILURE_BODY_CHARS]!r}"
        )
    stories = r.json()["stories"]
    for s in stories:
        # Mirrors mediacloud.api.SearchApi.story_list's own post-
        # processing -- publish_date comes back as a string (sometimes a
        # full timestamp), converted here to a plain date so downstream
        # code (_story_to_article below) can treat it uniformly.
        s["publish_date"] = date.fromisoformat(s["publish_date"][:10]) if s.get("publish_date") else None
    return stories


def _story_to_article(story: dict) -> RawArticle:
    publish_date = story.get("publish_date")
    # publish_date is a plain date (see _fetch_stories above), and
    # datetime.fromisoformat() (used downstream by clustering.py's
    # _parse_published) happily parses a bare "YYYY-MM-DD" string as
    # midnight UTC, so no extra conversion is needed here.
    published = publish_date.isoformat() if publish_date else None
    text = (story.get("text") or "")[:SUMMARY_CHARS_FROM_TEXT]
    return RawArticle(
        title=(story.get("title") or "").strip(),
        url=(story.get("url") or "").strip(),
        published=published,
        source_name=story.get("media_name") or "Unknown (via Media Cloud)",
        source_tier=SourceTier.GENERAL_NEWS,
        summary=text,
        fetch_source="Media Cloud",
    )


def fetch_mediacloud_articles(
    recipients: list[Recipient],
    triggers: list[str],
    collection_id: int,
    max_age_days: int,
    recipients_per_query: int = 8,
    triggers_per_query: int = 15,
) -> tuple[list[RawArticle], list[FetchFailure], MediaCloudFetchStats]:
    """Raises RuntimeError immediately if MEDIACLOUD_API_KEY isn't set —
    with every batch query bound to fail identically on a missing/invalid
    key, failing fast with one clear message beats silently recording the
    same failure ~7 times, several minutes apart, deep into a run (the
    same reasoning as extraction.py's FatalExtractionError for Bedrock).
    A real per-batch network/API error, by contrast, is recorded as a
    FetchFailure and does not stop the run — one bad batch shouldn't cost
    every other recipient's coverage from this source.
    """
    api_key = os.environ.get("MEDIACLOUD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "search.mediacloud_enabled is true in settings.yaml but MEDIACLOUD_API_KEY is not "
            "set. Get a free key at search.mediacloud.org and add it as a GitHub Actions secret "
            "(see .github/workflows/daily.yml)."
        )

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=max_age_days)

    def run_one_query(query: str, label: str) -> tuple[list, FetchFailure | None]:
        _pace_queries()
        try:
            return _fetch_stories(api_key, query, start_date, end_date, collection_id), None
        except Exception as e:
            error_text = str(e)
            if "API Server Error 429" not in error_text:
                # Broad on purpose: a real HTTP error surfaces as the
                # RuntimeError _fetch_stories raises above, but a network
                # blip surfaces as a plain requests exception instead --
                # neither should take down the whole run (see sources.py's
                # module docstring).
                logger.warning("Media Cloud fetch failed for %s: %s", label, e)
                return [], FetchFailure(label, f"collection {collection_id}", str(e))
            # A real run hit this even with pacing intended (the pacing
            # just added turned out to be needed because it wasn't
            # actually happening before -- see MIN_SECONDS_BETWEEN_
            # QUERIES above). One retry after a longer, deliberately
            # generous wait, on the chance the account is still working
            # off a burst from before this fix existed; anything past
            # that is treated as a real failure rather than burning the
            # rest of this source's time budget.
            logger.warning("Media Cloud rate-limited for %s, waiting %.0fs before one retry: %s",
                            label, MIN_SECONDS_BETWEEN_QUERIES * 2, e)
            time.sleep(MIN_SECONDS_BETWEEN_QUERIES * 2)
            try:
                return _fetch_stories(api_key, query, start_date, end_date, collection_id), None
            except Exception as retry_e:
                logger.warning("Media Cloud fetch failed for %s (after retry): %s", label, retry_e)
                return [], FetchFailure(label, f"collection {collection_id}", str(retry_e))

    all_articles: list[RawArticle] = []
    failures: list[FetchFailure] = []
    queries_made = 0

    for recipient_batch in _batched(recipients, recipients_per_query):
        for trigger_batch in _batched(triggers, triggers_per_query):
            label = f"Media Cloud: {', '.join(r.name for r in recipient_batch)}"
            query = _build_query(recipient_batch, trigger_batch)
            queries_made += 1
            stories, failure = run_one_query(query, label)
            if failure:
                failures.append(failure)
            all_articles.extend(_story_to_article(s) for s in stories)

    stats = MediaCloudFetchStats(queries_made=queries_made, candidates_found=len(all_articles))
    logger.info("Media Cloud: %d quer%s across %d recipients (batched), found %d candidate "
                "article(s) (%d fetch failures).", queries_made, "y" if queries_made == 1 else "ies",
                len(recipients), stats.candidates_found, len(failures))
    return all_articles, failures, stats
