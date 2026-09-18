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
not ORed together all at once, since Media Cloud's own setup
documentation states its API is rate-limited to "roughly 2 requests/
minute" and explicitly recommends pacing calls 31 seconds apart -- one
query per recipient across 50 recipients would take 25+ minutes for this
source alone even paced correctly, hence batching. Pacing is done
explicitly here (_pace_queries() below) rather than trusting a client
library, since a real run showed nothing here self-throttles -- 7 batch
queries fired within 0.79 seconds of each other in that run.

Calls the API directly via `requests` rather than through the
`mediacloud` pip client. Two unrelated real-run investigations drove
this, in order:

1. Every batch query, not just the first, and regardless of query size
   or pacing, failed with "API Server Error 403" -- ruling out rate
   limiting and query size as the cause. The pip client's own error
   handling (`RuntimeError(f"API Server Error {status}. Params: {params}")`)
   discards the actual response body on a non-200, so there was no way
   to tell an auth problem from a WAF block from that message alone.
   Calling the API directly instead captures the real response body and
   headers on any failure.
2. That paid off immediately: the captured body was
   `{"status": "error", "note": "You are not permitted to fetch
   `expanded` stories."}` -- a real, specific, plan-tier permission
   error, not a WAF or User-Agent issue at all (an earlier commit's
   User-Agent-spoofing guess, made before the real body was visible, is
   removed here as unneeded). `expanded=1` (full story body text) isn't
   available on this API key's plan; every request asked for it, so
   every request was rejected identically no matter the content, size,
   or timing. The fix is simply not asking for it.

Losing expanded=1 means story results no longer include body text, only
title/url/publish_date/media_name -- so RawArticle.summary is left empty
here (there's nothing to put in it) and RawArticle.matched_trigger is
set directly from the query's own trigger batch instead of being derived
from title+summary text the way clustering.filter_by_trigger_phrase does
for other sources. That's not a workaround, it's more correct: the
trigger phrase is already ANDed into the server-side query for this
batch, so a returned story is guaranteed to match one of them in its
full text even though we can't see that text locally to re-derive which
one. (See filter_by_trigger_phrase's own handling of an article that
already carries a matched_trigger.)

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

# How much of a failed response's body to keep in a FetchFailure/log line
# -- enough to show what the response actually says (a JSON error detail,
# an HTML block page, whatever it turns out to be) without dumping an
# entire page into the log. This is what surfaced the real cause of the
# 403s below -- see module docstring.
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

    Deliberately does NOT request expanded=1 (full story body text) --
    this account's plan doesn't have access to it (see module docstring),
    so results only carry title/url/publish_date/media_name.
    """
    params = {
        "q": query,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "platform": MEDIACLOUD_PLATFORM,
        "cs": str(collection_id),
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/json",
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
        # publish_date comes back as a string (sometimes a full
        # timestamp), converted here to a plain date so downstream code
        # (_story_to_article below) can treat it uniformly.
        s["publish_date"] = date.fromisoformat(s["publish_date"][:10]) if s.get("publish_date") else None
    return stories


def _story_to_article(story: dict, matched_trigger: str) -> RawArticle:
    publish_date = story.get("publish_date")
    # publish_date is a plain date (see _fetch_stories above), and
    # datetime.fromisoformat() (used downstream by clustering.py's
    # _parse_published) happily parses a bare "YYYY-MM-DD" string as
    # midnight UTC, so no extra conversion is needed here.
    published = publish_date.isoformat() if publish_date else None
    return RawArticle(
        title=(story.get("title") or "").strip(),
        url=(story.get("url") or "").strip(),
        published=published,
        source_name=story.get("media_name") or "Unknown (via Media Cloud)",
        source_tier=SourceTier.GENERAL_NEWS,
        # No body text available on this account's plan (see module
        # docstring) -- matched_trigger is set directly below instead of
        # being derived from title+summary text the way other sources'
        # candidates are.
        summary="",
        matched_trigger=matched_trigger,
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
            # Not the literal phrase that matched (we can't see body text
            # to tell) -- honest about that, but still a real, server-
            # verified match against one of this batch's phrases.
            matched_trigger = f"server-side match (1 of {len(trigger_batch)} trigger phrases)"
            queries_made += 1
            stories, failure = run_one_query(query, label)
            if failure:
                failures.append(failure)
            all_articles.extend(_story_to_article(s, matched_trigger) for s in stories)

    stats = MediaCloudFetchStats(queries_made=queries_made, candidates_found=len(all_articles))
    logger.info("Media Cloud: %d quer%s across %d recipients (batched), found %d candidate "
                "article(s) (%d fetch failures).", queries_made, "y" if queries_made == 1 else "ies",
                len(recipients), stats.candidates_found, len(failures))
    return all_articles, failures, stats
