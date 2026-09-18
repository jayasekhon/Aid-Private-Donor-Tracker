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

Query shape: recipients are BATCHED into one query per batch, not one
query per recipient like Google News does. Media Cloud's official client
self-throttles to 2 requests/minute (mediacloud.api.BaseApi.
RATE_LIMIT_PER_MINUTE) — one query per recipient across 50 recipients
would take 25+ minutes for this source alone. At the default batch size
(8 recipients/query), one daily run costs ~7 requests (~49/week) against
Media Cloud's default quota of 4,000 requests/week — comfortable
headroom for the batch size to shrink later if per-recipient attribution
turns out to matter more than query count. Each batch query ORs
together a group of recipients' searchable_names (see
config_loader.Recipient.searchable_names — excludes "WHO"/"CARE"-style
ambiguous aliases, the same fix already made for Google News and GDELT)
AND ORs together every trigger phrase, mirroring the boolean shape
query_builder.py already uses. Deliberately fetches only the first page
of results per batch (no pagination_token follow-up) to keep total
request count -- and therefore total runtime -- fully predictable given
the 2/minute limit; if real runs show this under-fetching, pagination can
be added later against that same rate-limit budget.

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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config_loader import Recipient
from .models import RawArticle, SourceTier
from .sources import FetchFailure

logger = logging.getLogger(__name__)

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


def _story_to_article(story: dict) -> RawArticle:
    publish_date = story.get("publish_date")
    # publish_date is a plain date (see mediacloud.types.Story), and
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

    import mediacloud.api

    search_api = mediacloud.api.SearchApi(api_key)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=max_age_days)

    all_articles: list[RawArticle] = []
    failures: list[FetchFailure] = []
    queries_made = 0

    for batch in _batched(recipients, recipients_per_query):
        label = f"Media Cloud: {', '.join(r.name for r in batch)}"
        query = _build_query(batch, triggers)
        queries_made += 1
        try:
            stories, _pagination_token = search_api.story_list(
                query, start_date=start_date, end_date=end_date,
                collection_ids=[collection_id], expanded=True,
            )
        except Exception as e:
            # Broad on purpose: the client can raise its own MCException/
            # APIResponseError for auth/quota/HTTP issues, or a plain
            # requests exception for a network blip — none of them should
            # take down the whole run (see sources.py's module docstring).
            logger.warning("Media Cloud fetch failed for %s: %s", label, e)
            failures.append(FetchFailure(label, f"collection {collection_id}", str(e)))
            continue

        all_articles.extend(_story_to_article(s) for s in stories)

    stats = MediaCloudFetchStats(queries_made=queries_made, candidates_found=len(all_articles))
    logger.info("Media Cloud: %d quer%s across %d recipients (batched), found %d candidate "
                "article(s) (%d fetch failures).", queries_made, "y" if queries_made == 1 else "ies",
                len(recipients), stats.candidates_found, len(failures))
    return all_articles, failures, stats
