"""Fetches raw items from Google News RSS and PR wire RSS feeds.

(GDELT is a separate source, handled entirely by gdelt_gkg.py — it comes
from bulk file downloads rather than a live search API, so it doesn't fit
this file's feedparser/RSS-shaped fetching. FetchFailure below is shared
with that module though, so both report failures the same way.)

Both sources here are public, syndication-oriented feeds (not an
adversarial scrape of a site that doesn't want automated access), which is
why this pipeline uses them as primary sources rather than scraping
company/news pages directly. See README for the reasoning.

Network failures here should never crash the whole run — a feed being
temporarily down is normal, and it gets recorded so it shows up on the
"Sources & limitations" page rather than silently disappearing.
"""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
import requests

from .models import RawArticle, SourceTier

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = "CorporateGivingMonitor/1.0 (+https://github.com/; non-commercial research)"


def _entry_published_iso(entry) -> str | None:
    """Prefers feedparser's already-parsed struct_time (published_parsed)
    over the raw published string — feedparser normalises whatever date
    format the feed used, so this is far more reliable for downstream age
    filtering than trying to re-parse entry.published ourselves.
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc).isoformat()
    return entry.get("published") or entry.get("updated")


@dataclass
class FetchFailure:
    source_label: str
    url: str
    error: str


def _fetch_feed(url: str, source_label: str) -> tuple[feedparser.FeedParserDict, FetchFailure | None]:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            return parsed, FetchFailure(source_label, url, f"Feed parse error: {parsed.bozo_exception}")
        return parsed, None
    except requests.RequestException as e:
        logger.warning("Fetch failed for %s (%s): %s", source_label, url, e)
        return feedparser.FeedParserDict(entries=[]), FetchFailure(source_label, url, str(e))


def fetch_google_news_query(url: str, recipient_name: str) -> tuple[list[RawArticle], FetchFailure | None]:
    parsed, failure = _fetch_feed(url, f"Google News: {recipient_name}")
    articles = []
    for entry in parsed.entries:
        # Google News RSS wraps the real publisher name in the 'source' field
        # when present; fall back to "Google News" if not.
        source_name = getattr(entry, "source", {}).get("title") if hasattr(entry, "source") else None
        articles.append(RawArticle(
            title=entry.get("title", "").strip(),
            url=entry.get("link", "").strip(),
            published=_entry_published_iso(entry),
            source_name=source_name or "Unknown (via Google News)",
            source_tier=SourceTier.GENERAL_NEWS,
            summary=entry.get("summary", ""),
            matched_recipient=recipient_name,
            fetch_source="Google News",
        ))
    return articles, failure


def fetch_pr_wire_feed(label: str, url: str) -> tuple[list[RawArticle], FetchFailure | None]:
    parsed, failure = _fetch_feed(url, label)
    articles = []
    for entry in parsed.entries:
        articles.append(RawArticle(
            title=entry.get("title", "").strip(),
            url=entry.get("link", "").strip(),
            published=_entry_published_iso(entry),
            source_name=label.split(" - ")[0],  # e.g. "PR Newswire"
            source_tier=SourceTier.WIRE,
            summary=entry.get("summary", ""),
            fetch_source="PR Wire",
        ))
    return articles, failure


def fetch_all(
    recipient_queries: list[tuple[str, str]],
    pr_wire_feeds: list[tuple[str, str]],
) -> tuple[list[RawArticle], list[FetchFailure]]:
    """recipient_queries: [(recipient_name, google_news_rss_url), ...] — a
    list, not a dict, because a recipient can have more than one query
    (trigger phrases are batched across several simpler queries; see
    query_builder.build_recipient_trigger_queries).
    """
    all_articles: list[RawArticle] = []
    failures: list[FetchFailure] = []

    for recipient_name, url in recipient_queries:
        articles, failure = fetch_google_news_query(url, recipient_name)
        all_articles.extend(articles)
        if failure:
            failures.append(failure)

    for label, url in pr_wire_feeds:
        articles, failure = fetch_pr_wire_feed(label, url)
        all_articles.extend(articles)
        if failure:
            failures.append(failure)

    return all_articles, failures
