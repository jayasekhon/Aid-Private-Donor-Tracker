"""Turns config (recipients, countries, trigger phrases) into a bounded set
of search queries.

Design note: we deliberately anchor every query on a RECIPIENT (a UN
agency / INGO / NGO), not a company, because the company side of this
is open-ended ("could be anyone anywhere" — see project brief). Anchoring
on the recipient list means a company we've never heard of still surfaces,
as long as it donated to an org we're watching.

We generate two query shapes per recipient:
  1. recipient + a rotating sample of trigger phrases (broad, catches most)
  2. recipient + country (for the subset of stories that name a place)
Shape 2 is intentionally optional / best-effort: a lot of real donations
never name a country at all (core/unearmarked funding), which is exactly
why "scope: unspecified" exists downstream instead of forcing a country match.
"""
from __future__ import annotations

import itertools
from urllib.parse import quote_plus

from .config_loader import Recipient

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"


def build_recipient_trigger_queries(
    recipients: list[Recipient],
    triggers: list[str],
    max_triggers_per_recipient: int = 6,
) -> list[str]:
    """One query per recipient, OR-ing together a capped sample of trigger
    phrases so we don't explode into recipients x triggers separate calls.
    """
    queries = []
    for r in recipients:
        name = r.name
        trigger_sample = triggers[:max_triggers_per_recipient]
        trigger_clause = " OR ".join(f'"{t}"' for t in trigger_sample)
        queries.append(f'"{name}" ({trigger_clause})')
    return queries


def build_recipient_country_queries(
    recipients: list[Recipient],
    countries: list[str],
) -> list[str]:
    """One query per (recipient, country) pair. This is the more expensive,
    exhaustive shape — used sparingly, e.g. once a week or when the broad
    query above is clearly missing regional stories.
    """
    queries = []
    for r, c in itertools.product(recipients, countries):
        queries.append(f'"{r.name}" "{c}" donation')
    return queries


def google_news_rss_url(query: str, language: str = "en-US", country: str = "US") -> str:
    encoded = quote_plus(query)
    return (
        f"{GOOGLE_NEWS_RSS_BASE}?q={encoded}"
        f"&hl={language}&gl={country}&ceid={country}:{language.split('-')[0]}"
    )


def build_daily_query_plan(
    recipients: list[Recipient],
    triggers: list[str],
    countries: list[str],
    include_country_queries: bool = False,
) -> list[str]:
    """Returns the Google News RSS URLs to fetch for one daily run."""
    queries = build_recipient_trigger_queries(recipients, triggers)
    if include_country_queries:
        queries += build_recipient_country_queries(recipients, countries)
    return [google_news_rss_url(q) for q in queries]
