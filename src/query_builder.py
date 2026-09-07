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
    max_triggers_per_recipient: int | None = None,
    max_age_days: int | None = None,
) -> list[str]:
    """One query per recipient, OR-ing together the trigger phrases so we
    don't explode into recipients x triggers separate calls.

    max_triggers_per_recipient caps how many trigger phrases go into each
    recipient's query, if you ever need to shorten it (e.g. hitting a feed's
    URL length limit) — but this doesn't change the number of Google News
    RSS requests made (still one per recipient either way), only which
    phrases that one request can match on. Previously defaulted to 6 out of
    ~26 phrases, which meant Google's search itself could never see most of
    the trigger list (only the post-fetch filter in clustering.py checked
    the full list) — a real story phrased with e.g. "donated to support"
    (trigger #25) would never even be searched for. Default is now "use all
    of them" — a 26-phrase query is under 1KB, nowhere near any URL limit.

    max_age_days, if given, adds Google News' "when:Nd" operator so the
    search itself is restricted to recent results — Google News ranks by
    relevance, not recency, so without this an old story that matches the
    keywords well can outrank (and crowd out) today's actual news.

    Also ORs in every alias from recipients.txt, not just the primary name —
    e.g. WFP's own line lists "WFP" and "World Food Program" as aliases
    specifically because that's how press actually refers to it, but until
    now only the formal "World Food Programme" (note the UK spelling) was
    ever searched for. Aliases existed in config and were fully documented
    as "other names/abbreviations the press might use" but were never
    actually read by the query builder — a real gap in coverage, not a
    tuning knob.
    """
    queries = []
    for r in recipients:
        name_clause = " OR ".join(f'"{n}"' for n in r.all_names)
        trigger_sample = triggers[:max_triggers_per_recipient] if max_triggers_per_recipient else triggers
        trigger_clause = " OR ".join(f'"{t}"' for t in trigger_sample)
        query = f'({name_clause}) ({trigger_clause})'
        if max_age_days:
            query += f" when:{max_age_days}d"
        queries.append(query)
    return queries


def build_recipient_country_queries(
    recipients: list[Recipient],
    countries: list[str],
    max_age_days: int | None = None,
) -> list[str]:
    """One query per (recipient, country) pair. This is the more expensive,
    exhaustive shape — used sparingly, e.g. once a week or when the broad
    query above is clearly missing regional stories.
    """
    queries = []
    for r, c in itertools.product(recipients, countries):
        name_clause = " OR ".join(f'"{n}"' for n in r.all_names)
        query = f'({name_clause}) "{c}" donation'
        if max_age_days:
            query += f" when:{max_age_days}d"
        queries.append(query)
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
    max_age_days: int | None = None,
) -> list[str]:
    """Returns the Google News RSS URLs to fetch for one daily run."""
    queries = build_recipient_trigger_queries(recipients, triggers, max_age_days=max_age_days)
    if include_country_queries:
        queries += build_recipient_country_queries(recipients, countries, max_age_days=max_age_days)
    return [google_news_rss_url(q) for q in queries]
