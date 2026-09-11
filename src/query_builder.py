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

# Aliases that double as ordinary English words are unusable as bare Google
# News search terms: Google matches the word anywhere in the article, not
# just as an org reference, so searching `"WHO"` alone matches the pronoun
# "who" in effectively any headline. A real run found 108 of 303 candidates
# that passed the trigger-phrase filter were false WHO matches (Taylor
# Swift donations, Trump policy news, etc.) — pure noise that still cost an
# AI call each to correctly reject. Dropped here, in query construction,
# only: "WHO" stays in recipients.txt and is still used everywhere alias
# matching runs against actual article text/domains rather than free-text
# search (e.g. tag_official_sources() matching a GDELT source against
# "who.int"), where the false-positive risk is negligible. The World
# Health Organization's full name is still searched via r.name, so WHO
# coverage isn't lost — just no longer keyed on the bare acronym.
AMBIGUOUS_SEARCH_ALIASES = {"who"}


def _searchable_names(r: Recipient) -> list[str]:
    return [n for n in r.all_names if n.strip().lower() not in AMBIGUOUS_SEARCH_ALIASES]


def build_recipient_trigger_queries(
    recipients: list[Recipient],
    triggers: list[str],
    triggers_per_query: int = 8,
    max_age_days: int | None = None,
) -> list[tuple[str, str]]:
    """Returns (recipient_name, query) pairs — one or more per recipient.

    Trigger phrases are split into batches of triggers_per_query, each its
    own query, rather than one query OR-ing all ~26 phrases together (and
    all of a recipient's aliases) in one giant boolean expression. A real
    run with the single-mega-query approach returned suspiciously few raw
    articles (377 across 50 recipients — ~7.5 each, far below what a single
    well-known org name alone normally returns from Google News), which
    looks like the same kind of unreliability we already found with the
    "when:" operator: Google's RSS search endpoint doesn't reliably
    evaluate very large/complex boolean queries, silently under-matching
    rather than erroring. Splitting into smaller, simpler queries trades
    more HTTP requests (cheap — public, unauthenticated RSS, no quota) for
    queries Google is actually likely to evaluate in full.

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
    pairs = []
    for r in recipients:
        name_clause = " OR ".join(f'"{n}"' for n in _searchable_names(r))
        for i in range(0, len(triggers), triggers_per_query):
            batch = triggers[i:i + triggers_per_query]
            trigger_clause = " OR ".join(f'"{t}"' for t in batch)
            query = f'({name_clause}) ({trigger_clause})'
            if max_age_days:
                query += f" when:{max_age_days}d"
            pairs.append((r.name, query))
    return pairs


def build_recipient_country_queries(
    recipients: list[Recipient],
    countries: list[str],
    max_age_days: int | None = None,
) -> list[tuple[str, str]]:
    """One (recipient_name, query) pair per (recipient, country) pair. This
    is the more expensive, exhaustive shape — used sparingly, e.g. once a
    week or when the broad query above is clearly missing regional stories.
    """
    pairs = []
    for r, c in itertools.product(recipients, countries):
        name_clause = " OR ".join(f'"{n}"' for n in _searchable_names(r))
        query = f'({name_clause}) "{c}" donation'
        if max_age_days:
            query += f" when:{max_age_days}d"
        pairs.append((r.name, query))
    return pairs


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
) -> list[tuple[str, str]]:
    """Returns (recipient_name, google_news_rss_url) pairs to fetch for one
    daily run."""
    pairs = build_recipient_trigger_queries(recipients, triggers, max_age_days=max_age_days)
    if include_country_queries:
        pairs += build_recipient_country_queries(recipients, countries, max_age_days=max_age_days)
    return [(name, google_news_rss_url(q)) for name, q in pairs]
