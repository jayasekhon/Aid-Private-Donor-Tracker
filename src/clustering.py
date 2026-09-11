"""Two jobs, both designed to keep AI (Gemini) calls to a minimum:

1. filter_by_trigger_phrase() — a free, instant keyword pass that throws
   out articles that clearly aren't about a donation at all. This is what
   keeps us inside the free-tier daily request budget: without it, a
   single day's Google News + PR wire haul could be 500-1000+ raw items,
   and we can't afford one AI call per item.

2. cluster_articles() — groups the survivors so the same real-world event
   covered by 5 outlets becomes ONE AI call, not five. Uses fuzzy title
   matching (rapidfuzz) plus same-recipient grouping as the similarity
   signal; good enough for "is this obviously the same story" without
   needing embeddings or another API call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from rapidfuzz import fuzz

from .config_loader import Recipient
from .models import ArticleCluster, RawArticle, SourceTier

logger = logging.getLogger(__name__)

# How many rejected-candidate titles to log per run, for eyeballing whether
# the trigger-phrase filter looks too strict (real donation stories being
# discarded) vs. correctly rejecting noise (recipient mentioned, but not in
# a donation context) — capped to keep logs readable on a high-volume day.
REJECTED_SAMPLE_LOG_LIMIT = 15


def _parse_published(published: str | None) -> datetime | None:
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

TITLE_SIMILARITY_THRESHOLD = 62  # 0-100, rapidfuzz token_set_ratio
# NOTE: this is a best-effort, imperfect signal. Headlines about the same
# real-world donation can be worded differently enough (different verbs,
# reordered clauses, one outlet adding a location the other omits) that
# some duplicates will slip past this and get sent to the AI step as two
# separate clusters. That's an acceptable failure mode BECAUSE the rolling
# event store (see store.py) does a second, more reliable dedupe pass
# after extraction, comparing actual (donor, recipient, amount, month)
# facts rather than headline text. Clustering here is purely a cost
# optimisation to reduce AI calls, not the final source of truth on
# duplicates.


def filter_by_recency(articles: list[RawArticle], max_age_days: int) -> tuple[list[RawArticle], int]:
    """Drops articles older than max_age_days. This is a backstop, not the
    primary control — the Google News queries already carry a "when:Nd"
    restriction (see query_builder.py) — but PR wire feeds have no
    equivalent, and Google's own restriction isn't airtight. An article
    with no parseable publish date is kept rather than dropped: we'd rather
    risk one stale item slipping through than silently lose a genuinely
    current story just because its feed entry omitted a date.
    Returns (kept_articles, number_dropped_as_stale).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    kept = []
    dropped = 0
    for a in articles:
        published = _parse_published(a.published)
        if published is not None and published < cutoff:
            dropped += 1
            continue
        kept.append(a)
    return kept, dropped


def filter_by_trigger_phrase(articles: list[RawArticle], triggers: list[str]) -> list[RawArticle]:
    kept = []
    rejected_sample = []
    for a in articles:
        haystack = f"{a.title} {a.summary}".lower()
        matched = False
        for phrase in triggers:
            if phrase in haystack:
                a.matched_trigger = phrase
                kept.append(a)
                matched = True
                break
        if not matched and len(rejected_sample) < REJECTED_SAMPLE_LOG_LIMIT:
            rejected_sample.append(a)

    if rejected_sample:
        logger.info("Sample of %d rejected candidate(s) (of %d total not passing the trigger-phrase "
                     "filter) — eyeball these for real donation stories the phrase list is missing:",
                     len(rejected_sample), len(articles) - len(kept))
        for a in rejected_sample:
            logger.info("  [%s, via %s] %s", a.matched_recipient or "?", a.fetch_source, a.title)

    if kept:
        # Per-source breakdown up front — without this, it's impossible to
        # tell from the log alone whether GDELT/PR-wire candidates are
        # even reaching this stage, versus Google News dominating simply
        # because it contributes far more raw volume to begin with.
        by_source: dict[str, int] = {}
        for a in kept:
            by_source[a.fetch_source] = by_source.get(a.fetch_source, 0) + 1
        breakdown = ", ".join(f"{count} {source}" for source, count in sorted(by_source.items()))
        logger.info("%d candidate(s) passed the trigger-phrase filter and are headed to the AI step (%s):",
                     len(kept), breakdown)
        for a in kept:
            logger.info("  [%s, matched %r, via %s] %s", a.matched_recipient or "?", a.matched_trigger, a.fetch_source, a.title)

    return kept


def tag_country(articles: list[RawArticle], countries: list[str]) -> None:
    """Mutates articles in place, tagging the first country name found
    (if any) in the title or summary. Leaves matched_country as None
    when no country is named — that is expected and handled downstream
    as 'scope: unspecified', not an error.
    """
    for a in articles:
        haystack = f"{a.title} {a.summary}"
        for country in countries:
            if country.lower() in haystack.lower():
                a.matched_country = country
                break


# Aliases shorter than this are excluded from the official-source check
# below. 3 is deliberately low enough to still catch the exact case this
# was built for — short acronyms (WFP, WHO, IOM) that genuinely appear
# in an org's own domain (wfp.org, who.int) — even though a couple of
# those (e.g. "who", a real English word) carry some real false-positive
# risk against an unrelated source name. Accepted: this is a confidence
# SIGNAL that nudges a score, not a publish/reject gate, so the cost of
# an occasional undeserved bump is low. A 1-2 character alias (if one
# ever existed) would still be excluded here as too risky even for that.
OFFICIAL_SOURCE_MIN_NAME_LENGTH = 3


def tag_official_sources(articles: list[RawArticle], recipients: list[Recipient]) -> None:
    """Mutates articles in place, upgrading source_tier to OFFICIAL when
    the article's source_name matches the matched recipient's own name or
    an alias — i.e. this looks like the recipient's own press channel
    talking about itself, the single most credible source there is.

    This is genuinely checkable at fetch time for GDELT and Google News,
    which both already carry a matched_recipient. It matters most for
    GDELT specifically: its source_name is the article's actual domain
    (e.g. "wfp.org"), and recipients.txt's aliases already include the
    short-form names ("WFP") that commonly appear IN such a domain —
    added specifically because press/domains refer to orgs that way (see
    query_builder.py's docstring on why aliases were added to Google
    News queries in the first place). Google News' source_name is a
    publisher display name rather than a domain, so this fires less
    often there, but still correctly catches cases where Google
    attributes an article directly to the org's own name.

    PR wire articles have no matched_recipient (PR wires aren't
    recipient-anchored at fetch time) and are left untouched — they keep
    their existing WIRE tier regardless.
    """
    recipients_by_name = {r.name: r for r in recipients}
    for a in articles:
        if not a.matched_recipient:
            continue
        recipient = recipients_by_name.get(a.matched_recipient)
        if recipient is None:
            continue
        source_l = a.source_name.strip().lower()
        for name in recipient.all_names:
            name_l = name.strip().lower()
            if len(name_l) < OFFICIAL_SOURCE_MIN_NAME_LENGTH:
                continue
            if name_l in source_l or source_l in name_l:
                a.source_tier = SourceTier.OFFICIAL
                break


def cluster_articles(articles: list[RawArticle]) -> list[ArticleCluster]:
    clusters: list[ArticleCluster] = []
    for article in articles:
        placed = False
        for cluster in clusters:
            # Only compare within the same recipient bucket (when known) —
            # cheap way to avoid false-merging unrelated stories that
            # happen to share generic wording ("donates funding to support...").
            same_recipient_bucket = (
                article.matched_recipient is None
                or any(m.matched_recipient == article.matched_recipient for m in cluster.articles)
            )
            if not same_recipient_bucket:
                continue
            for existing in cluster.articles:
                score = fuzz.token_set_ratio(article.title.lower(), existing.title.lower())
                if score >= TITLE_SIMILARITY_THRESHOLD:
                    cluster.articles.append(article)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append(ArticleCluster(cluster_id=f"c{len(clusters)+1}", articles=[article]))
    return clusters

def rank_clusters_by_priority(clusters: list[ArticleCluster]) -> list[ArticleCluster]:
    """Sorts clusters best-candidate-first, so that when the daily AI budget
    is scarce (as low as ~18-20 calls/day on Gemini's current free tier —
    see settings.yaml), the calls spent go on the most promising stories
    rather than whichever happened to appear first.

    Heuristic score, cheapest signals first (no AI involved):
      +3 if the cluster has more than one independent source (corroborated)
      +2 if the best source in the cluster is official/wire-tier
      +2 if any article's headline contains a currency symbol or "million"/
         "billion" (suggests a concrete figure is likely stated)
      +2 if the cluster's most recently published article is within 3 days
      +1 if the cluster's most recently published article is within 7 days
         (recency matters: without this, an old-but-well-corroborated story
         can keep outscoring today's actual news for the scarce AI budget)
      +1 if any article's headline contains a known corporate legal suffix
         (Inc, Corp, Ltd, plc, LLC) — a weak signal this is company-specific
         news rather than a generic sector/agency story
    """
    import re

    CURRENCY_PATTERN = re.compile(r"[$€£]|\bmillion\b|\bbillion\b", re.IGNORECASE)
    CORPORATE_SUFFIX_PATTERN = re.compile(r"\b(inc\.?|corp\.?|ltd\.?|plc|llc|co\.?)\b", re.IGNORECASE)

    def most_recent_published(cluster: ArticleCluster) -> datetime | None:
        dates = [d for d in (_parse_published(a.published) for a in cluster.articles) if d is not None]
        return max(dates) if dates else None

    def score(cluster: ArticleCluster) -> int:
        s = 0
        if cluster.independent_source_count > 1:
            s += 3
        if cluster.best_source_tier in (SourceTier.OFFICIAL, SourceTier.WIRE):
            s += 2
        titles = " ".join(a.title for a in cluster.articles)
        if CURRENCY_PATTERN.search(titles):
            s += 2
        newest = most_recent_published(cluster)
        if newest is not None:
            age = datetime.now(timezone.utc) - newest
            if age <= timedelta(days=3):
                s += 2
            elif age <= timedelta(days=7):
                s += 1
        if CORPORATE_SUFFIX_PATTERN.search(titles):
            s += 1
        return s

    return sorted(clusters, key=score, reverse=True)
