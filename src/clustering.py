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

from rapidfuzz import fuzz

from .models import ArticleCluster, RawArticle

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


def filter_by_trigger_phrase(articles: list[RawArticle], triggers: list[str]) -> list[RawArticle]:
    kept = []
    for a in articles:
        haystack = f"{a.title} {a.summary}".lower()
        for phrase in triggers:
            if phrase in haystack:
                a.matched_trigger = phrase
                kept.append(a)
                break
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
