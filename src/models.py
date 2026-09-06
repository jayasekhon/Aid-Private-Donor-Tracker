"""Shared data structures for the donation-tracking pipeline.

Keeping these in one place means every module (fetching, clustering,
extraction, storage, rendering) agrees on the exact shape of an "item"
as it moves through the system.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SourceTier(str, Enum):
    OFFICIAL = "official_press_release_or_agency_newsroom"
    WIRE = "wire_service"
    GENERAL_NEWS = "general_news"
    SOCIAL = "social_media_or_blog"


class EventStatus(str, Enum):
    NEW = "new_commitment"
    RENEWAL = "renewed_partnership"
    UNCLEAR = "unclear"


@dataclass
class RawArticle:
    """A single fetched item before any clustering or AI processing."""
    title: str
    url: str
    published: Optional[str]      # ISO string if known, else None
    source_name: str              # e.g. "Reuters", "PR Newswire"
    source_tier: SourceTier
    summary: str = ""             # snippet/description from the feed
    matched_recipient: Optional[str] = None
    matched_country: Optional[str] = None
    matched_trigger: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_tier"] = self.source_tier.value
        return d


@dataclass
class ArticleCluster:
    """A group of RawArticles believed to describe the same underlying event."""
    cluster_id: str
    articles: list[RawArticle] = field(default_factory=list)

    @property
    def best_source_tier(self) -> SourceTier:
        order = [SourceTier.OFFICIAL, SourceTier.WIRE,
                 SourceTier.GENERAL_NEWS, SourceTier.SOCIAL]
        tiers = {a.source_tier for a in self.articles}
        for t in order:
            if t in tiers:
                return t
        return SourceTier.SOCIAL

    @property
    def independent_source_count(self) -> int:
        return len({a.source_name for a in self.articles})


@dataclass
class DonationEntry:
    """A fully-extracted, scored donation event ready to publish."""
    entry_id: str
    donor: str
    recipient: str
    amount_text: Optional[str]        # e.g. "$5 million" — verbatim short figure clause, or None
    is_in_kind: bool
    in_kind_description: Optional[str]
    country_scope: str                # a name from countries.txt, or "Unspecified / global"
    status: EventStatus
    summary: str                      # paraphrased, non-verbatim summary
    figure_quote: Optional[str]       # short (<15 word) verbatim clause containing the figure, if any
    assumptions: list[str]            # anything the model had to infer or flag as unclear
    confidence_score: int             # 0-10
    confidence_breakdown: dict
    source_urls: list[str]
    source_names: list[str]
    date_found: str                   # ISO date this entry was first published on the site
    is_duplicate_of: Optional[str] = None   # entry_id of an earlier match, if any

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
