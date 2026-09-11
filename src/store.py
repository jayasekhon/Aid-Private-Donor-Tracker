"""A simple JSON-file-backed store of past donation events.

This is the pipeline's memory across daily runs. Before a newly-extracted
entry is published, we check it against this store to decide:
  - is this the exact same event we already reported (skip / do not republish)?
  - is this a RENEWAL of a known partnership (publish, but tag as renewal)?
  - is this genuinely new (publish as new)?

Deliberately file-based rather than a database: this project has modest
volume (a handful to a few dozen entries a day) and a JSON file that lives
in the git repo is transparent, diffable, and trivial for a non-technical
person to inspect if they're curious ("what has this system found before?")
without needing any tooling.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rapidfuzz import fuzz

from .models import DonationEntry

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "seen_events.json"

NAME_MATCH_THRESHOLD = 80        # rapidfuzz token_set_ratio, for donor/recipient names
AMOUNT_MATCH_TOLERANCE = 0.15    # 15% — treats "$5.1M" and "$5M" reporting drift as the same event

# When neither the candidate nor a matching prior entry has a parseable
# dollar figure (in-kind gifts, or vaguely-worded "supports research"-style
# partnership announcements), there's no numeric signal to confirm two
# write-ups describe the EXACT same donation — which is why that case
# below defaults to "likely_renewal" rather than silently skipping, so a
# genuinely new no-amount donation to a repeat donor/recipient pair isn't
# suppressed. But a match found within a few days is overwhelmingly more
# likely the SAME real-world event surfacing via a second article that
# clustering didn't merge (see clustering.py's TITLE_SIMILARITY_THRESHOLD)
# than a coincidentally-timed second donation. A real run confirmed this:
# two separate Sloan Foundation -> Stony Brook University clusters, and two
# separate Sumitomo Foundation -> Japan research clusters, all four with no
# stated amount, were published as four distinct entries in the same day's
# edition instead of two, because the "always renewal" branch never gets
# to compare a candidate against the OTHER copy of itself found minutes
# earlier in the same run.
SAME_EVENT_WINDOW_DAYS = 3


class EventStore:
    def __init__(self, path: Path = DEFAULT_STORE_PATH):
        self.path = path
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _parse_found_date(date_found: str | None) -> datetime | None:
        if not date_found:
            return None
        try:
            return datetime.fromisoformat(date_found)
        except ValueError:
            return None

    def _recent_entries(self, lookback_days: int) -> list[dict]:
        """Returns entries within lookback_days, MOST RECENT FIRST. Order
        matters: find_possible_match() below returns on the first name
        match it finds, so checking recent entries first means a same-run
        (or same-week) duplicate is compared against its own freshest
        copy — not against some much older entry that happens to appear
        earlier in the store's on-disk (oldest-first, append-only) order.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        out = []
        for e in self.entries:
            found = self._parse_found_date(e.get("date_found"))
            if found is not None and found >= cutoff:
                out.append((found, e))
        out.sort(key=lambda pair: pair[0], reverse=True)
        return [e for _, e in out]

    @staticmethod
    def _parse_amount_number(amount_text: str | None) -> float | None:
        """Very forgiving extraction of a numeric magnitude from strings like
        '$5 million', '$2.3M', '€500,000'. Returns None if nothing parseable
        (e.g. in-kind entries with no dollar figure) — callers must handle that.
        """
        if not amount_text:
            return None
        import re
        text = amount_text.lower().replace(",", "")
        match = re.search(r"([\d.]+)\s*(million|m|billion|bn|thousand|k)?", text)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        multiplier = {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9,
                      "thousand": 1e3, "k": 1e3}.get(match.group(2) or "", 1)
        return value * multiplier

    def find_possible_match(self, candidate: DonationEntry, lookback_days: int) -> tuple[dict | None, str]:
        """Returns (matching_prior_entry_or_None, reason).

        reason is one of: "exact_duplicate", "likely_renewal", "no_match"
        """
        candidate_amount = self._parse_amount_number(candidate.amount_text)
        for prior in self._recent_entries(lookback_days):
            donor_score = fuzz.token_set_ratio(candidate.donor.lower(), prior["donor"].lower())
            recipient_score = fuzz.token_set_ratio(candidate.recipient.lower(), prior["recipient"].lower())
            if donor_score < NAME_MATCH_THRESHOLD or recipient_score < NAME_MATCH_THRESHOLD:
                continue

            prior_amount = self._parse_amount_number(prior.get("amount_text"))
            if candidate_amount is not None and prior_amount is not None:
                if abs(candidate_amount - prior_amount) / max(prior_amount, 1) <= AMOUNT_MATCH_TOLERANCE:
                    return prior, "exact_duplicate"
                else:
                    # Same donor+recipient, different amount => likely a renewal
                    # or a new, separate commitment. We can't be fully sure from
                    # names+amount alone, so we flag it rather than assume.
                    return prior, "likely_renewal"
            elif candidate_amount is None and prior_amount is None:
                # Both in-kind / no figure — same donor+recipient pair
                # recurring. Treat a RECENT such match as the same event
                # reported twice (see SAME_EVENT_WINDOW_DAYS above); an
                # older one is genuinely ambiguous, so stays "likely_renewal".
                prior_found = self._parse_found_date(prior.get("date_found"))
                if prior_found is not None and \
                        datetime.now(timezone.utc) - prior_found <= timedelta(days=SAME_EVENT_WINDOW_DAYS):
                    return prior, "exact_duplicate"
                return prior, "likely_renewal"

        return None, "no_match"

    def add(self, entry: DonationEntry) -> None:
        self.entries.append(entry.to_dict())
