"""Fetches candidate donation-story leads from GDELT's Global Knowledge
Graph (GKG) 2.1 bulk files — GDELT's realtime, worldwide (65-language,
machine-translated) news monitoring feed, published as a new file every 15
minutes with no per-request rate limit. This replaces an earlier attempt at
using GDELT's DOC 2.0 search API, which proved unusable from GitHub
Actions' shared runner IPs (persistent 429s even with compliant pacing —
see git history for that investigation).

Schema below is confirmed against a real sample GKG file and GDELT's own
GKG 2.1 codebook, not assumed — 27 tab-delimited columns per row, one row
per article. We only use a handful of them:
  - column 2  (V2SOURCECOLLECTIONIDENTIFIER): filtered to "1" (WEB) only.
    GKG also carries a small number of TV-transcript-derived rows (value
    "6", Internet Archive TV News Archive closed captioning) that we don't
    want — this project tracks web/press coverage, not TV.
  - column 4  (V2DOCUMENTIDENTIFIER): the article URL.
  - column 3  (V2SOURCECOMMONNAME): the source domain.
  - column 1  (V2.1DATE): publish date, "YYYYMMDDHHMMSS".
  - column 13 (V1ORGANIZATIONS): semicolon-list of organizations
    mentioned — this is how GDELT's global firehose gets narrowed down to
    "mentions one of our monitored recipients" without a live search query.
  - column 26 (V2EXTRASXML): contains a <PAGE_TITLE> tag (confirmed present
    in 1560/1560 rows of a real sample file). GKG has no dedicated headline
    column — this is the only source of a usable article title.
  - columns 24/22 (V2.1AMOUNTS / V2.1QUOTATIONS): folded into a synthetic
    "summary", since GKG doesn't provide article body text at all — this
    gives the downstream trigger-phrase filter and the AI extraction step
    more to work with than a bare title.

Deliberately NOT using GKG's theme taxonomy as a donation-relevance signal:
checked both a real sample file and the full ~59,000-theme vocabulary (a
lookup table with real usage counts, not assumed) for donation/
philanthropy/CSR-flavoured themes and found essentially nothing usable — a
handful of specific named aid orgs and a "philanthropist" occupation tag,
nothing for "this article is about an act of giving". Donation-language
filtering for GKG-derived articles instead reuses the existing
trigger-phrase filter completely unchanged
(clustering.filter_by_trigger_phrase already checks title+summary
case-insensitively) — exactly like it already does for Google News and PR
wire articles, so no GKG-specific filtering code is needed there.
"""
from __future__ import annotations

import html
import io
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .config_loader import Recipient
from .models import RawArticle, SourceTier
from .sources import FetchFailure, USER_AGENT

logger = logging.getLogger(__name__)

GDELT_GKG_BASE = "http://data.gdeltproject.org/gdeltv2"
GKG_FILE_TIMEOUT_SECONDS = 60  # files run several MB; longer than the 15s used for small RSS feeds
FILE_INTERVAL_MINUTES = 15

# First run only (no prior state) — how far back to bootstrap. 24h keeps a
# first run to ~96 files rather than trying to catch up on weeks of
# history; after the first run, fetching always picks up exactly where the
# last run left off, so this constant stops mattering.
BOOTSTRAP_LOOKBACK_HOURS = 24

# Safety cap on files fetched in a single run — e.g. if the scheduled job
# doesn't run for several days, this stops a catch-up run from trying to
# download hundreds of files at once. State only advances to whatever was
# actually processed, so a capped run resumes the rest on the next run
# rather than silently skipping the gap.
MAX_FILES_PER_RUN = 200

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "gdelt_gkg_state.json"

PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)


@dataclass
class GkgFetchStats:
    files_pending: int
    files_processed: int
    candidates_found: int


def _load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _round_down_to_interval(dt: datetime) -> datetime:
    minute = (dt.minute // FILE_INTERVAL_MINUTES) * FILE_INTERVAL_MINUTES
    return dt.replace(minute=minute, second=0, microsecond=0)


def _pending_file_timestamps(last_processed: str | None, now: datetime) -> list[datetime]:
    """GDELT publishes a file every 15 minutes at a predictable timestamp —
    confirmed by inspecting real filenames, not assumed. The most recent
    COMPLETED window is used as the upper bound (not the current
    in-progress one, which may not be published yet).
    """
    latest_available = _round_down_to_interval(now) - timedelta(minutes=FILE_INTERVAL_MINUTES)
    if last_processed is None:
        start = latest_available - timedelta(hours=BOOTSTRAP_LOOKBACK_HOURS)
    else:
        start = (datetime.strptime(last_processed, "%Y%m%d%H%M%S")
                 .replace(tzinfo=timezone.utc) + timedelta(minutes=FILE_INTERVAL_MINUTES))

    timestamps = []
    t = start
    while t <= latest_available and len(timestamps) < MAX_FILES_PER_RUN:
        timestamps.append(t)
        t += timedelta(minutes=FILE_INTERVAL_MINUTES)
    return timestamps


def _fetch_gkg_file(ts: datetime, max_retries: int = 3) -> tuple[bytes | None, FetchFailure | None]:
    ts_str = ts.strftime("%Y%m%d%H%M%S")
    url = f"{GDELT_GKG_BASE}/{ts_str}.gkg.csv.zip"
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=GKG_FILE_TIMEOUT_SECONDS)
            if resp.status_code == 404:
                # GDELT occasionally skips a 15-minute window entirely —
                # not a failure, just nothing published for it.
                logger.info("No GDELT GKG file published for window %s (404) — skipping.", ts_str)
                return None, None
            resp.raise_for_status()
            return resp.content, None
        except requests.RequestException as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("GDELT GKG file fetch failed for %s (attempt %d/%d): %s. Retrying in %ds.",
                            ts_str, attempt, max_retries, e, wait)
            time.sleep(wait)
    logger.warning("Giving up on GDELT GKG file %s after %d failed attempts.", ts_str, max_retries)
    return None, FetchFailure(f"GDELT GKG: {ts_str}", url, str(last_error))


def _extract_page_title(xml_extras: str) -> str | None:
    m = PAGE_TITLE_RE.search(xml_extras)
    if not m:
        return None
    title = html.unescape(m.group(1)).strip()
    return title or None


def _parse_amounts(amounts_field: str) -> list[str]:
    out = []
    for block in amounts_field.split(";"):
        block = block.strip()
        if not block:
            continue
        parts = block.split(",")
        if len(parts) < 2:
            continue
        amount, obj = parts[0].strip(), parts[1].strip()
        out.append(f"{amount} ({obj})" if obj else amount)
    return out


# Cap on how many of an article's quotes to fold into the synthetic
# summary — enough to give the trigger-phrase filter a real shot at a
# donation-relevant quote that isn't literally the first one GDELT
# extracted, without letting a quote-heavy article balloon the summary.
MAX_QUOTES_IN_SUMMARY = 5


def _parse_quotes(quotes_field: str) -> list[str]:
    """Returns up to MAX_QUOTES_IN_SUMMARY quotes from the article, in the
    order GDELT extracted them. Previously only the FIRST quote was kept
    (_parse_first_quote) — a real recall gap, since the trigger-phrase
    filter only ever sees whatever text ends up in the synthetic summary,
    and a donation-relevant quote (e.g. an org spokesperson thanking a
    donor) is often not literally the first quote in the article; an
    earlier, unrelated quote (e.g. a local official describing the
    disaster itself) would previously have crowded it out entirely.
    """
    quotes = []
    for block in quotes_field.split("#"):
        parts = block.split("|")
        if len(parts) >= 4 and parts[3].strip():
            quotes.append(parts[3].strip())
        if len(quotes) >= MAX_QUOTES_IN_SUMMARY:
            break
    return quotes


def _build_summary(orgs: list[str], amounts_field: str, quotes_field: str) -> str:
    """GKG gives us no article body text, so this stitches together a
    short substitute from what it DOES extract — organization mentions,
    numeric amounts, and up to a handful of quoted statements — giving
    the trigger-phrase filter and the AI extraction step more to work
    with than a bare title.
    """
    parts = []
    if orgs:
        parts.append("Organizations mentioned: " + ", ".join(orgs) + ".")
    amounts = _parse_amounts(amounts_field)
    if amounts:
        parts.append("Amounts mentioned: " + "; ".join(amounts) + ".")
    quotes = _parse_quotes(quotes_field)
    if quotes:
        parts.append("Quotes: " + " | ".join(f'"{q}"' for q in quotes))
    return " ".join(parts)


def _build_name_lookup(recipients: list[Recipient]) -> dict[str, str]:
    """Maps every recipient name/alias (lowercased) to its canonical
    recipient name, for exact matching against GKG's organization mentions.
    Deliberately exact match, not substring: an earlier substring-based
    check against real sample data produced false positives like a
    3-letter alias ("IRC") matching inside unrelated words ("circuit").
    """
    lookup = {}
    for r in recipients:
        for n in r.all_names:
            lookup[n.lower().strip()] = r.name
    return lookup


def _parse_gkg_row(fields: list[str], name_lookup: dict[str, str]) -> RawArticle | None:
    if len(fields) < 27:
        return None
    if fields[2] != "1":  # V2SOURCECOLLECTIONIDENTIFIER — web articles only
        return None

    orgs_raw = [o.strip() for o in fields[13].split(";") if o.strip()]
    matched_recipient = None
    for org in orgs_raw:
        recipient = name_lookup.get(org.lower())
        if recipient:
            matched_recipient = recipient
            break
    if matched_recipient is None:
        return None

    title = _extract_page_title(fields[26])
    if not title:
        return None

    try:
        published = datetime.strptime(fields[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        published = None

    return RawArticle(
        title=title,
        url=fields[4].strip(),
        published=published,
        source_name=fields[3].strip() or "Unknown (via GDELT GKG)",
        source_tier=SourceTier.GENERAL_NEWS,
        summary=_build_summary(orgs_raw, fields[24], fields[22]),
        matched_recipient=matched_recipient,
        fetch_source="GDELT GKG",
    )


def _parse_gkg_zip(zip_bytes: bytes, name_lookup: dict[str, str]) -> list[RawArticle]:
    articles = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as raw_f:
            for raw_line in io.TextIOWrapper(raw_f, encoding="utf-8", errors="replace"):
                fields = raw_line.rstrip("\n").split("\t")
                article = _parse_gkg_row(fields, name_lookup)
                if article:
                    articles.append(article)
    return articles


def _fetch_and_parse_timestamps(
    timestamps: list[datetime], name_lookup: dict[str, str]
) -> tuple[list[RawArticle], list[FetchFailure], list[str]]:
    """Shared fetch/parse loop used by both the state-tracked incremental
    fetch and the state-independent backfill fetch below. Stops at the
    first real fetch/parse failure rather than skipping past it (a 404 —
    no file published for a window — is not treated as a failure, just
    continues). Returns (articles, failures, timestamps actually
    completed, in "%Y%m%d%H%M%S" string form) — callers decide what, if
    anything, to do with that completed list (e.g. persisting state).
    """
    all_articles: list[RawArticle] = []
    failures: list[FetchFailure] = []
    completed: list[str] = []

    for ts in timestamps:
        content, failure = _fetch_gkg_file(ts)
        if failure:
            failures.append(failure)
            break

        ts_str = ts.strftime("%Y%m%d%H%M%S")
        if content is None:
            completed.append(ts_str)
            continue

        try:
            articles = _parse_gkg_zip(content, name_lookup)
        except (zipfile.BadZipFile, UnicodeDecodeError) as e:
            logger.warning("Failed to parse GDELT GKG file for %s: %s", ts_str, e)
            failures.append(FetchFailure(f"GDELT GKG: {ts_str}", "", str(e)))
            break

        all_articles.extend(articles)
        completed.append(ts_str)

    return all_articles, failures, completed


def fetch_gdelt_gkg_articles(recipients: list[Recipient]) -> tuple[list[RawArticle], list[FetchFailure], GkgFetchStats]:
    """Fetches and parses every new GDELT GKG 15-minute file since the
    last successful run (tracked in data/gdelt_gkg_state.json), returning
    candidate articles that mention a monitored recipient.

    State is only advanced up to the last window actually completed, so a
    failure partway through leaves that window (and everything after it)
    to be picked up on the next run instead of silently creating a
    permanent gap.
    """
    state = _load_state()
    now = datetime.now(timezone.utc)
    timestamps = _pending_file_timestamps(state.get("last_processed_timestamp"), now)
    name_lookup = _build_name_lookup(recipients)

    all_articles, failures, completed = _fetch_and_parse_timestamps(timestamps, name_lookup)
    if completed:
        state["last_processed_timestamp"] = completed[-1]
        _save_state(state)

    stats = GkgFetchStats(files_pending=len(timestamps), files_processed=len(completed),
                           candidates_found=len(all_articles))
    logger.info("GDELT GKG: processed %d/%d file(s), found %d candidate article(s) mentioning a "
                "monitored recipient (%d fetch failures).",
                stats.files_processed, stats.files_pending, stats.candidates_found, len(failures))
    return all_articles, failures, stats


def fetch_gdelt_gkg_articles_for_range(
    recipients: list[Recipient], start: datetime, end: datetime
) -> tuple[list[RawArticle], list[FetchFailure], GkgFetchStats]:
    """Backfill/test variant of fetch_gdelt_gkg_articles: fetches every
    15-minute GKG file between start and end (inclusive, both rounded down
    to the nearest 15-minute boundary), for an arbitrary past date range —
    e.g. to check whether a specific known real-world event would have
    been caught. Completely independent of data/gdelt_gkg_state.json: does
    not read or write it, so running this can never disturb the real
    daily pipeline's incremental tracking. Still capped at
    MAX_FILES_PER_RUN as the same safety net against an accidentally huge
    range.
    """
    start = _round_down_to_interval(start)
    end = _round_down_to_interval(end)
    timestamps = []
    t = start
    while t <= end and len(timestamps) < MAX_FILES_PER_RUN:
        timestamps.append(t)
        t += timedelta(minutes=FILE_INTERVAL_MINUTES)

    name_lookup = _build_name_lookup(recipients)
    all_articles, failures, completed = _fetch_and_parse_timestamps(timestamps, name_lookup)

    stats = GkgFetchStats(files_pending=len(timestamps), files_processed=len(completed),
                           candidates_found=len(all_articles))
    logger.info("GDELT GKG backfill (%s to %s): processed %d/%d file(s), found %d candidate "
                "article(s) mentioning a monitored recipient (%d fetch failures).",
                start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M"),
                stats.files_processed, stats.files_pending, stats.candidates_found, len(failures))
    return all_articles, failures, stats
