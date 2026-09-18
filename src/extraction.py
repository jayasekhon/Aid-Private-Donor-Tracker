"""Reads an ArticleCluster and asks an LLM to pull out structured donation
facts, OR runs in mock mode (no API key needed) for local testing/demos.

COPYRIGHT NOTE — read before changing the prompt:
The model is explicitly instructed to return a PARAPHRASED summary plus,
separately, a short verbatim clause (under 15 words) that contains only
the figure/fact itself, never a reproduced sentence or paragraph from the
source article. This mirrors how a human researcher would take notes —
write the fact in your own words, quote only the number precisely. Do not
relax this instruction; it exists to avoid reproducing substantial
portions of copyrighted news text.

CONFIDENCE SCORING — the rubric lives in config/settings.yaml, not
hardcoded here, so a non-technical editor can rebalance the point values
without touching this file. This module only implements the arithmetic.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

from rapidfuzz import fuzz

from .config_loader import Recipient
from .models import ArticleCluster, DonationEntry, EventStatus, SourceTier, now_iso, today_str

logger = logging.getLogger(__name__)

# Phrases that mean "no specific company was actually named" — a backstop
# behind the extraction prompt's own instruction to set is_relevant=false
# in this case, for when the model doesn't comply. This tracker exists to
# identify WHICH company gave; an entry that can't name one isn't discovery,
# it's just noise confirming "some company" exists.
GENERIC_DONOR_MARKERS = (
    "unspecified", "not specified", "unnamed", "unknown", "unidentified",
    "various compan", "several compan", "multiple compan", "corporate partners",
    "a company", "companies involved", "not named", "n/a", "undisclosed",
)

# The prompt tells the model to set is_relevant=false when it can't name a
# specific donor, but it sometimes complies with the letter of "give a
# name" rather than the spirit -- extracting a bare industry descriptor as
# if it were the company's own name (a real case: donor="Housebuilder" for
# a story that never actually said which housebuilding company gave,
# with the model's own "assumptions" field admitting as much). Checked as
# an (article-stripped) EXACT match, never a substring -- a substring
# check would wrongly reject real single-word/short company names that
# happen to contain one of these words, e.g. "Bank of America".
GENERIC_DONOR_INDUSTRY_NOUNS = {
    "housebuilder", "house builder", "retailer", "developer", "manufacturer",
    "automaker", "car maker", "airline", "bank", "firm", "business",
    "company", "corporation", "organisation", "organization", "supermarket",
    "builder", "construction company", "property developer",
}


def is_generic_donor(donor: str) -> bool:
    if not donor or not donor.strip():
        return True
    d = donor.strip().lower()
    if any(marker in d for marker in GENERIC_DONOR_MARKERS):
        return True
    for article in ("a ", "an ", "the "):
        if d.startswith(article):
            d = d[len(article):]
            break
    return d in GENERIC_DONOR_INDUSTRY_NOUNS


# Aliases that double as ordinary English words are unsafe for the loose
# substring/fuzzy matching below: "CARE" (CARE International's alias)
# matches as a literal substring of "Charlotte Animal Care & Control",
# which is exactly what happened in a real run — that entry (a local
# animal shelter, unrelated to CARE International) was wrongly tagged
# "INGO" and given an undeserved monitored-recipient confidence bonus.
# "WHO" carries the same risk (it's a common standalone word in English
# text, e.g. "Anyone Who Cares Foundation"). Same class of bug as
# query_builder.py's Google News "WHO" fix, just hitting a different
# matcher. Only the loose checks (substring, fuzzy ratio) are skipped for
# these aliases — an EXACT match (candidate == name_l) is still trusted,
# so a genuine extraction of "WHO" or "CARE" as the recipient's full name
# still correctly resolves to World Health Organization / CARE
# International.
AMBIGUOUS_CURATED_ALIASES = {"who", "care"}


def match_curated_recipient(recipient: str, recipients: list[Recipient]) -> Recipient | None:
    """Returns the curated recipients.txt entry the extracted recipient
    corresponds to (including alias matches), or None if it names a real
    organization that just isn't on our curated 50-recipient watchlist.
    Being off that list is no longer a rejection reason on its own (see
    is_generic_recipient for the actual publish/reject gate) — this is
    used only to pull the authoritative org_type for curated recipients,
    which is more reliable than trusting the AI's own guess.
    """
    if not recipient or not recipient.strip():
        return None
    candidate = recipient.strip().lower()
    for r in recipients:
        for name in r.all_names:
            name_l = name.strip().lower()
            if not name_l:
                continue
            if candidate == name_l:
                return r
            if name_l in AMBIGUOUS_CURATED_ALIASES:
                continue
            if candidate in name_l or name_l in candidate:
                return r
            if fuzz.token_set_ratio(candidate, name_l) >= 88:
                return r
    return None


# Phrases that mean "no specific organization was actually named" — the
# recipient-side counterpart to GENERIC_DONOR_MARKERS. Per an explicit
# scope-broadening decision, a recipient no longer has to be one of the
# curated 50 to be published (see match_curated_recipient) — a donation to
# any small, local, or otherwise off-list nonprofit is a real finding as
# long as that nonprofit is actually NAMED. What's still rejected is a
# recipient that was never named at all, since "a company donated to a
# local charity" identifies WHO gave but not WHERE it went, and that's a
# materially weaker finding than either half being anonymous alone.
GENERIC_RECIPIENT_MARKERS = (
    "unspecified", "not specified", "unnamed", "unknown", "unidentified",
    "various nonprofit", "several nonprofit", "multiple nonprofit",
    "various ngo", "several ngo", "multiple ngo",
    "various charit", "several charit", "multiple charit",
    "local charity", "local charities", "local nonprofit", "local ngo",
    "a nonprofit", "a non-profit", "a charity", "an ngo", "an ingo",
    "aid organizations", "humanitarian organizations", "not named", "n/a", "undisclosed",
    # Plural/collective phrasings missed above — real cases: "Palestinian
    # aid groups" (Macklemore's $1M pledge) and "Nepal flood relief"
    # neither matched anything here, despite being just as generic as "an
    # aid organization". Deliberately 2-word phrases (not a bare "relief"
    # or "aid") since several REAL curated recipients have one of those
    # words as part of their actual proper name (Direct Relief, Islamic
    # Relief Worldwide, Catholic Relief Services) — these compounds don't
    # match any of them.
    "aid group", "relief group", "flood relief", "disaster relief",
    "humanitarian relief", "relief effort", "aid effort", "relief agenc",
    "aid agenc", "charitable organization",
    # Another real case: "Lagos railway security agencies" -- a vague
    # plural collective just like "aid groups" above, no specific agency
    # ever named.
    "security agenc",
)


def is_generic_recipient(recipient: str) -> bool:
    if not recipient or not recipient.strip():
        return True
    r = recipient.strip().lower()
    return any(marker in r for marker in GENERIC_RECIPIENT_MARKERS)


# This tracker exists to track PRIVATE-SECTOR donations to nonprofits —
# neither side of that should ever be a government body. A real run
# published two entries where BOTH sides were government: donor "Lagos
# State Government" -> recipient "Nigerian Railway Corporation" (a state-
# owned rail operator, not a nonprofit at all), and donor "Lagos State
# Security Trust Fund" -> recipient "Lagos railway security agencies" —
# neither a private company giving, nor a nonprofit receiving; both sides
# of both entries were the state. The extraction prompt already says to
# reject a government DONOR, but had no equivalent instruction for a
# government-owned/-run RECIPIENT, and evidently isn't reliably complied
# with either way — hence a code-level backstop, checked against BOTH
# fields the same way (one function, two call sites), same as
# is_generic_donor/is_generic_recipient's split for a different reason.
# Deliberately compound phrases, not bare words like "state" (which would
# wrongly reject real company names, e.g. "State Farm") — every marker
# here is checked against real curated recipients.txt entries too, none
# of which match any of them.
GOVERNMENT_ENTITY_MARKERS = (
    "government", "ministry of", "ministry for", "department of",
    "city council", "county council", "municipal council", "municipality of",
    "governor's office", "governor of", "mayor's office", "office of the mayor",
    "state house", "national assembly", "house of representatives",
    "state security trust fund", "security trust fund",
)


def is_government_entity(name: str) -> bool:
    if not name or not name.strip():
        return False
    n = name.strip().lower()
    return any(marker in n for marker in GOVERNMENT_ENTITY_MARKERS)


# GENERIC_DONOR_MARKERS/GENERIC_RECIPIENT_MARKERS above are a fixed
# phrase list, which will always be one step behind the open-ended ways
# a source can describe an unnamed entity — a real run published "A
# Plymouth-based developer has made a repeated donation to a local
# children's group" (donor "Plymouth developer", recipient "children's
# group") because neither string matched any marker, even though both
# are exactly as unusable as "a company" or "a local charity" ("Plymouth"
# and "children's" make them READ more specific without actually naming
# anyone). Model prompt strengthened for this case too (see
# EXTRACTION_PROMPT_TEMPLATE), but a code backstop shouldn't rely on
# prompt compliance alone -- and here there's a much more general signal
# than yet another marker to add: the model's OWN "assumptions" field
# already tends to admit the gap in its own words ("The specific name of
# the children's group is not provided", "...housebuilding company is
# not provided in the source text" -- both real, from entries that
# should have been rejected). This checks for that admission directly,
# rather than trying to keep pace with every new way of writing "no name
# given" as a hardcoded phrase.
NAME_MISSING_ADMISSION_PATTERN = re.compile(
    r"name of the .*\b(?:is not|isn't|wasn't|was not)\b.*\b(?:provided|specified|given|named|disclosed|stated)\b"
)
# Which side (donor vs. recipient) an admission is about is inferred from
# a word in the admission itself -- e.g. "name of the HOUSEBUILDING
# COMPANY is not provided" vs "name of the CHILDREN'S GROUP is not
# provided". Deliberately two separate small sets rather than one shared
# one: "company"/"business" is unambiguously donor-side, "charity"/
# "group"/"organization" unambiguously recipient-side, and keeping them
# apart avoids an admission about one field ever suppressing the other.
DONOR_NAME_CONTEXT_WORDS = {
    "company", "companies", "donor", "business", "firm", "corporation",
    "sponsor", "developer", "builder", "retailer", "manufacturer", "bank",
}
RECIPIENT_NAME_CONTEXT_WORDS = {
    "group", "organization", "organisation", "charity", "charities",
    "nonprofit", "non-profit", "ngo", "ingo", "association", "recipient",
    "shelter", "hospital", "school", "foundation", "fund",
}


def assumptions_admit_missing_name(assumptions: list[str], context_words: set[str]) -> bool:
    for a in assumptions or []:
        a_l = a.lower()
        if NAME_MISSING_ADMISSION_PATTERN.search(a_l) and any(w in a_l for w in context_words):
            return True
    return False


UNSPECIFIED_SCOPE = "unspecified / global"


def is_out_of_scope_country(country_scope: str, countries: list[str]) -> bool:
    """True if country_scope names a SPECIFIC place that isn't on the
    curated GHO+Nepal list (config/countries.txt) — a real, named
    location, just not one this tracker is scoped to (e.g. "South Korea",
    "Singapore", or a US city/state attached to domestic university/
    hospital/community philanthropy). "Unspecified / global" is NOT out
    of scope on its own — that's governed separately by
    publishing.include_unspecified_scope; this only catches a named-but-
    wrong place.

    The extraction prompt asks the AI to extract country_scope as free
    text from the source article, entirely independent of countries.txt
    — nothing upstream constrains it to the curated list, which is
    exactly the gap that let country_scope values like "South Korea" or
    "Singapore" reach publication. This is the backstop that closes it,
    matching the is_generic_donor/is_generic_recipient pattern.

    Matching is substring-based (e.g. "Syria" is a real substring of the
    curated "Syrian Arab Republic", "Congo" of "Democratic Republic of
    the Congo"), not exact — the AI tends to use the common/casual name
    for a place, not the formal one in countries.txt. This won't catch
    every real phrasing (a bare acronym like "DRC" won't match "Democratic
    Republic of the Congo" this way) — if that turns out to cause real
    false rejections in practice (visible via the "Rejecting cluster...
    out of scope country" log line), add the acronym as its own line in
    countries.txt rather than trying to guess every variant up front.
    """
    if not country_scope or country_scope.strip().lower() == UNSPECIFIED_SCOPE:
        return False
    candidate = country_scope.strip().lower()
    for c in countries:
        c_l = c.strip().lower()
        if candidate == c_l or candidate in c_l or c_l in candidate:
            return False
    return True


# Display labels for the type tag shown on every published entry. Curated
# recipients (recipients.txt) use their authoritative org_type; anything
# off-list falls back to the AI's own "recipient_type" guess from the
# extraction prompt, normalized to one of these four — never trusted
# blindly beyond that, since the AI can mis-classify an org it doesn't
# recognize.
RECIPIENT_TYPE_LABELS = {"UN": "UN Agency", "INGO": "INGO", "NGO": "NGO"}
OTHER_NONPROFIT_LABEL = "Other Nonprofit"
# Keyed lowercase since the prompt asks the model for lowercase-style
# values ("UN agency", "Other nonprofit") but the model's exact casing
# can't be relied on — matched case-insensitively, always returned as
# one of the canonical display labels above.
_AI_LABEL_BY_LOWER = {label.lower(): label for label in RECIPIENT_TYPE_LABELS.values()} | {
    OTHER_NONPROFIT_LABEL.lower(): OTHER_NONPROFIT_LABEL,
}


def classify_recipient_type(recipient: str, recipients: list[Recipient], ai_type_guess: str | None) -> str:
    """The display label for the entry's type tag. A curated-list match
    always wins (authoritative data beats a guess); otherwise falls back
    to the AI's own classification if it returned one of the four expected
    labels, or "Other Nonprofit" if it didn't (missing, malformed, or an
    unrecognized value) rather than publishing an untagged/miscategorized entry.
    """
    matched = match_curated_recipient(recipient, recipients)
    if matched is not None:
        return RECIPIENT_TYPE_LABELS[matched.org_type]
    guess = (ai_type_guess or "").strip().lower()
    return _AI_LABEL_BY_LOWER.get(guess, OTHER_NONPROFIT_LABEL)

# Free-tier Gemini 3.x Flash allows ~10 requests/minute (down from the 15
# RPM the older 2.0/2.5 Flash models had). Spacing calls 6.5s apart keeps us
# at ~9.2/minute, comfortably under the ceiling with margin for clock drift.
MIN_SECONDS_BETWEEN_AI_CALLS = 6.5
_last_ai_call_time: float = 0.0


def _pace_ai_calls() -> None:
    global _last_ai_call_time
    elapsed = time.monotonic() - _last_ai_call_time
    if elapsed < MIN_SECONDS_BETWEEN_AI_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_AI_CALLS - elapsed)
    _last_ai_call_time = time.monotonic()


class FatalExtractionError(Exception):
    """An error that will not resolve by retrying — e.g. the configured
    model no longer exists, or the API key is invalid. Raised to stop the
    whole run immediately (with a clear, actionable message) rather than
    burning through every remaining cluster's retry budget on an error
    that was never going to succeed. This is what caused the ~8 minute
    stall on a bad model name: 32 clusters x 3 retries x (2s+4s+8s) backoff
    each, for an error that would never resolve.
    """


# Substrings that mean "this will never succeed, no matter how many times
# we retry" — as opposed to transient issues (timeouts, 429 rate limits,
# 5xx server errors) where retrying is the right move.
NON_RETRYABLE_ERROR_MARKERS = (
    "is no longer available",
    "404",
    "not found",
    "api key not valid",
    "permission_denied",
    "invalid_argument",
)

# Quota-exhausted errors are a special case of non-retryable: retrying
# *within the same run* will never help (the daily cap has been hit and
# resets at midnight Pacific time), but the message a maintainer needs is
# different from "the model name is wrong" — so it gets its own markers
# and its own message, even though the control flow (stop immediately,
# don't burn retries) is identical.
QUOTA_EXHAUSTED_MARKERS = (
    "quota",
    "resource_exhausted",
    "generaterequestsperdayperprojectpermodel",
)

EXTRACTION_PROMPT_TEMPLATE = """You are a careful research assistant helping track private-sector \
(corporate) donations to UN agencies, INGOs, and NGOs. You will be shown one or more news \
items that a keyword filter GROUPED TOGETHER because they looked like they describe the same \
real-world donation/partnership event. That grouping is done by a cheap keyword/title-similarity \
check, not by you, and it is sometimes wrong: the source text(s) below can actually describe TWO \
OR MORE genuinely different donations that happen to share similar wording (most often one \
dominant, widely-syndicated story plus a smaller, distinct donation mentioned only in passing \
within one of the same articles). A real case this caused: a cluster of ~20 articles about a \
celebrity's $1M pledge "to Palestinian aid groups" (no specific org named) included one article \
that also reported a completely different, unrelated $10,000 donation from a different company to \
a specifically named recipient — extracting only the dominant story would silently lose that \
second, more specific, equally real finding. So: read ALL the source text carefully, and if you \
can identify more than one genuinely distinct donor+recipient combination, extract EACH as its \
own separate event below rather than merging them into one or defaulting to just the dominant one. \
The overwhelming majority of the time there is genuinely only one event — extract one.

Your job, for EACH distinct event you identify: decide if this is genuinely a private company (or corporate foundation) donating to, \
partnering with, or otherwise financially/materially supporting a nonprofit organization — a UN \
agency, INGO, NGO, or any other named charity/nonprofit, of any size. If it is NOT (e.g. the donor \
is a government body — a national/state/city government, a ministry, a government agency or \
fund — rather than a private company; OR the recipient is a government body or a state-owned/\
-run organization rather than a nonprofit, e.g. a state railway corporation, a government \
security agency, or a public university/hospital system run BY the state — even if press \
coverage calls it a "donation"; an unrelated story that matched keywords by coincidence; or pure \
speculation with no confirmed commitment), it is not a real event — do not include an entry for \
it. A real case that should have been excluded on both counts: "Lagos State Government" donating \
patrol vehicles to the "Nigerian Railway Corporation" — a government body giving to a state-\
owned corporation, nowhere close to a private company supporting a nonprofit. This tracker \
exists to track PRIVATE-SECTOR giving to NONPROFITS specifically; a transfer between two parts \
of government isn't that, no matter how it's phrased in the source text. This \
tracker exists to identify WHICH company gave — if the source text never names a specific company \
or corporate foundation (only vague language like "corporate partners", "several companies", or \
"a donor"), that is also not a real event: do not include an entry, and never invent a placeholder \
donor. This includes a description that SOUNDS specific but isn't actually a name — "a Plymouth \
developer" or "the housebuilder" name the donor's location or industry, not the company itself, \
and are exactly as unusable as "a company" or "corporate partners". Never make an exception for \
the donor, no matter how newsworthy the story is otherwise — identifying WHICH company gave is \
this tracker's entire purpose, so an unnamed donor is never a usable finding.

The recipient does NOT need to be a major/well-known organization — a donation to a small local \
nonprofit is just as relevant a finding as one to a large UN agency — and it should normally be a \
SPECIFIC, NAMED organization. If the source text only says something vague like "a local \
charity", "several nonprofits", or "the children's group" without ever naming which one, do NOT \
invent a placeholder — extract it as stated. There is ONE narrow exception where a vague \
recipient description is still worth capturing rather than discarding: when the donation is \
clearly tied to a SPECIFIC, NAMED place or crisis response (e.g. "Palestinian aid groups" when \
the story is plainly about Gaza/Palestine, or "Nepal flood relief efforts") — extract it normally \
with "country_scope" set to that specific place, and let a human reader judge whether the finding \
is useful despite the missing org name; this system scores that case lower automatically, it does \
not need you to pre-filter it. Outside that one exception — no specific place or crisis tying it \
down, just a bare category like "a local charity" or "the children's group" with nothing else — \
do not include an entry for it, and never invent a placeholder recipient.

For each event that IS relevant, extract the following fields. Follow these rules exactly:

1. "summary": a short PARAPHRASED description in your own words (2-3 sentences max). \
   Never copy sentences or distinctive phrasing from the source text.
2. "figure_quote": ONLY the specific figure/fact clause, verbatim, under 15 words \
   (e.g. "$2 million in emergency relief supplies"). If no specific figure is stated \
   anywhere in the source text, set this to null. Do not paraphrase this field — it must \
   be the exact wording used for the number, and nothing more.
3. "donor": the SPECIFIC company or corporate foundation name, as stated. Never write a \
   placeholder like "unspecified corporate partners" here — if you can't name a specific donor, \
   don't include an entry for this event at all instead (see above).
4. "recipient": the nonprofit organization's name, as stated.
5. "recipient_type": your best classification of the recipient, exactly one of "UN agency" \
   (a United Nations body, fund, or programme), "INGO" (an NGO that ITSELF operates across \
   MULTIPLE countries as one organization, e.g. Save the Children, Oxfam, the Red Cross/Red \
   Crescent movement), "NGO" (a national or regional NGO operating within one country), or \
   "Other nonprofit" (anything else — a single hospital, university, school, local charity, \
   foundation, or community organization, EVEN IF it is large, prestigious, or well-known, or \
   anything you're not confident fits the other three). INGO status depends ONLY on whether the \
   organization itself operates across multiple countries — never on size, funding, or \
   reputation. For example, a named hospital or university system (e.g. "Cooper University Health Care") \
   is "Other nonprofit", NOT "INGO", even though it may be large and well-regarded, because it \
   operates in one place, not across multiple countries as one organization. If \
   genuinely unsure, default to "Other nonprofit" rather than guessing at a more specific category \
   — it is the deliberately-safe fallback, not a last resort to avoid.
6. "is_in_kind": true if this is a donation of goods/services/logistics rather than cash.
7. "in_kind_description": if is_in_kind is true, describe what was given, in your own words, \
   with NO estimated dollar value invented. If is_in_kind is false, set to null.
8. "amount_text": the stated monetary amount as a short string (e.g. "$5 million"), or null \
   if none stated or if in-kind with no value given. Never estimate or infer a number that \
   is not explicitly stated in the source text.
9. "country_scope": a specific country/context name if the source text names one, otherwise \
   the string "Unspecified / global". Do NOT guess a country from context or from what you \
   know about the recipient's operations — only use one if it is explicitly named in the text.
10. "status": one of "new_commitment", "renewed_partnership", or "unclear" — based ONLY on \
    whether the source text itself describes this as new vs. a renewal/continuation. If the \
    text doesn't say, use "unclear" rather than guessing.
11. "assumptions": a list of short strings flagging ANYTHING you were unsure about, had to \
    infer, or found ambiguous or conflicting between sources — e.g. "amount differs slightly \
    between the two sources provided", "unclear whether this is a one-time gift or annual \
    pledge", "recipient name is a national committee, not the global agency". If there is \
    truly nothing ambiguous, return an empty list — do not invent caveats for the sake of it.

Return ONLY valid JSON, no other text: a single object with one key, "events", holding a JSON \
array with one object per distinct real event (matching the shape below), in the order you found \
them. If NOTHING relevant was found anywhere in the source text — no real event at all — return \
{{"events": []}}, an empty array, not an empty object and not an array with a placeholder inside.

{{
  "events": [
    {{
      "summary": "...",
      "figure_quote": "..." or null,
      "donor": "...",
      "recipient": "...",
      "recipient_type": "UN agency",
      "is_in_kind": false,
      "in_kind_description": null,
      "amount_text": "..." or null,
      "country_scope": "...",
      "status": "new_commitment",
      "assumptions": []
    }}
  ]
}}

SOURCE TEXT(S):
{source_text}
"""


@dataclass
class ExtractionResult:
    is_relevant: bool
    summary: str = ""
    figure_quote: str | None = None
    donor: str = ""
    recipient: str = ""
    recipient_type_guess: str | None = None  # raw AI classification, normalized later via classify_recipient_type
    is_in_kind: bool = False
    in_kind_description: str | None = None
    amount_text: str | None = None
    country_scope: str = "Unspecified / global"
    status: EventStatus = EventStatus.UNCLEAR
    assumptions: list[str] | None = None

    def __post_init__(self):
        if self.assumptions is None:
            self.assumptions = []


class ExtractionError(Exception):
    pass


def _build_source_text(cluster: ArticleCluster) -> str:
    parts = []
    for a in cluster.articles:
        parts.append(f"--- Source: {a.source_name} ---\nHeadline: {a.title}\nSnippet: {a.summary}\nURL: {a.url}\n")
    return "\n".join(parts)


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    """Thin wrapper around the Gemini API. Isolated in its own function so
    it's the only place that needs to change if the SDK/endpoint changes.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    response = gmodel.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
    )
    return response.text


def _call_bedrock(prompt: str, model: str, region: str) -> str:
    """Thin wrapper around AWS Bedrock's Converse API — same "isolated in
    one function" pattern as _call_gemini, so switching providers again
    later stays a contained change.

    Converse is deliberately model-agnostic (works the same way for Nova,
    Claude, Llama, etc. — whatever ai.bedrock_model in settings.yaml is
    set to), unlike the old implementation which used the Anthropic SDK's
    own Bedrock client and only ever worked with Claude. This means a
    future provider switch back to Claude (if Marketplace approval comes
    through later) is just a model ID change here, not new code.

    Auth is via whatever AWS credentials are already in the environment
    (set by the GitHub Actions OIDC step before this script runs) — nothing
    AWS-specific is configured here beyond the region. No response-format
    parameter exists the way Gemini has response_mime_type; the prompt's
    own "Return ONLY valid JSON" instruction plus the markdown-fence-
    stripping in extract_from_cluster below handles this reliably.
    """
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"]


def _mock_extract(cluster: ArticleCluster) -> str:
    """Deterministic, fake-but-structured response used when --mock is
    passed. Lets the rest of the pipeline (dedupe, scoring, site build,
    email) be developed and tested end-to-end without any AI provider
    credentials, and lets a maintainer smoke-test after every config
    change without spending anything.
    """
    first = cluster.articles[0]
    return json.dumps({"events": [{
        "summary": f"[MOCK] A company reportedly announced support for {first.matched_recipient or 'a monitored organisation'}.",
        "figure_quote": None,
        "donor": "Example Corp (mock)",
        "recipient": first.matched_recipient or "Unknown",
        "recipient_type": "NGO",
        "is_in_kind": False,
        "in_kind_description": None,
        "amount_text": None,
        "country_scope": first.matched_country or "Unspecified / global",
        "status": "unclear",
        "assumptions": ["This is placeholder mock data — run with --mock."],
    }]})


def _extract_via_gemini(prompt: str, model: str, cluster_id: str, max_retries: int) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Historically this fell back to mock output here; now the caller
        # decides mock vs. live before ever reaching this function, so a
        # missing key on a live run is a real configuration error.
        raise FatalExtractionError(
            "ai.provider is 'gemini' in settings.yaml but no GEMINI_API_KEY is set "
            "in the environment. Add it as a secret, or switch provider to 'bedrock'."
        )
    raw = None
    for attempt in range(1, max_retries + 1):
        try:
            _pace_ai_calls()
            raw = _call_gemini(prompt, model, api_key)
            break
        except Exception as e:  # noqa: BLE001 - broad on purpose, see note below
            error_text = str(e).lower()
            if any(marker in error_text for marker in QUOTA_EXHAUSTED_MARKERS):
                raise FatalExtractionError(
                    f"Gemini free-tier daily quota exhausted: {e}\n"
                    f"The free tier for '{model}' currently allows a limited number of "
                    f"requests per day (Google changes this without much notice — check "
                    f"https://aistudio.google.com/usage for your current limit). This resets "
                    f"at midnight Pacific time. Either lower max_ai_calls_per_run in "
                    f"settings.yaml to stay under your actual daily quota, or switch "
                    f"ai.provider to 'bedrock', which has no daily request ceiling."
                ) from e
            if any(marker in error_text for marker in NON_RETRYABLE_ERROR_MARKERS):
                # This will not fix itself by retrying. Stop the whole
                # run now with a clear, actionable message rather than
                # repeating the same failure for every remaining cluster.
                raise FatalExtractionError(
                    f"Gemini call failed with a non-retryable error: {e}\n"
                    f"This usually means the model name in config/settings.yaml "
                    f"(currently '{model}') is wrong or has been retired. "
                    f"Check https://ai.google.dev/gemini-api/docs/models for the "
                    f"current model name and update settings.yaml, or that your "
                    f"GEMINI_API_KEY secret is valid."
                ) from e
            # Free-tier quota errors, transient network issues, etc. all
            # land here. We back off and retry rather than crash the run;
            # if all retries fail we skip this one cluster and move on.
            wait = 2 ** attempt
            logger.warning("Gemini call failed (attempt %d/%d): %s. Retrying in %ds.",
                            attempt, max_retries, e, wait)
            time.sleep(wait)
    if raw is None:
        logger.error("Giving up on cluster %s after %d failed attempts.", cluster_id, max_retries)
        return None
    return raw



# Botocore error codes worth retrying: per-minute/account throttling, a
# 5xx-equivalent server error, or the model not yet warmed up. Unlike
# Gemini's free tier, Bedrock has no daily wall to hit — these are
# transient and worth retrying rather than giving up immediately.
BEDROCK_RETRYABLE_ERROR_CODES = {
    "ThrottlingException", "ModelTimeoutException", "InternalServerException",
    "ServiceUnavailableException", "ModelNotReadyException", "ServiceQuotaExceededException",
}


def _extract_via_bedrock(prompt: str, model: str, region: str, cluster_id: str, max_retries: int) -> str | None:
    import botocore.exceptions

    raw = None
    for attempt in range(1, max_retries + 1):
        try:
            _pace_ai_calls()
            raw = _call_bedrock(prompt, model, region)
            break
        except (botocore.exceptions.NoCredentialsError, botocore.exceptions.PartialCredentialsError) as e:
            raise FatalExtractionError(
                f"Bedrock call failed to resolve AWS credentials: {e}\n"
                f"Check that the GitHub workflow's aws-actions/configure-aws-credentials "
                f"step ran successfully before this step, that id-token: write is set in "
                f"the workflow's permissions, and that the IAM role's trust policy actually "
                f"matches this repo/branch."
            ) from e
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                raise FatalExtractionError(
                    f"Bedrock call failed: model '{model}' not found or not accessible in region "
                    f"'{region}': {e}\n"
                    f"Check that config/settings.yaml's ai.bedrock_model / ai.bedrock_region are "
                    f"correct — some models (including some Nova variants outside their home region) "
                    f"require a cross-region inference profile ID (e.g. 'eu.amazon.nova-pro-v1:0') "
                    f"rather than the bare foundation-model ID; check the Bedrock console's "
                    f"'Model access' page for the exact invokable ID in this region."
                ) from e
            if code == "AccessDeniedException":
                raise FatalExtractionError(
                    f"Bedrock call failed due to an access-denied error: {e}\n"
                    f"Check two SEPARATE things: (1) the IAM role's permission policy grants "
                    f"bedrock:InvokeModel on this exact model's ARN, and (2) model access is "
                    f"explicitly enabled for it on the Bedrock console's 'Model access' page for "
                    f"region '{region}' — this is a one-click, no-approval-wait step for first-party "
                    f"Amazon models like Nova, but it's a distinct step from both the IAM policy and "
                    f"from any AWS Marketplace subscription, and easy to miss."
                ) from e
            if code == "ValidationException":
                raise FatalExtractionError(f"Bedrock call failed with a validation error: {e}") from e
            if code in BEDROCK_RETRYABLE_ERROR_CODES:
                wait = 2 ** attempt
                logger.warning("Bedrock call failed (attempt %d/%d, %s): %s. Retrying in %ds.",
                                attempt, max_retries, code, e, wait)
                time.sleep(wait)
                continue
            raise  # an unrecognized error code — a real bug, don't mask it
    if raw is None:
        logger.error("Giving up on cluster %s after %d failed attempts.", cluster_id, max_retries)
        return None
    return raw


def _parse_event(data: dict) -> ExtractionResult:
    """Raises (ValueError, KeyError) on unexpected shape — caller decides
    whether that sinks the whole cluster or just this one event."""
    return ExtractionResult(
        is_relevant=True,
        summary=data.get("summary", ""),
        figure_quote=data.get("figure_quote"),
        donor=data.get("donor", "").strip(),
        recipient=data.get("recipient", "").strip(),
        recipient_type_guess=data.get("recipient_type"),
        is_in_kind=bool(data.get("is_in_kind", False)),
        in_kind_description=data.get("in_kind_description"),
        amount_text=data.get("amount_text"),
        country_scope=data.get("country_scope") or "Unspecified / global",
        status=EventStatus(data.get("status", "unclear")),
        assumptions=data.get("assumptions", []) or [],
    )


def extract_from_cluster(
    cluster: ArticleCluster,
    ai_settings: dict,
    mock: bool = False,
    max_retries: int = 3,
) -> list[ExtractionResult]:
    """Returns a list of events found in this cluster — usually zero or
    one, occasionally more than one. A cluster is a keyword/title-
    similarity GUESS that a group of articles describes the same event;
    that guess is sometimes wrong (a real case: ~20 articles about one
    celebrity's vague-recipient pledge included a completely different,
    specifically-named $10,000 donation mentioned in just one of them —
    extracting only the dominant story silently lost the second, more
    useful finding). The prompt now asks for every distinct event it can
    find rather than assuming there's exactly one, so this returns all of
    them. Empty list if nothing relevant was found, or if extraction
    failed after retries (logged, never raised — one bad cluster should
    not kill the whole day's run).

    ai_settings is the full settings.yaml "ai:" block — which provider-
    specific fields it needs depends on ai_settings["provider"].
    """
    source_text = _build_source_text(cluster)
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(source_text=source_text)

    if mock:
        raw = _mock_extract(cluster)
    else:
        provider = ai_settings.get("provider", "gemini")
        if provider == "bedrock":
            raw = _extract_via_bedrock(
                prompt, ai_settings["bedrock_model"], ai_settings["bedrock_region"],
                cluster.cluster_id, max_retries,
            )
        elif provider == "gemini":
            raw = _extract_via_gemini(prompt, ai_settings["model"], cluster.cluster_id, max_retries)
        else:
            raise FatalExtractionError(
                f"Unknown ai.provider '{provider}' in settings.yaml — must be 'bedrock' or 'gemini'."
            )
        if raw is None:
            return []

    try:
        cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Could not parse model output as JSON for cluster %s: %s\nRaw: %s",
                     cluster.cluster_id, e, raw[:500])
        return []

    if isinstance(data, dict) and "events" in data:
        event_dicts = data.get("events") or []
    elif isinstance(data, dict) and ("donor" in data or "recipient" in data):
        # Defensive fallback: the model occasionally ignores the "events"
        # wrapper and returns a single old-style object directly despite
        # the prompt's explicit shape. Treat it as a one-item list rather
        # than discarding a perfectly usable extraction over a formatting
        # slip.
        event_dicts = [data]
    else:
        event_dicts = []

    results = []
    for event_data in event_dicts:
        if not isinstance(event_data, dict):
            continue
        try:
            results.append(_parse_event(event_data))
        except (ValueError, KeyError) as e:
            # One malformed event in the array shouldn't cost every OTHER
            # genuine event the model found in the same cluster.
            logger.error("One event for cluster %s had unexpected shape, skipping just that "
                         "one: %s", cluster.cluster_id, e)
    return results


def score_confidence(
    cluster: ArticleCluster, result: ExtractionResult, confidence_cfg: dict, recipients: list[Recipient],
) -> tuple[int, dict]:
    """Implements the rubric defined in settings.yaml. Returns (total_score, breakdown_dict).

    Four factors, not three: source tier, explicit amount, multiple
    sources, and (new) whether the recipient is one of the curated 50
    this tracker is specifically anchored on. That last one is a
    deliberately different KIND of signal from the other three — it's
    not about how well-evidenced the story is, it's about how squarely
    it's in scope for this project's actual purpose. A single-source
    story with no dollar figure about a real donation to a recipient
    this tracker exists to watch (e.g. the World Bank) is a stronger
    hit for THIS project than the same evidentiary strength for an
    off-list local charity, even though "confidence the story is real"
    is arguably similar in both cases — see the settings.yaml comment
    on monitored_recipient_points for the concrete example that
    motivated adding this.

    The four factors can sum above 10 (e.g. official source + amount +
    multiple sources + monitored recipient); clipped to 10 so the "X/10"
    badge never reads oddly.
    """
    breakdown = {}

    tier_points = confidence_cfg["source_tier_points"]
    tier_key = cluster.best_source_tier.value
    breakdown["source_tier"] = {"value": cluster.best_source_tier.value, "points": tier_points.get(tier_key, 0)}

    amount_points = confidence_cfg["explicit_amount_stated_points"] if (result.amount_text or result.figure_quote) else 0
    breakdown["explicit_amount_stated"] = {"value": bool(result.amount_text or result.figure_quote), "points": amount_points}

    multi_source_points = confidence_cfg["multiple_independent_sources_points"] if cluster.independent_source_count > 1 else 0
    breakdown["multiple_independent_sources"] = {"value": cluster.independent_source_count, "points": multi_source_points}

    is_monitored = match_curated_recipient(result.recipient, recipients) is not None
    monitored_points = confidence_cfg["monitored_recipient_points"] if is_monitored else 0
    breakdown["monitored_recipient"] = {"value": is_monitored, "points": monitored_points}

    # Almost every entry reaching this point has a properly named
    # recipient and gets this automatically — the one exception is the
    # vague-but-in-scope-place case run_daily.py publishes anyway rather
    # than rejecting outright (see is_generic_recipient's caller), which
    # should score lower precisely because the recipient isn't actually
    # named. Recomputed here rather than threaded through as a parameter
    # since result.recipient/assumptions are already right here.
    recipient_is_vague = is_generic_recipient(result.recipient) or \
        assumptions_admit_missing_name(result.assumptions, RECIPIENT_NAME_CONTEXT_WORDS)
    named_points = 0 if recipient_is_vague else confidence_cfg["specific_recipient_named_points"]
    breakdown["specific_recipient_named"] = {"value": not recipient_is_vague, "points": named_points}

    total = sum(component["points"] for component in breakdown.values())
    return min(total, 10), breakdown


def _articles_for_event(cluster: ArticleCluster, result: ExtractionResult) -> list:
    """Narrows a multi-event cluster's full article list down to the ones
    that actually mention THIS event's donor — without this, every event
    split out of one cluster (see extract_from_cluster) would cite every
    article in the WHOLE cluster, including ones entirely about a
    different event. A real case: a $10,000 Pop Base donation ended up
    citing 24 sources, 23 of which were actually about an unrelated $1M
    Macklemore pledge that shared the same cluster — genuinely confusing
    to a reader ("why does this cite so many sources about someone
    else?"), not the merely-cosmetic over-attribution this was first
    treated as.

    Matches on the DONOR name specifically, not recipient — it's normally
    the more distinctive, reliably-literal string in a headline (a
    company/person's actual name is far less likely to coincidentally
    appear in an unrelated article than a generic recipient description
    is). Falls back to the full cluster if the donor name doesn't
    literally appear in ANY article (a paraphrased/reworded donor name,
    or the normal single-event case where this is a no-op safety net)
    rather than publish an entry with zero sources.
    """
    donor_l = result.donor.strip().lower()
    if not donor_l:
        return cluster.articles
    matching = [a for a in cluster.articles if donor_l in f"{a.title} {a.summary}".lower()]
    return matching or cluster.articles


def build_donation_entry(
    cluster: ArticleCluster,
    result: ExtractionResult,
    confidence_cfg: dict,
    recipients: list[Recipient],
) -> DonationEntry:
    source_articles = _articles_for_event(cluster, result)
    # score_confidence()'s source-tier/independent-source-count signals
    # must come from the SAME narrowed article set as the sources shown
    # on the entry -- not the full cluster. Without this, Pop Base's
    # $10,000 entry (see _articles_for_event's docstring) scored a full
    # 3/3 "multiple independent sources" credit for 23 articles, when
    # only 1 of them was actually about Pop Base at all; the other 22
    # corroborate a completely different ($1M Macklemore) donation.
    scoring_cluster = ArticleCluster(cluster.cluster_id, source_articles)
    score, breakdown = score_confidence(scoring_cluster, result, confidence_cfg, recipients)
    recipient_type = classify_recipient_type(result.recipient, recipients, result.recipient_type_guess)
    return DonationEntry(
        entry_id=str(uuid.uuid4()),
        donor=result.donor,
        recipient=result.recipient,
        recipient_type=recipient_type,
        amount_text=result.amount_text,
        is_in_kind=result.is_in_kind,
        in_kind_description=result.in_kind_description,
        country_scope=result.country_scope,
        status=result.status,
        summary=result.summary,
        figure_quote=result.figure_quote,
        assumptions=result.assumptions,
        confidence_score=score,
        confidence_breakdown=breakdown,
        source_urls=[a.url for a in source_articles],
        source_names=list({a.source_name for a in source_articles}),
        source_channels=sorted({a.fetch_source for a in source_articles}),
        date_found=now_iso(),
    )
