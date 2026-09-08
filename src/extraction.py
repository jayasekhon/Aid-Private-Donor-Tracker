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


def is_generic_donor(donor: str) -> bool:
    if not donor or not donor.strip():
        return True
    d = donor.strip().lower()
    return any(marker in d for marker in GENERIC_DONOR_MARKERS)


def recipient_is_monitored(recipient: str, recipients: list[Recipient]) -> bool:
    """True if the extracted recipient corresponds to one of the orgs this
    tracker is actually scoped to watch (recipients.txt, including
    aliases). Google News' search doesn't strictly enforce our query — a
    returned article can be about something else entirely that merely
    triggered a loose relevance match — so the model can (and does)
    sometimes extract a real donation to a real recipient that just isn't
    one we track. That's out of scope, not a tracked finding.
    """
    if not recipient or not recipient.strip():
        return False
    candidate = recipient.strip().lower()
    for r in recipients:
        for name in r.all_names:
            name_l = name.strip().lower()
            if not name_l:
                continue
            if candidate == name_l or candidate in name_l or name_l in candidate:
                return True
            if fuzz.token_set_ratio(candidate, name_l) >= 88:
                return True
    return False

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
items that a keyword filter believes describe the SAME real-world donation/partnership event.

Your job: decide if this is genuinely a private company (or corporate foundation) donating to, \
partnering with, or otherwise financially/materially supporting a UN agency, INGO, or NGO. If it \
is NOT (e.g. it's a government donation, an unrelated story that matched keywords by coincidence, \
or pure speculation with no confirmed commitment), say so clearly and set "is_relevant" to false. \
This tracker exists to identify WHICH company gave — if the source text never names a specific \
company or corporate foundation (only vague language like "corporate partners", "several \
companies", or "a donor"), that is also not relevant: set "is_relevant" to false rather than \
inventing a placeholder donor. Similarly, the recipient must be one of the organizations this \
tracker is actually scoped to (a UN agency, INGO, or NGO watched by name) — if the story is about \
some other organization that merely happened to be mentioned alongside a watched one (e.g. in an \
unrelated paragraph of the same article), set "is_relevant" to false.

If it IS relevant, extract the following as JSON. Follow these rules exactly:

1. "summary": a short PARAPHRASED description in your own words (2-3 sentences max). \
   Never copy sentences or distinctive phrasing from the source text.
2. "figure_quote": ONLY the specific figure/fact clause, verbatim, under 15 words \
   (e.g. "$2 million in emergency relief supplies"). If no specific figure is stated \
   anywhere in the source text, set this to null. Do not paraphrase this field — it must \
   be the exact wording used for the number, and nothing more.
3. "donor": the SPECIFIC company or corporate foundation name, as stated. Never write a \
   placeholder like "unspecified corporate partners" here — if you can't name a specific donor, \
   set "is_relevant" to false instead (see above).
4. "recipient": the UN agency / INGO / NGO name, as stated.
5. "is_in_kind": true if this is a donation of goods/services/logistics rather than cash.
6. "in_kind_description": if is_in_kind is true, describe what was given, in your own words, \
   with NO estimated dollar value invented. If is_in_kind is false, set to null.
7. "amount_text": the stated monetary amount as a short string (e.g. "$5 million"), or null \
   if none stated or if in-kind with no value given. Never estimate or infer a number that \
   is not explicitly stated in the source text.
8. "country_scope": a specific country/context name if the source text names one, otherwise \
   the string "Unspecified / global". Do NOT guess a country from context or from what you \
   know about the recipient's operations — only use one if it is explicitly named in the text.
9. "status": one of "new_commitment", "renewed_partnership", or "unclear" — based ONLY on \
   whether the source text itself describes this as new vs. a renewal/continuation. If the \
   text doesn't say, use "unclear" rather than guessing.
10. "assumptions": a list of short strings flagging ANYTHING you were unsure about, had to \
    infer, or found ambiguous or conflicting between sources — e.g. "amount differs slightly \
    between the two sources provided", "unclear whether this is a one-time gift or annual \
    pledge", "recipient name is a national committee, not the global agency". If there is \
    truly nothing ambiguous, return an empty list — do not invent caveats for the sake of it.
11. "is_relevant": true/false as described above.

Return ONLY valid JSON, no other text, matching this shape:
{{
  "is_relevant": true,
  "summary": "...",
  "figure_quote": "..." or null,
  "donor": "...",
  "recipient": "...",
  "is_in_kind": false,
  "in_kind_description": null,
  "amount_text": "..." or null,
  "country_scope": "...",
  "status": "new_commitment",
  "assumptions": []
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
    """Thin wrapper around Claude on Bedrock — same "isolated in one
    function" pattern as _call_gemini, so switching providers again later
    stays a contained change.

    Auth is via whatever AWS credentials are already in the environment
    (set by the GitHub Actions OIDC step before this script runs) — nothing
    AWS-specific is configured here beyond the region. No response-format
    parameter exists for Claude the way Gemini has response_mime_type; the
    prompt's own "Return ONLY valid JSON" instruction plus the markdown-
    fence-stripping in extract_from_cluster below handles this reliably.

    No temperature/sampling parameter is set — verified against the
    installed SDK that messages.create() on this client doesn't accept one
    at all (confirmed via inspect.signature, not assumed), so there's
    nothing to configure here; the model's default sampling behaviour is
    used.
    """
    from anthropic import AnthropicBedrockMantle

    client = AnthropicBedrockMantle(aws_region=region)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _mock_extract(cluster: ArticleCluster) -> str:
    """Deterministic, fake-but-structured response used when --mock is
    passed. Lets the rest of the pipeline (dedupe, scoring, site build,
    email) be developed and tested end-to-end without any AI provider
    credentials, and lets a maintainer smoke-test after every config
    change without spending anything.
    """
    first = cluster.articles[0]
    return json.dumps({
        "is_relevant": True,
        "summary": f"[MOCK] A company reportedly announced support for {first.matched_recipient or 'a monitored organisation'}.",
        "figure_quote": None,
        "donor": "Example Corp (mock)",
        "recipient": first.matched_recipient or "Unknown",
        "is_in_kind": False,
        "in_kind_description": None,
        "amount_text": None,
        "country_scope": first.matched_country or "Unspecified / global",
        "status": "unclear",
        "assumptions": ["This is placeholder mock data — run with --mock."],
    })


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


def _extract_via_bedrock(prompt: str, model: str, region: str, cluster_id: str, max_retries: int) -> str | None:
    import anthropic

    raw = None
    for attempt in range(1, max_retries + 1):
        try:
            _pace_ai_calls()
            raw = _call_bedrock(prompt, model, region)
            break
        except RuntimeError as e:
            # The Bedrock client raises a plain RuntimeError (not one of
            # the anthropic.* typed exceptions below) when it can't resolve
            # AWS credentials at all — confirmed by testing, not assumed.
            # This is a setup/config problem, not a transient one, so it
            # gets the same "stop now" treatment as the typed auth errors.
            if "credential" in str(e).lower():
                raise FatalExtractionError(
                    f"Bedrock call failed to resolve AWS credentials: {e}\n"
                    f"Check that the GitHub workflow's aws-actions/configure-aws-credentials "
                    f"step ran successfully before this step, that id-token: write is set in "
                    f"the workflow's permissions, and that the IAM role's trust policy actually "
                    f"matches this repo/branch."
                ) from e
            raise  # an unrelated RuntimeError — a real bug, don't mask it as a Bedrock auth issue
        except anthropic.NotFoundError as e:
            raise FatalExtractionError(
                f"Bedrock call failed: model '{model}' not found or not accessible in region "
                f"'{region}': {e}\n"
                f"Check that model access is enabled for this model in the AWS Bedrock console "
                f"for this region, and that config/settings.yaml's ai.bedrock_model / "
                f"ai.bedrock_region are correct."
            ) from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise FatalExtractionError(
                f"Bedrock call failed due to an authentication/permission error: {e}\n"
                f"Check the IAM role's trust policy (GitHub OIDC provider + repo/branch condition) "
                f"and its permission policy (bedrock:InvokeModel on the exact inference-profile and "
                f"foundation-model ARNs), and that the GitHub workflow's "
                f"aws-actions/configure-aws-credentials step is pointed at the right Role ARN."
            ) from e
        except anthropic.BadRequestError as e:
            # A malformed request (bad params, bad prompt shape) — retrying
            # the identical request will fail identically every time.
            raise FatalExtractionError(f"Bedrock call failed with a bad-request error: {e}") from e
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # Transient: per-minute throttling, a 5xx, or a network hiccup.
            # Unlike Gemini's free tier, Bedrock has no daily wall to hit —
            # these are worth retrying.
            wait = 2 ** attempt
            logger.warning("Bedrock call failed (attempt %d/%d): %s. Retrying in %ds.",
                            attempt, max_retries, e, wait)
            time.sleep(wait)
    if raw is None:
        logger.error("Giving up on cluster %s after %d failed attempts.", cluster_id, max_retries)
        return None
    return raw


def extract_from_cluster(
    cluster: ArticleCluster,
    ai_settings: dict,
    mock: bool = False,
    max_retries: int = 3,
) -> ExtractionResult | None:
    """Returns None if the model judged the cluster not actually relevant,
    or if extraction failed after retries (logged, never raised — one bad
    cluster should not kill the whole day's run).

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
            return None

    try:
        cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Could not parse model output as JSON for cluster %s: %s\nRaw: %s",
                     cluster.cluster_id, e, raw[:500])
        return None

    if not data.get("is_relevant", False):
        return None

    try:
        return ExtractionResult(
            is_relevant=True,
            summary=data.get("summary", ""),
            figure_quote=data.get("figure_quote"),
            donor=data.get("donor", "").strip(),
            recipient=data.get("recipient", "").strip(),
            is_in_kind=bool(data.get("is_in_kind", False)),
            in_kind_description=data.get("in_kind_description"),
            amount_text=data.get("amount_text"),
            country_scope=data.get("country_scope") or "Unspecified / global",
            status=EventStatus(data.get("status", "unclear")),
            assumptions=data.get("assumptions", []) or [],
        )
    except (ValueError, KeyError) as e:
        logger.error("Model output for cluster %s had unexpected shape: %s", cluster.cluster_id, e)
        return None


def score_confidence(cluster: ArticleCluster, result: ExtractionResult, confidence_cfg: dict) -> tuple[int, dict]:
    """Implements the rubric defined in settings.yaml. Returns (total_score, breakdown_dict)."""
    breakdown = {}

    tier_points = confidence_cfg["source_tier_points"]
    tier_key = cluster.best_source_tier.value
    breakdown["source_tier"] = {"value": cluster.best_source_tier.value, "points": tier_points.get(tier_key, 0)}

    amount_points = confidence_cfg["explicit_amount_stated_points"] if (result.amount_text or result.figure_quote) else 0
    breakdown["explicit_amount_stated"] = {"value": bool(result.amount_text or result.figure_quote), "points": amount_points}

    multi_source_points = confidence_cfg["multiple_independent_sources_points"] if cluster.independent_source_count > 1 else 0
    breakdown["multiple_independent_sources"] = {"value": cluster.independent_source_count, "points": multi_source_points}

    total = breakdown["source_tier"]["points"] + breakdown["explicit_amount_stated"]["points"] + breakdown["multiple_independent_sources"]["points"]
    return total, breakdown


def build_donation_entry(
    cluster: ArticleCluster,
    result: ExtractionResult,
    confidence_cfg: dict,
) -> DonationEntry:
    score, breakdown = score_confidence(cluster, result, confidence_cfg)
    return DonationEntry(
        entry_id=str(uuid.uuid4()),
        donor=result.donor,
        recipient=result.recipient,
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
        source_urls=[a.url for a in cluster.articles],
        source_names=list({a.source_name for a in cluster.articles}),
        date_found=now_iso(),
    )
