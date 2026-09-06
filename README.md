# Corporate Giving Monitor

A daily-updating tracker of private-sector (corporate) donations to UN
agencies, INGOs, and NGOs — built to surface donations that aren't
centrally logged anywhere else (e.g. in OCHA's FTS). Published as a
GitHub Pages site styled as a daily newspaper, with a weekly email digest
of the highest-confidence findings.

**This is a discovery aid, not an audited financial record.** Every entry
carries a confidence score and lists exactly what it's unsure about — see
`docs/sources.html` (or the "Sources & Limitations" page on the live site)
for what this can and can't catch.

---

## How it works, in short

Every day, an automated job:
1. Searches Google News + a set of PR wire feeds for stories mentioning a
   monitored UN agency/INGO/NGO alongside donation-related language
2. Filters out obviously irrelevant results with a cheap keyword pass
3. Groups near-duplicate stories about the same event together
4. Sends each event to Google's Gemini AI to extract structured facts
   (donor, recipient, amount, in-kind description, etc.)
5. Checks each finding against everything published before, to catch
   exact duplicates and flag likely partnership renewals
6. Publishes the day's findings as a new page on the site
7. Once a week, emails a digest of the best findings to a distribution list

## For non-technical editors: what you can change without touching code

Everything you're likely to want to adjust lives in the `config/` folder
as plain text files with instructions written at the top of each one:

| File | What it controls |
|---|---|
| `config/recipients.txt` | Which UN agencies, INGOs, and NGOs to watch for incoming donations |
| `config/countries.txt` | Which countries/crises count as "in scope" geographically |
| `config/trigger_phrases.txt` | The words/phrases used to spot donation-related stories |
| `config/pr_wire_feeds.txt` | Which press-release wire feeds get checked alongside Google News |
| `config/settings.yaml` | Confidence scoring weights, AI model choice, email settings |

To check you haven't broken a formatting rule after editing, run:
```
python src/config_loader.py
```
This will either print `OK: ...` or tell you exactly which file and line
number has a problem, without needing to run the whole pipeline.

**You do not need to touch anything in `src/`, `templates/`, or
`scripts/` to make routine changes.**

---

## One-time setup (for whoever is setting this up)

### 1. Get a free Gemini API key
Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and
create a free API key. The free tier gives ~1,500 requests/day on
`gemini-2.0-flash`, which is what this project is tuned for — the
pipeline caps itself at `max_ai_calls_per_run` in `settings.yaml` so it
can never accidentally exceed your daily quota.

### 2. Fork/clone this repo, then add secrets
In your GitHub repo: **Settings → Secrets and variables → Actions**, add:
- `GEMINI_API_KEY` — your key from step 1
- `RESEND_API_KEY` — (optional) if using Resend for the weekly email; see below
- `SMTP_PASSWORD` — (optional) if using SMTP instead of Resend

And under **Settings → Secrets and variables → Actions → Variables** tab:
- `SITE_URL` — your GitHub Pages URL once it's live, e.g.
  `https://yourorg.github.io/donation-tracker`

### 3. Turn on GitHub Pages
**Settings → Pages** → set source to "Deploy from a branch" → branch
`main`, folder `/docs`. Save. The site will appear at the URL GitHub
shows you there — put that URL into the `SITE_URL` variable above.

### 4. Set up the weekly email
Edit `config/settings.yaml`:
```yaml
email:
  enabled: true
  recipients: "person1@org.org, person2@org.org"
  method: "resend"          # or "smtp"
  from_address: "digest@yourdomain.org"
```
If using **Resend** (recommended — simple, generous free tier): sign up
at [resend.com](https://resend.com), verify a sending domain, create an
API key, add it as the `RESEND_API_KEY` secret above.

If using **SMTP** instead (e.g. a Gmail account): fill in `smtp_host`,
`smtp_port`, and `smtp_username` in `settings.yaml`, and add your SMTP
password (for Gmail, an "app password", not your normal password) as the
`SMTP_PASSWORD` secret.

### 5. Test before relying on it
The daily workflow can be triggered manually from the **Actions** tab
(`Daily donation scan` → `Run workflow`) without waiting for the 06:00 UTC
schedule — do this once after setup to confirm everything works.

---

## Running it locally (for development/testing)

```bash
pip install -r requirements.txt

# Check your config files are valid:
python src/config_loader.py

# Full dry run with fake data, no API calls or internet needed at all:
python scripts/run_daily.py --mock --mock-fetch

# Real fetching (Google News + PR wires) but fake AI extraction —
# good for checking your recipients/triggers/countries lists are
# actually finding real candidate articles, without spending API quota:
python scripts/run_daily.py --mock

# Full live run (needs GEMINI_API_KEY set as an environment variable):
export GEMINI_API_KEY=your_key_here
python scripts/run_daily.py

# Preview the weekly email without sending it:
python scripts/run_weekly_email.py --dry-run
```

After any run, open `docs/index.html` in a browser to see the site.

---

## Why these design choices (for whoever maintains the code)

- **Recipient-anchored search, not a company list.** Donors could be
  "anyone anywhere," so every search query is built from the known,
  stable list of UN agencies/INGOs/NGOs rather than a pre-built company
  list. New, previously-unknown donors surface naturally because the
  query is anchored on the org they gave to.
- **Two-stage deduplication.** Cheap fuzzy title matching (`clustering.py`)
  reduces AI calls by grouping same-day, same-story articles together —
  this is a cost optimisation, not the source of truth on duplicates.
  The real backstop is `store.py`, which compares actual extracted facts
  (donor, recipient, amount, timeframe) across the full lookback window,
  catching duplicates/renewals even when headlines look nothing alike.
- **Copyright-safe extraction.** The AI prompt requires a paraphrased
  summary plus, separately, only a short (<15 word) verbatim clause for
  the specific figure — never a reproduced sentence or paragraph from a
  news source.
- **"Unspecified / global" scope, never guessed.** A donation to an
  agency's general/unearmarked fund often doesn't name a country at all.
  Rather than have the AI infer a likely country from context (which
  would be an unstated assumption presented as fact), these are tagged
  "Unspecified / global" and still published.
- **Free-tier AI budget is a hard constraint, not an afterthought.**
  `max_ai_calls_per_run` in settings.yaml exists specifically so a busy
  news day can never blow through the Gemini free-tier daily quota and
  silently fail partway through a run.

## Known limitations (see also the live "Sources & Limitations" page)

- Will under-catch small, local, or purely social-media-announced
  donations — general news and PR wires are the ceiling of what this can see.
- Fuzzy clustering/dedup is approximate; very occasionally the same event
  may be published twice if wording and facts both diverge enough between
  outlets.
- Confidence scores reflect source strength and corroboration, not legal
  or financial verification — always check the linked source before
  citing a figure externally.
