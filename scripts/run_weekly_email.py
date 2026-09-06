#!/usr/bin/env python3
"""Sends a weekly email digest of the past 7 days' highlights.

"Highlights" = entries from the last 7 days, excluding low-confidence ones
(threshold set in config/settings.yaml), sorted by confidence score.

Two send methods are supported:
  - "resend": uses the Resend API (https://resend.com) — has a generous
    free tier and needs only one secret (RESEND_API_KEY). Recommended.
  - "smtp": uses a standard SMTP server (e.g. Gmail) if you'd rather not
    add a third-party service. Needs SMTP_PASSWORD as a secret alongside
    the smtp_username/host/port in settings.yaml.

If email.enabled is false in settings.yaml, this script exits without
sending anything — that's the default until you've filled in real
recipient addresses and sender details.

Usage:
    python scripts/run_weekly_email.py
    python scripts/run_weekly_email.py --dry-run   # print the email instead of sending
"""
from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_all
from src.site_builder import load_all_editions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_weekly_email")


def collect_week_highlights(low_confidence_threshold: int, days: int = 7) -> list[dict]:
    editions = load_all_editions()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    highlights = []
    for ed in editions:
        try:
            ed_date = datetime.strptime(ed["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ed_date < cutoff:
            continue
        for entry in ed["entries"]:
            if entry["confidence_score"] >= low_confidence_threshold:
                entry_with_date = dict(entry)
                entry_with_date["_edition_date"] = ed["date"]
                highlights.append(entry_with_date)
    highlights.sort(key=lambda e: e["confidence_score"], reverse=True)
    return highlights


def render_email_html(highlights: list[dict], site_title: str, site_url_base: str) -> str:
    if not highlights:
        body = "<p>No high-confidence donation entries were found this week.</p>"
    else:
        rows = []
        for h in highlights:
            figure = f"<br><em>&ldquo;{h['figure_quote']}&rdquo;</em>" if h.get("figure_quote") else ""
            rows.append(
                f"<li style='margin-bottom:16px;'>"
                f"<strong>{h['donor']} &rarr; {h['recipient']}</strong> "
                f"(confidence {h['confidence_score']}/10)<br>"
                f"{h['summary']}{figure}<br>"
                f"<a href='{site_url_base}/editions/{h['_edition_date']}.html'>View full entry</a>"
                f"</li>"
            )
        body = "<ul style='list-style:none;padding:0;'>" + "".join(rows) + "</ul>"
    return (
        f"<div style='font-family:Georgia,serif;max-width:640px;margin:0 auto;'>"
        f"<h2>{site_title} — Weekly Digest</h2>"
        f"{body}"
        f"<p style='color:#666;font-size:13px;'>Full daily editions: "
        f"<a href='{site_url_base}'>{site_url_base}</a></p>"
        f"</div>"
    )


def send_via_resend(subject: str, html: str, from_addr: str, to_addrs: list[str]) -> None:
    import requests
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not set. Add it as a GitHub Actions secret.")
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"from": from_addr, "to": to_addrs, "subject": subject, "html": html},
        timeout=15,
    )
    resp.raise_for_status()
    logger.info("Email sent via Resend to %d recipient(s).", len(to_addrs))


def send_via_smtp(subject: str, html: str, from_addr: str, to_addrs: list[str], smtp_cfg: dict) -> None:
    password = os.environ.get("SMTP_PASSWORD")
    if not password:
        raise RuntimeError("SMTP_PASSWORD is not set. Add it as a GitHub Actions secret.")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_cfg["smtp_host"], smtp_cfg["smtp_port"]) as server:
        server.starttls()
        server.login(smtp_cfg["smtp_username"], password)
        server.sendmail(from_addr, to_addrs, msg.as_string())
    logger.info("Email sent via SMTP to %d recipient(s).", len(to_addrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print the email instead of sending it.")
    parser.add_argument("--site-url", default=os.environ.get("SITE_URL", "https://example.github.io/donation-tracker"),
                         help="Base URL of the published GitHub Pages site, for links in the email.")
    args = parser.parse_args()

    cfg = load_all()
    settings = cfg["settings"]
    email_cfg = settings["email"]

    if not email_cfg.get("enabled", False) and not args.dry_run:
        logger.info("email.enabled is false in settings.yaml — skipping send. Use --dry-run to preview anyway.")
        return

    highlights = collect_week_highlights(settings["confidence"]["low_confidence_threshold"])
    logger.info("Found %d high-confidence entries from the past 7 days.", len(highlights))

    html = render_email_html(highlights, settings["site"]["title"], args.site_url)
    subject = f"{email_cfg['subject_prefix']} — {datetime.now(timezone.utc).strftime('%-d %b %Y')}"
    to_addrs = [a.strip() for a in email_cfg["recipients"].split(",") if a.strip()]

    if args.dry_run:
        print(f"SUBJECT: {subject}")
        print(f"TO: {to_addrs}")
        print("---")
        print(html)
        return

    if email_cfg["method"] == "resend":
        send_via_resend(subject, html, email_cfg["from_address"], to_addrs)
    elif email_cfg["method"] == "smtp":
        send_via_smtp(subject, html, email_cfg["from_address"], to_addrs, email_cfg)
    else:
        raise ValueError(f"Unknown email method in settings.yaml: {email_cfg['method']}")


if __name__ == "__main__":
    main()
