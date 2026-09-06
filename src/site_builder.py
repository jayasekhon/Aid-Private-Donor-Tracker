"""Builds the static site (for GitHub Pages) from a day's DonationEntry list.

Two outputs per day:
  - data/editions/YYYY-MM-DD.json   (the raw data — source of truth)
  - docs/editions/YYYY-MM-DD.html   (the rendered page)
Plus docs/index.html (always = latest edition), docs/archive/index.html,
and docs/sources.html, regenerated every run from the full edition history
so the archive and nav links stay correct.

Kept deliberately simple (Jinja2 + static files written directly to disk) —
no site-generator framework dependency, which keeps the non-technical
config story simple: nothing here needs a build toolchain beyond Python.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import DonationEntry

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
DOCS_DIR = ROOT / "docs"
EDITIONS_DATA_DIR = ROOT / "data" / "editions"

CONFIDENCE_HIGH = 8
CONFIDENCE_MID = 5


def _confidence_css_class(score: int) -> str:
    if score >= CONFIDENCE_HIGH:
        return "badge-confidence-high"
    if score >= CONFIDENCE_MID:
        return "badge-confidence-mid"
    return "badge-confidence-low"


def _status_display(status_value: str) -> str:
    return {
        "new_commitment": "New commitment",
        "renewed_partnership": "Renewed partnership",
        "unclear": "Status unclear",
    }.get(status_value, status_value)


def _entry_view(e: DonationEntry) -> dict:
    d = e.to_dict()
    d["confidence_css_class"] = _confidence_css_class(e.confidence_score)
    d["status_display"] = _status_display(e.status.value if hasattr(e.status, "value") else e.status)
    d["source_links"] = list(zip(e.source_names, e.source_urls)) if len(e.source_names) == len(e.source_urls) else \
        [(n, e.source_urls[i] if i < len(e.source_urls) else "#") for i, n in enumerate(e.source_names)]
    return d


def save_edition_json(date_str: str, entries: list[DonationEntry], stats: dict, fetch_failures: list[dict]) -> Path:
    EDITIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDITIONS_DATA_DIR / f"{date_str}.json"
    payload = {
        "date": date_str,
        "entries": [e.to_dict() for e in entries],
        "stats": stats,
        "fetch_failures": fetch_failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_all_editions() -> list[dict]:
    if not EDITIONS_DATA_DIR.exists():
        return []
    editions = []
    for path in sorted(EDITIONS_DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            editions.append(json.load(f))
    return editions  # sorted ascending by filename == date


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _date_display(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%A, %B %-d, %Y") if hasattr(dt, "strftime") else date_str


def build_site(settings: dict, recipient_counts: dict, low_confidence_threshold: int) -> None:
    """Regenerates the entire static site from every edition JSON on disk.
    Idempotent — safe to run every day; always reflects the full history.
    """
    editions = load_all_editions()
    if not editions:
        print("No editions found — nothing to build yet.")
        return

    env = _env()
    site_cfg = settings["site"]
    printed_time = datetime.now(timezone.utc).strftime("%H:%M UTC")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    docs_static = DOCS_DIR / "static"
    docs_static.mkdir(exist_ok=True)
    shutil.copy(STATIC_DIR / "style.css", docs_static / "style.css")

    (DOCS_DIR / "editions").mkdir(exist_ok=True)

    edition_template = env.get_template("edition.html")
    archive_template = env.get_template("archive.html")
    sources_template = env.get_template("sources.html")

    archive_rows = []
    dated_editions = sorted(editions, key=lambda e: e["date"])

    for i, ed in enumerate(dated_editions):
        date_str = ed["date"]
        entries_view = [_entry_view_from_dict(e) for e in ed["entries"]]
        prev_url = f"../editions/{dated_editions[i-1]['date']}.html" if i > 0 else None
        next_url = f"../editions/{dated_editions[i+1]['date']}.html" if i < len(dated_editions) - 1 else None
        is_latest = i == len(dated_editions) - 1

        html = edition_template.render(
            site=site_cfg,
            asset_prefix="../",
            edition_date_display=_date_display(date_str),
            printed_time=printed_time,
            active_nav="today" if is_latest else "",
            entries=entries_view,
            stats=ed["stats"],
            prev_edition_url=prev_url,
            next_edition_url=next_url,
            is_latest=is_latest,
        )
        out_path = DOCS_DIR / "editions" / f"{date_str}.html"
        out_path.write_text(html, encoding="utf-8")

        high_conf = sum(1 for e in ed["entries"] if e["confidence_score"] >= CONFIDENCE_HIGH)
        archive_rows.append({
            "date_display": _date_display(date_str),
            "url": f"editions/{date_str}.html" if False else f"{date_str}.html",  # relative to /archive/
            "entry_count": len(ed["entries"]),
            "high_confidence_count": high_conf,
        })

    # index.html = a copy of the latest edition, but with paths relative to site root
    latest = dated_editions[-1]
    latest_entries_view = [_entry_view_from_dict(e) for e in latest["entries"]]
    prev_url = f"editions/{dated_editions[-2]['date']}.html" if len(dated_editions) > 1 else None
    index_html = edition_template.render(
        site=site_cfg,
        asset_prefix="",
        edition_date_display=_date_display(latest["date"]),
        printed_time=printed_time,
        active_nav="today",
        entries=latest_entries_view,
        stats=latest["stats"],
        prev_edition_url=prev_url,
        next_edition_url=None,
        is_latest=True,
    )
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")

    # Archive index — links need to point into ../editions/ since archive/ is a subfolder
    (DOCS_DIR / "archive").mkdir(exist_ok=True)
    archive_rows_fixed = [
        {**row, "url": f"../editions/{d['date']}.html"} for row, d in zip(archive_rows, dated_editions)
    ]
    archive_html = archive_template.render(
        site=site_cfg,
        asset_prefix="../",
        edition_date_display="",
        printed_time=printed_time,
        active_nav="archive",
        editions=list(reversed(archive_rows_fixed)),
    )
    (DOCS_DIR / "archive" / "index.html").write_text(archive_html, encoding="utf-8")

    # Sources & limitations — reflects the latest run's stats
    sources_html = sources_template.render(
        site=site_cfg,
        asset_prefix="",
        edition_date_display="",
        printed_time=printed_time,
        active_nav="sources",
        stats=latest["stats"],
        fetch_failures=latest.get("fetch_failures", []),
        recipient_counts=recipient_counts,
        low_confidence_threshold=low_confidence_threshold,
    )
    (DOCS_DIR / "sources.html").write_text(sources_html, encoding="utf-8")

    print(f"Site built: {len(dated_editions)} edition(s), latest = {latest['date']}")


def _entry_view_from_dict(e: dict) -> dict:
    d = dict(e)
    d["confidence_css_class"] = _confidence_css_class(e["confidence_score"])
    d["status_display"] = _status_display(e["status"])
    source_names = e.get("source_names", [])
    source_urls = e.get("source_urls", [])
    d["source_links"] = list(zip(source_names, source_urls)) if len(source_names) == len(source_urls) else \
        [(n, source_urls[i] if i < len(source_urls) else "#") for i, n in enumerate(source_names)]
    return d
