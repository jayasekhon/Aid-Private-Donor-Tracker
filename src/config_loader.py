"""Loads every file in /config into plain Python objects.

Deliberately defensive: these files are meant to be hand-edited by
someone without a coding background, so a malformed line should produce
a clear, specific error message rather than a stack trace.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class ConfigError(Exception):
    """Raised when a config file is malformed. Carries a human-readable message."""


@dataclass
class Recipient:
    name: str
    aliases: list[str]
    org_type: str  # UN, INGO, NGO

    @property
    def all_names(self) -> list[str]:
        return [self.name] + self.aliases


def _strip_comments_and_blanks(lines: list[str]) -> list[tuple[int, str]]:
    """Returns (line_number, text) for every non-comment, non-blank line."""
    out = []
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append((i, line))
    return out


def load_recipients(path: Path = CONFIG_DIR / "recipients.txt") -> list[Recipient]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    recipients = []
    text = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in _strip_comments_and_blanks(text):
        parts = line.split("|")
        if len(parts) != 3:
            raise ConfigError(
                f"{path.name}, line {lineno}: expected 3 parts separated by '|' "
                f"(Name | aliases | TYPE), got {len(parts)}. Line was:\n  {line}"
            )
        name, aliases_raw, org_type = (p.strip() for p in parts)
        org_type = org_type.upper()
        if org_type not in {"UN", "INGO", "NGO"}:
            raise ConfigError(
                f"{path.name}, line {lineno}: TYPE must be UN, INGO or NGO, "
                f"got '{org_type}'. Line was:\n  {line}"
            )
        aliases = [a.strip() for a in aliases_raw.split(";") if a.strip()]
        recipients.append(Recipient(name=name, aliases=aliases, org_type=org_type))
    if not recipients:
        raise ConfigError(f"{path.name} has no active (non-comment) entries.")
    return recipients


def load_countries(path: Path = CONFIG_DIR / "countries.txt") -> list[str]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    text = path.read_text(encoding="utf-8").splitlines()
    countries = [line for _, line in _strip_comments_and_blanks(text)]
    if not countries:
        raise ConfigError(f"{path.name} has no active (non-comment) entries.")
    return countries


def load_trigger_phrases(path: Path = CONFIG_DIR / "trigger_phrases.txt") -> list[str]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    text = path.read_text(encoding="utf-8").splitlines()
    phrases = [line.lower() for _, line in _strip_comments_and_blanks(text)]
    if not phrases:
        raise ConfigError(f"{path.name} has no active (non-comment) entries.")
    return phrases


def load_pr_wire_feeds(path: Path = CONFIG_DIR / "pr_wire_feeds.txt") -> list[tuple[str, str]]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    feeds = []
    text = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in _strip_comments_and_blanks(text):
        parts = line.split("|")
        if len(parts) != 2:
            raise ConfigError(
                f"{path.name}, line {lineno}: expected 'Label | URL', "
                f"got:\n  {line}"
            )
        label, url = (p.strip() for p in parts)
        feeds.append((label, url))
    return feeds  # allowed to be empty — PR wires are optional


def load_settings(path: Path = CONFIG_DIR / "settings.yaml") -> dict:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML: {e}") from e
    required_top_level = {"site", "search", "ai", "confidence", "publishing", "email"}
    missing = required_top_level - data.keys()
    if missing:
        raise ConfigError(f"{path.name} is missing required section(s): {missing}")
    return data


def load_all() -> dict:
    """Convenience loader used by the pipeline entry points.

    Fails loudly and specifically rather than letting a bad config file
    cause a confusing crash halfway through a run.
    """
    try:
        return {
            "recipients": load_recipients(),
            "countries": load_countries(),
            "triggers": load_trigger_phrases(),
            "pr_wire_feeds": load_pr_wire_feeds(),
            "settings": load_settings(),
        }
    except ConfigError as e:
        print(f"\n[CONFIG ERROR] {e}\n", file=sys.stderr)
        print("Fix the file mentioned above and run again. No data was fetched.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Running this file directly is a quick way for a non-technical person
    # to check "did I break the config files?" without running the whole pipeline.
    cfg = load_all()
    print(f"OK: {len(cfg['recipients'])} recipients, "
          f"{len(cfg['countries'])} countries, "
          f"{len(cfg['triggers'])} trigger phrases, "
          f"{len(cfg['pr_wire_feeds'])} PR wire feeds loaded successfully.")
