#!/usr/bin/env python3
"""Strip Genius-style section headers and parenthetical ad-libs from pasted lyrics."""
from __future__ import annotations

import os
import re
from typing import TypedDict

_SECTION = re.compile(r"^\s*\[.+\]\s*$")
_ONLY_PARENS = re.compile(r"^\s*\([^)]*\)\s*$")
_INLINE_PARENS = re.compile(r"\([^)]*\)")


class CleanStats(TypedDict):
    removed_sections: int
    removed_paren_lines: int
    stripped_inline: int


def parse_lyrics(text: str, keep_adlibs: bool | None = None) -> tuple[list[str], CleanStats]:
    """Return sung lyric lines and counts of what was removed."""
    if keep_adlibs is None:
        keep_adlibs = os.environ.get("KEEP_ADLIBS") == "1"
    stats: CleanStats = {
        "removed_sections": 0,
        "removed_paren_lines": 0,
        "stripped_inline": 0,
    }
    lines: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SECTION.match(line):
            stats["removed_sections"] += 1
            continue
        if not keep_adlibs and _ONLY_PARENS.match(line):
            stats["removed_paren_lines"] += 1
            continue
        if not keep_adlibs and _INLINE_PARENS.search(line):
            stats["stripped_inline"] += 1
        cleaned = line if keep_adlibs else _INLINE_PARENS.sub("", line).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            lines.append(cleaned)

    return lines, stats


def format_lyrics(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def clean_lyrics_text(text: str, keep_adlibs: bool | None = None) -> tuple[str, CleanStats]:
    lines, stats = parse_lyrics(text, keep_adlibs=keep_adlibs)
    return format_lyrics(lines), stats


def summary(stats: CleanStats) -> str:
    parts: list[str] = []
    if stats["removed_sections"]:
        parts.append(f"{stats['removed_sections']} section header(s)")
    if stats["removed_paren_lines"]:
        parts.append(f"{stats['removed_paren_lines']} parenthetical line(s)")
    if stats["stripped_inline"]:
        parts.append(f"{stats['stripped_inline']} line(s) with inline (…) removed")
    return ", ".join(parts) if parts else "no section headers or parentheses found"


if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    ap = argparse.ArgumentParser(description="Clean pasted lyrics for sync")
    ap.add_argument("file", nargs="?", help="Input file (default: stdin)")
    ap.add_argument("--stats", action="store_true", help="Print cleanup stats to stderr")
    args = ap.parse_args()

    raw = pathlib.Path(args.file).read_text() if args.file else sys.stdin.read()
    cleaned, stats = clean_lyrics_text(raw)
    if args.stats:
        print(summary(stats), file=sys.stderr)
    sys.stdout.write(cleaned)
