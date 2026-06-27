#!/usr/bin/env python3
"""Build song.json from hybrid word timings and lyric lines."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clean_lyrics import parse_lyrics


def token_count(line: str) -> int:
    return len(re.findall(r"\S+", line))


def build_lines(words: list[dict], lyric_lines: list[str]) -> list[dict]:
    lines = []
    pos = 0
    for idx, text in enumerate(lyric_lines, 1):
        count = token_count(text)
        chunk = words[pos : pos + count]
        pos += count
        if not chunk:
            continue
        start = float(chunk[0]["start"])
        if lines:
            start = max(start, float(lines[-1]["end"]))
        lines.append(
            {
                "i": idx,
                "text": text,
                "start": round(start, 3),
                "end": chunk[-1]["end"],
                "words": chunk,
            }
        )
    return lines


def export_song_json(
    lyric_lines: list[str],
    words: list[dict],
    out_path: Path,
    meta: dict | None = None,
) -> dict:
    lines = build_lines(words, lyric_lines)
    payload = {
        "words": words,
        "lines": lines,
        "meta": meta or {},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build song.json for the Lyric Sync player")
    ap.add_argument("--lyrics", required=True, help="Lyrics text file")
    ap.add_argument("--words", required=True, help="Hybrid word timings JSON")
    ap.add_argument("--out", required=True, help="Output song.json path")
    ap.add_argument("--title", default="Lyric Sync")
    args = ap.parse_args()

    lyrics_path = Path(args.lyrics).resolve()
    words_path = Path(args.words).resolve()
    out_path = Path(args.out).resolve()

    lyric_lines, _ = parse_lyrics(lyrics_path.read_text())
    if not lyric_lines:
        print(f"No lyric lines in {lyrics_path}", file=sys.stderr)
        return 1

    words = json.loads(words_path.read_text())
    if not words:
        print(f"No words in {words_path}", file=sys.stderr)
        return 1

    export_song_json(lyric_lines, words, out_path, {"title": args.title})
    print(json.dumps({"words": len(words), "lines": len(lyric_lines), "song_json": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
