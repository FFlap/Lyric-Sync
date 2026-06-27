#!/usr/bin/env python3
"""Song library helpers for per-song folders under songs/."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SONGS = ROOT / "songs"


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return (s[:max_len] or "song").strip("-")


def youtube_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{6,})", url)
    return m.group(1) if m else None


def make_song_id(url: str, title: str | None = None) -> str:
    vid = youtube_id(url) or "clip"
    base = slugify(title) if title else "song"
    return f"{base}-{vid[:11]}"


def song_dir(song_id: str) -> Path:
    candidate = (SONGS / song_id).resolve()
    if candidate.parent != SONGS.resolve():
        raise ValueError(f"Invalid song id: {song_id!r}")
    return candidate


def meta_path(song_id: str) -> Path:
    return song_dir(song_id) / "meta.json"


def read_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_meta(song_id: str, meta: dict) -> None:
    d = song_dir(song_id)
    d.mkdir(parents=True, exist_ok=True)
    meta_path(song_id).write_text(json.dumps(meta, indent=2))


def list_songs() -> list[dict]:
    if not SONGS.exists():
        return []
    out = []
    for d in sorted(SONGS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = read_meta(d / "meta.json")
        if not meta and (d / "hybrid.json").exists():
            meta = {"id": d.name, "title": d.name, "created_at": d.stat().st_mtime}
        if meta:
            meta.setdefault("id", d.name)
            out.append(meta)
    return out


def migrate_root_song() -> str | None:
    """Move legacy root outputs into songs/ if present."""
    audio = ROOT / "audio.wav"
    hybrid = ROOT / "hybrid.json"
    if not audio.exists() or not hybrid.exists():
        return None
    SONGS.mkdir(parents=True, exist_ok=True)
    sid = "imported-song"
    n = 1
    while song_dir(sid).exists():
        sid = f"imported-song-{n}"
        n += 1
    dest = song_dir(sid)
    dest.mkdir(parents=True)
    for name in ("audio.wav", "hybrid.json", "lyrics.txt", "hybrid_picks.json", "song.json"):
        src = ROOT / name
        if src.exists():
            shutil.move(str(src), str(dest / name))
    if (ROOT / "work").exists():
        shutil.move(str(ROOT / "work"), str(dest / "work"))
    lyrics = ""
    lp = dest / "lyrics.txt"
    if lp.exists():
        lyrics = lp.read_text()
    first = next((ln.strip() for ln in lyrics.splitlines() if ln.strip()), sid)
    write_meta(
        sid,
        {
            "id": sid,
            "title": first[:60],
            "preview": first[:80],
            "url": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return sid


def scoop_root_outputs() -> str | None:
    """Move stray root-level sync outputs into songs/ (e.g. old server run)."""
    return migrate_root_song()
