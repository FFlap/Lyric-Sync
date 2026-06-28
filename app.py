#!/usr/bin/env python3
"""Lyric Sync: library UI with per-song folders."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from clean_lyrics import clean_lyrics_text, parse_lyrics, summary
from song_data import export_song_json
from song_lib import SONGS, list_songs, make_song_id, scoop_root_outputs, song_dir, write_meta

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
STATUS_FILE = WORK / "status.json"
app = Flask(__name__)
_job_lock = threading.Lock()


def _genius_lyrics(song_url: str) -> str:
    req = urllib.request.Request(song_url, headers={"User-Agent": "Mozilla/5.0 Lyric-Sync/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Could not load the Genius lyrics page") from exc

    from html import unescape
    from html.parser import HTMLParser

    class LyricsParser(HTMLParser):
        VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

        def __init__(self):
            super().__init__()
            self.depth = 0
            self.excluded_depth = 0
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if self.depth:
                if self.excluded_depth:
                    if tag not in self.VOID_TAGS:
                        self.excluded_depth += 1
                elif "data-exclude-from-selection" in attrs:
                    self.excluded_depth = 1
                elif tag == "br":
                    self.parts.append("\n")
                if tag not in self.VOID_TAGS:
                    self.depth += 1
            elif "data-lyrics-container" in attrs:
                self.depth = 1

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)

        def handle_endtag(self, tag):
            if self.depth and tag not in self.VOID_TAGS:
                if self.excluded_depth:
                    self.excluded_depth -= 1
                self.depth -= 1
                if self.depth == 0:
                    self.parts.append("\n")

        def handle_data(self, data):
            if self.depth and not self.excluded_depth:
                self.parts.append(data)

    parser = LyricsParser()
    parser.feed(html)
    lyrics = unescape("".join(parser.parts))
    lines = [line.strip() for line in lyrics.splitlines()]
    lyrics = "\n".join(line for line in lines if line)
    if not lyrics:
        raise RuntimeError("Genius did not return readable lyrics for this song")
    return lyrics


def _youtube_search(query: str) -> list[dict]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed") from exc
    options = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"ytsearch5:{query}", download=False)
    except Exception as exc:
        raise RuntimeError("YouTube search failed") from exc
    return [
        {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "channel": item.get("channel") or item.get("uploader") or "",
            "url": item.get("url") if str(item.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={item.get('id', '')}",
            "duration": item.get("duration"),
        }
        for item in (info.get("entries") or [])
        if item and item.get("id")
    ]


def _open_path(path: Path) -> None:
    """Open a local directory in the platform's file manager."""
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-a", "Finder", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _write_status(data: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, indent=2))


def _read_status() -> dict:
    if not STATUS_FILE.exists():
        return {"state": "idle", "log": [], "progress": 0}
    try:
        return json.loads(STATUS_FILE.read_text())
    except json.JSONDecodeError:
        return {"state": "idle", "log": [], "progress": 0}


def _log(status: dict, line: str) -> None:
    status.setdefault("log", []).append(line)
    if len(status["log"]) > 400:
        status["log"] = status["log"][-400:]
    _write_status(status)


def _song_payload(song_id: str) -> dict:
    d = song_dir(song_id)
    if not d.is_dir():
        raise FileNotFoundError(song_id)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {"id": song_id}
    song_json = d / "song.json"
    if song_json.exists():
        data = json.loads(song_json.read_text())
    else:
        lyrics, _ = parse_lyrics((d / "lyrics.txt").read_text(), keep_adlibs=bool(meta.get("keep_adlibs")))
        words = json.loads((d / "hybrid.json").read_text())
        data = export_song_json(lyrics, words, song_json, meta)
    data["meta"] = meta
    data["id"] = song_id
    return data


def _run_pipeline(url: str, lyrics: str, title: str | None, keep_adlibs: bool = False) -> None:
    lines, _ = parse_lyrics(lyrics, keep_adlibs=keep_adlibs)
    preview = lines[0] if lines else "Untitled"
    song_title = (title or preview)[:80]
    song_id = make_song_id(url, song_title)
    base = song_dir(song_id)
    if base.exists():
        n = 2
        while song_dir(f"{song_id}-{n}").exists():
            n += 1
        song_id = f"{song_id}-{n}"
        base = song_dir(song_id)

    status = {
        "state": "running",
        "step": "starting",
        "progress": 2,
        "log": [],
        "error": None,
        "song_id": song_id,
    }
    _write_status(status)

    try:
        base.mkdir(parents=True, exist_ok=True)
        cleaned, stats = clean_lyrics_text(lyrics, keep_adlibs=keep_adlibs)
        if not cleaned.strip():
            raise RuntimeError("No lyric lines left after removing section headers and parentheses")
        (base / "lyrics.txt").write_text(cleaned)
        _log(status, f"Song folder: songs/{song_id}")
        _log(status, f"Saved lyrics ({summary(stats)})")

        status["step"] = "download"
        status["progress"] = 10
        _write_status(status)
        _log(status, "Downloading audio from YouTube…")

        py = sys.executable
        audio_path = base / "audio.wav"
        dl = subprocess.run(
            [py, str(ROOT / "scripts/download_youtube.py"), url, "--out", str(audio_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if dl.returncode != 0:
            raise RuntimeError(dl.stderr.strip() or dl.stdout.strip() or "YouTube download failed")
        _log(status, "Downloaded audio.wav")

        status["step"] = "sync"
        status["progress"] = 20
        _write_status(status)
        _log(status, "Running hybrid sync…")

        venv_bin = str((ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")).resolve())
        song_dir_abs = str(base.resolve())
        env = {
            **os.environ,
            "DEVICE": os.environ.get("DEVICE", "cpu"),
            "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
            "PYTHON": py,
            "FROM_APP": "1",
            "SONG_DIR": song_dir_abs,
            "AUDIO": str((base / "audio.wav").resolve()),
            "LYRICS": str((base / "lyrics.txt").resolve()),
            "WORK": str((base / "work").resolve()),
            "HYBRID": str((base / "hybrid.json").resolve()),
            "SONG_JSON": str((base / "song.json").resolve()),
            "TITLE": song_title,
            "KEEP_ADLIBS": "1" if keep_adlibs else "0",
        }
        proc = subprocess.Popen(
            [py, str(ROOT / "scripts" / "run_sync.py")],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        step_progress = 20
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _log(status, line)
                if "Demucs" in line:
                    status["step"] = "demucs"
                    step_progress = max(step_progress, 35)
                elif "WhisperX" in line:
                    status["step"] = "whisperx"
                    step_progress = max(step_progress, 55)
                elif "Acoustic" in line:
                    status["step"] = "acoustic"
                    step_progress = max(step_progress, 70)
                elif "Hybrid" in line:
                    status["step"] = "blend"
                    step_progress = max(step_progress, 85)
                elif "Song data" in line:
                    status["step"] = "song"
                    step_progress = max(step_progress, 95)
                status["progress"] = step_progress
                _write_status(status)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Sync pipeline failed. See log above.")

        if not (base / "hybrid.json").exists():
            raise RuntimeError("hybrid.json was not created")

        write_meta(
            song_id,
            {
                "id": song_id,
                "title": song_title,
                "preview": preview[:100],
                "url": url,
                "keep_adlibs": keep_adlibs,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        status["state"] = "done"
        status["step"] = "done"
        status["progress"] = 100
        status["song_id"] = song_id
        _log(status, f"Done. Playing songs/{song_id}")
        _write_status(status)
    except Exception as exc:
        status["state"] = "error"
        status["error"] = str(exc)
        _log(status, f"ERROR: {exc}")
        _log(status, traceback.format_exc())
        _write_status(status)


@app.route("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.route("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(ROOT / "assets", filename)


@app.route("/api/songs")
def api_songs():
    return jsonify(list_songs())


@app.route("/api/songs/<song_id>/open-folder", methods=["POST"])
def api_open_song_folder(song_id: str):
    try:
        d = song_dir(song_id)
    except ValueError:
        return jsonify({"error": "not found"}), 404
    if not d.is_dir():
        return jsonify({"error": "not found"}), 404
    try:
        _open_path(d)
    except OSError:
        return jsonify({"ok": False, "error": "Could not open songs folder"}), 500
    return jsonify({"ok": True})


@app.route("/api/songs/<song_id>", methods=["DELETE"])
def api_delete_song(song_id: str):
    try:
        d = song_dir(song_id)
    except ValueError:
        return jsonify({"error": "not found"}), 404
    if not d.is_dir():
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(d)
    return jsonify({"ok": True})


@app.route("/api/songs/<song_id>")
def api_song(song_id: str):
    try:
        return jsonify(_song_payload(song_id))
    except (FileNotFoundError, ValueError):
        return jsonify({"error": "not found"}), 404


@app.route("/api/songs/<song_id>/audio")
def api_song_audio(song_id: str):
    try:
        d = song_dir(song_id)
    except ValueError:
        return jsonify({"error": "not found"}), 404
    if not (d / "audio.wav").exists():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(d, "audio.wav")


@app.route("/api/status")
def api_status():
    return jsonify(_read_status())


@app.route("/api/song-search")
def api_song_search():
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Enter a song name"}), 400
    try:
        return jsonify({"youtube": _youtube_search(query)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/genius-lyrics", methods=["POST"])
def api_genius_lyrics():
    song_url = ((request.get_json(force=True, silent=True) or {}).get("url") or "").strip()
    parsed = urllib.parse.urlparse(song_url)
    if parsed.scheme != "https" or parsed.hostname not in {"genius.com", "www.genius.com"}:
        return jsonify({"error": "Invalid Genius song URL"}), 400
    try:
        return jsonify({"lyrics": _genius_lyrics(song_url)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/sync", methods=["POST"])
def api_sync():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    lyrics = (data.get("lyrics") or "").strip()
    title = (data.get("title") or "").strip() or None
    keep_adlibs = data.get("keep_adlibs") is True

    if not url:
        return jsonify({"ok": False, "error": "Paste a YouTube link"}), 400
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"ok": False, "error": "URL must be a YouTube link"}), 400
    if not lyrics:
        return jsonify({"ok": False, "error": "Paste lyrics (one line per row)"}), 400

    lines, _ = parse_lyrics(lyrics, keep_adlibs=keep_adlibs)
    if not lines:
        return jsonify({"ok": False, "error": "Lyrics are empty after cleanup"}), 400

    if not _job_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "A sync is already running"}), 409

    _write_status({"state": "queued", "progress": 0, "log": ["Queued…"], "error": None})

    def worker():
        try:
            _run_pipeline(url, lyrics, title, keep_adlibs=keep_adlibs)
        finally:
            _job_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()

    SONGS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    scooped = scoop_root_outputs()
    if scooped:
        print(f"Moved stray root files → songs/{scooped}/")
    print(f"Lyric Sync → http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
