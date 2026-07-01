#!/usr/bin/env python3
"""BS-RoFormer → WhisperX → acoustic refine → hybrid blend → song.json."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from platform_utils import ROOT, augmented_path_env, project_python, run_python_script


def _env_path(name: str, default: str = "") -> Path | None:
    value = os.environ.get(name, default).strip()
    return Path(value).resolve() if value else None


def resolve_paths() -> dict[str, Path]:
    song_dir = _env_path("SONG_DIR")
    if song_dir:
        song_dir.mkdir(parents=True, exist_ok=True)
        audio = _env_path("AUDIO") or (song_dir / "audio.wav")
        lyrics = _env_path("LYRICS") or (song_dir / "lyrics.txt")
        work = _env_path("WORK") or (song_dir / "work")
        hybrid = _env_path("HYBRID") or (song_dir / "hybrid.json")
        song_json = _env_path("SONG_JSON") or (song_dir / "song.json")
    else:
        audio = _env_path("AUDIO") or (ROOT / "audio.wav")
        lyrics = _env_path("LYRICS") or (ROOT / "lyrics.txt")
        work = _env_path("WORK") or (ROOT / "work")
        hybrid = _env_path("HYBRID") or (ROOT / "hybrid.json")
        song_json = _env_path("SONG_JSON") or (ROOT / "song.json")

    if os.environ.get("FROM_APP") == "1" and not song_dir:
        print("FROM_APP requires SONG_DIR", file=sys.stderr)
        raise SystemExit(1)

    return {
        "audio": audio,
        "lyrics": lyrics,
        "work": work,
        "hybrid": hybrid,
        "song_json": song_json,
    }


def vocals_path(work: Path) -> Path:
    return work / "stems" / "bs_roformer" / "vocals.normalized.wav"


def main() -> int:
    paths = resolve_paths()
    audio = paths["audio"]
    lyrics = paths["lyrics"]
    work = paths["work"]
    hybrid = paths["hybrid"]
    song_json = paths["song_json"]

    device = os.environ.get("DEVICE", "cpu")
    title = os.environ.get("TITLE", "Lyric Sync")

    vocals = vocals_path(work)
    env = augmented_path_env()

    print(f"==> Output folder: {hybrid.parent}")

    if not lyrics.is_file():
        print(f"Missing lyrics file: {lyrics}", file=sys.stderr)
        return 1
    if not audio.is_file():
        print(f"Missing audio file: {audio}", file=sys.stderr)
        return 1

    (work / "align").mkdir(parents=True, exist_ok=True)

    if os.environ.get("SKIP_SEPARATION") != "1":
        print("==> BS-RoFormer vocal separation and normalization")
        py = project_python()
        proc = subprocess.run(
            [str(py), str(ROOT / "scripts" / "run_bs_roformer.py"), str(audio), str(work)],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return proc.returncode
    else:
        print("==> BS-RoFormer skipped (SKIP_SEPARATION=1)")

    if not vocals.is_file():
        print(f"Missing vocals stem: {vocals}", file=sys.stderr)
        return 1

    print("==> WhisperX forced alignment (vocals)")
    run_python_script(
        "force_align.py",
        ["--audio", str(vocals), "--lyrics", str(lyrics), "--out", str(work / "align"), "--device", device],
        env=env,
    )

    print("==> Acoustic boundary refinement")
    run_python_script(
        "refine_boundaries.py",
        [
            "--words",
            str(work / "align" / "whisperx_full.json"),
            "--audio",
            str(vocals),
            "--out",
            str(work / "acoustic_refined.json"),
        ],
        env=env,
    )

    print("==> Hybrid blend")
    run_python_script(
        "blend_alignments.py",
        [
            "--acoustic",
            str(work / "acoustic_refined.json"),
            "--line-windowed",
            str(work / "align" / "whisperx_line_windowed.json"),
            "--lyrics",
            str(lyrics),
            "--audio",
            str(vocals),
            "--out",
            str(hybrid),
        ],
        env=env,
    )

    print("==> Song data")
    run_python_script(
        "song_data.py",
        [
            "--lyrics",
            str(lyrics),
            "--words",
            str(hybrid),
            "--out",
            str(song_json),
            "--title",
            title,
        ],
        env=env,
    )

    print("")
    print("Done.")
    print(f"  {hybrid}")
    print(f"  {song_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
