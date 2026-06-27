#!/usr/bin/env python3
"""Download YouTube audio as WAV for the sync pipeline."""
import argparse
import pathlib
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube URL")
    ap.add_argument("--out", default="audio.wav", help="Output WAV path")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    template = str(out.with_suffix("")) + ".%(ext)s"

    if not shutil.which("ffmpeg"):
        from platform_utils import ffmpeg_hint

        raise SystemExit(f"ffmpeg not found. {ffmpeg_hint()}")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        template,
        "--no-playlist",
        args.url,
    ]

    print("Downloading:", args.url, file=sys.stderr)
    subprocess.run(cmd, check=True)

    wav = out.with_suffix(".wav")
    if wav != out and wav.exists():
        wav.replace(out)
    if not out.exists():
        raise SystemExit(f"Download failed. Expected {out}")

    print(str(out))


if __name__ == "__main__":
    main()
