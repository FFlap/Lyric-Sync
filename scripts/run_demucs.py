#!/usr/bin/env python3
"""Demucs vocal separation (cross-platform)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_utils import ROOT, augmented_path_env, demucs_executable, run_checked


def main() -> int:
    ap = argparse.ArgumentParser(description="Separate vocals with Demucs htdemucs")
    ap.add_argument("source", nargs="?", default="audio.wav", help="Input audio file")
    ap.add_argument("work", nargs="?", default="work", help="Work directory")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    work = Path(args.work).resolve()
    basename = source.stem
    vocals = work / "stems" / "htdemucs" / basename / "vocals.wav"

    if not source.is_file():
        print(f"Missing audio: {source}", file=sys.stderr)
        return 1

    try:
        demucs = demucs_executable()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"==> Demucs htdemucs: {source}")
    stems_dir = work / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    run_checked(
        [demucs, "--two-stems", "vocals", "-n", "htdemucs", "-o", str(stems_dir), str(source)],
        env=augmented_path_env(),
    )

    if not vocals.is_file():
        print(f"Expected vocals at {vocals}", file=sys.stderr)
        return 1

    print(f"Vocals: {vocals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
