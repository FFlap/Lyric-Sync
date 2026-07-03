#!/usr/bin/env python3
"""Audio-consistency check of word timings against the vocal stem.

Flags words highlighted over silence, words starting in dead air, sung audio
no word covers, and word starts far from any acoustic articulation evidence.
Run after a sync as a QA step:

    python scripts/verify_alignment.py songs/<song-dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("song_dir")
    ap.add_argument("--words", default=None, help="words JSON (default: <song>/hybrid.json)")
    ap.add_argument("--audio", default=None, help="vocal stem (default: bs_roformer stem)")
    ap.add_argument("--worst", type=int, default=8)
    args = ap.parse_args()

    import librosa

    from align_words import SR, AudioFeatures

    song = pathlib.Path(args.song_dir)
    words_path = pathlib.Path(args.words) if args.words else song / "hybrid.json"
    audio_path = (
        pathlib.Path(args.audio)
        if args.audio
        else song / "work/stems/bs_roformer/vocals.normalized.wav"
    )
    words = [w for w in json.loads(words_path.read_text()) if w.get("source") != "punct"]
    if not words:
        print("no words", file=sys.stderr)
        return 1
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    feats = AudioFeatures(y)

    def voiced_frac(a: float, b: float) -> float:
        i, j = feats.t2f(a), feats.t2f(b)
        if j <= i:
            return 1.0
        return float(np.mean(feats.voiced[i:j]))

    silent = [
        w
        for w in words
        if w["end"] - w["start"] >= 0.12 and voiced_frac(w["start"], w["end"]) < 0.15
    ]

    dead_start = [
        w
        for w in words
        if voiced_frac(w["start"], w["start"] + 0.08) <= 0.2
        and voiced_frac(w["start"] + 0.08, w["start"] + 0.35) <= 0.3
    ]

    spans = sorted((w["start"], w["end"]) for w in words)
    uncovered = []
    for v0, v1 in feats.voiced_intervals(0.0, feats.duration, min_len=0.6):
        cover = sum(max(0.0, min(b, v1) - max(a, v0)) for a, b in spans)
        if cover / (v1 - v0) < 0.30:
            uncovered.append((round(v0, 2), round(v1, 2)))

    cands = np.array(sorted(set(list(feats.onsets) + [t for t, _ in feats.novelty])))
    dists = np.array(
        [float(np.min(np.abs(cands - w["start"]))) if len(cands) else 99.0 for w in words]
    )
    unsupported = [(d, w) for d, w in zip(dists, words) if d > 0.30]

    print(f"{song.name}: {len(words)} words")
    print(
        f"  attack support: median {np.median(dists):.3f}s, P90 {np.percentile(dists, 90):.3f}s,"
        f" >0.3s: {len(unsupported)} ({len(unsupported) / len(words) * 100:.1f}%)"
    )
    print(f"  silent words: {len(silent)}")
    for w in silent[: args.worst]:
        print(f"    {w['word']!r} {w['start']}-{w['end']} [{w['source']}]")
    print(f"  dead-air starts: {len(dead_start)}")
    for w in dead_start[: args.worst]:
        print(f"    {w['word']!r} {w['start']}-{w['end']} [{w['source']}]")
    print(f"  uncovered vocal stretches: {len(uncovered)} (vocals absent from lyrics are expected here)")
    for v in uncovered[: args.worst]:
        print(f"    {v[0]}-{v[1]}s")
    for d, w in sorted(unsupported, reverse=True, key=lambda x: x[0])[: args.worst]:
        print(f"  weak start: {d:.2f}s off  {w['word']!r} {w['start']}-{w['end']} [{w['source']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
