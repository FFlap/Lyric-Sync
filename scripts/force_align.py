#!/usr/bin/env python3
"""WhisperX forced alignment: full-song + per-line windowed (inputs to hybrid blend)."""
import argparse
import csv
import json
import pathlib
import re

import whisperx

from clean_lyrics import parse_lyrics

MODEL_NAME = "WAV2VEC2_ASR_LARGE_LV60K_960H"


def token_count(line):
    return len(re.findall(r"\S+", line))


def clean_word(w, i):
    return {
        "i": i,
        "word": str(w.get("word", "")),
        "start": round(float(w["start"]), 3),
        "end": round(float(w["end"]), 3),
        "score": round(float(w.get("score", 0.0)), 3),
    }


def write_words(outdir, stem, words):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stem}.json").write_text(json.dumps(words, indent=2))
    with open(outdir / f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["i", "word", "start", "end", "score"])
        wr.writeheader()
        wr.writerows(words)


def force_align_full(audio_path, lyrics_text, device):
    audio = whisperx.load_audio(audio_path)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device, model_name=MODEL_NAME)
    duration = len(audio) / 16000.0
    segment = {"start": 0.0, "end": duration, "text": lyrics_text}
    result = whisperx.align([segment], model_a, metadata, audio, device, return_char_alignments=False)
    words = [
        clean_word(w, i + 1)
        for i, w in enumerate(result.get("word_segments", []))
        if "start" in w and "end" in w
    ]
    return words, audio, model_a, metadata


def line_windowed_align(audio, model_a, metadata, lines, base_words, device, pad):
    windows = []
    pos = 0
    duration = len(audio) / 16000.0
    for line in lines:
        count = token_count(line)
        chunk = base_words[pos : pos + count]
        pos += count
        if chunk:
            start = max(0.0, chunk[0]["start"] - pad)
            end = min(duration, chunk[-1]["end"] + pad)
        else:
            start, end = 0.0, duration
        windows.append({"start": start, "end": end, "text": line})
    result = whisperx.align(windows, model_a, metadata, audio, device, return_char_alignments=False)
    words = [
        clean_word(w, i + 1)
        for i, w in enumerate(result.get("word_segments", []))
        if "start" in w and "end" in w
    ]
    return words, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="work/stems/bs_roformer/vocals.normalized.wav")
    ap.add_argument("--lyrics", default="lyrics.txt")
    ap.add_argument("--out", default="work/align")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pad", type=float, default=1.2)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    lines, _ = parse_lyrics(pathlib.Path(args.lyrics).read_text())
    if not lines:
        raise SystemExit(f"No lyric lines in {args.lyrics}")
    lyrics_text = " ".join(lines)

    full_words, audio, model_a, metadata = force_align_full(args.audio, lyrics_text, args.device)
    write_words(outdir, "whisperx_full", full_words)

    line_words, windows = line_windowed_align(audio, model_a, metadata, lines, full_words, args.device, args.pad)
    write_words(outdir, "whisperx_line_windowed", line_words)
    (outdir / "line_windows.json").write_text(json.dumps(windows, indent=2))

    print(
        json.dumps(
            {
                "full_words": len(full_words),
                "line_windowed_words": len(line_words),
                "lines": len(lines),
                "out": str(outdir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
