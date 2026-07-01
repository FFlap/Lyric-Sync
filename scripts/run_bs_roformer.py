#!/usr/bin/env python3
"""Separate vocals with BS-RoFormer and normalize them without changing timing."""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_MODEL_DIR = Path("/tmp/audio-separator-models")


def stem_paths(source: Path, work: Path) -> tuple[Path, Path]:
    del source  # BS-RoFormer produces one canonical vocal stem per song work tree.
    stem_dir = work / "stems" / "bs_roformer"
    return stem_dir / "vocals.wav", stem_dir / "vocals.normalized.wav"


def stem_is_current(source: Path, stem: Path) -> bool:
    return stem.is_file() and stem.stat().st_mtime_ns >= source.stat().st_mtime_ns


def _frame_rms(audio: np.ndarray, frame_length: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float64)
    values = []
    for start in range(0, len(audio), frame_length):
        frame = audio[start : start + frame_length]
        values.append(math.sqrt(float(np.mean(np.square(frame, dtype=np.float64)))))
    return np.asarray(values, dtype=np.float64)


def _active_rms(audio: np.ndarray, frame_length: int) -> float:
    rms = _frame_rms(audio, frame_length)
    non_silent = rms[rms > 1e-7]
    if not len(non_silent):
        return 0.0

    # Ignore silent gaps and low-level separator residue while retaining quiet vocals.
    threshold = float(np.percentile(non_silent, 30.0))
    active = non_silent[non_silent >= threshold]
    return math.sqrt(float(np.mean(np.square(active))))


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def normalize_audio(
    source: Path,
    output: Path,
    *,
    target_active_rms_db: float = -16.0,
    peak_ceiling: float = 0.95,
    frame_length: int = 2048,
) -> dict[str, float | bool]:
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    active_rms = _active_rms(audio, frame_length)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    target_rms = 10.0 ** (target_active_rms_db / 20.0)
    rms_gain = target_rms / active_rms if active_rms > 0.0 else 1.0
    peak_gain = peak_ceiling / peak if peak > 0.0 else rms_gain
    gain = min(rms_gain, peak_gain)
    normalized = np.clip(audio * gain, -peak_ceiling, peak_ceiling)

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, normalized, sample_rate, subtype="PCM_16")

    output_active_rms = _active_rms(normalized, frame_length)
    output_peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    return {
        "input_active_rms_db": _db(active_rms),
        "output_active_rms_db": _db(output_active_rms),
        "input_peak": peak,
        "output_peak": output_peak,
        "gain_db": _db(gain),
        "peak_limited": peak_gain < rms_gain,
    }


def separate(source: Path, output: Path, model_dir: Path) -> None:
    from audio_separator.separator import Separator

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="separate-") as temp:
        temp_dir = Path(temp)
        separator = Separator(
            output_dir=str(temp_dir),
            output_format="wav",
            model_file_dir=str(model_dir),
        )
        separator.load_model(MODEL_NAME)
        files = separator.separate(str(source))
        vocal_file = next(
            (temp_dir / name for name in files if "(vocals)" in name.lower()),
            None,
        )
        if vocal_file is None or not vocal_file.is_file():
            raise RuntimeError(f"BS-RoFormer did not produce a vocals stem: {files}")
        shutil.move(str(vocal_file), output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate and normalize vocals with BS-RoFormer")
    parser.add_argument("source", nargs="?", default="audio.wav")
    parser.add_argument("work", nargs="?", default="work")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    work = Path(args.work).resolve()
    raw, normalized = stem_paths(source, work)
    model_dir = Path(os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", DEFAULT_MODEL_DIR))

    if not source.is_file():
        print(f"Missing audio: {source}", file=sys.stderr)
        return 1

    try:
        if args.force or not stem_is_current(source, raw):
            print(f"==> BS-RoFormer vocal separation: {source}")
            separate(source, raw, model_dir)
        else:
            print(f"==> Reusing BS-RoFormer stem: {raw}")

        stats = normalize_audio(raw, normalized)
    except Exception as exc:
        print(f"BS-RoFormer failed: {exc}", file=sys.stderr)
        return 1

    limited = " (peak limited)" if stats["peak_limited"] else ""
    print(
        "Normalized active RMS: "
        f"{stats['input_active_rms_db']:.2f} -> {stats['output_active_rms_db']:.2f} dBFS"
        f"{limited}"
    )
    print(f"Vocals: {normalized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
