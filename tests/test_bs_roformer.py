#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bs_roformer import normalize_audio, stem_is_current, stem_paths
from run_sync import vocals_path


class BsRoformerTests(unittest.TestCase):
    def test_stem_paths_use_normalized_bs_roformer_audio(self):
        raw, normalized = stem_paths(Path("song.wav"), Path("work"))

        self.assertEqual(raw, Path("work/stems/bs_roformer/vocals.wav"))
        self.assertEqual(
            normalized, Path("work/stems/bs_roformer/vocals.normalized.wav")
        )

    def test_sync_pipeline_uses_normalized_bs_roformer_audio(self):
        self.assertEqual(
            vocals_path(Path("work")),
            Path("work/stems/bs_roformer/vocals.normalized.wav"),
        )

    def test_stem_is_rebuilt_when_source_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            stem = Path(tmp) / "vocals.wav"
            source.touch()
            stem.touch()
            os.utime(stem, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))

            self.assertFalse(stem_is_current(source, stem))
            os.utime(stem, ns=(3_000_000_000, 3_000_000_000))
            self.assertTrue(stem_is_current(source, stem))
            stem.unlink()
            self.assertFalse(stem_is_current(source, stem))

    def test_normalization_preserves_sample_count_and_reaches_active_rms_target(self):
        sample_rate = 16000
        silence = np.zeros((sample_rate, 2), dtype=np.float32)
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        tone = (0.05 * np.sin(2 * np.pi * 220 * t))[:, None]
        audio = np.concatenate([silence, np.repeat(tone, 2, axis=1)])

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            output = Path(tmp) / "normalized.wav"
            sf.write(source, audio, sample_rate, subtype="FLOAT")

            stats = normalize_audio(source, output, target_active_rms_db=-16.0)
            normalized, output_rate = sf.read(output, always_2d=True)

        self.assertEqual(output_rate, sample_rate)
        self.assertEqual(normalized.shape, audio.shape)
        self.assertAlmostEqual(stats["output_active_rms_db"], -16.0, delta=0.15)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 0.9501)

    def test_normalization_respects_peak_ceiling(self):
        sample_rate = 8000
        audio = np.full((sample_rate, 1), 0.02, dtype=np.float32)
        audio[sample_rate // 2, 0] = 0.9

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            output = Path(tmp) / "normalized.wav"
            sf.write(source, audio, sample_rate, subtype="FLOAT")

            stats = normalize_audio(
                source,
                output,
                target_active_rms_db=-10.0,
                peak_ceiling=0.95,
            )
            normalized, _ = sf.read(output, always_2d=True)

        self.assertLessEqual(float(np.max(np.abs(normalized))), 0.9501)
        self.assertTrue(stats["peak_limited"])


if __name__ == "__main__":
    unittest.main()
