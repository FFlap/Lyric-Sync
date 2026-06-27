#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from blend_alignments import blend


class BlendSmokeTests(unittest.TestCase):
    def test_blend_returns_one_timing_per_lyric_word(self):
        acoustic = [
            {"i": 1, "word": "one", "start": 0.0, "end": 0.4},
            {"i": 2, "word": "two", "start": 0.5, "end": 0.9},
            {"i": 3, "word": "three", "start": 1.0, "end": 1.4},
        ]
        line_windowed = [
            {"i": 1, "word": "one", "start": 0.05, "end": 0.45},
            {"i": 2, "word": "two", "start": 0.55, "end": 0.95},
            {"i": 3, "word": "three", "start": 1.05, "end": 1.45},
        ]

        words, _ = blend(acoustic, line_windowed, ["one two", "three"])

        self.assertEqual(len(words), 3)
        self.assertEqual([w["word"] for w in words], ["one", "two", "three"])

    def test_blend_keeps_words_in_time_order(self):
        acoustic = [
            {"i": 1, "word": "a", "start": 0.0, "end": 0.3},
            {"i": 2, "word": "b", "start": 0.4, "end": 0.7},
        ]
        line_windowed = [
            {"i": 1, "word": "a", "start": 0.0, "end": 0.3},
            {"i": 2, "word": "b", "start": 0.4, "end": 0.7},
        ]

        words, _ = blend(acoustic, line_windowed, ["a b"])

        for prev, cur in zip(words, words[1:]):
            self.assertLessEqual(prev["start"], cur["start"])


if __name__ == "__main__":
    unittest.main()
