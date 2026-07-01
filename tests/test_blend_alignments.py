#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from blend_alignments import (
    apply_conservative_line_anchor,
    apply_supported_line_anchors,
    apply_lw_stride_gaps,
    blend,
    conservative_line_anchor,
    min_duration,
    optimize_line_boundaries,
    prefer_line_windowed,
    refine_line_openers,
    refine_absorbed_successors,
    shift_line_to_early_anchor,
)


class BlendSmokeTests(unittest.TestCase):
    @patch("blend_alignments.vowel_short_attack_start", return_value=2.25)
    def test_absorbed_short_word_rejects_late_attack_jump(self, _):
        words = [
            {"word": "good", "start": 1.0, "end": 2.0, "source": "acoustic"},
            {"word": "I'm", "start": 1.9, "end": 2.4, "source": "acoustic"},
        ]
        acoustic = [
            {"word": "good", "start": 1.0, "end": 2.0, "score": 0.7},
            {"word": "I'm", "start": 2.0, "end": 2.3, "score": 0.2},
        ]
        line_windowed = [
            {"word": "good", "start": 1.0, "end": 1.5, "score": 0.7},
            {"word": "I'm", "start": 1.6, "end": 1.8, "score": 0.2},
        ]

        refine_absorbed_successors(words, acoustic, line_windowed, [0.0], 16000)

        self.assertEqual(1.9, words[1]["start"])

    def test_lw_gap_does_not_mix_timelines_several_seconds_apart(self):
        chunk = [
            {"word": "Baby", "start": 5.0, "end": 10.4, "source": "acoustic"},
            {"word": "on", "start": 10.4, "end": 10.8, "source": "acoustic"},
        ]
        acoustic = [
            {"word": "Baby", "start": 10.0, "end": 10.4, "score": 0.7},
            {"word": "on", "start": 10.4, "end": 10.8, "score": 0.7},
        ]
        line_windowed = [
            {"word": "Baby", "start": 5.0, "end": 5.2, "score": 0.7},
            {"word": "on", "start": 5.3, "end": 5.5, "score": 0.7},
        ]

        apply_lw_stride_gaps(chunk, acoustic, line_windowed)

        self.assertEqual(10.4, chunk[0]["end"])
        self.assertEqual(10.4, chunk[1]["start"])

    @patch("blend_alignments.conservative_line_anchor", return_value=4.0)
    @patch("blend_alignments.local_onsets", return_value=[4.0])
    def test_supported_anchor_cannot_rewind_previous_line_by_seconds(self, _, __):
        previous = [
            {"word": "Baby", "start": 2.0, "end": 3.0, "source": "acoustic"},
            {"word": "on", "start": 3.0, "end": 6.0, "source": "acoustic"},
        ]
        current = [
            {"word": "Baby", "start": 7.0, "end": 7.5, "source": "acoustic"},
            {"word": "on", "start": 7.5, "end": 8.0, "source": "acoustic"},
        ]
        previous_acoustic = [dict(word, score=0.7) for word in previous]
        current_acoustic = [dict(word, score=0.7) for word in current]
        previous_windowed = [dict(word, score=0.7) for word in previous]
        current_windowed = [dict(word, score=0.7) for word in current]

        apply_supported_line_anchors(
            [
                (previous, previous_acoustic, previous_windowed),
                (current, current_acoustic, current_windowed),
            ],
            [0.0],
            16000,
        )

        self.assertEqual(6.0, previous[-1]["end"])
        self.assertEqual(7.0, current[0]["start"])

    @patch("blend_alignments.best_phrase_onset", return_value=1.4)
    @patch("blend_alignments.onset_anchors_ac_start", return_value=(True, 2.5))
    def test_line_opener_can_use_strong_earlier_onset_despite_late_acoustic_anchor(self, _, __):
        words = [
            {"word": "yeah", "start": 0.6, "end": 1.0, "source": "acoustic"},
            {"word": "I", "start": 2.5, "end": 2.8, "source": "acoustic"},
            {"word": "got", "start": 2.8, "end": 3.1, "source": "acoustic"},
        ]
        acoustic = [
            {"word": "yeah", "start": 0.6, "end": 1.0, "score": 0.7},
            {"word": "I", "start": 2.5, "end": 2.8, "score": 0.7},
            {"word": "got", "start": 2.8, "end": 3.1, "score": 0.7},
        ]
        line_windowed = [
            {"word": "yeah", "start": 0.6, "end": 1.0, "score": 0.7},
            {"word": "I", "start": 2.2, "end": 2.5, "score": 0.7},
            {"word": "got", "start": 2.8, "end": 3.1, "score": 0.7},
        ]

        refine_line_openers(
            words,
            acoustic,
            line_windowed,
            ["yeah", "I got"],
            [0.0],
            16000,
        )

        self.assertEqual(1.4, words[1]["start"])
        self.assertEqual(1.7, words[1]["end"])

    @patch("blend_alignments.has_line_windowed_early_drift", return_value=False)
    def test_blend_defaults_to_conservative_mode_without_global_drift(self, _):
        acoustic = [
            {
                "i": i + 1,
                "word": f"word{i}",
                "start": 10.0 + i * 0.5,
                "end": 10.35 + i * 0.5,
                "score": 0.6,
            }
            for i in range(8)
        ]
        line_windowed = [
            {
                "i": i + 1,
                "word": f"word{i}",
                "start": 9.3 + i * 0.5,
                "end": 9.4 + i * 0.5,
                "score": 0.58,
            }
            for i in range(8)
        ]

        words, _ = blend(
            acoustic,
            line_windowed,
            [" ".join(f"word{i}" for i in range(8))],
        )

        self.assertEqual(10.0, words[0]["start"])
        self.assertTrue(all("line_shift" not in word["source"] for word in words))

    def test_distant_later_window_does_not_replace_low_confidence_acoustic_word(self):
        acoustic = {"word": "eye", "start": 1.0, "end": 1.1, "score": 0.02}
        windowed = {"word": "eye", "start": 1.44, "end": 1.58, "score": 0.43}
        previous = {"word": "her", "start": 0.8, "end": 1.0, "score": 0.5}

        self.assertFalse(prefer_line_windowed(acoustic, windowed, previous, None))

    def test_far_later_window_does_not_replace_tightly_joined_word(self):
        previous = {"word": "real", "start": 1.0, "end": 1.3, "score": 0.55}
        acoustic = {"word": "good", "start": 1.3, "end": 1.55, "score": 0.35}
        windowed = {"word": "good", "start": 1.95, "end": 2.25, "score": 0.55}

        self.assertFalse(prefer_line_windowed(acoustic, windowed, previous, None))

    def test_clean_acoustic_boundary_is_not_pulled_toward_late_window(self):
        chunk = [
            {"i": 1, "word": "That's", "start": 1.0, "end": 1.4, "score": 0.46, "source": "acoustic"},
            {"i": 2, "word": "cute", "start": 1.4, "end": 2.0, "score": 0.57, "source": "acoustic"},
        ]
        acoustic = [
            {"i": 1, "word": "That's", "start": 1.0, "end": 1.4, "score": 0.46},
            {"i": 2, "word": "cute", "start": 1.4, "end": 2.0, "score": 0.57},
        ]
        line_windowed = [
            {"i": 1, "word": "That's", "start": 1.35, "end": 2.3, "score": 0.37},
            {"i": 2, "word": "cute", "start": 2.32, "end": 2.45, "score": 0.25},
        ]

        optimize_line_boundaries(chunk, acoustic, line_windowed)

        self.assertEqual(chunk[0]["end"], 1.4)
        self.assertEqual(chunk[1]["start"], 1.4)

    def test_rejects_line_opener_that_reverses_line_window_order(self):
        acoustic = [
            {"i": 1, "word": "tail", "start": 1.0, "end": 1.5, "score": 0.8},
            {"i": 2, "word": "My", "start": 1.7, "end": 2.1, "score": 0.8},
            {"i": 3, "word": "strategy", "start": 2.1, "end": 2.6, "score": 0.7},
        ]
        line_windowed = [
            {"i": 1, "word": "tail", "start": 1.0, "end": 1.5, "score": 0.8},
            {"i": 2, "word": "My", "start": 0.9, "end": 1.2, "score": 0.75},
            {"i": 3, "word": "strategy", "start": 1.2, "end": 2.5, "score": 0.7},
        ]

        words, _ = blend(acoustic, line_windowed, ["tail", "My strategy"])

        self.assertNotIn("line_windowed", words[1]["source"])
        self.assertGreaterEqual(words[1]["start"], 1.5)

    def test_audio_supported_ordered_opener_can_override_global_drift(self):
        previous_lw = [
            {"i": 1, "word": "loaded", "start": 2.8, "end": 3.5, "score": 0.6},
            {"i": 2, "word": "shotgun", "start": 3.5, "end": 4.0, "score": 0.55},
        ]
        acoustic = [
            {"i": 3, "word": "Ready", "start": 5.0, "end": 5.8, "score": 0.8},
            {"i": 4, "word": "to", "start": 5.8, "end": 6.1, "score": 0.7},
            {"i": 5, "word": "fire", "start": 6.1, "end": 6.5, "score": 0.6},
            {"i": 6, "word": "now", "start": 6.5, "end": 6.9, "score": 0.6},
        ]
        line_windowed = [
            {"i": 3, "word": "Ready", "start": 4.3, "end": 5.3, "score": 0.74},
            {"i": 4, "word": "to", "start": 5.6, "end": 5.9, "score": 0.7},
            {"i": 5, "word": "fire", "start": 6.08, "end": 6.4, "score": 0.6},
            {"i": 6, "word": "now", "start": 6.45, "end": 6.8, "score": 0.6},
        ]

        anchor = conservative_line_anchor(
            acoustic,
            line_windowed,
            previous_lw,
            onset_times=[4.26, 4.32],
        )

        self.assertEqual(anchor, 4.3)

    def test_uniform_early_alternate_timeline_is_not_a_boundary_anchor(self):
        previous_lw = [
            {"i": 1, "word": "tail", "start": 3.2, "end": 4.0, "score": 0.7},
        ]
        acoustic = [
            {"i": i + 2, "word": word, "start": 5.0 + i * 0.4, "end": 5.3 + i * 0.4, "score": 0.75}
            for i, word in enumerate(["Ready", "steady", "target", "again"])
        ]
        line_windowed = [
            dict(word, start=word["start"] - 0.7, end=word["end"] - 0.7, score=0.72)
            for word in acoustic
        ]

        anchor = conservative_line_anchor(
            acoustic,
            line_windowed,
            previous_lw,
            onset_times=[4.28, 4.32],
        )

        self.assertIsNone(anchor)

    def test_short_ambiguous_opener_is_not_used_as_a_line_anchor(self):
        previous_lw = [{"i": 1, "word": "tail", "start": 3.0, "end": 4.0, "score": 0.7}]
        acoustic = [
            {"i": 2, "word": "It", "start": 5.0, "end": 5.3, "score": 0.7},
            {"i": 3, "word": "went", "start": 5.3, "end": 5.7, "score": 0.7},
            {"i": 4, "word": "past", "start": 5.7, "end": 6.1, "score": 0.7},
        ]
        line_windowed = [
            {"i": 2, "word": "It", "start": 4.3, "end": 4.6, "score": 0.7},
            {"i": 3, "word": "went", "start": 5.25, "end": 5.6, "score": 0.7},
            {"i": 4, "word": "past", "start": 5.68, "end": 6.0, "score": 0.7},
        ]

        anchor = conservative_line_anchor(
            acoustic,
            line_windowed,
            previous_lw,
            onset_times=[4.28],
        )

        self.assertIsNone(anchor)

    def test_line_anchor_preserves_order_and_minimum_durations(self):
        previous = [
            {"i": 1, "word": "loaded", "start": 3.78, "end": 4.5, "score": 0.6, "source": "acoustic"},
            {"i": 2, "word": "shotgun", "start": 4.5, "end": 5.0, "score": 0.55, "source": "acoustic"},
        ]
        previous_acoustic = [dict(word) for word in previous]
        previous_windowed = [
            {"i": 1, "word": "loaded", "start": 2.8, "end": 3.5, "score": 0.6},
            {"i": 2, "word": "shotgun", "start": 3.5, "end": 4.0, "score": 0.55},
        ]
        current = [
            {"i": 3, "word": "Ready", "start": 5.0, "end": 5.8, "score": 0.8, "source": "acoustic"},
            {"i": 4, "word": "to", "start": 5.8, "end": 6.1, "score": 0.7, "source": "acoustic"},
        ]

        apply_conservative_line_anchor(
            previous,
            previous_acoustic,
            previous_windowed,
            current,
            anchor=4.3,
        )

        self.assertEqual(current[0]["start"], 4.3)
        self.assertEqual(previous[-1]["end"], 4.3)
        self.assertLessEqual(previous[0]["end"], previous[1]["start"])
        self.assertTrue(
            all(word["end"] - word["start"] >= min_duration(word["word"]) - 0.001 for word in previous)
        )

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

    def test_line_shift_needs_consistent_word_support(self):
        acoustic = [
            {"i": i + 1, "word": f"w{i}", "start": 10.0 + i * 0.4, "end": 10.3 + i * 0.4, "score": 0.6}
            for i in range(8)
        ]
        line_windowed = [
            {"i": 1, "word": "w0", "start": 9.3, "end": 9.38, "score": 0.58},
            *[
                {
                    "i": i + 1,
                    "word": f"w{i}",
                    "start": 10.0 + i * 0.4,
                    "end": 10.3 + i * 0.4,
                    "score": 0.58,
                }
                for i in range(1, 8)
            ],
        ]
        chunk = [
            {"i": w["i"], "word": w["word"], "start": w["start"], "end": w["end"], "score": w["score"], "source": "acoustic"}
            for w in acoustic
        ]

        shift_line_to_early_anchor(chunk, acoustic, line_windowed)

        self.assertEqual(10.0, chunk[0]["start"])

    def test_blend_suppresses_consistently_early_line_windowed_timeline(self):
        acoustic = [
            {"i": i + 1, "word": f"word{i}", "start": i * 0.5, "end": i * 0.5 + 0.3, "score": 0.55}
            for i in range(10)
        ]
        line_windowed = [
            {"i": i + 1, "word": f"word{i}", "start": i * 0.5 - 0.5, "end": i * 0.5 - 0.2, "score": 0.75}
            for i in range(10)
        ]
        line_windowed[0]["start"] = 0.0
        line_windowed[0]["end"] = 0.3

        words, _ = blend(acoustic, line_windowed, [" ".join(f"word{i}" for i in range(10))])

        self.assertTrue(all("line_windowed" not in word["source"] for word in words))

    def test_global_early_drift_still_allows_later_local_line_windowed_word(self):
        acoustic = [
            {"i": i + 1, "word": f"word{i}", "start": i * 0.5, "end": i * 0.5 + 0.3, "score": 0.55}
            for i in range(10)
        ]
        line_windowed = [
            {"i": i + 1, "word": f"word{i}", "start": i * 0.5 - 0.5, "end": i * 0.5 - 0.2, "score": 0.75}
            for i in range(10)
        ]
        acoustic[0]["score"] = 0.2
        line_windowed[0] = {"i": 1, "word": "word0", "start": 0.32, "end": 0.6, "score": 0.75}

        words, _ = blend(acoustic, line_windowed, [" ".join(f"word{i}" for i in range(10))])

        self.assertIn("line_windowed", words[0]["source"])
        self.assertEqual(0.32, words[0]["start"])

    def test_optimization_preserves_later_selected_line_windowed_onset(self):
        chunk = [
            {"i": 1, "word": "short", "start": 1.0, "end": 1.4, "source": "acoustic", "score": 0.6},
            {"i": 2, "word": "longer", "start": 2.0, "end": 2.6, "source": "line_windowed", "score": 0.75},
        ]
        acoustic = [
            {"i": 1, "word": "short", "start": 1.0, "end": 1.4, "score": 0.6},
            {"i": 2, "word": "longer", "start": 1.42, "end": 2.1, "score": 0.55},
        ]
        line_windowed = [
            {"i": 1, "word": "short", "start": 0.6, "end": 0.9, "score": 0.7},
            {"i": 2, "word": "longer", "start": 2.0, "end": 2.6, "score": 0.75},
        ]

        optimize_line_boundaries(chunk, acoustic, line_windowed)

        self.assertEqual(2.0, chunk[1]["start"])

    def test_later_line_windowed_boundary_does_not_overstretch_short_previous_word(self):
        chunk = [
            {"i": 1, "word": "no", "start": 1.0, "end": 1.4, "source": "acoustic", "score": 0.6},
            {"i": 2, "word": "reason", "start": 2.0, "end": 2.6, "source": "line_windowed", "score": 0.75},
        ]
        acoustic = [
            {"i": 1, "word": "no", "start": 1.0, "end": 1.4, "score": 0.6},
            {"i": 2, "word": "reason", "start": 1.42, "end": 2.1, "score": 0.55},
        ]
        line_windowed = [
            {"i": 1, "word": "no", "start": 0.6, "end": 0.9, "score": 0.7},
            {"i": 2, "word": "reason", "start": 2.0, "end": 2.6, "score": 0.75},
        ]

        optimize_line_boundaries(chunk, acoustic, line_windowed)

        self.assertGreater(chunk[0]["start"], 1.0)
        self.assertLessEqual(chunk[0]["end"] - chunk[0]["start"], 0.361)


if __name__ == "__main__":
    unittest.main()
