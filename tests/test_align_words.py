#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from align_words import (
    find_twin,
    interpolated_base_times,
    lyric_tokens,
    match_transcript,
    merge_anchors,
    min_duration,
    norm_token,
    paren_runs,
    pick_anchors,
    syllables,
    weak_segments,
)


class TextUtilTests(unittest.TestCase):
    def test_norm_token(self):
        self.assertEqual(norm_token("Walkin',"), "walkin'")
        self.assertEqual(norm_token("“Hey!”"), "hey")
        self.assertEqual(norm_token("2"), "two")
        self.assertEqual(norm_token("---"), "")

    def test_syllables(self):
        self.assertEqual(syllables("baby"), 2)
        self.assertEqual(syllables("straight"), 1)
        self.assertGreaterEqual(syllables(""), 1)

    def test_lyric_tokens_paren_flags(self):
        tokens = lyric_tokens(["Hey boy (get ya) now", "(Ooh) Am I wrong?"])
        flags = [t["paren"] for t in tokens]
        self.assertEqual(flags, [False, False, True, True, False, True, False, False, False])
        self.assertEqual([t["line"] for t in tokens[:5]], [0] * 5)

    def test_min_duration_grows_with_length(self):
        self.assertLess(min_duration("a"), min_duration("straight"))


class MatchTests(unittest.TestCase):
    def _tokens(self, words):
        return [{"norm": w, "word": w, "line": 0, "paren": False, "i": k + 1} for k, w in enumerate(words)]

    def test_exact_match_and_anchor_pick(self):
        toks = self._tokens(["she", "shot", "me", "bang", "bang"])
        hyp = [
            {"norm": "she", "start": 1.0, "end": 1.2, "p": 0.9},
            {"norm": "shot", "start": 1.3, "end": 1.5, "p": 0.9},
            {"norm": "me", "start": 1.6, "end": 1.7, "p": 0.9},
            {"norm": "bang", "start": 2.0, "end": 2.2, "p": 0.9},
            {"norm": "bang", "start": 2.4, "end": 2.6, "p": 0.9},
        ]
        pairs = match_transcript(toks, hyp)
        self.assertEqual(len(pairs), 5)
        anchors = pick_anchors(toks, hyp, pairs, [None] * 5)
        self.assertEqual([a["tok"] for a in anchors], [0, 1, 2, 3, 4])

    def test_time_penalty_prefers_right_repetition(self):
        # same line repeated; base times point at the second repetition
        toks = self._tokens(["she", "shot", "me"])
        hyp = [
            {"norm": "she", "start": 1.0, "end": 1.1, "p": 0.9},
            {"norm": "shot", "start": 1.2, "end": 1.3, "p": 0.9},
            {"norm": "me", "start": 1.4, "end": 1.5, "p": 0.9},
            {"norm": "she", "start": 8.0, "end": 8.1, "p": 0.9},
            {"norm": "shot", "start": 8.2, "end": 8.3, "p": 0.9},
            {"norm": "me", "start": 8.4, "end": 8.5, "p": 0.9},
        ]
        pairs = match_transcript(toks, hyp, base_times=[8.0, 8.2, 8.4])
        matched_hyp = [h for _, h, _ in pairs]
        self.assertEqual(matched_hyp, [3, 4, 5])

    def test_interpolated_base_times_fills_gaps(self):
        base = [{"start": 1.0}, None, {"start": 3.0}]
        self.assertEqual(interpolated_base_times(base), [1.0, 2.0, 3.0])

    def test_merge_anchors_keeps_monotone_times(self):
        primary = [
            {"tok": 0, "t": 1.0, "t_end": 1.2, "sim": 1.0},
            {"tok": 5, "t": 5.0, "t_end": 5.2, "sim": 1.0},
        ]
        extra = [
            {"tok": 2, "t": 2.0, "t_end": 2.2, "sim": 0.9},
            {"tok": 3, "t": 0.5, "t_end": 0.7, "sim": 0.9},  # violates order
        ]
        merged = merge_anchors(primary, extra)
        self.assertEqual([a["tok"] for a in merged], [0, 2, 5])


class EchoTests(unittest.TestCase):
    def test_paren_runs_and_twin(self):
        tokens = lyric_tokens(["I bet ya (I bet ya)"])
        runs = paren_runs(tokens)
        self.assertEqual(runs, [[3, 4, 5]])
        twin = find_twin(tokens, runs[0])
        self.assertEqual(twin, [0, 1, 2])

    def test_no_twin_for_unique_adlib(self):
        tokens = lyric_tokens(["I bet ya (whoa whoa)"])
        self.assertIsNone(find_twin(tokens, paren_runs(tokens)[0]))


class WeakSegmentTests(unittest.TestCase):
    def test_weak_segments_finds_long_distributed_runs(self):
        tokens = [{"norm": "x"}] * 6
        words = [
            {"start": 0.0, "end": 0.5, "source": "ctc"},
            {"start": 0.5, "end": 1.5, "source": "distributed"},
            {"start": 1.5, "end": 2.5, "source": "distributed"},
            {"start": 2.5, "end": 3.5, "source": "distributed"},
            {"start": 3.5, "end": 4.5, "source": "distributed"},
            {"start": 4.5, "end": 5.0, "source": "ctc"},
        ]
        segs = weak_segments(tokens, words)
        self.assertEqual(len(segs), 1)
        run, a, b = segs[0]
        self.assertEqual(run, [1, 2, 3, 4])
        self.assertEqual((a, b), (0.5, 4.5))

    def test_short_runs_ignored(self):
        tokens = [{"norm": "x"}] * 2
        words = [
            {"start": 0.0, "end": 3.0, "source": "distributed"},
            {"start": 3.0, "end": 6.0, "source": "distributed"},
        ]
        self.assertEqual(weak_segments(tokens, words), [])


if __name__ == "__main__":
    unittest.main()
