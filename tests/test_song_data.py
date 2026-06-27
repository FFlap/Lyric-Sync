#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from song_data import build_lines


class SongDataTests(unittest.TestCase):
    def test_overlapping_words_do_not_activate_next_line_early(self):
        words = [
            {"i": 1, "word": "first", "start": 1.0, "end": 1.4},
            {"i": 2, "word": "line", "start": 1.4, "end": 2.0},
            {"i": 3, "word": "next", "start": 1.7, "end": 2.1},
            {"i": 4, "word": "line", "start": 2.1, "end": 2.5},
        ]

        lines = build_lines(words, ["first line", "next line"])

        self.assertEqual(lines[1]["start"], 2.0)


if __name__ == "__main__":
    unittest.main()
