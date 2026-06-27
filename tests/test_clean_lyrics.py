#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clean_lyrics import parse_lyrics


class CleanLyricsTests(unittest.TestCase):
    def test_section_headers_removed(self):
        text = "[Intro: K'naan]\nShe shot me\n[Chorus]\nBang bang\n"
        lines, stats = parse_lyrics(text)
        self.assertEqual(lines, ["She shot me", "Bang bang"])
        self.assertEqual(stats["removed_sections"], 2)

    def test_paren_line_removed(self):
        text = "It went bang bang bang\n(Straight through my heart)\nAlthough I could\n"
        lines, stats = parse_lyrics(text)
        self.assertEqual(lines, ["It went bang bang bang", "Although I could"])
        self.assertEqual(stats["removed_paren_lines"], 1)

    def test_inline_parens_stripped(self):
        text = "(Ooh) Am I wrong?\nBut what is love without the pain to go along? (Ooh)\n"
        lines, stats = parse_lyrics(text)
        self.assertEqual(
            lines,
            ["Am I wrong?", "But what is love without the pain to go along?"],
        )
        self.assertEqual(stats["stripped_inline"], 2)

    def test_idempotent_on_clean_text(self):
        text = "She shot me, she shot me\nBang, bang, she shot me\n"
        a, _ = parse_lyrics(text)
        b, stats = parse_lyrics("\n".join(a))
        self.assertEqual(a, b)
        self.assertEqual(stats["removed_sections"], 0)


if __name__ == "__main__":
    unittest.main()
