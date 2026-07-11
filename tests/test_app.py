#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import app as app_module
from song_lib import song_dir


class SongPathSafetyTests(unittest.TestCase):
    def test_song_dir_rejects_parent_directory(self):
        with self.assertRaises(ValueError):
            song_dir("..")

    def test_song_dir_accepts_generated_song_id(self):
        self.assertEqual(song_dir("example-song-AbCdEf12345"), ROOT / "songs" / "example-song-AbCdEf12345")

    def test_song_endpoints_reject_parent_directory(self):
        with app_module.app.test_client() as client:
            self.assertEqual(client.get("/api/songs/%2E%2E").status_code, 404)
            self.assertEqual(client.get("/api/songs/%2E%2E/audio").status_code, 404)
            self.assertEqual(client.delete("/api/songs/%2E%2E").status_code, 404)


class SongTitleTests(unittest.TestCase):
    def test_youtube_title_reads_video_metadata(self):
        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def extract_info(self, url, download=False):
                self.request = (url, download)
                return {"title": "  Video title  "}

        with patch.dict(sys.modules, {"yt_dlp": type("YtDlp", (), {"YoutubeDL": FakeYDL})}):
            self.assertEqual(app_module._youtube_title("https://youtu.be/example"), "Video title")

    def test_edit_title_updates_meta_and_song_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "song"
            d.mkdir()
            (d / "meta.json").write_text(json.dumps({"id": "song", "title": "Old"}))
            (d / "song.json").write_text(json.dumps({"meta": {"id": "song", "title": "Old"}, "lines": []}))
            with (
                patch.object(app_module, "song_dir", return_value=d),
                app_module.app.test_client() as client,
            ):
                response = client.patch("/api/songs/song", json={"title": "New title"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads((d / "meta.json").read_text())["title"], "New title")
            self.assertEqual(json.loads((d / "song.json").read_text())["meta"]["title"], "New title")

    def test_edit_title_rejects_blank_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "song"
            d.mkdir()
            with (
                patch.object(app_module, "song_dir", return_value=d),
                app_module.app.test_client() as client,
            ):
                response = client.patch("/api/songs/song", json={"title": "  "})
        self.assertEqual(response.status_code, 400)


class OpenSongsFolderTests(unittest.TestCase):
    def test_open_path_activates_finder_on_macos(self):
        with (
            patch.object(app_module.os, "name", "posix"),
            patch.object(app_module.sys, "platform", "darwin"),
            patch.object(app_module.subprocess, "Popen") as popen,
        ):
            app_module._open_path(app_module.SONGS)

        popen.assert_called_once_with(["open", "-a", "Finder", str(app_module.SONGS)])

    def test_open_song_folder_opens_selected_song_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected_song = Path(tmp) / "selected-song"
            selected_song.mkdir()
            with (
                patch.object(app_module, "song_dir", return_value=selected_song),
                patch.object(app_module, "_open_path") as open_path,
                app_module.app.test_client() as client,
            ):
                response = client.post("/api/songs/selected-song/open-folder")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})
        open_path.assert_called_once_with(selected_song)

    def test_open_song_folder_reports_launch_failure(self):
        with (
            patch.object(app_module, "song_dir", return_value=app_module.SONGS),
            patch.object(app_module, "_open_path", side_effect=OSError("not available")),
            app_module.app.test_client() as client,
        ):
            response = client.post("/api/songs/selected-song/open-folder")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"ok": False, "error": "Could not open songs folder"})

    def test_folder_button_precedes_lyric_search_button(self):
        with app_module.app.test_client() as client:
            response = client.get("/")
            markup = response.get_data(as_text=True)
            response.close()

        self.assertIn('id="openSongsFolderBtn"', markup)
        folder_button = markup.index('id="openSongsFolderBtn"')
        search_button = markup.index('id="lyricSearchBtn"')
        self.assertLess(folder_button, search_button)
        self.assertIn('aria-label="Open current song folder"', markup)
        self.assertIn("encodeURIComponent(currentId) + '/open-folder'", markup)


class LyricTimingMarkupTests(unittest.TestCase):
    def test_active_word_must_not_be_held_past_its_end(self):
        with app_module.app.test_client() as client:
            markup = client.get("/").get_data(as_text=True)

        self.assertIn("t < Number(line.words[wi].end)", markup)


class SaveTimingsTests(unittest.TestCase):
    def test_save_timings_rejects_parent_directory(self):
        with app_module.app.test_client() as client:
            response = client.put("/api/songs/%2E%2E/timings", json={"words": []})
        self.assertEqual(response.status_code, 404)

    def test_save_timings_rejects_bad_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "song"
            d.mkdir()
            words = [{"i": 1, "word": "Hello", "start": 2.0, "end": 1.0}]
            with (
                patch.object(app_module, "song_dir", return_value=d),
                app_module.app.test_client() as client,
            ):
                response = client.put("/api/songs/song/timings", json={"words": words})
        self.assertEqual(response.status_code, 400)

    def test_save_timings_writes_hybrid_and_song_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "song"
            d.mkdir()
            (d / "lyrics.txt").write_text("Hello world\n")
            (d / "meta.json").write_text(json.dumps({"id": "song", "title": "Song"}))
            words = [
                {"i": 1, "word": "Hello", "start": 1.0, "end": 1.5, "source": "manual", "score": 1},
                {"i": 2, "word": "world", "start": 1.6, "end": 2.2004, "source": "ctc", "score": 0.9},
            ]
            with (
                patch.object(app_module, "song_dir", return_value=d),
                app_module.app.test_client() as client,
            ):
                response = client.put("/api/songs/song/timings", json={"words": words})

            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data["ok"])
            hybrid = json.loads((d / "hybrid.json").read_text())
            self.assertEqual(hybrid[1]["end"], 2.2)
            song = json.loads((d / "song.json").read_text())
            self.assertEqual(song["lines"][0]["text"], "Hello world")
            self.assertEqual(song["lines"][0]["start"], 1.0)
            self.assertEqual(data["lines"][0]["words"][0]["source"], "manual")


class SyncLockTests(unittest.TestCase):
    def test_busy_lock_returns_conflict_without_releasing_other_job(self):
        class BusyLock:
            def __init__(self):
                self.release_called = False

            def acquire(self, blocking=True):
                return False

            def release(self):
                self.release_called = True

        lock = BusyLock()
        payload = {
            "url": "https://youtu.be/AbCdEf12345",
            "lyrics": "A valid lyric line",
        }

        with (
            patch.object(app_module, "_job_lock", lock),
            patch.object(app_module, "_read_status", return_value={"state": "queued"}),
            app_module.app.test_client() as client,
        ):
            response = client.post("/api/sync", json=payload)

        self.assertEqual(response.status_code, 409)
        self.assertFalse(lock.release_called)


if __name__ == "__main__":
    unittest.main()
