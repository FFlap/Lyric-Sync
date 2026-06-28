#!/usr/bin/env python3
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
