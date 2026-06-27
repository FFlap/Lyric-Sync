#!/usr/bin/env python3
import sys
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
