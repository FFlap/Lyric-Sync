


https://github.com/user-attachments/assets/3b010986-0bb3-44b1-8d67-da3daf2cde1d

# Lyric Sync

Paste a YouTube link and lyrics, and get synchronized word-level timing in a local web player.

## What it does

* Syncs lyrics to any song from a YouTube URL.
* Lines up each word with the audio as it plays.
* Builds a personal library of songs on your machine.
* Highlights the current word so you can follow along.
* Cleans up pasted lyrics from Genius: section headers and ad-libs in parentheses are removed automatically.
* Works on macOS, Linux, and Windows.

## Install for development

**macOS / Linux**

```bash
git clone <your-repo-url> lyric-sync
cd lyric-sync
python3 scripts/install_deps.py
source .venv/bin/activate
brew install ffmpeg   # if needed
python3 app.py
```

**Windows**

```powershell
git clone <your-repo-url> lyric-sync
cd lyric-sync
py -3.11 scripts\install_deps.py
.\.venv\Scripts\Activate.ps1
winget install ffmpeg   # if needed
python app.py
```

Open **http://127.0.0.1:5050/**

Requires **Python 3.11+** and **ffmpeg** on your PATH. The first sync can take several minutes.

## Use

1. Click **Sync Song** and paste a YouTube URL plus lyrics (one line per row).
2. Wait for the sync to finish.
3. Select the song in the sidebar.
4. Press play and watch the lyrics move word by word with the music.

Lyrics must match the song in order. Empty lines are ignored.

## Commands

```bash
python3 app.py
python3 app.py --port 5050
DEVICE=cuda python3 app.py
python scripts/run_sync.py
python -m unittest discover -s tests -v
```
