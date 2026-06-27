#!/usr/bin/env python3
"""Create venv and install requirements (cross-platform)."""
from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def main() -> int:
    if not VENV.is_dir():
        print("Creating .venv…")
        venv.create(VENV, with_pip=True)

    py = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py.is_file():
        print(f"Missing venv python: {py}", file=sys.stderr)
        return 1

    print(f"Installing requirements into {VENV} …")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT, check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)], cwd=ROOT, check=True)

    print("")
    print("Verifying imports…")
    subprocess.run(
        [
            str(py),
            "-c",
            "import flask, whisperx, demucs, librosa, torch; print('ok:', torch.__version__)",
        ],
        cwd=ROOT,
        check=True,
    )

    print("")
    if sys.platform == "win32":
        print(r"Done. Run: .venv\Scripts\activate && python app.py")
    else:
        print("Done. Run: source .venv/bin/activate && python3 app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
