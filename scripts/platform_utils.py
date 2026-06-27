"""Cross-platform helpers for venv paths and subprocess execution."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def venv_dir() -> Path:
    return ROOT / ".venv"


def venv_bin_dir() -> Path:
    return venv_dir() / ("Scripts" if os.name == "nt" else "bin")


def project_python() -> Path:
    """Prefer project venv interpreter when present."""
    name = "python.exe" if os.name == "nt" else "python"
    candidate = venv_bin_dir() / name
    if candidate.is_file():
        return candidate
    return Path(sys.executable)


def augmented_path_env(extra: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    bindir = str(extra or venv_bin_dir())
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["PYTHON"] = str(project_python())
    return env


def run_checked(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or ROOT, env=env, check=True)


def run_python_script(
    script: str | Path,
    args: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [str(project_python()), str(SCRIPTS / script)]
    if args:
        cmd.extend(args)
    run_checked(cmd, env=env or augmented_path_env())


def demucs_executable() -> str:
    bindir = venv_bin_dir()
    name = "demucs.exe" if os.name == "nt" else "demucs"
    candidate = bindir / name
    if candidate.is_file():
        return str(candidate)
    import shutil

    found = shutil.which("demucs")
    if found:
        return found
    raise RuntimeError("demucs not on PATH. Run: python scripts/install_deps.py")


def ffmpeg_hint() -> str:
    if os.name == "nt":
        return "winget install ffmpeg  (or choco install ffmpeg)"
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    return "install ffmpeg with your package manager"
