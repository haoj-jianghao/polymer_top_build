"""Finding and running external software that lives in the user's own environments.

OpenFF (Sage engine) and AmberTools (GAFF2 engine) are large installations
that users usually already have in some conda/mamba environment. The builder
does not import or install them: it runs a small worker script with that
environment's Python and reads the files it writes. This module holds the
shared pieces: locating conda installations and environments that contain
given packages, and running a worker script with an environment's Python.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

CONDA_ROOT_GUESSES = ("miniforge3", "mambaforge", "miniconda3", "anaconda3", "micromamba")


def python_in(prefix: Path) -> Path:
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def prefix_of(python: str | Path) -> Path:
    """Environment prefix of an interpreter (``<prefix>/bin/python``)."""

    path = Path(python)
    return path.parent if os.name == "nt" else path.parent.parent


def has_conda_packages(prefix: Path, packages: tuple[str, ...]) -> bool:
    meta = prefix / "conda-meta"
    return all(any(meta.glob(f"{package}-[0-9]*.json")) for package in packages)


def conda_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("CONDA_EXE"):
        roots.append(Path(os.environ["CONDA_EXE"]).resolve().parent.parent)
    for variable in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX"):
        if os.environ.get(variable):
            prefix = Path(os.environ[variable]).resolve()
            roots.append(prefix.parent.parent if prefix.parent.name == "envs" else prefix)
    roots += [Path.home() / name for name in CONDA_ROOT_GUESSES]
    return list(dict.fromkeys(root for root in roots if root.is_dir()))


def conda_environments_with(packages: tuple[str, ...], roots: list[Path] | None = None) -> list[Path]:
    """Prefixes (base and envs/*) of every conda installation that hold all ``packages``."""

    found = []
    for root in conda_roots() if roots is None else roots:
        for prefix in [root] + sorted(path for path in (root / "envs").glob("*") if path.is_dir()):
            if has_conda_packages(prefix, packages) and python_in(prefix).exists():
                found.append(prefix)
    return found


def worker_environment(python: str | Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a worker: its own bin directory first, no PYTHONPATH/PYTHONHOME."""

    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env


def run_worker(
    python: str | Path,
    script: Path,
    arguments: list[str],
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, dict, str]:
    """Run ``python -I script arguments``; return (exit code, last JSON line of stdout, stderr)."""

    completed = subprocess.run(
        [str(python), "-I", str(script), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=worker_environment(python, extra_env),
    )
    payload: dict = {}
    for line in reversed(completed.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
            break
        except ValueError:
            continue
    return completed.returncode, payload, completed.stderr


def unique(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result = []
    for python, source in candidates:
        if python not in seen:
            seen.add(python)
            result.append((python, source))
    return result
