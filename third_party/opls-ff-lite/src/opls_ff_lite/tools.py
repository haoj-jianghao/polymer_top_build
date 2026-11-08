"""MOPAC lookup, config file, pre-flight check and `opls-ff-lite doctor`.

opls-ff-lite runs one external program, MOPAC (AM1 for the CM1A charges).
It is found, in this order:

1. ``OPLS_FF_LITE_MOPAC``;
2. the user config file, ``[programs]`` table (``opls-ff-lite config set mopac PATH``);
   the file is ``OPLS_FF_LITE_CONFIG``, else
   ``$XDG_CONFIG_HOME/opls-ff-lite/config.toml`` (default ``~/.config/...``);
3. ``mopac`` on ``PATH``.

Python packages: RDKit and NumPy only. No xTB, QM fit, OpenMM or GROMACS.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_ENV = "OPLS_FF_LITE_CONFIG"
MOPAC_ENV = "OPLS_FF_LITE_MOPAC"
MOPAC_INSTALL = "conda install -c conda-forge mopac=23.2.5"
KEYS = ("mopac",)

# Versions opls-ff-lite was validated with; others are expected to work, so a
# different version only gives a note, never an error.
TESTED_TEXT = {"mopac": "23.2.5", "rdkit": "2025.09-2026.09"}


# -- config file --------------------------------------------------------------
def config_path() -> Path:
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV]).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "opls-ff-lite" / "config.toml"


def read_config() -> dict[str, str]:
    """The ``[programs]`` table of the config file ({} if there is none)."""
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"cannot read the opls-ff-lite config file {path}: {exc}") from exc
    return {str(k): str(v) for k, v in data.get("programs", {}).items()}


def write_config(values: dict[str, str]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# opls-ff-lite settings (edit, or use: opls-ff-lite config set mopac PATH)",
             "[programs]"]
    for key in sorted(values):
        escaped = values[key].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n")
    return path


def set_config(key: str, value: str | None) -> Path:
    """Set (``value``) or remove (None) one config entry, after validating it."""
    if key not in KEYS:
        raise ValueError(f"unknown key {key!r}; the only key is 'mopac'")
    values = read_config()
    if value is None:
        values.pop(key, None)
        return write_config(values)
    path = Path(value).expanduser().resolve()
    if not _executable(path):
        raise ValueError(f"{path} is not an executable file")
    values[key] = str(path)
    return write_config(values)


# -- lookup -------------------------------------------------------------------
def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


@dataclass(frozen=True)
class Location:
    path: Path | None
    source: str  # "env OPLS_FF_LITE_MOPAC", "config ...", "PATH", or "not found"

    @property
    def ok(self) -> bool:
        return self.path is not None and _executable(self.path)


def locate(key: str = "mopac") -> Location:
    if key not in KEYS:
        raise KeyError(key)
    if os.environ.get(MOPAC_ENV):
        return Location(Path(os.environ[MOPAC_ENV]).expanduser(), f"env {MOPAC_ENV}")
    configured = read_config().get("mopac")
    if configured:
        return Location(Path(configured).expanduser(), f"config {config_path()}")
    found = shutil.which("mopac")
    return Location(Path(found), "PATH") if found else Location(None, "not found")


def program_path(key: str = "mopac") -> Path:
    """Path of MOPAC, or its bare name if it is not found (running it then
    fails with a 'not found' error)."""
    return locate(key).path or Path("mopac")


def program_version(key: str = "mopac", path: Path | None = None) -> str | None:
    path = path or locate(key).path
    if path is None or not _executable(path):
        return None
    try:
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                                timeout=30, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"MOPAC version (\S+)", result.stdout + result.stderr)
    return match.group(1) if match else "unknown"


def package_version(module: str) -> str | None:
    from importlib import metadata

    if importlib.util.find_spec(module) is None:
        return None
    try:
        return metadata.version(module)
    except metadata.PackageNotFoundError:
        return "unknown"


def version_note(key: str, version: str | None) -> str | None:
    """None for a tested version, else a note (never an error)."""
    if version is None or key not in TESTED_TEXT:
        return None
    label = {"mopac": "MOPAC", "rdkit": "RDKit"}[key]
    if version == "unknown":
        return f"{label}: the version could not be read; tested with {TESTED_TEXT[key]}"
    parts = re.findall(r"\d+", version)
    if key == "rdkit":
        tested = len(parts) >= 2 and (2025, 9) <= (int(parts[0]), int(parts[1])) <= (2026, 9)
    else:
        tested = parts[:1] == ["23"]
    if tested:
        return None
    return (f"{label} {version} differs from the tested {TESTED_TEXT[key]}; it is expected "
            "to work, but results are not validated with it")


def run_version_notes() -> list[str]:
    notes = []
    location = locate("mopac")
    if location.ok:
        notes.append(version_note("mopac", program_version("mopac", location.path)))
    notes.append(version_note("rdkit", package_version("rdkit")))
    return [note for note in notes if note]


# -- pre-flight ---------------------------------------------------------------
def _mopac_problem() -> str:
    location = locate("mopac")
    where = (f"{location.path} ({location.source}) is not an executable file"
             if location.path is not None else "not found")
    return (f"MOPAC: {where}; needed for the CM1A charges (AM1; every run).\n"
            f"    If you have it: opls-ff-lite config set mopac /path/to/mopac"
            f"  (or set {MOPAC_ENV})\n"
            f"    To install it:  {MOPAC_INSTALL}")


def preflight() -> tuple[list[str], list[str]]:
    """(errors, warnings) before a run: package data, the SI files, MOPAC."""
    from . import si
    from .datafiles import verify

    errors, warnings = verify()
    si_errors, si_warnings = si.check()
    errors += si_errors
    warnings += si_warnings
    if not locate("mopac").ok:
        errors.append(_mopac_problem())
    return errors, warnings


# -- doctor ---------------------------------------------------------------------
def _tag(ok: bool, required: bool = True) -> str:
    return f"{'[ok]' if ok else '[MISSING]' if required else '[missing]':<10}"


def doctor() -> tuple[str, bool]:
    """Status report; True if everything required is present."""
    import json

    from . import si
    from .datafiles import DATA_DIR, MANIFEST, verify

    lines = [f"config file: {config_path()}" + ("" if config_path().is_file() else " (none)"),
             "", "Program:"]
    location = locate("mopac")
    required_ok = location.ok
    if location.ok:
        version = program_version("mopac", location.path)
        lines.append(f"  {_tag(True)}MOPAC {version or ''} (required)\n"
                     f"            {location.path}  [{location.source}]")
        note = version_note("mopac", version)
        if note:
            lines.append(f"            note: {note}")
    else:
        lines.append(f"  {_tag(False)}" + _mopac_problem().replace("\n", "\n  "))
    lines += ["", "Python packages (this interpreter):"]
    for module, label in (("rdkit", "RDKit"), ("numpy", "NumPy")):
        version = package_version(module)
        required_ok &= version is not None
        lines.append(f"  {_tag(version is not None)}{label} {version or ''} (required)")
        note = version_note(module, version)
        if note:
            lines.append(f"            note: {note}")
    data_errors, data_warnings = verify()
    required_ok &= not data_errors
    lines += ["", "Package data:"]
    if not data_errors and not data_warnings:
        count = len(json.loads(MANIFEST.read_text())["files"])
        lines.append(f"  {_tag(True)}{count} files match {MANIFEST.name} ({DATA_DIR})")
    lines += [f"  {_tag(False)}{message}" for message in data_errors]
    lines += [f"  [warning] {message}" for message in data_warnings]
    directory, source = si.si_dir()
    si_errors, si_warnings = si.check()
    required_ok &= not si_errors
    lines += ["", f"OPLS/2020 SI (user-installed): {directory} [{source}]"]
    if si_errors:
        lines += [f"  {_tag(False)}" + message.replace("\n", "\n  ") for message in si_errors]
    else:
        for si_file in si.SI_FILES:
            lines.append(f"  {_tag(True)}{si_file.name} ({si_file.original})")
        lines += [f"  [warning] {message}" for message in si_warnings]
        if not si_warnings:
            lines.append("            checksums match the Oct 1, 2023 release")
    lines.append("" if required_ok else
                 "\nRequired dependencies are missing: runs will stop until they are installed.")
    return "\n".join(lines).rstrip() + "\n", required_ok
