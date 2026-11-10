"""Integrity of the package data (``data/MANIFEST.json``).

The manifest lists every data file the package ships (path relative to
``data/``, size, SHA-256), except the user-installed OPLS/2020 SI files in
``data/opls2020/``, which are never distributed. ``verify()`` runs before
every CLI run (``tools.preflight``) and in ``opls-ff-lite doctor``:

- a listed file that is missing, or a manifest that is missing, is an error:
  the installation is incomplete (e.g. a wheel built without package data);
- a file whose checksum differs is an error in an installed package (damaged
  or mixed installation) and a warning in a development checkout (the
  source tree, recognised by ``pyproject.toml`` two levels above the
  package), where data files are edited; regenerate the manifest with
  ``python src/scripts/build_data_manifest.py`` (the test suite fails until
  it matches; the scripts that rebuild data files regenerate it themselves).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"
EXCLUDED_DIRS = ("opls2020",)  # user-installed SI, never shipped


def development_checkout() -> bool:
    return (PACKAGE_DIR.parent.parent / "pyproject.toml").is_file()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shipped_files() -> list[Path]:
    """Data files that belong in the distribution (sorted)."""
    return sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
        and path.relative_to(DATA_DIR).parts[0] not in EXCLUDED_DIRS
    )


def build_manifest() -> dict:
    return {
        "description": "Package data of opls-ff-lite: path relative to data/, size and "
                       "SHA-256 (src/opls_ff_lite/datafiles.py). The OPLS/2020 SI "
                       "(data/opls2020/) is user-installed and not listed.",
        "files": {
            path.relative_to(DATA_DIR).as_posix(): {"size": path.stat().st_size,
                                                    "sha256": _sha256(path)}
            for path in shipped_files()
        },
    }


def write_manifest() -> Path:
    MANIFEST.write_text(json.dumps(build_manifest(), indent=1) + "\n")
    return MANIFEST


def verify() -> tuple[list[str], list[str]]:
    """(errors, warnings) about the package data."""
    reinstall = "reinstall opls-ff-lite from the release wheel"
    if not MANIFEST.is_file():
        return [f"package data incomplete: {MANIFEST} is missing; {reinstall}"], []
    try:
        listed = json.loads(MANIFEST.read_text())["files"]
    except (OSError, ValueError, KeyError) as exc:
        return [f"package data manifest {MANIFEST} is unreadable ({exc}); {reinstall}"], []
    missing = [name for name in listed if not (DATA_DIR / name).is_file()]
    changed = [name for name, entry in listed.items()
               if name not in missing and _sha256(DATA_DIR / name) != entry["sha256"]]
    errors, warnings = [], []
    if missing:
        errors.append(f"package data incomplete: {len(missing)} file(s) missing from "
                      f"{DATA_DIR}: {', '.join(missing)}; {reinstall}")
    if changed:
        message = (f"package data differs from its manifest: {', '.join(changed)}")
        if development_checkout():
            warnings.append(message + "; regenerate it with "
                            "'python src/scripts/build_data_manifest.py'")
        else:
            errors.append(message + f" (damaged or mixed installation); {reinstall}")
    return errors, warnings
