"""The user-installed OPLS/2020 Supporting Information files.

The OPLS/2020 parameters are the Supporting Information of Jorgensen et al.,
J. Phys. Chem. B (2024), doi:10.1021/acs.jpcb.3c06602 (ACS). They may not be
redistributed, so opls-ff-lite never ships them: each user downloads them
from the article page and installs them with

    opls-ff-lite install-si jp3c06602_si_002.txt jp3c06602_si_003.txt

which checks them against the SHA-256 of the release opls-ff-lite was
validated with (Oct 1, 2023) and copies them to a per-user directory that
survives upgrades of the package. If ``OPLS_FF_LITE_SI_DIR`` is set, the
files are read from that directory only; otherwise from the first of these
that holds both:

1. the user directory: ``$XDG_DATA_HOME/opls-ff-lite/opls2020`` (default
   ``~/.local/share/...``; ``%APPDATA%\\opls-ff-lite\\opls2020`` on Windows);
2. ``data/opls2020/`` inside the package (development checkouts).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

PACKAGE_SI_DIR = Path(__file__).resolve().parent / "data" / "opls2020"
SI_DIR_ENV = "OPLS_FF_LITE_SI_DIR"
DOI = "10.1021/acs.jpcb.3c06602"


@dataclass(frozen=True)
class SIFile:
    name: str  # installed name
    original: str  # name on the ACS site
    content: str
    marker: str  # text in the file's first lines that identifies it
    sha256: str  # the Oct 1, 2023 release


SI_FILES = (
    SIFile("oplsaa2020.par", "jp3c06602_si_002.txt", "non-bonded and torsion parameters",
           "OPLS All-Atom Parameters",
           "a7b8daed6e100ba79fe82378450a4e12d2908c1f9b5a70bd8111dd56b3e46c40"),
    SIFile("oplsaa2020.sb", "jp3c06602_si_003.txt", "bond and angle parameters",
           "Bond Stretching and Angle Bending",
           "889c622f8a91674363f14294b7209d10de5a5823494e83648eb7e1a96d96fa3a"),
)


def user_si_dir() -> Path:
    if os.name == "nt" and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"])
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "opls-ff-lite" / "opls2020"


def candidate_dirs() -> list[tuple[Path, str]]:
    found = []
    if os.environ.get(SI_DIR_ENV):
        found.append((Path(os.environ[SI_DIR_ENV]).expanduser(), f"env {SI_DIR_ENV}"))
    found += [(user_si_dir(), "user directory"), (PACKAGE_SI_DIR, "package directory")]
    return found


def si_dir() -> tuple[Path, str]:
    """The directory the SI is read from and how it was chosen:
    ``OPLS_FF_LITE_SI_DIR`` if set (even if incomplete: an explicit choice is
    never silently replaced), else the first complete candidate, else the user
    directory (where ``install-si`` puts the files)."""
    candidates = candidate_dirs()
    if os.environ.get(SI_DIR_ENV):
        directory, source = candidates[0]
        complete = all((directory / f.name).is_file() for f in SI_FILES)
        return directory, source if complete else f"{source}, not installed"
    for directory, source in candidates:
        if all((directory / f.name).is_file() for f in SI_FILES):
            return directory, source
    return candidates[0][0], f"{candidates[0][1]}, not installed"


def si_paths() -> tuple[Path, Path]:
    """(oplsaa2020.par, oplsaa2020.sb) as read by ``opls2020.py``."""
    directory, _ = si_dir()
    return directory / SI_FILES[0].name, directory / SI_FILES[1].name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


INSTALL_HINT = (f"    Download the Supporting Information of doi:{DOI} from the article page "
                "(it may not be redistributed),\n"
                "    then: opls-ff-lite install-si jp3c06602_si_002.txt jp3c06602_si_003.txt")


def check() -> tuple[list[str], list[str]]:
    """(errors, warnings): missing files are an error; files that differ from
    the validated release are a warning (results may differ)."""
    directory, source = si_dir()
    missing = [f for f in SI_FILES if not (directory / f.name).is_file()]
    if missing:
        return ([f"OPLS/2020 SI not installed (looked in {directory} [{source}]; "
                 f"missing {', '.join(f.name for f in missing)}).\n" + INSTALL_HINT], [])
    changed = [f.name for f in SI_FILES if _sha256(directory / f.name) != f.sha256]
    if changed:
        return [], [f"OPLS/2020 SI files {', '.join(changed)} in {directory} differ from the "
                    "Oct 1, 2023 release opls-ff-lite was validated with; results may differ"]
    return [], []


def identify(path: Path) -> SIFile | None:
    """Which SI file ``path`` is: by checksum, else by its header."""
    digest = _sha256(path)
    for si_file in SI_FILES:
        if digest == si_file.sha256:
            return si_file
    with path.open(errors="replace") as handle:
        head = "".join(handle.readline() for _ in range(5))
    for si_file in SI_FILES:
        if si_file.marker in head:
            return si_file
    return None


class SIInstallError(RuntimeError):
    pass


def install(paths: list[Path], dest: Path | None = None, force: bool = False) -> list[str]:
    """Check and copy the two SI files; returns a report line per file."""
    dest = Path(dest).expanduser() if dest is not None else user_si_dir()
    chosen: dict[str, tuple[Path, SIFile]] = {}
    for path in paths:
        path = Path(path).expanduser()
        if not path.is_file():
            raise SIInstallError(f"{path}: no such file")
        si_file = identify(path)
        if si_file is None:
            raise SIInstallError(
                f"{path}: not an OPLS/2020 SI parameter file (expected "
                + " or ".join(f"{f.original} ({f.content})" for f in SI_FILES) + ")")
        if si_file.name in chosen:
            raise SIInstallError(f"{path} and {chosen[si_file.name][0]} are both "
                                 f"{si_file.original}")
        if _sha256(path) != si_file.sha256 and not force:
            raise SIInstallError(
                f"{path} looks like {si_file.original} but differs from the Oct 1, 2023 "
                f"release opls-ff-lite was validated with (SHA-256 {si_file.sha256}); "
                "use --force to install it anyway")
        chosen[si_file.name] = (path, si_file)
    absent = [f.original for f in SI_FILES if f.name not in chosen]
    if absent and not all((dest / f.name).is_file() for f in SI_FILES if f.name not in chosen):
        raise SIInstallError(f"also give {', '.join(absent)} (both files are needed)")
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    for name, (path, si_file) in chosen.items():
        shutil.copyfile(path, dest / name)
        status = "verified" if _sha256(dest / name) == si_file.sha256 else "UNVERIFIED (--force)"
        report.append(f"{path} -> {dest / name}  [{si_file.content}; {status}]")
    return report
