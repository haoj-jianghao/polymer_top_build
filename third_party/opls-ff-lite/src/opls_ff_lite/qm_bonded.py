"""Recorded QM bonds and angles, looked up after the OPLS/2020 table.

``data/qm_bonded_terms.par`` (written by ``src/scripts/qm_bonded_terms.py``; not
part of the OPLS/2020 SI) holds bond and angle terms the table lacks, taken
from earlier on-the-fly QM fits (``qm_fit.py``: QM-optimized geometry, modified
Seminario force constants). ``bonded.py`` consults it after the OPLS/2020
lookup when no run-time QM fit is requested (with ``--qm`` every term the
table lacks is fitted), so a class pattern fitted once is reused. Its provenance (every contributing molecule and its
values) is ``data/qm_bonded_terms.json``.

Bonds are keyed by their two classes. Angles are keyed by their three classes
and the small-ring context of the angle (:func:`angle_ring_context`), so the
strained angles of 3- to 5-membered rings stay apart from open-chain and
larger-ring angles of the same classes. Each value is the mean over the
contributing molecules (symmetry copies within a molecule count once).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

DEFAULT_QM_BONDED = Path(__file__).resolve().parent / "data" / "qm_bonded_terms.par"
RULE_QM_TABLE = "qm-table"
SMALL_RING = 5  # largest ring size that gets its own angle context
NO_CONTEXT = "-"


def angle_ring_context(molecule: Chem.Mol, a: int, b: int, c: int) -> str:
    """``inN`` if angle a-b-c lies in a ring of N <= 5 atoms (smallest N),
    ``exoN`` if only its centre b does (the angle outside that ring), else ``-``."""
    rings = [ring for ring in molecule.GetRingInfo().AtomRings() if len(ring) <= SMALL_RING]
    inside = [len(ring) for ring in rings if a in ring and b in ring and c in ring]
    if inside:
        return f"in{min(inside)}"
    around = [len(ring) for ring in rings if b in ring]
    if around:
        return f"exo{min(around)}"
    return NO_CONTEXT


def bond_key(classes: tuple[str, str]) -> tuple[str, ...]:
    return min(tuple(classes), tuple(classes[::-1]))


def angle_key(classes: tuple[str, str, str], context: str) -> tuple[str, ...]:
    return min(tuple(classes), tuple(classes[::-1])) + (context,)


@dataclass(frozen=True)
class RecordedTerm:
    classes: tuple[str, ...]
    values: tuple[float, float]  # (b0 nm, k kJ/mol/nm^2) or (theta0 deg, k kJ/mol/rad^2)
    source_line: int
    comment: str


class QmBondedTable:
    def __init__(self, path: str | Path = DEFAULT_QM_BONDED):
        self.path = Path(path).resolve()
        self.bonds: dict[tuple[str, ...], RecordedTerm] = {}
        self.angles: dict[tuple[str, ...], RecordedTerm] = {}
        if not self.path.is_file():
            return
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            body, _, comment = line.partition(";")
            fields = body.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "bond" and len(fields) == 5:
                classes = tuple(fields[1:3])
                self.bonds[bond_key(classes)] = RecordedTerm(
                    classes, (float(fields[3]), float(fields[4])), line_number, comment.strip())
            elif fields[0] == "angle" and len(fields) == 7:
                classes = tuple(fields[1:4])
                self.angles[angle_key(classes, fields[4])] = RecordedTerm(
                    classes, (float(fields[5]), float(fields[6])), line_number, comment.strip())
            else:
                raise ValueError(f"{self.path}:{line_number}: cannot parse {line!r}")

    def bond(self, classes: tuple[str, str]) -> RecordedTerm | None:
        return self.bonds.get(bond_key(classes))

    def angle(self, molecule: Chem.Mol, atoms: tuple[int, int, int],
              classes: tuple[str, str, str]) -> RecordedTerm | None:
        """Recorded term for angle ``atoms`` (zero-based) with these classes."""
        return self.angles.get(angle_key(classes, angle_ring_context(molecule, *atoms)))


@lru_cache(maxsize=None)
def load_qm_bonded(path: str | Path = DEFAULT_QM_BONDED) -> QmBondedTable:
    return QmBondedTable(path)
