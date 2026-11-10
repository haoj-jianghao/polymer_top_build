"""GAFF2 bonds and angles for terms that involve a GAFF2-typed atom.

An atom OPLS cannot type gets a GAFF2 type (``amber.py``), but that LJ type
merges GAFF2 types with equal Lennard-Jones parameters, which bonded terms do
not allow. ``data/gaff2_bonded.json`` (``src/scripts/build_gaff2_bonded.py``; no
AmberTools at runtime) holds a second rule table with the full GAFF2 types
and the BOND/ANGL sections of gaff2.dat. A bond or angle that the OPLS/2020
table lacks and that involves a GAFF2-typed atom takes the gaff2.dat term of
the full GAFF2 types of all its atoms (OPLS-typed atoms included: their own
GAFF2 type in this molecule). The general X-b-c-X torsions of gaff2.dat are
offered to the torsion fallback (``bonded_substitution.py``), converted to
OPLS Fourier V1-V4 per quartet and to Ryckaert-Bellemans.

Conjugated pair types (cc/cd, ce/cf, ...) are assigned by alternation along
the Kekulé bonds of each conjugated system, as antechamber does: a single bond
between two pair atoms keeps the name, a double bond switches it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from .smarts_typing import SmartsRule, first_match

GAFF2_BONDED = Path(__file__).resolve().parent / "data" / "gaff2_bonded.json"
RULE_GAFF2_BONDED = "gaff2"
KCAL_TO_KJ = 4.184


@dataclass(frozen=True)
class Gaff2Term:
    classes: tuple[str, ...]  # full GAFF2 types of the entry
    # (b0 nm, k kJ/mol/nm^2), (theta0 deg, k kJ/mol/rad^2) or torsion RB C0-C5 (kJ/mol)
    values: tuple[float, ...]
    source_line: int


def amber_to_fourier(terms) -> tuple[float, float, float, float] | None:
    """gaff2.dat torsion terms [IDIVF, PK, phase, periodicity] -> OPLS V1-V4 per
    quartet (kcal/mol): V_n = +-2 PK/IDIVF; None if a term has no Fourier form."""
    v = [0.0] * 4
    for idivf, pk, phase, periodicity in terms:
        n, k = abs(int(periodicity)), pk / idivf
        if k == 0.0:
            continue
        if n > 4 or phase not in (0.0, 180.0):
            return None
        positive = (n % 2 == 1 and phase == 0.0) or (n % 2 == 0 and phase == 180.0)
        v[n - 1] += (2.0 if positive else -2.0) * k
    return tuple(v)


class Gaff2BondedTable:
    def __init__(self, path: str | Path = GAFF2_BONDED):
        payload = json.loads(Path(path).read_text())
        self.path = Path(path).resolve()
        self.rules = tuple(SmartsRule(r["type"], r["smarts"], r.get("note", ""))
                           for r in payload["rules"])
        self.pairs: dict[str, str] = payload["pairs"]
        # E = K (x - x0)^2 (kcal/mol, A or rad) -> GROMACS E = k/2 (x - x0)^2 (kJ/mol, nm or rad)
        self.bonds = {tuple(name.split("-")): (r0 / 10.0, 2.0 * k * KCAL_TO_KJ * 100.0, line)
                      for name, (k, r0, line) in payload["bonds"].items()}
        self.angles = {tuple(name.split("-")): (theta, 2.0 * k * KCAL_TO_KJ, line)
                       for name, (k, theta, line) in payload["angles"].items()}
        self.torsions = {tuple(name.split("-")[1:3]): (terms, line)
                         for name, (terms, line) in payload.get("torsions", {}).items()}

    def types(self, molecule: Chem.Mol) -> list[str | None]:
        """Full GAFF2 type of every atom (pairs resolved); None for an untypable
        atom (e.g. Si, which GAFF2 lacks), so only the terms with it lose GAFF2."""
        assigned = first_match(molecule, self.rules)
        types = [assigned[i].label if i in assigned else None
                 for i in range(molecule.GetNumAtoms())]
        six_rings = Counter(i for ring in molecule.GetRingInfo().AtomRings() if len(ring) == 6
                            for i in ring)
        # cp (antechamber [AR1,1RG6]): a fused AR1 atom (two 6-rings) is ca
        types = ["ca" if t == "cp" and six_rings[i] > 1 else t for i, t in enumerate(types)]
        return _alternate(molecule, types, self.pairs)

    def _lookup(self, entries: dict, types: tuple[str, ...]) -> Gaff2Term | None:
        for key in (types, types[::-1]):
            if key in entries:
                value, k, line = entries[key]
                return Gaff2Term(key, (value, k), line)
        return None

    def bond(self, types: tuple[str, str]) -> Gaff2Term | None:
        return self._lookup(self.bonds, types)

    def angle(self, types: tuple[str, str, str]) -> Gaff2Term | None:
        return self._lookup(self.angles, types)

    def general_torsion(self, b: str, c: str) -> Gaff2Term | None:
        """The gaff2.dat X-b-c-X term of central types b-c, as RB coefficients."""
        from .opls2020 import fourier_kcal_to_rb_kj

        for key in ((b, c), (c, b)):
            if key in self.torsions:
                terms, line = self.torsions[key]
                fourier = amber_to_fourier(terms)
                if fourier is None:
                    return None
                return Gaff2Term(("X", *key, "X"), fourier_kcal_to_rb_kj(*fourier), line)
        return None


def _alternate(molecule: Chem.Mol, types: list[str], pairs: dict[str, str]) -> list[str]:
    """Second pair name where the alternation along Kekulé bonds puts it."""
    kekule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001 - keep the first names
        return types
    parity: dict[int, int] = {}
    for start in range(len(types)):
        if types[start] not in pairs or start in parity:
            continue
        parity[start] = 0
        stack = [start]
        while stack:
            atom = stack.pop()
            for bond in kekule.GetAtomWithIdx(atom).GetBonds():
                other = bond.GetOtherAtomIdx(atom)
                if types[other] not in pairs or other in parity:
                    continue
                parity[other] = parity[atom] ^ (bond.GetBondType() == Chem.BondType.DOUBLE)
                stack.append(other)
    return [pairs[t] if parity.get(i) else t for i, t in enumerate(types)]


@lru_cache(maxsize=None)
def load_gaff2_bonded(path: str | Path = GAFF2_BONDED) -> Gaff2BondedTable:
    return Gaff2BondedTable(path)
