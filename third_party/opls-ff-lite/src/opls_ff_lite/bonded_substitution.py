"""Fallback for bonds, angles and torsions that no table has (after parmchk2's
chk_bond/chk_angle, and for torsions after the parmchk2-style substitution).

A bond or angle missing from the OPLS/2020 table, gaff2.dat (terms with a
GAFF2-typed atom) and the recorded QM table always gets a value here, so no
bond or angle needs a QM fit (the run-time QM fit is an option; ``qm_fit.py``).
The design follows a comparison with the xTB (Seminario) fits of about 900 NCI
molecules (validation/torsion_fallbacks/README.md, section 10; user decision
2026-10-28): a substituted bond's length is closer to xTB than the empirical
length, but the empirical force constant is closer than the substitute's, and
for angles the empirical rule is closer than any class substitution.

Bonds, in order:

1. ``gaff2-parmchk2-equa`` / ``gaff2-parmchk2-corr`` (terms with a GAFF2-typed
   atom): parmchk2's equal and corresponding types on the full GAFF2 types of
   both atoms against gaff2.dat, with the bond scores (bl, blf) and weights of
   ``data/parmchk/PARMCHK.DAT`` (AmberTools 26, unchanged).
2. ``parmchk2-equa`` / ``parmchk2-corr``: the same on the OPLS classes, with
   only the uppercase (AMBER) types of that file, as for torsions (``parmchk.py``).
3. ``default-substitution``: either or both atoms replaced by a default OPLS
   class read from the molecule (element, neighbour count, aromaticity and
   bonding; :func:`default_class`); one atom (penalty 1) before both (2).

   Steps 1-3 give b0 (and the substitute's classes); k is the empirical k of
   that b0 (below) when the elements have data, else the substitute's.
4. ``empirical``: parmchk2's empbond with the element data of
   ``data/parmchk/PARM_BLBA_GAFF2.DAT``: b0 is the file's single-bond
   reference length of the two elements, shortened for the bond order by
   Pauling's r(n) = r(1) - 0.71 log10(n) (aromatic n = 1.5); k = exp(kai - PC ln b0).

Angles: ``empirical``. theta0 is the mean OPLS/2020 angle about the centre's
default class (60, 90 or 108 degrees inside a 3-, 4- or 5-membered ring;
109.47/120/180 by neighbour count if the class has no angle); k is parmchk2's
empangle (GAFF, Wang et al. 2004), 143.9 Z_a C_b Z_c exp(-2 D) / (r_ab + r_bc)
/ sqrt(theta0), with the two bonds' assigned b0.

Proper torsions (user decisions 2026-10-29), in order:

1. ``default-general``: the general X-B-C-X term with either central atom
   replaced by its default class (penalty 1) before both (2), from the
   OPLS/2020 table, then the GAFF2-derived general terms
   (``data/gaff2_general_torsions.par``).
2. With a GAFF2-typed atom, ``gaff2-dat-general``: the gaff2.dat X-b-c-X term
   of the full GAFF2 types of the central atoms (``data/gaff2_bonded.json``).
3. All atoms OPLS-typed, ``central-pair``: the OPLS/2020 specific entry with
   the same central classes whose end classes are closest; per end, 0 for a
   match, 1 for the same element, 2 for another heavy element, 3 for H
   against a heavy atom (the penalty is the sum; ties: first in the file).
   For classes such as Si that have only specific torsions in OPLS/2020.
4. ``zero-torsion``: a torsion none of these covers is set to zero (user
   decision 2026-10-29; listed in the warnings).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

from rdkit import Chem

DATA = Path(__file__).resolve().parent / "data" / "parmchk"
DEFAULT_PARMCHK = DATA / "PARMCHK.DAT"
DEFAULT_BLBA = DATA / "PARM_BLBA_GAFF2.DAT"
KCAL_TO_KJ = 4.184

RULE_EQUA = "parmchk2-equa"
RULE_CORR = "parmchk2-corr"
RULE_GAFF2_EQUA = "gaff2-parmchk2-equa"
RULE_GAFF2_CORR = "gaff2-parmchk2-corr"
RULE_DEFAULT = "default-substitution"
RULE_EMPIRICAL = "empirical"
RULE_DEFAULT_GENERAL = "default-general"  # torsions: X-B-C-X with default central classes
RULE_GAFF2_DAT_GENERAL = "gaff2-dat-general"  # torsions: gaff2.dat X-b-c-X, full GAFF2 types
RULE_CENTRAL_PAIR = "central-pair"  # torsions: closest specific entry about the same centrals
RULE_ZERO_TORSION = "zero-torsion"  # torsions: last resort, all coefficients zero
RULES = (RULE_EQUA, RULE_CORR, RULE_GAFF2_EQUA, RULE_GAFF2_CORR, RULE_DEFAULT, RULE_EMPIRICAL,
         RULE_DEFAULT_GENERAL, RULE_GAFF2_DAT_GENERAL, RULE_CENTRAL_PAIR, RULE_ZERO_TORSION)

DEFAULT_PENALTY = 1.0  # per atom replaced by its default class
PAULING = 0.71  # Angstrom per decade of bond order
RING_ANGLE = {3: 60.0, 4: 90.0, 5: 108.0}
IDEAL_ANGLE = {4: 109.47, 3: 120.0, 2: 180.0}


@dataclass(frozen=True)
class Term:
    """Same layout as the table entries: classes, GROMACS values, source line."""

    classes: tuple[str, ...]
    values: tuple[float, float]  # (b0 nm, k kJ/mol/nm^2) or (theta0 deg, k kJ/mol/rad^2)
    source_line: int


@dataclass(frozen=True)
class Substitution:
    parameter: object  # table entry or Term
    rule: str
    penalty: float


# -- default classes ------------------------------------------------------------
def default_class(molecule: Chem.Mol, index: int) -> str:
    """Default OPLS/2020 class of an atom from its element and bonding."""
    atom = molecule.GetAtomWithIdx(index)
    element, degree = atom.GetSymbol(), atom.GetDegree()
    aromatic = atom.GetIsAromatic()
    orders = [bond.GetBondType() for bond in atom.GetBonds()]
    double = [bond.GetOtherAtom(atom).GetSymbol() for bond in atom.GetBonds()
              if bond.GetBondType() == Chem.BondType.DOUBLE]
    hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
    if element == "H":
        neighbor = atom.GetNeighbors()[0] if degree else None
        if neighbor is None or neighbor.GetSymbol() == "C":
            return "HA" if neighbor is not None and neighbor.GetIsAromatic() else "HC"
        return {"N": "H", "O": "HO", "S": "HS"}.get(neighbor.GetSymbol(), "HC")
    if element == "C":
        if degree == 4:
            return "CT"
        if degree == 3:
            if aromatic:
                return "CA"
            return "C" if any(x in ("O", "S") for x in double) else "CM"
        return "CZ"
    if element == "N":
        if degree == 4:
            return "N3"
        if degree == 1 or Chem.BondType.TRIPLE in orders:
            return "NZ"
        oxygens = sum(n.GetSymbol() == "O" for n in atom.GetNeighbors())
        if degree == 3 and oxygens >= 2:
            return "NO"
        if aromatic:
            return "NA" if degree == 3 or hydrogens else "NB"
        if degree == 3:
            planar = double or any(
                n.GetIsAromatic() or any(b.GetBondType() == Chem.BondType.DOUBLE
                                         for b in n.GetBonds())
                for n in atom.GetNeighbors())
            return "N" if planar else "NT"
        return "NC"
    if element == "O":
        if degree == 1:
            return "O"
        return "OH" if hydrogens else "OS"
    if element == "S":
        if degree >= 4:
            return "SY"
        if degree == 3:
            return "SZ"
        if aromatic:
            return "SA"
        if degree == 1:
            return "S"
        return "SH" if hydrogens else "S"
    return {"P": "P", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I", "Si": "Si",
            "B": "B"}.get(element, element)


# -- parmchk2 bond and angle scores ---------------------------------------------
@dataclass(frozen=True)
class _Corr:
    atomtype: str
    bl: float
    blf: float
    cba: float
    cbaf: float
    ba: float
    baf: float
    kind: int  # 0 itself, 1 equal type, 2 corresponding type


@dataclass(frozen=True)
class _Type:
    atomtype: str
    group: int
    equa: tuple[str, ...]
    corr: tuple[_Corr, ...]


class BondScores:
    """Atom-type equalities and bond correspondence scores of a PARMCHK.DAT."""

    def __init__(self, path: str | Path = DEFAULT_PARMCHK, uppercase_only: bool = True):
        keep = (lambda t: t[:1].isupper()) if uppercase_only else (lambda t: True)
        weights, defaults, raw = {}, {}, []
        for line in Path(path).read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "PARM":
                raw.append([fields[1], int(fields[3]), [], []])
            elif fields[0] == "EQUA":
                raw[-1][2].append(fields[1])
            elif fields[0] == "CORR":
                # sscanf order: bl blf cba cbaf ba baf ctor tor ps
                values = [float(v) for v in fields[2:8]] + [0.0] * (6 - len(fields[2:8]))
                raw[-1][3].append((fields[1], values))
            elif fields[0].startswith("WEIGHT_"):
                weights[fields[0][7:]] = float(fields[1])
            elif fields[0].startswith("DEFAULT_"):
                defaults[fields[0][8:]] = float(fields[1])
        self.w = weights
        names = ("BL", "BLF", "BA_CTR", "BAF_CTR", "BA", "BAF")
        self.types: dict[str, _Type] = {}
        for atomtype, group, equa, corr in raw:
            if not keep(atomtype) or atomtype in self.types:
                continue
            entries = [_Corr(atomtype, 0, 0, 0, 0, 0, 0, 0)]
            entries += [_Corr(name, 0, 0, 0, 0, 0, 0, 1) for name in equa if keep(name)]
            for name, values in corr:
                if keep(name):
                    entries.append(_Corr(name, *(defaults[d] if v < 0 else v
                                                 for d, v in zip(names, values)), 2))
            self.types[atomtype] = _Type(atomtype, group,
                                         tuple([atomtype] + [n for n in equa if keep(n)]),
                                         tuple(entries))
        for atomtype, entry in list(self.types.items()):  # targets must be types of the file
            self.types[atomtype] = _Type(
                atomtype, entry.group, tuple(n for n in entry.equa if n in self.types),
                tuple(c for c in entry.corr if c.atomtype in self.types))

    def _type(self, atomtype: str) -> _Type:
        return self.types.get(atomtype) or _Type(
            atomtype, 0, (atomtype,), (_Corr(atomtype, 0, 0, 0, 0, 0, 0, 0),))

    def _group(self, atomtype: str) -> int:
        return self._type(atomtype).group

    def bond(self, classes: tuple[str, str], lookup, rules=(RULE_EQUA, RULE_CORR)):
        """chk_bond: equal types (first hit), then corresponding types (lowest score)."""
        a, b = (self._type(x) for x in classes)
        for m, a2 in enumerate(a.equa):
            for n, b2 in enumerate(b.equa):
                if m == 0 and n == 0:
                    continue
                found = lookup((a2, b2))
                if found is not None:
                    return Substitution(found, rules[0], 0.0)
        best = None
        w = self.w
        for ca in a.corr:
            for cb in b.corr:
                if ca.kind <= 1 and cb.kind <= 1:
                    continue
                score = (ca.bl * w["BL"] + ca.blf * w["BLF"] + cb.bl * w["BL"] + cb.blf * w["BLF"])
                if self._group(ca.atomtype) != self._group(cb.atomtype):
                    score += w["GROUP"]
                if best is not None and score >= best.penalty:
                    continue
                found = lookup((ca.atomtype, cb.atomtype))
                if found is not None:
                    best = Substitution(found, rules[1], round(score, 1))
        return best


# -- empirical rules --------------------------------------------------------------
class EmpiricalRules:
    """parmchk2's empbond/empangle element data (PARM_BLBA_GAFF2.DAT)."""

    def __init__(self, path: str | Path = DEFAULT_BLBA):
        self.bonds: dict[tuple[int, int], tuple[float, float]] = {}  # (r_ref A, kai)
        self.angles: dict[int, tuple[float, float]] = {}  # atomic number -> (C, Z)
        for line in Path(path).read_text().splitlines():
            fields = line.split()
            if len(fields) < 3 or fields[0] != "PARM":
                continue
            if fields[1] == "PC":
                self.pc = float(fields[2])
            elif fields[1] == "BL":
                z1, z2 = int(fields[3]), int(fields[5])
                self.bonds.setdefault(tuple(sorted((z1, z2))), (float(fields[6]), float(fields[7])))
            elif fields[1] == "BA":
                self.angles[int(fields[3])] = (float(fields[4]), float(fields[5]))

    def _reference(self, z1: int, z2: int) -> tuple[float, float] | None:
        """(r_ref, kai); missing pairs mixed from the two diagonal entries as empbond does."""
        found = self.bonds.get(tuple(sorted((z1, z2))))
        if found is not None:
            return found
        d1, d2 = self.bonds.get((z1, z1)), self.bonds.get((z2, z2))
        if d1 is None or d2 is None:
            return None
        rref = 0.5 * (d1[0] + d2[0])
        dr1, dr2 = abs(rref - d1[0]), abs(rref - d2[0])
        kai = (d1[1] * dr1 + d2[1] * dr2) / (dr1 + dr2) if dr1 + dr2 > 0 else 0.5 * (d1[1] + d2[1])
        return rref, kai

    def bond_k(self, z1: int, z2: int, b0_nm: float) -> float | None:
        """empbond k (kJ/mol/nm^2) of a bond of length b0; None without element data."""
        reference = self._reference(z1, z2)
        if reference is None:
            return None
        k = math.exp(reference[1] - self.pc * math.log(b0_nm * 10.0))  # kcal/mol/A^2
        return 2.0 * k * KCAL_TO_KJ * 100.0

    def bond(self, z1: int, z2: int, order: float) -> tuple[float, float] | None:
        """(b0 nm, k kJ/mol/nm^2)."""
        reference = self._reference(z1, z2)
        if reference is None:
            return None
        rref, kai = reference
        b0 = rref - PAULING * math.log10(max(order, 1.0))
        k = math.exp(kai - self.pc * math.log(b0))  # kcal/mol/A^2, E = k (b - b0)^2
        return b0 / 10.0, 2.0 * k * KCAL_TO_KJ * 100.0

    def angle_k(self, za: int, zb: int, zc: int, b1_nm: float, b2_nm: float,
                theta0: float) -> float | None:
        """k (kJ/mol/rad^2) of empangle; None without element data."""
        if za not in self.angles or zb not in self.angles or zc not in self.angles:
            return None
        r1, r2 = b1_nm * 10.0, b2_nm * 10.0
        d = (r1 - r2) ** 2 / (r1 + r2) ** 2
        k = (143.9 * self.angles[za][1] * self.angles[zb][0] * self.angles[zc][1]
             * math.exp(-2.0 * d) / (r1 + r2) / math.sqrt(math.radians(theta0)))
        return 2.0 * k * KCAL_TO_KJ if k > 0 else None


def _bond_order(molecule: Chem.Mol, i: int, j: int) -> float:
    bond = molecule.GetBondBetweenAtoms(i, j)
    return 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()


def _ring_size(molecule: Chem.Mol, atoms: tuple[int, int, int]) -> int | None:
    """Smallest ring (<= 5 atoms) containing the whole angle."""
    sizes = [len(ring) for ring in molecule.GetRingInfo().AtomRings()
             if len(ring) <= 5 and all(a in ring for a in atoms)]
    return min(sizes) if sizes else None


# -- the fallback --------------------------------------------------------------------
class BondedSubstitution:
    def __init__(self, table, gaff2_table=None, general=None):
        self.table = table
        self.gaff2_table = gaff2_table
        self.general = general  # GAFF2-derived general torsions (gaff2_torsions.py)
        from .opls2020 import load_atom_types

        self.class_element: dict[str, int] = {}
        for record in load_atom_types().values():
            self.class_element.setdefault(record.opls_class, record.atomic_number)
        self.opls_scores = BondScores(uppercase_only=True)
        self.gaff2_scores = BondScores(uppercase_only=False) if gaff2_table else None
        self.empirical = EmpiricalRules()
        self._centre_angles: dict[str, float | None] = {}

    def _opls_bond(self, classes):
        found = self.table.lookup("bond", classes)
        return None if found is None else found[0]

    def _defaults(self, molecule, atoms, classes):
        own_default = [(c, default_class(molecule, i)) for i, c in zip(atoms, classes)]
        candidates = []
        for choice in product((0, 1), repeat=len(atoms)):
            if not any(choice):
                continue
            if any(c and own == default for c, (own, default) in zip(choice, own_default)):
                continue  # replacing a class by itself
            substitute = tuple(pair[c] for c, pair in zip(choice, own_default))
            candidates.append((DEFAULT_PENALTY * sum(choice), choice, substitute))
        for penalty, _, substitute in sorted(candidates):
            found = self._opls_bond(substitute)
            if found is not None:
                return Substitution(found, RULE_DEFAULT, penalty)
        return None

    def bond(self, molecule, atoms, classes, full_types=None) -> Substitution | None:
        """b0 from a substitute bond (GAFF2 types, then OPLS classes, then default
        classes; k empirical, the substitute's k without element data), else both
        empirical; None without element data."""
        z = [molecule.GetAtomWithIdx(i).GetAtomicNum() for i in atoms]
        hit = None
        if full_types is not None and self.gaff2_scores is not None:
            hit = self.gaff2_scores.bond(full_types, self.gaff2_table.bond,
                                         (RULE_GAFF2_EQUA, RULE_GAFF2_CORR))
        if hit is None:
            hit = self.opls_scores.bond(classes, self._opls_bond)
        if hit is None:
            hit = self._defaults(molecule, atoms, classes)
        if hit is not None:
            b0, k = hit.parameter.values[:2]
            empirical_k = self.empirical.bond_k(z[0], z[1], b0)
            term = Term(tuple(hit.parameter.classes), (b0, k if empirical_k is None else empirical_k),
                        hit.parameter.source_line)
            return Substitution(term, hit.rule, hit.penalty)
        values = self.empirical.bond(z[0], z[1], _bond_order(molecule, *atoms))
        if values is None:
            return None
        return Substitution(Term(tuple(classes), values, 0), RULE_EMPIRICAL, 0.0)

    def _centre_angle(self, centre_class: str) -> float | None:
        if centre_class not in self._centre_angles:
            values = [p.values[0] for key, p in self.table.angles.items() if key[1] == centre_class]
            self._centre_angles[centre_class] = sum(values) / len(values) if values else None
        return self._centre_angles[centre_class]

    def angle(self, molecule, atoms, classes, bond_b0: dict) -> Substitution | None:
        """Empirical angle; ``bond_b0``: assigned b0 (nm) per sorted 0-based atom pair."""
        a, b, c = atoms
        ring = _ring_size(molecule, atoms)
        if ring is not None:
            theta0 = RING_ANGLE[ring]
        else:
            theta0 = self._centre_angle(default_class(molecule, b))
            if theta0 is None:
                theta0 = IDEAL_ANGLE.get(molecule.GetAtomWithIdx(b).GetDegree(), 109.47)
        b1, b2 = bond_b0.get(tuple(sorted((a, b)))), bond_b0.get(tuple(sorted((b, c))))
        if b1 is None or b2 is None:
            return None  # a bond without a value: reported with the missing terms
        z = [molecule.GetAtomWithIdx(i).GetAtomicNum() for i in atoms]
        k = self.empirical.angle_k(*z, b1, b2, theta0)
        if k is None:
            return None
        return Substitution(Term(tuple(classes), (theta0, k), 0), RULE_EMPIRICAL, 0.0)

    def torsion(self, molecule, atoms, classes, full_types=None) -> Substitution:
        """Torsion fallback: default-general, then gaff2-dat-general (``full_types``:
        GAFF2 types when a GAFF2-typed atom is involved) or central-pair, else zero."""
        hit = self._default_general(molecule, atoms, classes)
        if hit is None and full_types is not None and self.gaff2_table is not None:
            term = self.gaff2_table.general_torsion(full_types[1], full_types[2])
            if term is not None:
                hit = Substitution(term, RULE_GAFF2_DAT_GENERAL, 0.0)
        if hit is None and full_types is None:
            hit = self._central_pair(classes)
        if hit is None:
            hit = Substitution(Term(tuple(classes), (0.0,) * 6, 0), RULE_ZERO_TORSION, 0.0)
        return hit

    def _end_score(self, pattern: str, value: str) -> int:
        from .opls2020 import _pattern_matches

        if _pattern_matches(pattern, value):
            return 0
        a, b = self.class_element.get(pattern), self.class_element.get(value)
        if a is not None and a == b:
            return 1
        return 3 if 1 in (a, b) and a is not None and b is not None else 2

    def _central_pair(self, classes) -> Substitution | None:
        from .opls2020 import _pattern_matches

        best, best_key = None, None
        for entry in self.table.specific_entries(classes[1], classes[2]):
            e = entry.classes
            if _pattern_matches(e[1], classes[1]) and _pattern_matches(e[2], classes[2]):
                ends = (e[0], e[3])
            else:
                ends = (e[3], e[0])
            score = self._end_score(ends[0], classes[0]) + self._end_score(ends[1], classes[3])
            key = (score, entry.source_line)
            if best_key is None or key < best_key:
                best, best_key = entry, key
        return None if best is None else Substitution(best, RULE_CENTRAL_PAIR, float(best_key[0]))

    def _default_general(self, molecule, atoms, classes) -> Substitution | None:
        """General X-B-C-X term with default classes for one, then both, centrals."""
        a, b, c, d = classes
        defaults = (default_class(molecule, atoms[1]), default_class(molecule, atoms[2]))
        candidates = []
        for choice in ((1, 0), (0, 1), (1, 1)):
            centrals = (defaults[0] if choice[0] else b, defaults[1] if choice[1] else c)
            if centrals == (b, c) or any(x and own == default for x, own, default
                                         in zip(choice, (b, c), defaults)):
                continue  # replacing a class by itself
            candidates.append((DEFAULT_PENALTY * sum(choice), centrals))
        ring = molecule.GetBondBetweenAtoms(atoms[1], atoms[2]).IsInRing()
        for penalty, (b_sub, c_sub) in sorted(candidates, key=lambda item: item[0]):
            found = self.table.proper_entry((a, b_sub, c_sub, d), True)
            if found is None and self.general is not None:
                found = self.general.lookup((a, b_sub, c_sub, d), ring)
            if found is not None:
                return Substitution(found, RULE_DEFAULT_GENERAL, penalty)
        return None


@lru_cache(maxsize=None)
def load_bonded_substitution(with_gaff2: bool) -> BondedSubstitution:
    from .gaff2_bonded import load_gaff2_bonded
    from .gaff2_torsions import load_gaff2_general
    from .opls2020 import load_bonded_table

    return BondedSubstitution(load_bonded_table(), load_gaff2_bonded() if with_gaff2 else None,
                              load_gaff2_general())
