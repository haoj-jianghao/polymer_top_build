"""Assign OPLS/2020 bonded interactions.

RDKit supplies the molecular graph, atom typing one OPLS bonded class per atom,
and the OPLS/2020 tables (``opls2020.py``) the numerical parameters. The
OPLS/2020 table is always the first choice for every term.

With the run-time QM fit (``defer_missing``; ``--qm xtb|dft``), every term the
OPLS/2020 table lacks is returned as a ``qm-pending`` placeholder that the
on-the-fly fit (``qm_fit.py``) replaces; nothing else is consulted.

Without it (default), a missing term goes through, in order:

- Bonds and angles: with a GAFF2-typed atom, the gaff2.dat term of the full
  GAFF2 types of its atoms (``gaff2_bonded.py``); the value recorded from
  earlier QM fits of the same classes (``qm_bonded.py``; angles also by
  small-ring context), unless ``qm_bonded_table`` is off; the fallback of
  ``bonded_substitution.py`` (bonds: b0 from a parmchk2-style or
  default-class substitute and the empirical k; angles: the empirical rule),
  unless ``bonded_fallback`` is off. Every bond and angle gets a value.
- Proper torsions, unless ``torsion_substitution`` is off: the GAFF2-derived
  general X-B-C-X term of its central classes (``gaff2_torsions.py``); the
  parmchk2-style substitution of AmberTools (``parmchk.py``); the torsion
  fallback of ``bonded_substitution.py`` (general X-B-C-X with default central
  classes; gaff2.dat X-b-c-X of the full GAFF2 types; closest OPLS/2020 entry
  about the same central classes; finally zero). Every torsion gets a value.

A term still missing (a lookup or fallback switched off) is reported in one
error. Torsions flagged for a QM check (``qm_check``) and zero torsions are
listed in the warnings. Impropers are never fitted: planar centres get the
OPLS improper family of their class (GAFF2 centres by element and bonding) and
allenes the OPLS/2020 twist term. Atom indices in the public records are
one-based so they can be written directly to GROMACS.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from rdkit import Chem

from .assignment import AssignmentError

# Provenance of each assigned bonded parameter, stored as ``rule`` on every
# interaction record.
RULE_EXACT = "exact"  # table entry for exactly these classes
RULE_WILDCARD = "wildcard"  # table torsion entry with ? / ?? wildcards
RULE_IMPROPER = "improper-family"  # OPLS improper family of the centre's class
RULE_GAFF2_IMPROPER = "gaff2-improper-family"  # GAFF2-typed centre mapped to an OPLS improper
# Missing from the table and deferred to the on-the-fly QM fit (qm_fit.py);
# the stored values are placeholders until the fit replaces them.
RULE_QM_PENDING = "qm-pending"
PLACEHOLDER_BOND_K = 250000.0  # kJ/mol/nm^2
PLACEHOLDER_ANGLE_K = 400.0  # kJ/mol/rad^2

# Rules that take a parameter straight from the table; any other rule is
# annotated in the ITP.
STANDARD_RULES = frozenset({RULE_EXACT, RULE_WILDCARD, RULE_IMPROPER})


@dataclass(frozen=True)
class BondInteraction:
    atoms: tuple[int, int]
    classes: tuple[str, str]
    length_nm: float
    force_constant_kj_mol_nm2: float
    source_line: int
    rule: str
    parameter_classes: tuple[str, ...] | None = None  # substitute's classes (fallback)
    substitution_penalty: float | None = None  # fallback substitution penalty


@dataclass(frozen=True)
class AngleInteraction:
    atoms: tuple[int, int, int]
    classes: tuple[str, str, str]
    angle_deg: float
    force_constant_kj_mol_rad2: float
    source_line: int
    rule: str


@dataclass(frozen=True)
class ProperTorsionInteraction:
    atoms: tuple[int, int, int, int]
    classes: tuple[str, str, str, str]
    coefficients_kj_mol: tuple[float, float, float, float, float, float]
    parameter_classes: tuple[str, str, str, str]
    source_line: int
    rule: str
    substitution_penalty: float | None = None  # parmchk2 penalty score (parmchk.RULES)
    qm_check: bool = False  # term flagged for a QM check (gaff2_torsions.py)


@dataclass(frozen=True)
class ImproperTorsionInteraction:
    atoms: tuple[int, int, int, int]
    classes: tuple[str, str, str, str]
    phase_deg: float
    force_constant_kj_mol: float
    multiplicity: int
    parameter_name: str
    source_line: int
    rule: str


@dataclass(frozen=True)
class PairInteraction:
    atoms: tuple[int, int]
    function: int = 1


@dataclass(frozen=True)
class BondedAssignment:
    parameter_file: Path
    bonds: tuple[BondInteraction, ...]
    angles: tuple[AngleInteraction, ...]
    proper_torsions: tuple[ProperTorsionInteraction, ...]
    improper_torsions: tuple[ImproperTorsionInteraction, ...]
    pairs: tuple[PairInteraction, ...]
    warnings: tuple[str, ...] = ()
    nrexcl: int = 3
    fudge_lj: float = 0.5
    fudge_qq: float = 0.5
    qm_method: str = ""  # set by qm_fit when missing terms were fitted

    def rule_counts(self) -> dict[str, int]:
        """Number of bonds, angles, and torsions assigned by each rule."""
        counts = Counter(
            value.rule
            for group in (
                self.bonds, self.angles, self.proper_torsions, self.improper_torsions
            )
            for value in group
        )
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class _Parameter:
    classes: tuple[str, ...]
    values: tuple[float, ...]
    source_line: int


def _canonical_path(path: tuple[int, ...]) -> tuple[int, ...]:
    return min(path, path[::-1])


def _enumerate_angles(molecule: Chem.Mol) -> list[tuple[int, int, int]]:
    angles = []
    for center in range(molecule.GetNumAtoms()):
        neighbors = sorted(atom.GetIdx() for atom in molecule.GetAtomWithIdx(center).GetNeighbors())
        angles.extend((left, center, right) for left, right in combinations(neighbors, 2))
    return angles


def _enumerate_propers(molecule: Chem.Mol) -> list[tuple[int, int, int, int]]:
    paths: set[tuple[int, int, int, int]] = set()
    for bond in molecule.GetBonds():
        center1, center2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = [a.GetIdx() for a in molecule.GetAtomWithIdx(center1).GetNeighbors() if a.GetIdx() != center2]
        right = [a.GetIdx() for a in molecule.GetAtomWithIdx(center2).GetNeighbors() if a.GetIdx() != center1]
        # An outer atom bonded to both central atoms closes a three-membered
        # ring; the "torsion" then spans a 1-3 pair and is not a dihedral of
        # the ring, so it is omitted (as BOSS/LigParGen do).
        left = [value for value in left if value not in right]
        for outer1 in left:
            for outer2 in right:
                if molecule.GetBondBetweenAtoms(outer2, center1) is None:
                    paths.add(_canonical_path((outer1, center1, center2, outer2)))
    return sorted(paths)


def _improper_name(
    center_atom: Chem.Atom, center_class: str, neighbor_classes: list[str],
) -> str | None:
    planar = center_atom.GetIsAromatic() or center_atom.GetHybridization() == Chem.HybridizationType.SP2
    if not planar:
        return None
    if _is_gaff2_class(center_class):
        return _gaff2_improper_name(center_atom)
    if center_class in {"C", "C$"} and any(value in {"O", "O2"} for value in neighbor_classes):
        return "improper_O_C_X_Y"
    # OPLS/2020 has no nitro improper; nitro N (NO) gets the amide-N term below.
    if center_class in {"N", "N*", "N2", "NA", "NB", "NC", "NT", "NY", "NZ",
                        "NM", "N$", "NS", "NX", "N=", "NO"}:
        return "improper_Z_N_X_Y"
    if center_class in {"CM", "C=", "C#"}:
        return "improper_Z_CM_X_Y"
    if center_class in {"CA", "CB", "CN", "CV", "CW", "CR", "CK", "CQ", "CS", "C*",
                        "CU", "CX", "CP", "C!"}:
        return "improper_Z_CA_X_Y"
    return None


def _is_gaff2_class(atom_class: str) -> bool:
    """GAFF2 types (the Amber fallback of untyped atoms) are lower case."""
    return atom_class[:1].islower()


def _gaff2_improper_name(center_atom: Chem.Atom) -> str | None:
    """OPLS improper family for a planar GAFF2-typed centre, by element and
    bonding: C=O carbon, aromatic C, other sp2 C, planar N; None otherwise."""
    element = center_atom.GetSymbol()
    if element == "C":
        if any(bond.GetBondType() == Chem.BondType.DOUBLE
               and bond.GetOtherAtom(center_atom).GetSymbol() == "O"
               for bond in center_atom.GetBonds()):
            return "improper_O_C_X_Y"
        return "improper_Z_CA_X_Y" if center_atom.GetIsAromatic() else "improper_Z_CM_X_Y"
    if element == "N":
        return "improper_Z_N_X_Y"
    return None


def _allene_twists(molecule: Chem.Mol, atom_classes, table) -> tuple[list, list[str]]:
    """Allene twist terms X-C1...C3-Y for every pair of substituents on the two
    ends of each C: centre (OPLS/2020 type 277), and warnings for pairs whose
    classes have no table entry."""
    terms, warnings = [], []
    for center in range(molecule.GetNumAtoms()):
        ends = [atom.GetIdx() for atom in molecule.GetAtomWithIdx(center).GetNeighbors()]
        if atom_classes[center] != "C:" or len(ends) != 2 or any(
            atom_classes[end] != "CM" for end in ends
        ):
            continue
        left, right = ends

        def substituents(end):
            return [a.GetIdx() for a in molecule.GetAtomWithIdx(end).GetNeighbors()
                    if a.GetIdx() != center]

        for x in substituents(left):
            for y in substituents(right):
                classes = (atom_classes[x], "CM", "CM", atom_classes[y])
                parameter = table.allene_twist(classes[0], classes[3])
                atoms = (x + 1, left + 1, right + 1, y + 1)
                if parameter is None:
                    warnings.append(
                        f"allene {left + 1}={center + 1}={right + 1}: no twist term for "
                        f"substituent classes {classes[0]}/{classes[3]} (atoms "
                        f"{atoms[0]}, {atoms[3]}); only the other substituent pairs hold the twist"
                    )
                    continue
                terms.append(ImproperTorsionInteraction(
                    atoms, classes, parameter.values[0], parameter.values[1],
                    int(parameter.values[2]), "allene_twist", parameter.source_line, RULE_EXACT,
                ))
    return terms, warnings


def assign_bonded_interactions(
    molecule: Chem.Mol,
    atom_classes: list[str] | tuple[str, ...],
    defer_missing: bool = False,
    torsion_substitution: bool = True,
    qm_bonded_table: bool = True,
    bonded_fallback: bool = True,
    gaff2_classes: list[str | None] | tuple[str | None, ...] | None = None,
) -> BondedAssignment:
    """Enumerate a molecular graph and assign OPLS/2020 bonded parameters.

    ``gaff2_classes`` gives the GAFF2 type of each atom that has an OPLS/2020
    analog type (``gaff2_analogs.py``; None for the other atoms). A term with
    such an atom is looked up in the OPLS/2020 table with the analog's class;
    if the table lacks it, every later lookup and fallback uses the GAFF2 type
    instead, as for a GAFF2-typed atom.

    Bonds and angles missing from the table first take the gaff2.dat term when
    they involve a GAFF2-typed atom, then the recorded QM value (``qm_bonded_table``).
    Proper torsions missing from the table first take the GAFF2-derived
    general term, then the parmchk2-style substitution (``torsion_substitution``). Interactions still missing are
    either returned as ``qm-pending`` placeholders (``defer_missing``; the
    on-the-fly QM fit replaces them) or, for bonds and angles, given the
    fallback of ``bonded_substitution.py`` (``bonded_fallback``); what is still
    missing is collected and reported in a single error.
    """
    from .opls2020 import load_bonded_table
    from .gaff2_torsions import RULE_GAFF2_GENERAL, load_gaff2_general
    from .parmchk import load_parmchk_scores
    from .gaff2_bonded import RULE_GAFF2_BONDED, load_gaff2_bonded
    from .qm_bonded import RULE_QM_TABLE, load_qm_bonded
    from .bonded_substitution import RULE_ZERO_TORSION, load_bonded_substitution

    if len(atom_classes) != molecule.GetNumAtoms():
        raise AssignmentError(
            f"Received {len(atom_classes)} OPLS classes for {molecule.GetNumAtoms()} atoms"
        )
    if gaff2_classes is not None and len(gaff2_classes) != len(atom_classes):
        raise AssignmentError(
            f"Received {len(gaff2_classes)} GAFF2 classes for {len(atom_classes)} atoms"
        )
    # classes for every lookup after the OPLS/2020 table: analog atoms as GAFF2
    lookup_classes = (list(atom_classes) if gaff2_classes is None else
                      [gaff2 or value for gaff2, value in zip(gaff2_classes, atom_classes)])
    table = load_bonded_table()
    # With the run-time QM fit (defer_missing), a term the OPLS/2020 table lacks
    # goes straight to the fit: no other table, substitution or fallback.
    lookups = not defer_missing
    substitution = load_parmchk_scores() if torsion_substitution and lookups else None
    general = load_gaff2_general() if torsion_substitution else None
    recorded = load_qm_bonded() if qm_bonded_table and lookups else None
    gaff2_table = gaff2_full = None  # full GAFF2 types, only with GAFF2-typed atoms
    if lookups and any(_is_gaff2_class(value) for value in lookup_classes):
        gaff2_table = load_gaff2_bonded()
        gaff2_full = gaff2_table.types(molecule)
    fallback = (load_bonded_substitution(gaff2_full is not None)
                if bonded_fallback and not defer_missing else None)
    bond_b0: dict[tuple[int, int], float] = {}  # assigned b0 per atom pair (empirical angles)
    bond_substitutes: dict[tuple[int, ...], tuple[tuple[str, ...], float]] = {}
    qm_checks: dict[tuple[int, ...], str] = {}  # torsion atoms -> flagged entry
    missing: dict[tuple[str, tuple[str, ...]], list[tuple[int, ...]]] = {}
    penalties: dict[tuple[int, ...], float] = {}  # substituted torsions
    conformer = molecule.GetConformer()

    def placeholder(kind: str, atoms: tuple[int, ...], classes: tuple[str, ...]) -> _Parameter:
        """Input-geometry values (bonds, angles) or zero (torsions)."""
        points = [conformer.GetAtomPosition(index) for index in atoms]
        if kind == "bond":
            return _Parameter(classes, ((points[0] - points[1]).Length() / 10.0,
                                        PLACEHOLDER_BOND_K), 0)
        if kind == "angle":
            angle = math.degrees((points[0] - points[1]).AngleTo(points[2] - points[1]))
            return _Parameter(classes, (angle, PLACEHOLDER_ANGLE_K), 0)
        return _Parameter(classes, (0.0,) * 6, 0)

    def full_types(atoms: tuple[int, ...], classes: tuple[str, ...],
                   needed: tuple[int, ...] | None = None) -> tuple[str | None, ...] | None:
        """Full GAFF2 types of a term with a GAFF2-typed atom, or None if one of
        the ``needed`` positions (default: all) has no GAFF2 type (e.g. Si)."""
        if gaff2_full is None or not any(_is_gaff2_class(value) for value in classes):
            return None
        types = tuple(gaff2_full[index] for index in atoms)
        needed = range(len(atoms)) if needed is None else needed
        return None if any(types[position] is None for position in needed) else types

    def resolve(
        kind: str, atoms: tuple[int, ...], table_classes: tuple[str, ...]
    ) -> tuple[_Parameter, str] | None:
        found = table.lookup(kind, table_classes)
        classes = tuple(lookup_classes[index] for index in atoms)
        if found is None and kind in ("bond", "angle"):
            types = full_types(atoms, classes)
            if types is not None:
                term = gaff2_table.bond(types) if kind == "bond" else gaff2_table.angle(types)
                if term is not None:
                    return term, RULE_GAFF2_BONDED
        if found is None and recorded is not None and kind in ("bond", "angle"):
            # recorded under the classes of the fitted topology: an analog atom's
            # OPLS class, or (fits before the analogs) its GAFF2 type
            for key in dict.fromkeys((table_classes, classes)):
                term = (recorded.bond(key) if kind == "bond"
                        else recorded.angle(molecule, atoms, key))
                if term is not None:
                    return term, RULE_QM_TABLE
        if found is None and fallback is not None and kind in ("bond", "angle"):
            if kind == "bond":
                hit = fallback.bond(molecule, atoms, classes, full_types(atoms, classes))
            else:
                hit = fallback.angle(molecule, atoms, classes, bond_b0)
            if hit is not None:
                if hit.rule != "empirical":
                    bond_substitutes[atoms] = (tuple(hit.parameter.classes), hit.penalty)
                return hit.parameter, hit.rule
        if kind == "proper" and general is not None:
            if found is not None:
                entry = found[0].classes
                if (entry[0] == entry[3] == "X"
                        and (table_classes[1], table_classes[2]) in general.qm_check_opls):
                    qm_checks[atoms] = f"OPLS/2020 {'-'.join(entry)}"
            elif lookups:
                term = general.lookup(classes, molecule.GetBondBetweenAtoms(
                    atoms[1], atoms[2]).IsInRing())
                if term is not None:
                    if term.qm_check:
                        qm_checks[atoms] = f"{RULE_GAFF2_GENERAL} {'-'.join(term.classes)}"
                    return term, RULE_GAFF2_GENERAL
        if found is None and kind == "proper" and substitution is not None:
            hit = substitution.substitute(table, classes)
            if hit is None and fallback is not None:
                hit = fallback.torsion(molecule, atoms, classes,
                                       full_types(atoms, classes, needed=(1, 2)))
            if hit is not None:
                if hit.rule != RULE_ZERO_TORSION:
                    penalties[atoms] = hit.penalty
                return hit.parameter, hit.rule
        if found is None:
            key = (kind, min(table_classes, table_classes[::-1]))
            missing.setdefault(key, []).append(tuple(index + 1 for index in atoms))
            if defer_missing:
                found = (placeholder(kind, atoms, table_classes), RULE_QM_PENDING)
        return found

    bonds = []
    for rdkit_bond in molecule.GetBonds():
        atoms = tuple(sorted((rdkit_bond.GetBeginAtomIdx(), rdkit_bond.GetEndAtomIdx())))
        classes = tuple(atom_classes[index] for index in atoms)
        found = resolve("bond", atoms, classes)
        if found is None:
            continue
        parameter, rule = found
        bond_b0[atoms] = parameter.values[0]
        substitute, penalty = bond_substitutes.get(atoms, (None, None))
        bonds.append(
            BondInteraction(
                tuple(index + 1 for index in atoms), classes,
                parameter.values[0], parameter.values[1], parameter.source_line, rule,
                substitute, penalty,
            )
        )

    angles = []
    for atoms in _enumerate_angles(molecule):
        classes = tuple(atom_classes[index] for index in atoms)
        found = resolve("angle", atoms, classes)
        if found is None:
            continue
        parameter, rule = found
        angles.append(
            AngleInteraction(
                tuple(index + 1 for index in atoms), classes,
                parameter.values[0], parameter.values[1], parameter.source_line, rule,
            )
        )

    propers = []
    for atoms in _enumerate_propers(molecule):
        classes = tuple(atom_classes[index] for index in atoms)
        found = resolve("proper", atoms, classes)
        if found is None:
            continue
        parameter, rule = found
        propers.append(
            ProperTorsionInteraction(
                tuple(index + 1 for index in atoms), classes,
                parameter.values, parameter.classes, parameter.source_line, rule,
                penalties.get(atoms), atoms in qm_checks,
            )
        )

    if missing and not defer_missing:
        lines = [
            f"  {kind} {'-'.join(classes)}: {len(atoms)}x, atoms "
            + ", ".join("-".join(map(str, value)) for value in atoms[:3])
            + (" ..." if len(atoms) > 3 else "")
            for (kind, classes), atoms in sorted(missing.items())
        ]
        raise AssignmentError(
            f"No OPLS/2020 parameter in {table.path} for {len(missing)} bonded "
            f"class pattern(s):\n" + "\n".join(lines)
        )

    warnings = [
        f"torsion term {entry} flagged for a QM check ({count} quartet(s))"
        for entry, count in sorted(Counter(qm_checks.values()).items())
    ]
    zeros = Counter(min(value.classes, value.classes[::-1]) for value in propers
                    if value.rule == RULE_ZERO_TORSION)
    warnings += [
        f"torsion {'-'.join(classes)} has no parameter in any table or fallback: set to zero "
        f"({count} quartet(s))"
        for classes, count in sorted(zeros.items())
    ]
    impropers = []
    for center in range(molecule.GetNumAtoms()):
        neighbors = sorted(
            atom.GetIdx() for atom in molecule.GetAtomWithIdx(center).GetNeighbors()
        )
        if len(neighbors) != 3:
            continue
        neighbor_classes = [atom_classes[index] for index in neighbors]
        name = _improper_name(
            molecule.GetAtomWithIdx(center), atom_classes[center], neighbor_classes
        )
        if name is None:
            continue
        parameter = table.improper(name)
        atoms = (neighbors[0], center, neighbors[1], neighbors[2])
        impropers.append(
            ImproperTorsionInteraction(
                tuple(index + 1 for index in atoms),
                tuple(atom_classes[index] for index in atoms),
                parameter.values[0], parameter.values[1], int(parameter.values[2]),
                name, parameter.source_line,
                RULE_GAFF2_IMPROPER if _is_gaff2_class(atom_classes[center]) else RULE_IMPROPER,
            )
        )

    twists, twist_warnings = _allene_twists(molecule, atom_classes, table)
    impropers.extend(twists)
    warnings.extend(twist_warnings)
    for center in range(molecule.GetNumAtoms()):
        atom = molecule.GetAtomWithIdx(center)
        if (_is_gaff2_class(atom_classes[center]) and atom.GetDegree() == 3
                and (atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2)
                and _gaff2_improper_name(atom) is None):
            warnings.append(
                f"atom {center + 1} ({atom.GetSymbol()}, GAFF2 {atom_classes[center]}) is a "
                f"planar centre with no OPLS improper family; no improper torsion written"
            )

    distance_matrix = Chem.GetDistanceMatrix(molecule)
    pairs = tuple(
        PairInteraction((left + 1, right + 1))
        for left in range(molecule.GetNumAtoms())
        for right in range(left + 1, molecule.GetNumAtoms())
        if int(distance_matrix[left, right]) == 3
    )

    return BondedAssignment(
        parameter_file=table.path,
        bonds=tuple(sorted(bonds, key=lambda value: value.atoms)),
        angles=tuple(sorted(angles, key=lambda value: value.atoms)),
        proper_torsions=tuple(sorted(propers, key=lambda value: value.atoms)),
        improper_torsions=tuple(sorted(impropers, key=lambda value: value.atoms)),
        pairs=pairs,
        warnings=tuple(warnings),
    )
