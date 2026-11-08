"""OPLS/2020 parameter tables in BOSS format (``oplsaa.par`` / ``oplsaa.sb``).

Source: Supporting Information of W. L. Jorgensen, M. M. Ghahremanpour,
A. Saar and J. Tirado-Rives, "OPLS/2020 Force Field for Unsaturated
Hydrocarbons, Alcohols, and Ethers", J. Phys. Chem. B (2024),
https://doi.org/10.1021/acs.jpcb.3c06602 (files ``jp3c06602_si_002.txt``,
non-bonded and torsional parameters, and ``jp3c06602_si_003.txt``, bond
stretching and angle bending parameters, both dated Oct 1, 2023). They are
not redistributable: each user installs them (``opls-ff-lite install-si``;
``si.py`` gives where they are read from).

Units in the files: sigma in angstrom, epsilon in kcal/mol, bond
E = K (r - r0)^2 with K in kcal/mol/A^2, angle E = K (theta - theta0)^2 with
K in kcal/mol/rad^2, and OPLS Fourier torsions
E = V1/2 (1 + cos phi) + V2/2 (1 - cos 2phi) + V3/2 (1 + cos 3phi)
  + V4/2 (1 - cos 4phi) with V in kcal/mol. Everything is converted to GROMACS
units here.

Lookup follows BOSS: bonds and angles by atom class; torsions by class
quartet (either direction) where ``?`` is a one-character wildcard (``??``
matches any class). An exact quartet beats one with wildcards, fewer
wildcard characters beat more, and among equals the first entry in the file
wins (the file lists generic entries before specialised ones, e.g. the
"butane only" CT-CT-CT-CT after the general one). Duplicate bonds and angles
are resolved the same way (first entry).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .assignment import AssignmentError
from .si import si_paths

DEFAULT_PAR, DEFAULT_SB = si_paths()  # user-installed SI (si.py)
DATA_DIR = DEFAULT_PAR.parent

KCAL_TO_KJ = 4.184
# E = K dr^2 (kcal/mol/A^2)  ->  E = k/2 dr^2 (kJ/mol/nm^2)
BOND_K_TO_GROMACS = 2.0 * KCAL_TO_KJ * 100.0
# E = K dtheta^2 (kcal/mol/rad^2)  ->  E = k/2 dtheta^2 (kJ/mol/rad^2)
ANGLE_K_TO_GROMACS = 2.0 * KCAL_TO_KJ

_TYPE_LINE = re.compile(r"^ *(\d+) (\d\d) (\S{1,2}) +(\S+) +(\S+) +(\S+)\s*(.*)$")
_TORSION_LINE = re.compile(
    r"^ *(\d+) +([-0-9.]+) +([-0-9.]+) +([-0-9.]+) +([-0-9.]+) +"
    r"(.{2})-(.{2})-(.{2})-(.{2})\s*(.*)$"
)
_ANGLE_LINE = re.compile(r"^(.{2})-(.{2})-(.{2}) +([0-9.]+) +([0-9.]+)\s*(.*)$")
_BOND_LINE = re.compile(r"^(.{2})-(.{2}) +([0-9.]+) +([0-9.]+)\s*(.*)$")


@dataclass(frozen=True)
class OPLSAtomType:
    number: int
    atomic_number: int
    opls_class: str
    charge_e: float
    sigma_angstrom: float
    epsilon_kcal_mol: float
    comment: str
    source_line: int

    @property
    def name(self) -> str:
        return f"opls_{self.number}"

    @property
    def sigma_nm(self) -> float:
        return self.sigma_angstrom / 10.0

    @property
    def epsilon_kj_mol(self) -> float:
        return self.epsilon_kcal_mol * KCAL_TO_KJ


@dataclass(frozen=True)
class _Parameter:
    """Same layout as ``bonded._Parameter`` (classes, GROMACS values, line)."""

    classes: tuple[str, ...]
    values: tuple[float, ...]
    source_line: int


@lru_cache(maxsize=None)
def load_atom_types(path: str | Path = DEFAULT_PAR) -> dict[int, OPLSAtomType]:
    """Atom types of an ``oplsaa.par`` file, keyed by type number."""
    types: dict[int, OPLSAtomType] = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if "Fourier coefficients" in line:
            break
        match = _TYPE_LINE.match(line)
        if match is None:
            continue
        number = int(match[1])
        if number in types:  # keep the first definition, as for bonded terms
            continue
        types[number] = OPLSAtomType(
            number=number,
            atomic_number=int(match[2]),
            opls_class=match[3],
            charge_e=float(match[4]),
            sigma_angstrom=float(match[5]),
            epsilon_kcal_mol=float(match[6]),
            comment=match[7].strip(),
            source_line=line_number,
        )
    if not types:
        raise AssignmentError(f"No atom types found in {path}")
    return types


def fourier_kcal_to_rb_kj(v1: float, v2: float, v3: float, v4: float) -> tuple[float, ...]:
    """OPLS Fourier V1-V4 (kcal/mol) to GROMACS Ryckaert-Bellemans C0-C5 (kJ/mol)."""
    v1, v2, v3, v4 = (KCAL_TO_KJ * value for value in (v1, v2, v3, v4))
    return (v2 + 0.5 * (v1 + v3), 0.5 * (-v1 + 3 * v3), -v2 + 4 * v4, -2 * v3, -4 * v4, 0.0)


def _class(field: str) -> str:
    return field.strip()


def _general(parameter: "_Parameter") -> bool:
    """X-B-C-X torsion entry (the general term of the central bond)."""
    return parameter.classes[0] == "X" and parameter.classes[3] == "X"


def _pattern_matches(pattern: str, value: str) -> bool:
    """BOSS class match: ``??`` is any class, ``?`` one character (or blank)."""
    if pattern == "??":
        return True
    padded_pattern, padded_value = pattern.ljust(2), value.ljust(2)
    return all(p == "?" or p == v for p, v in zip(padded_pattern, padded_value))


# Improper torsions of the OPLS/2020 file, by the names bonded.py uses.
# GROMACS periodic (funct 4): E = k (1 + cos(n phi - phi_s)); the OPLS term
# V2/2 (1 - cos 2phi) is k = V2/2, phi_s = 180, n = 2.
_IMPROPER_ENTRIES = {
    "improper_O_C_X_Y": ("O", "C", "X", "Y"),
    "improper_Z_N_X_Y": ("Z", "N", "X", "Y"),
    "improper_Z_CA_X_Y": ("Z", "CA", "X", "Y"),
    "improper_Z_CM_X_Y": ("Z", "CM", "X", "Y"),
}
# Allene twist ("allenes improper" CM-X-CM-Y, type 277): applied as the
# dihedral X-C1...C3-Y between substituents on the two allene ends, the only
# reading whose V2 < 0 term has its minimum at the 90-degree allene twist.
_ALLENE_TWIST_COMMENT = "allenes improper"


def _periodic_improper(v2_kcal: float) -> tuple[float, float, float]:
    """(phi_s, k kJ/mol, n) of V2/2 (1 - cos 2phi); a negative V2 is written
    as phi_s = 0 with k = -V2/2 (the same term up to a constant)."""
    k = v2_kcal / 2.0 * KCAL_TO_KJ
    return (180.0, k, 2.0) if k >= 0 else (0.0, -k, 2.0)


class OPLS2020BondedTable:
    """Bonded parameters of OPLS/2020 (values in GROMACS units).

    Duplicates resolve to the first entry, which also selects the
    trans-favouring tertiary-amide CT-N-CT-CT term (type 108) over the
    proline-ring CT-CT-N-CT one (109).
    """

    name = "OPLS/2020"

    def __init__(self, par_path: str | Path = DEFAULT_PAR, sb_path: str | Path = DEFAULT_SB):
        self.path = Path(par_path).resolve()
        self.sb_path = Path(sb_path).resolve()
        for path in (self.path, self.sb_path):
            if not path.is_file():
                raise AssignmentError(f"OPLS/2020 parameter file is unavailable: {path}")
        self.bonds: dict[tuple[str, str], _Parameter] = {}
        self.angles: dict[tuple[str, str, str], _Parameter] = {}
        self.propers: list[_Parameter] = []
        self.impropers: dict[str, _Parameter] = {}
        self.allene_twists: dict[tuple[str, str], _Parameter] = {}  # (X, Y) classes
        self.duplicates: list[tuple[str, tuple[str, ...], int, int]] = []
        self._torsion_types: dict[int, str] = {}  # source line -> BOSS torsion type id
        self._by_central: dict[tuple[str, str], list[_Parameter]] = {}
        self._parse_sb()
        self._parse_par()

    @staticmethod
    def _bond_key(classes: tuple[str, str]) -> tuple[str, str]:
        return min(classes, classes[::-1])

    @staticmethod
    def _angle_key(classes: tuple[str, str, str]) -> tuple[str, str, str]:
        return min(classes, classes[::-1])

    def _store(self, table: dict, key: tuple, parameter: _Parameter, kind: str) -> None:
        if key in table:
            if table[key].values != parameter.values:
                self.duplicates.append((kind, key, table[key].source_line, parameter.source_line))
            return
        table[key] = parameter

    def _parse_sb(self) -> None:
        for line_number, line in enumerate(self.sb_path.read_text().splitlines(), 1):
            if not line or line[0] in "*#\"":
                continue
            match = _ANGLE_LINE.match(line)
            if match:
                classes = tuple(_class(match[i]) for i in (1, 2, 3))
                values = (float(match[5]), float(match[4]) * ANGLE_K_TO_GROMACS)
                self._store(self.angles, self._angle_key(classes),
                            _Parameter(classes, values, line_number), "angle")
                continue
            match = _BOND_LINE.match(line)
            if match:
                classes = (_class(match[1]), _class(match[2]))
                values = (float(match[4]) / 10.0, float(match[3]) * BOND_K_TO_GROMACS)
                self._store(self.bonds, self._bond_key(classes),
                            _Parameter(classes, values, line_number), "bond")

    def _parse_par(self) -> None:
        in_torsions = False
        improper_rows = {}
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            if "Fourier coefficients" in line:
                in_torsions = True
                continue
            if not in_torsions or line.startswith("#"):
                continue
            match = _TORSION_LINE.match(line)
            if match is None:
                continue
            classes = tuple(_class(match[i]) for i in (6, 7, 8, 9))
            v = tuple(float(match[i]) for i in (2, 3, 4, 5))
            comment = match[10]
            if _ALLENE_TWIST_COMMENT in comment.lower():
                twist = (classes[1], classes[3])
                if twist not in self.allene_twists:
                    self.allene_twists[twist] = _Parameter(
                        classes, _periodic_improper(v[1]), line_number)
                continue
            if "improper" in comment.lower():
                improper_rows.setdefault(classes, (v, line_number))
                continue
            # "??" (any class) is stored as "X"; a partial "C?" stays a pattern.
            stored = tuple("X" if c == "??" else c for c in classes)
            self.propers.append(_Parameter(stored, fourier_kcal_to_rb_kj(*v), line_number))
            self._torsion_types[line_number] = match[1]
        for name, classes in _IMPROPER_ENTRIES.items():
            if classes not in improper_rows:
                raise AssignmentError(f"{self.path}: improper {'-'.join(classes)} not found")
            v, line_number = improper_rows[classes]
            self.impropers[name] = _Parameter(classes, _periodic_improper(v[1]), line_number)

    def allene_twist(self, x_class: str, y_class: str) -> _Parameter | None:
        """Twist term for substituents of classes X and Y on the two allene
        ends (either order), or None if the table has none."""
        return self.allene_twists.get((x_class, y_class)) or self.allene_twists.get((y_class, x_class))

    def torsion_type(self, source_line: int) -> str:
        """BOSS torsion type number of the entry at ``source_line``."""
        return self._torsion_types.get(source_line, "")

    # -- lookup ------------------------------------------------------------
    def lookup(self, kind: str, classes: tuple[str, ...]):
        from .bonded import RULE_EXACT, RULE_WILDCARD

        if kind == "bond":
            parameter = self.bonds.get(self._bond_key(classes))
            return None if parameter is None else (parameter, RULE_EXACT)
        if kind == "angle":
            parameter = self.angles.get(self._angle_key(classes))
            return None if parameter is None else (parameter, RULE_EXACT)
        if kind != "proper":
            raise ValueError(f"Unknown bonded interaction kind {kind!r}")
        parameter = self._proper_match(tuple(classes))
        if parameter is None:
            return None
        wildcard = any(c == "X" or "?" in c for c in parameter.classes)
        return parameter, (RULE_WILDCARD if wildcard else RULE_EXACT)

    def proper_entry(
        self, classes: tuple[str, str, str, str], general: bool
    ) -> _Parameter | None:
        """Best torsion entry of one kind: ``general`` X-B-C-X entries only, or
        (``general=False``) every other entry. Used by the parmchk2-style
        substitution (``parmchk.py``), which keeps the two kinds apart."""
        return self._proper_match(tuple(classes), general)

    def has_specific_central(self, b: str, c: str) -> bool:
        """Whether any non-X-B-C-X torsion entry can have centrals B-C."""
        return any(not _general(p) for p in self._central_entries(b, c))

    def specific_entries(self, b: str, c: str) -> list[_Parameter]:
        """Non-X-B-C-X torsion entries whose centrals match B-C (either direction)."""
        return [p for p in self._central_entries(b, c) if not _general(p)]

    def _central_entries(self, b: str, c: str) -> list[_Parameter]:
        """Torsion entries whose central classes match B-C in either direction
        (cached; the lookup only needs to test these)."""
        key = (b, c)
        if key not in self._by_central:
            self._by_central[key] = [
                parameter for parameter in self.propers
                if any(all(_pattern_matches("??" if p == "X" else p, q)
                           for p, q in zip(parameter.classes[1:3], query))
                       for query in ((b, c), (c, b)))
            ]
        return self._by_central[key]

    def _proper_match(
        self, classes: tuple[str, str, str, str], general: bool | None = None
    ) -> _Parameter | None:
        best, best_score = None, None
        for parameter in self._central_entries(classes[1], classes[2]):
            if general is not None and general != _general(parameter):
                continue
            for query in (classes, classes[::-1]):
                if all(_pattern_matches("??" if p == "X" else p, q)
                       for p, q in zip(parameter.classes, query)):
                    wildcards = sum(2 if p == "X" else p.count("?") for p in parameter.classes)
                    score = (wildcards, parameter.source_line)
                    if best_score is None or score < best_score:
                        best, best_score = parameter, score
                    break
        return best

    def bond(self, classes):
        return self._require(self.lookup("bond", classes), "bond", classes)

    def angle(self, classes):
        return self._require(self.lookup("angle", classes), "angle", classes)

    def proper(self, classes):
        return self._require(self.lookup("proper", classes), "proper torsion", classes)

    @staticmethod
    def _require(found, kind, classes):
        if found is None:
            raise AssignmentError(f"No OPLS/2020 {kind} parameter for {classes}")
        return found[0]

    def improper(self, name: str) -> _Parameter:
        try:
            return self.impropers[name]
        except KeyError as exc:
            raise AssignmentError(f"OPLS/2020 improper {name!r} was not defined") from exc


@lru_cache(maxsize=4)
def load_bonded_table(par_path: str | Path = DEFAULT_PAR,
                      sb_path: str | Path = DEFAULT_SB) -> OPLS2020BondedTable:
    return OPLS2020BondedTable(par_path, sb_path)
