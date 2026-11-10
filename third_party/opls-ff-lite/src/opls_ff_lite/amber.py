"""GAFF2 atom types and Lennard-Jones parameters for atoms OPLS cannot type.

LigParGen fills gaps in OPLS-AA typing by borrowing from AMBER. Here an atom
that neither the OPLS SMARTS rules nor the atom-type archive can type gets a GAFF2 atom type
and the GAFF2 Lennard-Jones parameters for that type. Its bonded class is the
GAFF2 type itself (lower case, so it never matches an OPLS class). Bonds and
angles that involve it take gaff2.dat terms of full GAFF2 types
(``gaff2_bonded.py``); torsions the general terms or the QM fit.

Types come from the ordered SMARTS rule table in ``data/gaff2_lj.json``
(no AmberTools needed). The table only separates GAFF2 types whose LJ
parameters differ; types with identical LJ get one representative name.
It was checked against antechamber (``src/scripts/gaff2_reference.py``,
``tests/test_gaff2.py``).

A GAFF2-typed atom with an OPLS/2020 analog takes that type instead
(``gaff2_analogs.py``). In the topology the remaining GAFF2 types are written
under cosmetic names in the unused OPLS number range (``opls_801`` ...,
``data/gaff2_display_types.json``); these are labels, not OPLS/2020 types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from .assignment import AssignmentError
from .smarts_typing import SmartsRule, first_match

GAFF2_TABLE = Path(__file__).resolve().parent / "data" / "gaff2_lj.json"
GAFF2_DISPLAY_TYPES = Path(__file__).resolve().parent / "data" / "gaff2_display_types.json"
# Rmin/2 (angstrom) -> sigma (nm): sigma = 2 Rmin/2 / 2^(1/6)
_RMIN_HALF_TO_SIGMA_NM = 2.0 / 2.0 ** (1.0 / 6.0) / 10.0
KCAL_TO_KJ = 4.184


@dataclass(frozen=True)
class Gaff2Table:
    rules: tuple[SmartsRule, ...]
    lennard_jones: dict[str, tuple[float, float]]  # type -> (sigma nm, epsilon kJ/mol)
    source: str


@lru_cache(maxsize=None)
def load_gaff2_table(path: str | Path = GAFF2_TABLE) -> Gaff2Table:
    payload = json.loads(Path(path).read_text())
    lennard_jones = {
        name: (value["rmin_half_angstrom"] * _RMIN_HALF_TO_SIGMA_NM,
               value["epsilon_kcal_mol"] * KCAL_TO_KJ)
        for name, value in payload["lennard_jones"].items()
    }
    rules = tuple(SmartsRule(r["type"], r["smarts"], r.get("note", "")) for r in payload["rules"])
    unknown = sorted({rule.label for rule in rules} - lennard_jones.keys())
    if unknown:
        raise AssignmentError(f"{path}: rules use types without LJ parameters: {unknown}")
    return Gaff2Table(rules, lennard_jones, payload.get("source", ""))


def gaff2_types(molecule: Chem.Mol, atoms: list[int] | None = None) -> dict[int, str]:
    """GAFF2 type of each atom in ``atoms`` (0-based; default all atoms).

    The molecule must have explicit hydrogens. Raises if an atom matches no rule.
    """
    table = load_gaff2_table()
    wanted = set(range(molecule.GetNumAtoms())) if atoms is None else set(atoms)
    assigned = first_match(molecule, table.rules, wanted)
    missing = sorted(wanted - assigned.keys())
    if missing:
        details = ", ".join(f"{i + 1}:{molecule.GetAtomWithIdx(i).GetSymbol()}" for i in missing)
        raise AssignmentError(f"No GAFF2 rule matches atoms {details}")
    return {index: rule.label for index, rule in assigned.items()}


def gaff2_lennard_jones() -> dict[str, tuple[float, float]]:
    """GAFF2 type -> (sigma nm, epsilon kJ/mol)."""
    return load_gaff2_table().lennard_jones


@lru_cache(maxsize=None)
def gaff2_display_types(path: str | Path = GAFF2_DISPLAY_TYPES) -> dict[str, str]:
    """GAFF2 type -> cosmetic topology name (``opls_801`` ...; not an OPLS type)."""
    return json.loads(Path(path).read_text())["names"]
