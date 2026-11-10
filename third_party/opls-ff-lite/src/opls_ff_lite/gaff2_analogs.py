"""OPLS/2020 analog types for atoms OPLS cannot type.

An atom that neither the OPLS SMARTS rules nor the atom-type archive types
has a GAFF2 type (``amber.py``). ``data/gaff2_opls_analogs.json`` maps such
an atom, by its GAFF2 type and a SMARTS context, to an OPLS/2020 type of a
class with the same element, hybridization and bond order (e.g. the S of a
sulfonic acid to the sulfone S). The atom then has that type's class and
Lennard-Jones; its charge is CM1A as for every atom. Atoms without a listed
analog (C=S, C=N, N-N, N=O, 5-ring heteroaromatics, ...) keep GAFF2.

Heavy atoms are mapped first. A rule marked ``parent_typed`` maps a hydrogen
only if its heavy atom then has an OPLS type, so a group that stays GAFF2
keeps its hydrogens.

For bonded terms an analog atom keeps its GAFF2 type: a term the OPLS/2020
table lacks is looked up further (gaff2.dat, GAFF2 general torsions,
substitution, fallbacks) with the GAFF2 type (``analog_gaff2_classes``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from .smarts_typing import _compile

GAFF2_OPLS_ANALOGS = Path(__file__).resolve().parent / "data" / "gaff2_opls_analogs.json"


@dataclass(frozen=True)
class AnalogRule:
    name: str
    gaff2: frozenset[str]  # GAFF2 types the rule applies to
    smarts: str  # atom mapped :1 is the typed atom
    opls_type: str
    parent_typed: bool = False
    note: str = ""


@lru_cache(maxsize=None)
def load_analog_rules(path: str | Path = GAFF2_OPLS_ANALOGS) -> tuple[AnalogRule, ...]:
    payload = json.loads(Path(path).read_text())
    return tuple(
        AnalogRule(r["name"], frozenset(r["gaff2"]), r["smarts"], r["type"],
                   r.get("parent_typed", False), r.get("note", ""))
        for r in payload["rules"]
    )


def analog_types(molecule: Chem.Mol, gaff_types: dict[int, str]) -> dict[int, AnalogRule]:
    """Atom index -> analog rule, for the GAFF2-typed atoms ``gaff_types``
    (index -> GAFF2 type) that have an OPLS/2020 analog."""
    rules = load_analog_rules()
    assigned: dict[int, AnalogRule] = {}
    for parent_pass in (False, True):
        for rule in rules:
            if rule.parent_typed != parent_pass:
                continue
            query, target = _compile(rule.smarts)
            for match in molecule.GetSubstructMatches(query, uniquify=False):
                index = match[target]
                if index in assigned or gaff_types.get(index) not in rule.gaff2:
                    continue
                if parent_pass and any(
                    neighbor.GetIdx() in gaff_types and neighbor.GetIdx() not in assigned
                    for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
                ):
                    continue
                assigned[index] = rule
    return assigned


def analog_gaff2_classes(molecule: Chem.Mol, type_rules: list[str]) -> list[str | None]:
    """GAFF2 type of each analog-typed atom (``type_rule`` ``analog:<rule>``),
    None for the others: the classes ``assign_bonded_interactions`` uses for
    terms the OPLS/2020 table lacks."""
    from .amber import gaff2_types

    analogs = [index for index, rule in enumerate(type_rules) if rule.startswith("analog:")]
    found = gaff2_types(molecule, analogs) if analogs else {}
    return [found.get(index) for index in range(len(type_rules))]
