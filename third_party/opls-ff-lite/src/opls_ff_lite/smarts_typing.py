"""Ordered SMARTS rule tables: the first rule that matches an atom types it.

A rule is an RDKit SMARTS whose atom mapped ``:1`` is the typed atom (an
unmapped pattern types its first atom). Rules are tried in table order and
an atom keeps the first rule that matches it, so specific rules go before
general ones. Used by the GAFF2 fallback (``amber.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem


class RuleError(ValueError):
    """Raised for an invalid rule table."""


@dataclass(frozen=True)
class SmartsRule:
    label: str
    smarts: str
    note: str = ""


@lru_cache(maxsize=None)
def _compile(smarts: str) -> tuple[Chem.Mol, int]:
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise RuleError(f"Invalid SMARTS {smarts!r}")
    mapped = [atom.GetIdx() for atom in query.GetAtoms() if atom.GetAtomMapNum() == 1]
    return query, mapped[0] if mapped else 0


def first_match(
    molecule: Chem.Mol, rules: tuple[SmartsRule, ...], atoms: set[int] | None = None
) -> dict[int, SmartsRule]:
    """Atom index -> first matching rule, for ``atoms`` (default: all atoms).

    Atoms that no rule matches are absent from the result.
    """
    pending = set(range(molecule.GetNumAtoms())) if atoms is None else set(atoms)
    assigned: dict[int, SmartsRule] = {}
    for rule in rules:
        if not pending:
            break
        query, target = _compile(rule.smarts)
        for match in molecule.GetSubstructMatches(query, uniquify=False, maxMatches=100000):
            index = match[target]
            if index in pending:
                assigned[index] = rule
                pending.discard(index)
    return assigned
