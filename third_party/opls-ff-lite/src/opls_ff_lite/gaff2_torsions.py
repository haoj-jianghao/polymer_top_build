"""Supplementary general torsions (X-B-C-X) converted from GAFF2.

``data/gaff2_general_torsions.par`` (written by
``src/scripts/gaff2_general_torsions.py``; not part of the OPLS/2020 SI) holds
one X-B-C-X term per OPLS central class pair that OPLS/2020 has no general
term for: the GAFF2 general term used about the same bonds, converted to OPLS
Fourier V1-V4 per quartet. ``bonded.py`` consults it after the OPLS/2020
lookup and before the parmchk2 type substitution (parmchk2's step 3 comes
before its type substitutions).

A term whose comment contains ``ring bonds only`` applies only to torsions
about a ring bond (there it comes before a term for every bond of the pair):
the xTB refit of CA-CU on fused pyrazolo-pyrimidine rings would be wrong on a
rotatable aryl-pyrazole CA-CU bond.

Terms marked ``qm-check`` (GAFF2 itself only has them by type substitution),
and the OPLS/2020 general terms listed as ``# qm-check-opls:``, are flagged
for a QM check: they are used, and the ITP line and a warning say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .opls2020 import _TORSION_LINE, _class, fourier_kcal_to_rb_kj

DEFAULT_GAFF2_GENERAL = Path(__file__).resolve().parent / "data" / "gaff2_general_torsions.par"
RULE_GAFF2_GENERAL = "gaff2-general"
RING_ONLY = "ring bonds only"
_QM_CHECK_OPLS = re.compile(r"^# qm-check-opls: (.{2})-(.{2})-(.{2})-(.{2})")


@dataclass(frozen=True)
class GeneralTorsion:
    classes: tuple[str, str, str, str]  # X, B, C, X
    values: tuple[float, ...]  # RB C0-C5, kJ/mol
    source_line: int
    comment: str
    qm_check: bool
    ring_only: bool = False


class Gaff2GeneralTorsions:
    def __init__(self, path: str | Path = DEFAULT_GAFF2_GENERAL):
        self.path = Path(path).resolve()
        self.by_central: dict[tuple[str, str], GeneralTorsion] = {}
        self.by_central_ring: dict[tuple[str, str], GeneralTorsion] = {}  # ring bonds only
        self.qm_check_opls: set[tuple[str, str]] = set()  # OPLS/2020 X-B-C-X centrals
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            flagged = _QM_CHECK_OPLS.match(line)
            if flagged:
                b, c = _class(flagged[2]), _class(flagged[3])
                self.qm_check_opls |= {(b, c), (c, b)}
                continue
            match = _TORSION_LINE.match(line)
            if match is None:
                continue
            classes = tuple(_class(match[i]) for i in (6, 7, 8, 9))
            v = tuple(float(match[i]) for i in (2, 3, 4, 5))
            comment = match[10]
            ring_only = RING_ONLY in comment
            term = GeneralTorsion(classes, fourier_kcal_to_rb_kj(*v), line_number, comment,
                                  comment.rstrip().endswith("qm-check"), ring_only)
            target = self.by_central_ring if ring_only else self.by_central
            for key in ((classes[1], classes[2]), (classes[2], classes[1])):
                target.setdefault(key, term)

    def lookup(self, classes: tuple[str, str, str, str],
               ring: bool = False) -> GeneralTorsion | None:
        """The X-B-C-X term for the central classes of A-B-C-D, if any; ``ring``:
        the central bond is in a ring (ring-only terms first)."""
        key = (classes[1], classes[2])
        if ring and key in self.by_central_ring:
            return self.by_central_ring[key]
        return self.by_central.get(key)


@lru_cache(maxsize=None)
def load_gaff2_general(path: str | Path = DEFAULT_GAFF2_GENERAL) -> Gaff2GeneralTorsions:
    return Gaff2GeneralTorsions(path)
