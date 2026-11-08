"""parmchk2-style substitution for proper torsions missing from OPLS/2020.

Follows ``chk_torsion`` of AmberTools 26 ``parmchk2.c`` (antechamber), with
the atom-type correspondence scores of its ``PARMCHK.DAT`` (copied unchanged
to ``data/parmchk/``). Only the uppercase (AMBER) atom types of that file are
used: their names are the OPLS classes, so an OPLS class takes the
correspondences of the AMBER type with the same name. OPLS classes without
such a type (C=, CS, CU, NO, SY, ...) and GAFF2-typed atoms correspond only
to themselves.

A torsion A-B-C-D that the table lookup (exact, ``?`` patterns, X-B-C-X)
misses goes through parmchk2's remaining steps, in order:

2. ``parmchk2-equa``: a specific entry with equal types (``EQUA``) at any
   position; penalty 0, the first hit in loop order wins.
4. ``parmchk2-equa-general``: an X-B'-C'-X entry with equal central types;
   penalty 0.
5. ``parmchk2-corr``: a specific entry with corresponding types (``CORR``) at
   any positions; penalty = tor(A) + 10 ctor(B) + 10 ctor(C) + tor(D) (+ the
   group and equivalent-type terms); the lowest penalty wins.
6. ``parmchk2-corr-general``: an X-B'-C'-X entry with corresponding central
   types; penalty = 10 ctor(B) + 10 ctor(C) (+ group and equivalent-type
   terms).

Steps 1 and 3 of parmchk2 (the exact entry, the general X-B-C-X entry) are
the table lookup itself. "Specific" is every table entry that is not X-B-C-X
(OPLS/2020 also has one-ended and ``?`` patterns, which parmchk2's GAFF tables
do not). As in ``read_parmchk_parm``, a missing (-1) score takes the file's
default and the central score is ctor = FRACT1 ctor + FRACT2 general score.
Among equal penalties the first combination in parmchk2's loop order wins
(its ``score < bestscore`` test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_PARMCHK = Path(__file__).resolve().parent / "data" / "parmchk" / "PARMCHK.DAT"

RULE_EQUA = "parmchk2-equa"
RULE_EQUA_GENERAL = "parmchk2-equa-general"
RULE_CORR = "parmchk2-corr"
RULE_CORR_GENERAL = "parmchk2-corr-general"
RULES = (RULE_EQUA, RULE_EQUA_GENERAL, RULE_CORR, RULE_CORR_GENERAL)


@dataclass(frozen=True)
class _Corr:
    atomtype: str
    tor: float  # outer-atom torsion score
    ctor: float  # central-atom torsion score (after the default mixing)
    kind: int  # 0 itself, 1 equal type, 2 corresponding type


@dataclass
class _Type:
    atomtype: str
    group: int
    equtype: int
    equa: list[str] = field(default_factory=list)  # [itself, EQUA ...]
    corr: list[_Corr] = field(default_factory=list)  # [itself, EQUA ..., CORR ...]


@dataclass(frozen=True)
class Substitution:
    parameter: object  # opls2020._Parameter
    rule: str
    penalty: float


def _uppercase(atomtype: str) -> bool:
    return atomtype[:1].isupper()


class ParmchkScores:
    """Atom-type equalities and correspondence scores of a PARMCHK.DAT."""

    def __init__(self, path: str | Path = DEFAULT_PARMCHK, uppercase_only: bool = True):
        """``uppercase_only`` keeps the AMBER types (the OPLS classes); False
        keeps every type, as parmchk2 itself does (GAFF types included)."""
        self.path = Path(path).resolve()
        keep = _uppercase if uppercase_only else (lambda atomtype: True)
        weights, defaults, raw = {}, {}, []
        for line in self.path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "PARM":
                raw.append([fields[1], int(fields[3]), int(fields[5]), [], []])
            elif fields[0] == "EQUA":
                raw[-1][3].append(fields[1])
            elif fields[0] == "CORR":
                # sscanf order: bl blf cba cbaf ba baf ctor tor ps
                values = [float(v) for v in fields[2:11]] + [0.0] * (9 - len(fields[2:11]))
                raw[-1][4].append((fields[1], values[6], values[7], values[8]))
            elif fields[0].startswith("WEIGHT_"):
                weights[fields[0][7:]] = float(fields[1])
            elif fields[0].startswith("DEFAULT_"):
                defaults[fields[0][8:]] = float(fields[1])
        self.tor_ctr = weights["TOR_CTR"]
        self.group_weight = weights["GROUP"]
        self.equtype_weight = weights["EQUTYPE"]
        fract1, fract2 = defaults["FRACT1"], defaults["FRACT2"]

        def central(ctor: float, ps: float) -> float:
            return (defaults["TOR_CTR"] if ctor < 0 else ctor) * fract1 + ps * fract2

        self.types: dict[str, _Type] = {}
        for atomtype, group, equtype, equa, corr in raw:
            if not keep(atomtype) or atomtype in self.types:
                continue
            entry = _Type(atomtype, group, equtype, [atomtype],
                          [_Corr(atomtype, 0.0, 0.0, 0)])
            for name in equa:
                if keep(name):
                    entry.equa.append(name)
                    entry.corr.append(_Corr(name, 0.0, 0.0, 1))
            for name, ctor, tor, ps in corr:
                if keep(name):
                    entry.corr.append(_Corr(
                        name, defaults["TOR"] if tor < 0 else tor, central(ctor, ps), 2))
            self.types[atomtype] = entry
        # Targets must be types of the file (parmchk2 resolves them by name).
        for entry in self.types.values():
            entry.equa = [name for name in entry.equa if name in self.types]
            entry.corr = [c for c in entry.corr if c.atomtype in self.types]

    def _type(self, atomtype: str) -> _Type:
        """The file's type, or one that corresponds only to itself (group 0,
        the group of every uppercase type, so it adds no group penalty)."""
        return self.types.get(atomtype) or _Type(
            atomtype, 0, 0, [atomtype], [_Corr(atomtype, 0.0, 0.0, 0)])

    def _group(self, atomtype: str) -> int:
        return self._type(atomtype).group

    def _equtype_penalty(self, b: str, c: str, b_new: str, c_new: str) -> float:
        n1, n2 = self._type(b).equtype, self._type(c).equtype
        n3, n4 = self._type(b_new).equtype, self._type(c_new).equtype
        if n1 == 0 and n2 == 0:
            return 0.0
        tn1, tn2 = abs(n1) + abs(n2), abs(n3) + abs(n4)
        if (tn2 == 3) != (tn1 == 3):
            return self.equtype_weight
        if n1 + n2 == 0 and n3 < 0 and n4 < 0:
            return 0.5 * self.equtype_weight
        return 0.0

    def substitute(self, table, classes: tuple[str, str, str, str]) -> Substitution | None:
        """parmchk2 steps 2, 4, 5 and 6 for a torsion the table lacks."""
        a, b, c, d = (self._type(x) for x in classes)

        # Step 2: equal types, specific entries; first hit wins.
        for b2 in b.equa:
            for c2 in c.equa:
                for a2 in a.equa:
                    for d2 in d.equa:
                        if (a2, b2, c2, d2) == tuple(classes):
                            continue
                        found = table.proper_entry((a2, b2, c2, d2), general=False)
                        if found is not None:
                            return Substitution(found, RULE_EQUA, 0.0)
        # Step 4: equal central types, general entries.
        for b2 in b.equa:
            for c2 in c.equa:
                if (b2, c2) == tuple(classes[1:3]):
                    continue
                found = table.proper_entry(("X", b2, c2, "X"), general=True)
                if found is not None:
                    return Substitution(found, RULE_EQUA_GENERAL, 0.0)
        # Step 5: corresponding types, specific entries; lowest penalty wins.
        candidates = []
        for m, ca in enumerate(a.corr):
            for n, cb in enumerate(b.corr):
                for p, cc in enumerate(c.corr):
                    if not table.has_specific_central(cb.atomtype, cc.atomtype):
                        continue
                    for q, cd in enumerate(d.corr):
                        if max(ca.kind, cb.kind, cc.kind, cd.kind) <= 1:
                            continue
                        score = (ca.tor + cb.ctor * self.tor_ctr + cc.ctor * self.tor_ctr
                                 + cd.tor + self.group_weight
                                 + self._equtype_penalty(b.atomtype, c.atomtype,
                                                         cb.atomtype, cc.atomtype))
                        if (self._group(ca.atomtype) == self._group(cb.atomtype)
                                == self._group(cc.atomtype) == self._group(cd.atomtype)):
                            score -= self.group_weight
                        candidates.append((score, (m, n, p, q),
                                           (ca.atomtype, cb.atomtype, cc.atomtype, cd.atomtype)))
        for score, _, quartet in sorted(candidates):
            found = table.proper_entry(quartet, general=False)
            if found is not None:
                return Substitution(found, RULE_CORR, score)
        # Step 6: corresponding central types, general entries.
        candidates = []
        for n, cb in enumerate(b.corr):
            for p, cc in enumerate(c.corr):
                if max(cb.kind, cc.kind) <= 1:
                    continue
                score = (cb.ctor * self.tor_ctr + cc.ctor * self.tor_ctr
                         + self._equtype_penalty(b.atomtype, c.atomtype,
                                                 cb.atomtype, cc.atomtype))
                if self._group(cb.atomtype) != self._group(cc.atomtype):
                    score += self.group_weight
                candidates.append((score, (n, p), ("X", cb.atomtype, cc.atomtype, "X")))
        for score, _, quartet in sorted(candidates):
            found = table.proper_entry(quartet, general=True)
            if found is not None:
                return Substitution(found, RULE_CORR_GENERAL, score)
        return None


@lru_cache(maxsize=None)
def load_parmchk_scores(path: str | Path = DEFAULT_PARMCHK,
                        uppercase_only: bool = True) -> ParmchkScores:
    return ParmchkScores(path, uppercase_only)
