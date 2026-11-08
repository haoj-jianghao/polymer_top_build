"""Published localized bond-charge corrections for 1.14*CM1A-LBCC."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


class LBCCError(RuntimeError):
    """Raised when an LBCC definition or application is invalid."""


@dataclass(frozen=True)
class LBCCRule:
    """A directed bond correction expressed as an atom-mapped SMARTS."""

    name: str
    smarts: str
    delta_e: float
    reference_molecule: str

    def query(self) -> Chem.Mol:
        query = Chem.MolFromSmarts(self.smarts)
        if query is None:
            raise LBCCError(f"Invalid SMARTS for LBCC rule {self.name}: {self.smarts}")
        maps = {atom.GetAtomMapNum() for atom in query.GetAtoms()}
        if not {1, 2}.issubset(maps):
            raise LBCCError(f"LBCC rule {self.name} must contain atom maps 1 and 2")
        return query


@dataclass(frozen=True)
class LBCCApplication:
    """One directed correction applied to a specific molecular bond."""

    rule: str
    first_atom_index: int
    second_atom_index: int
    delta_e: float


# Table 1 of Dodda et al., J. Phys. Chem. B 121, 3864-3870 (2017).
# The paper's B_X-Y value is added to X and subtracted from Y.
LBCC_RULES = (
    LBCCRule("HA-CA", "[H:1]-[c:2]", -0.01, "benzene"),
    LBCCRule("CA-NC", "[c:1]~[n;r6:2]", 0.07, "pyridine"),
    LBCCRule(
        "CA-NO",
        "[c:1]-[N;X3:2](~[O;X1])~[O;X1]",
        -0.08,
        "nitrobenzene",
    ),
    LBCCRule("CA-NP", "[c:1]-[N;X3;H2;+0:2]", 0.06, "aniline"),
    LBCCRule("H-NP", "[H:1]-[N;X3;H2;+0:2]", -0.05, "aniline"),
    LBCCRule("CA-F", "[c:1]-[F:2]", 0.13, "fluorobenzene"),
    LBCCRule("CA-Br", "[c:1]-[Br:2]", 0.19, "bromobenzene"),
    LBCCRule("CZA-NZ", "[c]-[C;X2:1]#[N;X1:2]", 0.09, "benzonitrile"),
    LBCCRule("CA-OH", "[c:1]-[O;X2;H1:2]", 0.22, "phenol"),
    LBCCRule(
        "OY-SZ",
        "[O;X1:1]~[S;X3:2]([#6])([#6])",
        0.06,
        "dimethyl sulfoxide",
    ),
    LBCCRule("CT-OH", "[C;X4:1]-[O;X2;H1:2]", 0.10, "ethanol"),
    LBCCRule(
        "C-O",
        "[C;X3;!$(C(=O)N);!$(C(=O)O[#6]):1]=[O;X1:2]",
        0.05,
        "acetone",
    ),
    LBCCRule("CT-S", "[C;X4:1]-[S;X2;H0:2]", 0.08, "dimethyl sulfide"),
    LBCCRule("CT-SH", "[C;X4:1]-[S;X2;H1:2]", 0.175, "ethanethiol"),
    LBCCRule(
        "NO-ON",
        "[N;X3;$(N(~[O;X1])~[O;X1]):1]~[O;X1:2]",
        -0.18,
        "nitroethane",
    ),
    LBCCRule("CT-Cl", "[C;X4:1]-[Cl:2]", 0.10, "ethyl chloride"),
    LBCCRule("CT-Br", "[C;X4:1]-[Br:2]", 0.08, "ethyl bromide"),
    LBCCRule("CT-NP", "[C;X4:1]-[N;X3;H2;+0:2]", 0.04, "ethylamine"),
    LBCCRule("CZT-NZ", "[!a]-[C;X2:1]#[N;X1:2]", 0.03, "acetonitrile"),
)


def find_lbcc_applications(
    molecule: Chem.Mol, rules: tuple[LBCCRule, ...] = LBCC_RULES
) -> list[LBCCApplication]:
    """Return all directed LBCC applications for an explicit-H molecule."""
    applications = []
    seen = set()

    for rule in rules:
        query = rule.query()
        query_map_indices = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in query.GetAtoms()
            if atom.GetAtomMapNum()
        }
        for match in molecule.GetSubstructMatches(query, uniquify=True):
            first = match[query_map_indices[1]]
            second = match[query_map_indices[2]]
            key = (rule.name, first, second)
            if key in seen:
                continue
            seen.add(key)
            if molecule.GetBondBetweenAtoms(first, second) is None:
                raise LBCCError(
                    f"LBCC rule {rule.name} mapped atoms that are not bonded: "
                    f"{first + 1}, {second + 1}"
                )
            applications.append(
                LBCCApplication(
                    rule=rule.name,
                    first_atom_index=first + 1,
                    second_atom_index=second + 1,
                    delta_e=rule.delta_e,
                )
            )

    applications.sort(
        key=lambda item: (item.first_atom_index, item.second_atom_index, item.rule)
    )
    return applications


def apply_lbcc(
    charges: list[float], applications: list[LBCCApplication]
) -> tuple[list[float], list[float]]:
    """Apply equal-and-opposite corrections and return charges and per-atom deltas."""
    corrected = list(charges)
    deltas = [0.0] * len(charges)
    for application in applications:
        first = application.first_atom_index - 1
        second = application.second_atom_index - 1
        if not (0 <= first < len(charges) and 0 <= second < len(charges)):
            raise LBCCError(f"Out-of-range atoms in {application}")
        corrected[first] += application.delta_e
        corrected[second] -= application.delta_e
        deltas[first] += application.delta_e
        deltas[second] -= application.delta_e

    if abs(sum(deltas)) > 1e-12:
        raise LBCCError(f"LBCC corrections changed net charge by {sum(deltas):+.3e} e")
    return corrected, deltas
