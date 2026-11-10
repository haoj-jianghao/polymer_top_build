"""opls-ff-lite: OPLS/2020 + CM1A-LBCC parameterization of organic molecules and
oligomers (neutral or charged) for GROMACS, with MOPAC as the only external program."""

from .assignment import (
    AssignmentError,
    AssignmentResult,
    AtomAssignment,
    assign_molecule,
    load_sdf,
    molecule_from_smiles,
)
from .bonded import (
    AngleInteraction,
    BondedAssignment,
    BondInteraction,
    ImproperTorsionInteraction,
    PairInteraction,
    ProperTorsionInteraction,
    assign_bonded_interactions,
)
from .workflow import GromacsTopologyResult, generate_gromacs_topology

__all__ = [
    "AssignmentError",
    "AssignmentResult",
    "AtomAssignment",
    "assign_molecule",
    "load_sdf",
    "molecule_from_smiles",
    "AngleInteraction",
    "BondedAssignment",
    "BondInteraction",
    "ImproperTorsionInteraction",
    "PairInteraction",
    "ProperTorsionInteraction",
    "assign_bonded_interactions",
    "GromacsTopologyResult",
    "generate_gromacs_topology",
]

__version__ = "0.1.0"
