"""OPLS-AA/2020 parameterization of the reference oligomer with opls-ff-lite.

opls-ff-lite is bundled in ``third_party/opls-ff-lite`` (copied by
``scripts/sync_opls_ff_lite.sh``; see its INTERFACE.md). It needs Python 3.12+,
RDKit >= 2025.9, NumPy and MOPAC, plus the OPLS/2020 SI files, which every
user installs once with

    python third_party/opls-ff-lite/run_opls_ff_lite.py install-si \\
        jp3c06602_si_002.txt jp3c06602_si_003.txt

This module only runs the engine on the reference oligomer and reads back its
GROMACS topology; the transfer to the target polymer is engine-neutral
(``topology_transfer``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from topology_transfer import GromacsTopology, parse_gromacs_topology


PROJECT_ROOT = Path(__file__).resolve().parent
LITE_ROOT = PROJECT_ROOT / "third_party" / "opls-ff-lite"
LITE_SOURCE = LITE_ROOT / "src"
ENGINE = "opls"
DESCRIPTION = "OPLS-AA/2020 + CM1A/1.14*CM1A-LBCC charges (opls-ff-lite)"
INSTALL_SI_HINT = (
    f"{sys.executable} {LITE_ROOT / 'run_opls_ff_lite.py'} install-si "
    "jp3c06602_si_002.txt jp3c06602_si_003.txt"
)


def _import_lite():
    if sys.version_info < (3, 12):
        raise ImportError(
            f"opls-ff-lite needs Python 3.12 or newer; {sys.executable} is {sys.version.split()[0]}"
        )
    source = str(LITE_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    import opls_ff_lite

    return opls_ff_lite


def preflight() -> tuple[list[str], list[str]]:
    """(errors, warnings): everything opls-ff-lite needs before a run."""

    try:
        _import_lite()
        from opls_ff_lite.tools import preflight as lite_preflight
    except ImportError as exc:
        return [f"opls-ff-lite cannot be imported ({exc}); it needs Python 3.12+, RDKit >= 2025.9 and NumPy"], []
    return lite_preflight()


def summarize_problems(errors: list[str]) -> list[str]:
    """Short names of the missing requirements (for the GUI and CLI summary)."""

    names = []
    for error in errors:
        head = error.splitlines()[0]
        if head.startswith("MOPAC"):
            names.append("MOPAC")
        elif "SI" in head or "oplsaa2020" in head or "jp3c06602" in head:
            names.append("OPLS/2020 SI files")
        elif head.startswith("package data"):
            names.append("opls-ff-lite package data")
        elif "cannot be imported" in head:
            names.append("opls-ff-lite (Python 3.12+, RDKit, NumPy)")
        else:
            names.append(head[:80])
    return list(dict.fromkeys(names))


def missing_requirements() -> list[str]:
    return summarize_problems(preflight()[0])


@dataclass
class ReferenceParameterization:
    topology: GromacsTopology
    files: list[Path]
    info: dict = field(default_factory=dict)


def parameterize_reference(
    structure_sdf: Path,
    outdir: Path,
    *,
    net_charge: int,
    name: str = "reference",
) -> ReferenceParameterization:
    """Run opls-ff-lite on an explicit-H, 3D structure; atom order is kept."""

    lite = _import_lite()
    from opls_ff_lite import AssignmentError, generate_gromacs_topology
    from opls_ff_lite.structure import prepare_structure

    try:
        molecule = prepare_structure(path=structure_sdf, name=name, optimize=None, net_charge=net_charge)
        result = generate_gromacs_topology(molecule, outdir, name=name)
    except AssignmentError as exc:
        raise RuntimeError(f"opls-ff-lite failed on the reference oligomer: {exc}") from exc

    topology = parse_gromacs_topology(result.top_path)
    if len(topology.atoms) != molecule.GetNumAtoms():
        raise RuntimeError(
            f"opls-ff-lite topology has {len(topology.atoms)} atoms, the structure {molecule.GetNumAtoms()}."
        )
    topology.labels = [atom.opls_type for atom in result.assignment.atoms]

    assignment = result.assignment
    files = [result.itp_path, result.top_path, result.gro_path, result.topology_json_path]
    files += [path for path in (assignment.json_path, assignment.csv_path, assignment.sdf_path) if path]
    info = {
        "engine": ENGINE,
        "engine_description": DESCRIPTION,
        "opls_ff_lite_version": getattr(lite, "__version__", "unknown"),
        "charge_model": assignment.charge_model,
        "reference_net_charge": assignment.net_charge_e,
        "reference_charge_sum": assignment.final_charge_sum_e,
        "bonded_rule_counts": result.bonded.rule_counts(),
        "atom_type_rules": _count(atom.type_rule.split(":")[0] for atom in assignment.atoms),
        "engine_warnings": list(result.bonded.warnings),
        "reference_topology": str(result.top_path),
    }
    return ReferenceParameterization(topology=topology, files=[Path(path) for path in files], info=info)


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
