"""OpenFF Sage parameterization of the reference oligomer, using the user's own OpenFF install.

OpenFF (openff-toolkit, openff-interchange, the force-field files and, for
Sage 2.3, the NAGL charge model with PyTorch) is never imported by the
builder. ``sage_worker.py`` runs as a subprocess with the Python interpreter
of an existing OpenFF environment, writes a GROMACS topology of the reference
oligomer, and the builder transfers it to the polymer with the same
engine-neutral algorithm as every other engine (``topology_transfer``).

The OpenFF interpreter is taken from, in order:

1. the ``sage_python`` engine setting (CLI ``--sage-python``, GUI field);
2. the ``POLYMER_SAGE_PYTHON`` environment variable;
3. the interpreter running the builder, if it can import OpenFF;
4. conda/mamba environments that have openff-toolkit and openff-interchange
   installed (found under the conda installations in ``CONDA_EXE``,
   ``CONDA_PREFIX``, ``MAMBA_ROOT_PREFIX`` and ~/miniforge3, ~/mambaforge,
   ~/miniconda3, ~/anaconda3).

Each candidate is probed (imports, force field, charge model) and the first
that works is used.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import external_env
from topology_transfer import GromacsTopology, parse_gromacs_topology


PROJECT_ROOT = Path(__file__).resolve().parent
WORKER = PROJECT_ROOT / "sage_worker.py"
ENGINE = "sage"
DEFAULT_FORCE_FIELD = "openff-2.3.0.offxml"
SAGE_PYTHON_ENV = "POLYMER_SAGE_PYTHON"
PROBE_TIMEOUT_S = 600
RUN_TIMEOUT_S = 3600
OPENFF_PACKAGES = ("openff-toolkit", "openff-interchange")


def description(settings: dict | None = None) -> str:
    force_field = (settings or {}).get("sage_forcefield") or DEFAULT_FORCE_FIELD
    return f"OpenFF {force_field} (SMIRNOFF; user's OpenFF installation via Interchange)"


DESCRIPTION = description()


# ---------------------------------------------------------------------------
# Finding an OpenFF interpreter
# ---------------------------------------------------------------------------


def _conda_roots() -> list[Path]:
    return external_env.conda_roots()


def candidate_interpreters(sage_python: str | None = None) -> list[tuple[str, str]]:
    """(python, how it was found) in order of preference."""

    candidates: list[tuple[str, str]] = []
    if sage_python:
        candidates.append((sage_python, "setting sage_python"))
    if os.environ.get(SAGE_PYTHON_ENV):
        candidates.append((os.environ[SAGE_PYTHON_ENV], f"environment variable {SAGE_PYTHON_ENV}"))
    if importlib.util.find_spec("openff") and importlib.util.find_spec("openff.interchange"):
        candidates.append((sys.executable, "the builder's own interpreter"))
    for prefix in external_env.conda_environments_with(OPENFF_PACKAGES, _conda_roots()):
        candidates.append((str(external_env.python_in(prefix)), f"conda environment {prefix}"))
    return external_env.unique(candidates)


def _run_worker(python: str, arguments: list[str], timeout: int) -> tuple[int, dict, str]:
    return external_env.run_worker(python, WORKER, arguments, timeout)


@functools.lru_cache(maxsize=16)
def probe(python: str, force_field: str) -> dict:
    if not Path(python).exists():
        return {"ok": False, "python": python, "errors": [f"{python} does not exist"]}
    try:
        code, payload, stderr = _run_worker(python, ["probe", "--forcefield", force_field], PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "python": python, "errors": [f"could not run {python}: {exc}"]}
    if not payload:
        return {"ok": False, "python": python, "errors": [f"{python} did not answer the probe: {stderr.strip()[-300:]}"]}
    return payload


@dataclass(frozen=True)
class Interpreter:
    python: str
    source: str
    probe: dict


def find_interpreter(settings: dict | None = None) -> tuple[Interpreter | None, list[str]]:
    """The first working OpenFF interpreter, and the problems of the candidates tried."""

    settings = settings or {}
    force_field = settings.get("sage_forcefield") or DEFAULT_FORCE_FIELD
    problems: list[str] = []
    for python, source in candidate_interpreters(settings.get("sage_python")):
        result = probe(python, force_field)
        if result.get("ok"):
            return Interpreter(python, source, result), problems
        problems.append(f"{python} ({source}): " + "; ".join(result.get("errors", ["probe failed"])))
        if source.startswith(("setting", "environment variable")):
            break  # an explicit choice is never silently replaced
    return None, problems


def preflight(settings: dict | None = None) -> tuple[list[str], list[str]]:
    interpreter, problems = find_interpreter(settings)
    if interpreter is None:
        searched = ", ".join(str(root) for root in _conda_roots()) or "no conda installation found"
        errors = [
            "OpenFF interpreter: none found with openff-toolkit, openff-interchange and a usable "
            f"{(settings or {}).get('sage_forcefield') or DEFAULT_FORCE_FIELD}. Point the builder at your "
            f"OpenFF environment's Python with --sage-python PATH or {SAGE_PYTHON_ENV}=PATH "
            f"(conda installations searched: {searched})."
        ]
        errors += [f"  tried {problem}" for problem in problems]
        return errors, []
    warnings = []
    charges = interpreter.probe.get("charges", {})
    if charges.get("method", "").startswith("AM1-BCC"):
        warnings.append(
            "this force field uses toolkit AM1-BCC charges (slow, needs AmberTools or OpenEye); "
            f"{DEFAULT_FORCE_FIELD} uses fast NAGL charges"
        )
    return [], warnings


def summarize_problems(errors: list[str]) -> list[str]:
    return ["OpenFF interpreter (openff-toolkit + openff-interchange)"] if errors else []


def missing_requirements(settings: dict | None = None) -> list[str]:
    return summarize_problems(preflight(settings)[0])


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------


@dataclass
class ReferenceParameterization:
    topology: GromacsTopology
    files: list[Path]
    info: dict = field(default_factory=dict)


def topology_from_worker_output(top_path: Path, worker_info: dict) -> GromacsTopology:
    """Read the Interchange topology and label atoms by their Sage vdW parameter id.

    Interchange writes one atom type per atom with no bond_type column; the
    vdW parameter id (e.g. ``n16``) becomes both the readable label and the
    bond_type, so atoms with identical vdW parameters can be merged.
    """

    topology = parse_gromacs_topology(top_path)
    labels = list(worker_info["vdw_parameter_ids"])
    if len(labels) != len(topology.atoms):
        raise RuntimeError(
            f"Sage worker labelled {len(labels)} atoms, the topology has {len(topology.atoms)}."
        )
    for atom, label in zip(topology.atoms, labels):
        atom_type = topology.atomtypes[atom.type_name]
        topology.atomtypes[atom.type_name] = dataclasses.replace(atom_type, bond_type=label)
    topology.labels = labels
    return topology


def parameterize_reference(
    structure_sdf: Path,
    outdir: Path,
    *,
    net_charge: int,
    name: str = "reference",
    settings: dict | None = None,
) -> ReferenceParameterization:
    settings = settings or {}
    force_field = settings.get("sage_forcefield") or DEFAULT_FORCE_FIELD
    interpreter, problems = find_interpreter(settings)
    if interpreter is None:
        raise RuntimeError("No usable OpenFF interpreter:\n" + "\n".join(problems))
    outdir.mkdir(parents=True, exist_ok=True)
    code, payload, stderr = _run_worker(
        interpreter.python,
        [
            "run",
            "--input", str(structure_sdf),
            "--output-dir", str(outdir),
            "--name", name,
            "--forcefield", force_field,
            "--net-charge", str(net_charge),
        ],
        RUN_TIMEOUT_S,
    )
    if code != 0 or not payload.get("ok"):
        message = payload.get("error") or stderr.strip()[-1000:] or f"exit code {code}"
        if "Unassigned" in message and "no parameters for element" not in message:
            message += f"\n{force_field} has no parameters for part of this molecule; try the OPLS-AA/2020 engine."
        raise RuntimeError(f"OpenFF/Sage failed on the reference oligomer ({interpreter.python}): {message}")
    worker_info = json.loads((outdir / f"{name}.sage.json").read_text())
    top_path = outdir / f"{name}.top"
    topology = topology_from_worker_output(top_path, worker_info)
    charges = worker_info.get("charges", {})
    charge_model = charges.get("method", "unknown") + (f" ({charges['model']})" if charges.get("model") else "")
    info = {
        "engine": ENGINE,
        "engine_description": description(settings),
        "force_field": force_field,
        "openff_python": interpreter.python,
        "openff_python_source": interpreter.source,
        "openff_versions": worker_info.get("versions", {}),
        "charge_model": charge_model,
        "reference_charge_sum": worker_info.get("charge_sum"),
        "atom_type_rules": _count(worker_info["vdw_parameter_ids"]),
        "engine_warnings": [worker_info.get("constrained_bonds_note", "")],
        "reference_topology": str(top_path),
    }
    return ReferenceParameterization(topology, [top_path, outdir / f"{name}.sage.json"], info)


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
