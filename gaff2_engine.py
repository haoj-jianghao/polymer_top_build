"""GAFF2/AM1-BCC parameterization of the reference oligomer, using the user's own AmberTools.

AmberTools (antechamber, sqm, parmchk2, tleap) and ParmEd are never imported
by the builder. ``amber_worker.py`` runs as a subprocess with the Python of
an AmberTools environment, parameterizes the reference oligomer and writes
its GROMACS topology; the builder transfers it to the polymer with the same
engine-neutral algorithm as every other engine (``topology_transfer``).

The AmberTools Python is taken from, in order:

1. the ``amber_python`` engine setting (CLI ``--amber-python``, GUI field);
2. the ``POLYMER_AMBER_PYTHON`` environment variable;
3. ``$AMBERHOME/bin/python``;
4. the interpreter running the builder, if it can import ParmEd and
   ``antechamber`` is on PATH;
5. conda/mamba environments with the ``ambertools`` package.

Each candidate is probed (executables, ParmEd, leaprc.gaff2) and the first
that works is used; an explicit choice (1, 2) is never silently replaced.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import external_env
from topology_transfer import GromacsTopology, parse_gromacs_topology


PROJECT_ROOT = Path(__file__).resolve().parent
WORKER = PROJECT_ROOT / "amber_worker.py"
ENGINE = "gaff2"
AMBER_PYTHON_ENV = "POLYMER_AMBER_PYTHON"
AMBERTOOLS_PACKAGES = ("ambertools",)
PROBE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 6 * 3600  # AM1-BCC (sqm) on a large reference oligomer can be slow
DESCRIPTION = "GAFF2 with AM1-BCC charges (user's AmberTools: antechamber, parmchk2, tleap; ParmEd)"


def description(settings: dict | None = None) -> str:
    return DESCRIPTION


def _conda_roots() -> list[Path]:
    return external_env.conda_roots()


def candidate_interpreters(amber_python: str | None = None) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if amber_python:
        candidates.append((amber_python, "setting amber_python"))
    if os.environ.get(AMBER_PYTHON_ENV):
        candidates.append((os.environ[AMBER_PYTHON_ENV], f"environment variable {AMBER_PYTHON_ENV}"))
    if os.environ.get("AMBERHOME"):
        python = external_env.python_in(Path(os.environ["AMBERHOME"]))
        if python.exists():
            candidates.append((str(python), "AMBERHOME"))
    if importlib.util.find_spec("parmed") and shutil.which("antechamber"):
        candidates.append((sys.executable, "the builder's own interpreter (antechamber on PATH)"))
    for prefix in external_env.conda_environments_with(AMBERTOOLS_PACKAGES, _conda_roots()):
        candidates.append((str(external_env.python_in(prefix)), f"conda environment {prefix}"))
    return external_env.unique(candidates)


@functools.lru_cache(maxsize=16)
def probe(python: str) -> dict:
    if not Path(python).exists():
        return {"ok": False, "python": python, "errors": [f"{python} does not exist"]}
    try:
        code, payload, stderr = external_env.run_worker(python, WORKER, ["probe"], PROBE_TIMEOUT_S)
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
    settings = settings or {}
    problems: list[str] = []
    for python, source in candidate_interpreters(settings.get("amber_python")):
        result = probe(python)
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
            "AmberTools: no environment found with antechamber, parmchk2, tleap, ParmEd and "
            f"leaprc.gaff2. Point the builder at your AmberTools environment's Python with "
            f"--amber-python PATH or {AMBER_PYTHON_ENV}=PATH (conda installations searched: {searched})."
        ]
        errors += [f"  tried {problem}" for problem in problems]
        return errors, []
    return [], []


def summarize_problems(errors: list[str]) -> list[str]:
    return ["AmberTools environment (antechamber, parmchk2, tleap, ParmEd)"] if errors else []


def missing_requirements(settings: dict | None = None) -> list[str]:
    return summarize_problems(preflight(settings)[0])


@dataclass
class ReferenceParameterization:
    topology: GromacsTopology
    files: list[Path]
    info: dict = field(default_factory=dict)


def topology_from_worker_output(top_path: Path, worker_info: dict) -> GromacsTopology:
    """Read the ParmEd topology; atoms are labelled by their GAFF2 type."""

    topology = parse_gromacs_topology(top_path)
    labels = list(worker_info["gaff2_types"])
    if len(labels) != len(topology.atoms):
        raise RuntimeError(f"AmberTools typed {len(labels)} atoms, the topology has {len(topology.atoms)}.")
    mismatched = [
        index for index, (atom, label) in enumerate(zip(topology.atoms, labels), start=1) if atom.type_name != label
    ]
    if mismatched:
        raise RuntimeError(f"ParmEd atom types differ from the antechamber GAFF2 types at atoms {mismatched[:10]}.")
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
    from polymer_from_oligomer import read_sdf, write_mol2  # local: avoids an import cycle

    settings = settings or {}
    interpreter, problems = find_interpreter(settings)
    if interpreter is None:
        raise RuntimeError("No usable AmberTools environment:\n" + "\n".join(problems))
    outdir.mkdir(parents=True, exist_ok=True)
    input_mol2 = outdir / f"{name}_antechamber_input.mol2"
    write_mol2(read_sdf(structure_sdf), input_mol2, residue_name="MOL", molecule_name=name)
    code, payload, stderr = external_env.run_worker(
        interpreter.python,
        WORKER,
        ["run", "--input", str(input_mol2), "--output-dir", str(outdir), "--name", name, "--net-charge", str(net_charge)],
        RUN_TIMEOUT_S,
    )
    if code != 0 or not payload.get("ok"):
        message = payload.get("error") or stderr.strip()[-1000:] or f"exit code {code}"
        raise RuntimeError(f"AmberTools failed on the reference oligomer ({interpreter.python}): {message}")
    worker_info = json.loads((outdir / f"{name}.amber.json").read_text())
    top_path = outdir / f"{name}.top"
    topology = topology_from_worker_output(top_path, worker_info)
    notes = []
    if worker_info.get("frcmod_terms_estimated"):
        notes.append(
            f"parmchk2 estimated {worker_info['frcmod_terms_estimated']} term(s) missing from gaff2.dat "
            f"(see {outdir / (name + '.frcmod')})"
        )
    info = {
        "engine": ENGINE,
        "engine_description": DESCRIPTION,
        "amber_python": interpreter.python,
        "amber_python_source": interpreter.source,
        "amber_tools": worker_info.get("tools"),
        "parmed_version": worker_info.get("parmed"),
        "charge_model": worker_info.get("charge_model"),
        "reference_charge_sum": worker_info.get("charge_sum"),
        "atom_type_rules": _count(worker_info["gaff2_types"]),
        "engine_warnings": notes,
        "reference_topology": str(top_path),
    }
    files = [
        input_mol2,
        outdir / f"{name}_typed.mol2",
        outdir / f"{name}.frcmod",
        outdir / f"{name}.prmtop",
        outdir / f"{name}.inpcrd",
        top_path,
        outdir / f"{name}.amber.json",
    ]
    return ReferenceParameterization(topology, files, info)


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
