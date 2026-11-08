#!/usr/bin/env python
"""Apply an OpenFF SMIRNOFF force field (Sage) to one molecule and write GROMACS files.

This script runs inside the user's own OpenFF environment, never inside the
polymer builder's: the builder calls it with that environment's Python
(``sage_engine.py``), so OpenFF, PyTorch and the NAGL charge models do not
have to be installed next to the builder. It imports only OpenFF and RDKit
(which OpenFF requires) and nothing from the builder.

    python sage_worker.py probe --forcefield openff-2.3.0.offxml
        -> JSON on stdout: versions, whether the force field and its charge
           model are usable
    python sage_worker.py run --input ref.sdf --output-dir DIR --name NAME \\
                              --forcefield openff-2.3.0.offxml --net-charge Q
        -> DIR/NAME.top (GROMACS, from Interchange) and DIR/NAME.sage.json
           (versions, charge method, per-atom vdW parameter ids)

The input SDF must contain all hydrogens; atom order is preserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
import warnings
from pathlib import Path

# Elements the Sage line of force fields has parameters for.
SAGE_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def _quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def _version(module_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(module_name)
    except Exception:
        return None


def _charge_method(force_field) -> dict:
    """Which handler assigns the partial charges, and whether it is usable here."""

    handlers = set(force_field.registered_parameter_handlers)
    if "NAGLCharges" in handlers:
        handler = force_field.get_parameter_handler("NAGLCharges")
        info = {"method": "NAGL", "model": handler.model_file, "usable": True}
        try:
            from openff.nagl_models import validate_nagl_model_path

            validate_nagl_model_path(handler.model_file)
        except Exception as exc:  # model or openff-nagl-models missing
            info.update(usable=False, problem=f"NAGL model {handler.model_file} unavailable: {exc}")
        return info
    if "ToolkitAM1BCC" in handlers:
        return {"method": "AM1-BCC (toolkit; AmberTools sqm or OpenEye)", "model": None, "usable": True}
    return {"method": "library charges only", "model": None, "usable": True}


def probe(forcefield: str) -> dict:
    result: dict = {"ok": False, "python": sys.executable, "errors": []}
    try:
        _quiet()
        from openff.toolkit import ForceField
        from openff.toolkit.typing.engines.smirnoff import get_available_force_fields
        import openff.interchange  # noqa: F401
    except Exception as exc:
        result["errors"].append(f"cannot import openff-toolkit/openff-interchange: {exc}")
        return result
    result["versions"] = {
        name: _version(name)
        for name in ("openff-toolkit", "openff-interchange", "openff-forcefields", "openff-nagl", "openff-nagl-models", "rdkit")
    }
    available = sorted(name for name in get_available_force_fields() if name.startswith("openff-"))
    result["available_force_fields"] = available
    try:
        force_field = ForceField(forcefield)
    except Exception as exc:
        result["errors"].append(f"force field {forcefield} cannot be loaded: {exc}")
        return result
    result["charges"] = _charge_method(force_field)
    if not result["charges"]["usable"]:
        result["errors"].append(result["charges"]["problem"])
    result["ok"] = not result["errors"]
    return result


def run(input_sdf: Path, output_dir: Path, name: str, forcefield: str, net_charge: int) -> dict:
    _quiet()
    from rdkit import Chem
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule
    from openff.units import unit

    records = [mol for mol in Chem.SDMolSupplier(str(input_sdf), removeHs=False) if mol is not None]
    if len(records) != 1:
        raise ValueError(f"{input_sdf}: expected one SDF record, found {len(records)}")
    rdmol = records[0]
    if Chem.GetFormalCharge(rdmol) != net_charge:
        raise ValueError(f"formal charges sum to {Chem.GetFormalCharge(rdmol)}, expected {net_charge}")
    molecule = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    elements = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    if [atom.symbol for atom in molecule.atoms] != elements:
        raise RuntimeError("OpenFF changed the atom order of the input structure")

    force_field = ForceField(forcefield)
    try:
        interchange = Interchange.from_smirnoff(force_field, [molecule])
    except Exception as exc:
        unsupported = sorted(set(elements) - SAGE_ELEMENTS)
        if unsupported:
            raise RuntimeError(
                f"{forcefield} has no parameters for element(s) {', '.join(unsupported)} "
                f"(Sage covers {', '.join(sorted(SAGE_ELEMENTS))}); use the OPLS-AA/2020 engine "
                f"for this polymer. OpenFF reported: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            ) from exc
        raise
    interchange.box = [10.0, 10.0, 10.0] * unit.nanometer  # GROMACS export needs a box; unused here

    output_dir.mkdir(parents=True, exist_ok=True)
    top_path = output_dir / f"{name}.top"
    interchange.to_top(str(top_path))

    vdw_handler = force_field.get_parameter_handler("vdW")
    smirks_to_id = {parameter.smirks: parameter.id for parameter in vdw_handler.parameters}
    by_atom = {key.atom_indices[0]: value.id for key, value in interchange["vdW"].key_map.items()}
    vdw_ids = [smirks_to_id.get(by_atom[index], by_atom[index]) for index in range(molecule.n_atoms)]

    charges = interchange["Electrostatics"].charges
    charge_sum = sum(value.m_as(unit.elementary_charge) for value in charges.values())
    info = {
        "force_field": forcefield,
        "python": sys.executable,
        "versions": {name: _version(name) for name in ("openff-toolkit", "openff-interchange", "openff-forcefields", "openff-nagl", "openff-nagl-models", "rdkit")},
        "charges": _charge_method(force_field),
        "charge_sum": charge_sum,
        "atom_count": molecule.n_atoms,
        "elements": elements,
        "vdw_parameter_ids": vdw_ids,
        "constrained_bonds_note": (
            "Sage is fitted with X-H bond constraints; the .top keeps all bonds as "
            "harmonic terms, so set constraints = h-bonds in the .mdp."
        ),
        "top": str(top_path),
    }
    (output_dir / f"{name}.sage.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("--forcefield", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--forcefield", required=True)
    run_parser.add_argument("--net-charge", type=int, required=True)
    args = parser.parse_args(argv)

    if args.command == "probe":
        print(json.dumps(probe(args.forcefield)))
        return 0
    try:
        info = run(args.input, args.output_dir, args.name, args.forcefield, args.net_charge)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"ok": False, "error": message, "traceback": traceback.format_exc()}))
        return 1
    print(json.dumps({"ok": True, "top": info["top"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
