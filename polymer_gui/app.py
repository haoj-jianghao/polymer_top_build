from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_from_oligomer import (
    CHAIN_LAYOUTS,
    DEFAULT_CHAIN_LAYOUT,
    DEFAULT_FF_ENGINE,
    FF_ENGINES,
    BuildOptions,
    MonomerBuildOptions,
    build_polymer_from_monomer,
    engine_problem_lines,
    build_polymer_from_oligomer,
    parse_index_list,
    read_sdf,
)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/preview")
def preview():
    sdf_text = request.form.get("sdf", "").strip()
    if not sdf_text:
        return jsonify({"error": "Draw a molecule or paste SDF/MOL text first."}), 400
    with tempfile.NamedTemporaryFile("w", suffix=".sdf", delete=False) as handle:
        handle.write(sdf_text + "\n")
        path = Path(handle.name)
    try:
        mol = read_sdf(path)
        atoms = [
            {
                "index": atom.index,
                "element": atom.element,
                "formal_charge": atom.formal_charge,
            }
            for atom in mol.atoms
        ]
        bonds = [
            {"a": bond.a, "b": bond.b, "order": bond.order, "stereo": bond.stereo}
            for bond in mol.bonds
        ]
        return jsonify(
            {
                "name": mol.name,
                "formula": mol.formula(),
                "molecular_weight": round(mol.molecular_weight(), 3),
                "formal_charge": mol.total_formal_charge(),
                "atoms": atoms,
                "bonds": bonds,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/build")
def build():
    sdf_text = request.form.get("sdf", "").strip()
    if not sdf_text:
        return jsonify({"error": "Draw a molecule or paste SDF/MOL text first."}), 400
    try:
        workflow = request.form.get("workflow", "monomer")
        dp = int(request.form.get("dp", "1"))
        previous_atom = int(request.form.get("previous_atom", "0") or "0")
        next_atom = int(request.form.get("next_atom", "0") or "0")
        reference_dp = int(request.form.get("reference_dp", "5") or "5")
        run_external = request.form.get("run_external") == "true"
        chain_layout = request.form.get("chain_layout", DEFAULT_CHAIN_LAYOUT) or DEFAULT_CHAIN_LAYOUT
        if chain_layout not in CHAIN_LAYOUTS:
            raise ValueError(f"unknown starting structure {chain_layout!r}")
        chain_seed = int(request.form.get("chain_seed", "0") or "0")
        ff_engine = request.form.get("ff_engine", DEFAULT_FF_ENGINE) or DEFAULT_FF_ENGINE
        if ff_engine not in FF_ENGINES:
            raise ValueError(f"unknown force field {ff_engine!r}")
        engine_settings = {}
        for key in ("sage_python", "amber_python"):
            value = request.form.get(key, "").strip()
            if value:
                engine_settings[key] = str(Path(value).expanduser())
    except Exception as exc:
        return jsonify({"error": f"Invalid settings: {exc}"}), 400

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outdir = OUTPUT_ROOT / request.form.get("job_name", "polymer_gui_job").strip().replace(" ", "_")
    input_sdf = outdir / "oligomer_input.sdf"
    outdir.mkdir(parents=True, exist_ok=True)
    input_sdf.write_text(sdf_text + "\n")
    try:
        if workflow == "monomer":
            result = build_polymer_from_monomer(
                input_sdf,
                MonomerBuildOptions(
                    previous_atom=previous_atom,
                    next_atom=next_atom,
                    reference_dp=reference_dp,
                    polymer_dp=dp,
                    outdir=outdir,
                    run_external=run_external,
                    ff_engine=ff_engine,
                    engine_settings=engine_settings,
                    chain_layout=chain_layout,
                    chain_seed=chain_seed,
                ),
            )
        else:
            repeat_atoms = parse_index_list(request.form.get("repeat_atoms", ""))
            result = build_polymer_from_oligomer(
                input_sdf,
                BuildOptions(
                    repeat_atoms=repeat_atoms,
                    dp=dp,
                    oligomer_charge=int(request.form.get("oligomer_charge", "0")),
                    repeat_charge=int(request.form.get("repeat_charge", "0")),
                    end_charge=int(request.form.get("end_charge", "0")),
                    outdir=outdir,
                    run_external=run_external,
                    ff_engine=ff_engine,
                    engine_settings=engine_settings,
                    chain_layout=chain_layout,
                    chain_seed=chain_seed,
                ),
            )
        metadata = _read_metadata(outdir)
        engine_info = metadata.get("engine", {})
        return jsonify(
            {
                "formula": result.molecule.formula(),
                "atoms": len(result.molecule.atoms),
                "bonds": len(result.molecule.bonds),
                "repeat_units": result.repeat_units,
                "target_charge": result.target_charge,
                "final_charge": result.final_charge,
                "missing_tools": result.missing_tools,
                "missing_details": _missing_details(ff_engine, engine_settings) if result.missing_tools else "",
                "ff_engine": ff_engine,
                "ff_engine_label": FF_ENGINES[ff_engine],
                "parameterization_mode": metadata.get("parameterization_mode"),
                "charge_model": engine_info.get("charge_model"),
                "atom_types": engine_info.get("merged_atom_types"),
                "warnings": engine_info.get("transfer_warnings", [])
                + [warning for warning in engine_info.get("engine_warnings", []) if warning],
                "openff_python": engine_info.get("openff_python"),
                "amber_python": engine_info.get("amber_python"),
                "chain_seed": engine_info.get("chain_seed"),
                "radius_of_gyration": engine_info.get("radius_of_gyration_A"),
                "files": [str(path.relative_to(PROJECT_ROOT)) for path in result.files],
                "download_base": f"/outputs/{outdir.name}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def _read_metadata(outdir: Path) -> dict:
    try:
        return json.loads((outdir / "build_metadata.json").read_text())
    except (OSError, ValueError):
        return {}


def _missing_details(ff_engine: str, engine_settings: dict) -> str:
    return " | ".join(line.strip().splitlines()[0] for line in engine_problem_lines(ff_engine, engine_settings))


@app.get("/outputs/<job>/<path:filename>")
def outputs(job: str, filename: str):
    return send_from_directory(OUTPUT_ROOT / job, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True)
