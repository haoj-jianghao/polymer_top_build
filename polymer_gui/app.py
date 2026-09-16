from __future__ import annotations

import tempfile
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
sys.path.insert(0, str(PROJECT_ROOT))

from polymer_from_oligomer import (
    BuildOptions,
    MonomerBuildOptions,
    build_polymer_from_monomer,
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
                ),
            )
        return jsonify(
            {
                "formula": result.molecule.formula(),
                "atoms": len(result.molecule.atoms),
                "bonds": len(result.molecule.bonds),
                "repeat_units": result.repeat_units,
                "target_charge": result.target_charge,
                "final_charge": result.final_charge,
                "missing_tools": result.missing_tools,
                "files": [str(path.relative_to(PROJECT_ROOT)) for path in result.files],
                "download_base": f"/outputs/{outdir.name}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/outputs/<job>/<path:filename>")
def outputs(job: str, filename: str):
    return send_from_directory(OUTPUT_ROOT / job, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True)
