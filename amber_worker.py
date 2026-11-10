#!/usr/bin/env python
"""Parameterize one molecule with GAFF2/AM1-BCC (AmberTools) and write a GROMACS topology.

This script runs with the Python of the user's AmberTools environment, never
inside the polymer builder: the builder calls it (``gaff2_engine.py``) so
that AmberTools and ParmEd do not have to be installed next to the builder.
It needs only ParmEd (shipped with conda AmberTools) and the AmberTools
executables antechamber, parmchk2 and tleap, looked up next to this
interpreter first and then on PATH.

    python amber_worker.py probe
        -> JSON on stdout: executables, ParmEd version, leaprc.gaff2 location
    python amber_worker.py run --input ref.mol2 --output-dir DIR --name NAME --net-charge Q
        -> in DIR: NAME_typed.mol2 (antechamber: GAFF2 types, AM1-BCC charges),
           NAME.frcmod (parmchk2 -s gaff2), NAME.prmtop/.inpcrd (tleap),
           NAME.top (ParmEd, GROMACS) and NAME.amber.json

Atom order of the input MOL2 is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

TOOLS = ("antechamber", "parmchk2", "tleap")


def _bin_dir() -> Path:
    return Path(sys.executable).parent


def _tool(name: str) -> str | None:
    local = _bin_dir() / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which(name)


def _amberhome() -> Path | None:
    candidates = [os.environ.get("AMBERHOME"), str(_bin_dir().parent)]
    tleap = _tool("tleap")
    if tleap:
        candidates.append(str(Path(tleap).resolve().parent.parent))
    for candidate in candidates:
        if candidate and (Path(candidate) / "dat" / "leap" / "cmd" / "leaprc.gaff2").is_file():
            return Path(candidate)
    return None


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(_bin_dir()) + os.pathsep + env.get("PATH", "")
    amberhome = _amberhome()
    if amberhome is not None:
        env["AMBERHOME"] = str(amberhome)  # conda sets it only on activation
    return env


def probe() -> dict:
    result: dict = {"ok": False, "python": sys.executable, "errors": []}
    result["tools"] = {name: _tool(name) for name in TOOLS}
    missing = [name for name, path in result["tools"].items() if path is None]
    if missing:
        result["errors"].append(f"AmberTools executable(s) not found: {', '.join(missing)}")
    try:
        import parmed

        result["parmed"] = parmed.__version__
    except Exception as exc:
        result["errors"].append(f"cannot import parmed: {exc}")
    amberhome = _amberhome()
    result["amberhome"] = str(amberhome) if amberhome else None
    if amberhome is None:
        result["errors"].append("leaprc.gaff2 not found (no AMBERHOME with dat/leap/cmd/leaprc.gaff2)")
    result["ok"] = not result["errors"]
    return result


def _call(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("w") as handle:
        completed = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=_environment())
    if completed.returncode != 0:
        tail = log.read_text().strip().splitlines()[-15:]
        raise RuntimeError(f"{Path(command[0]).name} failed (exit {completed.returncode}):\n" + "\n".join(tail))


def _mol2_atoms(path: Path) -> list[tuple[str, str, float]]:
    """(name, atom type, charge) of the ATOM records."""

    atoms, inside = [], False
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            inside = line.startswith("@<TRIPOS>ATOM")
            continue
        if inside and line.strip():
            fields = line.split()
            atoms.append((fields[1], fields[5], float(fields[8])))
    return atoms


def run(input_mol2: Path, output_dir: Path, name: str, net_charge: int) -> dict:
    import parmed

    output_dir.mkdir(parents=True, exist_ok=True)
    typed = output_dir / f"{name}_typed.mol2"
    frcmod = output_dir / f"{name}.frcmod"
    prmtop = output_dir / f"{name}.prmtop"
    inpcrd = output_dir / f"{name}.inpcrd"
    top = output_dir / f"{name}.top"
    shutil.copyfile(input_mol2, output_dir / f"{name}_input.mol2")

    _call(
        [_tool("antechamber"), "-i", f"{name}_input.mol2", "-fi", "mol2", "-o", typed.name, "-fo", "mol2",
         "-c", "bcc", "-s", "2", "-at", "gaff2", "-nc", str(net_charge), "-pf", "y"],
        output_dir, output_dir / "antechamber.log",
    )
    # -s gaff2: estimate missing terms against gaff2.dat (parmchk2's default is GAFF)
    _call([_tool("parmchk2"), "-i", typed.name, "-f", "mol2", "-o", frcmod.name, "-s", "gaff2"],
          output_dir, output_dir / "parmchk2.log")
    leap_in = output_dir / "tleap.in"
    leap_in.write_text(
        "source leaprc.gaff2\n"
        f"loadamberparams {frcmod.name}\n"
        f"mol = loadmol2 {typed.name}\n"
        f"saveamberparm mol {prmtop.name} {inpcrd.name}\n"
        "quit\n"
    )
    _call([_tool("tleap"), "-f", leap_in.name], output_dir, output_dir / "tleap.log")

    input_atoms = _mol2_atoms(input_mol2)
    typed_atoms = _mol2_atoms(typed)
    structure = parmed.load_file(str(prmtop), str(inpcrd))
    if len(structure.atoms) != len(input_atoms) or len(typed_atoms) != len(input_atoms):
        raise RuntimeError(
            f"atom count changed: input {len(input_atoms)}, antechamber {len(typed_atoms)}, tleap {len(structure.atoms)}"
        )
    if [atom.name for atom in structure.atoms] != [entry[0] for entry in input_atoms]:
        raise RuntimeError("tleap changed the atom order of the input structure")
    structure.save(str(top), format="gromacs", overwrite=True)

    frcmod_text = frcmod.read_text()
    info = {
        "python": sys.executable,
        "tools": {tool: _tool(tool) for tool in TOOLS},
        "parmed": parmed.__version__,
        "amberhome": os.environ.get("AMBERHOME") or str(_amberhome()),
        "charge_model": "AM1-BCC (antechamber/sqm)",
        "charge_sum": sum(charge for _, _, charge in typed_atoms),
        "gaff2_types": [atom_type for _, atom_type, _ in typed_atoms],
        "frcmod_terms_estimated": sum(1 for line in frcmod_text.splitlines() if "ATTN, need revision" in line),
        "top": str(top),
    }
    (output_dir / f"{name}.amber.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--net-charge", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        print(json.dumps(probe()))
        return 0
    os.environ.update({key: value for key, value in _environment().items() if key == "AMBERHOME"})
    try:
        info = run(args.input.resolve(), args.output_dir.resolve(), args.name, args.net_charge)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}))
        return 1
    print(json.dumps({"ok": True, "top": info["top"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
