"""CM1A partial charges from a MOPAC AM1 calculation.

CM1A (Charge Model 1 for AM1): J. W. Storer, D. J. Giesen, C. J. Cramer and
D. G. Truhlar, J. Comput.-Aided Mol. Des. 9, 87-110 (1995). The charge of
atom k is its AM1 Mulliken (Coulson) charge corrected through the bond orders
of its bonds:

    q_k = q0_k + B_k Delta_k - sum_{k' != k} B_kk' Delta_k'
    Delta_k = c_k q0_k + d_k + sum_{k' != k} B_kk' d_{Z_k Z_k'}

with q0_k = Z_core,k - sum_{mu on k} P_mu,mu, the bond order
B_kk' = sum_{mu on k, nu on k'} P_mu,nu^2 (no overlap in NDDO) and
B_k = sum_k' B_kk'. Pair offsets d_ZZ' exist for H bonded to N/O/Si, N bonded
to O (and to C through the N=C triple-bond switch below) and O bonded to S.
For an N-C pair, AM1 replaces c_N and adds d_NC through the switch
0.5 (1 - tanh(23 - 10 B_NC)), which is ~1 only for N#C triple bonds.

The AM1 wavefunction comes from MOPAC (open source, https://openmopac.net),
a single SCF on the input geometry. The parameters below are the published
CM1A values; they were cross-checked against AMSOL 7.1, and the charges
against AMSOL (validation/mopac_cm1a/).
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from .tools import program_path

# Published CM1A parameters (Storer et al. 1995), by element symbol.
CM1A_SCALE = {"N": 0.3846, "F": 0.1468, "S": -0.1311, "Cl": 0.0405, "Br": 0.1761, "I": 0.2380}
CM1A_OFFSET = {"O": -0.0283, "F": 0.0399, "S": -0.0956, "Cl": -0.0276, "Br": -0.0802,
               "I": -0.1819}
# d_{Z_k Z_k'}: offset of atom k (outer key) per bond order to an atom k' (inner key)
CM1A_PAIR_OFFSET = {
    "H": {"N": 0.0850, "O": 0.1447, "Si": 0.0640},
    "N": {"O": -0.0630},
    "O": {"S": -0.0600},
}
CM1A_NC_OFFSET = -0.0880  # d_NC, enters only through the N#C switch
MOPAC = program_path("mopac")  # OPLS_FF_LITE_MOPAC, config file, PATH (tools.py)


class CM1AError(RuntimeError):
    """MOPAC failed or its output could not be read."""


@dataclass(frozen=True)
class CM1AResult:
    charges: np.ndarray  # CM1A, per atom (e)
    mulliken: np.ndarray  # AM1 Mulliken/Coulson charges q0 (e)
    bond_orders: np.ndarray  # B_kk' (atoms x atoms, zero diagonal)
    heat_of_formation_kcal_mol: float
    mopac_input: Path
    mopac_output: Path


def write_mopac_input(path: Path, molecule: Chem.Mol, net_charge: int, title: str) -> None:
    """AM1 single SCF on the input coordinates; full-precision AUX output."""
    conformer = molecule.GetConformer()
    # SCFCRT: MOPAC's default SCF convergence leaves up to ~0.001 e of noise in the
    # charges of large conjugated molecules; 1e-10 removes it at little cost.
    lines = [f"AM1 1SCF CHARGE={net_charge} GEO-OK SCFCRT=1.D-10 AUX(PRECISION=12) THREADS=1",
             title[:78] or "molecule", ""]
    for index, atom in enumerate(molecule.GetAtoms()):
        p = conformer.GetAtomPosition(index)
        lines.append(f"{atom.GetSymbol():2s} {p.x:16.10f} 0 {p.y:16.10f} 0 {p.z:16.10f} 0")
    path.write_text("\n".join(lines) + "\n")


def _aux_block(text: str, key: str) -> list[str]:
    """Values of an AUX entry such as ``TOTAL_DENSITY_MATRIX[0171]=``."""
    match = re.search(rf"^\s*{re.escape(key)}(?::[A-Z/]+)?\[(\d+)\]=\s*\n?", text, re.M)
    if match is None:
        raise CM1AError(f"MOPAC AUX file has no {key}")
    count = int(match.group(1))
    tokens = [token for line in text[match.end():].splitlines()
              if not line.lstrip().startswith("#") for token in line.split()]
    values, i = [], 0
    while len(values) < count:
        values.append(tokens[i])
        i += 1
    return values


def read_mopac_aux(path: Path) -> dict:
    text = path.read_text()
    density = np.array([float(v.replace("D", "E")) for v in _aux_block(text, "TOTAL_DENSITY_MATRIX")])
    ao_atom = np.array([int(v) for v in _aux_block(text, "AO_ATOMINDEX")]) - 1
    core = np.array([float(v) for v in _aux_block(text, "ATOM_CORE")])
    heat = re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([-+0-9.D]+)", text)
    n = len(ao_atom)
    if len(density) != n * (n + 1) // 2:
        raise CM1AError(f"{path}: density matrix size {len(density)} does not match {n} AOs")
    matrix = np.zeros((n, n))
    matrix[np.tril_indices(n)] = density  # MOPAC stores the lower triangle row by row
    matrix = matrix + np.tril(matrix, -1).T
    return {"density": matrix, "ao_atom": ao_atom, "core": core,
            "heat_of_formation": float(heat.group(1).replace("D", "E")) if heat else float("nan")}


def cm1a_from_density(symbols: list[str], core: np.ndarray, density: np.ndarray,
                      ao_atom: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CM1A charges, Mulliken charges and bond orders from an AM1 density matrix."""
    n_atoms = len(symbols)
    onehot = np.zeros((len(ao_atom), n_atoms))
    onehot[np.arange(len(ao_atom)), ao_atom] = 1.0
    q0 = core - onehot.T @ np.diag(density)
    bond = onehot.T @ (density ** 2) @ onehot
    np.fill_diagonal(bond, 0.0)
    scale = np.array([CM1A_SCALE.get(s, 0.0) for s in symbols])
    offset = np.array([CM1A_OFFSET.get(s, 0.0) for s in symbols])
    for k, s in enumerate(symbols):
        for k2, s2 in enumerate(symbols):
            if k != k2:
                offset[k] += bond[k, k2] * CM1A_PAIR_OFFSET.get(s, {}).get(s2, 0.0)
    c_n = CM1A_SCALE["N"]
    for k, s in enumerate(symbols):
        if s != "N":
            continue
        for k2, s2 in enumerate(symbols):
            if s2 == "C":
                switch = 0.5 * (1.0 - math.tanh(23.0 - 10.0 * bond[k, k2]))
                scale[k] -= c_n * switch
                offset[k] += CM1A_NC_OFFSET * switch
    delta = scale * q0 + offset
    charges = q0 + bond.sum(axis=1) * delta - bond @ delta
    return charges, q0, bond


def run_cm1a(molecule: Chem.Mol, workdir: Path, name: str, net_charge: int = 0,
             timeout_seconds: int = 600) -> CM1AResult:
    """MOPAC AM1 single point and CM1A charges for ``molecule`` (explicit H, 3D)."""
    workdir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "molecule"
    mop = workdir / f"{stem}.mop"
    write_mopac_input(mop, molecule, net_charge, stem)
    try:
        result = subprocess.run([str(MOPAC), mop.name], cwd=workdir, capture_output=True,
                                text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        raise CM1AError(f"MOPAC executable not found: {MOPAC}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CM1AError(f"MOPAC exceeded the {timeout_seconds}-second timeout") from exc
    out, aux = workdir / f"{stem}.out", workdir / f"{stem}.aux"
    if result.returncode != 0 or not aux.exists():
        raise CM1AError(f"MOPAC failed ({result.returncode}): {result.stderr.strip()[-300:]}")
    text = out.read_text(errors="replace")
    if "SCF FIELD WAS ACHIEVED" not in text:
        raise CM1AError(f"{out}: MOPAC did not report SCF convergence")
    data = read_mopac_aux(aux)
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    if len(data["core"]) != len(symbols):
        raise CM1AError(f"MOPAC returned {len(data['core'])} atoms for {len(symbols)}")
    charges, q0, bond = cm1a_from_density(symbols, data["core"], data["density"], data["ao_atom"])
    return CM1AResult(charges, q0, bond, data["heat_of_formation"], mop, out)
