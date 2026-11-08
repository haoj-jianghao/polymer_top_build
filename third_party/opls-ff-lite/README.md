# opls-ff-lite

`opls-ff-lite` generates OPLS-AA (OPLS/2020) topologies for GROMACS, for small
molecules and short oligomers, neutral or charged. It is a reduced copy of
opls-ff-gen with the fewest possible dependencies.

| needs | what for |
|---|---|
| Python 3.12, RDKit (>= 2025.9), NumPy | everything; installable with pip or conda |
| **MOPAC** (23.2.5 tested) | AM1 for the CM1A charges, the only external program |
| the OPLS/2020 parameter files | you download them yourself, then install them with `opls-ff-lite install-si` |

It does not use or need xTB, QM fitting, PySCF, OpenMM, GROMACS or AmberTools.
Geometries come from RDKit (ETKDGv3 + MMFF94), or from your structure file.

## What it does

1. **Structure:** from a SMILES (ETKDGv3, then MMFF94), or from an SDF, MOL2
   or PDB file, with its coordinates as given or optimized with MMFF94. For
   PDB and MOL2 files, `--template-smiles` sets bond orders and formal charges.
2. **Atom types:** OPLS-AA SMARTS rules plus the atom-type archive, refined to
   OPLS/2020 type numbers. Atoms OPLS cannot type get a GAFF2 type, then the
   analogous OPLS/2020 type where there is one, else GAFF2 Lennard-Jones.
3. **Charges:** MOPAC AM1 single point, then CM1A.
   - With `--charge-model auto` (the default), **neutral molecules**
     (zwitterions included) get 1.14×CM1A with the localized bond-charge
     corrections (1.14×CM1A-LBCC).
   - **Ions** get unscaled CM1A.
   - The charges always sum exactly to the net charge.
4. **Bonded terms:** from the OPLS/2020 tables first. A term the tables lack
   is taken from:
   - GAFF2 (gaff2.dat);
   - values recorded from earlier QM fits (no QM is run);
   - parmchk2-style substitution;
   - empirical rules.
   Every term gets a value, and every non-table term is marked in the ITP.
5. **Output:**
   - `NAME.itp`, `NAME.top`, `NAME.gro`;
   - `NAME.topology.json`, with every term, its parameters and its source;
   - `NAME.assignment.json` and `.csv` (per-atom types and charges);
   - `NAME.assigned.sdf`;
   - the MOPAC files.

For the same structure, the topology is identical to opls-ff-gen's default
(`--qm none`) with MMFF94 geometry.

## Install

```bash
conda env create -f environment.yml && conda activate opls-ff-lite
pip install opls_ff_lite-<version>-py3-none-any.whl
opls-ff-lite install-si jp3c06602_si_002.txt jp3c06602_si_003.txt
opls-ff-lite doctor
```

- The OPLS/2020 files are the Supporting Information of W. L. Jorgensen et
  al., *J. Phys. Chem. B* (2024), https://doi.org/10.1021/acs.jpcb.3c06602.
  They may not be redistributed. `install-si` checks them against the release
  the tool was validated with and copies them to
  `~/.local/share/opls-ff-lite/opls2020/`.
- If MOPAC is installed somewhere else: `opls-ff-lite config set mopac
  /path/to/mopac` (or set `OPLS_FF_LITE_MOPAC`).
- Without installing: `python run_opls_ff_lite.py ...`, with any Python 3.12+
  that has RDKit and NumPy.

## Use

```bash
opls-ff-lite --smiles 'CC(C(=O)[O-])CC(C(=O)[O-])CC(C(=O)[O-])C' --name paa3 --output-dir out/paa3
opls-ff-lite --structure oligomer.mol2 --net-charge -3 --output-dir out/oligomer
opls-ff-lite --structure oligomer.pdb --template-smiles 'SMILES' --output-dir out/oligomer
```

```python
from opls_ff_lite import generate_gromacs_topology
from opls_ff_lite.structure import prepare_structure

molecule = prepare_structure(smiles="C[NH3+]", name="methylammonium")
result = generate_gromacs_topology(molecule, "out/methylammonium", name="methylammonium")
print(result.assignment.charge_model, result.assignment.final_charge_sum_e, result.itp_path)
```

The `.top` file carries the OPLS `[ defaults ]`: geometric combination rule,
fudgeLJ = fudgeQQ = 0.5.

## Development

opls-ff-lite lives in `lite/` of the opls-ff-gen repository.
- Its core modules (typing, charges, tables, bonded assignment and fallbacks)
  are copies of the full tool's, differing only in the package name.
- Its data files are byte-identical copies.
- `python src/scripts/check_sync.py` checks this; `--update` re-copies
  changed files.
- The tests (`python -m unittest discover -s tests`) also compare complete
  topologies with the full tool's.

Licence: MIT; two AmberTools data files are GPL-3.0 (`NOTICE`).
