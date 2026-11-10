# Polymer Topology Builder for Molecular Dynamics Simulation

A tool to build force-field topologies and starting structures of polymers
for molecular dynamics (MD) (currently only support GROMACS). 
It writes ready-to-use `.top`, `.itp` and `.gro` files. 
The main input is the monomer structure, which you can draw in a GUI.

You draw a monomer and select the connection atoms between monomers, then the program then works as follows:

- **Reference oligomer.** It builds a small hidden reference oligomer (5
  units by default) and parameterizes only that oligomer, with one of three
  force-field engines:
  - **OPLS-AA/2020** (default);
  - **OpenFF Sage** (your own OpenFF installation);
  - **GAFF2/AM1-BCC** (your own AmberTools installation).
- **Transfer.** The same engine-neutral step transfers the reference
  parameters to a polymer of the length you choose. Terms are copied by an
  index shift of whole monomer units, and the result is accepted only if an
  independent environment-matching method gives exactly the same atom types
  and terms.
- **Atom types** are merged across the chain only when their non-bonded
  parameters are identical.
- **Charges** are averaged over equivalent reference atoms and corrected so
  the total equals the formal charge exactly.
- **Starting structure.** The 3D coordinates come from a self-avoiding
  random walk of rigid monomer units taken from the reference geometry.


## Force-field engines

| engine | `--ff-engine` | force field and charges | needs |
|---|---|---|---|
| **OPLS-AA/2020** (default) | `opls` | OPLS/2020 atom types and bonded terms; CM1A charges (ions) or 1.14*CM1A-LBCC (neutral) 
| OpenFF Sage | `sage` | Sage 2.3.0 (`openff-2.3.0.offxml`) by default; NAGL (AshGC) charges | your own OpenFF installation (openff-toolkit, openff-interchange, openff-nagl + models); nothing is installed for it |
| GAFF2 | `gaff2` | GAFF2 types; AM1-BCC charges | your own AmberTools installation (`antechamber`, `parmchk2`, `tleap`, ParmEd); nothing is installed for it |

Every engine parameterizes only the hidden reference oligomer, and all three
reach the polymer through the same transfer step (see below).

- **OPLS-AA/2020** uses the bundled
  [opls-ff-lite](third_party/opls-ff-lite/README.md). It covers H, C, N, O,
  F, P, S, Cl, Br and I. Atoms that OPLS cannot type, such as Si, get GAFF2
  types.
- **Sage** covers the same elements but not Si; a Si-containing polymer
  stops with a message to use OPLS.

## What is included

- `polymer_gui/desktop.py`: desktop GUI (Tkinter): sketcher, build settings, background builds.
- `polymer_from_oligomer.py`: importable backend plus CLI.
- `third_party/opls-ff-lite/`: bundled copy of opls-ff-lite, a customized opls force field parameterization engine, 
   but without the OPLS/2020 SI files. `scripts/sync_opls_ff_lite.sh` updates it from the opls-ff-gen repository.

The GUI includes a small built-in sketcher, so it does not depend on an
external chemistry editor. It can also open MOL/SDF files and accept pasted
MOL/SDF text.

## Install

```bash
conda env create -f environment.yml          # OPLS-AA/2020 engine
conda activate polymer-gui
# or, to install AmberTools for the GAFF2 engine in the same environment
# (not needed if you already have AmberTools in another environment):
conda env create -f environment-amber.yml
conda activate polymer-gui-amber
```

The desktop GUI uses Tkinter, which is part of every conda Python. A
Linux distribution's own Python may need its Tk package
(`sudo apt install python3-tk` on Debian/Ubuntu).

The OPLS/2020 parameters are the Supporting Information of W. L. Jorgensen et
al., *J. Phys. Chem. B* 2024, https://doi.org/10.1021/acs.jpcb.3c06602. They
may not be redistributed. Download `jp3c06602_si_002.txt` and
`jp3c06602_si_003.txt` from the article page, then install them once:

```bash
python third_party/opls-ff-lite/run_opls_ff_lite.py install-si jp3c06602_si_002.txt jp3c06602_si_003.txt
python third_party/opls-ff-lite/run_opls_ff_lite.py doctor     # exit 0 = ready
```

- The files go to `~/.local/share/opls-ff-lite/opls2020/`. Set
  `OPLS_FF_LITE_SI_DIR` to read them from another directory.
- MOPAC is found on `PATH`, or through `OPLS_FF_LITE_MOPAC`.
- If a requirement is missing, a build writes placeholder files and the GUI
  or CLI names what is missing.

### Using your OpenFF installation for Sage

The builder never imports OpenFF and never installs it. `sage_worker.py`
runs as a subprocess with the Python interpreter of an existing OpenFF
environment, so the builder's own environment stays small (no PyTorch, no
NAGL weights). The OpenFF Python is chosen in this order:

1. `--sage-python PATH` (CLI), or the "OpenFF Python" field in the GUI;
2. the environment variable `POLYMER_SAGE_PYTHON`;
3. the builder's own interpreter, if it can import OpenFF;
4. a conda/mamba environment with `openff-toolkit` and `openff-interchange`
   installed. The search covers the installations in `CONDA_EXE`,
   `CONDA_PREFIX`, `MAMBA_ROOT_PREFIX` and `~/miniforge3`, `~/mambaforge`,
   `~/miniconda3`, `~/anaconda3`.

``

### Using your AmberTools installation for GAFF2

Like Sage, the GAFF2 engine uses an existing installation and installs
nothing. `amber_worker.py` runs with the Python of an AmberTools
environment. That environment must provide antechamber, parmchk2 and tleap
next to its Python or on `PATH`, ParmEd (shipped with conda AmberTools), and
`leaprc.gaff2`. `AMBERHOME` is set automatically when the environment is not
activated. The AmberTools Python is chosen in this order:

1. `--amber-python PATH` (CLI), or the "AmberTools Python" field in the GUI;
2. the environment variable `POLYMER_AMBER_PYTHON`;
3. `$AMBERHOME/bin/python`;
4. the builder's own interpreter, if it can import ParmEd and `antechamber`
   is on `PATH`;
5. a conda/mamba environment with the `ambertools` package.

Note that, the GAFF2 engine could be significantly slower than the default opls
engine due to the charge calculation. 


## Starting structure (random walk of monomers)

`polymer.gro` and `polymer.sdf` get 3D coordinates from a self-avoiding random
walk of rigid monomer units (`chain_builder.py`). This step does not depend
on the force-field engine.


## Run the GUI

The desktop GUI uses Tkinter, which ships with Python. It needs no web
browser, no server and no extra package:

```bash
python polymer_gui/desktop.py
```

## Run the CLI

Monomer-mode non-external smoke test:

```bash
python polymer_from_oligomer.py examples/PEG_repeat_monomer.sdf \
  --previous-atom 1 \
  --next-atom 3 \
  --reference-dp 5 \
  --dp 12 \
  --outdir outputs/PEG_repeat_DP12_cli \
  --no-external
```

Monomer-mode OPLS-AA/2020 build (default engine):

```bash
python polymer_from_oligomer.py examples/PEG_repeat_monomer.sdf \
  --previous-atom 1 \
  --next-atom 3 \
  --reference-dp 5 \
  --dp 12 \
  --outdir outputs/PEG_repeat_DP12_opls
```

Add `--ff-engine gaff2` to use AmberTools instead. If the engine's
requirements are missing, the backend reports them and still writes
placeholder GROMACS-shaped files for non-external GUI/backend testing.

Geometry minimization of the reference oligomer is enabled by default. Only
the Sage engine can skip it, because its NAGL charges do not depend on the
geometry; the OPLS CM1A and GAFF2 AM1-BCC charges need a 3D structure:

```bash
python polymer_from_oligomer.py examples/PEG_repeat_monomer.sdf \
  --previous-atom 1 \
  --next-atom 3 \
  --reference-dp 5 \
  --dp 12 \
  --ff-engine sage \
  --outdir outputs/PEG_repeat_DP12_sage_unminimized \
  --no-minimize-geometry
```

The older oligomer mode is still available for compatibility by passing
`--repeat-atoms` instead of `--previous-atom` and `--next-atom`.

