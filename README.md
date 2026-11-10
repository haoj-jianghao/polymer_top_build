# Polymer Topology Builder for Molecular Dynamics Simulation

A tool to build force-field topologies and starting structures of polymers
for molecular dynamics (MD) (currently only support GROMACS). 
It writes ready-to-use `.top`, `.itp` and `.gro` files. 
The main input is the monomer structure, which you can draw in a GUI or give as a MOL/SDF file.

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

```text
draw monomer
  -> choose previous/next connection atoms
  -> generate hidden reference oligomer, e.g. 5-mer
  -> add hydrogens, minimize the geometry (RDKit)
  -> parameterize the reference oligomer (opls | sage | gaff2)
  -> transfer its parameters to the requested polymer (index shift + cross-check)
  -> build 3D coordinates (random walk of monomers)
  -> write polymer.itp / polymer.top / polymer.gro (+ PARAMETER_TRANSFER_REPORT.txt)
```

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

- `polymer_from_oligomer.py`: importable backend plus CLI.
- `opls_engine.py`: runs the bundled opls-ff-lite on the reference oligomer.
- `sage_engine.py`: finds the user's OpenFF installation and runs
  `sage_worker.py` in it on the reference oligomer.
- `gaff2_engine.py`: finds the user's AmberTools installation and runs
  `amber_worker.py` in it on the reference oligomer.
- `external_env.py`: shared search for conda environments and worker runs.
- `chain_builder.py`: random walk of monomers for the starting 3D structure.
- `topology_transfer.py`: transfers a reference-oligomer GROMACS topology to
  the polymer. The transfer works the same way for every engine.
- `third_party/opls-ff-lite/`: bundled copy of opls-ff-lite, without the
  OPLS/2020 SI files. `scripts/sync_opls_ff_lite.sh` updates it from the
  opls-ff-gen repository.
- `polymer_gui/desktop.py`: desktop GUI (Tkinter): sketcher, build settings, background builds.
- `polymer_gui/sketch_model.py`: the sketcher's drawing model (no GUI code; unit-tested).
- `examples/PEG_repeat_monomer.sdf`: PEG-like repeat monomer example.
- `examples/PEG_5mer.sdf`: older oligomer-mode pentaethylene glycol example.
- `tests/test_polymer_non_amber.py`: tests for repeat detection, DP build, and charge normalization.
- `tests/test_topology_transfer.py`: tests for the topology transfer and the OPLS engine.
- `tests/test_sage_engine.py`: tests for the Sage engine and the search for an OpenFF interpreter.
- `tests/test_gaff2_engine.py`: tests for the GAFF2 engine, including a comparison with tleap run on the whole polymer.
- `tests/test_chain_builder.py`: tests for the random-walk starting structure.
- `tests/test_desktop_gui.py`: tests for the sketcher model and the desktop window (window tests need a display).

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

An explicitly given interpreter (1 or 2) is never silently replaced by
another one. Each candidate is checked once per session: the imports, the
force field and its charge model. To see which installation is used:

```bash
python polymer_from_oligomer.py --check-engine --ff-engine sage
```

`--sage-forcefield` selects another SMIRNOFF file, for example
`openff-2.2.1.offxml`. Sage 2.2 uses toolkit AM1-BCC charges, which are slow
and need AmberTools or OpenEye in the OpenFF environment, so a warning is
shown. Sage is fitted with X-H bond constraints: use `constraints = h-bonds`
in the `.mdp`.

## How the parameters reach the polymer

1. The engine writes a GROMACS topology of the reference oligomer.
   - OPLS: `opls_reference/reference.top` and `.itp`, plus the per-atom
     assignment and MOPAC files.
   - Sage: `sage_reference/reference.top` (Interchange) and
     `reference.sage.json` (versions, charge model, vdW parameter ids).
   - GAFF2: `gaff2_reference/reference.top` (ParmEd), plus the antechamber,
     parmchk2 and tleap files.
2. Atoms are addressed as (repeat unit, index within the unit). The
   reference and the polymer are built unit by unit from the same monomer.
   Each unit's hydrogens are numbered after its monomer atoms, by parent
   atom and slot.
3. **Bonded terms and atom types are copied by index shift.**
   - A reference term in units r..r+s is copied to polymer units p..p+s, and
     each atom keeps its index within its unit.
   - The source window follows a fixed rule. Polymer terms within the first
     or last E units come from the same position at the reference head or
     tail; E = 2 for the default 5-unit reference. All other terms come from
     the most central reference window.
   - Example with a 5-mer reference and a 12-mer polymer: unit 1 ← 1, unit 2
     ← 2, units 3–10 ← 3, unit 11 ← 4, unit 12 ← 5. The junction between
     units 6 and 7 is copied from the junction between reference units 2
     and 3.
4. **Cross-check:** an independent method places every reference term
   pattern at every polymer position. For each one it picks the reference
   copy whose bonded neighbourhood (up to 4 bond shells) matches the
   polymer's best.
   - The build is accepted only if both methods give exactly the same atom
     types and term lines: same atoms, function, parameter text and comment.
     Any difference stops the build and lists the differences.
   - The build also checks that every bond, angle, 1-4 pair and proper
     dihedral the polymer's graph requires is present.
5. **Reference size:** interior terms are copied from windows of interior
   reference units only. A term spanning n repeat units therefore needs a
   reference of at least n + 2 units, and smaller references are rejected
   with that number.
   - Monomers with a two-atom backbone (vinyl polymers, PE, PDMS) have
     torsions and 1-4 pairs spanning 3 units, so they need 5 units.
   - Larger repeat units need 4.
   - The default of 5 covers both.
6. **Charges:** each atom gets its charge averaged over the equivalent
   reference atoms, as for GAFF2.
   - The first unit uses the reference head unit, the last unit the tail
     unit, and middle units all reference interior units.
   - The charges are then corrected so the total equals the formal charge
     exactly.
7. **Atom types** are merged by their complete non-bonded parameter set:
   atomic number, mass, sigma and epsilon. Every bonded term is written with
   explicit parameters, so merging changes no energy. A PEG polymer needs 8
   types and PMMA 10. A type label used with different parameters gets a
   suffix, for example `opls_135_2`.
8. A polymer shorter than the reference oligomer is parameterized directly.

`PARAMETER_TRANSFER_REPORT.txt` records the following:
- the shift rule used and the cross-check result;
- term counts and coverage checks;
- the environment-match depth histogram;
- the opls-ff-lite rule counts;
- the charge spread between equivalent reference atoms, with a warning above
  0.02 e (a longer reference converges the interior charges better);
- any warnings.

`polymer.top` carries the engine's own `[ defaults ]`:
- OPLS: `1 3 yes 0.5 0.5` (geometric combination rule, fudgeLJ = fudgeQQ = 0.5);
- Sage and GAFF2: `1 2 yes 0.5 0.833333` (Lorentz-Berthelot, fudgeQQ = 1/1.2).

Sage atom types are named by their vdW parameter id, for example `n16`, and
GAFF2 atom types by their GAFF2 type, for example `c3`. `polymer.itp` holds
the `[ atomtypes ]` and the `[ moleculetype ]`. Each repeat unit is one
residue, and atoms are named element + number within the unit.

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

```bash
python polymer_from_oligomer.py --check-engine --ff-engine gaff2
```

On the reference oligomer only, the worker runs:
1. `antechamber -c bcc -at gaff2` (GAFF2 types, AM1-BCC charges through sqm);
2. `parmchk2 -s gaff2`;
3. `tleap` with `leaprc.gaff2` and that frcmod;
4. ParmEd, to write the GROMACS topology.

The files are kept in `gaff2_reference/`: the typed MOL2, frcmod, prmtop,
the GROMACS `.top` and the logs. The polymer is then built by the same index
shift and cross-check as for the other engines; tleap never sees the full
polymer.

`parmchk2` is run with `-s gaff2`. Its default parameter set is GAFF, and the
earlier version of this builder used that default. Terms missing from
`gaff.dat` were then estimated from GAFF data, and because the frcmod is
loaded after `leaprc.gaff2`, those estimates overrode GAFF2's own values.

## Starting structure (random walk of monomers)

`polymer.gro` and `polymer.sdf` get 3D coordinates from a self-avoiding random
walk of rigid monomer units (`chain_builder.py`). This step does not depend
on the force-field engine.

- **Templates.** Each polymer unit is a rigid copy of a unit of the
  minimized reference oligomer, chosen by the same rule as the parameters:
  head and tail units from the reference ends, the others from its central
  unit. The units therefore keep the drawn stereochemistry.
- **Joining units.** A new unit joins the previous one at the reference's
  junction bond length and angles.
- **The random choice.** The only random degree of freedom is the rotation
  about the junction bond: a staggered rotamer (0, 120 or 240 degrees from
  the reference conformation) plus up to ±15 degrees jitter. A double-bond
  junction only takes 0 or 180 degrees.
- **Clashes.** A placement is rejected when an atom comes closer than
  0.80 × the sum of van der Waals radii to an atom more than three bonds
  away. Hydrogen-bonded pairs (a polar H and an N/O/F acceptor) are allowed
  down to 0.60 ×. When no rotation fits, the walk backtracks.
- **Reproducibility.** `--chain-seed N` (or the GUI seed field) makes the
  walk reproducible; another seed gives another chain.
- **The old layout.** `--chain-layout 2d` keeps the flat layout of the
  drawing.

`PARAMETER_TRANSFER_REPORT.txt` lists the seed, the number of backtracks,
the closest contact, the end-to-end distance and the radius of gyration.
Bond lengths equal those of the reference units. On the tested polymers
(PEG, PE, poly(methyl acrylate), polystyrene, PAA, a polycation, PDMS,
nylon-6; DP 200), a GROMACS steepest-descent minimization reaches
Fmax < 1000 kJ/mol/nm within about 40 steps. A 30,000-atom chain is built in
a few seconds.

Units are rigid, so the torsions inside a monomer keep their reference
conformation; with very small monomers such as ethylene this makes the
starting chain more extended than a random coil. Minimize the structure and
equilibrate it before production MD as usual.

## Run the GUI

The desktop GUI uses Tkinter, which ships with Python. It needs no web
browser, no server and no extra package:

```bash
python polymer_gui/desktop.py
```

The window has three parts: the drawing tools and canvas on the left, the
**Build settings**, **MOL/SDF** and **Output** tabs on the right, and a
status bar.

- **Build in the background.** Builds run in a background thread, so the
  window stays responsive while the reference oligomer is parameterized
  (AM1-BCC with GAFF2 can take a minute).
- **Output tab.** It shows the transfer report and lists the written
  files. Double-click a file to open it, or use **Open output folder**.
  Under WSL the folder opens in Windows Explorer.
- **Check engine.** This button reports whether the chosen force field is
  ready and which OpenFF or AmberTools installation will be used.
- **File menu.** Open a MOL/SDF monomer, save the drawing as MOL, or load
  the PEG example.

Basic PEG workflow:

1. Click **PEG example**. The connection atoms are set to 1 (previous,
   green) and 3 (next, orange).
2. Keep the reference oligomer size at 5 and choose the polymer DP.
3. Choose the force field: OPLS-AA/2020, OpenFF Sage or GAFF2.
4. Click **Generate GROMACS files**. The job folder is
   `<output folder>/<job name>` (by default `outputs/`).

Drawing:

1. Choose an element under **Atoms**, then work on the canvas:
   - drag on the canvas to draw a single bond; free ends become carbon
     atoms;
   - drag from an existing atom to sprout a new atom and bond;
   - click empty space to add an atom of the chosen element;
   - click an atom to change it to the chosen element;
   - click a bond to cycle single → double → triple → single.
2. To delete, right-click an atom or bond, or select it and press
   `Delete`/`Backspace` or click **Delete selected**. Atoms are renumbered
   1..N.
3. Atom numbers are drawn beside each atom. Under **Connection atoms**,
   choose **Pick previous/next**. Then click the atom bonded to the
   previous monomer and the atom bonded to the next monomer. They are
   ringed green (previous) and orange (next), and the settings fields
   update. You can also type the numbers in **Build settings**.
4. The line above the canvas shows the monomer formula with hydrogens
   added, its molecular weight and its formal charge.

For a stereogenic single bond, choose **Wedge** or **Dash** under **Bond
style**. Then draw the bond from its narrow end to its wide end, or click
an existing single bond. Solid wedges are stored as V2000 stereo code `1`
and hashed wedges as code `6`. The codes are replicated with the monomer,
used by RDKit for the 3D reference geometry, and carried into every unit
of the random-walk polymer.

For an ionic repeat, choose −1 or +1 under **Formal charge** and click
the charged atom; choose 0 and click again to remove it.
- The charge is stored in the MOL/SDF `M  CHG` records and sets both the
  reference-oligomer and the polymer net charge.
- Hydrogen completion takes formal charges into account. N+ is allowed
  four bond orders, so a quaternary nitrogen with four single bonds gets
  no hydrogen, while a three-coordinate N+ gets one.

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

## Test

```bash
python -m unittest tests.test_polymer_non_amber tests.test_topology_transfer tests.test_sage_engine tests.test_gaff2_engine tests.test_chain_builder tests.test_desktop_gui
```

- `tests.test_polymer_non_amber` and the first two groups of
  `tests.test_topology_transfer` need neither AmberTools, MOPAC nor GROMACS.
  The transfer tests use a synthetic reference topology whose parameters
  encode each atom's local chemistry.
- `OplsLiteIntegrationTests` run the real opls-ff-lite and are skipped
  unless it is ready (MOPAC and the SI files). They check two things:
  - a polymer as long as the reference reproduces opls-ff-lite's own
    topology term for term;
  - a DP-20 PEG has an exact integer charge and passes `gmx grompp` when
    GROMACS is on `PATH`;
  - an 8-mer built from a 5-mer reference has exactly the terms, parameters
    and atom types of an 8-mer built independently from SMILES and
    parameterized directly.
- `SageIntegrationTests` run the same checks through your OpenFF
  installation and are skipped when none is found.
- `Gaff2IntegrationTests` run through your AmberTools installation and are
  skipped when none is found. They also check the transferred polymer
  against tleap's own type-based assembly of the whole polymer.

## V1 limitations

- Linear homopolymer backbones only.
- The GUI/CLI default is monomer mode. Oligomer mode remains available for
  compatibility and testing.
- The selected repeat atom list can include branches or ring atoms; the backend
  now copies the full selected repeat subgraph and preserves internal bond
  orders.
- SDF/MOL V2000 input only; `polymer.sdf` is written as V3000 above 999 atoms.
- Neighboring repeats must be connected by one inferred backbone junction bond.
- Side groups must be included in the selected repeat atom list if they should
  be replicated on every repeat.
- When external parameterization is disabled or the engine's requirements are
  missing, placeholder `.itp/.top/.gro` files are suitable only for testing
  data flow, atom counts, and charge normalization.
- The starting coordinates are a random walk of rigid reference units (see
  "Starting structure"). They are clash-free but not equilibrated:
  minimize and equilibrate before production MD. Placeholder builds
  (`--no-external`) keep the flat 2D layout.
