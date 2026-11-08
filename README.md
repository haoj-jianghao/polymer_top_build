# Polymer GROMACS Builder

A tool to build force-field topologies of polymers for molecular dynamics (MD)
simulations with GROMACS. It writes ready-to-use `.top`, `.itp` and `.gro`
files. The main input is the monomer structure, which you draw in a GUI.

You draw a monomer and choose the atoms that connect it to the previous and
next monomer. The program builds a small hidden reference oligomer (5 units by
default), parameterizes that oligomer with a force-field engine, and transfers
the parameters to a polymer of the length you choose.

```text
draw monomer
  -> choose previous/next connection atoms
  -> generate hidden reference oligomer, e.g. 5-mer
  -> add hydrogens, minimize the geometry (RDKit)
  -> parameterize the reference oligomer with the chosen force-field engine
  -> transfer its parameters to the requested polymer
  -> write polymer.itp / polymer.top / polymer.gro
```

## Force-field engines

| engine | `--ff-engine` | force field and charges | needs |
|---|---|---|---|
| **OPLS-AA/2020** (default) | `opls` | OPLS/2020 atom types and bonded terms; CM1A charges (ions) or 1.14*CM1A-LBCC (neutral) | bundled opls-ff-lite, MOPAC, the OPLS/2020 SI files |
| OpenFF Sage | `sage` | Sage 2.3.0 (`openff-2.3.0.offxml`) by default; NAGL (AshGC) charges | your own OpenFF installation (openff-toolkit, openff-interchange, openff-nagl + models); nothing is installed for it |
| GAFF2 | `gaff2` | GAFF2 types; AM1-BCC charges | AmberTools (`antechamber`, `parmchk2`, `tleap`), ParmEd |

Every engine parameterizes only the hidden reference oligomer. OPLS and Sage
parameters reach the polymer through the same transfer step (see below).

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
- `topology_transfer.py`: transfers a reference-oligomer GROMACS topology to
  the polymer. The transfer works the same way for every engine.
- `third_party/opls-ff-lite/`: bundled copy of opls-ff-lite, without the
  OPLS/2020 SI files. `scripts/sync_opls_ff_lite.sh` updates it from the
  opls-ff-gen repository.
- `polymer_gui/app.py`: Flask web server.
- `polymer_gui/templates/index.html`: local browser interface.
- `polymer_gui/static/app.js`: built-in 2D sketcher and backend calls.
- `examples/PEG_repeat_monomer.sdf`: PEG-like repeat monomer example.
- `examples/PEG_5mer.sdf`: older oligomer-mode pentaethylene glycol example.
- `tests/test_polymer_non_amber.py`: tests for repeat detection, DP build, and charge normalization.
- `tests/test_topology_transfer.py`: tests for the topology transfer and the OPLS engine.
- `tests/test_sage_engine.py`: tests for the Sage engine and the search for an OpenFF interpreter.

The GUI includes a small built-in sketcher, so it does not depend on an
external JavaScript chemistry editor. It can also accept pasted MOL/SDF text.

## Install

```bash
conda env create -f environment.yml          # OPLS-AA/2020 engine
conda activate polymer-gui
# or, to also have the GAFF2/AmberTools engine:
conda env create -f environment-amber.yml
conda activate polymer-gui-amber
```

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
- Sage: `1 2 yes 0.5 0.833333` (Lorentz-Berthelot, fudgeQQ = 1/1.2).

Sage atom types are named by their vdW parameter id, for example `n16`. `polymer.itp` holds
the `[ atomtypes ]` and the `[ moleculetype ]`. Each repeat unit is one
residue, and atoms are named element + number within the unit.

## GAFF2 engine (AmberTools)

AmberTools must provide these executables on `PATH`:

```bash
antechamber
parmchk2
tleap
```

ParmEd is also required for full AMBER-to-GROMACS conversion. A local GROMACS
install is useful for validation with `gmx grompp`, but this V1 backend does
not require `gmx` for the non-AmberTools tests.

When the AmberTools path is used, the generated explicit-H SDF is first
minimized with RDKit. The backend uses MMFF94s when parameters are available
and falls back to UFF otherwise. The minimized structure is written as
`polymer_minimized.sdf` alongside `MINIMIZATION_REPORT.txt`, then converted to
Tripos MOL2 for the actual Antechamber handoff.

In monomer mode, Antechamber is run on the hydrogen-completed hidden reference
oligomer, not the final requested polymer DP. The relevant files are:

```text
reference_oligomer_from_monomer.sdf      heavy-atom reference oligomer
reference_oligomer_for_antechamber.sdf   hydrogenated reference oligomer (all engines)
polymer.sdf                              final requested polymer structure
```

The AmberTools runner also enforces explicit hydrogens immediately before
Antechamber for every workflow. In a real external run it writes:

```text
antechamber_input_with_h.sdf
polymer_minimized.sdf        when minimization is enabled
antechamber_input.mol2       actual file passed to Antechamber
```

Antechamber is run with `-fi mol2` on `antechamber_input.mol2`; this avoids
AmberTools SDF parser failures for larger hidden oligomers whose atom and bond
counts exceed two digits.

For monomer mode, `reference_typed_charged.mol2` stores the GAFF2 atom types
and AM1-BCC charges assigned to the reference oligomer. The first and last
reference units supply terminal templates, while equivalent interior-unit
charges are averaged. The parameters are expanded onto the requested target
in `polymer_typed_charged.mol2`.

`parmchk2` runs on the reference. TLeap then loads the target MOL2 with GAFF2
and the reference-derived `polymer.frcmod`, generating target bonds, angles,
dihedrals, and impropers. The production `polymer.itp`, `polymer.top`, and
`polymer.gro` therefore describe the final requested DP and have the same atom
count as `polymer.sdf`.

## Run the GUI

```bash
python polymer_gui/app.py
```

Then open:

```text
http://127.0.0.1:8501
```

Basic PEG workflow:

1. Click `PEG example`.
2. Click `Preview atom indices`.
3. Keep previous connection atom as `1` and next connection atom as `3`.
4. Keep reference oligomer size as `5`.
5. Keep polymer DP as `12`.
6. Keep the force field as `OPLS-AA/2020` (or choose `GAFF2`).
7. Click `Generate GROMACS files`.

Manual drawing workflow:

1. Use the left toolbar to choose an atom type such as `C`, `O`, `N`, `S`, `H`, `Cl`, `Si`, `P`, `F`, `Br`, or `I`.
2. Drag a line on the canvas to create a single carbon-carbon bond.
3. Drag from an existing atom to an empty point to sprout a new carbon atom and bond.
4. Click an atom to change it to the currently selected element.
5. Click an existing bond to cycle it single -> double -> triple -> single.
6. Click an atom or bond to select it, then press `Delete`/`Backspace` or click `Delete` in the toolbar to remove it.
7. Right-click an atom or bond to remove it immediately.
8. Atom numbers are drawn beside each atom. Click atom tiles to fill the
   previous/next connection atom fields.
9. Click `Use drawing` to export the canvas to MOL/SDF text, then build.

For a stereogenic single bond, choose `Wedge` or `Dash` under **Bond
stereochemistry**, then draw the directed bond from its narrow end to its wide
end. You can also select a style and click an existing single bond. The GUI
stores solid wedges as V2000 stereo code `1` and hashed wedges as code `6`;
these codes are replicated with the monomer and used by RDKit when generating
the three-dimensional geometry of the reference oligomer and the polymer.

For an ionic repeat, choose `-1` or `+1` under **Formal charge**, then click
the charged atom. Choose `0` and click it again to remove the assignment. The
charge is stored in the MOL/SDF `M  CHG` records, displayed in the preview, and
used to infer both the reference-oligomer net charge and the requested
polymer charge. Hydrogen completion is formal-charge aware: for example, N+ is
allowed four bond orders, so a quaternary nitrogen with four single bonds gets
no additional hydrogen, while a three-coordinate N+ gets one.

The GUI writes job folders under `outputs/`.

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

Geometry minimization of the reference oligomer is enabled by default. To bypass it
(GAFF2 only; OPLS CM1A charges need a 3D geometry):

```bash
python polymer_from_oligomer.py examples/PEG_repeat_monomer.sdf \
  --previous-atom 1 \
  --next-atom 3 \
  --reference-dp 5 \
  --dp 12 \
  --outdir outputs/PEG_repeat_DP12_amber_unminimized \
  --no-minimize-geometry
```

The older oligomer mode is still available for compatibility by passing
`--repeat-atoms` instead of `--previous-atom` and `--next-atom`.

## Test

```bash
python -m unittest tests.test_polymer_non_amber tests.test_topology_transfer tests.test_sage_engine
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

## V1 limitations

- Linear homopolymer backbones only.
- The GUI/CLI default is monomer mode. Oligomer mode remains available for
  compatibility and testing.
- The selected repeat atom list can include branches or ring atoms; the backend
  now copies the full selected repeat subgraph and preserves internal bond
  orders.
- SDF/MOL V2000 input only.
- Neighboring repeats must be connected by one inferred backbone junction bond.
- Side groups must be included in the selected repeat atom list if they should
  be replicated on every repeat.
- When external parameterization is disabled or the engine's requirements are
  missing, placeholder `.itp/.top/.gro` files are suitable only for testing
  data flow, atom counts, and charge normalization.
- The polymer coordinates in `polymer.gro` come from the 2D monomer layout,
  except with wedge/dash stereochemistry or a polymer shorter than the
  reference, which get a 3D RDKit geometry. Minimize the structure before
  running dynamics.
