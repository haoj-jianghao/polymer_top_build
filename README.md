# Polymer GROMACS Builder

This project wraps `polymer_from_oligomer.py` with a practical local web GUI.
The browser page lets a user draw or paste a monomer, preview atom indices,
mark the atoms that connect to the previous and next monomer, choose the hidden
reference oligomer size and final polymer DP, then
call the same Python backend used by the command line.

The default workflow is now monomer-first:

```text
draw monomer
  -> choose previous/next connection atoms
  -> generate hidden reference oligomer, e.g. 5-mer
  -> add hydrogens to that reference oligomer
  -> run geometry minimization + AmberTools on that oligomer
  -> generate final polymer GROMACS files
```

## What is included

- `polymer_from_oligomer.py`: importable backend plus CLI.
- `polymer_gui/app.py`: Flask web server.
- `polymer_gui/templates/index.html`: local browser interface.
- `polymer_gui/static/app.js`: built-in 2D sketcher and backend calls.
- `examples/PEG_repeat_monomer.sdf`: PEG-like repeat monomer example.
- `examples/PEG_5mer.sdf`: older oligomer-mode pentaethylene glycol example.
- `tests/test_polymer_non_amber.py`: tests for repeat detection, DP build, and charge normalization.

The GUI includes a small built-in sketcher, so it does not depend on an
external JavaScript chemistry editor. It can also accept pasted MOL/SDF text.

## Install

For the GUI only:

```bash
python -m pip install -r requirements.txt
```

For real GAFF2/AM1-BCC parameterization and GROMACS-ready output:

```bash
conda env create -f environment.yml
conda activate polymer-gui
```

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
reference_oligomer_for_antechamber.sdf   hydrogenated reference oligomer
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

1. Click `PEG repeat`.
2. Click `Preview atom indices`.
3. Keep previous connection atom as `1` and next connection atom as `3`.
4. Keep reference oligomer size as `5`.
5. Keep polymer DP as `12`.
6. Click `Generate GROMACS files`.

Manual drawing workflow:

1. Use the left toolbar to choose an atom type such as `C`, `O`, `N`, `S`, `H`, or `Cl`.
2. Drag a line on the canvas to create a single carbon-carbon bond.
3. Drag from an existing atom to an empty point to sprout a new carbon atom and bond.
4. Click an atom to change it to the currently selected element.
5. Click an existing single bond to toggle it into a double bond; click again to return to single.
6. Click an atom or bond to select it, then press `Delete`/`Backspace` or click `Delete` in the toolbar to remove it.
7. Right-click an atom or bond to remove it immediately.
8. Atom numbers are drawn beside each atom. Click atom tiles to fill the
   previous/next connection atom fields.
9. Click `Use drawing` to export the canvas to MOL/SDF text, then build.

For an ionic repeat, choose `-1` or `+1` under **Formal charge**, then click
the charged atom. Choose `0` and click it again to remove the assignment. The
charge is stored in the MOL/SDF `M  CHG` records, displayed in the preview, and
used to infer both the reference-oligomer Antechamber charge and the requested
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

Monomer-mode AmberTools path:

```bash
python polymer_from_oligomer.py examples/PEG_repeat_monomer.sdf \
  --previous-atom 1 \
  --next-atom 3 \
  --reference-dp 5 \
  --dp 12 \
  --outdir outputs/PEG_repeat_DP12_amber
```

If AmberTools is missing, the backend reports the missing executables and still
writes placeholder GROMACS-shaped files for non-external GUI/backend testing.

Geometry minimization before Antechamber is enabled by default. To bypass it:

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
python -m unittest tests.test_polymer_non_amber
```

The tests do not require RDKit, AmberTools, ParmEd, Flask, or GROMACS.

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
- When external parameterization is disabled or AmberTools is unavailable,
  placeholder `.itp/.top/.gro` files are suitable only for testing data flow,
  atom counts, and charge normalization.
- Real force-field assignment still depends on AmberTools, ParmEd, and RDKit.
