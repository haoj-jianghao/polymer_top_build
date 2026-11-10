# opls-ff-lite: interface for the polymer builder

This is how a calling program (the polymer topology builder) uses
opls-ff-lite to get OPLS-AA (OPLS/2020) parameters for a molecule or a short
oligomer, neutral or charged. opls-ff-lite 0.1.0; package `opls_ff_lite`.

## 1. What the builder has to provide

| item | how |
|---|---|
| Python 3.12, RDKit >= 2025.9, NumPy | in the builder's environment (conda-forge or pip) |
| the `opls_ff_lite` package | copy `lite/src/opls_ff_lite/` into the builder, or `pip install` a wheel built from `lite/`. It has no other Python dependency and never imports opls-ff-gen. |
| MOPAC (23.2.5 tested) | conda-forge `mopac`. Found via `OPLS_FF_LITE_MOPAC`, then the config file (`opls-ff-lite config set mopac PATH`), then `PATH`. |
| the OPLS/2020 SI files | **each user** downloads them and runs `opls-ff-lite install-si jp3c06602_si_002.txt jp3c06602_si_003.txt`. They go to `~/.local/share/opls-ff-lite/opls2020/`, or `OPLS_FF_LITE_SI_DIR`. |

Rules when bundling opls-ff-lite into the builder's distribution:
- **Never include the SI files.** They are ACS material and may not be
  redistributed. `data/opls2020/` must stay out of any package.
- **Keep `data/` complete,** including `MANIFEST.json`. Every run checks the
  data against the manifest's checksums and stops if a file is missing or
  changed.
- **Ship `LICENSE` and `NOTICE`, and keep `data/parmchk/AmberTools-LICENSE.txt`
  and `README.txt`.** The code is MIT; the two parmchk data files are GPL-3.0.

To check an installation, run `opls-ff-lite doctor` (exit 0 = ready) or call
`opls_ff_lite.tools.doctor()`, which returns `(report_text, ok)`.

## 2. Python API (recommended)

```python
from opls_ff_lite import AssignmentError, generate_gromacs_topology
from opls_ff_lite.structure import prepare_structure
from opls_ff_lite.tools import preflight

errors, warnings = preflight()          # data, SI, MOPAC; errors -> cannot run
if errors:
    raise RuntimeError("\n".join(errors))

molecule = prepare_structure(
    path="oligomer.mol2",               # or smiles="..."; exactly one of the two
    name="paa3",
    optimize=None,                      # files: None/"none" = use the coordinates; "mmff"
    template_smiles=None,               # PDB/MOL2: bond orders + formal charges from this SMILES
    net_charge=-3,                      # optional check against the formal charges
)
result = generate_gromacs_topology(
    molecule, "out/paa3", name="paa3",
    charge_model="auto",                # "auto" | "cm1a" | "cm1a-lbcc"
)
```

### `prepare_structure(...)` returns an RDKit `Chem.Mol` with all H and 3D coordinates

- **SMILES:** ETKDGv3 conformer (fixed seed, so reproducible), then MMFF94.
  Always optimized; `optimize="none"` is an error.
- **SDF/MOL (one record), MOL2, PDB:**
  - Structure files must contain all hydrogens, unless `optimize="mmff"`,
    which adds the missing ones first.
  - PDB has no reliable bond orders: give `template_smiles`, or else they are
    perceived from the geometry and `net_charge`.
  - For MOL2, `template_smiles` also overrides the file's bond orders, which
    differ between writers for nitro and similar groups.
- After MMFF94, the connectivity must be unchanged (no bond broken or
  formed), or `AssignmentError` is raised.
- **Charges come from the formal charges in the input** (e.g. `[O-]`,
  `[NH3+]`). With `net_charge` given, a mismatch raises `AssignmentError`.
- **Atom order is preserved:**
  - structure file: the file's atom order;
  - SMILES: RDKit's heavy-atom order from the SMILES, then the hydrogens.

  Atom `i` (0-based) in the RDKit molecule is atom `i + 1` in every output.
  Build the oligomer as a structure file to control the order yourself.

### `generate_gromacs_topology(molecule, output_directory, *, name=None, charge_model="auto", ...)`

Writes the files of section 4 into `output_directory` and returns
`GromacsTopologyResult`.

Other keyword arguments:
- `torsion_substitution=True`: if False, a torsion missing from every table
  is an error instead of getting a fallback value.
- `qm_bonded_table=True`: use the recorded bond/angle values (section 6).
- `amber_fallback=True`: if False, atoms OPLS cannot type are an error
  instead of getting GAFF2 types.
- `atomtype_archive`: path of the atom-type archive (`None` = OPLS-AA rules only).
- `timeout_seconds=300`: MOPAC time limit.
- `box_padding_nm=1.0`: padding of the `.gro` box.

### The result object

| attribute | content |
|---|---|
| `result.assignment` | `AssignmentResult`: `name`, `net_charge_e`, `charge_model` (`"CM1A"` or `"1.14*CM1A-LBCC"`), `charge_scale`, `atom_count`, `final_charge_sum_e`, `raw_charge_sum_e`, `atoms`, `lbcc_applications`, `json_path`, `csv_path`, `sdf_path`, `charge_input_path`/`charge_output_path` (MOPAC files) |
| `result.assignment.atoms[i]` | `AtomAssignment`: `index` (1-based), `name`, `element`, `x/y/z_angstrom`, `symmetry_class`, `opls_type` (OPLS/2020 type number, e.g. `opls_55`; GAFF2 atoms `gaff2_<type>`), `opls_class` (bonded class, e.g. `CT`; GAFF2 atoms: lower-case GAFF2 type), `mass_u`, `sigma_nm`, `epsilon_kj_mol`, `cm1a_raw_e`, `cm1a_scaled_e`, `cm1a_symmetry_averaged_e`, `lbcc_delta_e`, `final_charge_e`, `type_rule` (how the type was assigned, section 6) |
| `result.bonded` | `BondedAssignment`: `bonds`, `angles`, `proper_torsions`, `improper_torsions`, `pairs`, `warnings`, `nrexcl` (3), `fudge_lj` (0.5), `fudge_qq` (0.5), and `rule_counts()` (rule → number of terms) |
| `result.bonded.bonds[k]` | `atoms` (1-based), `classes`, `length_nm`, `force_constant_kj_mol_nm2`, `rule`, `source_line`, `parameter_classes`, `substitution_penalty` |
| `result.bonded.angles[k]` | `atoms`, `classes`, `angle_deg`, `force_constant_kj_mol_rad2`, `rule`, `source_line` |
| `proper_torsions[k]` | `atoms`, `classes`, `coefficients_kj_mol` (Ryckaert-Bellemans C0-C5), `parameter_classes`, `rule`, `source_line`, `substitution_penalty`, `qm_check` |
| `improper_torsions[k]` | `atoms`, `classes`, `phase_deg`, `force_constant_kj_mol`, `multiplicity`, `parameter_name`, `rule` |
| `pairs[k]` | `atoms`, `function` (1) |
| `result.itp_path`, `top_path`, `gro_path`, `topology_json_path` | the written files |

### Errors

- `AssignmentError` (a `RuntimeError`) is raised for:
  - invalid input, missing hydrogens, or a charge mismatch;
  - a failed conformer or MMFF94 step;
  - an atom OPLS cannot type, when GAFF2 is off;
  - a MOPAC failure or time-out.
- Missing SI or MOPAC: `preflight()` reports it before any work starts.
  Without that check, the run fails at the first table lookup or the MOPAC
  step.
- Nothing is cached between runs. Every call recomputes everything.

## 3. Command line (alternative to the API)

```bash
opls-ff-lite --smiles 'SMILES' --name NAME --output-dir DIR
opls-ff-lite --structure oligomer.mol2 [--net-charge -3] [--optimize mmff] --output-dir DIR
opls-ff-lite --structure oligomer.pdb --template-smiles 'SMILES' --output-dir DIR
python run_opls_ff_lite.py ...          # same, from the source folder without installing
```

| exit code | meaning |
|---|---|
| 0 | done |
| 2 | bad arguments, or the check before the run failed (package data, SI or MOPAC); the message lists each missing item and how to fix it |
| 1 | the run failed (traceback ending in `AssignmentError`) |

- **On stderr:**
  - `WARNING: ...` for SI files that differ from the validated release;
  - `NOTE: ...` for a MOPAC or RDKit version other than the tested ones.
    This is information only; other versions are expected to work.
- **On stdout:** a summary (atoms, net charge, charge model, final charge,
  bonded rule counts, file paths).
- **No QM options exist.** `--qm` is rejected.

## 4. Output files (in the output directory)

| file | content |
|---|---|
| `NAME.itp` | `[ atomtypes ]`, `[ moleculetype ]`, `[ atoms ]`, `[ bonds ]`, `[ pairs ]`, `[ angles ]`, `[ dihedrals ]` (proper), `[ dihedrals ]` (improper) |
| `NAME.top` | `[ defaults ]` `1 3 yes 0.5 0.5`, then `#include "NAME.itp"`, `[ system ]` and `[ molecules ]` (one copy) |
| `NAME.gro` | coordinates in a box padded by `box_padding_nm` |
| `NAME.topology.json` | every interaction with parameters, rule and source (section 5) |
| `NAME.assignment.json` / `.csv` | per-atom types, charges, LJ, geometry record |
| `NAME.assigned.sdf` | the structure with types, classes and final charges as SDF properties |
| `mopac/NAME.mop`, `.out`, `.aux`, `.arc` | the AM1 calculation |

ITP conventions (GROMACS units: nm, kJ/mol, degrees):
- **`[ atomtypes ]`:** one type per atom, named `og<hash>_<NNN>`. The hash is
  8 characters derived from the name and the canonical SMILES, so the names
  are unique to the molecule. `bond_type` is the OPLS class. The section
  also holds σ (nm) and ε (kJ/mol).
- **`[ moleculetype ]`:** named `MOL_<name>_<hash>`, `nrexcl 3`, one residue
  (`resnr 1`, residue name = first 5 letters/digits of the name, upper case).
  Atom names are element + number (`C001`); charges are given to 10 decimals.
- **Bonds** use funct 1, with `b0` in nm and `kb` in kJ mol⁻¹ nm⁻². **Angles**
  use funct 1, with θ0 in degrees and k in kJ mol⁻¹ rad⁻². **Pairs** use
  funct 1; their LJ comes from `gen-pairs` with fudgeLJ.
- **Proper dihedrals** use funct 3: Ryckaert-Bellemans C0-C5 in kJ/mol,
  converted from the OPLS Fourier V1-V4.
- **Improper dihedrals** use funct 4 (periodic: phase, k, multiplicity).
- **Comments:** lines ending in `; <rule>` show where a term did not come
  straight from the OPLS/2020 table; see section 6.

When several such ITPs go into one GROMACS topology, every `[ atomtypes ]`
section must come before the first `[ moleculetype ]`. Move the atomtypes
of each ITP into the top-level file, or include the ITPs only after merging
the atomtypes. Because the names include each molecule's hash, they never
clash.

The builder must also use the OPLS `[ defaults ]`: `nbfunc 1`, `comb-rule 3`
(geometric σ and ε), `gen-pairs yes`, `fudgeLJ 0.5`, `fudgeQQ 0.5`. These
differ from GAFF/AMBER (comb-rule 2, fudgeQQ 0.8333).

## 5. JSON records

### `NAME.topology.json` (`schema_version` 1)

| key | content |
|---|---|
| `name`, `molecule_type`, `residue_name`, `charge_model`, `parameter_file` | identification |
| `defaults` | `{"nrexcl": 3, "fudge_lj": 0.5, "fudge_qq": 0.5}` |
| `atom_types` | the per-atom type names, in atom order |
| `bonds`, `angles`, `proper_torsions`, `improper_torsions`, `pairs` | lists of objects with the same fields as the API objects in section 2 (atoms 1-based) |
| `bonded_rule_counts`, `bonded_warnings`, `borrowed_atoms` | summary of the rules used and any warnings |
| `files` | paths of the ITP, GRO and TOP |
| `qm_method` | always `""` in opls-ff-lite |

### `NAME.assignment.json` (`schema_version` 1)

| key | content |
|---|---|
| `name`, `net_charge_e`, `charge_model`, `charge_scale`, `atom_count` | the molecule and its charge model |
| `raw_charge_sum_e`, `final_charge_sum_e` | charge sums |
| `am1_heat_of_formation_kcal_mol` | from MOPAC |
| `atoms[]` | the `AtomAssignment` fields of section 2 |
| `lbcc_applications[]` | the LBCC corrections applied (neutral molecules) |
| `geometry` | how the structure was made, e.g. `Input_SMILES`, `Input_File`, `Conformer_Method`, `Geometry_Optimization` |
| `charge_engine`, `charge_input`, `charge_output` | the MOPAC files |

## 6. Charges, typing and rules

**Charges** (`charge_model="auto"`):

| molecule | charge model |
|---|---|
| net charge ≠ 0 (any charge, e.g. −3 for a sulfonate trimer) | unscaled CM1A |
| net charge 0, zwitterions included | 1.14×CM1A plus the published LBCC corrections |

- Atoms equivalent by graph symmetry get the same charge.
- The final charges always sum exactly to the net charge (to 1e-6 e).
- `cm1a` or `cm1a-lbcc` forces one model for all molecules.
- The charges belong to the whole oligomer. Any per-monomer charge
  extraction or adjustment for longer chains is up to the builder.

**`type_rule`** of each atom (how its type was assigned):

| value | meaning |
|---|---|
| `smarts` / `smarts:<type>` | OPLS SMARTS rule (`smarts:opls_135` = rule type 135, refined to an OPLS/2020 number such as `opls_55`) |
| `archive:opls2020` | atom-type archive rule |
| `borrowed:<rule>` | an existing OPLS-AA type borrowed for chemistry without one |
| `collision-highest:<types>` | several rules matched with no precedence entry; the highest-numbered type was taken |
| `analog:<rule>` | the OPLS/2020 type analogous to the atom's GAFF2 type |
| `amber:gaff2:<type>` | GAFF2 type and Lennard-Jones (no OPLS type) |

**Bonded `rule`** of each term:

| rule | meaning |
|---|---|
| `exact`, `wildcard`, `improper-family` | the OPLS/2020 table itself |
| `gaff2`, `gaff2-general`, `gaff2-dat-general`, `gaff2-improper-family` | GAFF2 terms |
| `qm-table` | bonds and angles recorded from earlier QM fits of opls-ff-gen (no QM is run) |
| `parmchk2-equa`, `parmchk2-corr`, `parmchk2-equa-general`, `parmchk2-corr-general`, `gaff2-parmchk2-*` | parmchk2-style substitution, with a penalty |
| `default-substitution`, `default-general`, `central-pair` | substitution fallbacks |
| `empirical` | parmchk2's empirical bond and angle rules |
| `zero-torsion` | a torsion with no parameter anywhere, set to zero |

`result.bonded.rule_counts()` (or `bonded_rule_counts` in the JSON) shows at
a glance how much of a molecule came from fallbacks.

## 7. Behaviour to rely on

- **Deterministic:** the same input and options give the same output. SMILES
  use a fixed ETKDG seed and MMFF94; MOPAC is deterministic.
- **Charges depend on geometry:** CM1A is computed on the geometry used. For
  a structure file with `optimize="none"`, that's the builder's coordinates.
- **Speed:** small molecules and 3-5-unit oligomers take seconds. A 158-atom
  PMMA 10-mer took 2.5 s, MOPAC included.
- **Same result as opls-ff-gen:** for the same structure, the topology is
  identical to the full opls-ff-gen's default (`--qm none`, MMFF94 geometry).
  The lite test suite checks this.

## 8. Keeping opls-ff-lite current

opls-ff-lite is developed in `lite/` of the opls-ff-gen repository and
goes into the builder's own repository and distribution; it is not
published on its own.
- Its core modules are copies of opls-ff-gen's, differing only in the
  package name, and its data files are byte-identical.
- `python src/scripts/check_sync.py` (from `lite/`) lists drift from the full
  tool; `--update` re-copies the changed files.
- Then run `python -m unittest discover -s tests`. The tests include the
  dependency guard, charged molecules and oligomers, and parity with the full
  tool.
