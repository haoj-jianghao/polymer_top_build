"""OPLS/2020 atom typing (OPLS-AA SMARTS rules + refinement) and CM1A charges."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from .lbcc import LBCCApplication, apply_lbcc, find_lbcc_applications
from .opls_typer import find_atomtypes, resolve_atomtypes


# Charges: MOPAC AM1 + opls_ff_lite.cm1a (AMSOL was replaced 2026-10-18).
CHARGE_ENGINE = "mopac"
NEUTRAL_CM1A_SCALE = 1.14
SCHEMA_VERSION = 1

DEFAULT_ATOMTYPE_ARCHIVE = Path(__file__).resolve().parent / "data" / "atomtype_archive.json"
# OPLS-AA SMARTS typing rules (exported from Foyer 1.2.0's oplsaa.xml); only
# the SMARTS and overrides are used, classes and LJ come from OPLS/2020.
DEFAULT_OPLS_RULES = Path(__file__).resolve().parent / "data" / "opls_rules.json"


@dataclass(frozen=True)
class BorrowRule:
    """Assign an existing OPLS-AA type to an atom the SMARTS rules cannot type.

    ``smarts`` is an RDKit SMARTS whose atom mapped ``:1`` is the target.
    ``replaces`` lists SMARTS-rule types the rule may overwrite, and only on an
    atom bonded to a borrowed atom (e.g. the H on a borrowed ring carbon).
    """

    name: str
    smarts: str
    opls_type: str
    replaces: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TypeRule:
    """Replace SMARTS-rule type ``source`` by OPLS/2020 type ``target``.

    ``smarts`` (atom mapped ``:1`` is the typed atom) restricts the rule; an
    empty SMARTS is the default for ``source``. The first matching rule wins.
    ``source`` "*" matches any atom by SMARTS alone, including untyped ones.
    """

    source: str
    target: str
    smarts: str = ""


@dataclass(frozen=True)
class AtomTypeArchive:
    """Project extensions to the OPLS-AA SMARTS typing (``data/atomtype_archive.json``).

    - ``definitions``: SMARTS for OPLS-AA types whose Foyer definition is
      missing or too narrow (added to the OPLS rule set).
    - ``precedence``: preferred type -> types it overrides (added to the overrides).
    - ``borrowed``: existing OPLS-AA types for chemistry the SMARTS rules do
      not type, applied to atoms they leave untyped.
    - ``opls2020_types``: refinement of the rule set's GROMACS-numbered types
      into OPLS/2020 type numbers (new alkane, alkyl-arene, cyclobutane,
      alkyne-adjacent and allene types; numbers whose meaning changed).
    """

    path: Path | None
    definitions: dict[str, str]
    precedence: dict[str, frozenset[str]]
    borrowed: tuple[BorrowRule, ...]
    opls2020_types: tuple[TypeRule, ...] = ()


@lru_cache(maxsize=None)
def load_atomtype_archive(path: str | Path | None = DEFAULT_ATOMTYPE_ARCHIVE) -> AtomTypeArchive:
    """The archive at ``path``; None gives an empty archive (OPLS rules only)."""
    if path is None:
        return AtomTypeArchive(None, {}, {}, ())
    path = Path(path)
    payload = json.loads(path.read_text())
    return AtomTypeArchive(
        path=path,
        definitions={item["type"]: item["smarts"] for item in payload["definitions"]},
        precedence={
            preferred: frozenset().union(*(
                item["over"] for item in payload["precedence"] if item["type"] == preferred
            ))
            for preferred in {item["type"] for item in payload["precedence"]}
        },
        borrowed=tuple(
            BorrowRule(item["name"], item["smarts"], item["type"], frozenset(item["replaces"]))
            for item in payload["borrowed"]
        ),
        opls2020_types=tuple(
            TypeRule(item["from"], item["to"], item.get("smarts", ""))
            for item in payload.get("opls2020_types", [])
        ),
    )


class AssignmentError(RuntimeError):
    """Raised when a molecule cannot be assigned safely."""


@dataclass(frozen=True)
class AtomAssignment:
    index: int
    name: str
    element: str
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float
    symmetry_class: int
    opls_type: str
    opls_class: str
    mass_u: float
    sigma_nm: float
    epsilon_kj_mol: float
    cm1a_raw_e: float
    cm1a_scaled_e: float
    cm1a_symmetry_averaged_e: float
    lbcc_delta_e: float
    final_charge_e: float
    # "smarts" (OPLS SMARTS rules), "smarts:<rule type>" (refined to an
    # OPLS/2020 type), "collision-highest:<candidates>" (rule collision without
    # a precedence entry: highest-numbered type), "archive:opls2020" (untyped
    # atom typed by a "*" rule),
    # "borrowed:<rule name>", "analog:<rule name>" (OPLS/2020 analog of the
    # atom's GAFF2 type, data/gaff2_opls_analogs.json) or "amber:gaff2:<GAFF2 type>"
    type_rule: str = "smarts"


@dataclass(frozen=True)
class AssignmentResult:
    name: str
    net_charge_e: int
    charge_model: str
    charge_scale: float
    atom_count: int
    am1_heat_of_formation_kcal_mol: float
    raw_charge_sum_e: float
    final_charge_sum_e: float
    atoms: tuple[AtomAssignment, ...]
    lbcc_applications: tuple[LBCCApplication, ...]
    output_directory: Path
    json_path: Path
    csv_path: Path
    sdf_path: Path
    charge_engine: str
    charge_input_path: Path
    charge_output_path: Path


def load_sdf(path: str | Path) -> Chem.Mol:
    """Load one explicit-hydrogen, three-dimensional SDF record."""
    path = Path(path)
    records = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol]
    if len(records) != 1:
        raise AssignmentError(
            f"{path}: expected one valid SDF record, found {len(records)}"
        )
    molecule = records[0]
    _validate_molecule(molecule, source=str(path))
    return molecule


def molecule_from_smiles(
    smiles: str, name: str = "molecule", random_seed: int = 20261016
) -> Chem.Mol:
    """Build a deterministic explicit-H 3D conformer from a SMILES string."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AssignmentError(f"Invalid SMILES: {smiles!r}")
    molecule = Chem.AddHs(molecule)
    molecule.SetProp("_Name", name)
    molecule.SetProp("Input_SMILES", smiles)

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = random_seed
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise AssignmentError(f"RDKit could not generate a conformer for {smiles!r}")

    if AllChem.MMFFHasAllMoleculeParams(molecule):
        status = AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000)
        method = "MMFF94"
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        status = AllChem.UFFOptimizeMolecule(molecule, maxIters=1000)
        method = "UFF"
    else:
        raise AssignmentError(
            "No MMFF94 or UFF parameters are available for conformer optimization"
        )
    if status < 0:
        raise AssignmentError(f"{method} conformer optimization failed")
    molecule.SetProp("Conformer_Method", "ETKDGv3")
    molecule.SetProp("Geometry_Optimization", method)
    _validate_molecule(molecule, source=f"SMILES {smiles!r}")
    return molecule


def _validate_molecule(molecule: Chem.Mol, source: str) -> None:
    if molecule.GetNumAtoms() == 0:
        raise AssignmentError(f"{source}: molecule contains no atoms")
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise AssignmentError(f"{source}: exactly one 3D conformer is required")
    with_hydrogens = Chem.AddHs(Chem.Mol(molecule))
    if with_hydrogens.GetNumAtoms() != molecule.GetNumAtoms():
        raise AssignmentError(f"{source}: all hydrogen atoms must be explicit")
    radical_atoms = [
        atom.GetIdx() + 1
        for atom in molecule.GetAtoms()
        if atom.GetNumRadicalElectrons()
    ]
    if radical_atoms:
        raise AssignmentError(
            f"{source}: open-shell radicals are not supported (atoms {radical_atoms})"
        )


@lru_cache(maxsize=None)
def _opls_rule_records() -> tuple[dict, ...]:
    return tuple(json.loads(DEFAULT_OPLS_RULES.read_text())["types"])


def typing_rules(archive: AtomTypeArchive) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Ordered SMARTS definitions and overrides: the OPLS rule table with the
    archive's definitions (replacing a type's SMARTS in place, else appended)
    and precedence (added to the overrides), as Foyer combined them."""
    definitions: dict[str, str] = {}
    overrides: dict[str, set[str]] = {}
    for record in _opls_rule_records():
        if record["smarts"]:
            definitions[record["name"]] = record["smarts"]
        if record["overrides"]:
            overrides[record["name"]] = set(record["overrides"])
    definitions.update(archive.definitions)
    for preferred_type, shadowed_types in archive.precedence.items():
        overrides.setdefault(preferred_type, set()).update(shadowed_types)
    return definitions, overrides


def _borrow_types(
    molecule: Chem.Mol, atom_types: list[str | None], borrow_rules: tuple[BorrowRule, ...]
) -> tuple[list[str | None], list[str]]:
    """Fill untyped atoms with the archive's borrow rules; return types and rules."""
    types = list(atom_types)
    rules = ["smarts"] * len(types)
    proposals: dict[int, set[tuple[str, str]]] = {}
    replacements: dict[int, set[tuple[str, str]]] = {}
    for rule in borrow_rules:
        query = Chem.MolFromSmarts(rule.smarts)
        target = next(a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum() == 1)
        for match in molecule.GetSubstructMatches(query, uniquify=False):
            index = match[target]
            if types[index] is None:
                proposals.setdefault(index, set()).add((rule.opls_type, rule.name))
            elif types[index] in rule.replaces:
                replacements.setdefault(index, set()).add((rule.opls_type, rule.name))
    for index, options in {**proposals}.items():
        if len({opls_type for opls_type, _ in options}) > 1:
            raise AssignmentError(
                f"Borrow rules disagree for atom {index + 1}: {sorted(options)}"
            )
        types[index], name = sorted(options)[0]
        rules[index] = f"borrowed:{name}"
    for index, options in replacements.items():
        borrowed_neighbor = any(
            rules[neighbor.GetIdx()] != "smarts"
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
        )
        if not borrowed_neighbor:
            continue
        if len({opls_type for opls_type, _ in options}) > 1:
            raise AssignmentError(
                f"Borrow rules disagree for atom {index + 1}: {sorted(options)}"
            )
        types[index], name = sorted(options)[0]
        rules[index] = f"borrowed:{name}"
    return types, rules


def smarts_atom_types(
    molecule: Chem.Mol,
    atomtype_archive: str | Path | None = DEFAULT_ATOMTYPE_ARCHIVE,
) -> tuple[list[str | None], list[str]]:
    """First typing stage: OPLS-AA SMARTS rules with the archive's definitions
    and precedence, then its borrow rules for gaps. Returns the rule set's
    (GROMACS-numbered) types, None for atoms still untyped, and the rules
    (``collision-highest:<candidates>`` for a collision no precedence entry
    resolves, typed with the highest-numbered candidate)."""
    archive = load_atomtype_archive(
        None if atomtype_archive is None else Path(atomtype_archive)
    )
    atom_types, collisions = _smarts_atom_types(molecule, archive)
    type_rules = ["smarts"] * len(atom_types)
    for index, candidates in collisions.items():
        type_rules[index] = f"collision-highest:{'|'.join(candidates)}"
    if None in atom_types:
        atom_types, type_rules = _borrow_types(molecule, atom_types, archive.borrowed)
    return atom_types, type_rules


def _assign_atom_types(
    molecule: Chem.Mol,
    atomtype_archive: str | Path | None = DEFAULT_ATOMTYPE_ARCHIVE,
    amber_fallback: bool = True,
    gaff2_analogs: bool = True,
) -> tuple[list[str], dict, list[str]]:
    """OPLS/2020 atom types: ``smarts_atom_types``, refined to OPLS/2020 type
    numbers by the archive's ``opls2020_types`` rules, with classes and LJ
    from the OPLS/2020 table. Atoms still untyped get GAFF2 types when
    ``amber_fallback`` is set, otherwise they are an error; a GAFF2-typed atom
    with an OPLS/2020 analog (``gaff2_analogs.py``, unless ``gaff2_analogs``
    is off) takes that type, the others GAFF2 LJ.

    Returns the types, their parameters, and each atom's typing rule.
    """
    archive = load_atomtype_archive(
        None if atomtype_archive is None else Path(atomtype_archive)
    )
    atom_types, type_rules = smarts_atom_types(molecule, atomtype_archive)
    return _opls2020_atom_types(molecule, atom_types, type_rules, archive, amber_fallback,
                                gaff2_analogs)


def _smarts_targets(molecule: Chem.Mol, smarts: str) -> set[int]:
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise AssignmentError(f"Invalid SMARTS in the atom-type archive: {smarts!r}")
    target = next(a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum() == 1)
    return {match[target] for match in molecule.GetSubstructMatches(query, uniquify=False)}


def _opls2020_atom_types(
    molecule: Chem.Mol,
    atom_types: list[str | None],
    type_rules: list[str],
    archive: AtomTypeArchive,
    amber_fallback: bool = True,
    gaff2_analogs: bool = True,
) -> tuple[list[str], dict, list[str]]:
    """Refine SMARTS-rule types to OPLS/2020 numbers and parameterize them
    from the OPLS/2020 table; atoms that are still untyped take their GAFF2
    type's OPLS/2020 analog if there is one, else GAFF2."""
    from .opls2020 import load_atom_types

    table = load_atom_types()
    periodic_table = Chem.GetPeriodicTable()
    types, rules = list(atom_types), list(type_rules)
    hits: dict[str, set[int]] = {}
    for index, atom_type in enumerate(types):
        for rule in archive.opls2020_types:
            # "*" rules match by SMARTS alone, also atoms the rules left untyped.
            if rule.source != atom_type and rule.source != "*":
                continue
            if atom_type is None and rule.source != "*":
                continue
            if rule.smarts:
                if rule.smarts not in hits:
                    hits[rule.smarts] = _smarts_targets(molecule, rule.smarts)
                if index not in hits[rule.smarts]:
                    continue
            types[index] = rule.target
            if atom_type is None:
                rules[index] = "archive:opls2020"
            elif rules[index] == "smarts":
                rules[index] = f"smarts:{atom_type}"
            break

    parameters: dict[str, dict] = {}
    untyped = [index for index, atom_type in enumerate(types) if atom_type is None]
    if untyped and not amber_fallback:
        details = ", ".join(
            f"{i + 1}:{molecule.GetAtomWithIdx(i).GetSymbol()}" for i in untyped
        )
        raise AssignmentError(
            f"OPLS-AA atom typing failed: no type for atoms {details} "
            f"(not in the OPLS rules or the atom-type archive {archive.path})"
        )
    if untyped:
        from .amber import gaff2_lennard_jones, gaff2_types
        from .gaff2_analogs import analog_types

        gaff_types = gaff2_types(molecule, untyped)
        if gaff2_analogs:
            for index, analog in analog_types(molecule, gaff_types).items():
                types[index] = analog.opls_type
                rules[index] = f"analog:{analog.name}"
            untyped = [index for index in untyped if types[index] is None]
        lennard_jones = gaff2_lennard_jones()
        for index in untyped:
            gaff = gaff_types[index]
            sigma, epsilon = lennard_jones[gaff]
            name = f"gaff2_{gaff}"
            types[index] = name
            rules[index] = f"amber:gaff2:{gaff}"
            atom = molecule.GetAtomWithIdx(index)
            parameters[name] = {
                "class": gaff,
                "element": atom.GetSymbol(),
                "mass": periodic_table.GetAtomicWeight(atom.GetAtomicNum()),
                "sigma": sigma,
                "epsilon": epsilon,
            }

    for index, atom_type in enumerate(types):
        if atom_type in parameters:
            continue
        number = int(re.match(r"opls_(\d+)", atom_type).group(1))
        record = table.get(number)
        if record is None:
            raise AssignmentError(
                f"Type {atom_type} (atom {index + 1}) is not in the OPLS/2020 table"
            )
        atomic_number = molecule.GetAtomWithIdx(index).GetAtomicNum()
        if record.atomic_number != atomic_number:
            raise AssignmentError(
                f"Type {atom_type} is element {record.atomic_number} in OPLS/2020 but atom "
                f"{index + 1} is element {atomic_number}"
            )
        parameters[atom_type] = {
            "class": record.opls_class,
            "element": molecule.GetAtomWithIdx(index).GetSymbol(),
            "mass": periodic_table.GetAtomicWeight(atomic_number),
            "sigma": record.sigma_nm,
            "epsilon": record.epsilon_kj_mol,
        }
    return types, parameters, rules


def _type_number(name: str) -> int:
    match = re.fullmatch(r"opls_(\d+)", name)
    return int(match.group(1)) if match else -1


def _smarts_atom_types(
    molecule: Chem.Mol, archive: AtomTypeArchive
) -> tuple[list[str | None], dict[int, tuple[str, ...]]]:
    """SMARTS typing (opls_typer, Foyer's algorithm) with the OPLS rule table
    and the archive; None for atoms no rule types.

    An atom left with several types (a rule collision the archive's
    precedence does not resolve) takes the highest-numbered one; the second
    value maps each such atom to its candidates.
    """
    definitions, overrides = typing_rules(archive)
    try:
        typemap = find_atomtypes(molecule, definitions, overrides)
    except Exception as exc:
        raise AssignmentError(f"OPLS-AA atom typing failed: {exc}") from exc
    collisions: dict[int, tuple[str, ...]] = {}
    for index, atom in enumerate(typemap):
        candidates = atom["whitelist"] - atom["blacklist"]
        if len(candidates) > 1:
            collisions[index] = tuple(sorted(candidates, key=_type_number))
            atom["whitelist"] = {collisions[index][-1]}
            atom["blacklist"] = set()
    return resolve_atomtypes(typemap), collisions


def _cm1a_charges(
    molecule: Chem.Mol,
    output_directory: Path,
    safe_name: str,
    net_charge: int,
    timeout_seconds: int,
) -> tuple[list[float], float, Path, Path]:
    """Raw CM1A charges (MOPAC AM1 + opls_ff_lite.cm1a), AM1 heat of formation
    and the MOPAC input/output files."""
    from .cm1a import CM1AError, run_cm1a

    try:
        result = run_cm1a(molecule, output_directory / "mopac", safe_name, net_charge,
                          timeout_seconds)
    except CM1AError as exc:
        raise AssignmentError(f"CM1A charge calculation failed: {exc}") from exc
    return ([float(q) for q in result.charges], result.heat_of_formation_kcal_mol,
            result.mopac_input, result.mopac_output)




def _symmetry_average(values: list[float], classes: list[int]) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for symmetry_class, value in zip(classes, values, strict=True):
        grouped.setdefault(symmetry_class, []).append(value)
    averages = {
        symmetry_class: sum(group) / len(group)
        for symmetry_class, group in grouped.items()
    }
    return [averages[symmetry_class] for symmetry_class in classes]


def _normalize_charge(values: list[float], target: int) -> list[float]:
    """Remove printed-charge residual while preserving all charge differences."""
    correction = (target - sum(values)) / len(values)
    return [value + correction for value in values]


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    if not cleaned:
        raise AssignmentError("Molecule name contains no filename-safe characters")
    return cleaned


def assign_molecule(
    molecule: Chem.Mol,
    output_directory: str | Path,
    *,
    name: str | None = None,
    charge_model: str = "auto",
    timeout_seconds: int = 300,
    atomtype_archive: str | Path | None = DEFAULT_ATOMTYPE_ARCHIVE,
    amber_fallback: bool = True,
) -> AssignmentResult:
    """Assign OPLS-AA atom types and CM1A-derived charges to a molecule.

    ``charge_model`` may be ``auto``, ``cm1a``, or ``cm1a-lbcc``. For neutral
    molecules, CM1A is scaled by 1.14. For ions, unscaled CM1A is used and LBCC
    is rejected because the published corrections are defined for neutrals.
    """
    molecule = Chem.Mol(molecule)
    _validate_molecule(molecule, source=name or "molecule")
    if name is None:
        name = molecule.GetProp("_Name") if molecule.HasProp("_Name") else "molecule"
    safe_name = _safe_name(name)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    net_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    if charge_model not in {"auto", "cm1a", "cm1a-lbcc"}:
        raise AssignmentError(
            "charge_model must be one of: auto, cm1a, cm1a-lbcc"
        )
    if charge_model == "auto":
        charge_model = "cm1a-lbcc" if net_charge == 0 else "cm1a"
    if charge_model == "cm1a-lbcc" and net_charge != 0:
        raise AssignmentError("CM1A-LBCC is defined only for neutral molecules")
    charge_scale = NEUTRAL_CM1A_SCALE if net_charge == 0 else 1.0
    model_label = (
        "1.14*CM1A-LBCC"
        if charge_model == "cm1a-lbcc"
        else ("1.14*CM1A" if net_charge == 0 else "CM1A")
    )

    atom_types, opls_parameters, type_rules = _assign_atom_types(
        molecule, atomtype_archive, amber_fallback
    )
    raw_charges, heat_of_formation, charge_input, charge_output = _cm1a_charges(
        molecule, output_directory, safe_name, net_charge, timeout_seconds,
    )

    scaled_charges = [charge_scale * charge for charge in raw_charges]
    symmetry_classes = list(
        Chem.CanonicalRankAtoms(
            molecule, breakTies=False, includeChirality=True, includeIsotopes=True
        )
    )
    for symmetry_class in set(symmetry_classes):
        elements = {
            molecule.GetAtomWithIdx(index).GetSymbol()
            for index, value in enumerate(symmetry_classes)
            if value == symmetry_class
        }
        if len(elements) != 1:
            raise AssignmentError(
                f"RDKit symmetry class {symmetry_class} contains multiple elements"
            )
    symmetry_charges = _normalize_charge(
        _symmetry_average(scaled_charges, symmetry_classes), net_charge
    )

    if charge_model == "cm1a-lbcc":
        applications = find_lbcc_applications(molecule)
        final_charges, lbcc_deltas = apply_lbcc(symmetry_charges, applications)
    else:
        applications = []
        final_charges = symmetry_charges
        lbcc_deltas = [0.0] * molecule.GetNumAtoms()
    if abs(sum(final_charges) - net_charge) > 1e-10:
        raise AssignmentError(
            f"Final charge sum {sum(final_charges):+.12f} does not equal {net_charge}"
        )

    conformer = molecule.GetConformer()
    atoms = []
    for zero_index, atom in enumerate(molecule.GetAtoms()):
        position = conformer.GetAtomPosition(zero_index)
        opls_type = atom_types[zero_index]
        parameters = opls_parameters[opls_type]
        atoms.append(
            AtomAssignment(
                index=zero_index + 1,
                name=f"{atom.GetSymbol()}{zero_index + 1}",
                element=atom.GetSymbol(),
                x_angstrom=position.x,
                y_angstrom=position.y,
                z_angstrom=position.z,
                symmetry_class=int(symmetry_classes[zero_index]),
                opls_type=opls_type,
                opls_class=str(parameters["class"]),
                mass_u=float(parameters["mass"]),
                sigma_nm=float(parameters["sigma"]),
                epsilon_kj_mol=float(parameters["epsilon"]),
                cm1a_raw_e=raw_charges[zero_index],
                cm1a_scaled_e=scaled_charges[zero_index],
                cm1a_symmetry_averaged_e=symmetry_charges[zero_index],
                lbcc_delta_e=lbcc_deltas[zero_index],
                final_charge_e=final_charges[zero_index],
                type_rule=type_rules[zero_index],
            )
        )

    json_path = output_directory / f"{safe_name}.assignment.json"
    csv_path = output_directory / f"{safe_name}.assignment.csv"
    sdf_path = output_directory / f"{safe_name}.assigned.sdf"
    result = AssignmentResult(
        name=name,
        net_charge_e=net_charge,
        charge_model=model_label,
        charge_scale=charge_scale,
        atom_count=molecule.GetNumAtoms(),
        am1_heat_of_formation_kcal_mol=heat_of_formation,
        raw_charge_sum_e=sum(raw_charges),
        final_charge_sum_e=sum(final_charges),
        atoms=tuple(atoms),
        lbcc_applications=tuple(applications),
        output_directory=output_directory,
        json_path=json_path,
        csv_path=csv_path,
        sdf_path=sdf_path,
        charge_engine=CHARGE_ENGINE,
        charge_input_path=charge_input,
        charge_output_path=charge_output,
    )
    _write_outputs(result, molecule)
    return result


def _write_outputs(result: AssignmentResult, molecule: Chem.Mol) -> None:
    with result.csv_path.open("w", newline="") as handle:
        rows = [asdict(atom) for atom in result.atoms]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": result.name,
        "net_charge_e": result.net_charge_e,
        "charge_model": result.charge_model,
        "charge_scale": result.charge_scale,
        "atom_count": result.atom_count,
        "am1_heat_of_formation_kcal_mol": result.am1_heat_of_formation_kcal_mol,
        "raw_charge_sum_e": result.raw_charge_sum_e,
        "final_charge_sum_e": result.final_charge_sum_e,
        "atoms": [asdict(atom) for atom in result.atoms],
        "lbcc_applications": [
            asdict(application) for application in result.lbcc_applications
        ],
        "geometry": {
            key: molecule.GetProp(key)
            for key in ("Input_SMILES", "Input_File", "Input_Format", "Input_Bond_Orders",
                        "Input_Hydrogens_Added", "Conformer_Method", "Geometry_Optimization",
                        "Geometry_Optimization_Seconds", "Geometry_Fallback")
            if molecule.HasProp(key)
        },
        "charge_engine": result.charge_engine,
        "charge_input": str(result.charge_input_path),
        "charge_output": str(result.charge_output_path),
    }
    result.json_path.write_text(json.dumps(payload, indent=2) + "\n")

    assigned = Chem.Mol(molecule)
    assigned.SetProp("_Name", result.name)
    assigned.SetProp("Assignment_Schema_Version", str(SCHEMA_VERSION))
    assigned.SetProp("Charge_Model", result.charge_model)
    assigned.SetProp("Net_Charge_e", str(result.net_charge_e))
    assigned.SetProp(
        "OPLS_Atom_Types", " ".join(atom.opls_type for atom in result.atoms)
    )
    assigned.SetProp(
        "OPLS_Atom_Classes", " ".join(atom.opls_class for atom in result.atoms)
    )
    assigned.SetProp(
        "Partial_Charges_e",
        " ".join(f"{atom.final_charge_e:.10f}" for atom in result.atoms),
    )
    writer = Chem.SDWriter(str(result.sdf_path))
    writer.write(assigned)
    writer.close()
