"""Input structures and optional geometry optimization.

Inputs
    SMILES, or a structure file: SDF/MOL (one record), MOL2 (Tripos), or PDB.
    PDB has no reliable bond orders; they are assigned from a template SMILES
    when one is given (``template_smiles``), else perceived from the 3D
    geometry with RDKit's ``DetermineBondOrders`` and the net charge. MOL2
    bond orders come from the file, but writers differ for groups such as
    nitro (N.pl3 with single bonds to O.2 reads as N(O-)2), so a template
    SMILES also overrides them. ``net_charge``, when given, is checked against
    the formal charges for every format.
    Structure files must contain every hydrogen, unless the structure is
    optimized, in which case missing hydrogens are added first.

Geometry optimization (``optimize``)
    ``none``  use the input coordinates as they are (structure files only);
    ``mmff``  RDKit MMFF94 (UFF if MMFF94 lacks parameters).
    SMILES are always optimized: ETKDGv3 embedding, then MMFF94 (the only
    optimizer of opls-ff-lite; the full opls-ff-gen also offers xTB). After an
    optimization the bonded topology is re-derived from the new coordinates
    and must equal the input's (no bond broken or formed), or the run stops.

Aromaticity: every route sets the MMFF94 aromaticity model
(``apply_typing_aromaticity``), so a molecule types the same from SMILES,
SDF, MOL2 or PDB.

The CM1A charges and every geometry-derived quantity are computed on the
resulting structure. What was done is stored as molecule properties
(``Input_*``, ``Geometry_*``) and copied to the assignment JSON.
"""

from __future__ import annotations

import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from .assignment import AssignmentError, _validate_molecule

FORMATS = {".sdf": "sdf", ".mol": "sdf", ".sd": "sdf", ".mol2": "mol2", ".pdb": "pdb"}
OPTIMIZERS = ("none", "mmff")
EMBED_SEED = 20261016
MMFF_MAX_ITERATIONS = 1000  # as the original SMILES path, so earlier results are reproduced


def _check_hydrogens(molecule: Chem.Mol, source: str, add: bool) -> Chem.Mol:
    missing = sum(atom.GetNumImplicitHs() for atom in molecule.GetAtoms())
    if not missing:
        return molecule
    if not add:
        raise AssignmentError(
            f"{source}: {missing} hydrogen atom(s) are missing; add hydrogens to the "
            "structure, or use --optimize mmff to add and optimize them"
        )
    molecule = Chem.AddHs(molecule, addCoords=True)
    molecule.SetProp("Input_Hydrogens_Added", str(missing))
    return molecule


def load_structure(
    path: str | Path,
    *,
    template_smiles: str | None = None,
    net_charge: int | None = None,
    add_hydrogens: bool = False,
) -> Chem.Mol:
    """One molecule with 3D coordinates from an SDF/MOL, MOL2 or PDB file."""
    path = Path(path)
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise AssignmentError(f"{path}: unknown structure format (use .sdf, .mol, .mol2 or .pdb)")
    if not path.is_file():
        raise AssignmentError(f"{path}: file not found")
    bond_orders = "file"
    if fmt == "sdf":
        records = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m]
        if len(records) != 1:
            raise AssignmentError(f"{path}: expected one valid SDF record, found {len(records)}")
        molecule = records[0]
    elif fmt == "mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False)
        if molecule is None:
            raise AssignmentError(f"{path}: RDKit could not read the MOL2 file")
        if template_smiles:
            # MOL2 writers differ for nitro, N-oxide, amidinium etc.; a template
            # fixes bond orders and formal charges
            molecule = _bond_orders_from_template(molecule, template_smiles, path)
            bond_orders = "template SMILES"
    else:
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False,
                                       proximityBonding=True)
        if molecule is None:
            raise AssignmentError(f"{path}: RDKit could not read the PDB file")
        if template_smiles:
            molecule = _bond_orders_from_template(molecule, template_smiles, path)
            bond_orders = "template SMILES"
        else:
            try:
                for bond in molecule.GetBonds():
                    bond.SetBondType(Chem.BondType.SINGLE)
                rdDetermineBonds.DetermineBondOrders(molecule, charge=int(net_charge or 0))
                Chem.SanitizeMol(molecule)
            except Exception as exc:
                raise AssignmentError(
                    f"{path}: could not perceive bond orders from the geometry "
                    f"(net charge {net_charge or 0}); give --template-smiles or --net-charge: {exc}"
                ) from exc
            bond_orders = f"perceived from 3D geometry (net charge {net_charge or 0})"
        molecule = Chem.Mol(molecule)
    if len(Chem.GetMolFrags(molecule)) > 1:
        raise AssignmentError(f"{path}: the structure contains more than one molecule")
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise AssignmentError(f"{path}: a 3D structure is required")
    molecule = _check_hydrogens(molecule, str(path), add_hydrogens)
    if net_charge is not None and Chem.GetFormalCharge(molecule) != net_charge:
        raise AssignmentError(
            f"{path}: formal charges sum to {Chem.GetFormalCharge(molecule)}, "
            f"not the given net charge {net_charge} (read as "
            f"{Chem.MolToSmiles(Chem.RemoveHs(molecule))}); give --template-smiles to set "
            "bond orders and charges"
        )
    molecule.SetProp("_Name", molecule.GetProp("_Name") if molecule.HasProp("_Name")
                     and molecule.GetProp("_Name").strip() else path.stem)
    molecule.SetProp("Input_File", str(path.resolve()))
    molecule.SetProp("Input_Format", fmt)
    molecule.SetProp("Input_Bond_Orders", bond_orders)
    return molecule


def _bond_orders_from_template(molecule: Chem.Mol, smiles: str, path: Path) -> Chem.Mol:
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise AssignmentError(f"Invalid template SMILES {smiles!r}")
    has_h = any(a.GetAtomicNum() == 1 for a in molecule.GetAtoms())
    try:
        if has_h:
            return AllChem.AssignBondOrdersFromTemplate(Chem.AddHs(template), molecule)
        return AllChem.AssignBondOrdersFromTemplate(template, molecule)
    except Exception as exc:
        raise AssignmentError(
            f"{path}: the template SMILES does not match the structure: {exc}"
        ) from exc


def _bond_set(molecule: Chem.Mol) -> set[frozenset[int]]:
    return {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in molecule.GetBonds()}


def _topology_change(original: Chem.Mol, optimized: Chem.Mol) -> str | None:
    """None if the connectivity perceived from the optimized coordinates equals
    the input's, else a description of the broken and formed bonds."""
    probe = Chem.RWMol(Chem.Mol(optimized))
    for bond in list(probe.GetBonds()):
        probe.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    rdDetermineBonds.DetermineConnectivity(probe, useHueckel=False)
    before, after = _bond_set(original), _bond_set(probe)
    if before == after:
        return None
    broken = sorted(tuple(sorted(i + 1 for i in b)) for b in before - after)
    formed = sorted(tuple(sorted(i + 1 for i in b)) for b in after - before)
    return f"broken {broken}, formed {formed}"


def _check_topology_kept(original: Chem.Mol, optimized: Chem.Mol, method: str) -> None:
    """The connectivity perceived from the optimized coordinates must equal the input's."""
    change = _topology_change(original, optimized)
    if change is not None:
        raise AssignmentError(
            f"{method} optimization changed the bonding ({change}); "
            "check the input structure or protonation state"
        )


def _optimize_mmff(molecule: Chem.Mol) -> str:
    """MMFF94 (UFF if MMFF94 lacks parameters) in place; returns the label."""
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        status, label = AllChem.MMFFOptimizeMolecule(molecule, maxIters=MMFF_MAX_ITERATIONS), "MMFF94"
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        status, label = AllChem.UFFOptimizeMolecule(molecule, maxIters=MMFF_MAX_ITERATIONS), "UFF"
    else:
        raise AssignmentError("No MMFF94 or UFF parameters are available for optimization")
    if status < 0:
        raise AssignmentError(f"{label} optimization failed")
    if status == 1:
        label += f" (not converged in {MMFF_MAX_ITERATIONS} steps)"
    return label


def optimize_geometry(molecule: Chem.Mol, method: str) -> Chem.Mol:
    """Copy of ``molecule`` optimized with ``mmff`` (``none``: unchanged). The
    MMFF94 geometry must keep the bonding, or the run stops."""
    if method not in OPTIMIZERS:
        raise AssignmentError(f"Unknown optimization {method!r}; choose from {OPTIMIZERS}")
    molecule = Chem.Mol(molecule)
    if method == "none":
        molecule.SetProp("Geometry_Optimization", "none (input coordinates)")
        return molecule
    original = Chem.Mol(molecule)
    clock = time.perf_counter()
    label = _optimize_mmff(molecule)
    seconds = time.perf_counter() - clock
    _check_topology_kept(original, molecule, label)
    molecule.SetProp("Geometry_Optimization", label)
    molecule.SetProp("Geometry_Optimization_Seconds", f"{seconds:.2f}")
    return molecule


def apply_typing_aromaticity(molecule: Chem.Mol) -> Chem.Mol:
    """Set the aromaticity flags (in place) to the MMFF94 model used for typing.

    Atom typing, GAFF2 fallback and impropers read the aromaticity flags, so
    every input route gets the same model: MMFF94, which RDKit's MMFF set-up
    applies in place (it differs from RDKit's default model for e.g. coumarin, xanthine and
    4-hydroxycoumarin rings).
    """
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_MMFF94)
    return molecule


def molecule_from_smiles_optimized(smiles: str, name: str, method: str = "mmff") -> Chem.Mol:
    """ETKDGv3 conformer from SMILES, then MMFF94 optimization."""
    if method == "none":
        raise AssignmentError("A SMILES input is always optimized: use --optimize mmff")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AssignmentError(f"Invalid SMILES: {smiles!r}")
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = EMBED_SEED
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise AssignmentError(f"RDKit could not generate a conformer for {smiles!r}")
    molecule.SetProp("_Name", name)
    molecule.SetProp("Input_SMILES", smiles)
    molecule.SetProp("Conformer_Method", "ETKDGv3")
    apply_typing_aromaticity(molecule)
    molecule = optimize_geometry(molecule, method)
    _validate_molecule(molecule, source=f"SMILES {smiles!r}")
    return molecule


def prepare_structure(
    *,
    smiles: str | None = None,
    path: str | Path | None = None,
    name: str | None = None,
    optimize: str | None = None,
    template_smiles: str | None = None,
    net_charge: int | None = None,
) -> Chem.Mol:
    """The 3D molecule the workflow parameterizes (see the module docstring).

    ``optimize`` defaults to ``mmff`` for SMILES (always) and ``none`` for files.
    """
    if (smiles is None) == (path is None):
        raise AssignmentError("Give exactly one of a SMILES or a structure file")
    if smiles is not None:
        return molecule_from_smiles_optimized(smiles, name or "molecule", optimize or "mmff")
    method = optimize or "none"
    molecule = load_structure(path, template_smiles=template_smiles, net_charge=net_charge,
                              add_hydrogens=method != "none")
    apply_typing_aromaticity(molecule)
    molecule = optimize_geometry(molecule, method)
    if name:
        molecule.SetProp("_Name", name)
    _validate_molecule(molecule, source=str(path))
    return molecule
