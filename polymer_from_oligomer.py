#!/usr/bin/env python3
"""Build a homopolymer model from a drawn/reference oligomer.

This module is intentionally importable.  The CLI and the local web GUI both
call ``build_polymer_from_oligomer`` so the UI does not duplicate chemistry
logic.

V1 scope:
* linear homopolymer backbones with branched or ring-containing repeat units
* SDF/MOL V2000 input
* automatic repeat occurrence detection from one user-selected example
* simple valence-based hydrogen completion for non-AmberTools tests
* AmberTools/ParmEd orchestration hooks for real GAFF2/GROMACS output
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_VALENCE = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

FORMAL_VALENCE = {
    ("C", -1): 3,
    ("C", 1): 3,
    ("N", -1): 2,
    ("N", 0): 3,
    ("N", 1): 4,
    ("O", -1): 1,
    ("O", 0): 2,
    ("O", 1): 3,
    ("P", 1): 4,
    ("S", -1): 1,
    ("S", 0): 2,
    ("S", 1): 3,
}

V2000_CHARGE_CODE = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}

ATOMIC_MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}


@dataclass
class Atom:
    index: int
    element: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    atom_type: str = ""
    charge: float = 0.0
    repeat_atom: bool = False
    formal_charge: int = 0


@dataclass
class Bond:
    a: int
    b: int
    order: int = 1


@dataclass
class Molecule:
    name: str
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)

    def adjacency(self, include_h: bool = True) -> dict[int, list[int]]:
        keep = {atom.index for atom in self.atoms if include_h or atom.element != "H"}
        graph = {idx: [] for idx in keep}
        for bond in self.bonds:
            if bond.a in keep and bond.b in keep:
                graph[bond.a].append(bond.b)
                graph[bond.b].append(bond.a)
        return graph

    def atom(self, index: int) -> Atom:
        return self.atoms[index - 1]

    def formula(self) -> str:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.element] = counts.get(atom.element, 0) + 1
        ordered = []
        for element in ["C", "H"]:
            if element in counts:
                ordered.append((element, counts.pop(element)))
        ordered.extend(sorted(counts.items()))
        return "".join(f"{el}{'' if count == 1 else count}" for el, count in ordered)

    def molecular_weight(self) -> float:
        return sum(ATOMIC_MASS.get(atom.element, 0.0) for atom in self.atoms)

    def total_formal_charge(self) -> int:
        return sum(atom.formal_charge for atom in self.atoms)


@dataclass
class BuildOptions:
    repeat_atoms: list[int]
    dp: int
    oligomer_charge: int = 0
    repeat_charge: int = 0
    end_charge: int = 0
    outdir: Path = Path("polymer_out")
    run_external: bool = True
    minimize_geometry: bool = True


@dataclass
class MonomerBuildOptions:
    previous_atom: int
    next_atom: int
    polymer_dp: int
    reference_dp: int = 5
    outdir: Path = Path("polymer_out")
    run_external: bool = True
    minimize_geometry: bool = True
    junction_order: int = 1


@dataclass
class BuildResult:
    molecule: Molecule
    repeat_units: list[list[int]]
    target_charge: int
    raw_charge: float
    final_charge: float
    correction_per_repeat_atom: float
    files: list[Path]
    missing_tools: list[str]


@dataclass
class AmberTargetSpec:
    """Describe how AmberTools reference parameters are applied to a target."""

    molecule: Molecule
    target_charge: int
    unit_atom_count: int | None = None
    reference_dp: int | None = None
    target_dp: int | None = None
    raw_charge: float = 0.0
    final_charge: float = 0.0
    correction_per_repeat_atom: float = 0.0


@dataclass
class RepeatOccurrence:
    atoms: list[int]

    @property
    def atom_set(self) -> set[int]:
        return set(self.atoms)


@dataclass
class RepeatPlan:
    query_atoms: list[int]
    occurrences: list[RepeatOccurrence]
    junction_left_pos: int | None = None
    junction_right_pos: int | None = None
    junction_order: int = 1


def parse_index_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one repeat atom index is required.")
    if len(set(values)) != len(values):
        raise ValueError("Repeat atom indices must be unique.")
    return values


def read_sdf(path: Path) -> Molecule:
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path} is too short to be an SDF/MOL file.")
    name = lines[0].strip() or path.stem
    counts = lines[3]
    try:
        natoms = int(counts[0:3])
        nbonds = int(counts[3:6])
    except ValueError as exc:
        raise ValueError("Only V2000 SDF/MOL count lines are supported in V1.") from exc

    atoms: list[Atom] = []
    for idx, line in enumerate(lines[4 : 4 + natoms], start=1):
        charge_code_text = line[36:39].strip() if len(line) >= 39 else ""
        charge_code = int(charge_code_text) if charge_code_text else 0
        atoms.append(
            Atom(
                index=idx,
                x=float(line[0:10]),
                y=float(line[10:20]),
                z=float(line[20:30]),
                element=line[31:34].strip(),
                formal_charge=V2000_CHARGE_CODE.get(charge_code, 0),
            )
        )

    bonds: list[Bond] = []
    for line in lines[4 + natoms : 4 + natoms + nbonds]:
        bonds.append(Bond(a=int(line[0:3]), b=int(line[3:6]), order=int(line[6:9])))

    for line in lines[4 + natoms + nbonds :]:
        if not line.startswith("M  CHG"):
            continue
        fields = line.split()
        try:
            count = int(fields[2])
            pairs = fields[3:]
            if len(pairs) != count * 2:
                raise ValueError
            for pair_index in range(count):
                atom_index = int(pairs[pair_index * 2])
                formal_charge = int(pairs[pair_index * 2 + 1])
                if atom_index < 1 or atom_index > len(atoms):
                    raise ValueError
                atoms[atom_index - 1].formal_charge = formal_charge
        except ValueError as exc:
            raise ValueError(f"Malformed V2000 M  CHG record in {path}: {line}") from exc
    return Molecule(name=name, atoms=atoms, bonds=bonds)


def write_sdf(mol: Molecule, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [mol.name, "  polymer_from_oligomer", ""]
    lines.append(f"{len(mol.atoms):>3}{len(mol.bonds):>3}  0  0  0  0            999 V2000")
    for atom in mol.atoms:
        lines.append(
            f"{atom.x:10.4f}{atom.y:10.4f}{atom.z:10.4f} "
            f"{atom.element:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for bond in mol.bonds:
        lines.append(f"{bond.a:>3}{bond.b:>3}{bond.order:>3}  0  0  0  0")
    charged_atoms = [
        (atom.index, atom.formal_charge) for atom in mol.atoms if atom.formal_charge
    ]
    for start_index in range(0, len(charged_atoms), 8):
        chunk = charged_atoms[start_index : start_index + 8]
        lines.append(
            f"M  CHG{len(chunk):>3}"
            + "".join(f"{atom_index:>4}{charge:>4}" for atom_index, charge in chunk)
        )
    lines.extend(["M  END", "$$$$"])
    path.write_text("\n".join(lines) + "\n")


def _bond_order_label(order: int) -> str:
    return {1: "1", 2: "2", 3: "3"}.get(order, "1")


def _sybyl_atom_type(mol: Molecule, atom: Atom) -> str:
    element = atom.element
    attached_orders = [
        bond.order
        for bond in mol.bonds
        if bond.a == atom.index or bond.b == atom.index
    ]
    has_double = any(order == 2 for order in attached_orders)
    has_triple = any(order == 3 for order in attached_orders)
    if element == "C":
        if has_triple:
            return "C.1"
        if has_double:
            return "C.2"
        return "C.3"
    if element == "N":
        if atom.formal_charge > 0 and sum(attached_orders) >= 4:
            return "N.4"
        if has_triple:
            return "N.1"
        if has_double:
            return "N.2"
        return "N.3"
    if element == "O":
        if has_double:
            return "O.2"
        return "O.3"
    if element == "S":
        return "S.3"
    if element in {"F", "Cl", "Br", "I", "H", "P"}:
        return element
    return element


def write_mol2(
    mol: Molecule,
    path: Path,
    residue_name: str = "MOL",
    molecule_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_name = molecule_name or mol.name.strip() or path.stem
    # TLeap corrupts prmtop output and can segfault with molecule names longer
    # than its legacy fixed-width internal buffer.
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name)[:30] or "MOL"
    residue = re.sub(r"[^A-Za-z0-9_]", "", residue_name.upper())[:8] or "MOL"
    lines = [
        "@<TRIPOS>MOLECULE",
        name,
        f"{len(mol.atoms)} {len(mol.bonds)} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for atom in mol.atoms:
        atom_name = f"{atom.element}{atom.index}"
        atom_type = atom.atom_type or _sybyl_atom_type(mol, atom)
        lines.append(
            f"{atom.index:>7} {atom_name:<8} "
            f"{atom.x:>10.4f} {atom.y:>10.4f} {atom.z:>10.4f} "
            f"{atom_type:<6} 1 {residue:<8} {atom.charge:>12.8f}"
        )
    lines.extend(["", "@<TRIPOS>BOND"])
    for idx, bond in enumerate(mol.bonds, start=1):
        lines.append(f"{idx:>6} {bond.a:>5} {bond.b:>5} {_bond_order_label(bond.order)}")
    lines.append("")
    path.write_text("\n".join(lines))
    return None


def read_mol2_atom_parameters(path: Path) -> list[tuple[str, float]]:
    """Read atom types and charges from the ATOM section of a Tripos MOL2."""

    parameters: list[tuple[str, float]] = []
    in_atoms = False
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>"):
            if in_atoms:
                break
            continue
        if not in_atoms or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed MOL2 atom record in {path}: {line}")
        expected_index = len(parameters) + 1
        try:
            atom_index = int(fields[0])
            charge = float(fields[8])
        except ValueError as exc:
            raise ValueError(f"Malformed MOL2 atom record in {path}: {line}") from exc
        if atom_index != expected_index:
            raise ValueError(
                f"MOL2 atoms in {path} must be sequential and 1-based; "
                f"expected {expected_index}, found {atom_index}."
            )
        parameters.append((fields[5], charge))
    if not parameters:
        raise ValueError(f"No MOL2 atom parameters were found in {path}.")
    return parameters


def copy_typed_parameters_by_index(
    reference: Molecule,
    parameters: list[tuple[str, float]],
    target: Molecule,
) -> None:
    """Copy Antechamber results when reference and target are the same graph."""

    if len(reference.atoms) != len(parameters) or len(target.atoms) != len(parameters):
        raise ValueError(
            "AmberTools atom count mismatch: "
            f"reference={len(reference.atoms)}, typed MOL2={len(parameters)}, "
            f"target={len(target.atoms)}."
        )
    for target_atom, (atom_type, charge) in zip(target.atoms, parameters):
        target_atom.atom_type = atom_type
        target_atom.charge = charge


def _generated_hydrogens_by_parent(mol: Molecule, base_atom_count: int) -> dict[int, list[int]]:
    """Index valence-added hydrogens by their base atom, preserving atom order."""

    generated = {atom.index for atom in mol.atoms[base_atom_count:]}
    by_parent: dict[int, list[int]] = {idx: [] for idx in range(1, base_atom_count + 1)}
    for bond in mol.bonds:
        if bond.a in generated and bond.b <= base_atom_count:
            by_parent[bond.b].append(bond.a)
        elif bond.b in generated and bond.a <= base_atom_count:
            by_parent[bond.a].append(bond.b)
    for indices in by_parent.values():
        indices.sort()
    return by_parent


def transfer_reference_atom_parameters(
    reference: Molecule,
    parameters: list[tuple[str, float]],
    target: Molecule,
    unit_atom_count: int,
    reference_dp: int,
    target_dp: int,
) -> None:
    """Map reference head/interior/tail GAFF types and charges to a target."""

    if unit_atom_count < 1:
        raise ValueError("The monomer must contain at least one atom.")
    if reference_dp < 2 or target_dp < 1:
        raise ValueError("Reference DP must be at least 2 and target DP at least 1.")
    if target_dp > 2 and reference_dp < 3:
        raise ValueError("A target with interior units requires a reference oligomer of at least 3 units.")

    reference_base_count = unit_atom_count * reference_dp
    target_base_count = unit_atom_count * target_dp
    if len(reference.atoms) != len(parameters):
        raise ValueError(
            "Antechamber reference atom count mismatch: "
            f"reference has {len(reference.atoms)}, typed MOL2 has {len(parameters)}."
        )
    if len(reference.atoms) < reference_base_count:
        raise ValueError("Reference molecule is missing one or more monomer base atoms.")
    if len(target.atoms) < target_base_count:
        raise ValueError("Target molecule is missing one or more monomer base atoms.")

    reference_h = _generated_hydrogens_by_parent(reference, reference_base_count)
    target_h = _generated_hydrogens_by_parent(target, target_base_count)

    def reference_units_for(target_unit: int) -> list[int]:
        if target_dp == 1:
            return [0, reference_dp - 1]
        if target_unit == 0:
            return [0]
        if target_unit == target_dp - 1:
            return [reference_dp - 1]
        return list(range(1, reference_dp - 1))

    def averaged_parameter(reference_indices: list[int], description: str) -> tuple[str, float]:
        values = [parameters[index - 1] for index in reference_indices]
        atom_types = {atom_type for atom_type, _ in values}
        if len(atom_types) != 1:
            raise ValueError(
                f"Reference GAFF atom types disagree for {description}: "
                + ", ".join(sorted(atom_types))
            )
        atom_type = values[0][0]
        return atom_type, sum(charge for _, charge in values) / len(values)

    for target_unit in range(target_dp):
        reference_units = reference_units_for(target_unit)
        for local_position in range(unit_atom_count):
            target_index = target_unit * unit_atom_count + local_position + 1
            candidate_indices = [
                reference_unit * unit_atom_count + local_position + 1
                for reference_unit in reference_units
            ]
            target_hydrogens = target_h[target_index]
            reference_indices = [
                index
                for index in candidate_indices
                if len(reference_h[index]) == len(target_hydrogens)
            ]
            if not reference_indices:
                reference_counts = sorted({len(reference_h[index]) for index in candidate_indices})
                raise ValueError(
                    "Reference/target hydrogen environments disagree for "
                    f"unit atom {local_position + 1} in target unit {target_unit + 1}: "
                    f"reference counts={reference_counts}, target count={len(target_hydrogens)}."
                )
            atom_type, charge = averaged_parameter(
                reference_indices,
                f"unit atom {local_position + 1} in target unit {target_unit + 1}",
            )
            target_atom = target.atom(target_index)
            target_atom.atom_type = atom_type
            target_atom.charge = charge

            reference_hydrogen_lists = [reference_h[index] for index in reference_indices]
            for hydrogen_position, target_hydrogen in enumerate(target_hydrogens):
                hydrogen_references = [
                    indices[hydrogen_position] for indices in reference_hydrogen_lists
                ]
                atom_type, charge = averaged_parameter(
                    hydrogen_references,
                    f"hydrogen {hydrogen_position + 1} on unit atom {local_position + 1}",
                )
                target_atom = target.atom(target_hydrogen)
                target_atom.atom_type = atom_type
                target_atom.charge = charge


def clone_molecule(mol: Molecule, name: str | None = None) -> Molecule:
    return Molecule(
        name=name or mol.name,
        atoms=[
            Atom(
                index=atom.index,
                element=atom.element,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                atom_type=atom.atom_type,
                charge=atom.charge,
                repeat_atom=atom.repeat_atom,
                formal_charge=atom.formal_charge,
            )
            for atom in mol.atoms
        ],
        bonds=[Bond(bond.a, bond.b, bond.order) for bond in mol.bonds],
    )


def bond_between(mol: Molecule, a: int, b: int) -> Bond | None:
    for bond in mol.bonds:
        if (bond.a == a and bond.b == b) or (bond.a == b and bond.b == a):
            return bond
    return None


def selected_bond_orders(mol: Molecule, atoms: list[int]) -> dict[tuple[int, int], int]:
    selected = set(atoms)
    positions = {atom: pos for pos, atom in enumerate(atoms)}
    orders: dict[tuple[int, int], int] = {}
    for bond in mol.bonds:
        if bond.a in selected and bond.b in selected:
            i = positions[bond.a]
            j = positions[bond.b]
            orders[tuple(sorted((i, j)))] = bond.order
    return orders


def find_repeat_matches(mol: Molecule, repeat_atoms: list[int]) -> list[list[int]]:
    query_orders = selected_bond_orders(mol, repeat_atoms)
    candidates = [
        [
            atom.index
            for atom in mol.atoms
            if atom.element == mol.atom(query_atom).element
            and atom.formal_charge == mol.atom(query_atom).formal_charge
            and atom.element != "H"
        ]
        for query_atom in repeat_atoms
    ]
    order = sorted(range(len(repeat_atoms)), key=lambda pos: len(candidates[pos]))
    assignment: dict[int, int] = {}
    used: set[int] = set()
    matches: list[list[int]] = []

    def compatible(query_pos: int, target_atom: int) -> bool:
        for assigned_query_pos, assigned_target in assignment.items():
            query_bond = query_orders.get(tuple(sorted((query_pos, assigned_query_pos))))
            target_bond = bond_between(mol, target_atom, assigned_target)
            if query_bond is None and target_bond is not None:
                return False
            if query_bond is not None:
                if target_bond is None or target_bond.order != query_bond:
                    return False
        return True

    def backtrack(step: int) -> None:
        if step == len(order):
            matches.append([assignment[pos] for pos in range(len(repeat_atoms))])
            return
        query_pos = order[step]
        for target_atom in candidates[query_pos]:
            if target_atom in used:
                continue
            if compatible(query_pos, target_atom):
                assignment[query_pos] = target_atom
                used.add(target_atom)
                backtrack(step + 1)
                used.remove(target_atom)
                del assignment[query_pos]

    backtrack(0)
    return matches


def unique_occurrences(matches: list[list[int]], repeat_atoms: list[int]) -> list[RepeatOccurrence]:
    selected_set = set(repeat_atoms)
    by_set: dict[frozenset[int], list[int]] = {}
    for match in matches:
        key = frozenset(match)
        if key == selected_set:
            by_set[key] = repeat_atoms[:]
        elif key not in by_set:
            by_set[key] = match
    return [RepeatOccurrence(atoms=atoms) for _, atoms in sorted(by_set.items(), key=lambda item: min(item[0]))]


def external_positions(mol: Molecule, occurrence: RepeatOccurrence) -> set[int]:
    occurrence_set = occurrence.atom_set
    positions = {atom: pos for pos, atom in enumerate(occurrence.atoms)}
    external = set()
    for bond in mol.bonds:
        if bond.a in occurrence_set and bond.b not in occurrence_set:
            external.add(positions[bond.a])
        elif bond.b in occurrence_set and bond.a not in occurrence_set:
            external.add(positions[bond.b])
    return external


def inter_occurrence_bonds(mol: Molecule, left: RepeatOccurrence, right: RepeatOccurrence) -> list[tuple[int, int, int]]:
    left_set = left.atom_set
    right_set = right.atom_set
    bonds = []
    for bond in mol.bonds:
        if bond.a in left_set and bond.b in right_set:
            bonds.append((bond.a, bond.b, bond.order))
        elif bond.b in left_set and bond.a in right_set:
            bonds.append((bond.b, bond.a, bond.order))
    return bonds


def order_repeat_occurrences(mol: Molecule, occurrences: list[RepeatOccurrence], repeat_atoms: list[int]) -> list[RepeatOccurrence]:
    selected = next((occ for occ in occurrences if occ.atoms == repeat_atoms), None)
    if selected is None:
        selected = RepeatOccurrence(atoms=repeat_atoms)
        occurrences.append(selected)

    def grow(start: RepeatOccurrence, direction: int, blocked: set[frozenset[int]]) -> list[RepeatOccurrence]:
        chain = []
        current = start
        while True:
            current_min = min(current.atoms)
            candidates = []
            for occurrence in occurrences:
                key = frozenset(occurrence.atoms)
                if key in blocked or occurrence.atom_set & set().union(*(set(item) for item in blocked)):
                    continue
                if direction > 0 and min(occurrence.atoms) <= current_min:
                    continue
                if direction < 0 and min(occurrence.atoms) >= current_min:
                    continue
                if not inter_occurrence_bonds(mol, current, occurrence):
                    continue
                ext_count = len(external_positions(mol, occurrence))
                if ext_count > 2:
                    continue
                candidates.append((ext_count, abs(min(occurrence.atoms) - current_min), min(occurrence.atoms), occurrence))
            if not candidates:
                break
            occurrence = sorted(candidates, key=lambda item: item[:3])[0][3]
            chain.append(occurrence)
            blocked.add(frozenset(occurrence.atoms))
            current = occurrence
        return chain

    blocked = {frozenset(selected.atoms)}
    backward = grow(selected, -1, blocked)
    forward = grow(selected, 1, blocked)
    return list(reversed(backward)) + [selected] + forward


def make_repeat_plan(mol: Molecule, repeat_atoms: list[int]) -> RepeatPlan:
    matches = find_repeat_matches(mol, repeat_atoms)
    occurrences = order_repeat_occurrences(mol, unique_occurrences(matches, repeat_atoms), repeat_atoms)
    if not occurrences:
        raise ValueError("No repeat units matching the selected atom graph were detected.")
    plan = RepeatPlan(query_atoms=repeat_atoms, occurrences=occurrences)
    if len(occurrences) > 1:
        junctions = inter_occurrence_bonds(mol, occurrences[0], occurrences[1])
        if not junctions:
            raise ValueError("Detected repeat units are not connected by an inter-repeat bond.")
        if len(junctions) > 1:
            raise ValueError("V1 supports one backbone junction bond between neighboring repeat units.")
        left_atom, right_atom, order = junctions[0]
        plan.junction_left_pos = occurrences[0].atoms.index(left_atom)
        plan.junction_right_pos = occurrences[1].atoms.index(right_atom)
        plan.junction_order = order
    return plan


def detect_repeat_units(mol: Molecule, repeat_atoms: list[int]) -> tuple[list[int], list[list[int]]]:
    plan = make_repeat_plan(mol, repeat_atoms)
    return [atom for occurrence in plan.occurrences for atom in occurrence.atoms], [occurrence.atoms for occurrence in plan.occurrences]


def collect_external_fragment(mol: Molecule, start_occurrence: RepeatOccurrence, repeat_union: set[int]) -> set[int]:
    seeds = set()
    start_set = start_occurrence.atom_set
    for bond in mol.bonds:
        if bond.a in start_set and bond.b not in repeat_union:
            seeds.add(bond.b)
        elif bond.b in start_set and bond.a not in repeat_union:
            seeds.add(bond.a)
    fragment = set(seeds)
    queue = list(seeds)
    while queue:
        atom = queue.pop(0)
        for nbr in mol.adjacency(include_h=False).get(atom, []):
            if nbr in repeat_union or nbr in fragment:
                continue
            fragment.add(nbr)
            queue.append(nbr)
    return fragment


def copy_source_atoms(mol: Molecule, source_atoms: Iterable[int], out: Molecule, offset_x: float, repeat_atom: bool) -> dict[int, int]:
    mapping = {}
    for source_idx in sorted(source_atoms):
        source = mol.atom(source_idx)
        new_idx = len(out.atoms) + 1
        out.atoms.append(
            Atom(
                index=new_idx,
                element=source.element,
                x=source.x + offset_x,
                y=source.y,
                z=source.z,
                repeat_atom=repeat_atom,
                formal_charge=source.formal_charge,
            )
        )
        mapping[source_idx] = new_idx
    return mapping


def copy_bonds_between(mol: Molecule, out: Molecule, mapping: dict[int, int]) -> None:
    source_set = set(mapping)
    for bond in mol.bonds:
        if bond.a in source_set and bond.b in source_set:
            out.bonds.append(Bond(mapping[bond.a], mapping[bond.b], bond.order))


def monomer_spacing(monomer: Molecule, previous_atom: int, next_atom: int) -> float:
    previous = monomer.atom(previous_atom)
    next_ = monomer.atom(next_atom)
    connection_distance = math.hypot(next_.x - previous.x, next_.y - previous.y)
    min_x = min((atom.x for atom in monomer.atoms), default=0.0)
    max_x = max((atom.x for atom in monomer.atoms), default=0.0)
    width = max_x - min_x
    return max(3.0, width + connection_distance + 1.5)


def build_reference_oligomer_from_monomer(
    monomer: Molecule,
    previous_atom: int,
    next_atom: int,
    reference_dp: int,
    junction_order: int = 1,
) -> tuple[Molecule, list[int]]:
    if reference_dp < 2:
        raise ValueError("Reference oligomer size must be at least 2 monomers.")
    if previous_atom == next_atom:
        raise ValueError("Previous and next connection atoms must be different.")
    if previous_atom < 1 or previous_atom > len(monomer.atoms):
        raise ValueError("Previous connection atom index is outside the monomer.")
    if next_atom < 1 or next_atom > len(monomer.atoms):
        raise ValueError("Next connection atom index is outside the monomer.")

    oligomer = Molecule(name=f"{monomer.name}_reference_{reference_dp}mer")
    spacing = monomer_spacing(monomer, previous_atom, next_atom)
    mappings: list[dict[int, int]] = []
    for unit_idx in range(reference_dp):
        mapping = copy_source_atoms(monomer, [atom.index for atom in monomer.atoms], oligomer, unit_idx * spacing, True)
        copy_bonds_between(monomer, oligomer, mapping)
        mappings.append(mapping)
        if unit_idx > 0:
            oligomer.bonds.append(Bond(mappings[unit_idx - 1][next_atom], mapping[previous_atom], junction_order))
    repeat_atoms = [mappings[0][atom.index] for atom in monomer.atoms]
    return oligomer, repeat_atoms


def build_polymer_graph(mol: Molecule, plan: RepeatPlan, dp: int) -> Molecule:
    if dp < 1:
        raise ValueError("DP must be at least 1.")
    if len(plan.occurrences) > 1 and (plan.junction_left_pos is None or plan.junction_right_pos is None):
        raise ValueError("Could not infer the inter-repeat junction bond.")

    repeat_union = set().union(*(occurrence.atom_set for occurrence in plan.occurrences))
    head_atoms = collect_external_fragment(mol, plan.occurrences[0], repeat_union)
    tail_atoms = collect_external_fragment(mol, plan.occurrences[-1], repeat_union)
    tail_atoms -= head_atoms

    template = plan.occurrences[0]
    out = Molecule(name=f"{mol.name}_DP{dp}")
    repeat_mappings: list[dict[int, int]] = []
    if len(plan.occurrences) > 1:
        dx = mol.atom(plan.occurrences[1].atoms[0]).x - mol.atom(plan.occurrences[0].atoms[0]).x
        dy = mol.atom(plan.occurrences[1].atoms[0]).y - mol.atom(plan.occurrences[0].atoms[0]).y
        spacing = math.hypot(dx, dy) or 3.0
    else:
        spacing = max(3.0, len(template.atoms) * 1.4)

    head_mapping = copy_source_atoms(mol, head_atoms, out, 0.0, repeat_atom=False)
    copy_bonds_between(mol, out, head_mapping)

    template_set = template.atom_set
    for unit_idx in range(dp):
        offset = unit_idx * spacing
        mapping = copy_source_atoms(mol, template.atoms, out, offset, repeat_atom=True)
        copy_bonds_between(mol, out, mapping)
        repeat_mappings.append(mapping)
        if unit_idx > 0 and plan.junction_left_pos is not None and plan.junction_right_pos is not None:
            prev_source = template.atoms[plan.junction_left_pos]
            curr_source = template.atoms[plan.junction_right_pos]
            out.bonds.append(Bond(repeat_mappings[unit_idx - 1][prev_source], mapping[curr_source], plan.junction_order))

    tail_offset = (dp - 1) * spacing
    tail_mapping = copy_source_atoms(mol, tail_atoms, out, tail_offset, repeat_atom=False)
    copy_bonds_between(mol, out, tail_mapping)

    first_repeat_mapping = repeat_mappings[0]
    for bond in mol.bonds:
        if bond.a in head_mapping and bond.b in template_set:
            out.bonds.append(Bond(head_mapping[bond.a], first_repeat_mapping[bond.b], bond.order))
        elif bond.b in head_mapping and bond.a in template_set:
            out.bonds.append(Bond(head_mapping[bond.b], first_repeat_mapping[bond.a], bond.order))

    last_repeat_mapping = repeat_mappings[-1]
    last_occurrence = plan.occurrences[-1]
    for bond in mol.bonds:
        if bond.a in tail_mapping and bond.b in last_occurrence.atom_set:
            pos = last_occurrence.atoms.index(bond.b)
            out.bonds.append(Bond(tail_mapping[bond.a], last_repeat_mapping[template.atoms[pos]], bond.order))
        elif bond.b in tail_mapping and bond.a in last_occurrence.atom_set:
            pos = last_occurrence.atoms.index(bond.a)
            out.bonds.append(Bond(tail_mapping[bond.b], last_repeat_mapping[template.atoms[pos]], bond.order))

    add_valence_hydrogens(out)
    return out


def maximum_valence(atom: Atom) -> int | None:
    """Return the supported bond-order valence for an element/formal-charge pair."""

    return FORMAL_VALENCE.get(
        (atom.element, atom.formal_charge),
        DEFAULT_VALENCE.get(atom.element),
    )


def add_valence_hydrogens(mol: Molecule) -> None:
    degree = {atom.index: 0 for atom in mol.atoms}
    for bond in mol.bonds:
        degree[bond.a] += bond.order
        degree[bond.b] += bond.order
    next_idx = len(mol.atoms) + 1
    new_bonds: list[Bond] = []
    for atom in list(mol.atoms):
        if atom.element == "H":
            continue
        valence = maximum_valence(atom)
        if valence is None:
            continue
        bond_order_sum = degree[atom.index]
        needed = max(0, valence - bond_order_sum)
        for h_num in range(needed):
            angle = (2 * math.pi * h_num / max(1, needed)) + 0.7
            mol.atoms.append(
                Atom(
                    index=next_idx,
                    element="H",
                    x=atom.x + 0.85 * math.cos(angle),
                    y=atom.y + 0.85 * math.sin(angle),
                    z=0.15 * (h_num % 2),
                    repeat_atom=atom.repeat_atom,
                )
            )
            new_bonds.append(Bond(atom.index, next_idx, 1))
            next_idx += 1
    mol.bonds.extend(new_bonds)


def assign_placeholder_gaff(mol: Molecule) -> None:
    for atom in mol.atoms:
        atom.atom_type = {
            "C": "c3",
            "H": "h1",
            "O": "oh" if sum(1 for b in mol.bonds if atom.index in (b.a, b.b)) == 2 else "os",
            "N": "n4" if atom.formal_charge > 0 else "n3",
        }.get(atom.element, atom.element.lower())
        atom.charge = 0.0


def normalize_repeat_charges(mol: Molecule, target_charge: int) -> tuple[float, float, float]:
    raw = sum(atom.charge for atom in mol.atoms)
    eligible = [atom for atom in mol.atoms if atom.repeat_atom]
    if not eligible:
        raise ValueError("No repeat atoms are available for charge normalization.")
    correction = (target_charge - raw) / len(eligible)
    for atom in eligible:
        atom.charge += correction

    # MOL2 and placeholder ITP charges are written with eight decimal places.
    # Quantize here, then put the rounding residual on one repeat atom so the
    # serialized atomic charges still sum to the requested integer charge.
    for atom in mol.atoms:
        atom.charge = round(atom.charge, 8)
    rounding_residual = target_charge - sum(atom.charge for atom in mol.atoms)
    eligible[-1].charge = round(eligible[-1].charge + rounding_residual, 8)
    final = sum(atom.charge for atom in mol.atoms)
    return raw, final, correction


def write_gromacs_placeholders(mol: Molecule, outdir: Path, target_charge: int) -> tuple[list[Path], float, float, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    assign_placeholder_gaff(mol)
    raw, final, correction = normalize_repeat_charges(mol, target_charge)
    files: list[Path] = []

    gro = outdir / "polymer.gro"
    with gro.open("w") as handle:
        handle.write(f"{mol.name}\n{len(mol.atoms):5d}\n")
        for atom in mol.atoms:
            handle.write(
                f"{1:5d}{'POL':<5}{atom.element + str(atom.index):>5}{atom.index:5d}"
                f"{atom.x / 10:8.3f}{atom.y / 10:8.3f}{atom.z / 10:8.3f}\n"
            )
        handle.write("   5.00000   5.00000   5.00000\n")
    files.append(gro)

    itp = outdir / "polymer.itp"
    with itp.open("w") as handle:
        handle.write("[ moleculetype ]\n; name nrexcl\nPOL 3\n\n[ atoms ]\n")
        handle.write("; nr type resnr residue atom cgnr charge mass\n")
        for atom in mol.atoms:
            mass = ATOMIC_MASS.get(atom.element, 0.0)
            handle.write(
                f"{atom.index:5d} {atom.atom_type:<6} 1 POL {atom.element + str(atom.index):<6} "
                f"{atom.index:5d} {atom.charge: .8f} {mass: .4f}\n"
            )
        handle.write("\n[ bonds ]\n; ai aj funct\n")
        for bond in mol.bonds:
            handle.write(f"{bond.a:5d} {bond.b:5d} 1\n")
    files.append(itp)

    top = outdir / "polymer.top"
    top.write_text('#include "polymer.itp"\n\n[ system ]\nPolymer\n\n[ molecules ]\nPOL 1\n')
    files.append(top)

    report = outdir / "BUILD_REPORT.txt"
    report.write_text(
        "\n".join(
            [
                f"name: {mol.name}",
                f"formula: {mol.formula()}",
                f"atoms: {len(mol.atoms)}",
                f"bonds: {len(mol.bonds)}",
                f"target_charge: {target_charge}",
                f"raw_charge: {raw:.12g}",
                f"final_charge: {final:.12g}",
                f"correction_per_repeat_atom: {correction:.12g}",
                "",
                "These are standard-library placeholder topology files for GUI/backend testing.",
                "Install AmberTools and ParmEd for real GAFF2/AM1-BCC GROMACS topology generation.",
            ]
        )
        + "\n"
    )
    files.append(report)
    return files, raw, final, correction


def split_gromacs_topology(top_path: Path, itp_path: Path) -> None:
    """Split a ParmEd GROMACS topology into an includable ITP and wrapper TOP."""

    lines = top_path.read_text().splitlines()
    system_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"\s*\[\s*system\s*\]\s*", line, flags=re.IGNORECASE)
        ),
        None,
    )
    if system_index is None:
        raise ValueError(f"ParmEd topology does not contain a [ system ] section: {top_path}")
    molecular_lines = lines[:system_index]
    system_lines = lines[system_index:]
    if not any(
        re.fullmatch(r"\s*\[\s*moleculetype\s*\]\s*", line, flags=re.IGNORECASE)
        for line in molecular_lines
    ):
        raise ValueError(f"ParmEd topology does not contain a [ moleculetype ] section: {top_path}")
    itp_path.write_text("\n".join(molecular_lines).rstrip() + "\n")
    top_path.write_text(
        f'#include "{itp_path.name}"\n\n' + "\n".join(system_lines).rstrip() + "\n"
    )


def missing_external_tools() -> list[str]:
    return [tool for tool in ["antechamber", "parmchk2", "tleap"] if shutil.which(tool) is None]


def minimize_sdf_geometry(input_sdf: Path, output_sdf: Path, max_iters: int = 1000) -> tuple[Path, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("RDKit is required for geometry minimization before Antechamber.") from exc

    supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=True)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise RuntimeError(f"RDKit could not read SDF for minimization: {input_sdf}")

    needs_embed = mol.GetNumConformers() == 0
    if not needs_embed:
        conf = mol.GetConformer()
        z_values = [abs(conf.GetAtomPosition(idx).z) for idx in range(mol.GetNumAtoms())]
        needs_embed = max(z_values, default=0.0) < 1e-4

    if needs_embed:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0DEF
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise RuntimeError("RDKit ETKDG embedding failed before minimization.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        method = "MMFF94s"
        try:
            status = AllChem.MMFFOptimizeMolecule(mol, mmffVariant=method, maxIters=max_iters)
        except (TypeError, ValueError):
            method = "MMFF94"
            status = AllChem.MMFFOptimizeMolecule(mol, mmffVariant=method, maxIters=max_iters)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        status = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        method = "UFF"
    else:
        raise RuntimeError("RDKit has neither MMFF nor UFF parameters for this molecule.")

    if status < 0:
        raise RuntimeError(f"RDKit {method} minimization failed.")

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()
    return output_sdf, method


def ensure_explicit_hydrogen_sdf(input_sdf: Path, output_sdf: Path) -> Path:
    mol = read_sdf(input_sdf)
    hydrogenated = clone_molecule(mol, name=f"{mol.name}_H")
    add_valence_hydrogens(hydrogenated)
    write_sdf(hydrogenated, output_sdf)
    return output_sdf


def run_ambertools_pipeline(
    reference_sdf: Path,
    outdir: Path,
    formal_charge: int,
    minimize_geometry: bool = True,
    target_spec: AmberTargetSpec | None = None,
) -> tuple[list[Path], Path]:
    missing = missing_external_tools()
    if missing:
        raise RuntimeError("Required AmberTools executable(s) not found on PATH: " + ", ".join(missing))
    try:
        import parmed as pmd
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("ParmEd is required for AMBER to GROMACS conversion.") from exc

    outdir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    hydrogenated_sdf = ensure_explicit_hydrogen_sdf(reference_sdf, outdir / "antechamber_input_with_h.sdf")
    files.append(hydrogenated_sdf)
    antechamber_input = hydrogenated_sdf
    if minimize_geometry:
        minimized_sdf, minimizer = minimize_sdf_geometry(hydrogenated_sdf, outdir / "polymer_minimized.sdf")
        antechamber_input = minimized_sdf
        files.append(minimized_sdf)
        (outdir / "MINIMIZATION_REPORT.txt").write_text(
            "\n".join(
                [
                    f"input_sdf: {hydrogenated_sdf}",
                    f"output_sdf: {minimized_sdf}",
                    f"method: RDKit {minimizer}",
                    "stage: explicit-H geometry before Antechamber GAFF2/AM1-BCC parameterization",
                ]
            )
            + "\n"
        )
        files.append(outdir / "MINIMIZATION_REPORT.txt")

    antechamber_mol = read_sdf(antechamber_input)
    antechamber_mol2 = outdir / "antechamber_input.mol2"
    write_mol2(antechamber_mol, antechamber_mol2)
    files.append(antechamber_mol2)

    reference_mol2 = outdir / (
        "reference_typed_charged.mol2" if target_spec is not None else "polymer_typed_charged.mol2"
    )
    frcmod = outdir / "polymer.frcmod"
    prmtop = outdir / "polymer.prmtop"
    inpcrd = outdir / "polymer.inpcrd"
    leap_in = outdir / "tleap.in"
    subprocess.run(
        [
            "antechamber",
            "-i",
            str(antechamber_mol2),
            "-fi",
            "mol2",
            "-o",
            str(reference_mol2),
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-s",
            "2",
            "-at",
            "gaff2",
            "-nc",
            str(formal_charge),
        ],
        check=True,
    )
    subprocess.run(
        ["parmchk2", "-i", str(reference_mol2), "-f", "mol2", "-o", str(frcmod)],
        check=True,
    )

    final_mol2 = reference_mol2
    if target_spec is not None:
        parameters = read_mol2_atom_parameters(reference_mol2)
        mapping_values = (
            target_spec.unit_atom_count,
            target_spec.reference_dp,
            target_spec.target_dp,
        )
        if all(value is None for value in mapping_values):
            copy_typed_parameters_by_index(antechamber_mol, parameters, target_spec.molecule)
        elif all(value is not None for value in mapping_values):
            transfer_reference_atom_parameters(
                antechamber_mol,
                parameters,
                target_spec.molecule,
                unit_atom_count=target_spec.unit_atom_count,
                reference_dp=target_spec.reference_dp,
                target_dp=target_spec.target_dp,
            )
        else:
            raise ValueError(
                "unit_atom_count, reference_dp, and target_dp must be supplied together "
                "for reference-to-target parameter transfer."
            )
        raw, final, correction = normalize_repeat_charges(
            target_spec.molecule,
            target_spec.target_charge,
        )
        target_spec.raw_charge = raw
        target_spec.final_charge = final
        target_spec.correction_per_repeat_atom = correction
        final_mol2 = outdir / "polymer_typed_charged.mol2"
        write_mol2(
            target_spec.molecule,
            final_mol2,
            residue_name="POL",
            molecule_name="POLYMER",
        )

    leap_in.write_text(
        "\n".join(
            [
                "source leaprc.gaff2",
                f"loadamberparams {frcmod.name}",
                f"mol = loadmol2 {final_mol2.name}",
                f"saveamberparm mol {prmtop.name} {inpcrd.name}",
                "quit",
            ]
        )
        + "\n"
    )
    subprocess.run(["tleap", "-f", str(leap_in.name)], cwd=outdir, check=True)
    structure = pmd.load_file(str(prmtop), str(inpcrd))
    if target_spec is not None and len(structure.atoms) != len(target_spec.molecule.atoms):
        raise RuntimeError(
            "TLeap target atom count mismatch: "
            f"expected {len(target_spec.molecule.atoms)}, found {len(structure.atoms)}."
        )
    top_path = outdir / "polymer.top"
    gro_path = outdir / "polymer.gro"
    structure.save(str(top_path), overwrite=True)
    structure.save(str(gro_path), overwrite=True)
    itp_path = outdir / "polymer.itp"
    if target_spec is not None:
        split_gromacs_topology(top_path, itp_path)
    generated = [reference_mol2, frcmod]
    if final_mol2 != reference_mol2:
        generated.append(final_mol2)
    generated.extend([leap_in, prmtop, inpcrd, top_path, gro_path])
    if target_spec is not None:
        generated.append(itp_path)
    return files + generated, antechamber_mol2


def write_outputs_for_polymer(
    polymer: Molecule,
    repeats: list[list[int]],
    source_input: Path,
    repeat_atoms: list[int],
    options: BuildOptions,
    extra_metadata: dict | None = None,
    external_sdf: Path | None = None,
    external_formal_charge: int | None = None,
    reference_unit_atom_count: int | None = None,
    reference_dp: int | None = None,
) -> BuildResult:
    target_charge = options.dp * options.repeat_charge + options.end_charge
    options.outdir.mkdir(parents=True, exist_ok=True)
    polymer_sdf = options.outdir / "polymer.sdf"
    write_sdf(polymer, polymer_sdf)
    files = [polymer_sdf]

    missing = missing_external_tools()
    requested_antechamber_sdf = external_sdf or polymer_sdf
    actual_antechamber_input = requested_antechamber_sdf
    antechamber_charge = target_charge if external_formal_charge is None else external_formal_charge
    parameterization_mode = "placeholder"
    if options.run_external and not missing:
        if (reference_unit_atom_count is None) != (reference_dp is None):
            raise ValueError("reference_unit_atom_count and reference_dp must be supplied together.")
        parameterization_mode = (
            "reference_to_target" if reference_dp is not None else "direct_target"
        )
        target_spec = AmberTargetSpec(
            molecule=polymer,
            target_charge=target_charge,
            unit_atom_count=reference_unit_atom_count,
            reference_dp=reference_dp,
            target_dp=options.dp if reference_dp is not None else None,
        )
        external_files, actual_antechamber_input = run_ambertools_pipeline(
            requested_antechamber_sdf,
            options.outdir,
            antechamber_charge,
            minimize_geometry=options.minimize_geometry,
            target_spec=target_spec,
        )
        files.extend(external_files)
        raw = target_spec.raw_charge
        final = target_spec.final_charge
        correction = target_spec.correction_per_repeat_atom
    else:
        placeholder_files, raw, final, correction = write_gromacs_placeholders(polymer, options.outdir, target_charge)
        files.extend(placeholder_files)

    metadata = {
        "input": str(source_input),
        "repeat_atoms": repeat_atoms,
        "detected_repeat_units": repeats,
        "dp": options.dp,
        "formula": polymer.formula(),
        "atoms": len(polymer.atoms),
        "bonds": len(polymer.bonds),
        "target_charge": target_charge,
        "raw_charge": raw,
        "charge_correction_per_repeat_atom": correction,
        "parameterization_mode": parameterization_mode,
        "antechamber_source_sdf": str(requested_antechamber_sdf),
        "antechamber_input": str(actual_antechamber_input),
        "antechamber_input_format": actual_antechamber_input.suffix.lstrip(".").lower() or "unknown",
        "antechamber_input_sdf": str(requested_antechamber_sdf),
        "antechamber_formal_charge": antechamber_charge,
        "final_charge": final,
        "minimize_geometry": options.minimize_geometry,
        "missing_tools": missing,
        "files": [str(path) for path in files],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path = options.outdir / "build_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    files.append(metadata_path)

    return BuildResult(
        molecule=polymer,
        repeat_units=repeats,
        target_charge=target_charge,
        raw_charge=raw,
        final_charge=final,
        correction_per_repeat_atom=correction,
        files=files,
        missing_tools=missing,
    )


def build_polymer_from_oligomer(input_sdf: Path, options: BuildOptions) -> BuildResult:
    source = read_sdf(input_sdf)
    plan = make_repeat_plan(source, options.repeat_atoms)
    repeats = [occurrence.atoms for occurrence in plan.occurrences]
    polymer = build_polymer_graph(source, plan, options.dp)
    return write_outputs_for_polymer(polymer, repeats, input_sdf, options.repeat_atoms, options)


def build_polymer_from_monomer(input_sdf: Path, options: MonomerBuildOptions) -> BuildResult:
    monomer = read_sdf(input_sdf)
    monomer_formal_charge = monomer.total_formal_charge()
    options.outdir.mkdir(parents=True, exist_ok=True)
    reference_oligomer, repeat_atoms = build_reference_oligomer_from_monomer(
        monomer,
        previous_atom=options.previous_atom,
        next_atom=options.next_atom,
        reference_dp=options.reference_dp,
        junction_order=options.junction_order,
    )
    reference_formal_charge = reference_oligomer.total_formal_charge()
    reference_sdf = options.outdir / "reference_oligomer_from_monomer.sdf"
    write_sdf(reference_oligomer, reference_sdf)
    reference_for_antechamber = clone_molecule(reference_oligomer, name=f"{reference_oligomer.name}_H")
    add_valence_hydrogens(reference_for_antechamber)
    reference_antechamber_sdf = options.outdir / "reference_oligomer_for_antechamber.sdf"
    write_sdf(reference_for_antechamber, reference_antechamber_sdf)
    monomer_atom_order = [atom.index for atom in monomer.atoms]
    occurrence_atoms = [
        [unit_idx * len(monomer_atom_order) + atom_idx for atom_idx in range(1, len(monomer_atom_order) + 1)]
        for unit_idx in range(options.reference_dp)
    ]
    previous_pos = monomer_atom_order.index(options.previous_atom)
    next_pos = monomer_atom_order.index(options.next_atom)
    plan = RepeatPlan(
        query_atoms=repeat_atoms,
        occurrences=[RepeatOccurrence(atoms=atoms) for atoms in occurrence_atoms],
        junction_left_pos=next_pos,
        junction_right_pos=previous_pos,
        junction_order=options.junction_order,
    )
    polymer = build_polymer_graph(reference_oligomer, plan, options.polymer_dp)
    result = write_outputs_for_polymer(
        polymer,
        occurrence_atoms,
        reference_sdf,
        repeat_atoms,
        BuildOptions(
            repeat_atoms=repeat_atoms,
            dp=options.polymer_dp,
            repeat_charge=monomer_formal_charge,
            outdir=options.outdir,
            run_external=options.run_external,
            minimize_geometry=options.minimize_geometry,
        ),
        extra_metadata={
            "workflow": "monomer",
            "monomer_input": str(input_sdf),
            "previous_connection_atom": options.previous_atom,
            "next_connection_atom": options.next_atom,
            "reference_oligomer_dp": options.reference_dp,
            "reference_oligomer_sdf": str(reference_sdf),
            "reference_oligomer_antechamber_sdf": str(reference_antechamber_sdf),
            "reference_repeat_atoms": repeat_atoms,
            "monomer_atom_formal_charge": monomer_formal_charge,
            "reference_oligomer_formal_charge": reference_formal_charge,
            "target_polymer_formal_charge": monomer_formal_charge * options.polymer_dp,
            "formal_charge_inferred_from_atoms": True,
        },
        external_sdf=reference_antechamber_sdf,
        external_formal_charge=reference_formal_charge,
        reference_unit_atom_count=len(monomer_atom_order),
        reference_dp=options.reference_dp,
    )
    for path in [reference_sdf, reference_antechamber_sdf]:
        if path not in result.files:
            result.files.insert(0, path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_sdf", type=Path)
    parser.add_argument("--repeat-atoms", help="Oligomer mode: comma-separated 1-based repeat atom indices, e.g. 2,3,4")
    parser.add_argument("--previous-atom", type=int, help="Monomer mode: atom that connects to the previous monomer.")
    parser.add_argument("--next-atom", type=int, help="Monomer mode: atom that connects to the next monomer.")
    parser.add_argument("--reference-dp", type=int, default=5, help="Monomer mode: hidden reference oligomer size.")
    parser.add_argument("--junction-order", type=int, default=1, help="Monomer mode: monomer-monomer bond order.")
    parser.add_argument("--dp", type=int, required=True, help="Degree of polymerization")
    parser.add_argument("--oligomer-charge", type=int, default=0, help="Legacy oligomer mode only.")
    parser.add_argument("--repeat-charge", type=int, default=0, help="Legacy oligomer mode only.")
    parser.add_argument("--end-charge", type=int, default=0, help="Legacy oligomer mode only.")
    parser.add_argument("--outdir", type=Path, default=Path("polymer_out"))
    parser.add_argument(
        "--no-minimize-geometry",
        action="store_true",
        help="Skip RDKit MMFF/UFF geometry minimization before Antechamber.",
    )
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Skip AmberTools orchestration and write placeholder files for graph/GUI testing.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    monomer_mode = args.previous_atom is not None or args.next_atom is not None
    if monomer_mode:
        if args.previous_atom is None or args.next_atom is None:
            raise SystemExit("--previous-atom and --next-atom must be provided together for monomer mode.")
        result = build_polymer_from_monomer(
            args.input_sdf,
            MonomerBuildOptions(
                previous_atom=args.previous_atom,
                next_atom=args.next_atom,
                reference_dp=args.reference_dp,
                polymer_dp=args.dp,
                outdir=args.outdir,
                run_external=not args.no_external,
                minimize_geometry=not args.no_minimize_geometry,
                junction_order=args.junction_order,
            ),
        )
    else:
        if not args.repeat_atoms:
            raise SystemExit("--repeat-atoms is required for oligomer mode. For monomer mode, use --previous-atom and --next-atom.")
        result = build_polymer_from_oligomer(
            args.input_sdf,
            BuildOptions(
                repeat_atoms=parse_index_list(args.repeat_atoms),
                dp=args.dp,
                oligomer_charge=args.oligomer_charge,
                repeat_charge=args.repeat_charge,
                end_charge=args.end_charge,
                outdir=args.outdir,
                run_external=not args.no_external,
                minimize_geometry=not args.no_minimize_geometry,
            ),
        )
    print(f"Detected repeat units: {result.repeat_units}")
    print(f"Polymer formula: {result.molecule.formula()}")
    print(f"Atoms: {len(result.molecule.atoms)}")
    print(f"Bonds: {len(result.molecule.bonds)}")
    print(f"Final charge: {result.final_charge:.12g}")
    if result.missing_tools:
        print("AmberTools blocked: missing " + ", ".join(result.missing_tools))
        print("Wrote placeholder GROMACS-shaped files for non-external validation.")
    print(f"Output directory: {result.files[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
