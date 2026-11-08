"""Engine-neutral transfer of a reference-oligomer GROMACS topology to a polymer.

Every parameterization engine (OPLS-AA/2020 via opls-ff-lite, OpenFF Sage via
Interchange, GAFF2 via AmberTools/ParmEd) can write a GROMACS topology of the
small reference oligomer. This module reads that topology and builds the
topology of the long target polymer from it, so that all engines share one
algorithm:

1. ``parse_gromacs_topology`` reads the reference ``[ atomtypes ]``,
   ``[ atoms ]`` and every bonded line. Parameters are kept as the engine's
   exact text, so nothing is re-rounded.
2. Atoms are addressed as (repeat unit, index within the unit). The builder
   adds atoms unit by unit, so a monomer atom's unit and position follow
   from its index; hydrogens (appended after all monomer atoms) are placed in
   their parent's unit, ordered by parent atom and slot.
3. ``transfer_topology`` copies the reference terms by **index shift**
   (``transfer_by_index_shift``): a reference term occupying units r..r+s is
   copied to target units p..p+s with every atom keeping its index within its
   unit. Which reference window a target term comes from is a fixed rule
   (``shift_source_unit``): terms within the first or last E units of the
   target (E = ``end_unit_count``, 2 for a 5-unit reference) are copied from
   the same position relative to the reference head or tail; all other terms
   from the most central reference window. Atom types follow the same rule.
   The result is accepted only if an independent cross-check,
   ``transfer_by_environment_match``, gives exactly the same atom types and
   term lines. That check places every reference term pattern at every
   target position and picks the reference copy whose bonded neighbourhood
   matches the target to the most bond shells. Any difference raises
   ``TransferDisagreement``. Coverage of the graph-defined terms (bonds,
   angles, 1-4 pairs, proper dihedrals) is verified as well.
4. ``write_gromacs_files`` writes ``.itp``/``.top`` with atom types merged
   by their complete non-bonded parameter set. Every bonded term is written
   with explicit parameters, so a type name only carries mass, atomic number,
   sigma and epsilon, and two atoms share a name only if all of these are
   identical.

Charges are not chosen here: the caller averages them over the reference
atoms of the same role (head, interior or tail unit) and renormalizes them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


SECTION_ATOM_COUNT = {"bonds": 2, "pairs": 2, "angles": 3, "dihedrals": 4}
IMPROPER_FUNCTS = frozenset({2, 4})
PTYPES = frozenset({"A", "S", "V", "D"})
KIND_ORDER = ("bonds", "pairs", "angles", "propers", "impropers")
KIND_SECTION = {
    "bonds": "bonds",
    "pairs": "pairs",
    "angles": "angles",
    "propers": "dihedrals",
    "impropers": "dihedrals",
}
DEPTH_MAX = 4
DEPTH_MIN = 2

AtomLabel = tuple[int, tuple]


# ---------------------------------------------------------------------------
# Topology model and parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomType:
    name: str
    bond_type: str
    atomic_number: int | None
    mass: str
    charge: str
    ptype: str
    sigma: str
    epsilon: str

    def nonbonded_key(self) -> tuple:
        return (
            self.atomic_number,
            float(self.mass),
            self.ptype,
            float(self.sigma),
            float(self.epsilon),
        )


@dataclass
class TopologyAtom:
    index: int
    type_name: str
    charge: float
    mass: str


@dataclass(frozen=True)
class Term:
    section: str
    atoms: tuple[int, ...]
    funct: int
    params: tuple[str, ...]
    comment: str = ""

    @property
    def kind(self) -> str:
        return term_kind(self.section, self.funct)


@dataclass
class GromacsTopology:
    path: Path
    defaults: list[str] | None
    atomtypes: dict[str, AtomType]
    molecule_name: str
    nrexcl: int
    atoms: list[TopologyAtom]
    terms: list[Term]
    # Readable per-atom type label (e.g. the OPLS type "opls_182"); used to
    # name merged atom types. Defaults to the atom type's bond_type.
    labels: list[str] | None = None

    def atom_type(self, index: int) -> AtomType:
        return self.atomtypes[self.atoms[index - 1].type_name]

    def label(self, index: int) -> str:
        if self.labels is not None:
            return self.labels[index - 1]
        atom_type = self.atom_type(index)
        return atom_type.bond_type or atom_type.name


def term_kind(section: str, funct: int) -> str:
    if section == "dihedrals":
        return "impropers" if funct in IMPROPER_FUNCTS else "propers"
    return section


def canonical_atoms(kind: str, atoms: Sequence[int]) -> tuple[int, ...]:
    """Atom tuple that identifies a term regardless of its written direction.

    Improper dihedrals are order-sensitive and kept as written.
    """

    atoms = tuple(atoms)
    if kind == "impropers":
        return atoms
    return min(atoms, atoms[::-1])


def _expand_includes(path: Path, seen: set[Path] | None = None) -> Iterable[tuple[Path, str]]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Recursive #include of {path}")
    seen = seen | {path}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#include"):
            match = re.match(r'#include\s+"([^"]+)"', stripped)
            if match is None:
                raise ValueError(f"Unsupported #include in {path}: {line}")
            included = path.parent / match.group(1)
            if not included.is_file():
                raise ValueError(
                    f"{path} includes {match.group(1)}, which is not a local file; "
                    "the reference topology must be self-contained."
                )
            yield from _expand_includes(included, seen)
            continue
        yield path, line


def _parse_atomtype(fields: list[str], where: str) -> AtomType:
    ptype_index = len(fields) - 3
    if ptype_index < 3 or fields[ptype_index] not in PTYPES:
        raise ValueError(f"Unrecognized [ atomtypes ] line {where}: {' '.join(fields)}")
    name = fields[0]
    middle = fields[1 : ptype_index - 2]
    bond_type = name
    atomic_number: int | None = None
    if len(middle) == 2:
        bond_type, atomic_number = middle[0], int(middle[1])
    elif len(middle) == 1:
        if re.fullmatch(r"\d+", middle[0]):
            atomic_number = int(middle[0])
        else:
            bond_type = middle[0]
    elif middle:
        raise ValueError(f"Unrecognized [ atomtypes ] line {where}: {' '.join(fields)}")
    mass, charge = fields[ptype_index - 2], fields[ptype_index - 1]
    sigma, epsilon = fields[ptype_index + 1], fields[ptype_index + 2]
    return AtomType(name, bond_type, atomic_number, mass, charge, fields[ptype_index], sigma, epsilon)


def parse_gromacs_topology(path: Path) -> GromacsTopology:
    """Read a self-contained, single-molecule GROMACS topology (.top or .itp)."""

    path = Path(path)
    section: str | None = None
    defaults: list[str] | None = None
    atomtypes: dict[str, AtomType] = {}
    molecule_name: str | None = None
    nrexcl: int | None = None
    atoms: list[TopologyAtom] = []
    terms: list[Term] = []

    for source, raw in _expand_includes(path):
        body, _, comment = raw.partition(";")
        body = body.strip()
        comment = comment.strip()
        if not body:
            continue
        where = f"in {source}"
        if body.startswith("#"):
            raise ValueError(f"Unsupported preprocessor directive {where}: {raw.strip()}")
        header = re.fullmatch(r"\[\s*([A-Za-z_]+)\s*\]", body)
        if header:
            section = header.group(1).lower()
            if section == "moleculetype" and molecule_name is not None:
                raise ValueError(f"{path} defines more than one [ moleculetype ].")
            continue
        fields = body.split()
        if section == "defaults":
            defaults = fields
        elif section == "atomtypes":
            atom_type = _parse_atomtype(fields, where)
            atomtypes[atom_type.name] = atom_type
        elif section == "moleculetype":
            molecule_name, nrexcl = fields[0], int(fields[1])
        elif section == "atoms":
            if len(fields) < 7:
                raise ValueError(f"Malformed [ atoms ] line {where}: {raw.strip()}")
            index = int(fields[0])
            if index != len(atoms) + 1:
                raise ValueError(f"[ atoms ] must be numbered 1..N in order {where}: {raw.strip()}")
            type_name = fields[1]
            if type_name not in atomtypes:
                raise ValueError(f"Atom {index} uses undefined atom type {type_name} {where}.")
            mass = fields[7] if len(fields) >= 8 else atomtypes[type_name].mass
            atoms.append(TopologyAtom(index, type_name, float(fields[6]), mass))
        elif section in SECTION_ATOM_COUNT:
            count = SECTION_ATOM_COUNT[section]
            atom_indices = tuple(int(value) for value in fields[:count])
            funct = int(fields[count])
            params = tuple(fields[count + 1 :])
            if not params and section != "pairs":
                raise ValueError(
                    f"[ {section} ] line without explicit parameters {where}: {raw.strip()}"
                )
            terms.append(Term(section, atom_indices, funct, params, comment))
        elif section in {"system", "molecules"}:
            continue
        else:
            raise ValueError(f"Unsupported section [ {section} ] {where}.")

    if molecule_name is None or nrexcl is None:
        raise ValueError(f"{path} has no [ moleculetype ].")
    if not atoms:
        raise ValueError(f"{path} has no [ atoms ].")
    for term in terms:
        if min(term.atoms) < 1 or max(term.atoms) > len(atoms):
            raise ValueError(f"Term {term.atoms} in {path} refers to a missing atom.")
    return GromacsTopology(path, defaults, atomtypes, molecule_name, nrexcl, atoms, terms)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def adjacency(atom_count: int, bonds: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {index: set() for index in range(1, atom_count + 1)}
    for a, b in bonds:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def expected_terms(kind: str, graph: dict[int, set[int]]) -> set[tuple[int, ...]] | None:
    """Terms of a kind defined by the bond graph alone (None for impropers)."""

    if kind == "bonds":
        return {canonical_atoms(kind, (a, b)) for a in graph for b in graph[a] if a < b}
    if kind == "angles":
        return {
            canonical_atoms(kind, (a, center, b))
            for center in graph
            for a in graph[center]
            for b in graph[center]
            if a < b
        }
    if kind == "propers":
        # Every path a-b-c-d except those closing a three-membered ring
        # (as opls-ff-lite, BOSS and LigParGen do).
        paths = set()
        for b in graph:
            for c in graph[b]:
                for a in graph[b] - {c}:
                    if a in graph[c]:
                        continue
                    for d in graph[c] - {b}:
                        if d == a or d in graph[b]:
                            continue
                        paths.add(canonical_atoms(kind, (a, b, c, d)))
        return paths
    if kind == "pairs":
        pairs = set()
        for start in graph:
            distance = {start: 0}
            frontier = [start]
            for depth in range(1, 4):
                next_frontier = []
                for atom in frontier:
                    for neighbor in graph[atom]:
                        if neighbor not in distance:
                            distance[neighbor] = depth
                            next_frontier.append(neighbor)
                frontier = next_frontier
            pairs.update(
                canonical_atoms(kind, (start, other))
                for other, value in distance.items()
                if value == 3 and start < other
            )
        return pairs
    return None


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


@dataclass
class TransferredAtom:
    index: int
    atom_type: AtomType
    label: str
    mass: str
    depth: int


@dataclass
class TransferReport:
    reference_counts: dict[str, int] = field(default_factory=dict)
    target_counts: dict[str, int] = field(default_factory=dict)
    depth_histogram: Counter = field(default_factory=Counter)
    completeness: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    method: str = ""
    cross_check: str = ""

    def lines(self) -> list[str]:
        lines = []
        if self.method:
            lines.append(f"method: {self.method}")
        if self.cross_check:
            lines.append(f"cross-check: {self.cross_check}")
        lines.append("terms (reference -> target):")
        for kind in KIND_ORDER:
            lines.append(
                f"  {kind:<10s} {self.reference_counts.get(kind, 0):7d} -> "
                f"{self.target_counts.get(kind, 0):7d}   {self.completeness.get(kind, '')}"
            )
        lines.append(
            "environment match depth (bond shells identical between target and reference; "
            f"max {DEPTH_MAX}, required >= {DEPTH_MIN}):"
        )
        for depth in sorted(self.depth_histogram):
            lines.append(f"  depth {depth}: {self.depth_histogram[depth]} atoms/terms")
        if self.depth_histogram and min(self.depth_histogram) < DEPTH_MAX:
            lines.append(
                f"  note: depth < {DEPTH_MAX} means that, within {DEPTH_MAX} bonds of some terms, the "
                "best reference instance already sees a reference chain end; a longer reference "
                "oligomer gives a closer match (bonded parameters usually depend on 1-2 shells)."
            )
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        else:
            lines.append("warnings: none")
        return lines


@dataclass
class TransferResult:
    atoms: list[TransferredAtom]
    terms: list[Term]
    report: TransferReport


@dataclass
class _Instance:
    """One reference occurrence of a term pattern (several lines for multi-term dihedrals)."""

    first_unit: int
    lines: tuple[Term, ...]

    def parameters(self) -> tuple:
        return tuple((line.funct, line.params) for line in self.lines)


class _Matcher:
    def __init__(
        self,
        reference_graph: dict[int, set[int]],
        reference_labels: dict[int, AtomLabel],
        target_graph: dict[int, set[int]],
        target_labels: dict[int, AtomLabel],
        depth_max: int,
    ) -> None:
        self.reference_graph = reference_graph
        self.target_graph = target_graph
        self.reference_by_label = {label: index for index, label in reference_labels.items()}
        self.target_labels = target_labels
        self.depth_max = depth_max

    def depth(self, seeds: Sequence[int], shift: int) -> int:
        """Bond shells around ``seeds`` that map exactly onto the reference.

        Target atom with label (u, key) maps to reference atom (u + shift, key).
        Depth k: every atom closer than k bonds to a seed has a reference image
        whose neighbour set is exactly the image of its own neighbour set.
        """

        def image(atom: int) -> int | None:
            unit, key = self.target_labels[atom]
            return self.reference_by_label.get((unit + shift, key))

        visited = set(seeds)
        frontier = list(dict.fromkeys(seeds))
        for layer in range(self.depth_max):
            next_frontier = []
            for atom in frontier:
                reference_atom = image(atom)
                if reference_atom is None:
                    return layer
                mapped = {image(neighbor) for neighbor in self.target_graph[atom]}
                if None in mapped or mapped != self.reference_graph[reference_atom]:
                    return layer
                for neighbor in self.target_graph[atom]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return self.depth_max


def _position_order(
    first_unit: int, span: int, reference_dp: int, head_gap: int, tail_gap: int
) -> tuple[int, float, int]:
    """Tie-break among equally matching reference instances.

    Prefer the instance whose distances (in units) from the reference chain
    ends match the target term's distances from the target chain ends, each
    capped at half the reference length; then the most central instance.
    With target DP == reference DP this selects the identical instance, and
    far-interior target terms use the central reference units.
    """

    cap = max(1, (reference_dp - 1) // 2)
    reference_tail_gap = reference_dp - 1 - (first_unit + span)
    role_cost = abs(min(head_gap, cap) - min(first_unit, cap)) + abs(
        min(tail_gap, cap) - min(reference_tail_gap, cap)
    )
    center = (reference_dp - 1) / 2
    return (role_cost, abs(first_unit + span / 2 - center), first_unit)


def transfer_by_environment_match(
    reference: GromacsTopology,
    reference_bonds: Iterable[tuple[int, int]],
    reference_labels: dict[int, AtomLabel],
    target_bonds: Iterable[tuple[int, int]],
    target_labels: dict[int, AtomLabel],
    *,
    depth_max: int = DEPTH_MAX,
    depth_min: int = DEPTH_MIN,
) -> TransferResult:
    """Cross-check transfer: choose each term's reference copy by environment match.

    Every reference term pattern is placed at every target position; among
    the reference copies of the pattern, the one whose bonded neighbourhood
    matches the target's to the most bond shells is used (ties: the copy at
    the same distance from the chain ends, then the most central one).
    """

    _check_labels(reference, reference_labels, target_labels)
    reference_count = len(reference.atoms)
    target_count = len(target_labels)

    reference_graph = adjacency(reference_count, reference_bonds)
    target_graph = adjacency(target_count, target_bonds)
    reference_dp = max(unit for unit, _ in reference_labels.values()) + 1
    target_dp = max(unit for unit, _ in target_labels.values()) + 1
    target_by_label = {label: index for index, label in target_labels.items()}
    matcher = _Matcher(reference_graph, reference_labels, target_graph, target_labels, depth_max)
    report = TransferReport()
    clip = depth_max + 1
    depth_cache: dict[tuple, int] = {}

    def best_instances(
        pattern: tuple, span: int, placement: int, seeds: Sequence[int], instances: list[_Instance]
    ) -> tuple[int, list[_Instance]]:
        tail_gap = target_dp - 1 - (placement + span)
        scored = []
        for instance in instances:
            key = (pattern, instance.first_unit, min(placement, clip), min(tail_gap, clip))
            if key not in depth_cache:
                depth_cache[key] = matcher.depth(seeds, instance.first_unit - placement)
            scored.append((depth_cache[key], instance))
        best = max(depth for depth, _ in scored)
        chosen = sorted(
            (instance for depth, instance in scored if depth == best),
            key=lambda instance: _position_order(instance.first_unit, span, reference_dp, placement, tail_gap),
        )
        return best, chosen

    def describe(pattern: tuple, placement: int) -> str:
        return ", ".join(f"unit {placement + offset + 1} {key}" for offset, key in pattern)

    # -- atoms (single-atom patterns) --------------------------------------
    atom_instances: dict[tuple, list[_Instance]] = defaultdict(list)
    for index, (unit, key) in reference_labels.items():
        atom_instances[((0, key),)].append(_Instance(unit, ()))
    transferred_atoms: list[TransferredAtom] = []
    type_conflicts = 0
    for index in range(1, target_count + 1):
        unit, key = target_labels[index]
        pattern = ((0, key),)
        instances = atom_instances.get(pattern)
        if not instances:
            raise ValueError(f"Target atom {index} ({describe(pattern, unit)}) has no reference counterpart.")
        depth, chosen = best_instances(pattern, 0, unit, [index], instances)
        if depth < depth_min:
            raise ValueError(
                f"Target atom {index} ({describe(pattern, unit)}) matches the reference "
                f"environment only {depth} bond shell(s) deep (need {depth_min}); "
                "use a longer reference oligomer."
            )
        choices = []
        for instance in chosen:
            reference_index = matcher.reference_by_label[(instance.first_unit, key)]
            atom_type = reference.atom_type(reference_index)
            choices.append(
                (reference_index, atom_type, reference.label(reference_index), reference.atoms[reference_index - 1].mass)
            )
        reference_index, atom_type, label, mass = choices[0]
        signatures = {(choice[1].nonbonded_key(), choice[2]) for choice in choices}
        if len(signatures) > 1:
            type_conflicts += 1
            if type_conflicts <= 5:
                report.warnings.append(
                    f"target atom {index}: equally matching reference atoms disagree on the "
                    f"atom type ({', '.join(sorted({choice[2] for choice in choices}))}); "
                    f"the best-positioned one (reference atom {reference_index}) was used"
                )
        report.depth_histogram[depth] += 1
        transferred_atoms.append(TransferredAtom(index, atom_type, label, mass, depth))
    if type_conflicts > 5:
        report.warnings.append(f"... {type_conflicts - 5} more atom-type disagreements")

    # -- bonded terms ------------------------------------------------------
    grouped: dict[tuple, list[Term]] = defaultdict(list)
    for term in reference.terms:
        grouped[(term.section, term.atoms, term.funct)].append(term)
    pattern_instances: dict[tuple, list[_Instance]] = defaultdict(list)
    reference_kind_sets: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for (section, atoms, funct), lines in grouped.items():
        kind = term_kind(section, funct)
        reference_kind_sets[kind].add(canonical_atoms(kind, atoms))
        labels = [reference_labels[atom] for atom in atoms]
        first_unit = min(unit for unit, _ in labels)
        pattern = tuple((unit - first_unit, key) for unit, key in labels)
        pattern_instances[(section, funct, pattern)].append(_Instance(first_unit, tuple(lines)))
    for kind in KIND_ORDER:
        report.reference_counts[kind] = len(reference_kind_sets.get(kind, ()))

    selected: dict[tuple, tuple[int, list[Term]]] = {}
    parameter_conflicts = 0
    for (section, funct, pattern), instances in sorted(pattern_instances.items(), key=lambda item: repr(item[0])):
        kind = term_kind(section, funct)
        span = max(offset for offset, _ in pattern)
        for placement in range(0, target_dp - span):
            target_atoms = []
            for offset, key in pattern:
                target_atom = target_by_label.get((placement + offset, key))
                if target_atom is None:
                    break
                target_atoms.append(target_atom)
            else:
                identity = (section, kind, canonical_atoms(kind, target_atoms))
                depth, chosen = best_instances(pattern, span, placement, target_atoms, instances)
                previous = selected.get(identity)
                if previous is not None and previous[0] >= depth:
                    continue
                if depth < depth_min:
                    selected[identity] = (depth, [])
                    continue
                if len({instance.parameters() for instance in chosen}) > 1:
                    parameter_conflicts += 1
                    if parameter_conflicts <= 5:
                        report.warnings.append(
                            f"{kind} {tuple(target_atoms)}: equally matching reference instances "
                            "have different parameters; the best-positioned one was used"
                        )
                instance = chosen[0]
                atom_map = {}
                reference_atoms = instance.lines[0].atoms
                for reference_atom, target_atom in zip(reference_atoms, target_atoms):
                    atom_map[reference_atom] = target_atom
                lines = [
                    Term(line.section, tuple(atom_map[atom] for atom in line.atoms), line.funct, line.params, line.comment)
                    for line in instance.lines
                ]
                selected[identity] = (depth, lines)
    if parameter_conflicts > 5:
        report.warnings.append(f"... {parameter_conflicts - 5} more parameter disagreements")

    weak = [identity for identity, (depth, lines) in selected.items() if not lines]
    if weak:
        examples = "; ".join(f"{identity[1]} {identity[2]}" for identity in weak[:5])
        raise ValueError(
            f"{len(weak)} target term(s) match the reference environment less than {depth_min} "
            f"bond shells deep (e.g. {examples}); use a longer reference oligomer or a longer target."
        )

    target_terms: list[Term] = []
    target_kind_sets: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for (section, kind, atoms), (depth, lines) in selected.items():
        report.depth_histogram[depth] += 1
        target_terms.extend(lines)
        target_kind_sets[kind].add(atoms)
    for kind in KIND_ORDER:
        report.target_counts[kind] = len(target_kind_sets.get(kind, ()))

    _check_completeness(reference_kind_sets, target_kind_sets, reference_graph, target_graph, report)
    return TransferResult(transferred_atoms, target_terms, report)


def _check_completeness(
    reference_kind_sets: dict[str, set[tuple[int, ...]]],
    target_kind_sets: dict[str, set[tuple[int, ...]]],
    reference_graph: dict[int, set[int]],
    target_graph: dict[int, set[int]],
    report: TransferReport,
) -> None:
    """Graph-defined terms must cover the target whenever they cover the reference."""

    for kind in KIND_ORDER:
        reference_expected = expected_terms(kind, reference_graph)
        if reference_expected is None:
            report.completeness[kind] = "(not graph-defined; projected from the reference)"
            continue
        if reference_kind_sets.get(kind, set()) != reference_expected:
            report.completeness[kind] = "(reference itself is not graph-complete; not checked)"
            continue
        target_expected = expected_terms(kind, target_graph)
        target_actual = target_kind_sets.get(kind, set())
        missing = sorted(target_expected - target_actual)
        extra = sorted(target_actual - target_expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {len(missing)}, e.g. {missing[:5]}")
            if extra:
                details.append(f"unexpected {len(extra)}, e.g. {extra[:5]}")
            raise ValueError(
                f"Transferred {kind} do not cover the target bond graph ({'; '.join(details)}). "
                "The reference oligomer is probably too short for this monomer; increase the "
                "reference size."
            )
        report.completeness[kind] = "(complete for the target graph)"


def _check_labels(
    reference: GromacsTopology, reference_labels: dict[int, AtomLabel], target_labels: dict[int, AtomLabel]
) -> None:
    if set(reference_labels) != set(range(1, len(reference.atoms) + 1)):
        raise ValueError("Every reference atom needs a repeat-unit label.")
    if set(target_labels) != set(range(1, len(target_labels) + 1)):
        raise ValueError("Every target atom needs a repeat-unit label.")


@dataclass
class _UnitNumbering:
    """Atoms addressed as (unit, index within the unit).

    Within a unit the monomer atoms come first, in monomer order, followed by
    the hydrogens ordered by parent atom and slot. Shifting an atom by k units
    keeps its index within the unit.
    """

    unit_of: dict[int, int]
    local_of: dict[int, int]
    index_at: dict[tuple[int, int], int]
    composition: dict[int, tuple]

    @classmethod
    def from_labels(cls, labels: dict[int, AtomLabel]) -> "_UnitNumbering":
        by_unit: dict[int, list[tuple[tuple, int]]] = defaultdict(list)
        for index, (unit, key) in labels.items():
            by_unit[unit].append((key, index))
        unit_of, local_of, index_at, composition = {}, {}, {}, {}
        for unit, members in by_unit.items():
            members.sort()
            composition[unit] = tuple(key for key, _ in members)
            for local, (_, index) in enumerate(members):
                unit_of[index] = unit
                local_of[index] = local
                index_at[(unit, local)] = index
        return cls(unit_of, local_of, index_at, composition)


def end_unit_count(reference_dp: int) -> int:
    """Units at each chain end that are copied from the same position of the reference."""

    return max(1, (reference_dp - 1) // 2)


def shift_source_unit(first_unit: int, span: int, target_dp: int, reference_dp: int) -> int:
    """First reference unit a target term spanning ``first_unit``..``first_unit + span`` is copied from.

    - near the head (first unit among the first E): the same units of the reference (no shift);
    - near the tail (last unit among the last E): the same distance from the reference tail;
    - otherwise: the most central window of the reference.
    E = ``end_unit_count(reference_dp)``.
    """

    end_units = end_unit_count(reference_dp)
    head_gap = first_unit
    tail_gap = target_dp - 1 - (first_unit + span)
    if head_gap < end_units:
        return head_gap
    if tail_gap < end_units:
        return reference_dp - 1 - span - tail_gap
    return (reference_dp - 1 - span) // 2


def transfer_by_index_shift(
    reference: GromacsTopology,
    reference_bonds: Iterable[tuple[int, int]],
    reference_labels: dict[int, AtomLabel],
    target_bonds: Iterable[tuple[int, int]],
    target_labels: dict[int, AtomLabel],
) -> TransferResult:
    """Primary transfer: copy reference terms to the target by shifting whole units.

    A reference term occupying units r..r+span is copied to target units
    p..p+span (every atom keeps its index within its unit) exactly for the
    positions p whose source, by ``shift_source_unit``, is r. Each target atom
    takes the atom type of the same atom in its source unit. Source and target
    units must contain the same atoms (same monomer atoms and hydrogens).
    """

    _check_labels(reference, reference_labels, target_labels)
    reference_graph = adjacency(len(reference.atoms), reference_bonds)
    target_graph = adjacency(len(target_labels), target_bonds)
    ref = _UnitNumbering.from_labels(reference_labels)
    tgt = _UnitNumbering.from_labels(target_labels)
    reference_dp = max(ref.composition) + 1
    target_dp = max(tgt.composition) + 1
    report = TransferReport(
        method=(
            f"index shift; the first and last {end_unit_count(reference_dp)} unit(s) of the "
            f"{target_dp}-unit target are copied from the same positions of the {reference_dp}-unit "
            "reference, all other terms from its most central window"
        )
    )

    def check_composition(source_unit: int, target_unit: int) -> None:
        if ref.composition.get(source_unit) != tgt.composition[target_unit]:
            raise ValueError(
                f"Target unit {target_unit + 1} and its source reference unit {source_unit + 1} "
                "do not contain the same atoms, so the index shift is undefined; the target must be "
                "at least as long as the reference (shorter targets are parameterized directly)."
            )

    def check_window(source: int, span: int, what: str) -> None:
        if source < 0 or source + span > reference_dp - 1:
            raise ValueError(
                f"{what} spans {span + 1} units, more than the reference interior offers; "
                "increase the reference size."
            )

    # Interior target terms are copied from reference windows made only of
    # interior units: a term spanning n units needs n + 2 reference units.
    widest = max(
        (
            max(ref.unit_of[atom] for atom in term.atoms) - min(ref.unit_of[atom] for atom in term.atoms) + 1
            for term in reference.terms
        ),
        default=1,
    )
    if target_dp > reference_dp and reference_dp < widest + 2:
        raise ValueError(
            f"Some terms of this monomer span {widest} repeat units, so the index shift needs a "
            f"reference oligomer of at least {widest + 2} units (one interior window of {widest} "
            f"units plus both chain ends); the reference has {reference_dp}. Increase the "
            "reference oligomer size."
        )

    # -- atoms -------------------------------------------------------------
    atoms: list[TransferredAtom] = []
    for index in range(1, len(target_labels) + 1):
        unit = tgt.unit_of[index]
        source = shift_source_unit(unit, 0, target_dp, reference_dp)
        check_window(source, 0, f"Target atom {index}")
        check_composition(source, unit)
        reference_index = ref.index_at[(source, tgt.local_of[index])]
        atoms.append(
            TransferredAtom(
                index,
                reference.atom_type(reference_index),
                reference.label(reference_index),
                reference.atoms[reference_index - 1].mass,
                DEPTH_MAX,
            )
        )

    # -- bonded terms ------------------------------------------------------
    grouped: dict[tuple, list[Term]] = defaultdict(list)
    for term in reference.terms:
        grouped[(term.section, term.atoms, term.funct)].append(term)
    reference_kind_sets: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    target_kind_sets: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    target_terms: list[Term] = []
    for (section, reference_atoms, funct), lines in grouped.items():
        kind = term_kind(section, funct)
        reference_kind_sets[kind].add(canonical_atoms(kind, reference_atoms))
        units = [ref.unit_of[atom] for atom in reference_atoms]
        first_unit = min(units)
        span = max(units) - first_unit
        for placement in range(0, target_dp - span):
            if shift_source_unit(placement, span, target_dp, reference_dp) != first_unit:
                continue
            shift = placement - first_unit
            atom_map = {}
            for atom in reference_atoms:
                target_atom = tgt.index_at.get((ref.unit_of[atom] + shift, ref.local_of[atom]))
                if target_atom is None:
                    break
                check_composition(ref.unit_of[atom], ref.unit_of[atom] + shift)
                atom_map[atom] = target_atom
            else:
                identity = canonical_atoms(kind, [atom_map[atom] for atom in reference_atoms])
                if identity in target_kind_sets[kind]:
                    raise ValueError(f"Index shift produced the {kind} term {identity} twice.")
                target_kind_sets[kind].add(identity)
                target_terms.extend(
                    Term(line.section, tuple(atom_map[atom] for atom in line.atoms), line.funct, line.params, line.comment)
                    for line in lines
                )
    for kind in KIND_ORDER:
        report.reference_counts[kind] = len(reference_kind_sets.get(kind, ()))
        report.target_counts[kind] = len(target_kind_sets.get(kind, ()))
    _check_completeness(reference_kind_sets, target_kind_sets, reference_graph, target_graph, report)
    return TransferResult(atoms, target_terms, report)


def _atom_signature(atom: TransferredAtom) -> tuple:
    atom_type = atom.atom_type
    return (
        atom.label,
        atom_type.bond_type,
        atom_type.atomic_number,
        atom_type.mass,
        atom_type.ptype,
        atom_type.sigma,
        atom_type.epsilon,
        atom.mass,
    )


def _term_signature(term: Term) -> tuple:
    return (term.section, term.kind, canonical_atoms(term.kind, term.atoms), term.funct, term.params, term.comment)


def compare_transfers(primary: TransferResult, check: TransferResult, limit: int = 5) -> list[str]:
    """Differences between two transfers: per-atom types and the multiset of term lines.

    Atom types are compared by label, bond type and the exact parameter text
    (atomic number, mass, ptype, sigma, epsilon); terms by atoms (direction-
    independent except for impropers), function, parameter text and comment.
    """

    differences: list[str] = []
    if len(primary.atoms) != len(check.atoms):
        return [f"atom counts differ: {len(primary.atoms)} vs {len(check.atoms)}"]
    atom_differences = [
        (a.index, _atom_signature(a), _atom_signature(b))
        for a, b in zip(primary.atoms, check.atoms)
        if _atom_signature(a) != _atom_signature(b)
    ]
    for index, mine, theirs in atom_differences[:limit]:
        differences.append(f"atom {index}: index shift {mine} vs environment match {theirs}")
    if len(atom_differences) > limit:
        differences.append(f"... {len(atom_differences) - limit} more atom differences")
    primary_terms = Counter(map(_term_signature, primary.terms))
    check_terms = Counter(map(_term_signature, check.terms))
    only_primary = sorted((primary_terms - check_terms).elements())
    only_check = sorted((check_terms - primary_terms).elements())
    for signature in only_primary[:limit]:
        differences.append(f"only from index shift: {signature[1]} {signature[2]} funct {signature[3]} {signature[4]}")
    for signature in only_check[:limit]:
        differences.append(f"only from environment match: {signature[1]} {signature[2]} funct {signature[3]} {signature[4]}")
    extra = max(0, len(only_primary) - limit) + max(0, len(only_check) - limit)
    if extra:
        differences.append(f"... {extra} more term differences")
    return differences


class TransferDisagreement(ValueError):
    """The index-shift transfer and the environment-match cross-check differ."""


def transfer_topology(
    reference: GromacsTopology,
    reference_bonds: Iterable[tuple[int, int]],
    reference_labels: dict[int, AtomLabel],
    target_bonds: Iterable[tuple[int, int]],
    target_labels: dict[int, AtomLabel],
) -> TransferResult:
    """Transfer by index shift, accepted only if the environment-match cross-check agrees exactly."""

    reference_bonds = list(reference_bonds)
    target_bonds = list(target_bonds)
    shifted = transfer_by_index_shift(reference, reference_bonds, reference_labels, target_bonds, target_labels)
    matched = transfer_by_environment_match(reference, reference_bonds, reference_labels, target_bonds, target_labels)
    differences = compare_transfers(shifted, matched)
    if differences:
        raise TransferDisagreement(
            "The index-shift transfer and the environment-matching cross-check disagree, so the "
            "result was rejected. This usually means the reference oligomer is too short for this "
            "monomer; increase the reference size.\n  " + "\n  ".join(differences)
        )
    report = shifted.report
    report.depth_histogram = matched.report.depth_histogram
    report.warnings.extend(matched.report.warnings)
    report.cross_check = (
        f"environment matching gives identical atom types for all {len(shifted.atoms)} atoms and "
        f"identical parameters for all {len(shifted.terms)} term lines"
    )
    for primary, checked in zip(shifted.atoms, matched.atoms):
        primary.depth = checked.depth
    return shifted


def identity_transfer(reference: GromacsTopology) -> TransferResult:
    """Use a topology unchanged (the target was parameterized directly)."""

    atoms = [
        TransferredAtom(atom.index, reference.atom_type(atom.index), reference.label(atom.index), atom.mass, DEPTH_MAX)
        for atom in reference.atoms
    ]
    report = TransferReport()
    kinds: dict[str, set] = defaultdict(set)
    for term in reference.terms:
        kinds[term.kind].add(canonical_atoms(term.kind, term.atoms))
    for kind in KIND_ORDER:
        report.reference_counts[kind] = report.target_counts[kind] = len(kinds.get(kind, ()))
        report.completeness[kind] = "(target parameterized directly)"
    return TransferResult(atoms, list(reference.terms), report)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _type_name_base(label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_+\-]", "_", label).strip("_") or "X"
    return name if not name[0].isdigit() else f"t{name}"


def merge_atom_types(atoms: Sequence[TransferredAtom], merge: bool = True) -> tuple[list[str], list[tuple[str, AtomType]]]:
    """Name the target atom types.

    With ``merge`` (default), atoms share a type name exactly when their label,
    bond_type and complete non-bonded parameters (atomic number, mass, ptype,
    sigma, epsilon) are identical. A label reused with different parameters
    gets a suffix (``opls_135_2``). Without ``merge``, every atom gets its own
    type, which is useful to verify that merging changes nothing.
    """

    names: list[str] = []
    table: list[tuple[str, AtomType]] = []
    by_key: dict[tuple, str] = {}
    used: set[str] = set()
    for atom in atoms:
        key = (atom.label, atom.atom_type.bond_type, atom.atom_type.nonbonded_key())
        if merge and key in by_key:
            names.append(by_key[key])
            continue
        base = _type_name_base(atom.label)
        name = base if merge else f"{base}_a{atom.index}"
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        by_key[key] = name
        names.append(name)
        table.append((name, atom.atom_type))
    return names, table


def _format_term(term: Term) -> str:
    atoms = " ".join(f"{atom:6d}" for atom in term.atoms)
    params = " ".join(f"{value:>14s}" for value in term.params)
    comment = f" ; {term.comment}" if term.comment else ""
    return f"{atoms} {term.funct:5d} {params}".rstrip() + comment


def write_gromacs_files(
    itp_path: Path,
    top_path: Path,
    reference: GromacsTopology,
    transfer: TransferResult,
    charges: Sequence[float],
    atom_names: Sequence[str],
    residue_numbers: Sequence[int],
    *,
    molecule_name: str = "POL",
    residue_name: str = "POL",
    system_name: str = "Polymer",
    header: Sequence[str] = (),
    merge: bool = True,
) -> dict[str, int]:
    """Write the target ``.itp`` (atomtypes + moleculetype) and a wrapper ``.top``."""

    if reference.defaults is None:
        raise ValueError(f"Reference topology {reference.path} has no [ defaults ] section.")
    count = len(transfer.atoms)
    if not (len(charges) == len(atom_names) == len(residue_numbers) == count):
        raise ValueError("charges, atom names and residue numbers must cover every target atom.")
    names, table = merge_atom_types(transfer.atoms, merge=merge)

    lines = [f"; {line}" if line else ";" for line in header]
    lines += [
        "; Atom types are named by their force-field label and merged only when the",
        "; complete non-bonded parameter set (atomic number, mass, sigma, epsilon) is",
        "; identical; every bonded term below carries explicit parameters.",
        "",
        "[ atomtypes ]",
        "; name          bond_type  at.num          mass    charge ptype          sigma        epsilon",
    ]
    for name, atom_type in table:
        if atom_type.atomic_number is None:
            raise ValueError(f"Atom type {atom_type.name} has no atomic number.")
        lines.append(
            f"{name:<15s} {atom_type.bond_type:<10s} {atom_type.atomic_number:6d} "
            f"{atom_type.mass:>13s} {0.0:9.4f} {atom_type.ptype:>5s} "
            f"{atom_type.sigma:>14s} {atom_type.epsilon:>14s}"
        )
    lines += [
        "",
        "[ moleculetype ]",
        "; name   nrexcl",
        f"{molecule_name}  {reference.nrexcl}",
        "",
        "[ atoms ]",
        ";   nr  type            resnr residue  atom   cgnr        charge          mass",
    ]
    for atom, name, charge, atom_name, residue in zip(transfer.atoms, names, charges, atom_names, residue_numbers):
        lines.append(
            f"{atom.index:6d}  {name:<15s} {residue:5d} {residue_name:<7s} {atom_name:<6s} "
            f"{atom.index:6d} {charge:13.8f} {atom.mass:>13s} ; {atom.label}"
        )

    counts: dict[str, int] = {}
    for kind in KIND_ORDER:
        kind_terms = sorted(
            (term for term in transfer.terms if term.kind == kind),
            key=lambda term: canonical_atoms(kind, term.atoms),
        )
        counts[kind] = len(kind_terms)
        if not kind_terms:
            continue
        lines += ["", f"[ {KIND_SECTION[kind]} ]"]
        if kind in {"propers", "impropers"}:
            lines.append(f"; {kind[:-1]} dihedrals")
        lines.extend(_format_term(term) for term in kind_terms)

    itp_path.write_text("\n".join(lines) + "\n")
    top_path.write_text(
        "\n".join(
            [
                *(f"; {line}" if line else ";" for line in header),
                "",
                "[ defaults ]",
                "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
                "  ".join(reference.defaults),
                "",
                f'#include "{itp_path.name}"',
                "",
                "[ system ]",
                system_name,
                "",
                "[ molecules ]",
                f"{molecule_name}  1",
            ]
        )
        + "\n"
    )
    counts["atomtypes"] = len(table)
    return counts
