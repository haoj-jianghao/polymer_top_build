"""Starting 3D coordinates for a polymer: a self-avoiding random walk of rigid monomer units.

This is independent of the force field. The reference oligomer already has a
minimized 3D geometry (with all hydrogens and the drawn stereochemistry); its
units serve as rigid templates:

* every polymer unit copies the coordinates of the reference unit chosen by
  the same rule as the parameter transfer (``topology_transfer.shift_source_unit``:
  head and tail units from the reference ends, interior units from the
  central reference unit), so chain ends get end geometry;
* a new unit is attached to the previous one through the junction bond. Its
  direction comes from where the previous unit's next neighbour sat in the
  reference, and the bond length and the angles at the new unit's connection
  atom come from the reference junction;
* the one random degree of freedom is the rotation about the junction bond:
  a staggered rotamer (0, 120 or 240 degrees from the reference conformation;
  0 or 180 for a double-bond junction, none for a triple bond) plus a small
  uniform jitter (if no jittered rotamer fits, the exact rotamers are tried
  before backtracking);
* a placement is rejected if any new atom comes closer than
  ``CLASH_SCALE`` x (sum of van der Waals radii) to an already placed atom
  more than three bonds away (``HBOND_SCALE`` for a polar hydrogen and an
  N/O/F acceptor, which hydrogen-bond); when no rotamer of a unit fits, the
  walk backtracks. The random generator is seeded, so a build is reproducible.
  Contacts the minimized reference itself has are never clashes: if its
  closest non-bonded contact is below those limits (perfluorocarbons put 1-6
  F...F pairs at about 2.3 A against a 2.35 A limit), every limit is scaled
  down to ``REFERENCE_MARGIN`` x that contact.

Units are rigid copies of reference units, so stereocentres keep the
configuration of the reference (the drawn wedges and dashes).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from topology_transfer import shift_source_unit

AtomLabel = tuple[int, tuple]

VDW_RADIUS = {  # Bondi radii (Å); Si from Mantina et al.
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Si": 2.10,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}
CLASH_SCALE = 0.80
# Hydrogen-bonded pairs (a polar H and an N/O/F acceptor) legitimately come much
# closer than the sum of their van der Waals radii (H...O about 1.8-2.0 A).
HBOND_SCALE = 0.60
HBOND_DONOR_PARENTS = frozenset({"N", "O", "S"})
HBOND_ACCEPTORS = frozenset({"N", "O", "F"})
EXCLUDED_BOND_SEPARATION = 3  # pairs up to 1-4 are bonded neighbours, never clashes
REFERENCE_MARGIN = 0.95  # limits scaled to 95 % of the reference's own closest contact
ROTAMERS = {1: (0.0, 120.0, 240.0), 2: (0.0, 180.0)}  # by junction bond order; triple: (0,)
DEFAULT_JITTER_DEG = 15.0


@dataclass
class WalkResult:
    coordinates: dict[int, tuple[float, float, float]]
    rotamers: list[float] = field(default_factory=list)
    placements_tried: int = 0
    backtracks: int = 0
    min_contact_ratio: float = math.inf
    # 1.0, or < 1 when the reference's own closest contact required looser limits
    limit_scale: float = 1.0
    end_to_end: float = 0.0
    radius_of_gyration: float = 0.0
    seed: int = 0

    def lines(self) -> list[str]:
        return [
            "coordinates: self-avoiding random walk of rigid reference units",
            f"  seed: {self.seed}",
            f"  junction rotations (deg, first 12): {[round(value, 1) for value in self.rotamers[:12]]}",
            f"  placements tried: {self.placements_tried}; backtracks: {self.backtracks}",
            "  closest non-bonded contact (beyond 1-4) relative to its clash limit: "
            f"{self.min_contact_ratio:.3f} (>= 1; limits {CLASH_SCALE} x vdW sum, {HBOND_SCALE} for H-bond pairs"
            + (
                f", scaled by {self.limit_scale:.3f} to the reference's own closest contact)"
                if self.limit_scale < 1.0
                else ")"
            ),
            f"  end-to-end distance: {self.end_to_end:.2f} A; radius of gyration (heavy atoms): "
            f"{self.radius_of_gyration:.2f} A",
        ]


def _rotation_about(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    angle = math.radians(degrees)
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _frame(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns): e1 along a->b, e2 towards c, e3 = e1 x e2."""

    e1 = b - a
    e1 = e1 / np.linalg.norm(e1)
    v = c - a
    e2 = v - np.dot(v, e1) * e1
    norm = np.linalg.norm(e2)
    if norm < 1e-8:
        raise ValueError("Junction frame is degenerate (connection atoms are collinear).")
    e2 = e2 / norm
    return np.column_stack([e1, e2, np.cross(e1, e2)])


class _Grid:
    def __init__(self, cell: float) -> None:
        self.cell = cell
        self.cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def key(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(int(math.floor(value / self.cell)) for value in point)

    def add(self, index: int, point: np.ndarray) -> None:
        self.cells[self.key(point)].append(index)

    def remove(self, index: int, point: np.ndarray) -> None:
        self.cells[self.key(point)].remove(index)

    def near(self, point: np.ndarray):
        kx, ky, kz = self.key(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield from self.cells.get((kx + dx, ky + dy, kz + dz), ())


def _bond_neighbourhood(graph: dict[int, set[int]], start: int, depth: int) -> set[int]:
    seen = {start}
    frontier = deque([(start, 0)])
    while frontier:
        atom, distance = frontier.popleft()
        if distance == depth:
            continue
        for neighbour in graph[atom]:
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append((neighbour, distance + 1))
    return seen


def random_walk_coordinates(
    reference_coordinates: dict[int, tuple[float, float, float]],
    reference_labels: dict[int, AtomLabel],
    reference_bonds: list[tuple[int, int]],
    target_labels: dict[int, AtomLabel],
    target_bonds: list[tuple[int, int, int]],
    target_elements: dict[int, str],
    *,
    seed: int = 0,
    jitter_deg: float = DEFAULT_JITTER_DEG,
    max_placements: int | None = None,
) -> WalkResult:
    """3D coordinates (Å) for every target atom.

    ``target_bonds`` are (a, b, bond order); atoms carry repeat-unit labels as
    produced by ``polymer_from_oligomer.unit_atom_labels``.
    """

    rng = random.Random(seed)
    reference_dp = max(unit for unit, _ in reference_labels.values()) + 1
    target_dp = max(unit for unit, _ in target_labels.values()) + 1
    reference_by_label = {label: index for index, label in reference_labels.items()}
    reference_position = {index: np.asarray(xyz, dtype=float) for index, xyz in reference_coordinates.items()}

    units: dict[int, list[int]] = defaultdict(list)
    for index, (unit, _) in target_labels.items():
        units[unit].append(index)
    composition = {unit: sorted(target_labels[index][1] for index in members) for unit, members in units.items()}
    reference_composition: dict[int, list[tuple]] = defaultdict(list)
    for unit, key in reference_labels.values():
        reference_composition[unit].append(key)
    sources = [shift_source_unit(unit, 0, target_dp, reference_dp) for unit in range(target_dp)]
    for unit, source in enumerate(sources):
        if sorted(reference_composition[source]) != composition[unit]:
            raise ValueError(f"Target unit {unit + 1} and reference unit {source + 1} differ in atoms.")

    target_graph: dict[int, set[int]] = defaultdict(set)
    junction: dict[int, tuple[tuple, tuple, int]] = {}  # unit u -> (key of N in u-1, key of P in u, order)
    for a, b, order in target_bonds:
        target_graph[a].add(b)
        target_graph[b].add(a)
        (unit_a, key_a), (unit_b, key_b) = target_labels[a], target_labels[b]
        if unit_a != unit_b:
            if abs(unit_a - unit_b) != 1:
                raise ValueError("Only bonds between consecutive repeat units are supported.")
            (left, right) = ((key_a, key_b) if unit_a < unit_b else (key_b, key_a))
            unit = max(unit_a, unit_b)
            if unit in junction:
                raise ValueError("Neighbouring units must be joined by exactly one bond.")
            junction[unit] = (left, right, order)
    if target_dp > 1 and set(junction) != set(range(1, target_dp)):
        raise ValueError("Every pair of consecutive units must be joined by a bond.")

    reference_graph: dict[int, set[int]] = defaultdict(set)
    for a, b in reference_bonds:
        reference_graph[a].add(b)
        reference_graph[b].add(a)

    def ref_xyz(unit: int, key: tuple) -> np.ndarray:
        return reference_position[reference_by_label[(unit, key)]]

    def frame_atom_key(source_unit: int, n_key: tuple) -> tuple:
        """A neighbour of the connection atom N inside its own unit (heavy atoms first)."""

        n_index = reference_by_label[(source_unit, n_key)]
        same_unit = [reference_labels[i][1] for i in reference_graph[n_index] if reference_labels[i][0] == source_unit]
        if not same_unit:
            raise ValueError("The connection atom has no neighbour inside its unit to define the junction frame.")
        return sorted(same_unit, key=lambda key: (key[0] != "A", key))[0]

    radius = {index: VDW_RADIUS.get(element, 1.8) for index, element in target_elements.items()}
    polar_h = {
        index
        for index, element in target_elements.items()
        if element == "H" and any(target_elements[n] in HBOND_DONOR_PARENTS for n in target_graph[index])
    }
    acceptor = {index for index, element in target_elements.items() if element in HBOND_ACCEPTORS}

    def limit_for(a: int, b: int) -> float:
        hbond = (a in polar_h and b in acceptor) or (b in polar_h and a in acceptor)
        return (HBOND_SCALE if hbond else CLASH_SCALE) * limit_scale

    limit_scale = 1.0
    reference_ratio = _reference_min_contact(reference_position, reference_labels, reference_graph, target_labels, target_elements)
    if reference_ratio < 1.0:
        limit_scale = REFERENCE_MARGIN * reference_ratio

    cell = CLASH_SCALE * 2 * max(radius.values())
    grid = _Grid(cell)
    positions: dict[int, np.ndarray] = {}
    transforms: list[tuple[np.ndarray, np.ndarray]] = []
    neighbourhood_cache: dict[int, set[int]] = {}

    def neighbourhood(index: int) -> set[int]:
        if index not in neighbourhood_cache:
            neighbourhood_cache[index] = _bond_neighbourhood(target_graph, index, EXCLUDED_BOND_SEPARATION)
        return neighbourhood_cache[index]

    def place(unit: int, rotation: np.ndarray, translation: np.ndarray) -> dict[int, np.ndarray]:
        source = sources[unit]
        return {
            index: rotation @ ref_xyz(source, target_labels[index][1]) + translation for index in units[unit]
        }

    def clashes(candidate: dict[int, np.ndarray]) -> bool:
        for index, point in candidate.items():
            excluded = neighbourhood(index)
            for other in grid.near(point):
                if other in excluded:
                    continue
                distance = float(np.linalg.norm(point - positions[other]))
                if distance < limit_for(index, other) * (radius[index] + radius[other]):
                    return True
        return False

    def commit(candidate: dict[int, np.ndarray]) -> None:
        for index, point in candidate.items():
            positions[index] = point
            grid.add(index, point)

    def uncommit(unit: int) -> None:
        for index in units[unit]:
            grid.remove(index, positions.pop(index))

    # head unit: the reference coordinates of its source unit
    head = place(0, np.eye(3), np.zeros(3))
    commit(head)
    transforms.append((np.eye(3), np.zeros(3)))

    result = WalkResult(coordinates={}, seed=seed, limit_scale=limit_scale)
    limit = max_placements or 200 * target_dp + 1000
    options: list[list[float]] = [[]]
    unit = 1
    chosen: list[float] = []
    while unit < target_dp:
        if len(options) <= unit:
            order = junction[unit][2]
            rotamers = list(ROTAMERS.get(order, (0.0,)))
            rng.shuffle(rotamers)
            jittered = [value + rng.uniform(-jitter_deg, jitter_deg) * (order == 1) for value in rotamers]
            # Tried last to first: the jittered rotamers, then (crowded chains such as
            # perfluorocarbons, where jitter alone can cause a clash) the exact ones.
            exact = [value for value in reversed(rotamers)] if order == 1 and jitter_deg else []
            options.append(exact + jittered)
        if not options[unit]:
            # every rotamer of this unit clashed: undo the previous unit and try its next rotamer
            options.pop()
            unit -= 1
            if unit < 1:
                raise RuntimeError("The random walk could not place the second unit without clashes.")
            uncommit(unit)
            transforms.pop()
            chosen.pop()
            result.backtracks += 1
            continue
        if result.placements_tried >= limit:
            raise RuntimeError(
                f"The random walk could not grow a clash-free chain within {limit} placements "
                "(try another --chain-seed)."
            )
        phi = options[unit].pop()
        result.placements_tried += 1

        n_key, p_key, _ = junction[unit]
        rotation_prev, translation_prev = transforms[unit - 1]
        previous_source, source = sources[unit - 1], sources[unit]
        c_key = frame_atom_key(previous_source, n_key)
        # world: N and its in-unit neighbour C of the previous unit, and where its next
        # neighbour P sat in the reference
        n_world = rotation_prev @ ref_xyz(previous_source, n_key) + translation_prev
        c_world = rotation_prev @ ref_xyz(previous_source, c_key) + translation_prev
        p_world = rotation_prev @ ref_xyz(previous_source + 1, p_key) + translation_prev
        # template: the same three atoms around the junction into this unit's source unit
        n_ref = ref_xyz(source - 1, n_key)
        c_ref = ref_xyz(source - 1, c_key)
        p_ref = ref_xyz(source, p_key)
        base = _frame(n_world, p_world, c_world) @ _frame(n_ref, p_ref, c_ref).T
        spin = _rotation_about(p_world - n_world, phi)
        rotation = spin @ base
        translation = n_world - rotation @ n_ref
        candidate = place(unit, rotation, translation)
        if clashes(candidate):
            continue
        commit(candidate)
        transforms.append((rotation, translation))
        chosen.append(phi)
        unit += 1

    result.rotamers = chosen
    result.placements_tried = result.placements_tried
    result.coordinates = {index: tuple(float(value) for value in point) for index, point in positions.items()}
    result.min_contact_ratio = _overall_min_contact(positions, radius, grid, neighbourhood, limit_for)
    heavy = np.array([point for index, point in positions.items() if target_elements[index] != "H"])
    result.radius_of_gyration = float(np.sqrt(((heavy - heavy.mean(axis=0)) ** 2).sum(axis=1).mean()))
    if target_dp > 1:
        first = next(index for index in units[0] if target_labels[index][1] == junction[1][1])
        last = next(index for index in units[target_dp - 1] if target_labels[index][1] == junction[1][0])
        result.end_to_end = float(np.linalg.norm(positions[last] - positions[first]))
    return result


def _reference_min_contact(reference_position, reference_labels, reference_graph, target_labels, target_elements) -> float:
    """The reference's closest non-bonded contact (beyond 1-4) relative to the unscaled clash limits."""

    element_of_key = {key: target_elements[index] for index, (_, key) in target_labels.items()}
    element = {index: element_of_key.get(key, "C") for index, (_, key) in reference_labels.items()}
    radius = {index: VDW_RADIUS.get(value, 1.8) for index, value in element.items()}
    polar_h = {
        index for index, value in element.items()
        if value == "H" and any(element[n] in HBOND_DONOR_PARENTS for n in reference_graph[index])
    }
    acceptor = {index for index, value in element.items() if value in HBOND_ACCEPTORS}
    indices = sorted(reference_position)
    worst = math.inf
    for position, index in enumerate(indices):
        excluded = _bond_neighbourhood(reference_graph, index, EXCLUDED_BOND_SEPARATION)
        for other in indices[position + 1:]:
            if other in excluded:
                continue
            hbond = (index in polar_h and other in acceptor) or (other in polar_h and index in acceptor)
            limit = (HBOND_SCALE if hbond else CLASH_SCALE) * (radius[index] + radius[other])
            distance = float(np.linalg.norm(reference_position[index] - reference_position[other]))
            worst = min(worst, distance / limit)
    return worst


def _overall_min_contact(positions, radius, grid, neighbourhood, limit_for) -> float:
    """Closest non-bonded contact relative to its own clash limit (>= 1 means no clash)."""

    worst = math.inf
    for index, point in positions.items():
        excluded = neighbourhood(index)
        for other in grid.near(point):
            if other <= index or other in excluded:
                continue
            distance = float(np.linalg.norm(point - positions[other]))
            worst = min(worst, distance / (limit_for(index, other) * (radius[index] + radius[other])))
    return worst
