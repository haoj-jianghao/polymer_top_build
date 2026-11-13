"""Toolkit-independent model of the 2D monomer sketch.

The desktop GUI (``desktop.py``) draws this model on a Tk canvas and sends
mouse events to it; everything here is plain Python so it can be tested
without a display. Behaviour follows the original web sketcher:

* drag between two points draws a single bond, creating carbon atoms at free ends;
* click an atom to change its element (and choose it as a connection atom);
* click a bond to cycle single -> double -> triple -> single;
* bond-order tool: click a bond to make it single/double/triple, or drag to
  draw a bond of that order (bonds win over nearby atoms, so short bonds work);
* click empty space to add an atom of the current element;
* charge tool: click an atom to set its formal charge;
* stereo tool: draw or click a single bond to make it a wedge (1) or dash (6);
* delete an atom (with its bonds) or a bond; atoms are renumbered 1..N.

Canvas coordinates are pixels (y down); MOL coordinates are Ångström-like
units (y up) at ``SCALE`` pixels per unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ELEMENTS = ("C", "O", "N", "S", "H", "Cl", "Si", "P", "F", "Br", "I")
ELEMENT_COLORS = {
    "C": "#111111", "O": "#cf2f2f", "N": "#2356c4", "S": "#9a6b00",
    "H": "#64717b", "Cl": "#11824c", "Si": "#8a5a44", "P": "#db6f00",
    "F": "#5a9c3f", "Br": "#8b3a10", "I": "#5e2f8a",
}
BOND_ORDER_NAMES = {1: "single", 2: "double", 3: "triple"}
V2000_CHARGE_CODE = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
ATOM_HIT_RADIUS = 20.0
BOND_HIT_DISTANCE = 8.0
DRAG_THRESHOLD = 8.0
SCALE = 55.0  # pixels per MOL coordinate unit


@dataclass
class SketchAtom:
    id: int
    element: str
    x: float
    y: float
    formal_charge: int = 0


@dataclass
class SketchBond:
    a: int
    b: int
    order: int = 1
    stereo: int = 0


def charge_label(charge: int) -> str:
    if not charge:
        return ""
    magnitude = abs(charge)
    return ("" if magnitude == 1 else str(magnitude)) + ("+" if charge > 0 else "-")


def _distance_to_segment(px: float, py: float, a: SketchAtom, b: SketchAtom) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - a.x, py - a.y)
    t = max(0.0, min(1.0, ((px - a.x) * dx + (py - a.y) * dy) / length_sq))
    return math.hypot(px - (a.x + t * dx), py - (a.y + t * dy))


@dataclass
class SketchModel:
    atoms: list[SketchAtom] = field(default_factory=list)
    bonds: list[SketchBond] = field(default_factory=list)
    element: str = "C"
    tool: str = "atom"  # "atom" | "order" | "charge" | "stereo"
    bond_order: int = 1  # order used by the "order" tool
    formal_charge: int = 0
    bond_stereo: int = 0
    selected_atom: int | None = None
    selected_bond: SketchBond | None = None
    name: str = "Drawn_monomer"
    # {old id: new id} after an atom deletion renumbered the atoms; the GUI
    # uses it to follow the connection atoms (see take_renumbering).
    last_renumbering: dict[int, int] | None = None

    # -- queries -------------------------------------------------------------

    def atom(self, atom_id: int) -> SketchAtom | None:
        return next((atom for atom in self.atoms if atom.id == atom_id), None)

    def atom_at(self, x: float, y: float) -> SketchAtom | None:
        for atom in reversed(self.atoms):
            if math.hypot(atom.x - x, atom.y - y) <= ATOM_HIT_RADIUS:
                return atom
        return None

    def bond_at(self, x: float, y: float) -> SketchBond | None:
        for bond in reversed(self.bonds):
            a, b = self.atom(bond.a), self.atom(bond.b)
            if a and b and _distance_to_segment(x, y, a, b) <= BOND_HIT_DISTANCE:
                return bond
        return None

    def total_charge(self) -> int:
        return sum(atom.formal_charge for atom in self.atoms)

    def summary(self) -> str:
        return f"{self.name}    atoms={len(self.atoms)}    bonds={len(self.bonds)}    charge={self.total_charge()}"

    # -- edits ---------------------------------------------------------------

    def add_atom(self, element: str, x: float, y: float) -> SketchAtom:
        atom = SketchAtom(len(self.atoms) + 1, element, x, y)
        self.atoms.append(atom)
        return atom

    def add_bond(self, a: SketchAtom, b: SketchAtom) -> None:
        if a is None or b is None or a.id == b.id:
            return
        existing = next(
            (bond for bond in self.bonds if {bond.a, bond.b} == {a.id, b.id}),
            None,
        )
        if existing is not None:
            if self.tool == "order":
                existing.order = self.bond_order
                if existing.order != 1:
                    existing.stereo = 0
            elif self.tool == "stereo":
                if existing.order == 1:
                    existing.stereo = self.bond_stereo
            else:
                self.cycle_bond_order(existing)
        else:
            order = self.bond_order if self.tool == "order" else 1
            self.bonds.append(SketchBond(a.id, b.id, order, self.bond_stereo if self.tool == "stereo" else 0))

    @staticmethod
    def cycle_bond_order(bond: SketchBond) -> None:
        bond.order = 1 if bond.order >= 3 else bond.order + 1
        bond.stereo = 0

    def renumber(self) -> dict[int, int]:
        """Renumber atoms 1..N in drawing order; returns {old id: new id}."""

        old_to_new = {}
        for index, atom in enumerate(self.atoms, start=1):
            old_to_new[atom.id] = index
            atom.id = index
        self.bonds = [
            SketchBond(old_to_new[bond.a], old_to_new[bond.b], bond.order, bond.stereo)
            for bond in self.bonds
            if bond.a in old_to_new and bond.b in old_to_new
        ]
        return old_to_new

    def erase_atom(self, atom: SketchAtom) -> dict[int, int]:
        """Delete an atom and its bonds; returns {old id: new id} of the remaining atoms."""

        self.atoms = [item for item in self.atoms if item.id != atom.id]
        self.bonds = [bond for bond in self.bonds if atom.id not in (bond.a, bond.b)]
        self.selected_atom = None
        self.selected_bond = None
        self.last_renumbering = self.renumber()
        return self.last_renumbering

    def take_renumbering(self) -> dict[int, int] | None:
        """The {old id: new id} map of the last atom deletion, once (None if none since)."""

        mapping, self.last_renumbering = self.last_renumbering, None
        return mapping

    def erase_bond(self, bond: SketchBond) -> None:
        self.bonds = [item for item in self.bonds if item is not bond]
        self.selected_bond = None

    def delete_selected(self) -> str:
        if self.selected_atom is not None and self.atom(self.selected_atom):
            self.erase_atom(self.atom(self.selected_atom))
            return "Deleted the selected atom and its bonds."
        if self.selected_bond is not None:
            self.erase_bond(self.selected_bond)
            return "Deleted the selected bond."
        return "Select an atom or bond first, then delete it."

    def clear(self) -> None:
        self.atoms, self.bonds = [], []
        self.selected_atom, self.selected_bond = None, None

    # -- mouse gestures --------------------------------------------------------

    def release(self, start: tuple[float, float], end: tuple[float, float]) -> tuple[str, int | None]:
        """Apply a press at ``start`` released at ``end``.

        Returns (status message, id of an atom that was clicked, i.e. a
        candidate connection atom, or None).
        """

        start_atom = self.atom_at(*start)
        end_atom = self.atom_at(*end)
        dragged = math.hypot(end[0] - start[0], end[1] - start[1]) > DRAG_THRESHOLD

        if self.tool == "charge":
            if end_atom is None:
                return "Choose an existing atom to assign its formal charge.", None
            end_atom.formal_charge = self.formal_charge
            self.selected_atom, self.selected_bond = end_atom.id, None
            label = charge_label(self.formal_charge) or "0"
            return f"Assigned formal charge {label} to atom {end_atom.id}.", None

        if self.tool == "order" and not dragged:
            bond = self.bond_at(*end)
            if bond is None:
                return "Click a bond to set its order, or drag to draw a new bond of that order.", None
            bond.order = self.bond_order
            if bond.order != 1:
                bond.stereo = 0
            self.selected_bond, self.selected_atom = bond, None
            return f"Bond {bond.a}-{bond.b} is now {BOND_ORDER_NAMES[bond.order]}.", None

        if self.tool == "stereo" and not dragged:
            bond = self.bond_at(*end)
            if bond is None:
                return "Choose an existing single bond, or drag to draw a stereochemical bond.", None
            if bond.order != 1:
                return "Wedge and dash styles can only be assigned to single bonds.", None
            bond.stereo = self.bond_stereo
            self.selected_bond, self.selected_atom = bond, None
            return "Assigned the bond style.", None

        if dragged:
            a = start_atom or self.add_atom("C", *start)
            b = end_atom or self.add_atom("C", *end)
            self.add_bond(a, b)
            self.selected_atom, self.selected_bond = b.id, None
            return "Drew a bond.", None
        if end_atom is not None:
            end_atom.element = self.element
            self.selected_atom, self.selected_bond = end_atom.id, None
            return f"Atom {end_atom.id} is {end_atom.element}.", end_atom.id
        bond = self.bond_at(*end)
        if bond is not None:
            self.cycle_bond_order(bond)
            self.selected_bond, self.selected_atom = bond, None
            return f"Bond {bond.a}-{bond.b} is now {BOND_ORDER_NAMES[bond.order]}.", None
        atom = self.add_atom(self.element, *end)
        self.selected_atom, self.selected_bond = atom.id, None
        return f"Added atom {atom.id} ({atom.element}).", None

    def erase_at(self, x: float, y: float) -> str:
        atom = self.atom_at(x, y)
        if atom is not None:
            self.erase_atom(atom)
            return "Deleted the atom and its bonds."
        bond = self.bond_at(x, y)
        if bond is not None:
            self.erase_bond(bond)
            return "Deleted the bond."
        return ""

    # -- MOL/SDF ---------------------------------------------------------------

    def to_molblock(self, canvas_height: float = 520.0) -> str:
        if not self.atoms:
            raise ValueError("Draw at least one atom first.")
        lines = [self.name, "  polymer_gui", ""]
        lines.append(f"{len(self.atoms):>3}{len(self.bonds):>3}  0  0  0  0            999 V2000")
        for atom in self.atoms:
            x = atom.x / SCALE
            y = (canvas_height - atom.y) / SCALE
            lines.append(f"{x:10.4f}{y:10.4f}{0.0:10.4f} {atom.element:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
        for bond in self.bonds:
            lines.append(f"{bond.a:>3}{bond.b:>3}{bond.order:>3}{bond.stereo:>3}  0  0  0")
        charged = [atom for atom in self.atoms if atom.formal_charge]
        for start in range(0, len(charged), 8):
            chunk = charged[start : start + 8]
            lines.append(
                f"M  CHG{len(chunk):>3}" + "".join(f"{atom.id:>4}{atom.formal_charge:>4}" for atom in chunk)
            )
        lines += ["M  END", "$$$$"]
        return "\n".join(lines) + "\n"

    def load_molblock(self, text: str, width: float = 900.0, height: float = 520.0) -> None:
        """Load a V2000 MOL/SDF record, scaled and centred in a ``width`` x ``height`` canvas."""

        lines = text.splitlines()
        if len(lines) < 4:
            raise ValueError("The MOL/SDF text is too short.")
        try:
            atom_count, bond_count = int(lines[3][0:3]), int(lines[3][3:6])
        except ValueError as exc:
            raise ValueError("Only V2000 MOL/SDF counts lines can be loaded into the sketcher.") from exc
        atoms, bonds = [], []
        for index in range(atom_count):
            line = lines[4 + index]
            code = line[36:39].strip()
            atoms.append(
                SketchAtom(
                    index + 1,
                    line[31:34].strip(),
                    float(line[0:10]),
                    float(line[10:20]),
                    V2000_CHARGE_CODE.get(int(code) if code else 0, 0),
                )
            )
        for index in range(bond_count):
            line = lines[4 + atom_count + index]
            stereo = line[9:12].strip()
            bonds.append(SketchBond(int(line[0:3]), int(line[3:6]), int(line[6:9]) or 1, int(stereo) if stereo else 0))
        for line in lines[4 + atom_count + bond_count :]:
            if not line.startswith("M  CHG"):
                continue
            fields = line.split()
            for pair in range(int(fields[2])):
                atom_index, charge = int(fields[3 + 2 * pair]), int(fields[4 + 2 * pair])
                if 1 <= atom_index <= len(atoms):
                    atoms[atom_index - 1].formal_charge = charge
        # MOL units (y up) -> canvas pixels (y down), centred
        if atoms:
            xs = [atom.x for atom in atoms]
            ys = [atom.y for atom in atoms]
            centre_x, centre_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
            scale = min(SCALE, 0.8 * min(width, height) / span)
            for atom in atoms:
                atom.x, atom.y = width / 2 + (atom.x - centre_x) * scale, height / 2 - (atom.y - centre_y) * scale
        self.atoms, self.bonds = atoms, bonds
        self.name = lines[0].strip() or "Drawn_monomer"
        self.selected_atom, self.selected_bond = None, None
