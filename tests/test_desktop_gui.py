"""Tests of the Tkinter desktop GUI.

``SketchModelTests`` need no display. ``DesktopWindowTests`` open the real Tk
window and are skipped when no display is available.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "polymer_gui"))

from polymer_from_oligomer import read_sdf  # noqa: E402
from sketch_model import SketchModel  # noqa: E402


class SketchModelTests(unittest.TestCase):
    def draw_chain(self) -> SketchModel:
        model = SketchModel()
        model.release((100, 100), (160, 100))   # drag: two new C atoms and a bond
        model.release((160, 100), (220, 100))   # drag from atom 2: new atom 3
        return model

    def test_drag_creates_atoms_and_bonds(self):
        model = self.draw_chain()
        self.assertEqual([(a.id, a.element) for a in model.atoms], [(1, "C"), (2, "C"), (3, "C")])
        self.assertEqual([(b.a, b.b, b.order) for b in model.bonds], [(1, 2, 1), (2, 3, 1)])

    def test_click_gestures(self):
        model = self.draw_chain()
        model.element = "O"
        message, clicked = model.release((220, 100), (220, 100))       # click atom 3 -> O
        self.assertEqual((model.atom(3).element, clicked), ("O", 3))
        for expected in (2, 3, 1):                                     # click bond 1-2 cycles its order
            model.release((130, 100), (130, 100))
            self.assertEqual(model.bonds[0].order, expected)
        model.element = "N"
        model.release((300, 300), (300, 300))                          # click empty space -> new atom
        self.assertEqual(model.atoms[-1].element, "N")

    def test_charge_and_stereo_tools(self):
        model = self.draw_chain()
        model.tool, model.formal_charge = "charge", -1
        model.release((220, 100), (220, 100))
        self.assertEqual(model.atom(3).formal_charge, -1)
        model.tool, model.bond_stereo = "stereo", 1
        model.release((190, 100), (190, 100))                          # click bond 2-3 -> wedge
        self.assertEqual(model.bonds[1].stereo, 1)
        model.release((160, 100), (160, 40))                           # drawing with the tool makes a wedge bond
        self.assertEqual(model.bonds[-1].stereo, 1)

    def test_delete_renumbers_atoms_and_drops_bonds(self):
        model = self.draw_chain()
        model.add_atom("O", 400, 400)
        model.erase_at(160, 100)                                       # delete the middle atom
        self.assertEqual([(a.id, a.element) for a in model.atoms], [(1, "C"), (2, "C"), (3, "O")])
        self.assertEqual(model.bonds, [])

    def test_molblock_round_trip_through_the_backend_reader(self):
        model = self.draw_chain()
        model.atom(3).element, model.atom(3).formal_charge = "N", 1
        model.bonds[0].stereo = 6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monomer.sdf"
            path.write_text(model.to_molblock())
            molecule = read_sdf(path)
        self.assertEqual([a.element for a in molecule.atoms], ["C", "C", "N"])
        self.assertEqual(molecule.atoms[2].formal_charge, 1)
        self.assertEqual(molecule.bonds[0].stereo, 6)
        self.assertAlmostEqual(molecule.atoms[1].x - molecule.atoms[0].x, 60 / 55, places=3)

        reloaded = SketchModel()
        reloaded.load_molblock(model.to_molblock())
        self.assertEqual([(a.element, a.formal_charge) for a in reloaded.atoms], [("C", 0), ("C", 0), ("N", 1)])
        self.assertEqual([(b.a, b.b, b.order, b.stereo) for b in reloaded.bonds], [(1, 2, 1, 6), (2, 3, 1, 0)])

    def test_empty_drawing_cannot_be_exported(self):
        with self.assertRaises(ValueError):
            SketchModel().to_molblock()


def display_available() -> bool:
    if os.name != "nt" and sys.platform != "darwin" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import tkinter

        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(display_available(), "no display for Tk")
class DesktopWindowTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import desktop

        self.desktop = desktop
        self.root = tk.Tk()
        self.root.geometry("1400x800")
        self.app = desktop.PolymerBuilderApp(self.root)
        self.root.update()
        self.addCleanup(self.root.destroy)

    def gesture(self, x, y, x2=None, y2=None):
        canvas = self.app.canvas
        canvas.event_generate("<ButtonPress-1>", x=x, y=y)
        if x2 is not None:
            canvas.event_generate("<B1-Motion>", x=x2, y=y2)
        canvas.event_generate("<ButtonRelease-1>", x=x if x2 is None else x2, y=y if y2 is None else y2)
        self.root.update()

    def test_drawing_and_picking_connection_atoms_with_the_mouse(self):
        self.gesture(200, 200, 260, 200)
        self.gesture(260, 200, 320, 200)
        self.app._set_tool("atom:O")
        self.gesture(320, 200)
        self.app._set_tool("connect")
        self.gesture(200, 200)
        self.gesture(320, 200)
        self.assertEqual((self.app.previous_var.get(), self.app.next_var.get()), (1, 3))
        self.assertIn("C2H6O", self.app.summary_var.get())
        self.assertTrue(self.app.canvas.find_all())

    def test_placeholder_build_runs_in_the_background_and_reports(self):
        app = self.app
        app.load_peg_example()
        app.run_external_var.set(False)
        app.dp_var.set(6)
        with tempfile.TemporaryDirectory() as tmp:
            app.outdir_var.set(tmp)
            app.job_var.set("peg6")
            app.build()
            self.assertTrue(app.busy)
            deadline = time.monotonic() + 60
            while app.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.02)
            self.assertFalse(app.busy)
            self.assertTrue((Path(tmp) / "peg6" / "polymer.top").exists())
            self.assertGreater(app.files.size(), 0)
            self.assertIn("Polymer C12H26O6", app.log.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
