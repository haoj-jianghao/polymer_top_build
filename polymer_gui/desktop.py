#!/usr/bin/env python
"""Desktop GUI (Tkinter) of the polymer GROMACS builder.

    python polymer_gui/desktop.py

Draw a monomer, pick the previous/next connection atoms, choose the
force field and build settings, and generate the GROMACS files. The backend
(``polymer_from_oligomer``) is called directly in a background thread, so
no web browser or server is involved and the window stays responsive while
the reference oligomer is parameterized. Tkinter ships with Python; nothing
else is needed for the GUI itself.
"""

from __future__ import annotations

import json
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
for path in (str(GUI_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import polymer_from_oligomer as backend  # noqa: E402
from sketch_model import ELEMENT_COLORS, ELEMENTS, SketchModel, charge_label  # noqa: E402

APP_TITLE = "Polymer GROMACS Builder"
CANVAS_WIDTH, CANVAS_HEIGHT = 900, 520
GRID_STEP = 32
PEG_EXAMPLE = PROJECT_ROOT / "examples" / "PEG_repeat_monomer.sdf"
ENGINE_LABELS = {
    "opls": "OPLS-AA/2020 (bundled opls-ff-lite)",
    "sage": "OpenFF Sage 2.3 (your OpenFF install)",
    "gaff2": "GAFF2 / AM1-BCC (your AmberTools install)",
}
LAYOUT_LABELS = {"random-walk": "3D random walk of monomers", "2d": "Flat 2D layout"}
END_GROUP_LABELS = {
    "H": "H (hydrogen, default)",
    "F": "F (e.g. CF2 end -> CF3)",
    "Cl": "Cl",
    "CH3": "CH3 (methyl)",
    "CF3": "CF3 (trifluoromethyl)",
    "OH": "OH (hydroxyl)",
    "OCH3": "OCH3 (methoxy)",
    "NH2": "NH2 (amine)",
}
PREVIOUS_COLOR, NEXT_COLOR = "#1a8f3c", "#d9730d"


def open_path(path: Path) -> None:
    """Open a file or folder with the desktop's default application (Windows, macOS, Linux, WSL)."""

    path = Path(path)
    if platform.system() == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
        return
    if "microsoft" in platform.release().lower():  # WSL: hand the path to Windows
        if shutil.which("wslview"):
            subprocess.Popen(["wslview", str(path)])
            return
        if shutil.which("explorer.exe") and shutil.which("wslpath"):
            windows_path = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True).stdout.strip()
            subprocess.Popen(["explorer.exe", windows_path])
            return
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(path)])
        return
    raise RuntimeError(f"No program found to open {path}")


class PolymerBuilderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.model = SketchModel()
        self.press: tuple[float, float] | None = None
        self.drag_point: tuple[float, float] | None = None
        self.connect_next_is_previous = True
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.last_outdir: Path | None = None

        root.title(APP_TITLE)
        root.minsize(1100, 650)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Tool.TRadiobutton", padding=(6, 3))
        style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"))

        self._build_variables()
        self._build_menu()
        self._build_layout()
        self._bind_canvas()
        self._set_tool("atom:C")
        self.redraw()
        # A finished background job wakes the GUI by writing one byte to a pipe
        # that Tk watches (an OS-level event: no Tk call from the worker thread,
        # which is not thread-safe, and no timer). The timer poll is a fallback,
        # e.g. on Windows where Tk has no file handlers; Tk 8.6 timers follow the
        # wall clock and can stall when the system clock jumps (seen under WSL).
        self._wake_write: int | None = None
        if os.name != "nt" and hasattr(self.root, "createfilehandler"):
            read_end, self._wake_write = os.pipe()
            os.set_blocking(read_end, False)

            def on_wake(fd: int, _mask: int) -> None:
                try:
                    os.read(fd, 1024)
                except BlockingIOError:
                    pass
                self._drain_events()

            self.root.createfilehandler(read_end, tk.READABLE, on_wake)
        self.root.after(250, self._poll_events)

    # ------------------------------------------------------------------ setup

    def _build_variables(self) -> None:
        self.tool_var = tk.StringVar(value="atom:C")
        self.previous_var = tk.IntVar(value=1)
        self.next_var = tk.IntVar(value=3)
        self.reference_dp_var = tk.IntVar(value=5)
        self.dp_var = tk.IntVar(value=12)
        self.job_var = tk.StringVar(value="polymer_job")
        self.outdir_var = tk.StringVar(value=str(PROJECT_ROOT / "outputs"))
        self.engine_var = tk.StringVar(value=ENGINE_LABELS[backend.DEFAULT_FF_ENGINE])
        self.sage_python_var = tk.StringVar()
        self.amber_python_var = tk.StringVar()
        self.layout_var = tk.StringVar(value=LAYOUT_LABELS[backend.DEFAULT_CHAIN_LAYOUT])
        self.head_group_var = tk.StringVar(value=END_GROUP_LABELS[backend.DEFAULT_END_GROUP])
        self.tail_group_var = tk.StringVar(value=END_GROUP_LABELS[backend.DEFAULT_END_GROUP])
        self.seed_var = tk.IntVar(value=0)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Draw a monomer, or load the PEG example.")
        self.summary_var = tk.StringVar(value="")
        for variable in (self.previous_var, self.next_var):
            variable.trace_add("write", lambda *_: self.redraw())

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New drawing", command=self.clear_drawing, accelerator="Ctrl+N")
        file_menu.add_command(label="Open MOL/SDF...", command=self.open_molfile, accelerator="Ctrl+O")
        file_menu.add_command(label="Save drawing as MOL...", command=self.save_molfile, accelerator="Ctrl+S")
        file_menu.add_command(label="Load PEG example", command=self.load_peg_example)
        file_menu.add_separator()
        file_menu.add_command(label="Open output folder", command=self.open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy, accelerator="Ctrl+Q")
        menu.add_cascade(label="File", menu=file_menu)
        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Delete selected", command=self.delete_selected, accelerator="Del")
        edit_menu.add_command(label="Clear drawing", command=self.clear_drawing)
        menu.add_cascade(label="Edit", menu=edit_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Drawing help", command=self.show_help)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu)
        self.root.bind_all("<Control-n>", lambda _e: self.clear_drawing())
        self.root.bind_all("<Control-o>", lambda _e: self.open_molfile())
        self.root.bind_all("<Control-s>", lambda _e: self.save_molfile())
        self.root.bind_all("<Control-q>", lambda _e: self.root.destroy())

    def _build_layout(self) -> None:
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        # left: tools + canvas
        left = ttk.Frame(main)
        main.add(left, weight=3)
        tools = ttk.Frame(left, padding=(0, 0, 6, 0))
        tools.pack(side=tk.LEFT, fill=tk.Y)
        self._build_tools(tools)
        drawing = ttk.Frame(left)
        drawing.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(drawing, textvariable=self.summary_var, foreground="#33424e").pack(anchor=tk.W, pady=(0, 3))
        self.canvas = tk.Canvas(
            drawing, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, background="#fdfefe",
            highlightthickness=1, highlightbackground="#b8c6d4", cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # right: settings / MOL text / output
        right = ttk.Notebook(main)
        main.add(right, weight=2)
        self.notebook = right
        self._build_settings_tab(right)
        self._build_mol_tab(right)
        self._build_output_tab(right)

        status = ttk.Frame(self.root, padding=(6, 3))
        status.pack(fill=tk.X)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_tools(self, parent: ttk.Frame) -> None:
        def group(title: str) -> ttk.LabelFrame:
            frame = ttk.LabelFrame(parent, text=title, padding=4)
            frame.pack(fill=tk.X, pady=(0, 6))
            return frame

        atoms = group("Atoms")
        for index, element in enumerate(ELEMENTS):
            ttk.Radiobutton(
                atoms, text=element, value=f"atom:{element}", variable=self.tool_var,
                style="Tool.TRadiobutton", command=lambda: self._set_tool(self.tool_var.get()),
            ).grid(row=index // 3, column=index % 3, sticky=tk.EW)

        orders = group("Bond order")
        for column, (label, value) in enumerate((("Single", 1), ("Double", 2), ("Triple", 3))):
            ttk.Radiobutton(
                orders, text=label, value=f"order:{value}", variable=self.tool_var,
                style="Tool.TRadiobutton", command=lambda: self._set_tool(self.tool_var.get()),
            ).grid(row=0, column=column, sticky=tk.EW)

        charges = group("Formal charge")
        for column, (label, value) in enumerate((("-1", -1), ("0", 0), ("+1", 1))):
            ttk.Radiobutton(
                charges, text=label, value=f"charge:{value}", variable=self.tool_var,
                style="Tool.TRadiobutton", command=lambda: self._set_tool(self.tool_var.get()),
            ).grid(row=0, column=column, sticky=tk.EW)

        stereo = group("Bond style")
        for column, (label, value) in enumerate((("Plain", 0), ("Wedge", 1), ("Dash", 6))):
            ttk.Radiobutton(
                stereo, text=label, value=f"stereo:{value}", variable=self.tool_var,
                style="Tool.TRadiobutton", command=lambda: self._set_tool(self.tool_var.get()),
            ).grid(row=0, column=column, sticky=tk.EW)

        connect = group("Connection atoms")
        ttk.Radiobutton(
            connect, text="Pick previous/next", value="connect", variable=self.tool_var,
            style="Tool.TRadiobutton", command=lambda: self._set_tool("connect"),
        ).pack(fill=tk.X)
        legend = ttk.Frame(connect)
        legend.pack(fill=tk.X, pady=(3, 0))
        tk.Label(legend, text="* previous", foreground=PREVIOUS_COLOR).pack(side=tk.LEFT)
        tk.Label(legend, text="* next", foreground=NEXT_COLOR).pack(side=tk.LEFT, padx=(8, 0))

        actions = group("Drawing")
        ttk.Button(actions, text="PEG example", command=self.load_peg_example).pack(fill=tk.X)
        ttk.Button(actions, text="Delete selected", command=self.delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="New", command=self.clear_drawing).pack(fill=tk.X)

    def _build_settings_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="Build settings")
        tab.columnconfigure(1, weight=1)
        row = 0

        def add(label: str, widget: tk.Widget, button: tk.Widget | None = None) -> None:
            nonlocal row
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky=tk.W, pady=2, padx=(0, 6))
            widget.grid(row=row, column=1, sticky=tk.EW, pady=2)
            if button is not None:
                button.grid(row=row, column=2, padx=(4, 0), pady=2)
            row += 1

        add("Previous connection atom", ttk.Spinbox(tab, from_=1, to=999, textvariable=self.previous_var, width=6))
        add("Next connection atom", ttk.Spinbox(tab, from_=1, to=999, textvariable=self.next_var, width=6))
        add("Reference oligomer size", ttk.Spinbox(tab, from_=2, to=20, textvariable=self.reference_dp_var, width=6))
        add("Polymer DP", ttk.Spinbox(tab, from_=1, to=100000, textvariable=self.dp_var, width=8))
        add("Force field", ttk.Combobox(tab, textvariable=self.engine_var, values=list(ENGINE_LABELS.values()), state="readonly"))
        add(
            "OpenFF Python (Sage)",
            ttk.Entry(tab, textvariable=self.sage_python_var),
            ttk.Button(tab, text="...", width=3, command=lambda: self._browse_python(self.sage_python_var)),
        )
        add(
            "AmberTools Python (GAFF2)",
            ttk.Entry(tab, textvariable=self.amber_python_var),
            ttk.Button(tab, text="...", width=3, command=lambda: self._browse_python(self.amber_python_var)),
        )
        ttk.Label(tab, text="Leave blank to auto-detect an installed environment.", foreground="#5b6b77").grid(
            row=row, column=1, sticky=tk.W
        )
        row += 1
        add("Starting structure", ttk.Combobox(tab, textvariable=self.layout_var, values=list(LAYOUT_LABELS.values()), state="readonly"))
        for label, variable in (("Head end group (on previous atom)", self.head_group_var),
                                ("Tail end group (on next atom)", self.tail_group_var)):
            add(label, ttk.Combobox(tab, textvariable=variable, values=list(END_GROUP_LABELS.values()), state="readonly"))
        add("Random-walk seed", ttk.Spinbox(tab, from_=0, to=10**6, textvariable=self.seed_var, width=8))
        add("Job name", ttk.Entry(tab, textvariable=self.job_var))
        add(
            "Output folder",
            ttk.Entry(tab, textvariable=self.outdir_var),
            ttk.Button(tab, text="...", width=3, command=self._browse_outdir),
        )
        ttk.Checkbutton(
            tab,
            text="Dry run: do not parameterize, write placeholder files only (not for simulation)",
            variable=self.dry_run_var,
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))
        row += 1
        buttons = ttk.Frame(tab)
        buttons.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        ttk.Button(buttons, text="Check engine", command=self.check_engine).pack(side=tk.LEFT)
        self.build_button = ttk.Button(buttons, text="Generate GROMACS files", style="Primary.TButton", command=self.build)
        self.build_button.pack(side=tk.RIGHT)

    def _build_mol_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="MOL/SDF")
        self.mol_text = tk.Text(tab, height=20, width=60, font=("TkFixedFont", 9), undo=True)
        self.mol_text.pack(fill=tk.BOTH, expand=True)
        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(buttons, text="Update from drawing", command=self.update_mol_text).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Load text into drawing", command=self.load_mol_text).pack(side=tk.LEFT, padx=6)

    def _build_output_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="Output")
        self.output_tab = tab
        self.log = tk.Text(tab, height=18, width=60, font=("TkFixedFont", 9), wrap=tk.WORD, state=tk.DISABLED)
        scroll = ttk.Scrollbar(tab, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        files_label = ttk.Label(tab, text="Files")
        self.files = tk.Listbox(tab, height=8)
        buttons = ttk.Frame(tab)
        ttk.Button(buttons, text="Open output folder", command=self.open_output_folder).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Open selected file", command=self.open_selected_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Clear log", command=lambda: self._set_log("")).pack(side=tk.RIGHT)
        tab.rowconfigure(0, weight=3)
        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)
        self.log.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        files_label.grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.files.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)
        buttons.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        self.files.bind("<Double-Button-1>", lambda _e: self.open_selected_file())

    def _bind_canvas(self) -> None:
        canvas = self.canvas
        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<Button-3>", self._on_erase)
        if platform.system() == "Darwin":
            canvas.bind("<Button-2>", self._on_erase)
            canvas.bind("<Control-Button-1>", self._on_erase)
        for key in ("<Delete>", "<BackSpace>"):
            canvas.bind(key, lambda _e: self.delete_selected())
        canvas.bind("<Configure>", lambda _e: self.redraw())

    # ------------------------------------------------------------ tool state

    def _set_tool(self, value: str) -> None:
        self.tool_var.set(value)
        kind, _, argument = value.partition(":")
        model = self.model
        if kind == "atom":
            model.tool, model.element = "atom", argument
            self.set_status(f"Atom tool ({argument}): drag to draw bonds, click an atom to change its element.")
        elif kind == "order":
            model.tool, model.bond_order = "order", int(argument)
            name = {1: "single", 2: "double", 3: "triple"}[model.bond_order]
            self.set_status(f"Bond-order tool ({name}): click a bond to make it {name}, or drag to draw a {name} bond.")
        elif kind == "charge":
            model.tool, model.formal_charge = "charge", int(argument)
            if model.selected_atom is not None and model.atom(model.selected_atom):
                model.atom(model.selected_atom).formal_charge = model.formal_charge
                self.redraw()
            self.set_status(f"Charge tool: click an atom to set its formal charge to {charge_label(int(argument)) or '0'}.")
        elif kind == "stereo":
            model.tool, model.bond_stereo = "stereo", int(argument)
            self.set_status("Bond-style tool: draw a bond from its narrow end, or click a single bond.")
        elif kind == "connect":
            model.tool = "connect"
            self.connect_next_is_previous = True
            self.set_status("Click the atom bonded to the PREVIOUS monomer, then the atom bonded to the NEXT monomer.")

    # --------------------------------------------------------------- canvas

    def _on_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self.press = (event.x, event.y)
        self.drag_point = None

    def _on_drag(self, event: tk.Event) -> None:
        if self.press is None or self.model.tool in ("charge", "connect"):
            return
        if math.hypot(event.x - self.press[0], event.y - self.press[1]) > 8:
            self.drag_point = (event.x, event.y)
            self.redraw()

    def _on_release(self, event: tk.Event) -> None:
        if self.press is None:
            return
        start, end = self.press, (event.x, event.y)
        self.press = self.drag_point = None
        if self.model.tool == "connect":
            atom = self.model.atom_at(*end)
            if atom is None:
                self.set_status("Click an atom to choose it as a connection atom.")
            elif self.connect_next_is_previous:
                self.previous_var.set(atom.id)
                self.connect_next_is_previous = False
                self.set_status(f"Previous connection atom: {atom.id}. Now click the next connection atom.")
            else:
                self.next_var.set(atom.id)
                self.connect_next_is_previous = True
                self.set_status(f"Connection atoms: previous {self.previous_var.get()}, next {atom.id}.")
            self.redraw()
            return
        message, _ = self.model.release(start, end)
        self.set_status(message)
        self.redraw()

    def _on_erase(self, event: tk.Event) -> None:
        message = self.model.erase_at(event.x, event.y)
        if message:
            self.set_status(self._follow_renumbering(message))
        self.redraw()

    def _follow_renumbering(self, message: str) -> str:
        """Keep the connection atoms on the same drawn atoms after a deletion renumbered them.

        A connection atom that was itself deleted is cleared (the user has to
        pick it again). Returns ``message`` with a note of what changed.
        """

        mapping = self.model.take_renumbering()
        if mapping is None:
            return message
        notes = []
        for role, variable in (("previous", self.previous_var), ("next", self.next_var)):
            old = self._safe_int(variable)
            if old is None:
                continue
            new = mapping.get(old)
            if new is None:
                variable.set("")
                notes.append(f"the {role} connection atom was deleted; pick it again")
            elif new != old:
                variable.set(new)
                notes.append(f"{role} connection atom {old} is now atom {new}")
        return message + (" (" + "; ".join(notes) + ")" if notes else "")

    def _clear_connection_atoms(self) -> None:
        self.previous_var.set("")
        self.next_var.set("")

    def redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), CANVAS_WIDTH), max(canvas.winfo_height(), CANVAS_HEIGHT)
        for x in range(GRID_STEP, width, GRID_STEP):
            canvas.create_line(x, 0, x, height, fill="#eef3f7")
        for y in range(GRID_STEP, height, GRID_STEP):
            canvas.create_line(0, y, width, y, fill="#eef3f7")
        model = self.model
        for bond in model.bonds:
            a, b = model.atom(bond.a), model.atom(bond.b)
            if a is None or b is None:
                continue
            if bond is model.selected_bond:
                canvas.create_line(a.x, a.y, b.x, b.y, width=10, fill="#bcd8ef", capstyle=tk.ROUND)
            self._draw_bond(a, b, bond.order, bond.stereo)
        if self.press is not None and self.drag_point is not None:
            canvas.create_line(*self.press, *self.drag_point, fill="#1d6dab", width=2, dash=(6, 4))
        previous, nxt = self._safe_int(self.previous_var), self._safe_int(self.next_var)
        for atom in model.atoms:
            for target, color, letter in ((previous, PREVIOUS_COLOR, "prev"), (nxt, NEXT_COLOR, "next")):
                if atom.id == target:
                    canvas.create_oval(atom.x - 17, atom.y - 17, atom.x + 17, atom.y + 17, outline=color, width=3)
                    canvas.create_text(atom.x, atom.y - 26, text=letter, fill=color, font=("TkDefaultFont", 9, "bold"))
            if atom.id == model.selected_atom:
                canvas.create_rectangle(atom.x - 21, atom.y - 18, atom.x + 21, atom.y + 18, outline="#1d6dab", fill="#e3eff9")
            canvas.create_rectangle(atom.x - 12, atom.y - 10, atom.x + 12, atom.y + 9, outline="", fill="#fdfefe")
            canvas.create_text(atom.x, atom.y, text=atom.element, fill=ELEMENT_COLORS.get(atom.element, "#111111"),
                               font=("TkDefaultFont", 13, "bold"))
            if atom.formal_charge:
                canvas.create_text(atom.x + 15, atom.y - 12, text=charge_label(atom.formal_charge), fill="#8b1e3f",
                                   font=("TkDefaultFont", 10, "bold"))
            canvas.create_text(atom.x + 17, atom.y + 16, text=str(atom.id), fill="#0b5cab", font=("TkDefaultFont", 9))
        self.summary_var.set(self._summary_text())

    def _draw_bond(self, a, b, order: int, stereo: int) -> None:
        canvas = self.canvas
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / length * 4, dx / length * 4
        if order == 1 and stereo == 1:  # wedge: narrow at a, wide at b
            canvas.create_polygon(a.x, a.y, b.x + 2 * ox, b.y + 2 * oy, b.x - 2 * ox, b.y - 2 * oy, fill="#111111", outline="")
        elif order == 1 and stereo == 6:  # hashed wedge
            for step in range(1, 8):
                fraction = step / 8
                cx, cy, w = a.x + dx * fraction, a.y + dy * fraction, fraction * 2
                canvas.create_line(cx - ox * w, cy - oy * w, cx + ox * w, cy + oy * w, width=2, fill="#111111")
        elif order == 2:
            for sign in (1, -1):
                canvas.create_line(a.x + sign * ox, a.y + sign * oy, b.x + sign * ox, b.y + sign * oy, width=2, fill="#111111")
        elif order == 3:
            canvas.create_line(a.x, a.y, b.x, b.y, width=2, fill="#111111")
            for sign in (1.6, -1.6):
                canvas.create_line(a.x + sign * ox, a.y + sign * oy, b.x + sign * ox, b.y + sign * oy, width=2, fill="#111111")
        else:
            canvas.create_line(a.x, a.y, b.x, b.y, width=2, fill="#111111")

    def _summary_text(self) -> str:
        model = self.model
        if not model.atoms:
            return "Empty drawing"
        try:
            monomer = self._monomer_molecule()
            base = f"Monomer {monomer.formula()}  MW {monomer.molecular_weight():.2f}"
        except Exception:
            base = "Monomer"
        return f"{base}   atoms {len(model.atoms)}   bonds {len(model.bonds)}   charge {model.total_charge():+d}"

    def _monomer_molecule(self) -> backend.Molecule:
        """The drawn monomer with valence hydrogens (for the formula shown above the canvas)."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monomer.sdf"
            path.write_text(self.model.to_molblock(CANVAS_HEIGHT))
            molecule = backend.read_sdf(path)
        hydrogenated = backend.clone_molecule(molecule)
        backend.add_valence_hydrogens(hydrogenated)
        return hydrogenated

    @staticmethod
    def _safe_int(variable: tk.Variable) -> int | None:
        try:
            return int(variable.get())
        except (tk.TclError, ValueError):
            return None

    # ------------------------------------------------------------ actions

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def delete_selected(self) -> None:
        self.set_status(self._follow_renumbering(self.model.delete_selected()))
        self.redraw()

    def clear_drawing(self) -> None:
        self.model.clear()
        self._clear_connection_atoms()
        self.mol_text.delete("1.0", tk.END)
        self.redraw()
        self.set_status("Cleared the drawing.")

    def load_molblock(self, text: str, source: str) -> None:
        width, height = max(self.canvas.winfo_width(), CANVAS_WIDTH), max(self.canvas.winfo_height(), CANVAS_HEIGHT)
        self.model.load_molblock(text, width, height)
        self.update_mol_text()
        self.redraw()
        self.set_status(f"Loaded {source}.")

    def load_peg_example(self) -> None:
        self.load_molblock(PEG_EXAMPLE.read_text(), "the PEG example monomer")
        self.previous_var.set(1)
        self.next_var.set(3)
        self.job_var.set("PEG_example")

    def open_molfile(self) -> None:
        path = filedialog.askopenfilename(
            title="Open monomer", filetypes=[("MOL/SDF", "*.mol *.sdf *.sd"), ("All files", "*")]
        )
        if path:
            try:
                self.load_molblock(Path(path).read_text(), Path(path).name)
                self.job_var.set(Path(path).stem)
                self._clear_connection_atoms()
                self.set_status(f"Loaded {Path(path).name}. Now pick the previous and next connection atoms.")
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"Could not load {path}:\n{exc}")

    def save_molfile(self) -> None:
        try:
            text = self.model.to_molblock(CANVAS_HEIGHT)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        path = filedialog.asksaveasfilename(title="Save monomer", defaultextension=".mol", filetypes=[("MOL", "*.mol"), ("SDF", "*.sdf")])
        if path:
            Path(path).write_text(text)
            self.set_status(f"Saved {path}.")

    def update_mol_text(self) -> None:
        self.mol_text.delete("1.0", tk.END)
        if self.model.atoms:
            self.mol_text.insert("1.0", self.model.to_molblock(CANVAS_HEIGHT))

    def load_mol_text(self) -> None:
        try:
            self.load_molblock(self.mol_text.get("1.0", tk.END), "the MOL/SDF text")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not read the MOL/SDF text:\n{exc}")

    def _browse_python(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="Choose a Python interpreter")
        if path:
            variable.set(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Choose the output folder", initialdir=self.outdir_var.get())
        if path:
            self.outdir_var.set(path)

    def show_help(self) -> None:
        messagebox.showinfo(
            APP_TITLE,
            "Drawing\n"
            "- Drag to draw a bond (free ends become carbon atoms).\n"
            "- Click empty space to add an atom of the chosen element.\n"
            "- Click an atom to change its element.\n"
            "- Bond order tool: click a bond to make it single, double or triple\n"
            "  (with an element tool, clicking a bond cycles single -> double -> triple).\n"
            "- Right-click an atom or bond to delete it; Del deletes the selection.\n"
            "- Charge tool: click an atom. Bond style: draw a bond from its narrow end or click it.\n\n"
            "Connection atoms\n"
            "- 'Pick previous/next', then click the atom bonded to the previous monomer\n"
            "  and the atom bonded to the next monomer (green = previous, orange = next).",
        )

    # --------------------------------------------------------------- engine

    def _engine(self) -> str:
        return next(key for key, label in ENGINE_LABELS.items() if label == self.engine_var.get())

    def _engine_settings(self) -> dict:
        settings = {}
        if self.sage_python_var.get().strip():
            settings["sage_python"] = str(Path(self.sage_python_var.get().strip()).expanduser())
        if self.amber_python_var.get().strip():
            settings["amber_python"] = str(Path(self.amber_python_var.get().strip()).expanduser())
        return settings

    def check_engine(self) -> None:
        engine, settings = self._engine(), self._engine_settings()
        self._start_job(f"Checking the {engine} engine...", self._check_engine_job, engine, settings)

    @staticmethod
    def _check_engine_job(engine: str, settings: dict) -> dict:
        lines = [f"Force field: {backend.FF_ENGINES[engine]}"]
        problems = backend.engine_problem_lines(engine, settings)
        if engine in ("sage", "gaff2"):
            interpreter, _ = backend.engine_module(engine).find_interpreter(settings)
            if interpreter is not None:
                lines.append(f"Python: {interpreter.python} ({interpreter.source})")
                for key in ("versions", "charges", "tools", "parmed"):
                    if key in interpreter.probe:
                        lines.append(f"{key}: {interpreter.probe[key]}")
        lines += problems
        lines.append("Ready." if not problems else "NOT ready.")
        return {"kind": "check", "text": "\n".join(lines), "ok": not problems}

    # ---------------------------------------------------------------- build

    def _build_parameters(self) -> dict:
        if not self.model.atoms:
            raise ValueError("Draw a monomer first.")
        previous, nxt = self._safe_int(self.previous_var), self._safe_int(self.next_var)
        count = len(self.model.atoms)
        if previous is None or nxt is None:
            raise ValueError("Pick the previous and next connection atoms first (\"Pick previous/next\" tool).")
        if not (1 <= previous <= count and 1 <= nxt <= count):
            raise ValueError(f"Connection atoms must be between 1 and {count}.")
        if previous == nxt:
            raise ValueError("The previous and next connection atoms must be different.")
        job = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.job_var.get().strip()) or "polymer_job"
        return {
            "molblock": self.model.to_molblock(CANVAS_HEIGHT),
            "outdir": Path(self.outdir_var.get()).expanduser() / job,
            "options": dict(
                previous_atom=previous,
                next_atom=nxt,
                reference_dp=int(self.reference_dp_var.get()),
                polymer_dp=int(self.dp_var.get()),
                run_external=not self.dry_run_var.get(),
                ff_engine=self._engine(),
                engine_settings=self._engine_settings(),
                chain_layout=next(key for key, label in LAYOUT_LABELS.items() if label == self.layout_var.get()),
                head_group=next(key for key, label in END_GROUP_LABELS.items() if label == self.head_group_var.get()),
                tail_group=next(key for key, label in END_GROUP_LABELS.items() if label == self.tail_group_var.get()),
                chain_seed=int(self.seed_var.get()),
            ),
        }

    def build(self) -> None:
        try:
            parameters = self._build_parameters()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        outdir = parameters["outdir"]
        if outdir.exists() and any(outdir.iterdir()):
            if not messagebox.askyesno(APP_TITLE, f"{outdir} already contains files. Overwrite them?"):
                return
        self._start_job(f"Building in {outdir}...", self._build_job, parameters)

    @staticmethod
    def _build_job(parameters: dict) -> dict:
        outdir: Path = parameters["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        monomer_sdf = outdir / "monomer_input.sdf"
        monomer_sdf.write_text(parameters["molblock"])
        options = backend.MonomerBuildOptions(outdir=outdir, **parameters["options"])
        result = backend.build_polymer_from_monomer(monomer_sdf, options)
        metadata = json.loads((outdir / "build_metadata.json").read_text())
        report_path = outdir / "PARAMETER_TRANSFER_REPORT.txt"
        lines = [
            f"Polymer {result.molecule.formula()}: {len(result.molecule.atoms)} atoms, {len(result.molecule.bonds)} bonds",
            f"Force field: {metadata.get('ff_engine_description', metadata.get('ff_engine'))}",
            f"Mode: {metadata.get('parameterization_mode')}; final charge {result.final_charge:+.6f}",
        ]
        lines.append(f"End groups: head {metadata.get('head_end_group', 'H')}, tail {metadata.get('tail_end_group', 'H')}")
        if metadata.get("leaving_hydrogens_removed"):
            atoms = ", ".join(str(index) for index in metadata["leaving_hydrogens_removed"])
            lines.append(
                f"Leaving hydrogens: drawn H atom(s) {atoms} on the connection atoms were removed to make "
                "room for the inter-unit bonds (chain ends get them back as end-cap hydrogens)."
            )
        if not options.run_external:
            lines.append("DRY RUN: placeholder files only (no force-field parameters); not for simulation.")
        if result.missing_tools:
            lines.append("Placeholder files only; missing: " + ", ".join(result.missing_tools))
            lines += backend.engine_problem_lines(options.ff_engine, options.engine_settings)
        if report_path.exists():
            lines += ["", report_path.read_text()]
        files = sorted({str(path) for path in result.files if Path(path).exists()})
        return {"kind": "build", "text": "\n".join(lines), "files": files, "outdir": str(outdir),
                "ok": not result.missing_tools}

    # ------------------------------------------------------ background jobs

    def _start_job(self, message: str, function, *arguments) -> None:
        if self.busy:
            messagebox.showinfo(APP_TITLE, "Please wait for the current job to finish.")
            return
        self.busy = True
        self.build_button.state(["disabled"])
        self.progress.start(12)
        self.set_status(message)

        def run() -> None:
            try:
                self.events.put(function(*arguments))
            except Exception as exc:
                self.events.put({"kind": "error", "text": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
            if self._wake_write is not None:
                try:
                    os.write(self._wake_write, b"x")
                except OSError:
                    pass  # the timer poll picks the result up

        threading.Thread(target=run, daemon=True).start()

    def _drain_events(self) -> None:
        try:
            while True:
                self._finish_job(self.events.get_nowait())
        except queue.Empty:
            pass

    def _poll_events(self) -> None:
        self._drain_events()
        self.root.after(250, self._poll_events)

    def _finish_job(self, event: dict) -> None:
        self.busy = False
        self.build_button.state(["!disabled"])
        self.progress.stop()
        self.notebook.select(self.output_tab)
        if event["kind"] == "error":
            self._append_log("ERROR\n" + event["text"] + "\n\n" + event["trace"])
            self.set_status("Failed: " + event["text"].splitlines()[0][:150])
            return
        self._append_log(event["text"])
        if event["kind"] == "build":
            self.last_outdir = Path(event["outdir"])
            self.files.delete(0, tk.END)
            for path in event["files"]:
                self.files.insert(tk.END, path)
            self.set_status(("Build finished: " if event["ok"] else "Placeholder files written: ") + event["outdir"])
        else:
            self.set_status("Engine ready." if event["ok"] else "Engine not ready; see the Output tab.")

    def _set_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, text)
        self.log.configure(state=tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        if self.log.get("1.0", tk.END).strip():
            self.log.insert(tk.END, "\n" + "-" * 60 + "\n")
        self.log.insert(tk.END, text.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def open_output_folder(self) -> None:
        folder = self.last_outdir or Path(self.outdir_var.get()).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            open_path(folder)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open {folder}:\n{exc}")

    def open_selected_file(self) -> None:
        selection = self.files.curselection()
        if not selection:
            return
        try:
            open_path(Path(self.files.get(selection[0])))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))


def _bring_to_front(root: tk.Tk) -> None:
    """Make sure the main window really appears on screen.

    Under WSLg (and some other compositors) a Tk window can be mapped on the
    X side yet stay stuck as a taskbar entry on Windows. Re-showing it after
    the event loop starts, placing it explicitly on the visible screen and
    briefly raising it above other windows gets the compositor to present it.
    """

    root.update_idletasks()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    width = min(max(root.winfo_reqwidth(), 1100), screen_w - 40)
    height = min(max(root.winfo_reqheight(), 650), screen_h - 80)
    x = max((screen_w - width) // 2, 0)
    y = max((screen_h - height) // 3, 0)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    # Drop "always on top" again so the window behaves normally afterwards.
    root.after(300, lambda: root.attributes("-topmost", False))


def main() -> None:
    root = tk.Tk()
    # Build the UI hidden, then show it once the event loop is running: a
    # window that is mapped before it is fully built is what WSLg tends to
    # leave stuck in the Windows taskbar.
    root.withdraw()
    PolymerBuilderApp(root)
    root.after(100, lambda: _bring_to_front(root))
    root.mainloop()


if __name__ == "__main__":
    main()
