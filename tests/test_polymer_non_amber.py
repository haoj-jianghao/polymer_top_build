from pathlib import Path
import json
import math
import sys
import tempfile
import types
import unittest
from unittest import mock

from polymer_from_oligomer import (
    AmberTargetSpec,
    Atom,
    add_valence_hydrogens,
    Bond,
    BuildOptions,
    MonomerBuildOptions,
    Molecule,
    build_polymer_from_monomer,
    build_polymer_from_oligomer,
    build_reference_oligomer_from_monomer,
    detect_repeat_units,
    ensure_explicit_hydrogen_sdf,
    minimize_sdf_geometry,
    parse_index_list,
    run_ambertools_pipeline,
    read_mol2_atom_parameters,
    read_sdf,
    write_mol2,
    write_sdf,
)


ROOT = Path(__file__).resolve().parents[1]


class PolymerNonAmberTests(unittest.TestCase):
    def test_detects_peg5_repeats_from_first_unit(self):
        mol = read_sdf(ROOT / "examples" / "PEG_5mer.sdf")
        _, units = detect_repeat_units(mol, [2, 3, 4])
        self.assertEqual(units, [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]])

    def test_builds_dp12_peg_counts_and_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = build_polymer_from_oligomer(
                ROOT / "examples" / "PEG_5mer.sdf",
                BuildOptions(
                    repeat_atoms=parse_index_list("2,3,4"),
                    dp=12,
                    oligomer_charge=0,
                    repeat_charge=0,
                    end_charge=0,
                    outdir=tmp_path / "PEG12",
                    run_external=False,
                ),
            )
            self.assertEqual(result.molecule.formula(), "C24H50O13")
            self.assertEqual(len(result.molecule.atoms), 87)
            self.assertEqual(len(result.molecule.bonds), 86)
            self.assertLess(abs(result.final_charge), 1e-9)
            self.assertTrue((tmp_path / "PEG12" / "polymer.itp").exists())
            self.assertTrue((tmp_path / "PEG12" / "polymer.gro").exists())
            self.assertTrue((tmp_path / "PEG12" / "polymer.top").exists())

    def test_builds_branched_repeat_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            atoms = [
                Atom(1, "C", 0, 0),
                Atom(2, "C", 1.5, 0),
                Atom(3, "C", 1.5, 1.2),
                Atom(4, "C", 3.0, 0),
                Atom(5, "C", 4.5, 0),
                Atom(6, "C", 4.5, 1.2),
                Atom(7, "C", 6.0, 0),
                Atom(8, "C", 7.5, 0),
                Atom(9, "C", 7.5, 1.2),
            ]
            bonds = [
                Bond(1, 2),
                Bond(2, 3),
                Bond(2, 4),
                Bond(4, 5),
                Bond(5, 6),
                Bond(5, 7),
                Bond(7, 8),
                Bond(8, 9),
            ]
            sdf = tmp_path / "branched.sdf"
            write_sdf(Molecule("branched_trimer", atoms, bonds), sdf)

            mol = read_sdf(sdf)
            _, units = detect_repeat_units(mol, [1, 2, 3])
            self.assertEqual(units, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

            result = build_polymer_from_oligomer(
                sdf,
                BuildOptions(
                    repeat_atoms=[1, 2, 3],
                    dp=4,
                    outdir=tmp_path / "branched_DP4",
                    run_external=False,
                ),
            )
            self.assertEqual(result.molecule.formula(), "C12H26")
            self.assertEqual(len([atom for atom in result.molecule.atoms if atom.element != "H"]), 12)
            self.assertEqual(len(result.molecule.bonds), len(result.molecule.atoms) - 1)

    def test_builds_ring_containing_repeat_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            atoms = []
            bonds = []
            for ring_idx, x_offset in enumerate([0.0, 3.2]):
                base = ring_idx * 6
                for i in range(6):
                    atoms.append(
                        Atom(
                            base + i + 1,
                            "C",
                            x_offset + 1.1 * math.cos(i * math.pi / 3),
                            1.1 * math.sin(i * math.pi / 3),
                        )
                    )
                for i in range(6):
                    bonds.append(Bond(base + i + 1, base + ((i + 1) % 6) + 1, 2 if i % 2 == 0 else 1))
            bonds.append(Bond(1, 7, 1))
            sdf = tmp_path / "ring.sdf"
            write_sdf(Molecule("ring_dimer", atoms, bonds), sdf)

            mol = read_sdf(sdf)
            _, units = detect_repeat_units(mol, [1, 2, 3, 4, 5, 6])
            self.assertEqual(len(units), 2)
            self.assertEqual([set(unit) for unit in units], [set(range(1, 7)), set(range(7, 13))])

            result = build_polymer_from_oligomer(
                sdf,
                BuildOptions(
                    repeat_atoms=[1, 2, 3, 4, 5, 6],
                    dp=3,
                    outdir=tmp_path / "ring_DP3",
                    run_external=False,
                ),
            )
            heavy_atoms = [atom for atom in result.molecule.atoms if atom.element != "H"]
            heavy_bonds = [
                bond
                for bond in result.molecule.bonds
                if result.molecule.atom(bond.a).element != "H" and result.molecule.atom(bond.b).element != "H"
            ]
            self.assertEqual(len(heavy_atoms), 18)
            self.assertEqual(len(heavy_bonds), 20)
            self.assertTrue(any(bond.order == 2 for bond in heavy_bonds))

    def test_generates_hidden_reference_oligomer_from_monomer(self):
        monomer = Molecule(
            "peg_repeat",
            atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.45, 0), Atom(3, "O", 2.9, 0)],
            bonds=[Bond(1, 2), Bond(2, 3)],
        )
        oligomer, repeat_atoms = build_reference_oligomer_from_monomer(
            monomer,
            previous_atom=1,
            next_atom=3,
            reference_dp=5,
        )
        self.assertEqual(repeat_atoms, [1, 2, 3])
        self.assertEqual(len(oligomer.atoms), 15)
        self.assertEqual(len(oligomer.bonds), 14)
        self.assertEqual([bond.order for bond in oligomer.bonds[-4:]], [1, 1, 1, 1])

    def test_silicon_phosphorus_valence_and_mass_are_supported(self):
        molecule = Molecule(
            "silicon_phosphate",
            atoms=[
                Atom(1, "Si"),
                Atom(2, "C"), Atom(3, "C"), Atom(4, "C"), Atom(5, "C"),
                Atom(6, "P"),
                Atom(7, "O"), Atom(8, "O"), Atom(9, "O"), Atom(10, "O"),
            ],
            bonds=[
                Bond(1, 2), Bond(1, 3), Bond(1, 4), Bond(1, 5),
                Bond(6, 7, 2), Bond(6, 8), Bond(6, 9), Bond(6, 10),
            ],
        )
        add_valence_hydrogens(molecule)
        for center in (1, 6):
            neighbors = [
                molecule.atom(bond.b if bond.a == center else bond.a)
                for bond in molecule.bonds
                if center in (bond.a, bond.b)
            ]
            self.assertFalse(any(atom.element == "H" for atom in neighbors))
        self.assertGreater(molecule.molecular_weight(), 28.085 + 30.974)

    def test_wedge_and_dash_stereo_round_trip_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chiral_monomer.sdf"
            monomer = Molecule(
                "chiral_monomer",
                atoms=[
                    Atom(1, "C"), Atom(2, "C", 1.5, 0),
                    Atom(3, "N", -1.5, 0), Atom(4, "O", 0, 1.5),
                ],
                bonds=[Bond(1, 2, 1, 1), Bond(1, 3, 1, 6), Bond(1, 4)],
            )
            write_sdf(monomer, path)
            restored = read_sdf(path)
            self.assertEqual([bond.stereo for bond in restored.bonds], [1, 6, 0])
            oligomer, _ = build_reference_oligomer_from_monomer(
                restored, previous_atom=2, next_atom=3, reference_dp=3
            )
            self.assertEqual(sum(bond.stereo == 1 for bond in oligomer.bonds), 3)
            self.assertEqual(sum(bond.stereo == 6 for bond in oligomer.bonds), 3)

            result = build_polymer_from_monomer(
                path,
                MonomerBuildOptions(
                    previous_atom=2,
                    next_atom=3,
                    reference_dp=3,
                    polymer_dp=4,
                    outdir=Path(tmp) / "chiral_DP4",
                    run_external=False,
                ),
            )
            self.assertEqual(sum(bond.stereo == 1 for bond in result.molecule.bonds), 4)
            self.assertEqual(sum(bond.stereo == 6 for bond in result.molecule.bonds), 4)
            self.assertTrue(
                all(abs(atom.z) < 1e-12 for atom in result.molecule.atoms),
                "Generated hydrogens must not override a 2D wedge drawing.",
            )

    def test_builds_polymer_from_monomer_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "peg_repeat.sdf"
            write_sdf(
                Molecule(
                    "peg_repeat",
                    atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.45, 0), Atom(3, "O", 2.9, 0)],
                    bonds=[Bond(1, 2), Bond(2, 3)],
                ),
                monomer_sdf,
            )
            result = build_polymer_from_monomer(
                monomer_sdf,
                MonomerBuildOptions(
                    previous_atom=1,
                    next_atom=3,
                    reference_dp=5,
                    polymer_dp=12,
                    outdir=tmp_path / "PEG_repeat_DP12",
                    run_external=False,
                ),
            )
            self.assertEqual(result.molecule.formula(), "C24H50O12")
            self.assertEqual(len([atom for atom in result.molecule.atoms if atom.element != "H"]), 36)
            self.assertTrue((tmp_path / "PEG_repeat_DP12" / "reference_oligomer_from_monomer.sdf").exists())
            self.assertTrue((tmp_path / "PEG_repeat_DP12" / "reference_oligomer_for_antechamber.sdf").exists())
            self.assertTrue((tmp_path / "PEG_repeat_DP12" / "polymer.itp").exists())
            metadata = json.loads((tmp_path / "PEG_repeat_DP12" / "build_metadata.json").read_text())
            self.assertEqual(metadata["antechamber_source_sdf"], metadata["reference_oligomer_antechamber_sdf"])

    def test_user_monomer_writes_hydrogenated_reference_for_antechamber(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "user_monomer.sdf"
            monomer_sdf.write_text(
                """Drawn_monomer
  polymer_gui

  8  7  0  0  0  0            999 V2000
    0.8624    5.3472    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.8212    5.9297    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5935    5.2800    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.5256    5.9073    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.6468    4.4062    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.7945    3.9357    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.2593    3.7565    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.7679    3.1068    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  3  5  1  0  0  0  0
  5  6  1  0  0  0  0
  5  7  2  0  0  0  0
  6  8  1  0  0  0  0
M  END
$$$$
"""
            )
            result = build_polymer_from_monomer(
                monomer_sdf,
                MonomerBuildOptions(
                    previous_atom=1,
                    next_atom=4,
                    reference_dp=5,
                    polymer_dp=12,
                    outdir=tmp_path / "user_build",
                    run_external=False,
                ),
            )
            self.assertEqual(result.molecule.formula(), "C72H122O24")
            ref_heavy = read_sdf(tmp_path / "user_build" / "reference_oligomer_from_monomer.sdf")
            ref_ante = read_sdf(tmp_path / "user_build" / "reference_oligomer_for_antechamber.sdf")
            self.assertEqual(ref_heavy.formula(), "C30O10")
            self.assertEqual(ref_ante.formula(), "C30H52O10")
            metadata = json.loads((tmp_path / "user_build" / "build_metadata.json").read_text())
            self.assertEqual(metadata["antechamber_source_sdf"], metadata["reference_oligomer_antechamber_sdf"])

    def test_formal_charge_round_trips_through_v2000_m_chg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ammonium.sdf"
            molecule = Molecule(
                "charged",
                atoms=[Atom(1, "N", formal_charge=1), Atom(2, "Cl", formal_charge=-1)],
            )
            write_sdf(molecule, path)
            text = path.read_text()
            self.assertIn("M  CHG  2   1   1   2  -1", text)
            restored = read_sdf(path)
            self.assertEqual([atom.formal_charge for atom in restored.atoms], [1, -1])
            self.assertEqual(restored.total_formal_charge(), 0)

    def test_positive_nitrogen_valence_controls_hydrogen_count(self):
        three_coordinate = Molecule(
            "protonated_amine",
            atoms=[
                Atom(1, "N", formal_charge=1),
                Atom(2, "C"),
                Atom(3, "C"),
                Atom(4, "C"),
            ],
            bonds=[Bond(1, 2), Bond(1, 3), Bond(1, 4)],
        )
        add_valence_hydrogens(three_coordinate)
        nitrogen_hydrogens = [
            bond
            for bond in three_coordinate.bonds
            if 1 in (bond.a, bond.b)
            and three_coordinate.atom(bond.b if bond.a == 1 else bond.a).element == "H"
        ]
        self.assertEqual(len(nitrogen_hydrogens), 1)

        four_coordinate = Molecule(
            "quaternary_ammonium",
            atoms=[
                Atom(1, "N", formal_charge=1),
                Atom(2, "C"),
                Atom(3, "C"),
                Atom(4, "C"),
                Atom(5, "C"),
            ],
            bonds=[Bond(1, 2), Bond(1, 3), Bond(1, 4), Bond(1, 5)],
        )
        add_valence_hydrogens(four_coordinate)
        nitrogen_hydrogens = [
            bond
            for bond in four_coordinate.bonds
            if 1 in (bond.a, bond.b)
            and four_coordinate.atom(bond.b if bond.a == 1 else bond.a).element == "H"
        ]
        self.assertEqual(len(nitrogen_hydrogens), 0)


    def test_ionic_monomer_charge_is_inferred_for_reference_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "quaternary_repeat.sdf"
            write_sdf(
                Molecule(
                    "quaternary_repeat",
                    atoms=[
                        Atom(1, "N", formal_charge=1),
                        Atom(2, "C", 1.5, 0),
                        Atom(3, "C", -1.5, 0),
                        Atom(4, "C", 0, 1.5),
                        Atom(5, "C", 0, -1.5),
                    ],
                    bonds=[Bond(1, 2), Bond(1, 3), Bond(1, 4), Bond(1, 5)],
                ),
                monomer_sdf,
            )
            result = build_polymer_from_monomer(
                monomer_sdf,
                MonomerBuildOptions(
                    previous_atom=2,
                    next_atom=3,
                    reference_dp=4,
                    polymer_dp=12,
                    outdir=tmp_path / "ionic_build",
                    run_external=False,
                ),
            )
            self.assertEqual(result.target_charge, 12)
            charged_nitrogens = [
                atom
                for atom in result.molecule.atoms
                if atom.element == "N" and atom.formal_charge == 1
            ]
            self.assertEqual(len(charged_nitrogens), 12)
            for nitrogen in charged_nitrogens:
                neighbors = [
                    result.molecule.atom(bond.b if bond.a == nitrogen.index else bond.a)
                    for bond in result.molecule.bonds
                    if nitrogen.index in (bond.a, bond.b)
                ]
                self.assertFalse(any(atom.element == "H" for atom in neighbors))
            metadata = json.loads((tmp_path / "ionic_build" / "build_metadata.json").read_text())
            self.assertEqual(metadata["monomer_atom_formal_charge"], 1)
            self.assertEqual(metadata["antechamber_formal_charge"], 4)
            self.assertEqual(metadata["target_polymer_formal_charge"], 12)
            atom_lines = []
            in_atoms = False
            for line in (tmp_path / "ionic_build" / "polymer.itp").read_text().splitlines():
                if line.strip() == "[ atoms ]":
                    in_atoms = True
                    continue
                if in_atoms and line.lstrip().startswith("["):
                    break
                if in_atoms and line.strip() and not line.lstrip().startswith(";"):
                    atom_lines.append(line)
            itp_charge = sum(float(line.split()[6]) for line in atom_lines)
            self.assertAlmostEqual(itp_charge, 12.0, places=7)
            self.assertTrue(metadata["formal_charge_inferred_from_atoms"])
    def test_explicit_hydrogen_sdf_helper_adds_missing_hydrogens(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            heavy_sdf = tmp_path / "heavy.sdf"
            hydrogenated_sdf = tmp_path / "with_h.sdf"
            write_sdf(
                Molecule(
                    "ethane_heavy",
                    atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0)],
                    bonds=[Bond(1, 2)],
                ),
                heavy_sdf,
            )
            ensure_explicit_hydrogen_sdf(heavy_sdf, hydrogenated_sdf)
            mol = read_sdf(hydrogenated_sdf)
            self.assertEqual(mol.formula(), "C2H6")
            self.assertEqual(len(mol.atoms), 8)

    def test_mol2_writer_handles_three_digit_atom_and_bond_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            atoms = [Atom(idx, "C", idx * 1.5, 0, 0) for idx in range(1, 103)]
            bonds = [Bond(idx, idx + 1, 1) for idx in range(1, 102)]
            mol2_path = tmp_path / "large.mol2"
            write_mol2(Molecule("large_reference", atoms, bonds), mol2_path)

            text = mol2_path.read_text()
            self.assertIn("@<TRIPOS>MOLECULE\nlarge_reference\n102 101 0 0 0", text)
            self.assertNotIn("102101", text)

    def test_rdkit_mmff_optimizer_called_with_variant_keyword(self):
        calls = []
        embedding_params = []

        class FakePoint:
            z = 0.0

        class FakeConformer:
            def GetAtomPosition(self, _idx):
                return FakePoint()

        class FakeMol:
            def GetNumConformers(self):
                return 1

            def GetConformer(self):
                return FakeConformer()

            def GetNumAtoms(self):
                return 1

            def RemoveAllConformers(self):
                pass

        class FakeSupplier:
            def __init__(self, *_args, **_kwargs):
                self.mol = FakeMol()

            def __len__(self):
                return 1

            def __getitem__(self, _idx):
                return self.mol

        class FakeWriter:
            def __init__(self, path):
                self.path = path

            def write(self, _mol):
                Path(self.path).write_text("minimized\n")

            def close(self):
                pass

        def fake_mmff_optimize(*args, **kwargs):
            calls.append((args, kwargs))
            if len(args) != 1:
                raise TypeError("MMFFOptimizeMolecule expects mol as the only positional argument")
            if kwargs.get("mmffVariant") != "MMFF94s":
                raise AssertionError("mmffVariant keyword was not used")
            return 0

        fake_chem = types.ModuleType("rdkit.Chem")
        fake_chem.SDMolSupplier = FakeSupplier
        fake_chem.SDWriter = FakeWriter
        fake_all_chem = types.ModuleType("rdkit.Chem.AllChem")
        fake_all_chem.ETKDGv3 = lambda: types.SimpleNamespace(
            randomSeed=0, useRandomCoords=False, maxIterations=0
        )

        def fake_embed(_mol, params):
            embedding_params.append(params)
            return -1 if len(embedding_params) == 1 else 0

        fake_all_chem.EmbedMolecule = fake_embed
        fake_all_chem.MMFFHasAllMoleculeParams = lambda _mol: True
        fake_all_chem.MMFFOptimizeMolecule = fake_mmff_optimize
        fake_all_chem.UFFHasAllMoleculeParams = lambda _mol: False
        fake_all_chem.UFFOptimizeMolecule = lambda *_args, **_kwargs: 0
        fake_chem.AllChem = fake_all_chem
        fake_rdkit = types.ModuleType("rdkit")
        fake_rdkit.Chem = fake_chem

        original = {name: sys.modules.get(name) for name in ["rdkit", "rdkit.Chem", "rdkit.Chem.AllChem"]}
        sys.modules["rdkit"] = fake_rdkit
        sys.modules["rdkit.Chem"] = fake_chem
        sys.modules["rdkit.Chem.AllChem"] = fake_all_chem
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                input_sdf = tmp_path / "input.sdf"
                input_sdf.write_text("stub\n")
                output_sdf, method = minimize_sdf_geometry(input_sdf, tmp_path / "output.sdf")
                self.assertEqual(method, "MMFF94s")
                self.assertTrue(output_sdf.exists())
                self.assertEqual(len(calls), 1)
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_reference_parameters_are_applied_to_target_before_tleap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer = Molecule(
                "ethylene_repeat",
                atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0)],
                bonds=[Bond(1, 2)],
            )
            reference, _ = build_reference_oligomer_from_monomer(monomer, 1, 2, 4)
            add_valence_hydrogens(reference)
            reference_sdf = tmp_path / "reference.sdf"
            write_sdf(reference, reference_sdf)

            target, _ = build_reference_oligomer_from_monomer(monomer, 1, 2, 6)
            add_valence_hydrogens(target)
            typed_reference = Molecule(
                reference.name,
                atoms=[
                    Atom(
                        atom.index,
                        atom.element,
                        atom.x,
                        atom.y,
                        atom.z,
                        atom_type="c3" if atom.element == "C" else "hc",
                        charge=(atom.index / 1000.0) * (1 if atom.element == "C" else -1),
                    )
                    for atom in reference.atoms
                ],
                bonds=[Bond(bond.a, bond.b, bond.order) for bond in reference.bonds],
            )
            commands = []

            def fake_run(cmd, **kwargs):
                commands.append((cmd, kwargs))
                if cmd[0] == "antechamber":
                    write_mol2(typed_reference, Path(cmd[cmd.index("-o") + 1]))
                elif cmd[0] == "parmchk2":
                    Path(cmd[cmd.index("-o") + 1]).write_text("# fake frcmod\n")
                elif cmd[0] == "tleap":
                    cwd = Path(kwargs["cwd"])
                    (cwd / "polymer.prmtop").write_text("fake\n")
                    (cwd / "polymer.inpcrd").write_text("fake\n")
                return types.SimpleNamespace(returncode=0)

            class FakeStructure:
                atoms = [None] * len(target.atoms)

                def save(self, path, overwrite=False):
                    path = Path(path)
                    if path.suffix == ".top":
                        path.write_text(
                            "[ defaults ]\n1 2 yes 0.5 0.833333\n\n"
                            "[ moleculetype ]\nPOL 3\n\n"
                            f"[ atoms ]\n; target atoms: {len(self.atoms)}\n\n"
                            "[ system ]\nPolymer\n\n[ molecules ]\nPOL 1\n"
                        )
                    else:
                        path.write_text(f"target atoms: {len(self.atoms)}\n")

            fake_parmed = types.ModuleType("parmed")
            fake_parmed.load_file = lambda *_args, **_kwargs: FakeStructure()
            original_parmed = sys.modules.get("parmed")
            sys.modules["parmed"] = fake_parmed
            try:
                spec = AmberTargetSpec(
                    molecule=target,
                    target_charge=0,
                    unit_atom_count=2,
                    reference_dp=4,
                    target_dp=6,
                )
                with mock.patch("polymer_from_oligomer.shutil.which", return_value="/usr/bin/tool"):
                    with mock.patch("polymer_from_oligomer.subprocess.run", side_effect=fake_run):
                        files, _ = run_ambertools_pipeline(
                            reference_sdf,
                            tmp_path / "amber",
                            formal_charge=0,
                            minimize_geometry=False,
                            target_spec=spec,
                        )
            finally:
                if original_parmed is None:
                    sys.modules.pop("parmed", None)
                else:
                    sys.modules["parmed"] = original_parmed

            target_mol2 = tmp_path / "amber" / "polymer_typed_charged.mol2"
            reference_mol2 = tmp_path / "amber" / "reference_typed_charged.mol2"
            self.assertEqual(len(read_mol2_atom_parameters(reference_mol2)), len(reference.atoms))
            self.assertEqual(len(read_mol2_atom_parameters(target_mol2)), len(target.atoms))
            self.assertEqual(target_mol2.read_text().splitlines()[1], "POLYMER")
            self.assertTrue(all(atom.atom_type in {"c3", "hc"} for atom in target.atoms))
            self.assertAlmostEqual(spec.final_charge, 0.0, places=10)
            self.assertIn("mol = loadmol2 polymer_typed_charged.mol2", (tmp_path / "amber" / "tleap.in").read_text())
            top_path = tmp_path / "amber" / "polymer.top"
            itp_path = tmp_path / "amber" / "polymer.itp"
            self.assertIn(top_path, files)
            self.assertIn(itp_path, files)
            self.assertIn('#include "polymer.itp"', top_path.read_text())
            self.assertIn(f"; target atoms: {len(target.atoms)}", itp_path.read_text())

    def test_ambertools_pipeline_uses_explicit_hydrogen_mol2_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            heavy_sdf = tmp_path / "heavy.sdf"
            write_sdf(
                Molecule(
                    "ethane_heavy",
                    atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0)],
                    bonds=[Bond(1, 2)],
                ),
                heavy_sdf,
            )
            commands = []

            def fake_run(cmd, **_kwargs):
                commands.append(cmd)
                return types.SimpleNamespace(returncode=0)

            class FakeStructure:
                def save(self, path, overwrite=False):
                    Path(path).write_text("fake\n")

            fake_parmed = types.ModuleType("parmed")
            fake_parmed.load_file = lambda *_args, **_kwargs: FakeStructure()
            original_parmed = sys.modules.get("parmed")
            sys.modules["parmed"] = fake_parmed
            try:
                with mock.patch("polymer_from_oligomer.shutil.which", return_value="/usr/bin/tool"):
                    with mock.patch("polymer_from_oligomer.subprocess.run", side_effect=fake_run):
                        files, antechamber_input = run_ambertools_pipeline(
                            heavy_sdf,
                            tmp_path / "amber",
                            formal_charge=0,
                            minimize_geometry=False,
                        )
            finally:
                if original_parmed is None:
                    sys.modules.pop("parmed", None)
                else:
                    sys.modules["parmed"] = original_parmed

            self.assertEqual(antechamber_input.name, "antechamber_input.mol2")
            antechamber_cmd = commands[0]
            self.assertEqual(antechamber_cmd[antechamber_cmd.index("-i") + 1], str(antechamber_input))
            self.assertEqual(antechamber_cmd[antechamber_cmd.index("-fi") + 1], "mol2")
            hydrogenated_sdf = next(path for path in files if path.name == "antechamber_input_with_h.sdf")
            self.assertEqual(read_sdf(hydrogenated_sdf).formula(), "C2H6")
            self.assertTrue(antechamber_input.exists())
            self.assertTrue(any(path.name == "antechamber_input_with_h.sdf" for path in files))


if __name__ == "__main__":
    unittest.main()
