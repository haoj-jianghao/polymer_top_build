"""Tests of the starting-structure random walk (chain_builder).

They need RDKit (for the minimized 3D reference oligomer) and NumPy, but no
force-field engine.
"""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import opls_engine
import topology_transfer as tt
from polymer_from_oligomer import (
    Atom,
    Bond,
    Molecule,
    MonomerBuildOptions,
    add_valence_hydrogens,
    apply_random_walk_coordinates,
    build_polymer_from_monomer,
    build_reference_oligomer_from_monomer,
    minimize_sdf_geometry,
    read_sdf,
    unit_atom_labels,
    write_sdf,
)
from tests.test_topology_transfer import DIRECT_8MER_CASES, fake_engine_topology, mma_monomer

HAS_RDKIT = importlib.util.find_spec("rdkit") is not None


def minimized_reference(monomer: Molecule, previous: int, nxt: int, reference_dp: int, tmp: Path) -> Molecule:
    reference, _ = build_reference_oligomer_from_monomer(monomer, previous, nxt, reference_dp)
    add_valence_hydrogens(reference)
    write_sdf(reference, tmp / "reference.sdf")
    minimized, _ = minimize_sdf_geometry(tmp / "reference.sdf", tmp / "reference_min.sdf")
    return read_sdf(minimized)


def target_for(monomer: Molecule, previous: int, nxt: int, dp: int) -> Molecule:
    target, _ = build_reference_oligomer_from_monomer(monomer, previous, nxt, dp)
    add_valence_hydrogens(target)
    return target


def xyz(mol: Molecule) -> dict[int, np.ndarray]:
    return {atom.index: np.array([atom.x, atom.y, atom.z]) for atom in mol.atoms}


@unittest.skipUnless(HAS_RDKIT, "RDKit is needed for the minimized reference geometry")
class RandomWalkTests(unittest.TestCase):
    def walk(self, name: str, dp: int, seed: int = 0, reference_dp: int = 5):
        atoms, bonds, previous, nxt, _ = DIRECT_8MER_CASES[name]
        monomer = Molecule(name, atoms=atoms, bonds=bonds)
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        reference = minimized_reference(monomer, previous, nxt, reference_dp, tmp)
        target = target_for(monomer, previous, nxt, dp)
        result = apply_random_walk_coordinates(reference, target, len(monomer.atoms), reference_dp, dp, seed)
        return monomer, reference, target, result

    def test_bond_lengths_are_those_of_the_source_units_and_there_are_no_clashes(self):
        import chain_builder

        for name in ("PEG", "acrylate", "styrene", "nylon6"):
            with self.subTest(polymer=name):
                monomer, reference, target, result = self.walk(name, 25)
                n = len(monomer.atoms)
                target_labels = unit_atom_labels(target, n, 25)
                reference_by_label = {label: index for index, label in unit_atom_labels(reference, n, 5).items()}
                tx, rx = xyz(target), xyz(reference)
                for bond in target.bonds:
                    anchor = max(target_labels[bond.a][0], target_labels[bond.b][0])
                    source = tt.shift_source_unit(anchor, 0, 25, 5)

                    def reference_atom(index):
                        unit, key = target_labels[index]
                        return reference_by_label[(source + unit - anchor, key)]

                    expected = np.linalg.norm(rx[reference_atom(bond.a)] - rx[reference_atom(bond.b)])
                    self.assertAlmostEqual(np.linalg.norm(tx[bond.a] - tx[bond.b]), expected, places=6)
                self.assertGreaterEqual(result.min_contact_ratio, 1.0)
                self.assertEqual(len(result.rotamers), 24)
                z_spread = max(atom.z for atom in target.atoms) - min(atom.z for atom in target.atoms)
                self.assertGreater(z_spread, 1.0)  # genuinely three-dimensional

    def test_perfluorinated_chain_uses_the_reference_contacts_as_limit(self):
        # -CF2-CF2-: the minimized reference has F...F contacts below the plain
        # clash limit, which used to stop the walk at the second unit.
        monomer = Molecule(
            "PTFE",
            atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0), Atom(3, "F", 1.5, 1.3),
                   Atom(4, "F", 1.5, -1.3), Atom(5, "F", 0, 1.3), Atom(6, "F", 0, -1.3)],
            bonds=[Bond(1, 2), Bond(1, 5), Bond(1, 6), Bond(2, 3), Bond(2, 4)],
        )
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        reference = minimized_reference(monomer, 1, 2, 5, tmp)
        target = target_for(monomer, 1, 2, 12)
        result = apply_random_walk_coordinates(reference, target, len(monomer.atoms), 5, 12, 0)
        self.assertEqual(len(result.rotamers), 11)
        self.assertLess(result.limit_scale, 1.0)
        self.assertGreaterEqual(result.min_contact_ratio, 1.0)
        self.assertTrue(any("scaled by" in line for line in result.lines()))

    def test_walk_with_end_groups(self):
        from polymer_from_oligomer import add_end_groups

        monomer = Molecule(
            "PTFE",
            atoms=[Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0), Atom(3, "F", 1.5, 1.3),
                   Atom(4, "F", 1.5, -1.3), Atom(5, "F", 0, 1.3), Atom(6, "F", 0, -1.3)],
            bonds=[Bond(1, 2), Bond(1, 5), Bond(1, 6), Bond(2, 3), Bond(2, 4)],
        )
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

        def capped(dp):
            chain, _ = build_reference_oligomer_from_monomer(monomer, 1, 2, dp)
            end_atoms = add_end_groups(chain, 1, (dp - 1) * 6 + 2, "F", "CF3")
            add_valence_hydrogens(chain)
            return chain, end_atoms

        reference, end_atoms = capped(5)
        write_sdf(reference, tmp / "reference.sdf")
        reference = read_sdf(minimize_sdf_geometry(tmp / "reference.sdf", tmp / "reference_min.sdf")[0])
        target, _ = capped(12)
        result = apply_random_walk_coordinates(reference, target, 6, 5, 12, 0, end_atoms=end_atoms)
        self.assertEqual(len(result.rotamers), 11)
        self.assertGreaterEqual(result.min_contact_ratio, 1.0)
        self.assertEqual(len(result.coordinates), len(target.atoms))

    def test_walk_is_reproducible_for_a_seed(self):
        _, _, first, _ = self.walk("acrylate", 20, seed=3)
        _, _, again, _ = self.walk("acrylate", 20, seed=3)
        _, _, other, _ = self.walk("acrylate", 20, seed=4)
        self.assertEqual([(a.x, a.y, a.z) for a in first.atoms], [(a.x, a.y, a.z) for a in again.atoms])
        self.assertNotEqual([(a.x, a.y, a.z) for a in first.atoms], [(a.x, a.y, a.z) for a in other.atoms])

    def test_drawn_stereochemistry_is_kept_in_every_unit(self):
        from rdkit import Chem

        monomer = Molecule(
            "chiral",
            atoms=[Atom(1, "C"), Atom(2, "C", 1.5, 0), Atom(3, "N", -1.5, 0), Atom(4, "O", 0, 1.5)],
            bonds=[Bond(1, 2, 1, 1), Bond(1, 3, 1, 6), Bond(1, 4)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = minimized_reference(monomer, 2, 3, 5, tmp_path)
            target = target_for(monomer, 2, 3, 15)
            apply_random_walk_coordinates(reference, target, 4, 5, 15, seed=0)
            write_sdf(target, tmp_path / "target.sdf")

            def centre_labels(path, units):
                mol = Chem.MolFromMolFile(str(path), removeHs=False)
                Chem.AssignStereochemistryFrom3D(mol)
                return [mol.GetAtomWithIdx(4 * unit).GetProp("_CIPCode") for unit in range(units)]

            reference_labels = centre_labels(tmp_path / "reference_min.sdf", 5)
            target_labels = centre_labels(tmp_path / "target.sdf", 15)
            self.assertEqual(len(set(reference_labels)), 1)
            self.assertEqual(set(target_labels), set(reference_labels))

    def test_unfinished_walk_is_an_error(self):
        import chain_builder

        atoms, bonds, previous, nxt, _ = DIRECT_8MER_CASES["acrylate"]
        monomer = Molecule("acrylate", atoms=atoms, bonds=bonds)
        with tempfile.TemporaryDirectory() as tmp:
            reference = minimized_reference(monomer, previous, nxt, 5, Path(tmp))
            target = target_for(monomer, previous, nxt, 20)
            n = len(monomer.atoms)
            with self.assertRaisesRegex(RuntimeError, "chain-seed"):
                chain_builder.random_walk_coordinates(
                    {a.index: (a.x, a.y, a.z) for a in reference.atoms},
                    unit_atom_labels(reference, n, 5),
                    [(b.a, b.b) for b in reference.bonds],
                    unit_atom_labels(target, n, 20),
                    [(b.a, b.b, b.order) for b in target.bonds],
                    {a.index: a.element for a in target.atoms},
                    max_placements=5,
                )


@unittest.skipUnless(HAS_RDKIT, "RDKit is needed for the minimized reference geometry")
class PipelineLayoutTests(unittest.TestCase):
    def build(self, tmp_path: Path, layout: str):
        def fake_parameterize(structure_sdf, outdir, *, net_charge, name="reference", settings=None):
            outdir.mkdir(parents=True, exist_ok=True)
            top = outdir / f"{name}.top"
            fake_engine_topology(read_sdf(structure_sdf), top)
            return opls_engine.ReferenceParameterization(
                tt.parse_gromacs_topology(top), [top], {"engine": "opls", "charge_model": "fake", "engine_warnings": []}
            )

        monomer_sdf = tmp_path / "monomer.sdf"
        write_sdf(mma_monomer(), monomer_sdf)
        out = tmp_path / layout
        with mock.patch.object(opls_engine, "preflight", return_value=([], [])), mock.patch.object(
            opls_engine, "parameterize_reference", side_effect=fake_parameterize
        ):
            build_polymer_from_monomer(
                monomer_sdf, MonomerBuildOptions(1, 4, 20, reference_dp=5, outdir=out, ff_engine="opls", chain_layout=layout)
            )
        return out

    def test_random_walk_is_the_default_and_2d_keeps_the_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            walked = self.build(tmp_path, "random-walk")
            flat = self.build(tmp_path, "2d")
            report = (walked / "PARAMETER_TRANSFER_REPORT.txt").read_text()
            self.assertIn("self-avoiding random walk", report)
            self.assertEqual(json.loads((walked / "build_metadata.json").read_text())["engine"]["chain_layout"], "random-walk")
            self.assertEqual(json.loads((flat / "build_metadata.json").read_text())["engine"]["chain_layout"], "2d")

            def z_spread(directory):
                z = [float(line[36:44]) for line in (directory / "polymer.gro").read_text().splitlines()[2:-1]]
                return max(z) - min(z)

            self.assertGreater(z_spread(walked), 0.5)
            self.assertLess(z_spread(flat), 0.1)
            # polymer.sdf carries the walked coordinates too (the .gro is only translated into its box)
            sdf = xyz(read_sdf(walked / "polymer.sdf"))
            gro = [
                np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])]) * 10
                for line in (walked / "polymer.gro").read_text().splitlines()[2:-1]
            ]
            last = len(gro)
            self.assertAlmostEqual(np.linalg.norm(gro[0] - gro[last - 1]), np.linalg.norm(sdf[1] - sdf[last]), delta=0.02)


if __name__ == "__main__":
    unittest.main()
