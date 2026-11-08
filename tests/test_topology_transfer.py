"""Tests of the engine-neutral reference-to-target topology transfer and the OPLS engine.

The first group needs no external program: a synthetic "engine" writes a
reference topology whose parameters encode each atom's local chemistry
(element, hydrogen count, degree), so a correct transfer must reproduce, for
every target term, the parameters derived from the target atoms themselves.

The second group runs the bundled opls-ff-lite and is skipped unless its
preflight passes (MOPAC and the OPLS/2020 SI files installed).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import opls_engine
import topology_transfer as tt
from polymer_from_oligomer import (
    ATOMIC_MASS,
    Atom,
    Bond,
    Molecule,
    MonomerBuildOptions,
    add_valence_hydrogens,
    build_polymer_from_monomer,
    build_reference_oligomer_from_monomer,
    read_sdf,
    unit_atom_labels,
    write_sdf,
)


ROOT = Path(__file__).resolve().parents[1]
ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "Si": 14}


def atom_class(mol: Molecule, graph: dict[int, set[int]], index: int) -> str:
    element = mol.atom(index).element
    hydrogens = sum(1 for neighbor in graph[index] if mol.atom(neighbor).element == "H")
    return f"{element}{hydrogens}d{len(graph[index])}"


def fake_engine_topology(mol: Molecule, path: Path, multi_term_propers: bool = False) -> None:
    """Write a reference topology whose parameters are functions of local chemistry."""

    graph = tt.adjacency(len(mol.atoms), [(bond.a, bond.b) for bond in mol.bonds])
    classes = {atom.index: atom_class(mol, graph, atom.index) for atom in mol.atoms}
    lines = ["[ defaults ]", "1 3 yes 0.5 0.5", "", "[ atomtypes ]"]
    for atom in mol.atoms:
        cls = classes[atom.index]
        sigma = 0.25 + 0.01 * (sum(map(ord, cls)) % 17)
        lines.append(
            f"t{atom.index:04d} {cls} {ATOMIC_NUMBER[atom.element]} {ATOMIC_MASS[atom.element]:.3f} "
            f"0.0 A {sigma:.6e} {0.1 + 0.01 * len(cls):.6e}"
        )
    lines += ["", "[ moleculetype ]", "REF 3", "", "[ atoms ]"]
    for atom in mol.atoms:
        charge = 0.01 * (sum(map(ord, classes[atom.index])) % 23) - 0.1
        lines.append(
            f"{atom.index} t{atom.index:04d} 1 REF {atom.element}{atom.index} 1 {charge:.6f} "
            f"{ATOMIC_MASS[atom.element]:.3f}"
        )

    def params(atoms):
        names = [classes[atom] for atom in atoms]
        return "-".join(min(names, names[::-1]))

    sections = [("bonds", "bonds", 1), ("pairs", "pairs", 1), ("angles", "angles", 1)]
    for section, kind, funct in sections:
        lines += ["", f"[ {section} ]"]
        for atoms in sorted(tt.expected_terms(kind, graph)):
            extra = "" if section == "pairs" else f" {params(atoms)}"
            lines.append(" ".join(map(str, atoms)) + f" {funct}{extra} ; {kind}")
    lines += ["", "[ dihedrals ]"]
    for atoms in sorted(tt.expected_terms("propers", graph)):
        if multi_term_propers:
            lines.append(" ".join(map(str, atoms)) + f" 9 0.0 {params(atoms)} 1")
            lines.append(" ".join(map(str, atoms)) + f" 9 0.0 {params(atoms)} 3")
        else:
            lines.append(" ".join(map(str, atoms)) + f" 3 {params(atoms)}")
    # Impropers on every heavy atom with exactly three neighbours (order matters).
    lines += ["", "[ dihedrals ]"]
    for center in sorted(graph):
        neighbors = sorted(graph[center])
        if mol.atom(center).element != "H" and len(neighbors) == 3:
            atoms = (neighbors[0], center, neighbors[1], neighbors[2])
            lines.append(" ".join(map(str, atoms)) + f" 4 180.0 {params(atoms)} 2")
    path.write_text("\n".join(lines) + "\n")


def build_pair(monomer: Molecule, previous_atom: int, next_atom: int, reference_dp: int, target_dp: int):
    reference, _ = build_reference_oligomer_from_monomer(monomer, previous_atom, next_atom, reference_dp)
    add_valence_hydrogens(reference)
    target, _ = build_reference_oligomer_from_monomer(monomer, previous_atom, next_atom, target_dp)
    add_valence_hydrogens(target)
    return reference, target


def mma_monomer() -> Molecule:
    return Molecule(
        "mma",
        atoms=[
            Atom(1, "C", 0, 0, 0.2),
            Atom(2, "C", 1.5, 0),
            Atom(3, "C", 1.5, 1.5),
            Atom(4, "C", 3, 0),
            Atom(5, "O", 2.5, 2.5),
            Atom(6, "O", 0.5, 2.5),
            Atom(7, "C", 3.5, 3.5),
        ],
        bonds=[Bond(1, 2), Bond(2, 3), Bond(2, 4), Bond(3, 5, 2), Bond(3, 6), Bond(6, 7)],
    )


def transfer(reference: Molecule, target: Molecule, unit: int, reference_dp: int, target_dp: int, path: Path, **kw):
    fake_engine_topology(reference, path, **kw)
    reference_top = tt.parse_gromacs_topology(path)
    result = tt.transfer_topology(
        reference_top,
        [(bond.a, bond.b) for bond in reference.bonds],
        unit_atom_labels(reference, unit, reference_dp),
        [(bond.a, bond.b) for bond in target.bonds],
        unit_atom_labels(target, unit, target_dp),
    )
    return reference_top, result


class TopologyTransferTests(unittest.TestCase):
    def assert_local_chemistry_reproduced(self, target: Molecule, result: tt.TransferResult) -> None:
        graph = tt.adjacency(len(target.atoms), [(bond.a, bond.b) for bond in target.bonds])
        classes = {atom.index: atom_class(target, graph, atom.index) for atom in target.atoms}
        for atom in result.atoms:
            self.assertEqual(atom.atom_type.bond_type, classes[atom.index], f"atom {atom.index}")
        for term in result.terms:
            if term.section == "pairs":
                continue
            names = [classes[index] for index in term.atoms]
            expected = "-".join(min(names, names[::-1]))
            self.assertIn(expected, term.params, f"{term.kind} {term.atoms}")

    def test_transfer_reproduces_local_chemistry_and_covers_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, target = build_pair(mma_monomer(), 1, 4, 5, 20)
            reference_top, result = transfer(reference, target, 7, 5, 20, Path(tmp) / "ref.top")
            self.assertEqual(len(result.atoms), len(target.atoms))
            self.assert_local_chemistry_reproduced(target, result)
            graph = tt.adjacency(len(target.atoms), [(bond.a, bond.b) for bond in target.bonds])
            for kind in ("bonds", "angles", "pairs", "propers"):
                actual = {tt.canonical_atoms(kind, term.atoms) for term in result.terms if term.kind == kind}
                self.assertEqual(actual, tt.expected_terms(kind, graph), kind)
            impropers = [term for term in result.terms if term.kind == "impropers"]
            self.assertEqual(len(impropers), 20)  # one planar centre (the ester carbonyl C) per unit
            self.assertEqual(result.report.warnings, [])

    def test_same_length_target_reproduces_reference_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, target = build_pair(mma_monomer(), 1, 4, 5, 5)
            reference_top, result = transfer(reference, target, 7, 5, 5, Path(tmp) / "ref.top")
            key = lambda term: (term.kind, tt.canonical_atoms(term.kind, term.atoms), term.funct, term.params)
            self.assertEqual(sorted(map(key, result.terms)), sorted(map(key, reference_top.terms)))
            for atom in result.atoms:
                self.assertEqual(atom.atom_type, reference_top.atom_type(atom.index))

    def test_multi_term_dihedrals_are_kept_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            monomer = Molecule("pe", atoms=[Atom(1, "C", 0, 0, 0.1), Atom(2, "C", 1.5, 0)], bonds=[Bond(1, 2)])
            reference, target = build_pair(monomer, 1, 2, 6, 15)
            _, result = transfer(reference, target, 2, 6, 15, Path(tmp) / "ref.top", multi_term_propers=True)
            per_torsion = Counter(term.atoms for term in result.terms if term.kind == "propers")
            self.assertTrue(per_torsion)
            self.assertEqual(set(per_torsion.values()), {2})
            multiplicities = Counter(term.params[-1] for term in result.terms if term.kind == "propers")
            self.assertEqual(multiplicities["1"], multiplicities["3"])
            self.assert_local_chemistry_reproduced(target, result)

    def test_too_short_reference_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            monomer = Molecule("pe", atoms=[Atom(1, "C", 0, 0, 0.1), Atom(2, "C", 1.5, 0)], bonds=[Bond(1, 2)])
            reference, target = build_pair(monomer, 1, 2, 5, 2)
            with self.assertRaisesRegex(ValueError, "at least as long as the reference"):
                transfer(reference, target, 2, 5, 2, Path(tmp) / "ref.top")
            fake_engine_topology(reference, Path(tmp) / "ref2.top")
            with self.assertRaisesRegex(ValueError, "longer reference oligomer"):
                tt.transfer_by_environment_match(
                    tt.parse_gromacs_topology(Path(tmp) / "ref2.top"),
                    [(bond.a, bond.b) for bond in reference.bonds],
                    unit_atom_labels(reference, 2, 5),
                    [(bond.a, bond.b) for bond in target.bonds],
                    unit_atom_labels(target, 2, 2),
                )

    def test_index_shift_rule(self):
        # 5-mer reference (2 end units each side) -> 12-mer target
        sources = [tt.shift_source_unit(unit, 0, 12, 5) for unit in range(12)]
        self.assertEqual(sources, [0, 1] + [2] * 8 + [3, 4])
        self.assertEqual(tt.shift_source_unit(5, 1, 12, 5), 1)   # junction 6-7 <- central window 2-3
        self.assertEqual(tt.shift_source_unit(10, 1, 12, 5), 3)  # junction 11-12 <- reference 4-5
        # same length: every term is copied from its own position
        self.assertTrue(all(tt.shift_source_unit(u, 0, 5, 5) == u for u in range(5)))

    def test_wrong_shift_into_an_end_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, target = build_pair(mma_monomer(), 1, 4, 5, 12)
            # a wrong rule: copy every term from the most central window, chain ends included
            wrong = lambda first, span, target_dp, reference_dp: (reference_dp - 1 - span) // 2
            with mock.patch.object(tt, "shift_source_unit", side_effect=wrong):
                with self.assertRaisesRegex(ValueError, "do not contain the same atoms"):
                    transfer(reference, target, 7, 5, 12, Path(tmp) / "ref.top")

    def test_result_is_rejected_when_the_cross_check_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, target = build_pair(mma_monomer(), 1, 4, 5, 12)
            original = tt.transfer_by_environment_match

            def one_parameter_off(*args, **kwargs):
                result = original(*args, **kwargs)
                term = result.terms[0]
                result.terms[0] = tt.Term(term.section, term.atoms, term.funct, term.params + ("0.1",), term.comment)
                return result

            with mock.patch.object(tt, "transfer_by_environment_match", side_effect=one_parameter_off):
                with self.assertRaises(tt.TransferDisagreement) as caught:
                    transfer(reference, target, 7, 5, 12, Path(tmp) / "ref.top")
            self.assertIn("only from environment match", str(caught.exception))
            self.assertIn("only from index shift", str(caught.exception))

    def test_compare_transfers_detects_a_single_parameter_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, target = build_pair(mma_monomer(), 1, 4, 5, 9)
            _, result = transfer(reference, target, 7, 5, 9, Path(tmp) / "ref.top")
            self.assertIn("identical parameters", result.report.cross_check)
            changed = list(result.terms)
            term = changed[10]
            changed[10] = tt.Term(term.section, term.atoms, term.funct, term.params[:-1] + ("X",), term.comment)
            altered = tt.TransferResult(result.atoms, changed, result.report)
            differences = tt.compare_transfers(altered, result)
            self.assertEqual(len(differences), 2)
            self.assertEqual(tt.compare_transfers(result, result), [])

    def test_merge_splits_same_label_with_different_nonbonded_parameters(self):
        def atom(index, label, sigma):
            atom_type = tt.AtomType(f"x{index}", "CT", 6, "12.011", "0", "A", sigma, "0.276144")
            return tt.TransferredAtom(index, atom_type, label, "12.011", 4)

        atoms = [atom(1, "opls_135", "0.355"), atom(2, "opls_135", "0.350"), atom(3, "opls_135", "0.3550"), atom(4, "opls_136", "0.350")]
        names, table = tt.merge_atom_types(atoms)
        self.assertEqual(names, ["opls_135", "opls_135_2", "opls_135", "opls_136"])
        self.assertEqual(len(table), 3)
        names, table = tt.merge_atom_types(atoms, merge=False)
        self.assertEqual(len(set(names)), 4)

    def test_parser_reads_seven_column_atomtypes_and_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "mol.itp").write_text(
                "[ atomtypes ]\n"
                "AT_0  6  12.011  0.0  A  0.339  0.457\n"
                "AT_1  1  1.008   0.0  A  0.260  0.066 ; hydrogen\n\n"
                "[ moleculetype ]\nMOL 3\n\n[ atoms ]\n"
                "1 AT_0 1 MOL C1 1 -0.4 12.011\n2 AT_1 1 MOL H1 2 0.4\n\n"
                "[ bonds ]\n1 2 1 0.109 300000.0 ; c-h\n"
            )
            (tmp_path / "mol.top").write_text(
                '[ defaults ]\n1 2 yes 0.5 0.8333\n\n#include "mol.itp"\n\n[ system ]\nx\n\n[ molecules ]\nMOL 1\n'
            )
            top = tt.parse_gromacs_topology(tmp_path / "mol.top")
            self.assertEqual(top.defaults, ["1", "2", "yes", "0.5", "0.8333"])
            self.assertEqual(top.atom_type(1).atomic_number, 6)
            self.assertEqual(top.atom_type(2).sigma, "0.260")
            self.assertEqual(top.atoms[1].mass, "1.008")
            self.assertEqual(top.terms[0].params, ("0.109", "300000.0"))
            self.assertEqual(top.terms[0].comment, "c-h")


class OplsPipelineWithFakeEngineTests(unittest.TestCase):
    """Runs the full OPLS pipeline with opls-ff-lite replaced by the synthetic engine."""

    def run_build(self, tmp_path: Path, monomer: Molecule, previous: int, nxt: int, reference_dp: int, dp: int):
        monomer_sdf = tmp_path / "monomer.sdf"
        write_sdf(monomer, monomer_sdf)

        def fake_parameterize(structure_sdf, outdir, *, net_charge, name="reference"):
            outdir.mkdir(parents=True, exist_ok=True)
            top = outdir / f"{name}.top"
            fake_engine_topology(read_sdf(structure_sdf), top)
            topology = tt.parse_gromacs_topology(top)
            return opls_engine.ReferenceParameterization(
                topology=topology, files=[top], info={"engine": "opls", "charge_model": "fake", "engine_warnings": []}
            )

        with mock.patch.object(opls_engine, "preflight", return_value=([], [])), mock.patch.object(
            opls_engine, "parameterize_reference", side_effect=fake_parameterize
        ):
            return build_polymer_from_monomer(
                monomer_sdf,
                MonomerBuildOptions(
                    previous_atom=previous,
                    next_atom=nxt,
                    reference_dp=reference_dp,
                    polymer_dp=dp,
                    outdir=tmp_path / "out",
                    minimize_geometry=False,
                    ff_engine="opls",
                ),
            )

    def test_pipeline_writes_merged_topology_with_exact_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = self.run_build(tmp_path, mma_monomer(), 1, 4, 5, 12)
            out = tmp_path / "out"
            self.assertEqual(result.missing_tools, [])
            top = tt.parse_gromacs_topology(out / "polymer.top")
            self.assertEqual(len(top.atoms), len(result.molecule.atoms))
            self.assertEqual(top.defaults, ["1", "3", "yes", "0.5", "0.5"])
            self.assertLess(len(top.atomtypes), 15)
            self.assertAlmostEqual(sum(atom.charge for atom in top.atoms), 0.0, places=7)
            gro_lines = (out / "polymer.gro").read_text().splitlines()
            self.assertEqual(int(gro_lines[1]), len(result.molecule.atoms))
            self.assertTrue(all(float(value) >= 3.0 for value in gro_lines[-1].split()))
            metadata = json.loads((out / "build_metadata.json").read_text())
            self.assertEqual(metadata["ff_engine"], "opls")
            self.assertEqual(metadata["parameterization_mode"], "reference_to_target")
            self.assertEqual(metadata["engine"]["parameterization_mode"], "reference_to_target")
            report = (out / "PARAMETER_TRANSFER_REPORT.txt").read_text()
            self.assertIn("(complete for the target graph)", report)

    def test_target_shorter_than_reference_is_parameterized_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer = Molecule("pe", atoms=[Atom(1, "C", 0, 0, 0.1), Atom(2, "C", 1.5, 0)], bonds=[Bond(1, 2)])
            result = self.run_build(tmp_path, monomer, 1, 2, 5, 2)
            metadata = json.loads((tmp_path / "out" / "build_metadata.json").read_text())
            self.assertEqual(metadata["parameterization_mode"], "direct_target")
            self.assertEqual(metadata["parameterization_source_sdf"], str(tmp_path / "out" / "polymer.sdf"))
            self.assertEqual(result.molecule.formula(), "C4H10")

    def test_missing_requirements_fall_back_to_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "monomer.sdf"
            write_sdf(mma_monomer(), monomer_sdf)
            problem = (["MOPAC: not found; needed for the CM1A charges"], [])
            with mock.patch.object(opls_engine, "preflight", return_value=problem):
                result = build_polymer_from_monomer(
                    monomer_sdf,
                    MonomerBuildOptions(1, 4, 6, reference_dp=5, outdir=tmp_path / "out", ff_engine="opls"),
                )
            self.assertEqual(result.missing_tools, ["MOPAC"])
            metadata = json.loads((tmp_path / "out" / "build_metadata.json").read_text())
            self.assertEqual(metadata["parameterization_mode"], "placeholder")


def lite_ready() -> bool:
    try:
        return not opls_engine.preflight()[0]
    except Exception:
        return False


@unittest.skipUnless(lite_ready(), "opls-ff-lite is not ready (needs Python 3.12+, RDKit, MOPAC and the OPLS/2020 SI)")
class OplsLiteIntegrationTests(unittest.TestCase):
    def build(self, tmp_path: Path, dp: int, reference_dp: int = 5) -> Path:
        out = tmp_path / f"peg_dp{dp}"
        build_polymer_from_monomer(
            ROOT / "examples" / "PEG_repeat_monomer.sdf",
            MonomerBuildOptions(1, 3, dp, reference_dp=reference_dp, outdir=out, ff_engine="opls"),
        )
        return out

    def test_same_length_transfer_matches_opls_ff_lite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.build(Path(tmp), 5)
            reference = tt.parse_gromacs_topology(out / "opls_reference" / "reference.top")
            target = tt.parse_gromacs_topology(out / "polymer.top")
            key = lambda term: (term.kind, tt.canonical_atoms(term.kind, term.atoms), term.funct, term.params)
            self.assertEqual(sorted(map(key, target.terms)), sorted(map(key, reference.terms)))
            for index in range(1, len(reference.atoms) + 1):
                self.assertEqual(
                    reference.atom_type(index).nonbonded_key(), target.atom_type(index).nonbonded_key()
                )
            self.assertLess(len(target.atomtypes), len(reference.atomtypes))

    def test_long_polymer_has_exact_charge_and_passes_grompp(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.build(Path(tmp), 20)
            top = tt.parse_gromacs_topology(out / "polymer.top")
            self.assertAlmostEqual(sum(atom.charge for atom in top.atoms), 0.0, places=7)
            metadata = json.loads((out / "build_metadata.json").read_text())
            self.assertEqual(metadata["engine"]["charge_model"], "1.14*CM1A-LBCC")
            gmx = shutil.which("gmx") or shutil.which("gmx_mpi")
            if gmx is None:
                self.skipTest("GROMACS not installed")
            mdp = out / "check.mdp"
            mdp.write_text("integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\nrvdw = 1.2\nrcoulomb = 1.2\n")
            run = subprocess.run(
                [gmx, "grompp", "-f", mdp.name, "-c", "polymer.gro", "-p", "polymer.top", "-o", "check.tpr", "-maxwarn", "2"],
                cwd=out,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])


DIRECT_8MER_CASES = {
    # name: (monomer atoms, bonds, previous atom, next atom, independent SMILES of the H-capped 8-mer)
    "PEG": ([Atom(1, "C", 0, 0), Atom(2, "C", 1.45, 0), Atom(3, "O", 2.9, 0)], [Bond(1, 2), Bond(2, 3)], 1, 3, "CCO" * 8),
    "PE": ([Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0)], [Bond(1, 2)], 1, 2, "CC" * 8),
    "acrylate": (mma_monomer().atoms, mma_monomer().bonds, 1, 4, "CC(C(=O)OC)C" * 8),
    "styrene": (
        [Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0), Atom(3, "C", 1.5, 1.5), Atom(4, "C", 0.3, 2.2),
         Atom(5, "C", 0.3, 3.6), Atom(6, "C", 1.5, 4.3), Atom(7, "C", 2.7, 3.6), Atom(8, "C", 2.7, 2.2)],
        [Bond(1, 2), Bond(2, 3), Bond(3, 4, 2), Bond(4, 5), Bond(5, 6, 2), Bond(6, 7), Bond(7, 8, 2), Bond(8, 3)],
        1, 2, "CC(c1ccccc1)" * 8,
    ),
    "PAA-anion": (
        [Atom(1, "C", 0, 0), Atom(2, "C", 1.5, 0), Atom(3, "C", 1.5, 1.5), Atom(4, "O", 0.5, 2.3),
         Atom(5, "O", 2.5, 2.3, formal_charge=-1)],
        [Bond(1, 2), Bond(2, 3), Bond(3, 4, 2), Bond(3, 5)], 1, 2, "CC(C(=O)[O-])" * 8,
    ),
    "quat-N+": (
        [Atom(1, "N", formal_charge=1), Atom(2, "C", 1.5, 0), Atom(3, "C", -1.5, 0), Atom(4, "C", 0, 1.5), Atom(5, "C", 0, -1.5)],
        [Bond(1, 2), Bond(1, 3), Bond(1, 4), Bond(1, 5)], 2, 3, "C[N+](C)(C)C" * 8,
    ),
    "PDMS": (
        [Atom(1, "Si", 0, 0), Atom(2, "O", 1.6, 0), Atom(3, "C", 0, 1.8), Atom(4, "C", 0, -1.8)],
        [Bond(1, 2), Bond(1, 3), Bond(1, 4)], 1, 2, "[SiH](C)(C)O" + "[Si](C)(C)O" * 7,
    ),
    "nylon6": (
        [Atom(1, "N", 0, 0), Atom(2, "C", 1.4, 0), Atom(3, "O", 1.4, 1.3), Atom(4, "C", 2.8, 0),
         Atom(5, "C", 4.2, 0), Atom(6, "C", 5.6, 0), Atom(7, "C", 7.0, 0), Atom(8, "C", 8.4, 0)],
        [Bond(1, 2), Bond(2, 3, 2), Bond(2, 4), Bond(4, 5), Bond(5, 6), Bond(6, 7), Bond(7, 8)], 1, 8, "NC(=O)CCCCC" * 8,
    ),
}


@unittest.skipUnless(lite_ready(), "opls-ff-lite is not ready (needs Python 3.12+, RDKit, MOPAC and the OPLS/2020 SI)")
class DirectOligomerComparisonTests(unittest.TestCase):
    """The builder's 8-mer (5-mer reference + index shift) must have exactly the terms, parameters
    and atom types of an 8-mer built independently from SMILES and parameterized directly."""

    @staticmethod
    def direct_structure(smiles: str, workdir: Path, name: str):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 7
        params.useRandomCoords = True  # robust for long flexible chains
        if AllChem.EmbedMolecule(rd, params) != 0:
            raise RuntimeError(f"RDKit could not embed {smiles}")
        AllChem.MMFFOptimizeMolecule(rd, maxIters=5000)
        path = workdir / f"{name}_direct.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(rd)
        writer.close()
        return path

    def test_builder_8mer_matches_directly_parameterized_8mer(self):
        from rdkit import Chem

        opls_engine._import_lite()
        from opls_ff_lite import generate_gromacs_topology
        from opls_ff_lite.structure import prepare_structure

        def key(kind, atoms):
            if kind == "impropers":  # written order depends on atom numbering
                return (atoms[1], frozenset((atoms[0], atoms[2], atoms[3])))
            return tt.canonical_atoms(kind, atoms)

        def table(top, mapping=None):
            result = {}
            for term in top.terms:
                atoms = tuple(mapping[a] for a in term.atoms) if mapping else term.atoms
                result.setdefault((term.kind, key(term.kind, atoms)), []).append((term.funct, term.params))
            return result

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name, (atoms, bonds, previous, nxt, smiles) in DIRECT_8MER_CASES.items():
                with self.subTest(polymer=name):
                    monomer_sdf = tmp_path / f"{name}.sdf"
                    write_sdf(Molecule(name, atoms=atoms, bonds=bonds), monomer_sdf)
                    built_dir = tmp_path / f"{name}_built"
                    build_polymer_from_monomer(
                        monomer_sdf, MonomerBuildOptions(previous, nxt, 8, reference_dp=5, outdir=built_dir, ff_engine="opls")
                    )
                    built = tt.parse_gromacs_topology(built_dir / "polymer.top")
                    built_mol = Chem.SDMolSupplier(str(built_dir / "polymer.sdf"), removeHs=False)[0]

                    structure = self.direct_structure(smiles, tmp_path, name)
                    direct_mol = prepare_structure(path=structure, name=f"{name}8", optimize=None)
                    direct_result = generate_gromacs_topology(direct_mol, tmp_path / f"{name}_direct", name=f"{name}8")
                    direct = tt.parse_gromacs_topology(direct_result.top_path)
                    labels = [atom.opls_type for atom in direct_result.assignment.atoms]

                    match = built_mol.GetSubstructMatch(direct_mol)
                    self.assertEqual(len(match), direct_mol.GetNumAtoms(), "the two 8-mers are not the same graph")
                    mapping = {d + 1: b + 1 for d, b in enumerate(match)}
                    self.assertEqual(table(direct, mapping), table(built))
                    for d, b in mapping.items():
                        self.assertEqual(
                            (labels[d - 1], direct.atom_type(d).bond_type, direct.atom_type(d).nonbonded_key()),
                            (built.atoms[b - 1].type_name, built.atom_type(b).bond_type, built.atom_type(b).nonbonded_key()),
                            f"atom {b}",
                        )


if __name__ == "__main__":
    unittest.main()
