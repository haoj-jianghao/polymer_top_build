"""Tests of the OpenFF Sage engine (external OpenFF installation driven through sage_worker.py).

The first groups need no OpenFF: they cover reading Interchange output, the
search for an OpenFF interpreter (with fake conda environments and fake
interpreters), and the full pipeline with the worker replaced by a synthetic
engine. The integration group runs the real worker in the user's OpenFF
environment and is skipped when none is found.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sage_engine
import topology_transfer as tt
from polymer_from_oligomer import (
    Molecule,
    MonomerBuildOptions,
    build_polymer_from_monomer,
    read_sdf,
    write_sdf,
)
from tests.test_topology_transfer import (
    DIRECT_8MER_CASES,
    exact_term_table,
    fake_engine_topology,
    mma_monomer,
    same_numbering_3d_sdf,
    styrene_monomer,
)


ROOT = Path(__file__).resolve().parents[1]

INTERCHANGE_STYLE_TOP = """[ defaults ]
; nbfunc\tcomb-rule\tgen-pairs\tfudgeLJ\tfudgeQQ
     1\t     2\tyes   \t0.500000\t0.833333

[ atomtypes ]
;type, bondingtype, atomic_number, mass, charge, ptype, sigma, epsilon
MOL_0\t     6\t12.01078\t0.0000000000000000\tA    \t0.3387971073428981\t0.5044644278506009
MOL_1\t     8\t15.99943\t0.0000000000000000\tA    \t0.3033376560335655\t0.3570019770949033
MOL_2\t     1\t1.007947\t0.0000000000000000\tA    \t0.2558603767225489\t0.07916117199426585
MOL_3\t     1\t1.007947\t0.0000000000000000\tA    \t0.2558603767225489\t0.07916117199426585

[ moleculetype ]
MOL\t         3

[ atoms ]
;index, atom type, resnum, resname, name, cgnr, charge, mass
     1 MOL_0       1 MOL C          1      0.1 12.010780000000
     2 MOL_1       1 MOL O          1     -0.3 15.999430000000
     3 MOL_2       1 MOL H          1      0.1 1.007947000000
     4 MOL_3       1 MOL H          1      0.1 1.007947000000

[ bonds ]
     1     2     1      0.14     300000.0
     1     3     1      0.109    280000.0
     1     4     1      0.109    280000.0

[ dihedrals ]
;ai    aj   ak   al   funct phi  k
     3     1     2     4     1      0.000000000000      0.5                 3
     3     1     2     4     1    180.000000000000      0.2                 1

[ settles ]
;i  funct   dOH  dHH


[ exclusions ]
;ai    aj


[ system ]
;name
FOO

[ molecules ]
;name\tnumber
MOL\t1
"""


def make_fake_python(path: Path, payload: dict) -> Path:
    """An executable that answers any invocation (e.g. a worker probe) with ``payload`` as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\ncat <<'JSON'\n" + json.dumps(payload) + "\nJSON\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make_conda_env(prefix: Path, packages: list[str]) -> Path:
    (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
    for package in packages:
        (prefix / "conda-meta" / f"{package}-1.0-py_0.json").write_text("{}")
    python = prefix / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    return python


class SageWorkerOutputTests(unittest.TestCase):
    def test_interchange_topology_is_labelled_by_vdw_parameter_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            top = Path(tmp) / "reference.top"
            top.write_text(INTERCHANGE_STYLE_TOP)
            topology = sage_engine.topology_from_worker_output(top, {"vdw_parameter_ids": ["n16", "n18", "n3", "n3"]})
            self.assertEqual(topology.labels, ["n16", "n18", "n3", "n3"])
            self.assertEqual([topology.atom_type(i).bond_type for i in range(1, 5)], ["n16", "n18", "n3", "n3"])
            self.assertEqual(topology.defaults, ["1", "2", "yes", "0.500000", "0.833333"])
            torsion_lines = [term for term in topology.terms if term.kind == "propers"]
            self.assertEqual(len(torsion_lines), 2)
            atoms = tt.identity_transfer(topology).atoms
            names, table = tt.merge_atom_types(atoms)
            self.assertEqual(names, ["n16", "n18", "n3", "n3"])
            self.assertEqual(len(table), 3)

    def test_label_count_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            top = Path(tmp) / "reference.top"
            top.write_text(INTERCHANGE_STYLE_TOP)
            with self.assertRaises(RuntimeError):
                sage_engine.topology_from_worker_output(top, {"vdw_parameter_ids": ["n16"]})


class InterpreterSearchTests(unittest.TestCase):
    def setUp(self):
        sage_engine.probe.cache_clear()
        self.addCleanup(sage_engine.probe.cache_clear)

    def test_candidates_are_ordered_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "miniforge3"
            base = make_conda_env(root, ["python"])
            good = make_conda_env(root / "envs" / "openff", ["openff-toolkit", "openff-interchange"])
            make_conda_env(root / "envs" / "toolkit_only", ["openff-toolkit"])
            with mock.patch.object(sage_engine, "_conda_roots", return_value=[root]), mock.patch.dict(
                os.environ, {sage_engine.SAGE_PYTHON_ENV: "/env/var/python"}
            ), mock.patch.object(sage_engine.importlib.util, "find_spec", return_value=None):
                candidates = sage_engine.candidate_interpreters("/explicit/python")
            pythons = [python for python, _ in candidates]
            self.assertEqual(pythons, ["/explicit/python", "/env/var/python", str(good)])
            self.assertNotIn(str(base), pythons)

    def test_first_working_interpreter_is_used_and_explicit_choice_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            broken = make_fake_python(tmp_path / "broken" / "python", {"ok": False, "errors": ["no openff"]})
            working = make_fake_python(
                tmp_path / "working" / "python",
                {"ok": True, "errors": [], "charges": {"method": "NAGL", "model": "m.pt", "usable": True}},
            )
            candidates = [(str(broken), "conda environment broken"), (str(working), "conda environment working")]
            with mock.patch.object(sage_engine, "candidate_interpreters", return_value=candidates):
                interpreter, problems = sage_engine.find_interpreter({})
            self.assertEqual(interpreter.python, str(working))
            self.assertEqual(len(problems), 1)
            self.assertEqual(sage_engine.preflight.__name__, "preflight")

            explicit = [(str(broken), "setting sage_python"), (str(working), "conda environment working")]
            with mock.patch.object(sage_engine, "candidate_interpreters", return_value=explicit):
                interpreter, problems = sage_engine.find_interpreter({"sage_python": str(broken)})
                errors, _ = sage_engine.preflight({"sage_python": str(broken)})
            self.assertIsNone(interpreter)
            self.assertTrue(any("no openff" in line for line in errors))

    def test_am1bcc_force_field_gets_a_speed_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = make_fake_python(
                Path(tmp) / "python",
                {"ok": True, "errors": [], "charges": {"method": "AM1-BCC (toolkit; AmberTools sqm or OpenEye)", "usable": True}},
            )
            with mock.patch.object(sage_engine, "candidate_interpreters", return_value=[(str(python), "setting sage_python")]):
                errors, warnings = sage_engine.preflight({"sage_python": str(python), "sage_forcefield": "openff-2.2.1.offxml"})
            self.assertEqual(errors, [])
            self.assertTrue(any("NAGL" in warning for warning in warnings))


class SagePipelineWithFakeEngineTests(unittest.TestCase):
    def test_pipeline_uses_sage_defaults_and_labels(self):
        def fake_parameterize(structure_sdf, outdir, *, net_charge, name="reference", settings=None):
            outdir.mkdir(parents=True, exist_ok=True)
            top = outdir / f"{name}.top"
            fake_engine_topology(read_sdf(structure_sdf), top)
            top.write_text(top.read_text().replace("1 3 yes 0.5 0.5", "1 2 yes 0.5 0.833333"))
            topology = tt.parse_gromacs_topology(top)
            return sage_engine.ReferenceParameterization(
                topology, [top], {"engine": "sage", "charge_model": "NAGL (fake)", "engine_warnings": []}
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "monomer.sdf"
            write_sdf(mma_monomer(), monomer_sdf)
            with mock.patch.object(sage_engine, "preflight", return_value=([], [])), mock.patch.object(
                sage_engine, "parameterize_reference", side_effect=fake_parameterize
            ):
                result = build_polymer_from_monomer(
                    monomer_sdf,
                    MonomerBuildOptions(1, 4, 10, reference_dp=5, outdir=tmp_path / "out", minimize_geometry=False, ff_engine="sage"),
                )
            top = tt.parse_gromacs_topology(tmp_path / "out" / "polymer.top")
            self.assertEqual(top.defaults, ["1", "2", "yes", "0.5", "0.833333"])
            self.assertEqual(len(top.atoms), len(result.molecule.atoms))
            self.assertAlmostEqual(sum(atom.charge for atom in top.atoms), 0.0, places=7)
            metadata = json.loads((tmp_path / "out" / "build_metadata.json").read_text())
            self.assertEqual(metadata["ff_engine"], "sage")
            self.assertTrue((tmp_path / "out" / "sage_reference" / "reference.top").exists())
            self.assertIn("identical parameters", (tmp_path / "out" / "PARAMETER_TRANSFER_REPORT.txt").read_text())


def openff_available() -> bool:
    try:
        return sage_engine.find_interpreter({})[0] is not None
    except Exception:
        return False


@unittest.skipUnless(openff_available(), "no OpenFF interpreter found (set POLYMER_SAGE_PYTHON to run these tests)")
class SageIntegrationTests(unittest.TestCase):
    def test_same_length_transfer_matches_the_sage_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "peg"
            build_polymer_from_monomer(
                ROOT / "examples" / "PEG_repeat_monomer.sdf",
                MonomerBuildOptions(1, 3, 5, reference_dp=5, outdir=out, ff_engine="sage"),
            )
            reference = tt.parse_gromacs_topology(out / "sage_reference" / "reference.top")
            target = tt.parse_gromacs_topology(out / "polymer.top")
            key = lambda term: (term.kind, tt.canonical_atoms(term.kind, term.atoms), term.funct, term.params)
            self.assertEqual(sorted(map(key, target.terms)), sorted(map(key, reference.terms)))
            self.assertEqual(target.defaults, reference.defaults)
            for index in range(1, len(reference.atoms) + 1):
                self.assertEqual(reference.atom_type(index).nonbonded_key(), target.atom_type(index).nonbonded_key())

    def test_transfer_matches_sage_run_on_the_same_polymer(self):
        """Sage run directly on the builder's own 8-mer (same numbering) gives exactly the transferred
        terms, impropers (three per planar centre) in their exact written atom order included."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer, previous, nxt = styrene_monomer()
            monomer_sdf = tmp_path / "styrene.sdf"
            write_sdf(monomer, monomer_sdf)
            out = tmp_path / "built"
            build_polymer_from_monomer(monomer_sdf, MonomerBuildOptions(previous, nxt, 8, reference_dp=5, outdir=out, ff_engine="sage"))
            structure = same_numbering_3d_sdf(out / "polymer.sdf", tmp_path / "same_numbering.sdf")
            direct = sage_engine.parameterize_reference(structure, tmp_path / "direct", net_charge=0, name="target").topology
            built = tt.parse_gromacs_topology(out / "polymer.top")
            self.assertGreater(sum(1 for term in built.terms if term.kind == "impropers"), 0)
            self.assertEqual(exact_term_table(direct), exact_term_table(built))

    def test_builder_8mer_matches_directly_parameterized_8mer(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        def table(top, graph, mapping=None):
            result = {}
            for term in top.terms:
                atoms = tuple(mapping[a] for a in term.atoms) if mapping else term.atoms
                if term.kind == "impropers":  # Interchange writes the central atom first; compare as a set
                    center = next(a for a in atoms if all(o in graph[a] for o in atoms if o != a))
                    key = (center, frozenset(a for a in atoms if a != center))
                else:
                    key = tt.canonical_atoms(term.kind, atoms)
                result.setdefault((term.kind, key), []).append((term.funct, term.params))
            return {k: sorted(v) for k, v in result.items()}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("PEG", "acrylate", "PAA-anion", "nylon6"):
                atoms, bonds, previous, nxt, smiles = DIRECT_8MER_CASES[name]
                with self.subTest(polymer=name):
                    monomer_sdf = tmp_path / f"{name}.sdf"
                    write_sdf(Molecule(name, atoms=atoms, bonds=bonds), monomer_sdf)
                    built_dir = tmp_path / f"{name}_built"
                    build_polymer_from_monomer(
                        monomer_sdf, MonomerBuildOptions(previous, nxt, 8, reference_dp=5, outdir=built_dir, ff_engine="sage")
                    )
                    built = tt.parse_gromacs_topology(built_dir / "polymer.top")
                    built_mol = Chem.SDMolSupplier(str(built_dir / "polymer.sdf"), removeHs=False)[0]

                    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 7
                    params.useRandomCoords = True
                    self.assertEqual(AllChem.EmbedMolecule(rd, params), 0)
                    direct_sdf = tmp_path / f"{name}_direct.sdf"
                    writer = Chem.SDWriter(str(direct_sdf))
                    writer.write(rd)
                    writer.close()
                    direct = sage_engine.parameterize_reference(
                        direct_sdf, tmp_path / f"{name}_direct", net_charge=Chem.GetFormalCharge(rd), name="direct8"
                    ).topology

                    match = built_mol.GetSubstructMatch(rd)
                    self.assertEqual(len(match), rd.GetNumAtoms())
                    mapping = {d + 1: b + 1 for d, b in enumerate(match)}
                    graph = tt.adjacency(
                        built_mol.GetNumAtoms(),
                        [(b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1) for b in built_mol.GetBonds()],
                    )
                    self.assertEqual(table(direct, graph, mapping), table(built, graph))
                    for d, b in mapping.items():
                        self.assertEqual(
                            (direct.label(d), direct.atom_type(d).nonbonded_key()),
                            (built.atoms[b - 1].type_name, built.atom_type(b).nonbonded_key()),
                        )

    def test_silicon_is_reported_as_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            atoms, bonds, previous, nxt, _ = DIRECT_8MER_CASES["PDMS"]
            monomer_sdf = tmp_path / "pdms.sdf"
            write_sdf(Molecule("pdms", atoms=atoms, bonds=bonds), monomer_sdf)
            with self.assertRaisesRegex(RuntimeError, "no parameters for element\\(s\\) Si"):
                build_polymer_from_monomer(
                    monomer_sdf, MonomerBuildOptions(previous, nxt, 8, reference_dp=5, outdir=tmp_path / "out", ff_engine="sage")
                )


if __name__ == "__main__":
    unittest.main()
