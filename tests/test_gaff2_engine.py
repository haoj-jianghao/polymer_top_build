"""Tests of the GAFF2/AM1-BCC engine (external AmberTools installation driven through amber_worker.py).

The first groups need no AmberTools. The integration group runs the real
worker in the user's AmberTools environment and is skipped when none is found.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import external_env
import gaff2_engine
import topology_transfer as tt
from polymer_from_oligomer import Molecule, MonomerBuildOptions, build_polymer_from_monomer, read_sdf, write_sdf
from tests.test_sage_engine import make_conda_env
from tests.test_topology_transfer import DIRECT_8MER_CASES, exact_term_table, fake_engine_topology, mma_monomer


ROOT = Path(__file__).resolve().parents[1]

PARMED_STYLE_TOP = """;
;   File ref.top  was generated
;

[ defaults ]
; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
1               2               yes             0.5          0.83333333

[ atomtypes ]
; name    at.num    mass    charge ptype  sigma      epsilon
c3             6  12.010000  0.00000000  A     0.33977095      0.4510352
hc             1   1.008000  0.00000000  A      0.2600177      0.0870272

[ moleculetype ]
; Name            nrexcl
MOL          3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr    charge       mass  typeB    chargeB      massB
; residue    1 MOL rtp MOL q 0.0
    1         c3      1    MOL     C1      1 -0.06000000  12.010000   ; qtot -0.060000
    2         hc      1    MOL     H2      2 0.02000000   1.008000   ; qtot -0.040000
    3         hc      1    MOL     H3      3 0.02000000   1.008000   ; qtot -0.020000
    4         hc      1    MOL     H4      4 0.02000000   1.008000   ; qtot 0.000000

[ bonds ]
;    ai     aj funct         c0         c1         c2         c3
      1      2     1   0.10930 281080.320000
      1      3     1   0.10930 281080.320000
      1      4     1   0.10930 281080.320000

[ angles ]
;    ai     aj     ak funct         c0         c1         c2         c3
      2      1      3     1   107.5800000 326.352000
      2      1      4     1   107.5800000 326.352000
      3      1      4     1   107.5800000 326.352000

[ system ]
; Name
Generic title

[ molecules ]
; Compound       #mols
MOL                  1
"""


class Gaff2WorkerOutputTests(unittest.TestCase):
    def test_parmed_topology_is_labelled_by_gaff2_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            top = Path(tmp) / "reference.top"
            top.write_text(PARMED_STYLE_TOP)
            topology = gaff2_engine.topology_from_worker_output(top, {"gaff2_types": ["c3", "hc", "hc", "hc"]})
            self.assertEqual(topology.labels, ["c3", "hc", "hc", "hc"])
            self.assertEqual(topology.defaults, ["1", "2", "yes", "0.5", "0.83333333"])
            names, table = tt.merge_atom_types(tt.identity_transfer(topology).atoms)
            self.assertEqual(names, ["c3", "hc", "hc", "hc"])
            self.assertEqual(len(table), 2)

    def test_type_mismatch_between_antechamber_and_parmed_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            top = Path(tmp) / "reference.top"
            top.write_text(PARMED_STYLE_TOP)
            with self.assertRaisesRegex(RuntimeError, "differ"):
                gaff2_engine.topology_from_worker_output(top, {"gaff2_types": ["c3", "hc", "hc", "h1"]})


class AmberInterpreterSearchTests(unittest.TestCase):
    def setUp(self):
        gaff2_engine.probe.cache_clear()
        self.addCleanup(gaff2_engine.probe.cache_clear)

    def test_candidates_are_ordered_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "miniforge3"
            make_conda_env(root, ["python"])
            amber = make_conda_env(root / "envs" / "amber", ["ambertools", "parmed"])
            amberhome = make_conda_env(Path(tmp) / "amberhome", ["ambertools"])
            make_conda_env(root / "envs" / "other", ["openff-toolkit"])
            environment = {gaff2_engine.AMBER_PYTHON_ENV: "/env/var/python", "AMBERHOME": str(amberhome.parent.parent)}
            with mock.patch.object(gaff2_engine, "_conda_roots", return_value=[root]), mock.patch.dict(
                os.environ, environment
            ), mock.patch.object(gaff2_engine.importlib.util, "find_spec", return_value=None):
                pythons = [python for python, _ in gaff2_engine.candidate_interpreters("/explicit/python")]
            self.assertEqual(pythons, ["/explicit/python", "/env/var/python", str(amberhome), str(amber)])

    def test_explicit_choice_is_not_replaced(self):
        candidates = [("/missing/python", "setting amber_python"), ("/other/python", "conda environment x")]
        with mock.patch.object(gaff2_engine, "candidate_interpreters", return_value=candidates):
            interpreter, problems = gaff2_engine.find_interpreter({"amber_python": "/missing/python"})
            errors, _ = gaff2_engine.preflight({"amber_python": "/missing/python"})
        self.assertIsNone(interpreter)
        self.assertEqual(len(problems), 1)
        self.assertTrue(any("does not exist" in line for line in errors))


class Gaff2PipelineWithFakeEngineTests(unittest.TestCase):
    def test_pipeline_routes_gaff2_through_the_shared_transfer(self):
        def fake_parameterize(structure_sdf, outdir, *, net_charge, name="reference", settings=None):
            outdir.mkdir(parents=True, exist_ok=True)
            top = outdir / f"{name}.top"
            fake_engine_topology(read_sdf(structure_sdf), top)
            top.write_text(top.read_text().replace("1 3 yes 0.5 0.5", "1 2 yes 0.5 0.83333333"))
            return gaff2_engine.ReferenceParameterization(
                tt.parse_gromacs_topology(top), [top], {"engine": "gaff2", "charge_model": "AM1-BCC (fake)", "engine_warnings": []}
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "monomer.sdf"
            write_sdf(mma_monomer(), monomer_sdf)
            with mock.patch.object(gaff2_engine, "preflight", return_value=([], [])), mock.patch.object(
                gaff2_engine, "parameterize_reference", side_effect=fake_parameterize
            ):
                build_polymer_from_monomer(
                    monomer_sdf,
                    MonomerBuildOptions(1, 4, 9, reference_dp=5, outdir=tmp_path / "out", minimize_geometry=False, ff_engine="gaff2"),
                )
            top = tt.parse_gromacs_topology(tmp_path / "out" / "polymer.top")
            self.assertEqual(top.defaults, ["1", "2", "yes", "0.5", "0.83333333"])
            self.assertAlmostEqual(sum(atom.charge for atom in top.atoms), 0.0, places=7)
            self.assertTrue((tmp_path / "out" / "gaff2_reference" / "reference.top").exists())
            metadata = json.loads((tmp_path / "out" / "build_metadata.json").read_text())
            self.assertEqual(metadata["ff_engine"], "gaff2")
            self.assertEqual(metadata["parameterization_mode"], "reference_to_target")

    def test_missing_ambertools_falls_back_to_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer_sdf = tmp_path / "monomer.sdf"
            write_sdf(mma_monomer(), monomer_sdf)
            with mock.patch.object(gaff2_engine, "preflight", return_value=(["AmberTools: none"], [])):
                result = build_polymer_from_monomer(
                    monomer_sdf, MonomerBuildOptions(1, 4, 6, reference_dp=5, outdir=tmp_path / "out", ff_engine="gaff2")
                )
            self.assertEqual(result.missing_tools, ["AmberTools environment (antechamber, parmchk2, tleap, ParmEd)"])


def ambertools_available() -> bool:
    try:
        return gaff2_engine.find_interpreter({})[0] is not None
    except Exception:
        return False


@unittest.skipUnless(ambertools_available(), "no AmberTools environment found (set POLYMER_AMBER_PYTHON to run these tests)")
class Gaff2IntegrationTests(unittest.TestCase):
    def build(self, tmp_path: Path, name: str, dp: int) -> Path:
        atoms, bonds, previous, nxt, _ = DIRECT_8MER_CASES[name]
        monomer_sdf = tmp_path / f"{name}.sdf"
        write_sdf(Molecule(name, atoms=atoms, bonds=bonds), monomer_sdf)
        out = tmp_path / f"{name}_dp{dp}"
        build_polymer_from_monomer(monomer_sdf, MonomerBuildOptions(previous, nxt, dp, reference_dp=5, outdir=out, ff_engine="gaff2"))
        return out

    def test_same_length_transfer_matches_the_gaff2_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.build(Path(tmp), "PEG", 5)
            reference = tt.parse_gromacs_topology(out / "gaff2_reference" / "reference.top")
            target = tt.parse_gromacs_topology(out / "polymer.top")
            key = lambda term: (term.kind, tt.canonical_atoms(term.kind, term.atoms), term.funct, term.params)
            self.assertEqual(sorted(map(key, target.terms)), sorted(map(key, reference.terms)))
            self.assertEqual(target.defaults, reference.defaults)

    def test_transfer_matches_amber_type_based_assembly_of_the_target(self):
        """tleap on the whole target (reference frcmod, transferred types and charges) gives the same terms.

        Impropers are compared in their exact written atom order: tleap orders
        some of them differently in end and interior units, and the transfer
        must reproduce that. PAA has an improper per unit (carboxylate C).
        """

        interpreter, _ = gaff2_engine.find_interpreter({})
        bin_dir = Path(interpreter.python).parent
        env = external_env.worker_environment(interpreter.python, {"AMBERHOME": interpreter.probe["amberhome"]})
        with tempfile.TemporaryDirectory() as tmp:
            out = self.build(Path(tmp), "PAA-anion", 9)
            old = out / "tleap_target"
            old.mkdir()
            (old / "tleap.in").write_text(
                "source leaprc.gaff2\n"
                f"loadamberparams {out / 'gaff2_reference' / 'reference.frcmod'}\n"
                f"mol = loadmol2 {out / 'polymer_typed_charged.mol2'}\n"
                "saveamberparm mol t.prmtop t.inpcrd\nquit\n"
            )
            subprocess.run([str(bin_dir / "tleap"), "-f", "tleap.in"], cwd=old, env=env, check=True, capture_output=True)
            subprocess.run(
                [interpreter.python, "-c", "import parmed; parmed.load_file('t.prmtop', 't.inpcrd').save('t.top', overwrite=True)"],
                cwd=old, env=env, check=True, capture_output=True,
            )
            leap = tt.parse_gromacs_topology(old / "t.top")
            ours = tt.parse_gromacs_topology(out / "polymer.top")
            self.assertGreater(sum(1 for term in ours.terms if term.kind == "impropers"), 0)
            self.assertEqual(exact_term_table(leap), exact_term_table(ours))
            for leap_atom, our_atom in zip(leap.atoms, ours.atoms):
                self.assertEqual(leap_atom.type_name, our_atom.type_name)
                self.assertAlmostEqual(leap_atom.charge, our_atom.charge, places=6)


if __name__ == "__main__":
    unittest.main()
