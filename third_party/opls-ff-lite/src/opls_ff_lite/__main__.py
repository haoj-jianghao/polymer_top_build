"""Command-line interface of opls-ff-lite: OPLS/2020 + CM1A-LBCC GROMACS topologies
with MOPAC as the only external program (no QM fit, no xTB, no OpenMM/GROMACS)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem

from .assignment import DEFAULT_ATOMTYPE_ARCHIVE
from .structure import prepare_structure
from .workflow import generate_gromacs_topology


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opls-ff-lite",
        allow_abbrev=False,  # so that e.g. --qm is unknown, not taken for --qm-bonded-table
        description=(
            "Generate a GROMACS-ready OPLS/2020 topology for a molecule or oligomer, "
            "neutral or charged: OPLS/2020 atom types and parameters (SMARTS typing, "
            "GAFF2 fallbacks for atoms OPLS cannot type), CM1A charges from MOPAC AM1 "
            "(1.14*CM1A-LBCC for neutral molecules, unscaled CM1A for ions), and bonded "
            "terms from the OPLS/2020 tables with table-based fallbacks for missing terms. "
            "Geometry: RDKit MMFF94. No QM fit."
        ),
        epilog="Other commands: 'opls-ff-lite doctor' (status of every dependency), "
        "'opls-ff-lite config' (MOPAC path), "
        "'opls-ff-lite install-si' (install the OPLS/2020 SI files).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--structure", type=Path,
                        help="3D structure file: .sdf/.mol (one record), .mol2 or .pdb")
    source.add_argument("--sdf", type=Path, help="one-record 3D SDF (same as --structure)")
    source.add_argument("--smiles", help="SMILES; a 3D conformer is generated (ETKDGv3) "
                        "and optimized with MMFF94")
    parser.add_argument("--optimize", choices=("none", "mmff"),
                        help="geometry before the charges: none = the input coordinates "
                        "(default for structure files), mmff = RDKit MMFF94 (always for SMILES)")
    parser.add_argument("--template-smiles",
                        help="PDB or MOL2 input: assign bond orders and formal charges "
                        "from this SMILES")
    parser.add_argument("--net-charge", type=int,
                        help="net charge: used for PDB bond-order perception and checked "
                        "against the formal charges of any structure file")
    parser.add_argument("--name", help="molecule/output name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--charge-model", choices=("auto", "cm1a", "cm1a-lbcc"), default="auto",
                        help="auto (default): 1.14*CM1A-LBCC for neutral molecules (zwitterions "
                        "included), unscaled CM1A for ions")
    parser.add_argument("--torsion-substitution", choices=("parmchk2", "none"),
                        default="parmchk2",
                        help="torsions missing from OPLS/2020: GAFF2-derived general term, "
                        "parmchk2-style substitution and fallback, else zero (default); or "
                        "none (a missing torsion is an error)")
    parser.add_argument("--qm-bonded-table", choices=("use", "none"), default="use",
                        help="bonds and angles missing from OPLS/2020: values recorded from "
                        "earlier QM fits of opls-ff-gen (data/qm_bonded_terms.par; no QM is "
                        "run) before the fallback (default); or none")
    parser.add_argument("--no-amber-fallback", action="store_true",
                        help="fail on atoms OPLS cannot type instead of using GAFF2 types and LJ")
    parser.add_argument("--atomtype-archive", type=Path, default=DEFAULT_ATOMTYPE_ARCHIVE,
                        help="atom-type archive: extra SMARTS, precedence, OPLS/2020 "
                        "refinements and borrowed types")
    parser.add_argument("--no-atomtype-archive", action="store_true",
                        help="type with the OPLS-AA rule set only (data/opls_rules.json)")
    parser.add_argument("--timeout", type=int, default=300, help="MOPAC time limit (s)")
    parser.add_argument("--box-padding-nm", type=float, default=1.0,
                        help="padding around the molecule in the generated GRO box")
    return parser


CONFIG_HELP = """\
opls-ff-lite config: where opls-ff-lite finds MOPAC.

  opls-ff-lite config show                 list the setting and the config file
  opls-ff-lite config set mopac PATH       use this MOPAC (checked to be executable)
  opls-ff-lite config unset mopac          forget it (fall back to PATH)

The environment variable OPLS_FF_LITE_MOPAC takes precedence over the file.
Check with: opls-ff-lite doctor
"""


def _config_command(argv: list[str]) -> int:
    from . import tools

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(CONFIG_HELP, end="")
        return 0
    action, rest = argv[0], argv[1:]
    try:
        if action == "show" and not rest:
            values = tools.read_config()
            print(f"config file: {tools.config_path()}"
                  + ("" if tools.config_path().is_file() else " (none)"))
            for key in tools.KEYS:
                print(f"  {key:<8} {values.get(key, '(not set)')}")
            return 0
        if action == "set" and len(rest) == 2:
            path = tools.set_config(rest[0], rest[1])
            print(f"{rest[0]} = {tools.read_config()[rest[0]]}  ({path})")
            return 0
        if action == "unset" and len(rest) == 1:
            path = tools.set_config(rest[0], None)
            print(f"{rest[0]} removed ({path})")
            return 0
    except (ValueError, RuntimeError) as exc:
        print(f"opls-ff-lite config: {exc}", file=sys.stderr)
        return 2
    print(CONFIG_HELP, end="", file=sys.stderr)
    return 2


def _doctor_command(argv: list[str]) -> int:
    from . import tools

    if argv:
        print("usage: opls-ff-lite doctor  (status of every dependency)", file=sys.stderr)
        return 2
    report, required_ok = tools.doctor()
    print(report, end="")
    return 0 if required_ok else 1


def _install_si_command(argv: list[str]) -> int:
    from . import si

    parser = argparse.ArgumentParser(
        prog="opls-ff-lite install-si",
        description="Install the OPLS/2020 Supporting Information files "
        f"(doi:{si.DOI}; not redistributable, so download them yourself from the "
        "article page). Both files are checked against the release opls-ff-lite was "
        "validated with and copied to a per-user directory that survives upgrades "
        f"(default {si.user_si_dir()}).")
    parser.add_argument("files", nargs="+", type=Path,
                        help="jp3c06602_si_002.txt and jp3c06602_si_003.txt (any order)")
    parser.add_argument("--dest", type=Path,
                        help=f"install here instead (then set {si.SI_DIR_ENV} to it)")
    parser.add_argument("--force", action="store_true",
                        help="install files that differ from the validated release")
    args = parser.parse_args(argv)
    try:
        report = si.install(args.files, dest=args.dest, force=args.force)
    except si.SIInstallError as exc:
        print(f"opls-ff-lite install-si: {exc}", file=sys.stderr)
        return 2
    print("\n".join(report))
    if args.dest is not None:
        print(f"Set {si.SI_DIR_ENV}={args.dest.expanduser().resolve()} so that opls-ff-lite "
              "reads them from there.")
    directory, source = si.si_dir()
    print(f"opls-ff-lite reads the SI from: {directory} [{source}]")
    return 0


COMMANDS = {"config": _config_command, "doctor": _doctor_command,
            "install-si": _install_si_command}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        raise SystemExit(COMMANDS[argv[0]](argv[1:]))
    args = build_parser().parse_args(argv)
    from .tools import preflight, run_version_notes

    errors, warnings = preflight()
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        print("opls-ff-lite: cannot start:\n  " + "\n  ".join(errors)
              + "\nStatus of every dependency: opls-ff-lite doctor", file=sys.stderr)
        raise SystemExit(2)
    for note in run_version_notes():
        print(f"NOTE: {note}", file=sys.stderr)

    structure_path = args.structure or args.sdf
    name = args.name or (structure_path.stem if structure_path else "molecule")
    molecule = prepare_structure(
        smiles=args.smiles,
        path=structure_path,
        name=name,
        optimize=args.optimize,
        template_smiles=args.template_smiles,
        net_charge=args.net_charge,
    )
    result = generate_gromacs_topology(
        molecule,
        args.output_dir,
        name=name,
        charge_model=args.charge_model,
        atomtype_archive=None if args.no_atomtype_archive else args.atomtype_archive,
        amber_fallback=not args.no_amber_fallback,
        torsion_substitution=args.torsion_substitution == "parmchk2",
        qm_bonded_table=args.qm_bonded_table == "use",
        timeout_seconds=args.timeout,
        box_padding_nm=args.box_padding_nm,
    )
    print(
        f"{result.assignment.name}: {result.assignment.atom_count} atoms; "
        f"net charge {Chem.GetFormalCharge(molecule):+d}; {result.assignment.charge_model}; "
        f"final charge={result.assignment.final_charge_sum_e:+.10f} e"
    )
    rule_counts = ", ".join(
        f"{count} {rule}" for rule, count in result.bonded.rule_counts().items()
    )
    print(f"bonded terms: {rule_counts}")
    for warning in result.bonded.warnings:
        print(f"WARNING: {warning}")
    print(f"GROMACS ITP:     {result.itp_path}")
    print(f"GROMACS GRO:     {result.gro_path}")
    print(f"GROMACS TOP:     {result.top_path}")
    print(f"topology JSON:   {result.topology_json_path}")
    print(f"assignment JSON: {result.assignment.json_path}")
    print(f"assigned SDF:    {result.assignment.sdf_path}")
    print(f"structure read:  {Chem.MolToSmiles(Chem.RemoveHs(molecule))} "
          f"(net charge {Chem.GetFormalCharge(molecule):+d})")
    print(f"geometry:        {molecule.GetProp('Geometry_Optimization')}")


if __name__ == "__main__":
    main()
