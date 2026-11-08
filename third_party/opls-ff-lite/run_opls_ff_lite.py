#!/usr/bin/env python3
"""Run opls-ff-lite straight from the source tree, without installing it.

Use any Python 3.12+ that can import RDKit and NumPy, for example one of your
existing conda environments:

    /path/to/env/bin/python run_opls_ff_lite.py doctor
    /path/to/env/bin/python run_opls_ff_lite.py install-si jp3c06602_si_002.txt jp3c06602_si_003.txt
    /path/to/env/bin/python run_opls_ff_lite.py --smiles 'CCO' --name ethanol --output-dir out/ethanol

The arguments are those of the installed `opls-ff-lite` command. MOPAC is
found the same way: OPLS_FF_LITE_MOPAC, the config file
(`run_opls_ff_lite.py config set mopac PATH`), PATH. RDKit and NumPy must be
importable by the Python that runs this script.
"""

import importlib.util
import sys
from pathlib import Path

if sys.version_info < (3, 12):
    sys.exit(f"opls-ff-lite needs Python 3.12 or newer; {sys.executable} is "
             f"{sys.version.split()[0]}. Run this script with a newer Python.")

missing = [name for name in ("rdkit", "numpy") if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(
        f"opls-ff-lite needs {' and '.join(missing)}, which {sys.executable} cannot import.\n"
        "Run this script with a Python that has them, e.g.\n"
        "    /path/to/env/bin/python run_opls_ff_lite.py ...\n"
        "or install them into this one:\n"
        "    conda install -c conda-forge 'rdkit>=2025.9' numpy   (or: pip install rdkit numpy)")

# the source tree comes first, so an installed copy of the package is not used
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from opls_ff_lite.__main__ import main  # noqa: E402

main()
