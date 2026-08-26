"""Runs the full W1-W9 pipeline in order on the current `morfologia_data_new.parquet`.

The ordering is numeric and logical at once: W1-W4 are the four headline results
from the plan of the study, W5-W9 are supporting analyses and
appendices. Manuscript inputs are built separately by
`paper/fill_numbers.py`. Every step writes into `results/<name>/`.

Usage:
    python src/run_all.py                 # everything
    python src/run_all.py w1 w2           # selected steps
    python src/run_all.py --margin THR    # margin for W1/W3/W4/W5/W6/W7
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

# (module, accepts --margin) - W2 scans the margin grid itself, while W8 and W9
# work on raw results and therefore take no margin.
STEPS: list[tuple[str, bool]] = [
    ("w1_structure", True),
    ("w2_threshold", False),
    ("w3_redundancy", True),
    ("w4_unseen", True),
    ("w5_powerlaw", True),
    ("w6_geometry", True),
    ("w7_spaceviz", True),
    ("w8_correlation", False),
    ("w9_undefined", False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="Which steps to run (e.g. w1 w4). Default: all.")
    ap.add_argument("--margin", type=float, default=None, help="Discretisation margin for the steps that accept one.")
    args = ap.parse_args()

    wanted = [s.lower().rstrip("_") for s in args.steps]
    steps = [(m, f) for m, f in STEPS
             if not wanted or any(m == w or m.startswith(w + "_") for w in wanted)]
    if not steps:
        sys.exit(f"Nothing matches {args.steps}. Available: {[m for m, _ in STEPS]}")

    failed = []
    for num, (module, takes_margin) in enumerate(steps, 1):
        cmd = [sys.executable, str(SRC / f"{module}.py")]
        if takes_margin and args.margin is not None:
            cmd += ["--margin", str(args.margin)]

        print(f"\n{'=' * 70}\n[{num}/{len(steps)}] {module}\n{'=' * 70}", flush=True)
        started = time.time()
        rc = subprocess.run(cmd).returncode
        took = time.time() - started
        if rc == 0:
            print(f"--- {module}: OK ({took:.1f}s)")
        else:
            print(f"--- {module}: FAILED (exit {rc}, {took:.1f}s)")
            failed.append(module)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"Failed steps: {', '.join(failed)}")
        return 1
    print(f"All steps ({len(steps)}) completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
