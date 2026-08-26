"""W3 — algebraic redundancy: the profile space over 14 vs 10 vs 9 analytes.

Part 1 — algebraic residuals (Sysmex XN: HCT measured, red cell indices derived):
    r_mcv  = MCV  - HCT/RBC*10
    r_mch  = MCH  - HGB/RBC*10
    r_mchc = MCHC - HGB/HCT*100
    r_wbc  = WBC  - (NEUT+LYMPH+MONO+EOS+BASO)
If |residual| is of the order of the reporting rounding step, the analyte is
computed rather than measured.

Part 2 — table of space variants (at a common margin thr):
    V14  the full 14 analytes
    V10  without {MCV, MCH, MCHC, BASO} - pure algebraic redundancy
         (3 derived red cell indices + 1 fraction, since WBC = sum of 5 fractions)
    V9   V10 - {WBC} - the leukocyte aggregate removed as well (sensitivity check:
         one real but weakly informative axis fewer)
For each variant: number of profiles, 90% coverage, singleton fraction.

Per laboratory/analyser: skipped - the data carry no column identifying the
analyser (recorded as a limitation).

Usage:
    python src/w3_redundancy.py [--margin THR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, discretize, load_data, profile_ids

# space variants as subsets of the PARAMS_14 short names
DROP_V10 = ["MCV", "MCH", "MCHC", "BASO"]
DROP_V9 = DROP_V10 + ["WBC"]
VARIANTS = {
    "V14": list(PARAMS_14),
    "V10": [s for s in PARAMS_14 if s not in DROP_V10],
    "V9": [s for s in PARAMS_14 if s not in DROP_V9],
}

# algebraic identities: (name, residual function, reporting precision -> rounding threshold)
IDENTITIES = {
    "r_mcv": (["MCV", "HCT", "RBC"], lambda R: R["MCV"] - R["HCT"] / R["RBC"] * 10, 0.05),
    "r_mch": (["MCH", "HGB", "RBC"], lambda R: R["MCH"] - R["HGB"] / R["RBC"] * 10, 0.05),
    "r_mchc": (["MCHC", "HGB", "HCT"], lambda R: R["MCHC"] - R["HGB"] / R["HCT"] * 100, 0.05),
    "r_wbc": (["WBC", "NEUT", "LYMPH", "MONO", "EOS", "BASO"],
              lambda R: R["WBC"] - (R["NEUT"] + R["LYMPH"] + R["MONO"] + R["EOS"] + R["BASO"]),
              0.005),
}


def coverage90(counts: np.ndarray) -> int:
    cum = np.cumsum(counts) / counts.sum()
    return int(np.searchsorted(cum, 0.90) + 1)


def residual_stats(df) -> dict:
    out = {}
    for name, (needed, fn, round_thr) in IDENTITIES.items():
        R = {s: df[f"RESULT_{PARAMS_14[s]}"] for s in needed}
        r = fn(R).dropna()
        a = r.abs()
        out[name] = {
            "n": int(len(r)),
            "median_abs": round(float(a.median()), 5),
            "p90_abs": round(float(a.quantile(0.90)), 5),
            "p99_abs": round(float(a.quantile(0.99)), 5),
            "max_abs": round(float(a.max()), 3),
            "rounding_threshold": round_thr,
            "frac_within_rounding": round(float((a <= round_thr).mean()), 5),
            "frac_beyond_0.5": round(float((a > 0.5).mean()), 6),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    args = ap.parse_args()

    common.set_seed()
    exp = common.experiment_dir("w3_redundancy")

    df = load_data()

    # Part 1: algebraic residuals (on raw results, no thresholding)
    residuals = residual_stats(df)

    # Part 2: variant table at a common margin.
    # Note: each variant filters "undefined" using ONLY its own analytes,
    # so N may differ slightly between variants.
    variants_out = {}
    for name, shorts in VARIANTS.items():
        params = {s: PARAMS_14[s] for s in shorts}
        ordinal = discretize(df, margin=args.margin, params=params)
        vc = profile_ids(ordinal).value_counts()
        counts = vc.to_numpy()
        n_singwhen = int((counts == 1).sum())
        variants_out[name] = {
            "n_params": len(shorts),
            "params": shorts,
            "n_records": int(len(ordinal)),
            "n_profiles": int(len(counts)),
            "coverage90_profiles": coverage90(counts),
            "top1_share": round(float(counts[0] / len(ordinal)), 4),
            "n_singletons": n_singwhen,
            "singleton_fraction": round(n_singwhen / len(counts), 4),
        }

    summary = {
        "margin": args.margin,
        "residuals": residuals,
        "variants": variants_out,
        "per_lab": "skipped - the data carry no column identifying the analyser",
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # variant table as CSV
    import pandas as pd
    rows = [{"variant": k, **{kk: vv for kk, vv in v.items() if kk != "params"}}
            for k, v in variants_out.items()]
    pd.DataFrame(rows).to_csv(exp / "variants.csv", index=False)

    common.write_config(exp, {"experiment": "w3_redundancy", "margin": args.margin,
                              "variants": {k: v for k, v in VARIANTS.items()}})
    common.write_env(exp)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
