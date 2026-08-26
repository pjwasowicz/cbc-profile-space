"""W9 — audit of discarded ("undefined") records; appendix to W1.

Discretisation discards a record whenever any of the 14 analytes has a missing
result or reference range (P = NaN). We check whether that exclusion is random
or systematic (selection bias), because a reviewer will ask.

Reports:
- how many records were discarded and which analyte is most often responsible,
- the distribution of the number of missing analytes per record (single gaps vs
  entirely empty records),
- a comparison of the core analytes (present despite other gaps) between kept
  and discarded records — the bias test.

Usage:
    python src/w9_undefined.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, compute_p, load_data

# core analytes, almost always present — used for the kept vs dropped comparison
CORE = ["HGB", "WBC", "PLT", "RBC", "HCT"]


def main() -> None:
    common.set_seed()
    exp = common.experiment_dir("w9_undefined")
    df = load_data()
    N_raw = len(df)

    # P matrix and the mask of missing values
    P = np.column_stack([compute_p(df, PARAMS_14[s]).to_numpy(float) for s in PARAMS_14])
    isna = np.isnan(P)
    nmiss = isna.sum(axis=1)
    dropped = nmiss > 0
    n_dropped = int(dropped.sum())

    # which analyte is missing most often
    miss_by_param = {s: int(isna[:, j].sum()) for j, s in enumerate(PARAMS_14)}
    miss_by_param = dict(sorted(miss_by_param.items(), key=lambda x: -x[1]))

    # distribution of the gap count among discarded records
    vals, cnts = np.unique(nmiss[dropped], return_counts=True)
    dist_nmiss = {int(v): int(c) for v, c in zip(vals, cnts)}
    n_all_missing = dist_nmiss.get(14, 0)      # entirely empty records

    # bias check: kept vs dropped (empty records excluded) on the core analytes
    empty = nmiss == 14
    dropped_nonempty = dropped & ~empty
    kept = ~dropped
    bias = {}
    for s in CORE:
        col = df[f"RESULT_{PARAMS_14[s]}"].to_numpy(float)
        k = col[kept]; d = col[dropped_nonempty]
        k = k[~np.isnan(k)]; d = d[~np.isnan(d)]
        bias[s] = {
            "kept_median": round(float(np.median(k)), 3),
            "dropped_median": round(float(np.median(d)), 3),
            "kept_mean": round(float(np.mean(k)), 3),
            "dropped_mean": round(float(np.mean(d)), 3),
            "n_dropped_with_value": int(len(d)),
        }

    summary = {
        "n_raw": N_raw,
        "n_dropped": n_dropped,
        "dropped_fraction": round(n_dropped / N_raw, 5),
        "n_fully_empty_records": n_all_missing,
        "missing_by_param": miss_by_param,
        "dist_n_missing_among_dropped": dist_nmiss,
        "core_analyte_bias_kept_vs_dropped": bias,
        "note": ("Exclusion is NOT random: it is dominated by missing RDW-SD (an "
                 "analyte that is not always reported) plus a pool of entirely "
                 "empty records. The core-analyte comparison tests whether the "
                 "discarded records (empty ones aside) differ clinically from "
                 "the records that were kept."),
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    common.write_config(exp, {"experiment": "w9_undefined", "core_analytes": CORE})
    common.write_env(exp)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
