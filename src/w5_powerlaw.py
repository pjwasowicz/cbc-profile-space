"""W5 — power-law goodness of fit (Clauset, Shalizi, Newman 2009); supports W1.

Fits a power law to the distribution of profile COUNTS (n_i) by MLE, selecting
x_min through the minimum KS statistic, then computes a bootstrap p-value: it
generates synthetic data sets (tail ~ the fitted power law, the part below x_min
resampled empirically), refits each one and compares its KS against the KS of
the data.

Interpretation: p >= 0.1 means the power law is a plausible hypothesis (not
rejected). p < 0.1 calls for cautious wording ("consistent with", not "follows a
power law").

Note: continuous approximation (the counts are large); x_min candidates are
capped at XMIN_CAP for speed (the actual x_min is on the order of tens).

Usage:
    python src/w5_powerlaw.py [--margin THR] [--B 200]
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

XMIN_CAP = 300          # only consider x_min <= this value (the true x_min is small)
MIN_TAIL = 50           # minimum tail size accepted when fitting


def fit_powerlaw(x: np.ndarray, candidates: np.ndarray):
    """MLE alpha plus x_min selection by minimum KS (continuous approximation)."""
    x = np.sort(x.astype(float))
    best = None
    for xmin in candidates:
        tail = x[x >= xmin]
        n = tail.size
        if n < MIN_TAIL:
            continue
        s = np.log(tail / xmin)
        denom = s.sum()
        if denom <= 0:
            continue
        alpha = 1.0 + n / denom
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
        D = np.max(np.abs(cdf_emp - cdf_theo))
        if best is None or D < best["D"]:
            best = {"alpha": alpha, "xmin": xmin, "D": D, "ntail": n}
    return best


def candidate_xmins(x: np.ndarray) -> np.ndarray:
    u = np.unique(x)
    return u[(u >= 1) & (u <= XMIN_CAP)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    ap.add_argument("--B", type=int, default=200, help="number of bootstrap replicates")
    args = ap.parse_args()

    common.set_seed()
    rng = np.random.default_rng(common.SEED)
    exp = common.experiment_dir("w5_powerlaw")

    df = load_data()
    ordinal = discretize(df, margin=args.margin)
    counts = profile_ids(ordinal).value_counts().to_numpy().astype(float)
    S = counts.size

    cand = candidate_xmins(counts)
    fit = fit_powerlaw(counts, cand)
    alpha, xmin, D_data, ntail = fit["alpha"], fit["xmin"], fit["D"], fit["ntail"]
    p_tail = ntail / S
    below = counts[counts < xmin]

    # bootstrap p-value (Clauset)
    ge = 0
    for _ in range(args.B):
        n_pl = int(rng.binomial(S, p_tail))
        u = rng.random(n_pl)
        pl_samp = xmin * (1 - u) ** (-1.0 / (alpha - 1.0))          # continuous power law
        if below.size and S - n_pl > 0:
            below_samp = rng.choice(below, size=S - n_pl, replace=True)
        else:
            below_samp = np.empty(0)
        synth = np.concatenate([pl_samp, below_samp])
        cand_s = np.unique(synth)
        cand_s = cand_s[cand_s <= XMIN_CAP]
        fit_s = fit_powerlaw(synth, cand_s)
        if fit_s and fit_s["D"] >= D_data:
            ge += 1
    p_value = ge / args.B

    result = {
        "margin": args.margin,
        "B": args.B,
        "S_profiles": int(S),
        "fit": {"alpha": round(float(alpha), 4), "xmin": float(xmin),
                "KS_D": round(float(D_data), 4), "ntail": int(ntail)},
        "p_value": round(p_value, 3),
        "verdict": ("power law is plausible (not rejected)" if p_value >= 0.1
                    else "poor fit - write 'consistent with', not 'follows'"),
        "note": ("p = fraction of synthetic data sets with KS >= KS of the data; "
                 f"x_min candidates <= {XMIN_CAP}, minimum tail {MIN_TAIL}."),
    }
    (exp / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    common.write_config(exp, {"experiment": "w5_powerlaw", "margin": args.margin,
                              "B": args.B, "xmin_cap": XMIN_CAP, "min_tail": MIN_TAIL})
    common.write_env(exp)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
