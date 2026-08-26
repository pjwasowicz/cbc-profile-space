"""W1 — structure of the discretised CBC profile space.

Computes: the number of records and of unique profiles, the cumulative coverage
curve, how many profiles are needed to cover 50/80/90/95/99% of records, the
share of the top-1 profile, the singleton fraction, and the power-law fit
(Zipf OLS + Clauset MLE). Writes a CSV of profiles with their counts, two
figures and env/config.

Usage:
    python src/w1_structure.py [--margin THR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, discretize, load_data, profile_ids

COVERAGE_TARGETS = [0.50, 0.80, 0.90, 0.95, 0.99]


def profiles_for_coverage(counts: np.ndarray, target: float) -> int:
    """How many of the most frequent profiles cover the `target` fraction of records."""
    cum = np.cumsum(counts) / counts.sum()
    return int(np.searchsorted(cum, target) + 1)


def zipf_ols(counts: np.ndarray) -> dict:
    """Slope of the log-log line for the rank-frequency relation."""
    rank = np.arange(1, len(counts) + 1)
    x, y = np.log(rank), np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"zipf_slope": float(slope), "zipf_r2": float(r2)}


def clauset_powerlaw(counts: np.ndarray) -> dict:
    """MLE of the alpha exponent of the profile-count distribution, with xmin selection (KS).

    Continuous approximation (Clauset, Shalizi, Newman 2009): for a given xmin,
    alpha = 1 + n / sum(ln(x_i / xmin)); xmin is chosen by the minimum
    Kolmogorov-Smirnov statistic between the empirical and theoretical tail CDF.
    """
    x = np.sort(counts.astype(float))
    candidates = np.unique(x)
    candidates = candidates[candidates >= 1]
    best = None
    for xmin in candidates[:-2] if len(candidates) > 2 else candidates:
        tail = x[x >= xmin]
        n = len(tail)
        if n < 10:
            continue
        alpha = 1.0 + n / np.sum(np.log(tail / xmin))
        # empirical vs theoretical tail CDF
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
        ks = np.max(np.abs(cdf_emp - cdf_theo))
        if best is None or ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": float(xmin), "ks": float(ks), "n_tail": int(n)}
    return best or {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}


def fig_coverage(counts: np.ndarray, out: Path) -> None:
    cum = np.cumsum(counts) / counts.sum()
    k = np.arange(1, len(counts) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(k, cum, color="#2b6cb0", lw=1.6)
    for t in (0.90,):
        n = profiles_for_coverage(counts, t)
        ax.axhline(t, color="#a0aec0", ls="--", lw=0.8)
        ax.axvline(n, color="#a0aec0", ls="--", lw=0.8)
        ax.annotate(f"{n} profiles -> {int(t*100)}%", (n, t), xytext=(8, -12),
                    textcoords="offset points", fontsize=8, color="#4a5568")
    ax.set_xscale("log")
    ax.set_xlabel("Number of most frequent profiles (log scale)")
    ax.set_ylabel("Cumulative coverage of records")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_rank_frequency(counts: np.ndarray, pl: dict, out: Path) -> None:
    rank = np.arange(1, len(counts) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(rank, counts, ".", ms=2.5, color="#2b6cb0", alpha=0.5)
    ax.set_xlabel("Profile rank")
    ax.set_ylabel("Count")
    if pl.get("alpha"):
        ax.set_title(f"Clauset MLE α={pl['alpha']:.2f} (x_min={pl['xmin']:.0f}, KS={pl['ks']:.3f})",
                     fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    args = ap.parse_args()

    common.set_seed()
    exp = common.experiment_dir("w1_structure")

    df = load_data()
    ordinal = discretize(df, margin=args.margin)
    n_records = len(ordinal)

    pid = profile_ids(ordinal)
    vc = pid.value_counts()                       # sorted in decreasing order
    counts = vc.to_numpy()
    n_profiles = len(counts)

    coverage = {f"{int(t*100)}%": profiles_for_coverage(counts, t) for t in COVERAGE_TARGETS}
    top1_share = float(counts[0] / n_records)
    n_singletons = int((counts == 1).sum())
    singleton_frac = n_singletons / n_profiles

    zipf = zipf_ols(counts)
    pl = clauset_powerlaw(counts)

    # profile CSV (aggregate, no PII) - for Data Availability / Zenodo
    prof_df = vc.rename_axis("profile").reset_index(name="count")
    prof_df["percent"] = prof_df["count"] / n_records * 100
    prof_df["percent_cumulative"] = prof_df["percent"].cumsum()
    prof_df.to_csv(exp / "profile_counts.csv", index=False)

    fig_coverage(counts, exp / "fig_coverage.png")
    fig_rank_frequency(counts, pl, exp / "fig_rank_frequency.png")

    summary = {
        "n_records": int(n_records),
        "n_records_raw": int(len(df)),
        "n_dropped_undefined": int(len(df) - n_records),
        "n_profiles": int(n_profiles),
        "coverage_profiles": coverage,
        "top1_share": top1_share,
        "n_singletons": n_singletons,
        "singleton_fraction": singleton_frac,
        "zipf_ols": zipf,
        "powerlaw_clauset": pl,
        "margin": args.margin,
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    common.write_config(exp, {"experiment": "w1_structure", "margin": args.margin,
                              "params": list(PARAMS_14), "coverage_targets": COVERAGE_TARGETS})
    common.write_env(exp)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
