"""W2 — encoding resolution vs the discretisation margin, C(thr).

Justifies the choice of margin WITHOUT the argument "it maximises diversity":
we show that the encoding resolution has a FLAT maximum, so the choice is
robust, and we report the plateau it sits in.

Over a dense grid of margins (0.08-0.22 in steps of 0.01, plus sparse points)
it computes:
  - C(thr)     = number of unique profiles
  - cov90(thr) = how many of the most frequent profiles cover 90% of records
    (the frequency-weighted view, less sensitive to rare profiles)
plus a bootstrap over 50% subsamples: where argmax C(thr) falls across replicates.

P is computed ONCE; each margin only re-bins it, which is fast.

Usage:
    python src/w2_threshold.py [--boot 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, compute_p, discretize_series, load_data

# base-5 profile code (14 states 0..4) -> a single int64; 5**14 ~ 6.1e9 fits in int64
POW5 = (5 ** np.arange(14)).astype(np.int64)

DENSE = [round(0.08 + 0.01 * i, 2) for i in range(15)]      # 0.08..0.22
SPARSE = [0.00, 0.05, 0.25, 0.40, 0.50]
GRID = sorted(set(DENSE + SPARSE))


_T0 = None


def log(msg: str) -> None:
    """Progress message on stderr, stamped with elapsed time."""
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


def encode(P: np.ndarray, margin: float) -> np.ndarray:
    """Profile codes (int64) at margin `margin`, for all records at once."""
    states = np.empty(P.shape, dtype=np.int8)
    for j in range(P.shape[1]):
        states[:, j] = discretize_series(P[:, j], margin=margin)
    return states.astype(np.int64) @ POW5


def n_profiles(codes: np.ndarray) -> int:
    return int(np.unique(codes).size)


def cov90(codes: np.ndarray) -> int:
    counts = np.sort(np.unique(codes, return_counts=True)[1])[::-1]
    cum = np.cumsum(counts) / counts.sum()
    return int(np.searchsorted(cum, 0.90) + 1)


def compute_plateau(thr: np.ndarray, C: np.ndarray, tol: float):
    """Computed plateau: the contiguous margin interval where C(thr) stays within
    `tol` of the maximum and which contains the argmax. Returns (thr_argmax, lo, hi)."""
    order = np.argsort(thr)
    thr, C = thr[order], C[order]
    imax = int(np.argmax(C))
    cutoff = C[imax] * (1.0 - tol)
    lo = imax
    while lo - 1 >= 0 and C[lo - 1] >= cutoff:
        lo -= 1
    hi = imax
    while hi + 1 < len(C) and C[hi + 1] >= cutoff:
        hi += 1
    return float(thr[imax]), float(thr[lo]), float(thr[hi])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200, help="number of 50% bootstrap replicates")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="plateau tolerance: margins with C >= (1-tol)*Cmax (default 1%)")
    args = ap.parse_args()

    common.set_seed()
    rng = np.random.default_rng(common.SEED)
    exp = common.experiment_dir("w2_threshold")

    log("loading parquet...")
    df = load_data()
    # P once for the 14 analytes; drop records with any NaN (undefined is margin-independent)
    P = np.column_stack([compute_p(df, PARAMS_14[s]).to_numpy(float) for s in PARAMS_14])
    valid = ~np.isnan(P).any(axis=1)
    P = P[valid]
    N = P.shape[0]
    log(f"P computed: N={N} valid records, {P.shape[1]} analytes")

    # full sample
    log(f"full sample: grid of {len(GRID)} margins {GRID}")
    rows = []
    for i, thr in enumerate(GRID, 1):
        codes = encode(P, thr)
        C = n_profiles(codes)
        rows.append({"thr": thr, "C": C, "cov90": cov90(codes)})
        log(f"  [{i}/{len(GRID)}] thr={thr:.2f}  C={C}")
    thr_arr = np.array([r["thr"] for r in rows], dtype=float)
    C_arr = np.array([r["C"] for r in rows], dtype=float)
    argmax_full, plateau_lo, plateau_hi = compute_plateau(thr_arr, C_arr, args.tol)
    # Relative change is quoted against the selected margin, i.e. the argmax of
    # C(thr) found right above, so the column follows the data instead of a
    # margin fixed by hand.
    ref = next(r["C"] for r in rows if r["thr"] == argmax_full)
    for r in rows:
        r["per10k"] = round(r["C"] / N * 10000, 2)
        r["deltaC_pct_vs_argmax"] = round((r["C"] - ref) / ref * 100, 2)
    log(f"full sample done: argmax C(thr) = {argmax_full}; "
        f"plateau (tol {args.tol:.0%}) = [{plateau_lo}, {plateau_hi}]")

    # 50% bootstrap - where argmax C(thr) lands on the dense grid
    m = N // 2
    argmax_counts: dict[float, int] = {t: 0 for t in DENSE}
    log(f"bootstrap: {args.boot} replicates x {len(DENSE)} margins, 50% subsample (m={m})")
    t_boot = time.perf_counter()
    for b in range(1, args.boot + 1):
        idx = rng.choice(N, size=m, replace=False)
        Pb = P[idx]
        best_thr, best_C = None, -1
        for thr in DENSE:
            C = n_profiles(encode(Pb, thr))
            if C > best_C:
                best_C, best_thr = C, thr
        argmax_counts[best_thr] += 1
        if b % 10 == 0 or b == args.boot:
            elapsed = time.perf_counter() - t_boot
            eta = elapsed / b * (args.boot - b)
            top = max(argmax_counts, key=argmax_counts.get)
            log(f"  boot {b}/{args.boot}  ETA {eta:4.0f}s  "
                f"argmax leader: thr={top} ({argmax_counts[top]}/{b})")
    argmax_frac = {t: c / args.boot for t, c in argmax_counts.items()}

    # C(thr) figure: computed argmax and computed plateau (nothing hard-coded)
    order = np.argsort(thr_arr)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(thr_arr[order], C_arr[order], "-o", ms=3, color="#2b6cb0")
    ax.axvspan(plateau_lo, plateau_hi, color="#c6f6d5", alpha=0.5,
               label=f"plateau {plateau_lo:g}-{plateau_hi:g} (C >= {1-args.tol:.0%} Cmax)")
    ax.axvline(argmax_full, color="#e53e3e", ls="--", lw=1,
               label=f"argmax C(thr) = {argmax_full:g}")
    ax.set_xlabel("Discretisation margin thr")
    ax.set_ylabel("Number of unique profiles C(thr)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(exp / "fig_C_thr.png", dpi=200)
    plt.close(fig)

    import pandas as pd
    pd.DataFrame(rows).sort_values("thr").to_csv(exp / "c_thr.csv", index=False)

    summary = {
        "N_valid": int(N),
        "grid": GRID,
        "argmax_C_full_sample": argmax_full,
        "plateau_tol": args.tol,
        "plateau_thr_range": [plateau_lo, plateau_hi],
        "table": sorted(rows, key=lambda r: r["thr"]),
        "bootstrap": {
            "reps": args.boot,
            "subsample": "50% without replacement",
            "argmax_fraction_by_thr": argmax_frac,
        },
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    common.write_config(exp, {"experiment": "w2_threshold", "grid": GRID,
                              "bootstrap_reps": args.boot})
    common.write_env(exp)
    log(f"results written to {exp}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
