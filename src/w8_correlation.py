"""W8 — correlation heatmap across the 14 CBC analytes; illustrative figure for W3.

Spearman (rank) correlation on raw results, robust to outliers. Analytes are
ordered in blocks: red cell series (RBC, HCT, HGB plus the derived indices MCV,
MCH, MCHC, RDW-SD), platelets (PLT) and white cells (WBC plus the five
differential fractions), so that the algebraic redundancy blocks from W3 stand
out as bright squares.

Diverging palette (RdBu, CVD-safe) centred on zero: red = positive, blue =
negative, white = no correlation. vmin/vmax = +/-1.

Usage:
    python src/w8_correlation.py [--method spearman|pearson]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, load_data

# thematic ordering (redundancy blocks visible along the diagonal)
ORDER = ["RBC", "HCT", "HGB", "MCV", "MCH", "MCHC", "RDW-SD",
         "PLT", "WBC", "NEUT", "LYMPH", "MONO", "EOS", "BASO"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    args = ap.parse_args()

    common.set_seed()
    exp = common.experiment_dir("w8_correlation")
    df = load_data()

    data = df[[f"RESULT_{PARAMS_14[s]}" for s in ORDER]].copy()
    data.columns = ORDER
    corr = data.corr(method=args.method)

    corr.to_csv(exp / f"corr_{args.method}.csv")

    n = len(ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ORDER, fontsize=8)

    # value annotations; text colour follows background saturation
    M = corr.to_numpy()
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if abs(v) > 0.55 else "#1a202c")

    # thin rules separating the thematic blocks
    for b in (7, 8):
        ax.axhline(b - 0.5, color="#2d3748", lw=0.8)
        ax.axvline(b - 0.5, color="#2d3748", lw=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{args.method.capitalize()} correlation", fontsize=9)
    ax.set_title(f"{args.method.capitalize()} correlation between CBC analytes (N={len(data)})",
                 fontsize=10)
    fig.tight_layout()
    out = exp / f"fig_heatmap_{args.method}.png"
    fig.savefig(out, dpi=common.FIG_DPI)
    plt.close(fig)

    common.write_config(exp, {"experiment": "w8_correlation", "method": args.method,
                              "order": ORDER})
    common.write_env(exp)
    print(f"saved {out}")
    print("strongest pairs (|r| >= 0.7, diagonal excluded):")
    seen = set()
    for i in range(n):
        for j in range(n):
            if i < j and abs(M[i, j]) >= 0.7:
                print(f"  {ORDER[i]:7s}~{ORDER[j]:7s} {M[i, j]:+.2f}")


if __name__ == "__main__":
    main()
