"""W6 — space geometry and effective diversity (Hamming, Hill numbers); extends W1.

1. Hamming distance from the all-norm profile - how many analytes deviate from
   the reference range in a typical patient (record-weighted distribution).
   This is what "small and concentrated" looks like geometrically.
2. Entropy and Hill numbers q=0,1,2 - single-number measures of concentration
   and diversity, robust to how rare profiles are weighted.
3. Frequency spectrum f_r (frequency of frequencies) - how many profiles occurred
   exactly r times; the basis of Good-Turing/Chao1 (W4) and material for the
   Data Availability statement (Zenodo).

Margin thr defaults to common.MARGIN. Entropy/Hill/f_r are computed for V14 and
V10; Hamming for V14 (the full panel).

Usage:
    python src/w6_geometry.py [--margin THR]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, discretize, load_data, profile_ids

DROP_V10 = ["MCV", "MCH", "MCHC", "BASO"]
NORM_STATE = 2
BLUE, ORANGE = "#2b6cb0", "#dd6b20"


def diversity(counts: np.ndarray) -> dict:
    """Shannon entropy, Hill numbers q=0,1,2 and Pielou evenness."""
    N = counts.sum()
    p = counts / N
    H_nats = float(-(p * np.log(p)).sum())
    S = len(counts)
    return {
        "S_hill_q0": int(S),
        "shannon_H_nats": round(H_nats, 4),
        "shannon_H_bits": round(H_nats / np.log(2), 4),
        "hill_q1_expH": round(float(np.exp(H_nats)), 1),
        "hill_q2_invSimpson": round(float(1.0 / (p ** 2).sum()), 1),
        "pielou_evenness": round(H_nats / np.log(S), 4),
    }


def freq_spectrum(counts: np.ndarray) -> dict:
    r, fr = np.unique(counts, return_counts=True)
    return {int(rr): int(ff) for rr, ff in zip(r, fr)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    args = ap.parse_args()

    common.set_seed()
    exp = common.experiment_dir("w6_geometry")
    df = load_data()

    variants = {"V14": list(PARAMS_14), "V10": [s for s in PARAMS_14 if s not in DROP_V10]}
    ordinals, counts_by = {}, {}
    for k, shorts in variants.items():
        params = {s: PARAMS_14[s] for s in shorts}
        ordinals[k] = discretize(df, margin=args.margin, params=params)
        counts_by[k] = profile_ids(ordinals[k]).value_counts().to_numpy()

    # --- diversity ---
    diversity_out = {k: diversity(counts_by[k]) for k in variants}

    # --- f_r spectrum ---
    spectra = {k: freq_spectrum(counts_by[k]) for k in variants}

    # --- Hamming distance from all-norm (V14) ---
    ord14 = ordinals["V14"]
    ndev = (ord14.to_numpy() != NORM_STATE).sum(axis=1)          # deviated analytes per record
    N14 = len(ndev)
    dev_bins = np.bincount(ndev, minlength=15)                   # 0..14
    dev_frac = dev_bins / N14
    # severity-weighted variant: slight=1, clear=2
    sev_map = np.array([2, 1, 0, 1, 2])                          # states 0..4 -> severity
    sev_sum = sev_map[ord14.to_numpy()].sum(axis=1)
    hamming_out = {
        "n_records": int(N14),
        "mean_deviated_analytes": round(float(ndev.mean()), 3),
        "median_deviated_analytes": int(np.median(ndev)),
        "frac_0_deviations": round(float(dev_frac[0]), 4),
        "frac_le1": round(float(dev_frac[:2].sum()), 4),
        "frac_le2": round(float(dev_frac[:3].sum()), 4),
        "frac_le3": round(float(dev_frac[:4].sum()), 4),
        "distribution_by_n_deviated": {int(i): round(float(dev_frac[i]), 5)
                                       for i in range(15)},
        "mean_severity_sum": round(float(sev_sum.mean()), 3),
    }

    # ---------- FIGURES ----------
    # Hamming: single bar series with direct % labels
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = np.arange(15)
    ax.bar(xs, dev_frac * 100, color=BLUE, width=0.8)
    for i in range(6):                                           # label bars 0..5
        if dev_frac[i] * 100 >= 0.5:
            ax.text(i, dev_frac[i] * 100 + 0.6, f"{dev_frac[i]*100:.1f}",
                    ha="center", fontsize=7, color="#1a202c")
    ax.set_xlabel("Number of analytes outside the reference range (distance from all-norm)")
    ax.set_ylabel("Share of records (%)")
    ax.set_xticks(xs)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(exp / "fig_hamming.png", dpi=200)
    plt.close(fig)

    # f_r spectrum: log-log, two series
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k, col in (("V14", BLUE), ("V10", ORANGE)):
        r = np.array(sorted(spectra[k]))
        fr = np.array([spectra[k][int(x)] for x in r])
        ax.loglog(r, fr, "o", ms=3, color=col, alpha=0.7, label=k)
    ax.set_xlabel("Profile count r (times observed)")
    ax.set_ylabel("Number of profiles with count r  (f_r)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(exp / "fig_freq_spectrum.png", dpi=200)
    plt.close(fig)

    summary = {
        "margin": args.margin,
        "diversity": diversity_out,
        "hamming_from_allnorm_V14": hamming_out,
        "freq_spectrum_head": {k: {r: spectra[k][r] for r in sorted(spectra[k])[:10]}
                               for k in variants},
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # full f_r to CSV (Zenodo / Data Availability)
    import pandas as pd
    for k in variants:
        s = spectra[k]
        pd.DataFrame({"r": sorted(s), "f_r": [s[r] for r in sorted(s)]}).to_csv(
            exp / f"freq_spectrum_{k}.csv", index=False)

    common.write_config(exp, {"experiment": "w6_geometry", "margin": args.margin,
                              "variants": variants})
    common.write_env(exp)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
