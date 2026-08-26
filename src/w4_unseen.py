"""W4 — unobserved profiles: Good-Turing, Chao1, rarefaction.

Good-Turing: P(the next record is a new profile) ~ f1 / N, where f1 is the
number of profiles seen exactly once (singletons). Sample coverage = 1 - f1/N.

Chao1: lower bound on the total number of profiles, unseen ones included:
    S_chao1 = S_obs + f1(f1-1) / (2(f2+1))            (bias-corrected form)
    (classic: S_obs + f1^2/(2 f2), when f2 > 0)
where f2 is the number of profiles seen exactly twice.

Rarefaction: the expected number of unique profiles E[S(m)] when drawing m of N
records - computed ANALYTICALLY (Hurlbert/Coleman), with no Monte Carlo noise:
    E[S(m)] = S_obs - sum_i P(profile i absent from a sample of m)
    P(absent) = C(N-n_i, m) / C(N, m)   (0 when m > N-n_i)
The question: does the space saturate at 700k records?

Computed for V14 (full) and V10 (the independent set from W3); margin thr
defaults to common.MARGIN.

Usage:
    python src/w4_unseen.py [--margin THR]
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
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, discretize, load_data, profile_ids

DROP_V10 = ["MCV", "MCH", "MCHC", "BASO"]


def good_turing(counts: np.ndarray, N: int) -> dict:
    f1 = int((counts == 1).sum())
    f2 = int((counts == 2).sum())
    p0 = f1 / N
    return {"f1": f1, "f2": f2, "P_new_profile": round(p0, 5),
            "sample_coverage": round(1 - p0, 5)}


def chao1(counts: np.ndarray, S_obs: int) -> dict:
    f1 = float((counts == 1).sum())
    f2 = float((counts == 2).sum())
    corrected = S_obs + f1 * (f1 - 1) / (2 * (f2 + 1))
    classic = S_obs + (f1 ** 2) / (2 * f2) if f2 > 0 else None
    out = {
        "S_obs": int(S_obs),
        "chao1_corrected": round(corrected, 1),
        "chao1_classic": round(classic, 1) if classic else None,
        "unseen_corrected": round(corrected - S_obs, 1),
    }
    # log-normal 95% CI (Chao 1987) on the classic estimator, when f2 > 0
    if f2 > 0:
        r = f1 / f2
        var = f2 * (0.5 * r ** 2 + r ** 3 + 0.25 * r ** 4)
        T = (f1 ** 2) / (2 * f2)           # estimated unseen richness
        if T > 0:
            K = np.exp(1.96 * np.sqrt(np.log(1 + var / T ** 2)))
            out["chao1_classic_CI95"] = [round(S_obs + T / K, 1), round(S_obs + T * K, 1)]
            out["chao1_classic_SE"] = round(float(np.sqrt(var)), 1)
    return out


def other_estimators(counts: np.ndarray, S_obs: int, N: int) -> dict:
    """Jackknife 1/2 and ACE - robustness check on the richness estimate."""
    f1 = float((counts == 1).sum())
    f2 = float((counts == 2).sum())
    jack1 = S_obs + f1 * (N - 1) / N
    jack2 = S_obs + (f1 * (2 * N - 3) / N - f2 * (N - 2) ** 2 / (N * (N - 1)))

    # ACE: split into rare (<=10) and abundant (>10)
    rare_mask = counts <= 10
    S_rare = float(rare_mask.sum())
    S_abund = float((~rare_mask).sum())
    N_rare = float(counts[rare_mask].sum())
    ace = None
    if N_rare > 0 and f1 < N_rare:
        C_ace = 1 - f1 / N_rare
        i = np.arange(1, 11)
        fi = np.array([(counts == k).sum() for k in i], dtype=float)
        num = float((i * (i - 1) * fi).sum())
        gamma2 = max(S_rare / C_ace * num / (N_rare * (N_rare - 1)) - 1, 0.0)
        ace = S_abund + S_rare / C_ace + f1 / C_ace * gamma2
    return {
        "jackknife1": round(jack1, 1),
        "jackknife2": round(jack2, 1),
        "ace": round(ace, 1) if ace is not None else None,
    }


def rarefaction(counts: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytical E[S(m)] over a grid of sample sizes m."""
    N = int(counts.sum())
    S = len(counts)
    lg_Nni = gammaln(N - counts + 1)          # log (N-n_i)! - computed once
    lg_N1 = gammaln(N + 1)
    es = np.empty(len(grid), dtype=float)
    for j, m in enumerate(grid):
        # log P(absent) = lg(N-n_i)! - lg(N-n_i-m)! - lg N! + lg(N-m)!
        rem = N - counts - m
        with np.errstate(invalid="ignore"):
            log_absent = lg_Nni - gammaln(rem + 1) - lg_N1 + gammaln(N - m + 1)
        absent = np.where(rem >= 0, np.exp(log_absent), 0.0)
        es[j] = S - absent.sum()
    return grid, es


def analyze_variant(df, shorts, margin, label):
    params = {s: PARAMS_14[s] for s in shorts}
    ordinal = discretize(df, margin=margin, params=params)
    counts = profile_ids(ordinal).value_counts().to_numpy()
    N, S = int(counts.sum()), len(counts)

    gt = good_turing(counts, N)
    ch = chao1(counts, S)
    other = other_estimators(counts, S, N)

    # rarefaction grid: denser at the low end, up to N
    grid = np.unique(np.concatenate([
        np.array([1000, 2000, 5000]),
        np.arange(10000, N, 10000),
        np.array([N]),
    ]))
    grid = grid[grid <= N]
    m_arr, es = rarefaction(counts, grid)

    # end slope ~ the rate at which new profiles appear (cross-check against Good-Turing)
    end_slope = float((es[-1] - es[-2]) / (m_arr[-1] - m_arr[-2]))

    return {
        "label": label, "n_params": len(shorts), "N": N, "S_obs": S,
        "good_turing": gt, "chao1": ch, "other_estimators": other,
        "rarefaction_end_slope_per_record": round(end_slope, 5),
        "rarefaction": {"m": m_arr.tolist(), "E_S": [round(float(x), 1) for x in es]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    args = ap.parse_args()

    common.set_seed()
    exp = common.experiment_dir("w4_unseen")
    df = load_data()

    variants = {
        "V14": list(PARAMS_14),
        "V10": [s for s in PARAMS_14 if s not in DROP_V10],
    }
    out = {k: analyze_variant(df, v, args.margin, k) for k, v in variants.items()}

    # rarefaction figure
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = {"V14": "#2b6cb0", "V10": "#dd6b20"}
    for k, r in out.items():
        m = np.array(r["rarefaction"]["m"]) / 1000
        ax.plot(m, r["rarefaction"]["E_S"], "-", lw=1.6, color=colors[k],
                label=f"{k} ({r['n_params']} analytes): S_obs={r['S_obs']}, "
                      f"Chao1~{r['chao1']['chao1_corrected']:.0f}")
        ax.axhline(r["chao1"]["chao1_corrected"], color=colors[k], ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of sampled records (thousands)")
    ax.set_ylabel("Expected number of unique profiles E[S(m)]")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(exp / "fig_rarefaction.png", dpi=200)
    plt.close(fig)

    summary = {"margin": args.margin, "variants": out}
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    common.write_config(exp, {"experiment": "w4_unseen", "margin": args.margin,
                              "variants": variants})
    common.write_env(exp)

    # short summary on stdout
    for k, r in out.items():
        print(f"{k}: N={r['N']} S_obs={r['S_obs']} "
              f"P(new)={r['good_turing']['P_new_profile']} "
              f"Chao1={r['chao1']['chao1_corrected']} "
              f"unseen={r['chao1']['unseen_corrected']} "
              f"end_slope={r['rarefaction_end_slope_per_record']}")


if __name__ == "__main__":
    main()
