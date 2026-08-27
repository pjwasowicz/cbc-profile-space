"""W7 — visualisation of profile-space geometry (MDS, t-SNE, Hamming graph, icicle); illustrates W6.

Unlike the figures in w1_structure.py / w6_geometry.py, which show marginal
distributions (coverage, rank-frequency, Hamming, f_r spectrum), this script
shows the STRUCTURE of the 14-dimensional profile space itself. Four figures:

1. fig_graph_hamming.png - Hamming adjacency graph: nodes = the most frequent
   profiles, an edge = a difference in exactly one analyte (Hamming = 1). Node
   size ~ log frequency, colour = number of deviations from all-norm. Layout by
   Fruchterman-Reingold (own deterministic implementation). Reveals the all-norm
   hub and the thinning shells around it.
2. fig_projection_mds.png - 2D projection by classical multidimensional scaling
   (PCoA on the Hamming distance matrix). One point = one profile, size ~ log
   frequency, colour = number of deviations from all-norm. The "map" of the space.
3. fig_profile_heatmap.png - matrix of the most frequent profiles x 14 analytes;
   cell colour = ordinal state (0..4, diverging low-norm-high), with a log
   frequency bar alongside. Shows WHICH patterns dominate, not merely that the
   space is concentrated.
4. fig_icicle_shells.png - icicle plot (two-level partition) of record coverage by
   Hamming shell from all-norm: level 1 = shells 0/1/2/3/>=4, level 2 = individual
   profiles within a shell. Ties concentration (W1) to distance from normal (W6).

Margin thr defaults to common.MARGIN (the argmax of C(thr) from W2). Everything
is deterministic (SEED=42).

Usage:
    python src/w7_spaceviz.py [--margin THR] [--n-graph 250]
        [--n-proj 1500] [--n-heat 40]
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, discretize, load_data

# V10 panel - without the 4 algebraically dependent analytes (as in W3/W4, w6_geometry).
DROP_V10 = ["MCV", "MCH", "MCHC", "BASO"]

NORM_STATE = 2
# Five-state diverging scale: clearly_low .. norm .. clearly_high (RdBu-like).
STATE_COLORS = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), STATE_CMAP.N)
# Hamming shell colour (number of deviations from all-norm): 0..4+.
SHELL_COLORS = ["#2b6cb0", "#38a169", "#dd6b20", "#c53030", "#6b46c1"]


def unique_profiles(ordinal) -> tuple[np.ndarray, np.ndarray]:
    """Returns (profiles as 14-vectors, counts) sorted by decreasing count."""
    arr = ordinal.to_numpy(dtype=np.int8)
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    order = np.argsort(-counts)
    return uniq[order], counts[order]


def deviations(profiles: np.ndarray) -> np.ndarray:
    """Number of analytes outside the reference range (Hamming from all-norm) per profile."""
    return (profiles != NORM_STATE).sum(axis=1)


# --------------------------------------------------------------------------- #
# 1. Hamming adjacency graph + Fruchterman-Reingold layout
# --------------------------------------------------------------------------- #
def spring_layout(adj: np.ndarray, seed: int, iters: int = 300) -> np.ndarray:
    """Deterministic FR layout. adj: symmetric {0,1} adjacency matrix."""
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 2))
    k = 1.0 / np.sqrt(n)            # optimal distance between nodes
    t = 0.1                        # temperature (max displacement per step)
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]          # (n, n, 2)
        dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9
        # repulsion between all pairs: k^2/d along +delta
        rep = ((k * k / dist) / dist)[..., None] * delta
        disp = rep.sum(axis=1)
        # attraction along edges: d^2/k along -delta
        att_mag = (dist * dist / k) * adj
        disp -= ((att_mag / dist)[..., None] * delta).sum(axis=1)
        length = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        pos += (disp / length[..., None]) * np.minimum(length, t)[..., None]
        t *= 0.99
    # centre the layout
    pos -= pos.mean(axis=0)
    return pos


def fig_graph_hamming(profiles: np.ndarray, counts: np.ndarray, seed: int, out: Path) -> dict:
    ham = squareform(pdist(profiles, metric="hamming") * profiles.shape[1]).round().astype(int)
    adj_full = (ham == 1).astype(float)
    # keep only the largest connected component - isolated profiles and small
    # components drift away under the FR layout and squash the core without
    # carrying any structure.
    n_comp, labels = connected_components(adj_full, directed=False)
    giant = np.bincount(labels).argmax()
    keep = labels == giant
    n_isolated = int((~keep).sum())
    profiles, counts = profiles[keep], counts[keep]
    adj = adj_full[np.ix_(keep, keep)]
    pos = spring_layout(adj, seed=seed)

    dev = deviations(profiles)
    size = 40 + 260 * (np.log1p(counts) / np.log1p(counts.max()))
    color = [SHELL_COLORS[min(d, 4)] for d in dev]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # edges
    ei, ej = np.where(np.triu(adj) > 0)
    for i, j in zip(ei, ej):
        ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="#cbd5e0", lw=0.4, alpha=0.6, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=size, c=color, edgecolors="white",
               linewidths=0.4, zorder=2)
    # label the all-norm hub when present
    allnorm = np.where(dev == 0)[0]
    if len(allnorm):
        i = allnorm[0]
        ax.annotate("all-norm", (pos[i, 0], pos[i, 1]), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, weight="bold", color="#1a202c")

    handles = [plt.Line2D([0], [0], marker="o", ls="", mec="white", mfc=SHELL_COLORS[d],
                          ms=8, label=f"{d} deviations" if d < 4 else "≥4 deviations")
               for d in range(5)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5,
              frameon=False, title="from all-norm", title_fontsize=8)
    ax.set_title(f"Hamming adjacency graph - {len(profiles)} connected profiles\n"
                 f"(edge = one-analyte difference; {int(adj.sum() / 2)} edges; "
                 f"{n_isolated} outside the largest component)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=common.FIG_DPI)
    plt.close(fig)
    return {"n_nodes": int(len(profiles)), "n_edges": int(adj.sum() / 2),
            "n_isolated_dropped": n_isolated}


# --------------------------------------------------------------------------- #
# 2. 2D projection - classical MDS (PCoA) on Hamming distances
# --------------------------------------------------------------------------- #
def pcoa(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classical scaling (Torgerson): returns 2D coordinates and explained variance."""
    n = dist.shape[0]
    d2 = dist ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j                       # double-centred
    vals, vecs = np.linalg.eigh(b)
    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]
    pos = vecs[:, :2] * np.sqrt(np.clip(vals[:2], 0, None))
    explained = np.clip(vals, 0, None)
    explained = explained[:2] / explained.sum()
    return pos, explained


def fig_projection_mds(profiles: np.ndarray, counts: np.ndarray, out: Path) -> dict:
    dist = squareform(pdist(profiles, metric="hamming") * profiles.shape[1])
    pos, expl = pcoa(dist)
    dev = deviations(profiles)
    size = 8 + 140 * (np.log1p(counts) / np.log1p(counts.max()))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=size, c=dev, cmap=ListedColormap(SHELL_COLORS),
                    vmin=-0.5, vmax=4.5, alpha=0.8, edgecolors="none")
    allnorm = np.where(dev == 0)[0]
    if len(allnorm):
        i = allnorm[0]
        ax.scatter(pos[i, 0], pos[i, 1], s=size[i] + 60, facecolors="none",
                   edgecolors="#1a202c", linewidths=1.2, zorder=3)
        ax.annotate("all-norm", (pos[i, 0], pos[i, 1]), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, weight="bold")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(5), fraction=0.046, pad=0.04)
    cbar.set_label("deviations from all-norm", fontsize=8)
    cbar.ax.set_yticklabels(["0", "1", "2", "3", "≥4"])
    ax.set_title(f"MDS projection (PCoA on Hamming distances) - {len(profiles)} profiles\n"
                 f"axes: {expl[0]*100:.0f}% and {expl[1]*100:.0f}% of variance; size ~ log frequency",
                 fontsize=9)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=common.FIG_DPI)
    plt.close(fig)
    return {"n_profiles": int(len(profiles)),
            "explained_variance": [float(x) for x in expl]}


def fig_projection_tsne(profiles: np.ndarray, counts: np.ndarray, seed: int, out: Path) -> dict:
    """t-SNE projection on Hamming distances - brings out local clusters (non-linearly)."""
    dist = squareform(pdist(profiles, metric="hamming") * profiles.shape[1])
    n = len(profiles)
    perplexity = float(min(30, max(5, (n - 1) // 3)))
    ts = TSNE(n_components=2, metric="precomputed", init="random",
              perplexity=perplexity, random_state=seed)
    pos = ts.fit_transform(dist)
    dev = deviations(profiles)
    size = 8 + 140 * (np.log1p(counts) / np.log1p(counts.max()))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=size, c=dev, cmap=ListedColormap(SHELL_COLORS),
                    vmin=-0.5, vmax=4.5, alpha=0.8, edgecolors="none")
    allnorm = np.where(dev == 0)[0]
    if len(allnorm):
        i = allnorm[0]
        ax.scatter(pos[i, 0], pos[i, 1], s=size[i] + 60, facecolors="none",
                   edgecolors="#1a202c", linewidths=1.2, zorder=3)
        ax.annotate("all-norm", (pos[i, 0], pos[i, 1]), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, weight="bold")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(5), fraction=0.046, pad=0.04)
    cbar.set_label("deviations from all-norm", fontsize=8)
    cbar.ax.set_yticklabels(["0", "1", "2", "3", "≥4"])
    ax.set_title(f"t-SNE projection (Hamming distance) - {n} profiles\n"
                 f"perplexity={perplexity:.0f}; size ~ log frequency", fontsize=9)
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=common.FIG_DPI)
    plt.close(fig)
    return {"n_profiles": int(n), "perplexity": perplexity}


# --------------------------------------------------------------------------- #
# 3. Profile x analyte heatmap
# --------------------------------------------------------------------------- #
def fig_profile_heatmap(profiles: np.ndarray, counts: np.ndarray, n_records: int,
                        analytes: list[str], out: Path) -> None:
    n = len(profiles)
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(7.2, max(4.0, 0.22 * n + 1.2)),
        gridspec_kw={"width_ratios": [len(analytes), 3], "wspace": 0.05},
        sharey=True)

    ax.pcolormesh(profiles[::-1], cmap=STATE_CMAP, norm=STATE_NORM,
                  edgecolors="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(analytes)) + 0.5)
    ax.set_xticklabels(analytes, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([f"#{r}" for r in range(n, 0, -1)], fontsize=6)
    ax.set_title(f"{n} most frequent profiles x {len(analytes)} analytes", fontsize=9)

    pct = counts / n_records * 100
    axb.barh(np.arange(n) + 0.5, pct[::-1], height=0.8, color="#2b6cb0")
    axb.set_xscale("log")
    axb.set_xlabel("% of records\n(log scale)", fontsize=7)
    axb.tick_params(labelsize=6)
    axb.grid(axis="x", alpha=0.25, lw=0.5)

    # state legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=STATE_COLORS[s]) for s in range(5)]
    labels = ["clearly_low", "slightly_low", "norm", "slightly_high", "clearly_high"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=common.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Icicle of Hamming shells (two-level partition of coverage)
# --------------------------------------------------------------------------- #
def fig_icicle_shells(profiles: np.ndarray, counts: np.ndarray, n_records: int,
                      out: Path) -> dict:
    dev = deviations(profiles)
    shells = {}
    for d in range(5):
        mask = dev == d if d < 4 else dev >= 4
        shells[d] = counts[mask]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x0 = 0.0
    shell_share = {}
    for d in range(5):
        c = np.sort(shells[d])[::-1]
        w = c.sum() / n_records
        shell_share[d] = float(w)
        # level 1 - the shell
        ax.add_patch(plt.Rectangle((x0, 0.55), w, 0.4, facecolor=SHELL_COLORS[d],
                                   edgecolor="white", linewidth=1.0))
        if w > 0.02:
            ax.text(x0 + w / 2, 0.75, f"{d if d < 4 else '≥4'}\n{w*100:.1f}%",
                    ha="center", va="center", fontsize=8, color="white", weight="bold")
        # level 2 - individual profiles within the shell
        xp = x0
        for cnt in c:
            wp = cnt / n_records
            ax.add_patch(plt.Rectangle((xp, 0.1), wp, 0.4, facecolor=SHELL_COLORS[d],
                                       edgecolor="white", linewidth=0.2, alpha=0.75))
            xp += wp
        x0 += w

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_yticks([0.3, 0.75])
    ax.set_yticklabels(["profiles", "shell"], fontsize=8)
    ax.set_xlabel("Share of records (widths sum to 1)")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)])
    ax.set_title("Coverage icicle by Hamming shell from all-norm\n"
                 "(top band: shells 0..≥4; bottom: individual profiles)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.savefig(out, dpi=common.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return {"shell_record_share": shell_share}


def run_variant(ordinal, analytes: list[str], suffix: str, args, exp: Path) -> dict:
    """Generates the full figure set for one analyte panel. suffix='' for V14, '_v10' for V10."""
    n_records = len(ordinal)
    profiles, counts = unique_profiles(ordinal[analytes])

    g = fig_graph_hamming(profiles[:args.n_graph], counts[:args.n_graph],
                          seed=common.SEED, out=exp / f"fig_graph_hamming{suffix}.png")
    p = fig_projection_mds(profiles[:args.n_proj], counts[:args.n_proj],
                           out=exp / f"fig_projection_mds{suffix}.png")
    t = fig_projection_tsne(profiles[:args.n_proj], counts[:args.n_proj],
                            seed=common.SEED, out=exp / f"fig_projection_tsne{suffix}.png")
    fig_profile_heatmap(profiles[:args.n_heat], counts[:args.n_heat], n_records,
                        analytes, out=exp / f"fig_profile_heatmap{suffix}.png")
    ic = fig_icicle_shells(profiles, counts, n_records, out=exp / f"fig_icicle_shells{suffix}.png")
    return {"n_records": int(n_records), "n_profiles": int(len(profiles)),
            "n_analytes": len(analytes), "graph": g, "projection_mds": p,
            "projection_tsne": t, "icicle": ic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    ap.add_argument("--n-graph", type=int, default=250, help="number of profiles in the Hamming graph")
    ap.add_argument("--n-proj", type=int, default=1500, help="number of profiles in the MDS/t-SNE projections")
    ap.add_argument("--n-heat", type=int, default=40, help="number of profiles in the heatmap")
    args = ap.parse_args()

    common.set_seed()
    exp = common.experiment_dir("w7_spaceviz")

    df = load_data()
    ordinal = discretize(df, margin=args.margin)
    analytes14 = list(PARAMS_14)
    analytes10 = [a for a in analytes14 if a not in DROP_V10]

    summary = {
        "margin": args.margin,
        "V14": run_variant(ordinal, analytes14, "", args, exp),
        "V10": run_variant(ordinal, analytes10, "_v10", args, exp),
    }
    (exp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    common.write_config(exp, {"experiment": "w7_spaceviz", "margin": args.margin,
                              "n_graph": args.n_graph, "n_proj": args.n_proj,
                              "n_heat": args.n_heat, "params_V14": analytes14,
                              "params_V10": analytes10, "drop_V10": DROP_V10})
    common.write_env(exp)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
