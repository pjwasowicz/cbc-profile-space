"""Recomputes the manuscript's numbers from the deposited aggregate data alone.

This is the check a reviewer can run without any access to patient records. It
reads only `data/` and the macro file the manuscript inputs, derives
every quantity that is a function of the deposited data, and compares the result
against the published value character for character.

It deliberately shares no code with the analysis pipeline. The formulas are
written out here from their published definitions, so agreement means the
manuscript's numbers follow from the deposit, not that two copies of the same
implementation agree with each other.

Every macro in `numbers_en.tex` must appear in the registry below, either with a
way to recompute it or with a stated reason why the deposit cannot reach it. A
macro that appears in neither is a hard error, so the classification cannot go
stale as the manuscript changes.

Statuses:
    ok      recomputed from the deposit and identical to the published value
    bound   not point-identified from the deposit, but the published value lies
            inside the interval the deposit implies (the withheld tail is the
            only freedom left)
    n/a     out of reach of the deposit, with the reason printed
    FAIL    recomputed and different, or outside the implied interval

Usage:
    python src/verify_deposit.py
    python src/verify_deposit.py --bootstrap 200   # also re-run the W5 p-value
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPOSIT = REPO_ROOT / "data"
NUMBERS = REPO_ROOT / "paper" / "numbers_en.tex"

SEED = 42
NORM_STATE = 2
N_ANALYTES = 14
N_NON_NORM = 4                      # states 0,1,3,4 - anything that is not `norm`


# --------------------------------------------------------------- formatting --
# Mirrors paper/fill_numbers.py: the comparison is on the rendered
# string, so a difference in the last printed digit is a failure, not a rounding
# opinion.
def as_int(value) -> str:
    return f"{int(round(float(value))):,}"


def as_dec(value, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}"


def as_pct(fraction, decimals: int = 1) -> str:
    return f"{float(fraction) * 100.0:.{decimals}f}"


# ------------------------------------------------------------------ loading --
def read_spectrum(path: Path) -> np.ndarray:
    """Rebuilds the full count vector from an (r, f_r) spectrum."""
    with path.open() as fh:
        rows = [(int(row["r"]), int(row["f_r"])) for row in csv.DictReader(fh)]
    r = np.array([a for a, _ in rows])
    f = np.array([b for _, b in rows])
    return np.sort(np.repeat(r, f))[::-1]


def read_profiles(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The released head: analyte columns, the state matrix and the counts."""
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader]
    meta = {"count", "percent", "percent_cumulative"}
    analytes = [c for c in header if c not in meta]
    idx = [header.index(a) for a in analytes]
    states = np.array([[int(row[i]) for i in idx] for row in rows], dtype=np.int8)
    counts = np.array([int(row[header.index("count")]) for row in rows], dtype=np.int64)
    return analytes, states, counts


def read_macros(path: Path) -> dict[str, str]:
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
    out = {}
    for line in path.read_text().splitlines():
        hit = pattern.match(line.strip())
        if hit:
            out[hit.group(1)] = hit.group(2)
    return out


# ---------------------------------------------------------------- estimators --
def coverage_profiles(counts: np.ndarray, target: float) -> int:
    return int(np.searchsorted(np.cumsum(counts) / counts.sum(), target) + 1)


def zipf_ols(counts: np.ndarray) -> tuple[float, float]:
    x = np.log(np.arange(1, len(counts) + 1))
    y = np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(r2)


def clauset_fit(counts: np.ndarray, xmin_cap: float = 300.0,
                min_tail: int = 50) -> dict:
    """Continuous-approximation MLE with x_min chosen by minimum KS distance.

    Clauset, Shalizi & Newman (2009): for a candidate x_min,
    alpha = 1 + n / sum(ln(x_i / x_min)), and x_min minimises the KS distance
    between the empirical and the fitted tail CDF.
    """
    x = np.sort(counts.astype(float))
    best = None
    for xmin in np.unique(x):
        if xmin > xmin_cap:
            break
        tail = x[x >= xmin]
        n = len(tail)
        if n < min_tail:
            continue
        alpha = 1.0 + n / np.log(tail / xmin).sum()
        emp = np.arange(1, n + 1) / n
        theo = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
        ks = float(np.abs(emp - theo).max())
        if best is None or ks < best["ks"]:
            best = {"alpha": float(alpha), "xmin": float(xmin), "ks": ks, "ntail": n}
    return best


def diversity(counts: np.ndarray) -> dict:
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return {
        "richness": len(counts),
        "shannon": round(h, 4),
        "hill1": round(float(np.exp(h)), 1),
        "hill2": round(float(1.0 / (p ** 2).sum()), 1),
        "pielou": round(h / math.log(len(counts)), 4),
    }


def unseen(counts: np.ndarray) -> dict:
    """Good-Turing, Chao1 with its log-normal CI, jackknife 1/2 and ACE."""
    n_records = int(counts.sum())
    s_obs = len(counts)
    f1 = float((counts == 1).sum())
    f2 = float((counts == 2).sum())

    out = {
        "S_obs": s_obs,
        "f1": int(f1),
        "f2": int(f2),
        "p_new": round(f1 / n_records, 5),
        "coverage": round(1 - f1 / n_records, 5),
        "chao1": round(s_obs + f1 * (f1 - 1) / (2 * (f2 + 1)), 1),
        "jack1": round(s_obs + f1 * (n_records - 1) / n_records, 1),
        "jack2": round(s_obs + f1 * (2 * n_records - 3) / n_records
                       - f2 * (n_records - 2) ** 2 / (n_records * (n_records - 1)), 1),
    }

    # Chao (1987) log-normal interval on the classic estimator
    ratio = f1 / f2
    var = f2 * (0.5 * ratio ** 2 + ratio ** 3 + 0.25 * ratio ** 4)
    unseen_hat = f1 ** 2 / (2 * f2)
    k = math.exp(1.96 * math.sqrt(math.log(1 + var / unseen_hat ** 2)))
    out["chao_lo"] = round(s_obs + unseen_hat / k, 1)
    out["chao_hi"] = round(s_obs + unseen_hat * k, 1)

    # ACE, splitting at abundance 10
    rare = counts <= 10
    s_rare, s_abund = float(rare.sum()), float((~rare).sum())
    n_rare = float(counts[rare].sum())
    c_ace = 1 - f1 / n_rare
    i = np.arange(1, 11)
    fi = np.array([(counts == k_).sum() for k_ in i], dtype=float)
    gamma2 = max(s_rare / c_ace * (i * (i - 1) * fi).sum() / (n_rare * (n_rare - 1)) - 1, 0.0)
    out["ace"] = round(s_abund + s_rare / c_ace + f1 / c_ace * gamma2, 1)

    out["end_slope"] = rarefaction_end_slope(counts)
    return out


def rarefaction_end_slope(counts: np.ndarray) -> float:
    """Slope of the analytical rarefaction curve at the last grid step.

    E[S(m)] = S - sum_i P(profile i absent from a sample of m), Hurlbert/Coleman.
    Only the last two grid points are needed, and the grid is the one the
    manuscript's curve is drawn on.
    """
    n_records = int(counts.sum())
    grid = np.unique(np.concatenate([
        np.array([1000, 2000, 5000]),
        np.arange(10000, n_records, 10000),
        np.array([n_records]),
    ]))
    grid = grid[grid <= n_records][-2:]

    lg_rest = gammaln(n_records - counts + 1)
    lg_total = gammaln(n_records + 1)
    values = []
    for m in grid:
        remaining = n_records - counts - m
        with np.errstate(invalid="ignore"):
            log_absent = lg_rest - gammaln(remaining + 1) - lg_total + gammaln(n_records - m + 1)
        absent = np.where(remaining >= 0, np.exp(log_absent), 0.0)
        values.append(len(counts) - absent.sum())
    return float((values[1] - values[0]) / (grid[1] - grid[0]))


def clauset_pvalue(counts: np.ndarray, reps: int, rng: np.random.Generator) -> float:
    """Bootstrap goodness-of-fit: the share of synthetic sets with KS >= observed."""
    fit = clauset_fit(counts)
    x = counts.astype(float)
    below = x[x < fit["xmin"]]
    n_tail, n_total = fit["ntail"], len(x)
    worse = 0
    for _ in range(reps):
        n_from_tail = int(rng.binomial(n_total, n_tail / n_total))
        tail = fit["xmin"] * (1 - rng.random(n_from_tail)) ** (-1.0 / (fit["alpha"] - 1.0))
        rest = rng.choice(below, size=n_total - n_from_tail, replace=True) if len(below) else np.empty(0)
        synthetic = np.concatenate([tail, rest])
        synthetic_fit = clauset_fit(np.round(synthetic).astype(np.int64))
        if synthetic_fit and synthetic_fit["ks"] >= fit["ks"]:
            worse += 1
    return worse / reps


# ----------------------------------------------------------- Hamming (head) --
def hamming_head(states: np.ndarray, counts: np.ndarray, n_corpus: int,
                 min_count: int) -> dict:
    """Distance-from-all-norm shares computed on the released head only.

    Every share is expressed against the FULL corpus, so the head value is a
    lower bound. The freedom left to the withheld tail is not merely "the
    withheld mass could be anywhere": a shell at distance d holds at most
    C(14, d) * 4^d distinct profiles, the head already shows which of them are
    present, and every absent profile carries at most min_count - 1 records by
    definition of the threshold. That caps how much mass each shell can still
    gain, which is far tighter than the total withheld mass for the inner
    shells.
    """
    deviated = (states != NORM_STATE).sum(axis=1)
    mass = np.bincount(deviated, weights=counts, minlength=N_ANALYTES + 1) / n_corpus
    slack = 1.0 - counts.sum() / n_corpus

    seen = np.bincount(deviated, minlength=N_ANALYTES + 1).astype(float)
    possible = np.array([math.comb(N_ANALYTES, d) * N_NON_NORM ** d
                         for d in range(N_ANALYTES + 1)], dtype=float)
    capacity = (possible - seen) * (min_count - 1) / n_corpus

    return {"mass": mass, "slack": slack, "capacity": capacity,
            "seen": seen, "possible": possible,
            "head_mean_share": float((deviated * counts).sum() / n_corpus)}


def shell_interval(ham: dict, shells: set[int]) -> tuple[float, float]:
    """Two-sided bound on the record share of a set of Hamming shells.

    Upper: the shells can absorb at most their own spare capacity, and never
    more than the total withheld mass. Lower: whatever the OTHER shells cannot
    possibly hold has to land inside this set.
    """
    mass, slack, cap = ham["mass"], ham["slack"], ham["capacity"]
    inside = float(sum(cap[d] for d in shells))
    outside = float(sum(cap[d] for d in range(N_ANALYTES + 1) if d not in shells))
    base = float(sum(mass[d] for d in shells))
    return base + max(0.0, slack - outside), base + min(slack, inside)


def mean_interval(ham: dict) -> tuple[float, float]:
    """Bound on the mean distance, by placing the withheld mass as low, then as
    high, as the per-shell capacities allow."""
    slack, cap = ham["slack"], ham["capacity"]

    def fill(order):
        extra, remaining = 0.0, slack
        for d in order:
            take = min(remaining, float(cap[d]))
            extra += d * take
            remaining -= take
            if remaining <= 1e-12:
                break
        return extra

    base = ham["head_mean_share"]
    return (base + fill(range(N_ANALYTES + 1)),
            base + fill(range(N_ANALYTES, -1, -1)))


def bounded_median(mass: np.ndarray, slack: float) -> int | None:
    """The median distance, when the withheld mass cannot move it."""
    cum = np.cumsum(mass)
    for d in range(len(mass)):
        below = cum[d - 1] if d else 0.0
        if cum[d] > 0.5 and below + slack < 0.5:
            return d
    return None


# ------------------------------------------------------------ what is where --
P_NEEDED = ("needs the continuous position P inside each band; the deposit "
            "records only the discretised state")
RAW_NEEDED = "needs analyte values in physical units, which the deposit does not carry"
DROPPED_NEEDED = ("needs the discarded records; the deposit contains only complete ones")
STOCHASTIC = "bootstrap goodness-of-fit; re-run with --bootstrap N"

OUT_OF_SCOPE = {
    **{m: DROPPED_NEEDED for m in
       ("ResRecordsRaw", "ResDropped", "ResKeptPct", "ResDroppedPct",
        "AuditRaw", "AuditDropped", "AuditDroppedPct", "AuditEmpty",
        "AuditTopMissingName", "AuditTopMissingN")},
    **{m: P_NEEDED for m in
       ("ThrPlateauLo", "ThrPlateauHi", "ThrPlateauTolPct", "ThrBootReps",
        "ThrBootFirstMargin", "ThrBootFirstPct", "ThrBootSecondMargin",
        "ThrBootSecondPct", "ThrBootThirdMargin", "ThrBootThirdPct",
        "CthrZeroTen", "CthrZeroEleven", "CthrZeroTwelve", "CthrZeroThirteen",
        "CthrZeroFourteen", "CthrZeroFifteen", "CthrZeroSixteen", "CthrZeroTwenty")},
    **{f"Resid{a}{b}": RAW_NEEDED
       for a in ("Mcv", "Mch", "Mchc", "Wbc")
       for b in ("Median", "PNinetyNine", "WithinPct")},
    "PlPvalue": STOCHASTIC,
}

GROUPS = [
    ("W1  size and concentration", "Res"),
    ("W2  margin selection", ("Thr", "Cthr")),
    ("W3  algebraic redundancy", ("Resid", "Red")),
    ("W4  unobserved profiles", "Unseen"),
    ("W5  power-law fit", "Pl"),
    ("W6/W7  geometry and diversity", ("Geom", "Div")),
    ("W9  audit of discarded records", "Audit"),
    ("deposit", "Rel"),
]


def recompute(deposit: Path) -> tuple[dict[str, str], dict[str, tuple], list[str]]:
    """Returns exact values, bounded values and deposit-integrity notes."""
    v14 = read_spectrum(deposit / "frequency_spectrum.csv")
    v10 = read_spectrum(deposit / "frequency_spectrum_v10.csv")
    v9 = read_spectrum(deposit / "frequency_spectrum_v9.csv")
    analytes, states, head_counts = read_profiles(deposit / "profiles.csv")
    report = json.loads((deposit / "release_summary.json").read_text())

    n_records, n_profiles = int(v14.sum()), len(v14)
    exact: dict[str, str] = {}

    # ---- W1 ---------------------------------------------------------------- #
    exact["ResRecordsKept"] = as_int(n_records)
    exact["ResProfiles"] = as_int(n_profiles)
    for target, suffix in ((0.50, "Fifty"), (0.80, "Eighty"), (0.90, "Ninety"),
                           (0.95, "NinetyFive"), (0.99, "NinetyNine")):
        exact[f"ResCov{suffix}"] = as_int(coverage_profiles(v14, target))
    exact["ResTopOnePct"] = as_pct(v14[0] / n_records)
    f1_v14 = int((v14 == 1).sum())
    exact["ResSingletons"] = as_int(f1_v14)
    exact["ResSingletonPct"] = as_pct(f1_v14 / n_profiles)
    slope, r2 = zipf_ols(v14)
    exact["ResZipfSlope"] = as_dec(slope, 2)
    exact["ResZipfRsq"] = as_dec(r2, 2)

    # ---- W2: only the margin the deposit was built at is checkable --------- #
    exact["ThrArgmax"] = as_dec(report["margin"], 2)

    # ---- W5 ---------------------------------------------------------------- #
    fit = clauset_fit(v14)
    exact["PlAlpha"] = as_dec(fit["alpha"], 2)
    exact["PlXmin"] = as_int(fit["xmin"])
    exact["PlNTail"] = as_int(fit["ntail"])

    # ---- W3: variant tables ------------------------------------------------ #
    for suffix, counts in (("VFourteen", v14), ("VTen", v10), ("VNine", v9)):
        exact[f"Red{suffix}Profiles"] = as_int(len(counts))
        exact[f"Red{suffix}SingletonPct"] = as_pct((counts == 1).sum() / len(counts))
    for suffix, counts in (("VTen", v10), ("VNine", v9)):
        exact[f"RedDrop{suffix}Pct"] = as_pct((n_profiles - len(counts)) / n_profiles)

    # ---- W6: diversity, and W4: richness estimators ------------------------ #
    for suffix, counts in (("VFourteen", v14), ("VTen", v10)):
        div = diversity(counts)
        exact[f"DivShannon{suffix}"] = as_dec(div["shannon"], 2)
        exact[f"DivHillOne{suffix}"] = as_dec(div["hill1"], 1)
        exact[f"DivHillTwo{suffix}"] = as_dec(div["hill2"], 1)
        exact[f"DivPielou{suffix}"] = as_dec(div["pielou"], 2)
        exact[f"DivRichness{suffix}"] = as_int(div["richness"])

        est = unseen(counts)
        exact[f"Unseen{suffix}Sobs"] = as_int(est["S_obs"])
        exact[f"Unseen{suffix}FOne"] = as_int(est["f1"])
        exact[f"Unseen{suffix}FTwo"] = as_int(est["f2"])
        exact[f"Unseen{suffix}PNewPct"] = as_pct(est["p_new"], 2)
        exact[f"Unseen{suffix}CoveragePct"] = as_pct(est["coverage"])
        exact[f"Unseen{suffix}JackOne"] = as_int(est["jack1"])
        exact[f"Unseen{suffix}JackTwo"] = as_int(est["jack2"])
        exact[f"Unseen{suffix}Ace"] = as_int(est["ace"])
        exact[f"Unseen{suffix}Chao"] = as_int(est["chao1"])
        exact[f"Unseen{suffix}ChaoLo"] = as_int(est["chao_lo"])
        exact[f"Unseen{suffix}ChaoHi"] = as_int(est["chao_hi"])
        exact[f"Unseen{suffix}EndSlope"] = as_dec(est["end_slope"], 3)
        exact[f"Unseen{suffix}ObservedPct"] = as_pct(est["S_obs"] / est["chao1"], 0)

    # ---- deposit self-description ------------------------------------------ #
    exact["RelMinCount"] = as_int(report["min_count"])
    exact["RelProfilesTotal"] = as_int(report["n_profiles"])
    exact["RelProfilesReleased"] = as_int(report["profiles_released"])
    exact["RelProfilesWithheld"] = as_int(report["profiles_suppressed"])
    exact["RelProfilesWithheldPct"] = as_dec(report["profiles_suppressed_pct"], 1)
    exact["RelRecordsWithheld"] = as_int(report["records_suppressed"])
    exact["RelRecordsWithheldPct"] = as_dec(report["records_suppressed_pct"], 2)
    exact["RelRecordsCoveredPct"] = as_pct(1 - report["records_suppressed_pct"] / 100, 2)

    # ---- W6/W7: Hamming geometry, only partly reachable -------------------- #
    ham = hamming_head(states, head_counts, n_records, report["min_count"])
    bounded: dict[str, tuple] = {}

    def place(name: str, shells: set[int], decimals: int = 1) -> None:
        lo, hi = shell_interval(ham, shells)
        if round(lo * 100, decimals) == round(hi * 100, decimals):
            exact[name] = as_dec(lo * 100, decimals)          # the tail cannot move it
        else:
            bounded[name] = (lo * 100, hi * 100, decimals)

    place("GeomFracZeroPct", {0})
    place("GeomShellZeroPct", {0})
    place("GeomFracLeOnePct", {0, 1})
    place("GeomShellOnePct", {1})
    place("GeomFracLeTwoPct", {0, 1, 2})
    place("GeomShellTwoPct", {2})
    place("GeomShellThreePct", {3})
    place("GeomShellFourPlusPct", set(range(4, N_ANALYTES + 1)))

    lo_mean, hi_mean = mean_interval(ham)
    if round(lo_mean, 2) == round(hi_mean, 2):
        exact["GeomHammingMean"] = as_dec(lo_mean, 2)
    else:
        bounded["GeomHammingMean"] = (lo_mean, hi_mean, 2)

    median = bounded_median(ham["mass"], ham["slack"])
    if median is not None:
        exact["GeomHammingMedian"] = as_int(median)

    # ---- integrity of the deposit against its own report ------------------- #
    notes = []
    checks = [
        ("profiles in frequency_spectrum vs release_summary", n_profiles, report["n_profiles"]),
        ("records in frequency_spectrum vs release_summary", n_records, report["n_records"]),
        ("rows in profiles.csv vs release_summary", len(head_counts), report["profiles_released"]),
        ("withheld records", n_records - int(head_counts.sum()), report["records_suppressed"]),
        ("analyte columns in profiles.csv", len(analytes), N_ANALYTES),
        ("head respects the threshold", int(head_counts.min()), report["min_count"]),
    ]
    for label, got, want in checks:
        if label == "head respects the threshold":
            notes.append((label, got >= want, f"min count in head = {got}, threshold {want}"))
        else:
            notes.append((label, got == want, f"{got:,} vs {want:,}"))
    inner = ", ".join(f"{d}: {int(ham['seen'][d])}/{int(ham['possible'][d])}"
                      for d in range(4))
    notes.append(("Hamming shells present in the head (of all possible)", True, inner))
    return exact, bounded, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposit", type=Path, default=DEPOSIT)
    ap.add_argument("--numbers", type=Path, default=NUMBERS)
    ap.add_argument("--bootstrap", type=int, metavar="N", default=0,
                    help="Re-run the W5 goodness-of-fit with N synthetic sets (slow).")
    ap.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = ap.parse_args()

    for path in (args.deposit, args.numbers):
        if not path.exists():
            sys.exit(f"missing {path}")

    published = read_macros(args.numbers)
    exact, bounded, notes = recompute(args.deposit)

    if args.bootstrap:
        counts = read_spectrum(args.deposit / "frequency_spectrum.csv")
        p = clauset_pvalue(counts, args.bootstrap, np.random.default_rng(SEED))
        bounded["PlPvalue"] = (min(p, 0.10), 1.0, 3)     # the claim is "not rejected"
        OUT_OF_SCOPE.pop("PlPvalue", None)

    rows, tally = [], {"ok": 0, "bound": 0, "n/a": 0, "FAIL": 0}
    for name, value in sorted(published.items()):
        if name in exact:
            got = exact[name]
            status = "ok" if got == value else "FAIL"
            rows.append((name, status, value, got, ""))
        elif name in bounded:
            lo, hi, decimals = bounded[name]
            tol = 0.5 * 10 ** -decimals
            inside = lo - tol <= float(value) <= hi + tol
            rows.append((name, "bound" if inside else "FAIL", value,
                         f"[{lo:.{decimals}f}, {hi:.{decimals}f}]",
                         "withheld tail is the only freedom"))
        elif name in OUT_OF_SCOPE:
            rows.append((name, "n/a", value, "-", OUT_OF_SCOPE[name]))
        else:
            rows.append((name, "FAIL", value, "-", "macro not classified in this script"))
        tally[rows[-1][1]] += 1

    print(f"deposit  {args.deposit}")
    print(f"macros   {args.numbers}\n")
    print("Deposit integrity")
    for label, good, detail in notes:
        print(f"  [{'ok' if good else 'FAIL'}] {label}: {detail}")
        if not good:
            tally["FAIL"] += 1

    if not args.quiet:
        for title, prefixes in GROUPS:
            prefixes = (prefixes,) if isinstance(prefixes, str) else prefixes
            group = [r for r in rows if r[0].startswith(prefixes)
                     and not (title.startswith("W1") and r[0].startswith("Resid"))]
            if not group:
                continue
            print(f"\n{title}")
            for name, status, want, got, note in group:
                line = f"  [{status:>5}] {name:<28}{want:>12}"
                if status == "ok":
                    print(line)
                elif status == "bound":
                    print(f"{line}   in {got}")
                elif status == "n/a":
                    print(f"{line}   {note}")
                else:
                    print(f"{line}   got {got}   {note}")

    total = sum(tally.values())
    print(f"\n{total} macros: {tally['ok']} recomputed and identical, "
          f"{tally['bound']} consistent with the deposit's bounds, "
          f"{tally['n/a']} out of reach of the deposit, {tally['FAIL']} failing")
    if tally["FAIL"]:
        print("VERIFICATION FAILED")
        return 1
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
