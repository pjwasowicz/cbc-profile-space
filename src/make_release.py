"""Builds the publishable (Zenodo / Data Availability) dataset from the CBC data.

Three files, chosen so that every number in the manuscript stays reproducible
while individual rare patterns are never exposed:

    profiles.csv            one row per profile with count >= --min-count,
                            one column per analyte plus count and coverage
    frequency_spectrum.csv  r, f_r - how many profiles occurred exactly r times;
                            released in full, it carries no profile identities
    frequency_spectrum_v10.csv,
    frequency_spectrum_v9.csv
                            the same spectrum for the algebraically reduced
                            panels of W3, so the variant statistics are
                            reproducible from the deposit and not only recorded
    README.md               file description, state legend, suppression report
    release_summary.json    the same report as data, read by the manuscript's
                            fill_numbers.py so the Data Availability statement
                            quotes the actual suppression

The split matters. Coverage, top-1 share, singleton fraction, the Zipf slope,
the Clauset fit, Shannon/Hill numbers, Good-Turing, Chao1, jackknife, ACE and
the rarefaction curve are all functions of the count vector alone, so the
frequency spectrum reproduces them exactly. Profile identities are needed only
for the Hamming geometry and the V10/V9 variants, and there the head of the
distribution is what the manuscript actually shows.

Suppression is reported, never silent: the number of withheld profiles and the
share of records they represent go into README.md and into the printed summary.

Usage:
    python src/make_release.py                      # threshold 5, 5 states
    python src/make_release.py --min-count 1        # no suppression
    python src/make_release.py --states 3           # low / norm / high
    python src/make_release.py --out-dir data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from discretize import PARAMS_14, STATE_NAMES, discretize, load_data

DEFAULT_OUT = common.REPO_ROOT / "data"
# Withholding threshold. Profiles rarer than this are dropped from profiles.csv;
# their counts survive in frequency_spectrum.csv either way. Five is the
# conventional threshold in statistical disclosure control; lowering it to 3
# would add only ~1.6 percentage points of released record coverage, because the
# withheld mass is dominated by singletons that no threshold above 1 can reach.
DEFAULT_MIN_COUNT = 5

# Reduced panels of W3. V10 drops the three derived red cell indices and BASO
# (algebraic redundancy); V9 additionally drops the leukocyte aggregate. Each is
# re-discretised from the raw frame rather than projected from the 14-analyte
# table, because completeness is judged over the selected analytes only - the
# reduced panels therefore retain a few records that V14 discards.
DROP_V10 = ["MCV", "MCH", "MCHC", "BASO"]
DROP_V9 = DROP_V10 + ["WBC"]

# 5 -> 3 states: slightly_* is merged into the neighbouring clearly_* band.
COARSE_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
COARSE_NAMES = ["low", "norm", "high"]


def profile_table(ordinal: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct profile, sorted by decreasing count."""
    analytes = list(ordinal.columns)
    table = (ordinal.groupby(analytes, sort=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False, kind="stable")
                    .reset_index(drop=True))
    n_records = len(ordinal)
    table["percent"] = table["count"] / n_records * 100
    table["percent_cumulative"] = table["percent"].cumsum()
    return table


def frequency_spectrum(counts: np.ndarray) -> pd.DataFrame:
    """f_r: how many profiles were observed exactly r times."""
    r, f_r = np.unique(counts, return_counts=True)
    return pd.DataFrame({"r": r, "f_r": f_r})


def write_readme(path: Path, report: dict, analytes: list[str], state_names: list[str]) -> None:
    legend = "\n".join(f"| {i} | `{name}` |" for i, name in enumerate(state_names))
    suppressed_note = (
        f"Profiles observed fewer than {report['min_count']} times are withheld from "
        f"`profiles.csv` as a disclosure-control measure: {report['profiles_suppressed']:,} "
        f"profiles ({report['profiles_suppressed_pct']:.1f}% of all profiles), together "
        f"accounting for {report['records_suppressed']:,} records "
        f"({report['records_suppressed_pct']:.2f}% of the corpus). Their counts are still "
        f"present in `frequency_spectrum.csv`, so no statistic is lost - only the identity "
        f"of the rare patterns is."
        if report["profiles_suppressed"] else
        "No profiles were withheld: `profiles.csv` is the complete distribution."
    )

    path.write_text(f"""# Discretized CBC profile space - aggregate release

Aggregate data underlying the manuscript on the structure of the discretized
complete blood count (CBC) profile space. No record-level data, no identifiers,
no dates, no demographics.

## Files

### `profiles.csv`
One row per distinct qualitative profile. Columns: {len(analytes)} analytes
({", ".join(analytes)}), then:

- `count` - number of records with that profile
- `percent` - share of the corpus
- `percent_cumulative` - running total over profiles sorted by decreasing count

Percentages are computed against the **full** corpus of {report['n_records']:,}
records, so they remain correct despite the suppression described below.

### `frequency_spectrum.csv`
`r`, `f_r` - the number of profiles observed exactly `r` times, released in full.
This file carries no profile identities and is sufficient to reproduce coverage
curves, the top-1 share, the singleton fraction, the Zipf slope, the Clauset
power-law fit, Shannon entropy and Hill numbers, Good-Turing novelty, Chao1,
jackknife, ACE and the rarefaction curve.

### `frequency_spectrum_v10.csv`, `frequency_spectrum_v9.csv`
The same spectrum for the algebraically reduced panels used in W3/W4/W6. V10
drops the three derived red cell indices (MCV, MCH, MCHC) and BASO; V9
additionally drops WBC. Each panel is re-discretized from the source records
rather than projected from the 14-analyte table, because a record counts as
complete over the selected analytes only - the reduced panels therefore retain a
few records that the full panel discards. Released in full, no identities.

## State legend

Each analyte is expressed as its normalized position within the reference
interval, P = (result - low) / (high - low), discretized with margin
tau = {report['margin']}:

| value | state |
|---|---|
{legend}

## Corpus

| | |
|---|---|
| Records analyzed | {report['n_records']:,} |
| Records discarded (incomplete) | {report['n_dropped']:,} |
| Distinct profiles | {report['n_profiles']:,} |
| Profiles released in `profiles.csv` | {report['profiles_released']:,} |
| Discretization margin tau | {report['margin']} |
| Analytes | {len(analytes)} |

## Disclosure control

{suppressed_note}

## Provenance

Generated by `src/make_release.py` from the analysis pipeline
(`src/run_all.py`, SEED={common.SEED}). Records lacking any analyte
value or reference limit were excluded before discretization.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=common.MARGIN)
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                    help=f"Withhold profiles seen fewer than this many times "
                         f"(default {DEFAULT_MIN_COUNT}; 1 releases everything).")
    ap.add_argument("--states", type=int, choices=[3, 5], default=5,
                    help="5 = the states used in the paper; 3 merges slightly_* into "
                         "clearly_* (coarser, far fewer unique profiles).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    common.set_seed()
    df = load_data()
    ordinal = discretize(df, margin=args.margin)
    n_dropped = len(df) - len(ordinal)

    state_names = STATE_NAMES
    if args.states == 3:
        ordinal = ordinal.replace(COARSE_MAP)
        state_names = COARSE_NAMES

    table = profile_table(ordinal)
    counts = table["count"].to_numpy()
    n_records, n_profiles = len(ordinal), len(table)

    released = table[table["count"] >= args.min_count]
    suppressed = table[table["count"] < args.min_count]

    report = {
        "margin": args.margin,
        "min_count": args.min_count,
        "n_records": n_records,
        "n_dropped": n_dropped,
        "n_profiles": n_profiles,
        "profiles_released": len(released),
        "profiles_suppressed": len(suppressed),
        "profiles_suppressed_pct": len(suppressed) / n_profiles * 100,
        "records_suppressed": int(suppressed["count"].sum()),
        "records_suppressed_pct": float(suppressed["count"].sum()) / n_records * 100,
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    released.to_csv(out / "profiles.csv", index=False, float_format="%.6f")
    frequency_spectrum(counts).to_csv(out / "frequency_spectrum.csv", index=False)

    # Reduced-panel spectra. No profile identities, so they are released in full.
    variant_rows = []
    for name, dropped in (("v10", DROP_V10), ("v9", DROP_V9)):
        params = {s: p for s, p in PARAMS_14.items() if s not in dropped}
        ord_v = discretize(df, margin=args.margin, params=params)
        if args.states == 3:
            ord_v = ord_v.replace(COARSE_MAP)
        counts_v = profile_table(ord_v)["count"].to_numpy()
        frequency_spectrum(counts_v).to_csv(out / f"frequency_spectrum_{name}.csv",
                                            index=False)
        variant_rows.append((name.upper(), len(params), len(ord_v), len(counts_v)))
    write_readme(out / "README.md", report, list(ordinal.columns), state_names)
    # read back by paper/fill_numbers.py, so the Data Availability
    # statement quotes the actual suppression rather than a hand-typed figure
    (out / "release_summary.json").write_text(json.dumps(report, indent=2))

    print(f"records          {n_records:,} ({n_dropped:,} dropped as incomplete)")
    print(f"profiles         {n_profiles:,} distinct at tau={args.margin}, "
          f"{args.states} states")
    print(f"released         {report['profiles_released']:,} profiles "
          f"(count >= {args.min_count})")
    print(f"withheld         {report['profiles_suppressed']:,} profiles "
          f"({report['profiles_suppressed_pct']:.1f}% of profiles), covering "
          f"{report['records_suppressed']:,} records "
          f"({report['records_suppressed_pct']:.2f}% of the corpus)")
    print(f"                 their counts remain in frequency_spectrum.csv")
    for label, n_params, n_rec, n_prof in variant_rows:
        print(f"{label:<17}{n_prof:,} profiles over {n_params} analytes, "
              f"{n_rec:,} records (spectrum only)")
    print(f"written to       {out}")


if __name__ == "__main__":
    main()
