# Structure of the Discretized Complete Blood Count Profile Space

Analysis code and aggregate data for:

> Paweł Wąsowicz, Jakub Swadźba, Tomasz Anyszek.
> *Structure of the Discretized Complete Blood Count Profile Space: A Reproducible
> Analysis of 700,000 Records.*

Every CBC result is turned into an ordinal state relative to its own reference
interval, `P = (result − low) / (high − low)`, discretized into five states with a
margin `τ`. A record therefore becomes a *qualitative profile* — a 14-tuple over
`{0,…,4}`. This repository contains the pipeline that maps 729,469 records into
that space and measures its structure, plus the aggregate distribution the results
rest on.

The record-level laboratory data are **not** here and cannot be published; see
[Data](#data). What *is* here is enough to check the arithmetic, and the
[Levels of verification](#levels-of-verification) section says precisely how far
each level goes.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make verify
```

No data access, no network, a few seconds. `make verify` recomputes the
manuscript's numbers from the deposited aggregate data and compares them,
character for character, against the macro file the paper typesets from:

```
119 macros: 73 recomputed and identical, 5 consistent with the deposit's bounds, 41 out of reach of the deposit, 0 failing
```

## How verification works

`src/verify_deposit.py` reads only `data/` and `paper/numbers_en.tex`. It shares
no code with the analysis pipeline — the estimators are written out from their
published definitions — so agreement means the manuscript's numbers follow from
the deposit, not merely that two copies of one implementation agree with each
other.

Every macro the manuscript quotes must appear in the script's registry, either
with a way to recompute it or with a stated reason why the deposit cannot reach
it. A macro in neither is a hard error, so the classification cannot go stale as
the paper changes. Four statuses:

| status | meaning |
|---|---|
| `ok` | recomputed from the deposit and identical to the published value |
| `bound` | not point-identified, but the published value lies inside the interval the deposit implies |
| `n/a` | out of reach of the deposit; the reason is printed alongside |
| `FAIL` | recomputed and different, or outside the implied interval — the run exits non-zero |

It also checks the deposit against its own report: that the spectrum and
`profiles.csv` agree with `release_summary.json` on the record and profile
counts, that the released head really does respect the withholding threshold,
and how many profiles of each Hamming shell the head contains.

### Why a few quantities come out as bounds

Everything that is a function of the count vector alone is point-identified,
because `frequency_spectrum.csv` carries the count vector complete and
unsuppressed. The exceptions are the quantities that need profile *identities* —
the Hamming geometry — which are computed on a head covering
93.33% of the corpus.

The remainder is not simply unknown. A shell at Hamming distance *d* from the
all-normal profile holds at most C(14, *d*) · 4^*d* distinct profiles; the head
shows which of them are present; and every absent profile carries at most
4 records, by definition of the threshold. That caps how
much mass each shell can still gain, and for the inner shells the cap is far
tighter than the total withheld mass — tight enough that shells 0 and 1 are
pinned to the published value outright, with no freedom left to the tail.

### What the deposit cannot reach

Four analyses are functions of the record-level data rather than of the
deposited aggregates. The deposit holds their *results*, not the means to
recompute them, and `make verify` reports each with its reason rather than
passing over it:

| Quantity | Manuscript | Why | Recorded output |
|---|---|---|---|
| Coding-resolution grid `C(τ)`, plateau, bootstrap over margins | W2 | needs the continuous position `P` inside each band; the deposit records only the discretised state | `results/w2_threshold/c_thr.csv` |
| Algebraic residuals `r_mcv`, `r_mch`, `r_mchc`, `r_wbc` | W3 | needs analyte values in physical units | `results/w3_redundancy/summary.json` |
| Spearman correlation matrix over the 14 analytes | W8 | same | `results/w8_correlation/corr_spearman.csv` |
| Audit of discarded records, kept-vs-dropped comparison | W9 | needs the discarded records; the deposit contains only complete ones | `results/w9_undefined/summary.json` |

The V10 and V9 variant statistics are *not* in this list: their frequency spectra
are deposited in full, so the reduced-panel richness, singleton fractions and
diversity numbers are recomputed exactly like the V14 ones.

## Two further checks

`make check` regenerates `paper/numbers_en.tex` from the recorded pipeline
outputs in `results/` and diffs it. Where `make verify` asks whether the numbers
are *right*, this asks whether they were *typed* — every figure in the manuscript
is a generated macro, so nothing can be edited into the text by hand. It needs no
data access either, and CI runs both on every push.

`make all` re-runs the entire pipeline and requires approved access to the
record-level parquet; see [Data](#data).

---

## Layout

```
├── src/                    the pipeline: discretization + experiments W1–W9
│   ├── common.py           seed, margin τ, paths, env/config recording
│   ├── discretize.py       P → ordinal states; the 14-analyte panel
│   ├── run_all.py          runs W1–W9 in order
│   ├── w1_structure.py     … w9_undefined.py
│   ├── make_release.py     builds data/ from the record-level input
│   └── verify_deposit.py   recomputes the manuscript from data/ alone
├── data/                   the deposit — aggregate, publishable
│   ├── profiles.csv        one row per profile with count ≥ 5
│   ├── frequency_spectrum.csv   f_r, complete, unsuppressed
│   ├── frequency_spectrum_v10.csv, _v9.csv   the same, reduced panels
│   ├── release_summary.json     disclosure-control report
│   └── README.md           dataset documentation (also the Zenodo landing text)
├── data-restricted/        empty; where the record-level parquet goes
├── results/                per-experiment outputs, one directory per experiment
│   └── w*/                 summary.json, config.json, env.json, CSVs, figures
└── paper/
    ├── fill_numbers.py     results/ → LaTeX macros
    └── numbers_en.tex      the 119 macros the manuscript quotes
```

Each `results/w*/` directory carries `config.json` (the hyperparameters that run
used) and `env.json` (Python version, platform, full `pip freeze`, seed), written
by the run itself. The environment that produced the published numbers is
therefore recorded per experiment, not merely asserted in this README.

## Data

### The deposit (`data/`)

`profiles.csv` gives, for each distinct qualitative profile, the ordinal state of
each of the 14 analytes together with its count and its share of the corpus.
`frequency_spectrum.csv` gives the frequency-of-frequencies distribution.
`data/README.md` documents both, including the state legend.

| | |
|---|---|
| Records analyzed | 727,982 |
| Records discarded as incomplete | 1,487 |
| Distinct profiles | 40,376 |
| Profiles in `profiles.csv` | 5,410 |
| Discretization margin τ | 0.14 |
| Analytes | 14 |

**Disclosure control.** Profiles observed fewer than 5 times are
withheld from `profiles.csv`:

| | profiles | records | share of corpus |
|---|---|---|---|
| released | 5,410 | 679,391 | 93.33% |
| withheld | 34,966 | 48,591 | 6.67% |

A 14-dimensional ordinal pattern seen only a handful of times is, in principle, a
quasi-identifier, so patterns below the threshold are withheld. The withheld mass
is dominated by singletons — 26,214 of those records belong to
profiles observed exactly once — which no threshold above 1 can release. The
threshold trades the share of the corpus the released table covers against the
rarity of the patterns it exposes; it is recorded in `release_summary.json`, and
`python src/make_release.py --min-count N` regenerates the deposit at any other
value.

The withheld counts remain in `frequency_spectrum.csv`, which carries no profile
identities, so no *statistic* is lost — only the identity of the rare patterns.
`paper/fill_numbers.py` reads `release_summary.json` back when generating the
manuscript's macros, so the Data Availability statement quotes the suppression
that actually happened rather than a figure typed in alongside it.

For the same reason, the complete profile table including singletons
(`profile_counts.csv`, an intermediate output of W1) is deliberately **not**
published; it would defeat the threshold.

### The record-level data

729,469 complete blood counts extracted from the laboratory information system
of Diagnostyka S.A. They cannot be shared publicly: they are secondary clinical
laboratory data subject to patient-privacy restrictions. Requests for access may
be directed to the corresponding author (pawel.wasowicz@diag.pl) and are subject
to institutional and legal approval.

The extraction query itself is internal to the data provider and is not part of
this repository, so the inclusion criteria it encodes are stated here instead:

- complete blood count orders only, selected by test code in the laboratory
  information system;
- **Sysmex XN-10 analysers only.** Other instruments in the network (XN-1000,
  XN-550, XN-430) are excluded, so that reference intervals and reporting
  resolution are uniform across the corpus; results with no recorded analyser
  are excluded as well.
- **adults**: patients aged 18 or over on the date of the order. Patients with
  no recorded date of birth are excluded, because age cannot be confirmed.
- **per-patient reference limits**: `low` and `high` are the norms the laboratory
  actually applied to that result, not values from a global table, so `P` is
  computed against the interval that was in force for that patient.
- consecutive months of routine operation, taken as a whole rather than sampled.

A record enters the analysis only if all 14 analytes carry both a result and a
reference interval; the 1,487 records that fail this are audited in W9, which
tests whether the exclusion is selective.

Holders of an approved copy can run the full pipeline:

```bash
export CBC_PARQUET=/path/to/morfologia_data_new.parquet   # or drop it in data-restricted/
make all          # W1–W9, writes into results/
make deposit      # rebuilds data/ from the records
make numbers      # regenerates paper/numbers_en.tex
```

`make all` takes on the order of tens of minutes; W2 (a bootstrap over a margin
grid), W5 (a bootstrap power-law goodness-of-fit) and W7 (t-SNE and MDS on the
Hamming metric) dominate the runtime. Everything is seeded (`SEED = 42`) and
deterministic.

## Experiments

| | Question | Key outputs |
|---|---|---|
| W1 | How large and how concentrated is the profile space? | coverage curve, top-1 share, singleton fraction, Zipf + Clauset fits |
| W2 | Is the choice of margin τ arbitrary? | `C(τ)` grid, its plateau, bootstrap over the argmax |
| W3 | How much of the 14-analyte panel is algebraically redundant? | residuals of the derived indices, V14/V10/V9 variant table |
| W4 | Does the space saturate at 700k records? | Good–Turing, Chao1, jackknife, ACE, analytical rarefaction |
| W5 | Is a power law a defensible description of the tail? | Clauset MLE with a bootstrap *p*-value |
| W6 | What does the geometry look like? | Hamming distance from all-norm, entropy, Hill numbers, `f_r` |
| W7 | Visualization of the space | Hamming graph, MDS, t-SNE, profile heatmap, icicle shells |
| W8 | Is the redundancy of W3 visible in the correlations? | Spearman matrix over the 14 analytes |
| W9 | Is the exclusion of incomplete records selective? | missingness per analyte, kept-vs-dropped comparison |

`τ = 0.14` throughout, the argmax of `C(τ)` from W2, which also sits inside its
1% plateau. It is defined once, in `src/common.py`; the earlier arrangement of
one copy per script is how the steps drifted apart.

## Environment

Python 3.12 with the pinned versions in `requirements.txt`. The exact
environment behind the published numbers is recorded in each
`results/w*/env.json`, so it can be reconstructed per experiment rather than
merely asserted here.

## Ethics

The study was conducted in accordance with the Declaration of Helsinki and
approved by the Bioethics Committee of Andrzej Frycz Modrzewski Krakow University
(resolution no. 48/2026 of 16 July 2026; opinion no. KB/UAFM/48/2026). It is a
retrospective analysis of anonymized secondary laboratory data with no
intervention: no patients were recruited, no additional visits were conducted and
no additional samples were drawn. Patient consent was waived on that basis.

## Citation

See `CITATION.cff`. Until the paper appears, cite this repository.

## License

Code in `src/` and `paper/`: MIT (`LICENSE`).
Data in `data/` and `results/`: CC BY 4.0 (`LICENSE-DATA`).
