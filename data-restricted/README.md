# Record-level data (not included)

This directory is intentionally empty. The pipeline reads a single parquet file
of complete blood count records, one row per order, with three columns per
analyte (`RESULT_<name>`, `LOW_RANGE_<name>`, `HIGH_RANGE_<name>`); the analyte
names are listed in `src/discretize.py` (`PARAMS_14`).

Those records are secondary clinical laboratory data subject to patient-privacy
restrictions and cannot be published. Requests for access may be directed to the
corresponding author (pawel.wasowicz@diag.pl) and are subject to institutional
and legal approval.

With an approved copy in hand:

```bash
cp /path/to/morfologia_data_new.parquet data-restricted/
# or, without moving it:
export CBC_PARQUET=/path/to/morfologia_data_new.parquet
make all
```

Everything in this directory except this file is gitignored.
