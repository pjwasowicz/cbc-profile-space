"""Ordinal discretisation of CBC profiles.

P = (result - low) / (high - low), where [low, high] is the per-patient
reference range. P is discretised into 5 ordinal states that depend on the
`margin` parameter (default `common.MARGIN`, the argmax of C(thr) from W2):

    0 clearly_low    P <= -m
    1 slightly_low   -m < P < 0
    2 norm            0 <= P <= 1
    3 slightly_high   1 < P < 1+m
    4 clearly_high    P >= 1+m

Rows in which any analyte has a missing result or reference range (P = NaN) are
discarded - the equivalent of "undefined" in the make_combinations prototype.

The implementation is vectorised (no row-wise apply): it handles ~700k records
in a few seconds and produces the same partition as the prototype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import MARGIN

# The full 14-analyte panel (names as in the parquet: RESULT_/LOW_RANGE_/HIGH_RANGE_).
# The short names follow the convention of the W3 plan.
#
# The names come from the LIMS `parameters.name` column. For the five
# differential fractions we take the absolute "#" variants, not the percentage
# "%" ones: in older files (before the switch to `par.name`) both variants
# shared a single `par.label`, which collapsed them into one column carrying
# mixed units.
PARAMS_14: dict[str, str] = {
    "RBC": "MR5 - erytrocyty",
    "HCT": "MR5 - hematokryt",
    "HGB": "MR5 - hemoglobina",
    "WBC": "MR5 - leukocyty",
    "MCH": "MR5 - MCH",
    "MCHC": "MR5 - MCHC",
    "MCV": "MR5 - MCV",
    "RDW-SD": "MR5 - RDW-SD",
    "PLT": "MR5 - płytki krwi",
    "NEUT": "MR5 - neutrofile #",
    "LYMPH": "MR5 - limfocyty #",
    "MONO": "MR5 - monocyty #",
    "EOS": "MR5 - gran. kwasochłonne #",
    "BASO": "MR5 - bazofile #",
}

# Derived indices (algebraically dependent - W3). The independent set (10):
#   RBC, HCT, HGB, RDW-SD, WBC, PLT + 4 of the 5 fractions (BASO follows from WBC
#   and the rest).
DERIVED_INDICES = ["MCV", "MCH", "MCHC"]        # from RBC/HCT/HGB
DERIVED_FRACTION = ["BASO"]                     # WBC = NEUT+LYMPH+MONO+EOS+BASO

STATE_NAMES = ["clearly_low", "slightly_low", "norm", "slightly_high", "clearly_high"]


def compute_p(df: pd.DataFrame, pl_name: str) -> pd.Series:
    """Normalised position P relative to the reference range."""
    res = df[f"RESULT_{pl_name}"]
    low = df[f"LOW_RANGE_{pl_name}"]
    high = df[f"HIGH_RANGE_{pl_name}"]
    return (res - low) / (high - low)


def discretize_series(p: pd.Series | np.ndarray, margin: float = MARGIN) -> np.ndarray:
    """Vectorised discretisation of P into states 0..4 (NaN -> -1)."""
    p = np.asarray(p, dtype=float)
    conds = [
        p <= -margin,                    # clearly_low
        (p > -margin) & (p < 0),         # slightly_low
        (p >= 0) & (p <= 1),             # norm
        (p > 1) & (p < 1 + margin),      # slightly_high
        p >= 1 + margin,                 # clearly_high
    ]
    out = np.select(conds, [0, 1, 2, 3, 4], default=-1)
    return out.astype(np.int8)


def discretize(
    df: pd.DataFrame,
    margin: float = MARGIN,
    params: dict[str, str] | None = None,
    drop_undefined: bool = True,
) -> pd.DataFrame:
    """Returns a DataFrame of ordinal states (columns = analyte short names).

    The input index is preserved. Rows carrying any state of -1 (missing data)
    are dropped when drop_undefined=True.
    """
    params = params or PARAMS_14
    cols = {}
    for short, pl_name in params.items():
        need = [f"RESULT_{pl_name}", f"LOW_RANGE_{pl_name}", f"HIGH_RANGE_{pl_name}"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            available = sorted(c.split("RESULT_", 1)[1] for c in df.columns
                               if c.startswith("RESULT_"))
            raise KeyError(
                f"{short}: missing columns {missing}. The parquet holds: {available}. "
                "If this file predates the switch to par.name, it has to be "
                "re-extracted from the source system."
            )
        cols[short] = discretize_series(compute_p(df, pl_name), margin=margin)

    ordinal = pd.DataFrame(cols, index=df.index)
    if drop_undefined:
        keep = (ordinal >= 0).all(axis=1)
        ordinal = ordinal[keep]
    return ordinal


def profile_ids(ordinal: pd.DataFrame) -> pd.Series:
    """Joins the states into a profile identifier string, e.g. '2-2-1-...'."""
    return ordinal.astype(str).agg("-".join, axis=1)


def load_data(path=None) -> pd.DataFrame:
    from common import DATA_PARQUET

    return pd.read_parquet(path or DATA_PARQUET)
