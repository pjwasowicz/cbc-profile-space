"""Shared reproducibility helpers for the CBC profile-space analysis.

A single seed SEED=42, output into results/<experiment>/, and
alongside the CSV an env.json (package versions, seed, timestamp) plus a
config.json (hyperparameters).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 42

# Discretisation margin used across the whole pipeline. Data-driven value:
# argmax C(thr) from W2, which also sits inside its 1% plateau. Keep it here
# only - duplicating it per script is how the steps drifted apart before.
MARGIN = 0.14

# Raster resolution for every saved figure. 300 dpi is the threshold most
# publishers state; defined here for the same reason as MARGIN, so the figures
# cannot drift apart from one another.
FIG_DPI = 300

# Repo root = one level above this file (src/common.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Record-level input. NOT part of this repository: the CBC records are subject to
# patient-privacy restrictions (see README, section "Data"). Holders of an
# approved copy either drop it into data-restricted/ under the name below or
# point the CBC_PARQUET environment variable at it.
DATA_PARQUET = Path(os.environ.get(
    "CBC_PARQUET",
    REPO_ROOT / "data-restricted" / "morfologia_data_new.parquet",
))

RESULTS_ROOT = REPO_ROOT / "results"


def set_seed(seed: int = SEED) -> None:
    """Sets the global seed (numpy; extend to torch/sklearn in Paper 2)."""
    np.random.seed(seed)


def experiment_dir(name: str) -> Path:
    d = RESULTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        )
        return sorted(out.strip().splitlines())
    except Exception:
        return []


def write_env(exp_dir: Path, seed: int = SEED) -> Path:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _pip_freeze(),
    }
    p = exp_dir / "env.json"
    p.write_text(json.dumps(env, indent=2, ensure_ascii=False))
    return p


def write_config(exp_dir: Path, config: dict) -> Path:
    p = exp_dir / "config.json"
    p.write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))
    return p
