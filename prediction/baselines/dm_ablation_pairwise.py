"""
Pairwise Diebold-Mariano comparison across the C1-C4 ablation matrix.

For each (split, horizon, variant) combination this script runs DM tests
between every pair of ablation configurations using `diebold_mariano_test`
from `naive_baseline.py`. The goal is the Bab 4 question: does adding HMM
or sentiment features yield a statistically significant lift over the
price-only baseline (C1)?

Inputs (read-only):
    prediction/output_horizons_cpo_only/Daily/horizon_{h}/{split}_predictions_Daily_h{h}.csv  # C1
    prediction/output_horizons_cpo_hmm/Daily/horizon_{h}/{split}_predictions_Daily_h{h}.csv   # C2
    prediction/output_horizons_cpo_sentiment/Daily/horizon_{h}/{split}_predictions_Daily_h{h}.csv  # C3
    prediction/output_horizons/Daily/horizon_{h}/{split}_predictions_Daily_h{h}.csv           # C4

Outputs:
    prediction/baselines/output/dm_ablation_pairwise_Daily_{split}.csv

Schema:
    horizon, split, variant, model_a, model_b, n, dm_stat, p_value,
    significant_at_0.05, winner

Usage:
    python prediction/baselines/dm_ablation_pairwise.py
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTION_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_PREDICTION_DIR)
if _PREDICTION_DIR not in sys.path:
    sys.path.insert(0, _PREDICTION_DIR)

from naive_baseline import diebold_mariano_test  # noqa: E402


ABLATIONS: Dict[str, str] = {
    "C1_cpo_only":      os.path.join(_PREDICTION_DIR, "output_horizons_cpo_only"),
    "C2_cpo_hmm":       os.path.join(_PREDICTION_DIR, "output_horizons_cpo_hmm"),
    "C3_cpo_sentiment": os.path.join(_PREDICTION_DIR, "output_horizons_cpo_sentiment"),
    "C4_full":          os.path.join(_PREDICTION_DIR, "output_horizons"),
}

VARIANTS: Tuple[str, ...] = ("base", "csa")
HORIZONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
SPLITS: Tuple[str, ...] = ("testing", "validation")
INTERVAL = "Daily"
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")


def _predictions_path(variant_dir: str, horizon: int, split: str) -> str:
    return os.path.join(
        variant_dir, INTERVAL, f"horizon_{horizon}",
        f"{split}_predictions_{INTERVAL}_h{horizon}.csv",
    )


def _load_aligned_errors(
    dir_a: str, dir_b: str, horizon: int, split: str, variant: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (errors_a, errors_b, n) inner-joined on Date. Errors in log-return space."""
    path_a = _predictions_path(dir_a, horizon, split)
    path_b = _predictions_path(dir_b, horizon, split)
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    pred_col = f"xgboost_{variant}_LogReturn"
    for df, path in ((df_a, path_a), (df_b, path_b)):
        for required in ("Date", "Actual_LogReturn", pred_col):
            if required not in df.columns:
                raise KeyError(f"missing column '{required}' in {path}")

    merged = df_a[["Date", "Actual_LogReturn", pred_col]].merge(
        df_b[["Date", "Actual_LogReturn", pred_col]],
        on="Date", suffixes=("_a", "_b"),
    )
    # Sanity: the two ablations should agree on the actuals (same data).
    if not np.allclose(
        merged["Actual_LogReturn_a"].to_numpy(dtype=float),
        merged["Actual_LogReturn_b"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError(
            f"Actual_LogReturn disagrees between {path_a} and {path_b} on overlapping dates"
        )

    y_true = merged["Actual_LogReturn_a"].to_numpy(dtype=float)
    e_a = y_true - merged[f"{pred_col}_a"].to_numpy(dtype=float)
    e_b = y_true - merged[f"{pred_col}_b"].to_numpy(dtype=float)
    mask = ~(np.isnan(e_a) | np.isnan(e_b))
    return e_a[mask], e_b[mask], int(mask.sum())


def _verdict(dm_stat: float, p_value: float, label_a: str, label_b: str) -> Tuple[bool, str]:
    if np.isnan(dm_stat) or np.isnan(p_value):
        return False, "undetermined"
    significant = p_value < 0.05
    if not significant:
        return False, "tie"
    return True, label_a if dm_stat < 0 else label_b


def run_split(split: str) -> pd.DataFrame:
    rows: List[Dict] = []
    pairs = list(itertools.combinations(ABLATIONS.keys(), 2))
    for horizon in HORIZONS:
        for variant in VARIANTS:
            for label_a, label_b in pairs:
                dir_a, dir_b = ABLATIONS[label_a], ABLATIONS[label_b]
                try:
                    e_a, e_b, n = _load_aligned_errors(dir_a, dir_b, horizon, split, variant)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    print(f"  [{split} h{horizon} {variant}] skip {label_a} vs {label_b}: {exc}")
                    continue
                if n < 5:
                    print(f"  [{split} h{horizon} {variant}] skip {label_a} vs {label_b}: only {n} aligned rows")
                    continue
                dm_stat, p_value = diebold_mariano_test(e_a, e_b, h=horizon, loss="squared")
                significant, winner = _verdict(dm_stat, p_value, label_a, label_b)
                rows.append({
                    "horizon": horizon,
                    "split": split,
                    "variant": variant,
                    "model_a": label_a,
                    "model_b": label_b,
                    "n": n,
                    "dm_stat": round(dm_stat, 6) if not np.isnan(dm_stat) else float("nan"),
                    "p_value": round(p_value, 6) if not np.isnan(p_value) else float("nan"),
                    "significant_at_0.05": significant,
                    "winner": winner,
                })
    cols = ["horizon", "split", "variant", "model_a", "model_b", "n",
            "dm_stat", "p_value", "significant_at_0.05", "winner"]
    return pd.DataFrame(rows, columns=cols)


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print("  PAIRWISE DIEBOLD-MARIANO ACROSS C1-C4 ABLATIONS")
    print(f"  Variants: {VARIANTS}  Horizons: {HORIZONS}  Splits: {SPLITS}")
    print("=" * 72)

    overall: List[pd.DataFrame] = []
    for split in SPLITS:
        print(f"\n[{split}]")
        df = run_split(split)
        if df.empty:
            print(f"  no rows produced for {split}")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"dm_ablation_pairwise_{INTERVAL}_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"  wrote {len(df)} rows -> {os.path.relpath(out_path, _PROJECT_ROOT)}")

        sig_count = int(df["significant_at_0.05"].sum())
        print(f"  significant (p<0.05): {sig_count}/{len(df)}")
        overall.append(df)

    if overall:
        combined = pd.concat(overall, ignore_index=True)
        sig_summary = (
            combined.groupby(["variant", "split"])["significant_at_0.05"]
            .agg(["sum", "count"]).reset_index()
            .rename(columns={"sum": "significant", "count": "total"})
        )
        print("\nSignificance summary:")
        print(sig_summary.to_string(index=False))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
