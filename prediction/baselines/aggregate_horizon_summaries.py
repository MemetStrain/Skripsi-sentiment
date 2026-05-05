"""
Aggregate per-config horizon summaries (C1, C2, C3, C4) into a single
wide table per split, so the four ablation configurations can be
compared side-by-side at a glance.

Inputs (read-only) — the post-naive-integration summary files:
    prediction/output_horizons_cpo_only/Daily/horizon_summary_Daily_{split}.csv      # C1
    prediction/output_horizons_cpo_hmm/Daily/horizon_summary_Daily_{split}.csv       # C2
    prediction/output_horizons_cpo_sentiment/Daily/horizon_summary_Daily_{split}.csv # C3
    prediction/output_horizons/Daily/horizon_summary_Daily_{split}.csv               # C4

Output (committable):
    prediction/baselines/output/horizon_summary_combined_Daily_{split}.csv

Schema (wide):
    Horizon, Model, Optimization, then for every metric in
    {MAPE, sMAPE, RMSE, Directional_Accuracy, R2_Price, R2_LogReturn}
    one column per config:
        <metric>_C1_cpo_only, <metric>_C2_cpo_hmm,
        <metric>_C3_cpo_sentiment, <metric>_C4_full
    Rows are sorted CSA -> BASE -> NAIVE, then by Horizon, then Model.

Usage:
    python prediction/baselines/aggregate_horizon_summaries.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTION_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_PREDICTION_DIR)
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")

CONFIGS: Dict[str, str] = {
    "C1_cpo_only":      os.path.join(_PREDICTION_DIR, "output_horizons_cpo_only"),
    "C2_cpo_hmm":       os.path.join(_PREDICTION_DIR, "output_horizons_cpo_hmm"),
    "C3_cpo_sentiment": os.path.join(_PREDICTION_DIR, "output_horizons_cpo_sentiment"),
    "C4_full":          os.path.join(_PREDICTION_DIR, "output_horizons"),
}
SPLITS = ("testing", "validation")
INTERVAL = "Daily"
METRICS: List[str] = ["MAPE", "sMAPE", "RMSE", "Directional_Accuracy", "R2_Price", "R2_LogReturn"]
KEY_COLS: List[str] = ["Horizon", "Model", "Optimization"]
OPT_ORDER = {"CSA": 0, "BASE": 1, "NAIVE": 2}


def _summary_path(variant_dir: str, split: str) -> str:
    return os.path.join(variant_dir, INTERVAL, f"horizon_summary_{INTERVAL}_{split}.csv")


def aggregate_split(split: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for config, variant_dir in CONFIGS.items():
        path = _summary_path(variant_dir, split)
        if not os.path.exists(path):
            print(f"  [{split}] {config}: missing {path} - skipping")
            continue
        df = pd.read_csv(path)
        missing = [c for c in KEY_COLS + METRICS if c not in df.columns]
        if missing:
            print(f"  [{split}] {config}: missing columns {missing} - skipping")
            continue
        df = df[KEY_COLS + METRICS].copy()
        df["Config"] = config
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)

    # Wide pivot: rows = (Horizon, Model, Optimization), columns grouped by metric x config.
    wide_df = long_df.pivot_table(
        index=KEY_COLS, columns="Config", values=METRICS, aggfunc="first",
    )
    # Flatten ("MAPE", "C1_cpo_only") -> "MAPE_C1_cpo_only".
    wide_df.columns = [f"{metric}_{config}" for metric, config in wide_df.columns]

    # Stable column ordering: keep the metric-major / config-minor layout for readability.
    metric_to_configs = {m: [f"{m}_{c}" for c in CONFIGS.keys()] for m in METRICS}
    ordered = [col for m in METRICS for col in metric_to_configs[m] if col in wide_df.columns]
    wide_df = wide_df.reindex(ordered, axis=1).reset_index()

    wide_df["_opt_sort"] = wide_df["Optimization"].str.upper().map(OPT_ORDER).fillna(99)
    wide_df = (
        wide_df.sort_values(["_opt_sort", "Horizon", "Model"], kind="stable")
        .drop(columns="_opt_sort")
        .reset_index(drop=True)
    )
    return wide_df


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print("  AGGREGATE HORIZON SUMMARIES (C1, C2, C3, C4)")
    print("=" * 72)
    for split in SPLITS:
        print(f"\n[{split}]")
        df = aggregate_split(split)
        if df.empty:
            print("  no data — skipped")
            continue
        out_path = os.path.join(
            OUTPUT_DIR, f"horizon_summary_combined_{INTERVAL}_{split}.csv"
        )
        df.to_csv(out_path, index=False)
        print(f"  wrote {len(df)} rows x {len(df.columns)} cols -> "
              f"{os.path.relpath(out_path, _PROJECT_ROOT)}")
        # Spot-print just the CSA-MAPE columns for quick eyeballing.
        mape_cols = [c for c in df.columns if c.startswith("MAPE_C")]
        if mape_cols and "Optimization" in df.columns:
            csa = df[df["Optimization"].str.upper() == "CSA"][["Horizon"] + mape_cols]
            if not csa.empty:
                print("  CSA MAPE by config:")
                print(csa.to_string(index=False))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
