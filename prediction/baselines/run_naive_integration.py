"""
End-to-end orchestrator for naive-baseline integration (Hypothesis H4).

What it does
------------
1. Reads parametric rows from the existing `horizon_summary_Daily_{split}.csv`
   under `prediction/output_horizons/` (the full CPO + sentiment + HMM variant).
2. Computes naive baselines (naive_rw, historical_mean, seasonal_naive_7)
   via `naive_evaluator.evaluate_all_naive_baselines` without re-fitting any
   parametric model.
3. Makes a timestamped backup of each existing summary CSV, then rewrites it
   with parametric rows (untouched) first and naive rows appended at the end.
4. Runs `dm_comparison.compare_best_vs_naive` on both splits and writes
   `dm_comparison_Daily_{split}.csv` next to the summary files.
5. Prints a defense-ready summary table per split: best parametric vs
   naive_rw per horizon, with DM verdict.

Design constraints
------------------
- Does NOT re-run horizon_forecast.py. All parametric numbers come from the
  already-saved CSVs; naive rows are merely appended.
- Does NOT modify `naive_baseline.py`.
- Only operates on Daily interval and only on the full model variant
  (`output_horizons/`) per user instruction.

Usage
-----
    python prediction/baselines/run_naive_integration.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTION_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_PREDICTION_DIR)
if _PREDICTION_DIR not in sys.path:
    sys.path.insert(0, _PREDICTION_DIR)

from baselines.naive_evaluator import evaluate_all_naive_baselines, NAIVE_MODELS  # noqa: E402
from baselines.dm_comparison import compare_best_vs_naive, _select_best_parametric_at_h1  # noqa: E402


SCHEMA_COLUMNS = [
    "Horizon", "Model", "Optimization",
    "MAPE", "sMAPE", "RMSE",
    "Directional_Accuracy", "R2_Price", "R2_LogReturn",
]
PARAMETRIC_MODELS = ("xgboost", "random_forest", "arimax", "sarimax")
PARAMETRIC_OPTS = ("BASE", "CSA", "BAYESIAN")


# =============================================================================
# Summary I/O
# =============================================================================

def _summary_path(variant_dir: str, interval: str, split: str) -> str:
    """
    Resolve the horizon_summary CSV path for a split.

    Handles a wrinkle in this repo: horizon_forecast.py writes
    `horizon_summary_Daily_validation.csv`, but the current on-disk file was
    manually renamed to `horizon_summary_Daily_validation_comb.csv` (likely to
    differentiate from an older run). We try the canonical name first and fall
    back to the `_comb` suffix for the validation split.
    """
    base = os.path.join(variant_dir, interval, f"horizon_summary_{interval}_{split}.csv")
    if os.path.exists(base):
        return base
    if split == "validation":
        alt = os.path.join(variant_dir, interval, f"horizon_summary_{interval}_{split}_comb.csv")
        if os.path.exists(alt):
            return alt
    return base


def _backup_csv(path: str) -> str:
    """Copy `path` to `<path>.pre_naive_backup_<UTC-timestamp>`; returns backup path."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup = f"{path}.pre_naive_backup_{ts}"
    shutil.copy2(path, backup)
    return backup


def _strip_any_prior_naive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NAIVE rows from a summary DataFrame to keep the run idempotent."""
    if df.empty or "Optimization" not in df.columns:
        return df
    return df[df["Optimization"].astype(str).str.upper() != "NAIVE"].copy()


def _merge_parametric_and_naive(
    parametric_df: pd.DataFrame,
    naive_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parametric rows first in their original on-disk order (preserved verbatim);
    naive rows appended after, sorted by Horizon then Model for determinism.
    """
    # Ensure all schema columns exist in both frames
    for df in (parametric_df, naive_df):
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

    parametric_preserved = parametric_df.reset_index(drop=True)
    naive_sorted = naive_df.sort_values(["Horizon", "Model"]).reset_index(drop=True)

    combined = pd.concat(
        [parametric_preserved[SCHEMA_COLUMNS], naive_sorted[SCHEMA_COLUMNS]],
        ignore_index=True,
    )
    return combined


# =============================================================================
# Defense-ready print table
# =============================================================================

def _best_parametric_per_horizon(
    parametric_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    dm_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each horizon, pick the best parametric (min MAPE) and pair with
    naive_rw metrics and the DM verdict. Used purely for the printed summary."""
    rows: List[Dict] = []
    if parametric_df.empty:
        return pd.DataFrame()

    horizons = sorted(parametric_df["Horizon"].dropna().unique().astype(int).tolist())
    for h in horizons:
        para_h = parametric_df[parametric_df["Horizon"] == h]
        para_h = para_h[para_h["Model"].isin(PARAMETRIC_MODELS)
                        & para_h["Optimization"].isin(PARAMETRIC_OPTS)]
        para_h = para_h.dropna(subset=["MAPE"])
        if para_h.empty:
            continue
        best = para_h.sort_values("MAPE").iloc[0]
        naive_rw_row = naive_df[
            (naive_df["Horizon"] == h) & (naive_df["Model"] == "naive_rw")
        ]
        naive_mape = float(naive_rw_row["MAPE"].iloc[0]) if not naive_rw_row.empty else float("nan")
        naive_da = (float(naive_rw_row["Directional_Accuracy"].iloc[0])
                    if not naive_rw_row.empty else float("nan"))

        dm_row = dm_df[dm_df["horizon"] == h] if not dm_df.empty else pd.DataFrame()
        dm_stat = float(dm_row["dm_stat"].iloc[0]) if not dm_row.empty else float("nan")
        dm_p = float(dm_row["p_value"].iloc[0]) if not dm_row.empty else float("nan")
        verdict = str(dm_row["better_model"].iloc[0]) if not dm_row.empty else "-"

        rows.append({
            "h": h,
            "best_model": f"{best['Model']}_{str(best['Optimization']).lower()}",
            "best_MAPE": round(float(best["MAPE"]), 4),
            "best_DA": round(float(best["Directional_Accuracy"]), 2),
            "naive_MAPE": round(naive_mape, 4) if not np.isnan(naive_mape) else np.nan,
            "naive_DA": round(naive_da, 2) if not np.isnan(naive_da) else np.nan,
            "dm_stat": round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
            "p_value": round(dm_p, 4) if not np.isnan(dm_p) else np.nan,
            "winner": verdict,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main integration
# =============================================================================

def run_integration(
    variant_dir: str,
    interval: str = "Daily",
    horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    splits: Tuple[str, ...] = ("testing", "validation"),
    make_backup: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Execute the full naive-integration pipeline. Returns dict split -> combined DataFrame."""
    variant_dir = os.path.abspath(variant_dir)
    if not os.path.isdir(variant_dir):
        raise FileNotFoundError(f"variant_dir does not exist: {variant_dir}")

    print("=" * 72)
    print(f"  NAIVE BASELINE INTEGRATION - variant: {variant_dir}")
    print(f"  Interval: {interval}  Horizons: {list(horizons)}  Splits: {list(splits)}")
    print("=" * 72)

    # ---- Step 1: compute naive rows ----
    print("\n[1/4] Computing naive baselines...")
    naive_per_split = evaluate_all_naive_baselines(
        variant_dir=variant_dir, interval=interval,
        horizons=horizons, splits=splits,
    )
    for split, df in naive_per_split.items():
        print(f"    {split}: {len(df)} naive rows")

    # ---- Step 2: append to horizon_summary CSVs ----
    print("\n[2/4] Rewriting horizon_summary CSVs (parametric first, naive appended)...")
    combined_by_split: Dict[str, pd.DataFrame] = {}
    for split in splits:
        path = _summary_path(variant_dir, interval, split)
        if not os.path.exists(path):
            print(f"    {split}: {path} not found - skipping")
            continue

        try:
            existing = pd.read_csv(path)
        except Exception as exc:
            print(f"    {split}: failed to read existing summary - skipping ({exc})")
            continue

        parametric_only = _strip_any_prior_naive_rows(existing)
        naive_rows = naive_per_split.get(split, pd.DataFrame(columns=SCHEMA_COLUMNS))
        combined = _merge_parametric_and_naive(parametric_only, naive_rows)

        if make_backup:
            backup = _backup_csv(path)
            print(f"    {split}: backup -> {os.path.relpath(backup, _PROJECT_ROOT)}")
        try:
            combined.to_csv(path, index=False)
        except Exception as exc:
            print(f"    {split}: FAILED to write {path} ({exc})")
            continue
        print(f"    {split}: wrote {len(combined)} rows ({len(parametric_only)} parametric + "
              f"{len(naive_rows)} naive) -> {os.path.relpath(path, _PROJECT_ROOT)}")
        combined_by_split[split] = combined

    # ---- Step 3: DM comparison per split ----
    print("\n[3/4] Running Diebold-Mariano comparison...")
    dm_by_split: Dict[str, pd.DataFrame] = {}
    for split in splits:
        try:
            dm_df = compare_best_vs_naive(
                variant_dir=variant_dir, interval=interval,
                horizons=horizons, split=split,
            )
        except FileNotFoundError as exc:
            print(f"    {split}: {exc} - skipping DM")
            continue
        except ValueError as exc:
            print(f"    {split}: {exc} - skipping DM")
            continue

        dm_path = os.path.join(variant_dir, interval, f"dm_comparison_{interval}_{split}.csv")
        try:
            dm_df.to_csv(dm_path, index=False)
        except Exception as exc:
            print(f"    {split}: FAILED to write {dm_path} ({exc})")
            continue
        print(f"    {split}: wrote {len(dm_df)} rows -> {os.path.relpath(dm_path, _PROJECT_ROOT)}")
        dm_by_split[split] = dm_df

    # ---- Step 4: defense-ready summary printing ----
    print("\n[4/4] Comparison summary (best parametric per horizon vs naive_rw):")
    for split in splits:
        if split not in combined_by_split:
            continue
        combined = combined_by_split[split]
        parametric_only = combined[combined["Optimization"].astype(str).str.upper() != "NAIVE"]
        naive_only = combined[combined["Optimization"].astype(str).str.upper() == "NAIVE"]
        dm_df = dm_by_split.get(split, pd.DataFrame())

        print(f"\n  -- {split.upper()} split --")
        tbl = _best_parametric_per_horizon(parametric_only, naive_only, dm_df)
        if tbl.empty:
            print("    (no data)")
        else:
            print(tbl.to_string(index=False))

    print("\nNote: Directional Accuracy for naive_rw under calculate_metrics() "
          "uses the bucket test (lr_true > 0) == (lr_pred > 0). With naive_rw "
          "predicting lr_pred = 0, (lr_pred > 0) is uniformly False, so DA = "
          "fraction of non-positive actual log returns - empirically ~48% on "
          "CPO daily data. Interpret as a 50% coin-flip benchmark, NOT 0%. "
          "Both definitions say 'no directional signal'; only the reporting "
          "convention differs.")
    print("=" * 72)
    return combined_by_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Naive-baseline integration (H4 control experiment)")
    parser.add_argument("--variant-dir", type=str,
                        default=os.path.join(_PREDICTION_DIR, "output_horizons"),
                        help="Root directory of a horizon_forecast variant "
                             "(default: prediction/output_horizons, the full model)")
    parser.add_argument("--interval", type=str, default="Daily", choices=["Daily"])
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up existing summary CSVs before rewriting")
    args = parser.parse_args()

    try:
        run_integration(
            variant_dir=args.variant_dir,
            interval=args.interval,
            horizons=(1, 2, 3, 4, 5, 6, 7),
            splits=("testing", "validation"),
            make_backup=not args.no_backup,
        )
    except Exception as exc:
        print(f"\nERROR: integration failed - {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
