"""
Diebold-Mariano pairwise comparison: best parametric model vs naive_rw.

Implements Task 4 of the H4 integration:
  1. Determine the best-performing parametric model at horizon h=1 by min MAPE
     on the testing split summary CSV.
  2. For every horizon, load saved predictions from the per-horizon CSVs,
     compute forecast errors in log-return space for both the best parametric
     model and naive_rw (which predicts zero log return), and run the
     Diebold-Mariano test via `naive_baseline.diebold_mariano_test`.
  3. Emit a CSV with columns:
        horizon, best_model, naive_model, dm_stat, p_value,
        significant_at_0.05, better_model

No refit of any parametric model is required: predictions are read from the
already-saved `{split}_predictions_Daily_h{h}.csv` files produced by
horizon_forecast.py.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTION_DIR = os.path.dirname(_THIS_DIR)
if _PREDICTION_DIR not in sys.path:
    sys.path.insert(0, _PREDICTION_DIR)

from naive_baseline import diebold_mariano_test  # noqa: E402


PARAMETRIC_MODELS: Tuple[str, ...] = ("xgboost",)
PARAMETRIC_OPTS: Tuple[str, ...] = ("BASE", "CSA")


# =============================================================================
# Helpers
# =============================================================================

def _summary_path(variant_dir: str, interval: str, split: str) -> str:
    """
    Resolve the horizon_summary CSV path for a split.

    Falls back to `horizon_summary_{interval}_validation_comb.csv` when the
    canonical `_validation.csv` is absent (the repo currently ships the _comb
    rename for the validation split).
    """
    base = os.path.join(variant_dir, interval, f"horizon_summary_{interval}_{split}.csv")
    if os.path.exists(base):
        return base
    if split == "validation":
        alt = os.path.join(variant_dir, interval, f"horizon_summary_{interval}_{split}_comb.csv")
        if os.path.exists(alt):
            return alt
    return base


def _predictions_path(variant_dir: str, interval: str, horizon: int, split: str) -> str:
    return os.path.join(
        variant_dir, interval, f"horizon_{horizon}",
        f"{split}_predictions_{interval}_h{horizon}.csv",
    )


def _select_best_parametric_at_h1(
    summary_df: pd.DataFrame,
    metric: str = "MAPE",
    lower_is_better: bool = True,
) -> Tuple[str, str]:
    """
    From a horizon_summary DataFrame, return (model, optimization) of the best
    parametric model at h=1 by the given metric.

    Parametric candidates are restricted to the in-scope (xgboost) × {base,csa}
    grid. Legacy rows (RF / ARIMAX / SARIMAX / Bayesian) and naive rows already
    appended are ignored — they remain in the historical CSVs for reference.
    """
    if summary_df.empty:
        raise ValueError("summary DataFrame is empty — cannot select best parametric model")

    h1 = summary_df[summary_df["Horizon"] == 1].copy()
    h1 = h1[h1["Model"].isin(PARAMETRIC_MODELS) & h1["Optimization"].isin(PARAMETRIC_OPTS)]
    if h1.empty:
        raise ValueError("No parametric rows found at horizon=1 in summary")

    if metric not in h1.columns:
        raise ValueError(f"metric '{metric}' not in summary columns: {list(h1.columns)}")

    # Drop rows where the metric is NaN (failed fits)
    h1 = h1.dropna(subset=[metric])
    if h1.empty:
        raise ValueError(f"All parametric rows at h=1 have NaN {metric}")

    h1_sorted = h1.sort_values(metric, ascending=lower_is_better)
    best = h1_sorted.iloc[0]
    return str(best["Model"]), str(best["Optimization"]).upper()


def _load_predictions_csv(
    variant_dir: str, interval: str, horizon: int, split: str
) -> Optional[pd.DataFrame]:
    path = _predictions_path(variant_dir, interval, horizon, split)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise IOError(f"Failed to read {path}: {exc}") from exc


def _best_model_col(model: str, opt: str) -> str:
    """Column name in predictions CSV for the (model, opt) log-return series."""
    return f"{model}_{opt.lower()}_LogReturn"


# =============================================================================
# Public API
# =============================================================================

def compare_best_vs_naive(
    variant_dir: str,
    interval: str = "Daily",
    horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    split: str = "testing",
    metric_for_selection: str = "MAPE",
) -> pd.DataFrame:
    """
    Run the Diebold-Mariano comparison for one split.

    Parameters
    ----------
    variant_dir : str
        Variant root (e.g. `prediction/output_horizons`).
    interval : str
    horizons : tuple of int
    split : {'testing', 'validation'}
        Split whose horizon_summary we use to pick "best parametric at h=1" and
        whose prediction CSVs we evaluate the DM test against.
    metric_for_selection : str
        Metric used to pick the best parametric at h=1 (default 'MAPE').

    Returns
    -------
    pd.DataFrame with columns
        ['horizon', 'best_model', 'naive_model', 'dm_stat', 'p_value',
         'significant_at_0.05', 'better_model'].
    """
    summary_path = _summary_path(variant_dir, interval, split)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"horizon_summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    best_model, best_opt = _select_best_parametric_at_h1(summary_df, metric=metric_for_selection)
    best_label = f"{best_model}_{best_opt.lower()}"
    print(f"  [DM] Best parametric at h=1 by {metric_for_selection} ({split}): {best_label}")

    rows: List[Dict] = []
    for horizon in horizons:
        preds_df = _load_predictions_csv(variant_dir, interval, horizon, split)
        if preds_df is None:
            print(f"  [DM] skip horizon {horizon}: predictions CSV missing")
            continue
        if "Actual_LogReturn" not in preds_df.columns:
            print(f"  [DM] skip horizon {horizon}: Actual_LogReturn column missing")
            continue

        best_col = _best_model_col(best_model, best_opt)
        if best_col not in preds_df.columns:
            print(f"  [DM] skip horizon {horizon}: column '{best_col}' missing")
            continue

        y_true_lr = preds_df["Actual_LogReturn"].to_numpy(dtype=float)
        y_pred_best_lr = preds_df[best_col].to_numpy(dtype=float)
        y_pred_naive_lr = np.zeros_like(y_true_lr)

        err_best = y_true_lr - y_pred_best_lr
        err_naive = y_true_lr - y_pred_naive_lr

        # Drop any rows where either error is NaN (model fit failure)
        mask = ~(np.isnan(err_best) | np.isnan(err_naive))
        if mask.sum() < 5:
            print(f"  [DM] skip horizon {horizon}: too few valid rows ({mask.sum()})")
            continue

        dm_stat, p_value = diebold_mariano_test(
            errors_model_a=err_best[mask],
            errors_model_b=err_naive[mask],
            h=horizon,
            loss="squared",
        )

        # Interpretation: H0 = equal accuracy. Low p-value → reject H0.
        # errors_model_a is the best parametric; negative dm_stat means A has
        # lower squared loss (smaller errors) than B — i.e., parametric beats
        # naive. Positive dm_stat with low p → naive beats parametric.
        significant = bool(not np.isnan(p_value) and p_value < 0.05)
        if np.isnan(dm_stat):
            better = "undetermined"
        elif not significant:
            better = "tie"
        elif dm_stat < 0:
            better = best_label
        else:
            better = "naive_rw"

        rows.append({
            "horizon": horizon,
            "best_model": best_label,
            "naive_model": "naive_rw",
            "dm_stat": round(dm_stat, 6) if not np.isnan(dm_stat) else float("nan"),
            "p_value": round(p_value, 6) if not np.isnan(p_value) else float("nan"),
            "significant_at_0.05": significant,
            "better_model": better,
        })

    cols = ["horizon", "best_model", "naive_model", "dm_stat", "p_value",
            "significant_at_0.05", "better_model"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("horizon").reset_index(drop=True)


# =============================================================================
# Self-test
# =============================================================================

def _self_test() -> None:
    project_root = os.path.dirname(_PREDICTION_DIR)
    variant_dir = os.path.join(project_root, "prediction", "output_horizons")
    print(f"Self-test against variant_dir = {variant_dir}")
    if not os.path.isdir(variant_dir):
        print("  SKIP: variant_dir not present")
        return

    for split in ("testing", "validation"):
        try:
            df = compare_best_vs_naive(variant_dir, interval="Daily",
                                        horizons=(1, 2, 3, 4, 5, 6, 7),
                                        split=split)
        except FileNotFoundError as exc:
            print(f"  {split}: {exc}")
            continue
        print(f"\n[{split}] {len(df)} rows")
        if not df.empty:
            print(df.to_string(index=False))


if __name__ == "__main__":
    _self_test()
