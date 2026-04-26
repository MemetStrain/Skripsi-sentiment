"""
Naive baseline evaluator for multi-horizon CPO forecasting.

Reads the per-horizon prediction CSVs written by horizon_forecast.py
(`training_predictions_Daily_h{h}.csv`,
 `testing_predictions_Daily_h{h}.csv`,
 `validation_predictions_Daily_h{h}.csv`) and computes three naive
baselines (naive_rw, historical_mean, seasonal_naive_7) in **log-return
space**, feeding them through the same `calculate_metrics()` helper the
parametric models use so the comparison is apples-to-apples.

Design notes
------------
- naive_baseline.py is a frozen dependency: we call `predict_historical_mean`
  (natural API fit), `compute_naive_metrics`, and `diebold_mariano_test`
  from it, but DO NOT modify it.
- For naive_rw and seasonal_naive_7, naive_baseline.py's price-space API
  doesn't align with horizon_forecast.py's per-row anchor semantics (each
  test row has its own Close_t anchor and a h-step-ahead log-return target
  stored at that row). This wrapper re-expresses those two baselines in
  log-return space directly.
- Directional Accuracy for naive_rw with `calculate_metrics` is ~50% (a
  coin-flip benchmark), NOT 0%. `calculate_metrics` uses the bucket test
  `(lr_true > 0) == (lr_pred > 0)`: when lr_pred is zero, (lr_pred > 0)
  is uniformly False, so DA equals the fraction of non-positive actual
  log returns — empirically ~48% on CPO daily data. This is still a "no
  directional signal" interpretation, just framed against the 50%
  coin-flip threshold rather than 0%.

Schema written to horizon_summary_*.csv:
    Horizon, Model, Optimization, MAPE, sMAPE, RMSE,
    Directional_Accuracy, R2_Price, R2_LogReturn
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure `prediction/` is importable when this module is invoked as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTION_DIR = os.path.dirname(_THIS_DIR)
if _PREDICTION_DIR not in sys.path:
    sys.path.insert(0, _PREDICTION_DIR)

from utils.forecast_utils import calculate_metrics  # noqa: E402
from naive_baseline import predict_historical_mean  # noqa: E402


NAIVE_MODELS: Tuple[str, ...] = ("naive_rw", "historical_mean", "seasonal_naive_7")
SEASON_LENGTH: int = 7


# =============================================================================
# Predictors in log-return space (aligned per test row)
# =============================================================================

def _predict_naive_rw_lr(y_true_lr: np.ndarray) -> np.ndarray:
    """
    Naive random walk in log-return space: y_hat_lr[i] = 0.

    Rationale: horizon_forecast.py stores the h-step log return at each test
    row. The random walk assumption (price_{t+h} = price_t) implies
    log(price_{t+h} / price_t) = 0.

    Returns a zero array with the same shape as y_true_lr.
    """
    return np.zeros(len(y_true_lr), dtype=float)


def _predict_historical_mean_lr(
    y_train_lr: np.ndarray,
    close_train: np.ndarray,
    close_anchor_split: np.ndarray,
) -> np.ndarray:
    """
    Historical mean baseline in log-return space.

    Uses `predict_historical_mean(close_train, n_predictions)` from
    `naive_baseline.py` (which predicts a constant *price* equal to the
    training-set mean close) and converts each constant-price prediction
    into the log-return required at that test row:

        y_pred_lr[i] = log(mean(close_train) / close_anchor[i])

    This matches the spirit of naive_baseline.py while producing a
    log-return prediction compatible with `calculate_metrics`.
    """
    # predict_historical_mean returns a constant array in price space
    price_preds = predict_historical_mean(
        close_train=close_train, n_predictions=len(close_anchor_split)
    )
    # Convert constant price prediction to per-row log-return prediction.
    # clip for numerical stability in log and exp guards downstream
    safe_anchor = np.where(close_anchor_split > 1e-9, close_anchor_split, 1e-9)
    y_pred_lr = np.log(np.clip(price_preds, 1e-9, None) / safe_anchor)
    # y_train_lr is accepted for API symmetry and future extension but
    # unused — historical-mean uses the price-space mean by convention.
    _ = y_train_lr
    return y_pred_lr


def _predict_seasonal_naive_lr(
    y_train_lr: np.ndarray,
    y_test_lr: np.ndarray,
    season_length: int = SEASON_LENGTH,
) -> np.ndarray:
    """
    Seasonal naive baseline in log-return space.

    y_pred_lr[i] = log-return observed `season_length` rows before row i.
    For i < season_length, looks back into y_train_lr (tail).

    This is the log-return analogue of naive_baseline.predict_seasonal_naive:
    instead of "price from 7 periods ago," we forecast "the log return
    observed 7 periods ago" — both express the assumption that the same
    weekly pattern repeats.
    """
    n = len(y_test_lr)
    y_pred = np.empty(n, dtype=float)
    for i in range(n):
        if i >= season_length:
            y_pred[i] = y_test_lr[i - season_length]
        else:
            # Fall back to the training tail
            idx = len(y_train_lr) - season_length + i
            y_pred[i] = y_train_lr[idx] if 0 <= idx < len(y_train_lr) else 0.0
    return y_pred


# =============================================================================
# Input loading
# =============================================================================

def _load_split_predictions_csv(
    variant_dir: str,
    interval: str,
    horizon: int,
    split: str,
) -> Optional[pd.DataFrame]:
    """
    Load a split's predictions CSV for one horizon.

    Returns None if the file is missing (the validation split is empty when
    there is no post-2026 data).
    """
    path = os.path.join(
        variant_dir, interval, f"horizon_{horizon}",
        f"{split}_predictions_{interval}_h{horizon}.csv",
    )
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise IOError(f"Failed to read {path}: {exc}") from exc
    required = {"Close_Anchor", "Actual_LogReturn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def _training_series(
    variant_dir: str, interval: str, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (close_train, y_train_lr) for the given horizon.

    Used as inputs for historical_mean (close_train) and seasonal_naive_7
    fallback (y_train_lr tail).
    """
    train_df = _load_split_predictions_csv(variant_dir, interval, horizon, "training")
    if train_df is None:
        raise FileNotFoundError(
            f"training_predictions_{interval}_h{horizon}.csv not found under "
            f"{variant_dir}/{interval}/horizon_{horizon}/. "
            "Run horizon_forecast.py first to generate predictions."
        )
    close_train = train_df["Close_Anchor"].to_numpy(dtype=float)
    y_train_lr = train_df["Actual_LogReturn"].to_numpy(dtype=float)
    return close_train, y_train_lr


# =============================================================================
# Public API
# =============================================================================

def evaluate_all_naive_baselines(
    variant_dir: str,
    interval: str = "Daily",
    horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    splits: Tuple[str, ...] = ("testing", "validation"),
) -> Dict[str, pd.DataFrame]:
    """
    Compute naive baselines for every (split, horizon) pair found under variant_dir.

    Parameters
    ----------
    variant_dir : str
        Root output directory of one horizon_forecast variant, e.g.
        `prediction/output_horizons` for the full model.
    interval : str, default 'Daily'
        Interval subdirectory name.
    horizons : tuple of int
        Horizons to evaluate. Must match horizons the parametric pipeline
        produced predictions for.
    splits : tuple of str
        Splits to evaluate. Typically ('testing', 'validation').

    Returns
    -------
    dict mapping split -> DataFrame with columns
        ['Horizon', 'Model', 'Optimization', 'MAPE', 'sMAPE', 'RMSE',
         'Directional_Accuracy', 'R2_Price', 'R2_LogReturn'].
        Empty DataFrame for splits where no per-horizon CSVs were found.
    """
    if not os.path.isdir(variant_dir):
        raise FileNotFoundError(f"variant_dir not found: {variant_dir}")

    results: Dict[str, List[dict]] = {split: [] for split in splits}

    for horizon in horizons:
        # Training series is shared across splits for the same horizon
        try:
            close_train, y_train_lr = _training_series(variant_dir, interval, horizon)
        except FileNotFoundError as exc:
            print(f"  [skip] horizon {horizon}: {exc}")
            continue

        for split in splits:
            split_df = _load_split_predictions_csv(variant_dir, interval, horizon, split)
            if split_df is None or split_df.empty:
                continue

            close_anchor = split_df["Close_Anchor"].to_numpy(dtype=float)
            y_true_lr = split_df["Actual_LogReturn"].to_numpy(dtype=float)

            if len(y_true_lr) < 2:
                # calculate_metrics requires at least 2 samples
                continue

            # --- naive_rw -------------------------------------------------
            y_pred_rw = _predict_naive_rw_lr(y_true_lr)
            m_rw = calculate_metrics(y_true_lr, y_pred_rw, close_anchor)
            results[split].append(
                {"Horizon": horizon, "Model": "naive_rw", "Optimization": "NAIVE", **m_rw}
            )

            # --- historical_mean -----------------------------------------
            y_pred_hm = _predict_historical_mean_lr(
                y_train_lr=y_train_lr,
                close_train=close_train,
                close_anchor_split=close_anchor,
            )
            m_hm = calculate_metrics(y_true_lr, y_pred_hm, close_anchor)
            results[split].append(
                {"Horizon": horizon, "Model": "historical_mean",
                 "Optimization": "NAIVE", **m_hm}
            )

            # --- seasonal_naive_7 ----------------------------------------
            y_pred_sn = _predict_seasonal_naive_lr(
                y_train_lr=y_train_lr, y_test_lr=y_true_lr,
                season_length=SEASON_LENGTH,
            )
            m_sn = calculate_metrics(y_true_lr, y_pred_sn, close_anchor)
            results[split].append(
                {"Horizon": horizon, "Model": "seasonal_naive_7",
                 "Optimization": "NAIVE", **m_sn}
            )

    # Normalize into schema DataFrames
    columns = ["Horizon", "Model", "Optimization", "MAPE", "sMAPE", "RMSE",
               "Directional_Accuracy", "R2_Price", "R2_LogReturn"]
    out: Dict[str, pd.DataFrame] = {}
    for split, rows in results.items():
        if not rows:
            out[split] = pd.DataFrame(columns=columns)
            continue
        df = pd.DataFrame(rows)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        out[split] = df[columns].sort_values(["Horizon", "Model"]).reset_index(drop=True)
    return out


# =============================================================================
# Self-test
# =============================================================================

def _self_test() -> None:
    """Smoke-test against the real output_horizons/Daily/ artifacts."""
    project_root = os.path.dirname(_PREDICTION_DIR)
    variant_dir = os.path.join(project_root, "prediction", "output_horizons")
    print(f"Self-test against variant_dir = {variant_dir}")
    if not os.path.isdir(variant_dir):
        print("  SKIP: variant_dir not present")
        return

    out = evaluate_all_naive_baselines(
        variant_dir=variant_dir, interval="Daily",
        horizons=(1, 2, 3, 4, 5, 6, 7),
        splits=("testing", "validation"),
    )
    for split, df in out.items():
        print(f"\n[{split}] {len(df)} rows")
        if not df.empty:
            print(df.to_string(index=False))


if __name__ == "__main__":
    _self_test()
