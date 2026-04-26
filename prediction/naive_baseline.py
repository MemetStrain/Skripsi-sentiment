"""
Naive Baseline Models for CPO Price Prediction Control Experiment.

Provides non-parametric baseline predictors (random walk, historical mean,
seasonal naive) as experimental control for Hypothesis H4:
    "Model gabungan sentimen + HMM + lagged price mengalahkan naive random walk."

Design notes
------------
- No training required; all baselines are deterministic functions of y_t.
- Returns same output shape as parametric models (ndarray of predictions
  aligned with test_dates) so it drops into existing evaluation pipeline
  without special-casing.
- Multi-horizon aware: for horizon h, naive random walk predicts y_{t+h} = y_t.
- Scale-agnostic: no scaler, no transformation; works on raw prices.

Integration
-----------
Use alongside existing `forecast_utils.py` evaluation functions. Append naive
baseline results to the same metric table used for RF/XGBoost/ARIMAX/SARIMAX.

Example
-------
>>> from naive_baseline import predict_random_walk, evaluate_naive
>>> y_hat = predict_random_walk(close_test, horizon=1)
>>> metrics = evaluate_naive(y_test, y_hat, close_test)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Naive Predictors
# =============================================================================

def predict_random_walk(close_test: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Random walk / persistence baseline: y_hat_{t+h} = y_t.

    For horizon h >= 1, the prediction for time t+h is the last observed
    value y_t. This is the standard non-trivial baseline for financial
    time-series forecasting (Makridakis et al., 2018).

    Parameters
    ----------
    close_test : np.ndarray
        Close prices on the test set, shape (n,).
    horizon : int, default 1
        Forecast horizon (steps ahead). Must be >= 1.

    Returns
    -------
    np.ndarray
        Predictions shape (n - horizon,), aligned such that y_hat[i]
        predicts close_test[i + horizon].

    Raises
    ------
    ValueError
        If horizon < 1 or close_test length <= horizon.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(close_test, np.ndarray):
        close_test = np.asarray(close_test, dtype=float)
    if close_test.ndim != 1:
        raise ValueError(f"close_test must be 1D, got shape {close_test.shape}")
    if len(close_test) <= horizon:
        raise ValueError(
            f"close_test length ({len(close_test)}) must exceed horizon ({horizon})"
        )

    # y_hat_{t+h} = y_t  →  shift by h positions
    # prediction for index t+h is value at index t
    y_hat = close_test[:-horizon].copy()
    return y_hat


def predict_historical_mean(
    close_train: np.ndarray,
    n_predictions: int,
) -> np.ndarray:
    """
    Historical mean baseline: y_hat_t = mean(y_train).

    Predicts the in-sample training mean for every test observation. This
    is the "null hypothesis" baseline against which R^2 is computed; a
    model with R^2 < 0 is worse than this baseline.

    Parameters
    ----------
    close_train : np.ndarray
        Training close prices, shape (n_train,).
    n_predictions : int
        Number of predictions to generate (length of test set).

    Returns
    -------
    np.ndarray
        Constant prediction array, shape (n_predictions,).
    """
    if not isinstance(close_train, np.ndarray):
        close_train = np.asarray(close_train, dtype=float)
    if len(close_train) == 0:
        raise ValueError("close_train is empty")
    if n_predictions < 1:
        raise ValueError(f"n_predictions must be >= 1, got {n_predictions}")

    mean_value = float(np.mean(close_train))
    return np.full(n_predictions, mean_value)


def predict_seasonal_naive(
    close_test: np.ndarray,
    season_length: int = 7,
    horizon: int = 1,
) -> np.ndarray:
    """
    Seasonal naive baseline: y_hat_{t+h} = y_{t+h-s}.

    Useful for detecting weekly (s=7) or monthly (s=30) seasonality.
    If no seasonality exists in CPO prices, this baseline should perform
    worse than random walk — useful diagnostic information.

    Parameters
    ----------
    close_test : np.ndarray
        Close prices on the test set (concatenated with last season_length
        observations of train set if needed), shape (n,).
    season_length : int, default 7
        Seasonal period in time steps (7 = weekly, 30 = monthly).
    horizon : int, default 1
        Forecast horizon (steps ahead).

    Returns
    -------
    np.ndarray
        Predictions shape (n - max(horizon, season_length),).
    """
    if season_length < 1:
        raise ValueError(f"season_length must be >= 1, got {season_length}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(close_test, np.ndarray):
        close_test = np.asarray(close_test, dtype=float)

    lookback = season_length  # need s periods before first prediction
    if len(close_test) <= lookback:
        raise ValueError(
            f"close_test length ({len(close_test)}) must exceed "
            f"season_length ({season_length})"
        )

    # y_hat_{t+h} = y_{t+h-s}  →  for index i, predict using i-s+h
    # simplified: shift by season_length
    y_hat = close_test[:-season_length].copy()
    return y_hat


# =============================================================================
# Evaluation
# =============================================================================

def compute_naive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute MAPE, sMAPE, R^2, and Directional Accuracy for a naive prediction.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values (must be same length as y_true).
    y_prev : np.ndarray, optional
        Previous-period actual values for directional accuracy computation:
        DA = 1{sign(y_pred - y_prev) == sign(y_true - y_prev)}. If None,
        DA is set to NaN.

    Returns
    -------
    dict
        Keys: mape, smape, r2, da, n_samples
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("y_true is empty")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAPE — guard against division by zero (common in price data this is fine,
    # but log returns can cross zero)
    nonzero_mask = np.abs(y_true) > 1e-9
    if nonzero_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                           / y_true[nonzero_mask])) * 100.0
        )

    # sMAPE — symmetric variant, stable near zero
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom_safe = np.where(denom > 1e-9, denom, 1e-9)
    smape = float(np.mean(np.abs(y_true - y_pred) / denom_safe) * 100.0)

    # R^2 — fraction of variance explained; negative means worse than mean
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    # Directional Accuracy
    if y_prev is not None and len(y_prev) == len(y_true):
        actual_direction = np.sign(y_true - y_prev)
        pred_direction = np.sign(y_pred - y_prev)
        # exclude zero-change cases from DA computation (ambiguous)
        nonzero_dir = actual_direction != 0
        if nonzero_dir.sum() > 0:
            da = float(
                np.mean(actual_direction[nonzero_dir] == pred_direction[nonzero_dir])
                * 100.0
            )
        else:
            da = float("nan")
    else:
        da = float("nan")

    return {
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "r2": round(r2, 4),
        "da": round(da, 4),
        "n_samples": len(y_true),
    }


# =============================================================================
# Diebold-Mariano Test
# =============================================================================

def diebold_mariano_test(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy between two models.

    Tests H0: E[loss(a) - loss(b)] = 0 against H1: E[loss(a) - loss(b)] != 0.
    A negative DM statistic with low p-value indicates model A has lower loss
    (better accuracy) than model B.

    Reference: Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
    accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.

    Parameters
    ----------
    errors_model_a : np.ndarray
        Forecast errors (y_true - y_pred) from model A.
    errors_model_b : np.ndarray
        Forecast errors from model B; must be same length as model A.
    h : int, default 1
        Forecast horizon; used for Newey-West lag selection.
    loss : {'squared', 'absolute'}, default 'squared'
        Loss function for comparison.

    Returns
    -------
    (dm_statistic, p_value) : tuple of float
        Two-sided p-value from standard normal approximation.
    """
    if len(errors_model_a) != len(errors_model_b):
        raise ValueError("error series must have equal length")
    if loss not in ("squared", "absolute"):
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss}")

    e_a = np.asarray(errors_model_a, dtype=float)
    e_b = np.asarray(errors_model_b, dtype=float)

    # loss differential d_t
    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    else:
        d = np.abs(e_a) - np.abs(e_b)

    n = len(d)
    d_mean = float(np.mean(d))

    # Newey-West long-run variance with lag = h - 1 (no correction for h=1)
    # gamma_0 = variance; gamma_k = autocovariance at lag k
    gamma_0 = float(np.var(d, ddof=0))
    lrv = gamma_0
    for k in range(1, h):
        # autocovariance at lag k
        cov_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        lrv += 2.0 * cov_k  # symmetric

    if lrv <= 0:
        # happens when d is constant or negative variance from small sample
        return float("nan"), float("nan")

    dm_stat = d_mean / np.sqrt(lrv / n)

    # two-sided p-value from standard normal
    from scipy import stats  # local import to avoid hard dependency at module load
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)


# =============================================================================
# Orchestration — integrate with existing evaluation pipeline
# =============================================================================

def run_naive_baselines(
    close_train: np.ndarray,
    close_test: np.ndarray,
    horizons: List[int] = (1, 2, 3, 5, 7),
) -> pd.DataFrame:
    """
    Run all naive baselines across multiple horizons; return metric DataFrame.

    Output schema matches existing validation_summary.csv columns where
    applicable (model, horizon, MAPE, R2, DirAcc) plus sMAPE.

    Parameters
    ----------
    close_train : np.ndarray
        Training close prices (for historical_mean baseline).
    close_test : np.ndarray
        Test close prices.
    horizons : list of int
        Forecast horizons to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: model, horizon, mape, smape, r2, da, n_samples
    """
    if len(close_test) < max(horizons) + 1:
        raise ValueError(
            f"close_test length ({len(close_test)}) insufficient for "
            f"max horizon ({max(horizons)})"
        )

    rows = []

    for h in horizons:
        # Random walk: y_hat_{t+h} = y_t
        y_prev = close_test[:-h]                # aligned "previous" value
        y_true = close_test[h:]                 # actual t+h values
        y_hat = predict_random_walk(close_test, horizon=h)
        metrics = compute_naive_metrics(y_true, y_hat, y_prev=y_prev)
        rows.append({"model": "naive_rw", "horizon": h, **metrics})

        # Historical mean: constant prediction
        y_hat_hm = predict_historical_mean(close_train, n_predictions=len(y_true))
        metrics_hm = compute_naive_metrics(y_true, y_hat_hm, y_prev=y_prev)
        rows.append({"model": "historical_mean", "horizon": h, **metrics_hm})

        # Seasonal naive (weekly)
        if len(close_test) > 7 + h:
            y_true_sn = close_test[7:]          # skip first season_length
            y_prev_sn = close_test[7 - 1:-1]    # aligned previous for DA
            y_hat_sn = predict_seasonal_naive(close_test, season_length=7, horizon=h)
            # align lengths
            min_len = min(len(y_true_sn), len(y_hat_sn), len(y_prev_sn))
            metrics_sn = compute_naive_metrics(
                y_true_sn[:min_len], y_hat_sn[:min_len], y_prev=y_prev_sn[:min_len]
            )
            rows.append({"model": "seasonal_naive_7", "horizon": h, **metrics_sn})

    return pd.DataFrame(rows)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    # Synthetic random walk with upward drift + noise
    np.random.seed(42)
    n = 500
    drift = 0.05
    returns = np.random.normal(drift, 1.0, n)
    prices = 3500 + np.cumsum(returns)  # CPO-like price level

    train_size = int(0.8 * n)
    close_train = prices[:train_size]
    close_test = prices[train_size:]

    print("=" * 70)
    print("Naive Baseline Self-Test")
    print("=" * 70)
    print(f"Train size: {len(close_train)}, Test size: {len(close_test)}")
    print(f"Train mean: {np.mean(close_train):.2f}, "
          f"Test mean: {np.mean(close_test):.2f}")
    print()

    df = run_naive_baselines(close_train, close_test, horizons=[1, 2, 5])
    print(df.to_string(index=False))
    print()

    # Example DM test: random walk vs historical mean at h=1
    y_prev = close_test[:-1]
    y_true = close_test[1:]

    err_rw = y_true - predict_random_walk(close_test, horizon=1)
    err_hm = y_true - predict_historical_mean(close_train, len(y_true))

    dm_stat, dm_p = diebold_mariano_test(err_rw, err_hm, h=1, loss="squared")
    print(f"Diebold-Mariano (RW vs Historical Mean, h=1):")
    print(f"  DM stat = {dm_stat:.4f}, p-value = {dm_p:.4f}")
    print(f"  {'Significant' if dm_p < 0.05 else 'Not significant'} at alpha=0.05")
