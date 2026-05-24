"""
Naive Baseline Models for CPO Price Prediction Control Experiment.

Provides the two baseline pieces still used by the H4 evaluation pipeline:

- :func:`predict_historical_mean` - constant-price prediction used by
  ``baselines/naive_evaluator.py`` when converting to log-return space.
- :func:`diebold_mariano_test` - DM test with the Harvey, Leybourne &
  Newbold (1997) small-sample correction; used by
  ``baselines/dm_comparison.py`` and ``baselines/dm_ablation_pairwise.py``.
- :func:`pesaran_timmermann_test` - Pesaran-Timmermann (1992) directional
  / market-timing test; used by ``baselines/pt_directional.py`` to
  evaluate sign-prediction skill of each ablation configuration.

Older helpers (``predict_random_walk``, ``predict_seasonal_naive``,
``compute_naive_metrics``, ``run_naive_baselines``) were removed in the
FASE 1 cleanup - their logic now lives in
``prediction/baselines/naive_evaluator.py`` (which expresses the same
baselines directly in log-return space, the actual training target).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# =============================================================================
# Naive Predictors
# =============================================================================

def predict_historical_mean(
    close_train: np.ndarray,
    n_predictions: int,
) -> np.ndarray:
    """Historical mean baseline: y_hat = mean(close_train), broadcast.

    Parameters
    ----------
    close_train : np.ndarray
        Training close prices.
    n_predictions : int
        Number of predictions to generate (length of test set).
    """
    if not isinstance(close_train, np.ndarray):
        close_train = np.asarray(close_train, dtype=float)
    if len(close_train) == 0:
        raise ValueError("close_train is empty")
    if n_predictions < 1:
        raise ValueError(f"n_predictions must be >= 1, got {n_predictions}")

    mean_value = float(np.mean(close_train))
    return np.full(n_predictions, mean_value)


# =============================================================================
# Diebold-Mariano Test (with Harvey-Leybourne-Newbold 1997 correction)
# =============================================================================

def diebold_mariano_test(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> Tuple[float, float]:
    """DM test with Harvey-Leybourne-Newbold (1997) small-sample correction.

    Returns ``(HLN-corrected DM*, two-sided p-value from t_{n-1})``.

    A negative DM* with a low p-value indicates model A has lower loss
    (better accuracy) than model B. The HLN multiplier shrinks the raw
    DM statistic and the test uses a t_{n-1} reference distribution
    rather than the standard normal, both of which make the test
    noticeably more conservative for small n and h > 1.

    References
    ----------
    - Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
      accuracy. *Journal of Business & Economic Statistics*, 13(3),
      253-263.
    - Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the
      equality of prediction mean squared errors. *International
      Journal of Forecasting*, 13(2), 281-291.
    """
    e_a = np.asarray(errors_model_a, dtype=float)
    e_b = np.asarray(errors_model_b, dtype=float)
    if len(e_a) != len(e_b):
        raise ValueError("error series must have equal length")
    if loss not in ("squared", "absolute"):
        raise ValueError(f"loss must be 'squared'|'absolute', got {loss}")

    d = e_a ** 2 - e_b ** 2 if loss == "squared" else np.abs(e_a) - np.abs(e_b)
    n = len(d)
    d_mean = float(np.mean(d))

    # Rectangular-kernel long-run variance, truncated at lag h-1 (DM 1995)
    lrv = float(np.var(d, ddof=0))
    for k in range(1, h):
        cov_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        lrv += 2.0 * cov_k
    if lrv <= 0:
        return float("nan"), float("nan")

    dm = d_mean / np.sqrt(lrv / n)
    # HLN small-sample correction factor.
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * hln

    from scipy import stats
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_star), df=n - 1))
    return float(dm_star), float(p_value)


# =============================================================================
# Pesaran-Timmermann (1992) directional / market-timing test
# =============================================================================

def pesaran_timmermann_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drop_zero_actual: bool = True,
) -> Tuple[float, float, int]:
    """Pesaran-Timmermann (1992) directional / market-timing test.

    H0: arah prediksi independen dari arah aktual (tidak ada directional skill).
    H1 (one-sided): hit rate melebihi rate yang diharapkan di bawah independensi.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Deret log-return aktual dan prediksi (ruang yang sama dgn DM test).
    drop_zero_actual : bool
        Buang baris di mana perubahan aktual == 0 (tidak ada arah benar),
        meniru exclusion Directional_Accuracy di calculate_metrics.

    Returns
    -------
    (PT_stat, p_value_one_sided, n_used) ; (nan, nan, n) bila degenerate.

    References
    ----------
    - Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test
      of predictive performance. *Journal of Business & Economic
      Statistics*, 10(4), 461-465.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if drop_zero_actual:
        mask &= (y_true != 0.0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = int(y_true.size)
    if n < 2:
        return float("nan"), float("nan"), n

    z = (y_true > 0).astype(float)   # indikator aktual naik
    x = (y_pred > 0).astype(float)   # indikator prediksi naik

    p_hat = float(np.mean(z == x))   # hit rate
    py = float(np.mean(z))           # P(aktual naik)
    px = float(np.mean(x))           # P(prediksi naik)
    p_star = py * px + (1 - py) * (1 - px)   # hit rate di bawah independensi

    var_phat = p_star * (1 - p_star) / n
    var_pstar = (
        ((2 * py - 1) ** 2) * px * (1 - px) / n
        + ((2 * px - 1) ** 2) * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / (n ** 2)
    )
    denom_var = var_phat - var_pstar
    if denom_var <= 0 or np.isnan(denom_var):
        return float("nan"), float("nan"), n

    pt_stat = (p_hat - p_star) / np.sqrt(denom_var)  # ~ N(0,1) asimtotik
    from scipy.stats import norm
    p_value = float(1.0 - norm.cdf(pt_stat))         # one-sided (skill = hit > chance)
    return float(pt_stat), p_value, n


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    n = 200

    # Two identical error series -> DM ~ 0, p ~ 1
    e_same = np.random.normal(0, 1, n)
    dm0, p0 = diebold_mariano_test(e_same, e_same, h=1, loss="squared")
    print(f"identical errors -> DM={dm0:.4f}  p={p0:.4f}  (expect ~0, ~1)")

    # Model A much better than B -> DM << 0, p small
    e_better = np.random.normal(0, 0.5, n)
    e_worse = np.random.normal(0, 2.0, n)
    dm1, p1 = diebold_mariano_test(e_better, e_worse, h=1, loss="squared")
    print(f"A < B in loss   -> DM={dm1:.4f}  p={p1:.4f}  (expect DM<<0)")

    train = np.linspace(3500, 4000, 100)
    hm = predict_historical_mean(train, n_predictions=10)
    assert np.allclose(hm, np.mean(train)), "historical mean broken"
    print(f"historical_mean ok: constant {hm[0]:.2f} for 10 predictions")
