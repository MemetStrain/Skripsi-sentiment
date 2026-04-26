"""
Bayesian Optimization hyperparameter tuner for CPO prediction models.
======================================================================

Uses Gaussian Process surrogate (scikit-optimize) with Expected Improvement (EI)
acquisition function to find optimal hyperparameters via time-series CV.

Supports the same 4 model types as CSATimeSeriesOptimizer:
  xgboost, random_forest, arimax, sarimax

Drop-in replacement: BayesianTimeSeriesOptimizer.optimize() returns a BayesResult
that has the same interface as CSAResult (.best_params, .best_score,
.convergence_history, .total_iterations, .convergence_iteration).

Dependency: pip install scikit-optimize
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
except ImportError:
    raise ImportError(
        "scikit-optimize is required for Bayesian optimization.\n"
        "Install it with:  pip install scikit-optimize"
    )

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Result container (mirrors CSAResult interface)
# ---------------------------------------------------------------------------

@dataclass
class BayesResult:
    """
    Result object from Bayesian optimization.
    Mirrors the interface of CSAResult for drop-in compatibility.
    """
    best_params: Dict
    best_score: float
    convergence_history: List[float] = field(default_factory=list)
    total_iterations: int = 0
    convergence_iteration: int = 0


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class BayesianTimeSeriesOptimizer:
    """
    Gaussian Process Bayesian Optimization for ML model hyperparameters.

    Uses TimeSeriesSplit CV with RMSE as the objective (lower is better).
    The GP surrogate model is updated after each evaluation, guiding the
    search toward promising regions more efficiently than random or grid search.

    Parameters
    ----------
    model_type : str
        One of 'xgboost', 'random_forest', 'arimax', 'sarimax'.
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Training targets.
    config : IntervalConfig
        Interval configuration (used for seasonal_period in SARIMAX).
    cv_folds : int, default=3
        Number of TimeSeriesSplit folds.
    n_calls : int, default=50
        Total number of objective function evaluations.
        For ARIMAX/SARIMAX this is capped at 30.
    n_initial_points : int, default=10
        Random evaluations before the GP starts guiding the search.
    random_state : int, default=42
        Reproducibility seed.
    """

    # Parameter search spaces — identical bounds to CSATimeSeriesOptimizer
    PARAM_SPACES = {
        'xgboost': [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 15,  name='max_depth'),
            Real(1e-3, 0.3, prior='log-uniform', name='learning_rate'),
            Real(0.6, 1.0,  name='subsample'),
            Real(0.6, 1.0,  name='colsample_bytree'),
            Integer(1, 10,  name='min_child_weight'),
        ],
        'random_forest': [
            Integer(50, 500, name='n_estimators'),
            Integer(5, 30,   name='max_depth'),
            Integer(2, 20,   name='min_samples_split'),
            Integer(1, 10,   name='min_samples_leaf'),
            Real(0.3, 0.9,   name='max_features'),
        ],
        'arimax': [
            Integer(0, 5, name='p'),
            Integer(0, 2, name='d'),
            Integer(0, 5, name='q'),
        ],
        'sarimax': [
            Integer(0, 3, name='p'),
            Integer(0, 2, name='d'),
            Integer(0, 3, name='q'),
            Integer(0, 2, name='P'),
            Integer(0, 1, name='D'),
            Integer(0, 2, name='Q'),
        ],
    }

    def __init__(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config,
        cv_folds: int = 3,
        n_calls: int = 50,
        n_initial_points: int = 10,
        random_state: int = RANDOM_STATE,
    ):
        if model_type not in self.PARAM_SPACES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Must be one of: {list(self.PARAM_SPACES)}"
            )
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.cv_folds = cv_folds
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state

        # Pre-select top-N exogenous features for time-series models
        if model_type in ('arimax', 'sarimax'):
            self.exog_train, self.exog_indices = self._select_top_exog(
                X_train, y_train, n=min(10, X_train.shape[1])
            )
        else:
            self.exog_train = None
            self.exog_indices = None

    # -------------------------------------------------------------------------
    # Internal helpers (self-contained — no imports from adaptive_prediction)
    # -------------------------------------------------------------------------

    @staticmethod
    def _select_top_exog(
        X: np.ndarray, y: np.ndarray, n: int
    ) -> Tuple[np.ndarray, List[int]]:
        """Select top-N features by absolute Pearson correlation with y."""
        correlations = np.array([
            abs(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 0 else 0.0
            for i in range(X.shape[1])
        ])
        indices = np.argsort(correlations)[-n:].tolist()
        return X[:, indices], indices

    def _create_sklearn_model(self, params: Dict):
        """Instantiate XGBoost or Random Forest with given hyperparameters."""
        if self.model_type == 'xgboost':
            valid = set(XGBRegressor().get_params())
            p = {k: v for k, v in params.items() if k in valid}
            return XGBRegressor(**p, verbosity=0, random_state=self.random_state, n_jobs=1)
        else:
            return RandomForestRegressor(
                **params, random_state=self.random_state, n_jobs=1
            )

    def _fit_statsmodels(self, y_train, exog_train, order, seasonal_order):
        """Fit SARIMAX/ARIMAX; returns fitted result or None on failure."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = SARIMAX(
                    endog=y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
            return result
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Cross-validation objective
    # -------------------------------------------------------------------------

    def _cv_score(self, params: Dict) -> float:
        """Return mean RMSE across TimeSeriesSplit folds (lower is better).

        np.inf is never returned — failed folds use a large finite penalty so
        that the Gaussian Process surrogate (which validates inputs) stays happy.
        """
        _PENALTY = 1e6  # large but finite; signals a bad hyperparameter set
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores: List[float] = []

        if self.model_type in ('xgboost', 'random_forest'):
            model = self._create_sklearn_model(params)
            for tr, va in tscv.split(self.X_train):
                try:
                    model.fit(self.X_train[tr], self.y_train[tr])
                    y_pred = model.predict(self.X_train[va])
                    rmse = float(np.sqrt(mean_squared_error(self.y_train[va], y_pred)))
                    scores.append(rmse if np.isfinite(rmse) else _PENALTY)
                except Exception:
                    scores.append(_PENALTY)

        elif self.model_type == 'arimax':
            order = (int(params['p']), int(params['d']), int(params['q']))
            for tr, va in tscv.split(self.exog_train):
                fitted = self._fit_statsmodels(
                    self.y_train[tr], self.exog_train[tr], order, (0, 0, 0, 0)
                )
                if fitted is None:
                    scores.append(_PENALTY)
                    continue
                try:
                    preds = np.array(
                        fitted.forecast(steps=len(va), exog=self.exog_train[va])
                    )
                    if np.any(~np.isfinite(preds)):
                        scores.append(_PENALTY)
                    else:
                        rmse = float(np.sqrt(mean_squared_error(self.y_train[va], preds)))
                        scores.append(rmse if np.isfinite(rmse) else _PENALTY)
                except Exception:
                    scores.append(_PENALTY)

        elif self.model_type == 'sarimax':
            order = (int(params['p']), int(params['d']), int(params['q']))
            s = self.config.seasonal_period
            seasonal_order = (
                int(params['P']), int(params['D']), int(params['Q']), s
            )
            for tr, va in tscv.split(self.exog_train):
                if len(tr) < s * 2:
                    scores.append(_PENALTY)
                    continue
                fitted = self._fit_statsmodels(
                    self.y_train[tr], self.exog_train[tr], order, seasonal_order
                )
                if fitted is None:
                    scores.append(_PENALTY)
                    continue
                try:
                    preds = np.array(
                        fitted.forecast(steps=len(va), exog=self.exog_train[va])
                    )
                    if np.any(~np.isfinite(preds)):
                        scores.append(_PENALTY)
                    else:
                        rmse = float(np.sqrt(mean_squared_error(self.y_train[va], preds)))
                        scores.append(rmse if np.isfinite(rmse) else _PENALTY)
                except Exception:
                    scores.append(_PENALTY)

        return float(np.mean(scores)) if scores else _PENALTY

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def optimize(self) -> BayesResult:
        """
        Run Bayesian (GP) optimization over the parameter space.

        Returns
        -------
        BayesResult
            Contains best_params, best_score, convergence_history,
            total_iterations, and convergence_iteration.
        """
        space = self.PARAM_SPACES[self.model_type]

        # Reduce budget for slow time-series models
        n_calls = self.n_calls
        n_init = self.n_initial_points
        if self.model_type in ('arimax', 'sarimax'):
            n_calls = min(n_calls, 30)
            n_init = min(n_init, 8)

        convergence: List[float] = []

        def _objective(x: list) -> float:
            params = {dim.name: val for dim, val in zip(space, x)}
            return self._cv_score(params)

        def _callback(res):
            # res.fun is the best score found so far (skopt minimizes)
            convergence.append(float(res.fun))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            skopt_result = gp_minimize(
                func=_objective,
                dimensions=space,
                n_calls=n_calls,
                n_initial_points=n_init,
                acq_func='EI',          # Expected Improvement
                noise='gaussian',
                random_state=self.random_state,
                callback=_callback,
            )

        # Build best_params with Python-native types (skopt returns numpy scalars)
        best_params: Dict = {}
        for dim, val in zip(space, skopt_result.x):
            best_params[dim.name] = int(val) if isinstance(dim, Integer) else float(val)

        # Find the first iteration that achieved the best score
        best_val = float(skopt_result.fun)
        conv_iter = len(convergence) - 1
        for i, v in enumerate(convergence):
            if v <= best_val + 1e-9:
                conv_iter = i
                break

        return BayesResult(
            best_params=best_params,
            best_score=best_val,
            convergence_history=convergence,
            total_iterations=len(convergence),
            convergence_iteration=conv_iter,
        )
