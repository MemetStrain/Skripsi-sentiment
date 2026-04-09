"""
Shared utilities for multi-horizon CPO price forecasting.

Contains constants and functions that are identical across all four
horizon forecast files (horizon_forecast.py, horizon_forecast_cpo_hmm.py,
horizon_forecast_cpo_only.py, horizon_forecast_cpo_sentiment.py).

Phase 1: original function signatures extracted verbatim.
Phase 2 modifications (log return target, price-space metrics) are applied
to this file after the smoke test gate passes.
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from xgboost import XGBRegressor

# crow_search_optimizer lives in the prediction/ directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crow_search_optimizer import CrowSearchOptimizer, ParameterSpec, CSAResult  # noqa: F401

# =============================================================================
# Shared constants
# =============================================================================

RANDOM_STATE = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HORIZONS = {
    'Daily': [1, 2, 3, 4, 5, 6, 7],
    'Weekly': [1, 2, 3, 4],
    'Monthly': [1, 2, 3, 4, 5, 6],
}

BASE_PARAMS = {
    'xgboost': {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1,
        'random_state': RANDOM_STATE,
    },
    'random_forest': {
        'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
        'min_samples_leaf': 2, 'max_features': 0.7, 'random_state': RANDOM_STATE,
    },
    'arimax': {'order': (2, 1, 2)},
    'sarimax': {'order': (1, 1, 1), 'seasonal_order_pdq': (1, 0, 1)},
}

CSA_PARAM_SPACES = {
    'xgboost': [
        ParameterSpec('n_estimators', 50, 500, 'discrete'),
        ParameterSpec('max_depth', 3, 15, 'discrete'),
        ParameterSpec('learning_rate', 0.001, 0.3, 'continuous'),
        ParameterSpec('subsample', 0.6, 1.0, 'continuous'),
        ParameterSpec('colsample_bytree', 0.6, 1.0, 'continuous'),
        ParameterSpec('min_child_weight', 1, 10, 'discrete'),
    ],
    'random_forest': [
        ParameterSpec('n_estimators', 50, 500, 'discrete'),
        ParameterSpec('max_depth', 5, 30, 'discrete'),
        ParameterSpec('min_samples_split', 2, 20, 'discrete'),
        ParameterSpec('min_samples_leaf', 1, 10, 'discrete'),
        ParameterSpec('max_features', 0.3, 0.9, 'continuous'),
    ],
    'arimax': [
        ParameterSpec('p', 0, 5, 'discrete'),
        ParameterSpec('d', 0, 2, 'discrete'),
        ParameterSpec('q', 0, 5, 'discrete'),
    ],
    'sarimax': [
        ParameterSpec('p', 0, 3, 'discrete'),
        ParameterSpec('d', 0, 2, 'discrete'),
        ParameterSpec('q', 0, 3, 'discrete'),
        ParameterSpec('P', 0, 2, 'discrete'),
        ParameterSpec('D', 0, 1, 'discrete'),
        ParameterSpec('Q', 0, 2, 'discrete'),
    ],
}


# =============================================================================
# Train/test split
# =============================================================================

def prepare_train_test(df: pd.DataFrame, feature_cols: List[str], test_ratio: float) -> Dict:
    """Chronological train/test split with RobustScaler."""
    assert df['Close'].isna().sum() == 0, "Close has NaN — alignment broken"
    assert not df.isnull().values.any(), "NaN rows remain — check dropna()"

    split_idx = int(len(df) * (1 - test_ratio))

    X = df[feature_cols].values
    y = df['Target'].values
    dates = df['Date'].values
    close_prices = df['Close'].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]
    close_train, close_test = close_prices[:split_idx], close_prices[split_idx:]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'train_dates': train_dates, 'test_dates': test_dates,
        'scaler': scaler, 'feature_names': feature_cols,
        'close_train': close_train, 'close_test': close_test,
    }


# =============================================================================
# Model helpers
# =============================================================================

def create_sklearn_model(model_type: str, params: Optional[Dict] = None):
    p = dict(params or BASE_PARAMS[model_type])
    p.pop('random_state', None)
    if model_type == 'xgboost':
        valid_keys = set(XGBRegressor().get_params().keys())
        filtered = {k: v for k, v in p.items() if k in valid_keys}
        return XGBRegressor(**filtered, verbosity=0, random_state=RANDOM_STATE)
    elif model_type == 'random_forest':
        return RandomForestRegressor(**p, random_state=RANDOM_STATE)


def select_top_exog(X, y, n=10):
    correlations = np.array([abs(np.corrcoef(X[:, i], y)[0, 1])
                             if np.std(X[:, i]) > 0 else 0
                             for i in range(X.shape[1])])
    top_indices = np.argsort(correlations)[-n:]
    return X[:, top_indices], top_indices.tolist()


def train_statsmodels(model_type, y_train, exog_train, order, seasonal_order):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SM_SARIMAX(
                endog=y_train, exog=exog_train, order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=200)
    except Exception:
        return None


def predict_statsmodels(fitted, exog_test):
    try:
        forecast = fitted.forecast(steps=len(exog_test), exog=exog_test)
        return np.array(forecast)
    except Exception:
        return np.full(len(exog_test), np.nan)


def calculate_metrics(y_true_lr, y_pred_lr, close_anchor):
    """
    Compute all metrics in original price space by inverting log return predictions.

    Parameters
    ----------
    y_true_lr    : np.ndarray — actual h-step log returns (test targets)
    y_pred_lr    : np.ndarray — predicted h-step log returns
    close_anchor : np.ndarray — Close price at each prediction row (Close_t)

    Inverse transform:
        price_actual[i]    = close_anchor[i] * exp(y_true_lr[i])
        price_predicted[i] = close_anchor[i] * exp(y_pred_lr[i])
    """
    mask = ~np.isnan(y_pred_lr)
    if mask.sum() < 2:
        return {
            'MAPE': np.inf, 'sMAPE': np.inf,
            'RMSE': np.inf,
            'Directional_Accuracy': 0.0,
            'R2_Price': -np.inf, 'R2_LogReturn': -np.inf,
        }

    lr_true    = y_true_lr[mask]
    lr_pred    = y_pred_lr[mask]
    lr_clipped = np.clip(lr_pred, -10, 10)          # guard exp() overflow on predictions
    anchor     = close_anchor[mask]

    # Clip lr_true for metric computation only — real data is unlikely to overflow
    # but guard defensively. CSV Actual_Price uses raw unclipped values for honest reporting.
    lr_true_clipped = np.clip(lr_true, -10, 10)
    price_actual    = anchor * np.exp(lr_true_clipped)
    price_predicted = anchor * np.exp(lr_clipped)

    # MAPE in price space
    mape = np.mean(np.abs((price_actual - price_predicted)
                          / (np.abs(price_actual) + 1e-8))) * 100
    # sMAPE in price space
    smape = np.mean(200 * np.abs(price_actual - price_predicted)
                    / (np.abs(price_actual) + np.abs(price_predicted) + 1e-8))

    rmse     = np.sqrt(mean_squared_error(price_actual, price_predicted))
    r2_price = r2_score(price_actual, price_predicted)
    r2_lr    = r2_score(lr_true, lr_pred)

    # Directional accuracy uses unclipped lr_pred — intentional.
    # Clipping only guards exp() stability and should not affect sign detection.
    dir_acc = np.mean((lr_true > 0) == (lr_pred > 0)) * 100 if len(lr_true) > 1 else 0.0

    return {
        'MAPE':                 round(mape,     4),
        'sMAPE':                round(smape,    4),
        'RMSE':                 round(rmse,     4),
        'Directional_Accuracy': round(dir_acc,  4),
        'R2_Price':             round(r2_price, 4),
        'R2_LogReturn':         round(r2_lr,    4),
    }


# =============================================================================
# CSA Optimization
# =============================================================================

def csa_objective_sklearn(model_type, X_train, y_train, cv_folds):
    def objective(params):
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        model = create_sklearn_model(model_type, params)
        for train_idx, val_idx in tscv.split(X_train):
            try:
                model.fit(X_train[train_idx], y_train[train_idx])
                y_pred = model.predict(X_train[val_idx])
                scores.append(np.sqrt(mean_squared_error(y_train[val_idx], y_pred)))
            except Exception:
                scores.append(np.inf)
        return np.mean(scores)
    return objective


def csa_objective_arimax(y_train, exog_train, cv_folds):
    def objective(params):
        order = (int(params['p']), int(params['d']), int(params['q']))
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(exog_train):
            fitted = train_statsmodels('arimax', y_train[train_idx],
                                       exog_train[train_idx], order, (0, 0, 0, 0))
            if fitted is None:
                scores.append(np.inf)
                continue
            preds = predict_statsmodels(fitted, exog_train[val_idx])
            if np.any(np.isnan(preds)):
                scores.append(np.inf)
            else:
                scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))
        return np.mean(scores)
    return objective


def csa_objective_sarimax(y_train, exog_train, seasonal_period, cv_folds):
    def objective(params):
        order = (int(params['p']), int(params['d']), int(params['q']))
        seasonal_order = (int(params['P']), int(params['D']),
                          int(params['Q']), seasonal_period)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(exog_train):
            if len(train_idx) < seasonal_period * 2:
                scores.append(np.inf)
                continue
            fitted = train_statsmodels('sarimax', y_train[train_idx],
                                       exog_train[train_idx], order, seasonal_order)
            if fitted is None:
                scores.append(np.inf)
                continue
            preds = predict_statsmodels(fitted, exog_train[val_idx])
            if np.any(np.isnan(preds)):
                scores.append(np.inf)
            else:
                scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))
        return np.mean(scores)
    return objective


def run_csa(model_type, objective_fn, population_size, max_iterations):
    if model_type in ('arimax', 'sarimax'):
        max_iterations = min(max_iterations, 30)
        population_size = min(population_size, 15)

    optimizer = CrowSearchOptimizer(
        objective_function=objective_fn,
        parameter_specs=CSA_PARAM_SPACES[model_type],
        population_size=population_size,
        max_iterations=max_iterations,
        awareness_probability=0.1,
        flight_length=2.0,
        early_stopping_patience=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    return optimizer.optimize()
