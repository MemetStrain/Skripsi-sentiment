"""
Shared utilities for multi-horizon CPO price forecasting.

Used by all four ablation horizon-forecast files
(horizon_forecast_C{1..4}_*.py). Scope-trimmed to XGBoost-only after the
2026-04-26 thesis-scope-reduction sweep — Random Forest, ARIMAX, SARIMAX
helpers and Bayesian optimisation hooks are no longer present.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
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
MODELS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')

HORIZONS = [1, 2, 3, 4, 5, 6, 7]

BASE_PARAMS = {
    'xgboost': {
        'n_estimators': 2000, 'max_depth': 6, 'learning_rate': 0.001,
        'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1,
        'early_stopping_rounds': 50,
        'random_state': RANDOM_STATE, 'verbose': True,
    },
}

CSA_PARAM_SPACES = {
    'xgboost': [
        ParameterSpec('n_estimators', 50, 1500, 'discrete'),
        ParameterSpec('max_depth', 3, 9, 'discrete'),
        ParameterSpec('learning_rate', 0.01, 0.5, 'continuous'),
        ParameterSpec('subsample', 0.6, 1.0, 'continuous'),
        ParameterSpec('colsample_bytree', 0.6, 1.0, 'continuous'),
        ParameterSpec('min_child_weight', 1, 10, 'discrete'),
    ],
}


# Hard cutoff: data before this date = train+test; from this date onwards = validation
VAL_CUTOFF = pd.Timestamp('2026-01-01')


# Raw same-day OHLCV columns from the CPO file. These should be dropped at load
# time — feeding them to the model alongside Target = log(Close[t+h]/Close[t])
# is a leakage path. The lagged returns / spreads / SMAs in CPO_TECH_COLS
# already encode the safe price information.
CPO_VARS_DROP = {'Open', 'High', 'Low', 'Volume', 'Change_Pct'}

# Curated CPO technical-indicator whitelist (end-of-day, lag-1 or derived).
CPO_TECH_COLS = [
    'Return_t-1', 'Return_t-2', 'Volume_t-1',
    'High_Low_Spread', 'Open_Close_Spread',
    'SMA_3', 'SMA_6', 'EMA_3', 'EMA_6',
    'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Band_Width',
]


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


def prepare_train_test_val(df: pd.DataFrame, feature_cols: List[str],
                           test_ratio: float,
                           val_cutoff: pd.Timestamp = VAL_CUTOFF) -> Dict:
    """Chronological train / test / validation split with RobustScaler.

    - train + test : rows where Date < val_cutoff, split 80/20
    - val          : rows where Date >= val_cutoff
    Scaler is fit on train only (no leakage into test or val).
    """
    pre    = df[df['Date'] < val_cutoff].reset_index(drop=True)
    val_df = df[df['Date'] >= val_cutoff].reset_index(drop=True)

    assert len(pre) > 0, f"No pre-{val_cutoff.year} rows found — check the Date column"

    split_idx = int(len(pre) * (1 - test_ratio))

    X_pre      = pre[feature_cols].values
    y_pre      = pre['Target'].values
    dates_pre  = pre['Date'].values
    close_pre  = pre['Close'].values

    n_feat = len(feature_cols)
    if len(val_df):
        X_val     = val_df[feature_cols].values
        y_val     = val_df['Target'].values
        dates_val = val_df['Date'].values
        close_val = val_df['Close'].values
    else:
        X_val = np.empty((0, n_feat), dtype=float)
        y_val = dates_val = close_val = np.array([])

    X_train, X_test = X_pre[:split_idx], X_pre[split_idx:]
    y_train, y_test = y_pre[:split_idx], y_pre[split_idx:]
    train_dates, test_dates = dates_pre[:split_idx], dates_pre[split_idx:]
    close_train, close_test = close_pre[:split_idx], close_pre[split_idx:]

    scaler        = RobustScaler()
    X_train_s     = scaler.fit_transform(X_train)
    X_test_s      = scaler.transform(X_test)
    X_val_s       = scaler.transform(X_val) if len(X_val) else X_val

    return {
        'X_train': X_train_s,   'X_test': X_test_s,   'X_val': X_val_s,
        'y_train': y_train,     'y_test': y_test,      'y_val': y_val,
        'train_dates': train_dates, 'test_dates': test_dates, 'val_dates': dates_val,
        'scaler': scaler,       'feature_names': feature_cols,
        'close_train': close_train, 'close_test': close_test, 'close_val': close_val,
    }


def prepare_cv_test_split(df: pd.DataFrame, feature_cols: List[str],
                          n_cv_folds: int,
                          val_cutoff: pd.Timestamp = VAL_CUTOFF) -> Dict:
    """Time-series cross-validation split.

    - pre-test rows (Date < val_cutoff): TimeSeriesSplit into n_cv_folds folds
    - test rows    (Date >= val_cutoff): final holdout

    One global RobustScaler is fit on all pre-test data and reused across folds
    and the final model — matches horizon_forecast_configurable._build_data.
    """
    pre  = df[df['Date'] < val_cutoff].reset_index(drop=True)
    test = df[df['Date'] >= val_cutoff].reset_index(drop=True)

    assert len(pre) > 0, f"No pre-{val_cutoff.year} rows found — check the Date column"

    X_pre_raw  = pre[feature_cols].values
    y_pre      = pre['Target'].values
    dates_pre  = pre['Date'].values
    close_pre  = pre['Close'].values

    n_feat = len(feature_cols)
    if len(test):
        X_test_raw  = test[feature_cols].values
        y_test      = test['Target'].values
        dates_test  = test['Date'].values
        close_test  = test['Close'].values
    else:
        X_test_raw = np.empty((0, n_feat))
        y_test = dates_test = close_test = np.array([])

    scaler   = RobustScaler()
    X_pre_s  = scaler.fit_transform(X_pre_raw)
    X_test_s = scaler.transform(X_test_raw) if len(X_test_raw) else X_test_raw

    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    cv_splits = []
    for fold_idx, (train_idx, cv_idx) in enumerate(tscv.split(X_pre_s)):
        cv_splits.append({
            'fold':        fold_idx + 1,
            'X_train':     X_pre_s[train_idx],
            'y_train':     y_pre[train_idx],
            'dates_train': dates_pre[train_idx],
            'close_train': close_pre[train_idx],
            'X_cv':        X_pre_s[cv_idx],
            'y_cv':        y_pre[cv_idx],
            'dates_cv':    dates_pre[cv_idx],
            'close_cv':    close_pre[cv_idx],
        })

    return {
        'X_pre': X_pre_s,   'y_pre': y_pre,
        'dates_pre': dates_pre, 'close_pre': close_pre,
        'X_pre_raw': X_pre_raw,
        'X_test': X_test_s, 'y_test': y_test,
        'dates_test': dates_test, 'close_test': close_test,
        'X_test_raw': X_test_raw,
        'cv_splits': cv_splits,
        'scaler':    scaler,
        'feature_names': feature_cols,
    }


# =============================================================================
# Model helpers
# =============================================================================

def create_sklearn_model(model_type: str, params: Optional[Dict] = None):
    if model_type != 'xgboost':
        raise ValueError(f"Unsupported model_type '{model_type}' — only 'xgboost' is in scope.")
    p = dict(params or BASE_PARAMS[model_type])
    p.pop('random_state', None)
    valid_keys = set(XGBRegressor().get_params().keys())
    filtered = {k: v for k, v in p.items() if k in valid_keys}
    return XGBRegressor(**filtered, verbosity=1, random_state=RANDOM_STATE)


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

    mape = np.mean(np.abs((price_actual - price_predicted)
                          / (np.abs(price_actual) + 1e-8))) * 100
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
                y_true = y_train[val_idx]
                mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
                scores.append(mape)
            except Exception:
                scores.append(np.inf)
        return np.mean(scores)
    return objective


def run_csa(model_type, objective_fn, population_size, max_iterations):
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


# =============================================================================
# Model artifact persistence (local only)
# =============================================================================

def save_model_artifacts(
    model,
    model_type: str,
    scaler,
    feature_cols: List[str],
    params: dict,
    save_dir: str,
) -> None:
    """Save a trained XGBoost model, its scaler, and metadata to *save_dir*."""
    os.makedirs(save_dir, exist_ok=True)

    if model is not None:
        joblib.dump(model, os.path.join(save_dir, 'model.pkl'))

    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    def _serialise(v):
        if callable(v):
            return v.__name__
        if hasattr(v, 'tolist'):
            return v.tolist()
        return v

    meta = {
        'model_type':   model_type,
        'feature_cols': list(feature_cols),
        'params':       {k: _serialise(v) for k, v in params.items()},
        'saved_at':     pd.Timestamp.now().isoformat(),
    }
    with open(os.path.join(save_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)


def load_model_artifacts(load_dir: str) -> Optional[Dict]:
    """Load artifacts previously saved by :func:`save_model_artifacts`."""
    meta_path = os.path.join(load_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as fh:
        meta = json.load(fh)

    model = None
    p = os.path.join(load_dir, 'model.pkl')
    if os.path.exists(p):
        model = joblib.load(p)

    scaler = None
    sp = os.path.join(load_dir, 'scaler.pkl')
    if os.path.exists(sp):
        scaler = joblib.load(sp)

    return {
        'model':        model,
        'scaler':       scaler,
        'model_type':   meta['model_type'],
        'feature_cols': meta['feature_cols'],
        'params':       meta['params'],
        'saved_at':     meta.get('saved_at'),
    }


# =============================================================================
# Feature importance
# =============================================================================

def save_feature_importance(model: XGBRegressor, feature_cols: List[str],
                            out_dir, horizon: int, top_n: int = 20,
                            tag: str = 'BASE') -> None:
    """Save XGBoost feature importance (gain + weight) as CSV and bar chart.

    The model is trained on a numpy array, so XGBoost names features f0, f1, …
    internally — we remap to actual column names before lookup.
    """
    out_dir = Path(out_dir)
    booster = model.get_booster()

    gain_raw   = booster.get_score(importance_type='gain')
    weight_raw = booster.get_score(importance_type='weight')
    gain   = {feature_cols[int(k[1:])]: v for k, v in gain_raw.items()}
    weight = {feature_cols[int(k[1:])]: v for k, v in weight_raw.items()}

    rows = [
        {'Feature': feat,
         'Importance_Gain':   gain.get(feat, 0.0),
         'Importance_Weight': weight.get(feat, 0.0)}
        for feat in feature_cols
    ]
    df_imp = (pd.DataFrame(rows)
              .sort_values('Importance_Gain', ascending=False)
              .reset_index(drop=True))
    df_imp.insert(0, 'Rank', df_imp.index + 1)

    tag_lower = tag.lower()
    df_imp.to_csv(out_dir / f'feature_importance_{tag_lower}_h{horizon}.csv',
                  index=False)

    top = df_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    ax.barh(top['Feature'][::-1], top['Importance_Gain'][::-1], color='#2E86AB')
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'Feature Importance ({tag}) — Horizon {horizon} '
                 f'(top {len(top)} by gain)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f'feature_importance_{tag_lower}_h{horizon}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Feature importance [{tag}] (top {min(top_n, len(df_imp))} by gain):")
    print(f"  {'Rank':>4}  {'Feature':<35}  {'Gain':>10}  {'Weight':>8}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*10}  {'-'*8}")
    for _, row in df_imp.head(top_n).iterrows():
        print(f"  {int(row['Rank']):>4}  {row['Feature']:<35}  "
              f"{row['Importance_Gain']:>10.4f}  {int(row['Importance_Weight']):>8}")
