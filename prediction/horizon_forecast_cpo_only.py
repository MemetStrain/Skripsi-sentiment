"""
Multi-Horizon CPO Price Forecasting with CSA and Bayesian Optimization (CPO Only)
==================================================================================

Uses ONLY CPO price variables (cpo_variables_{interval}.csv).
No sentiment or HMM data.

Forecasts CPO prices at daily horizons 1–7.
Each horizon is preprocessed independently to prevent data leakage.
Uses 4 model types (XGBoost, Random Forest, ARIMAX, SARIMAX), each as
base, CSA-optimized, and Bayesian-optimized variants.

Usage:
    python horizon_forecast_cpo_only.py --interval daily
    python horizon_forecast_cpo_only.py --interval daily --optimizer bayesian
    python horizon_forecast_cpo_only.py --interval daily --optimizer both
"""

import os
import sys
import json
import time
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.forecast_utils import (
    PROJECT_ROOT, HORIZONS, BASE_PARAMS, MODELS_DIR,
    prepare_train_test_val, create_sklearn_model, select_top_exog,
    train_statsmodels, predict_statsmodels, calculate_metrics,
    csa_objective_sklearn, csa_objective_arimax, csa_objective_sarimax, run_csa,
    save_model_artifacts,
)
from bayesian_optimizer import BayesianTimeSeriesOptimizer


SCRIPT_TAG = 'cpo_only'


class _Cfg:
    """Minimal config shim for BayesianTimeSeriesOptimizer."""
    def __init__(self, seasonal_period: int, name: str = 'Daily'):
        self.seasonal_period = seasonal_period
        self.name = name


warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Interval configurations (CPO only)
INTERVAL_CONFIGS = {
    'Daily': {
        'cpo_file': os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Daily.csv'),
        'seasonal_period': 5,
        'base_lag_periods': [1, 2, 3, 5, 10, 20],
        'min_samples': 100,
        'test_ratio': 0.2,
    },
}


# =============================================================================
# Data Loading (CPO Only)
# =============================================================================

def load_and_merge_data(interval: str) -> pd.DataFrame:
    """Load CPO data only."""
    cfg = INTERVAL_CONFIGS[interval]

    print(f"  Loading CPO data...")
    cpo = pd.read_csv(cfg['cpo_file'])
    cpo['Date'] = pd.to_datetime(cpo['Date'])

    cpo = cpo.sort_values('Date').reset_index(drop=True)

    print(f"  Data: {len(cpo)} rows, {cpo['Date'].min()} to {cpo['Date'].max()}")

    if len(cpo) < cfg['min_samples']:
        raise ValueError(f"Only {len(cpo)} rows, minimum required: {cfg['min_samples']}")

    return cpo


# =============================================================================
# Feature Engineering (horizon-aware to prevent leakage)
# =============================================================================

def engineer_features_for_horizon(df: pd.DataFrame, interval: str, horizon: int
                                  ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features for a specific forecast horizon using CPO data only.

    To prevent data leakage / look-ahead bias:
    - Lag features use lags >= horizon (so we never peek into the forecast window)
    - Target is Close shifted by -horizon
    """
    cfg = INTERVAL_CONFIGS[interval]
    df = df.copy()

    # Temporal features (safe - derived from date, not target)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)

    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
    df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)

    # Lag features - only use lags >= horizon to prevent look-ahead bias
    safe_lags = [lag for lag in cfg['base_lag_periods'] if lag >= horizon]
    if not safe_lags:
        safe_lags = [horizon]

    lag_cols = ['Close']
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in safe_lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Target: h-step cumulative log return (stationary; inverse-transform for price-space errors)
    df['Target'] = np.log(df['Close'].shift(-horizon) / df['Close'])

    # Drop rows with NaN (from lags and target shift)
    df = df.dropna().reset_index(drop=True)

    # Feature columns
    exclude = ['Date', 'Target']
    feature_cols = [c for c in df.columns
                    if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    return df, feature_cols


# =============================================================================
# Single Horizon Pipeline
# =============================================================================

def _save_split_outputs(split_name: str, dates, y_true, close_anchor,
                        predictions: Dict, results: Dict,
                        horizon_dir: str, interval: str, horizon: int):
    pred_df = pd.DataFrame({
        'Date': dates, 'Close_Anchor': close_anchor,
        'Actual_LogReturn': y_true,
        'Actual_Price': close_anchor * np.exp(np.clip(y_true, -10, 10)),
    })
    for name, preds in predictions.items():
        pred_df[f'{name}_LogReturn'] = preds
        safe = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
        pred_df[f'{name}_Price'] = close_anchor * np.exp(safe)
    pred_df.to_csv(
        os.path.join(horizon_dir, f'{split_name}_predictions_{interval}_h{horizon}.csv'),
        index=False)
    rows = [{'Model': k.rsplit('_', 1)[0], 'Optimization': k.rsplit('_', 1)[1].upper(), **v}
            for k, v in results.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(horizon_dir, f'{split_name}_results_{interval}_h{horizon}.csv'),
        index=False)


def run_single_horizon(interval: str, horizon: int, merged_df: pd.DataFrame,
                       output_dir: str, csa_config: Dict,
                       bayes_config: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full prediction pipeline for one interval+horizon combination."""
    cfg = INTERVAL_CONFIGS[interval]
    model_types = ['xgboost', 'random_forest', 'arimax', 'sarimax']

    print(f"\n{'='*60}")
    print(f"  {interval} - Horizon {horizon} (CPO Only)")
    print(f"{'='*60}")

    df, feature_cols = engineer_features_for_horizon(merged_df, interval, horizon)
    print(f"  Features: {len(feature_cols)}, Samples: {len(df)}")

    data = prepare_train_test_val(df, feature_cols, cfg['test_ratio'])
    print(f"  Train: {len(data['X_train'])}, Test: {len(data['X_test'])}, "
          f"Val (2026+): {len(data['X_val'])}")

    exog_train, exog_indices = select_top_exog(
        data['X_train'], data['y_train'], n=min(10, data['X_train'].shape[1]))
    exog_test = data['X_test'][:, exog_indices]
    exog_val  = (data['X_val'][:, exog_indices]
                 if len(data['X_val']) else np.empty((0, len(exog_indices))))

    X_all_pre    = np.vstack([data['X_train'], data['X_test']])
    y_all_pre    = np.concatenate([data['y_train'], data['y_test']])
    exog_all_pre = np.vstack([exog_train, exog_test])

    artifacts_dir = os.path.join(MODELS_DIR, SCRIPT_TAG, interval, f'h{horizon}')
    os.makedirs(artifacts_dir, exist_ok=True)

    train_preds, train_results = {}, {}
    test_preds,  test_results  = {}, {}
    val_preds,   val_results   = {}, {}
    all_params = {}

    for model_type in model_types:
        print(f"\n  {model_type.upper()}:")

        # BASE
        t0 = time.time()
        if model_type in ('xgboost', 'random_forest'):
            model = create_sklearn_model(model_type)
            model.fit(data['X_train'], data['y_train'])
            y_pred_test  = model.predict(data['X_test'])
            y_pred_train = model.predict(data['X_train'])
            model_val = create_sklearn_model(model_type)
            model_val.fit(X_all_pre, y_all_pre)
            y_pred_val = (model_val.predict(data['X_val'])
                          if len(data['X_val']) else np.array([]))
            base_params = BASE_PARAMS[model_type].copy()
            save_model_artifacts(
                model=model_val, model_type=model_type, scaler=data['scaler'],
                feature_cols=feature_cols, exog_indices=list(exog_indices), params=base_params,
                save_dir=os.path.join(artifacts_dir, f'{model_type}_base'),
                gcs_bucket=os.environ.get('GCS_BUCKET'),
                gcs_prefix=f'models/{SCRIPT_TAG}/{interval}/h{horizon}/{model_type}_base',
            )
        else:
            bp = BASE_PARAMS[model_type]
            order = bp['order']
            seasonal_order = ((*bp['seasonal_order_pdq'], cfg['seasonal_period'])
                              if model_type == 'sarimax' else (0, 0, 0, 0))
            fitted = train_statsmodels(model_type, data['y_train'], exog_train, order, seasonal_order)
            if fitted is not None:
                y_pred_test  = predict_statsmodels(fitted, exog_test)
                y_pred_train = np.array(fitted.fittedvalues)
            else:
                y_pred_test  = np.full(len(data['y_test']),  np.mean(data['y_train']))
                y_pred_train = np.full(len(data['y_train']), np.mean(data['y_train']))
            fitted_val = train_statsmodels(model_type, y_all_pre, exog_all_pre, order, seasonal_order)
            y_pred_val = (predict_statsmodels(fitted_val, exog_val)
                          if fitted_val is not None and len(exog_val)
                          else np.full(len(data['y_val']), np.nan))
            base_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}
            save_model_artifacts(
                model=fitted_val, model_type=model_type, scaler=data['scaler'],
                feature_cols=feature_cols, exog_indices=list(exog_indices), params=base_params,
                save_dir=os.path.join(artifacts_dir, f'{model_type}_base'),
                gcs_bucket=os.environ.get('GCS_BUCKET'),
                gcs_prefix=f'models/{SCRIPT_TAG}/{interval}/h{horizon}/{model_type}_base',
            )

        key = f'{model_type}_base'
        train_preds[key] = y_pred_train; test_preds[key] = y_pred_test; val_preds[key] = y_pred_val
        train_results[key] = calculate_metrics(data['y_train'], y_pred_train, data['close_train'])
        test_results[key]  = calculate_metrics(data['y_test'],  y_pred_test,  data['close_test'])
        val_results[key]   = (calculate_metrics(data['y_val'], y_pred_val, data['close_val'])
                              if len(data['y_val']) else {})
        all_params[key] = base_params
        m = test_results[key]
        print(f"    BASE  - MAPE: {m['MAPE']:.2f}%  RMSE: {m['RMSE']:.2f}  "
              f"R²(price): {m['R2_Price']:.4f}  R²(lr): {m['R2_LogReturn']:.4f}  ({time.time()-t0:.1f}s)")

        # CSA
        if csa_config.get('enabled', True):
            t0 = time.time()
            if model_type in ('xgboost', 'random_forest'):
                obj_fn = csa_objective_sklearn(model_type, data['X_train'], data['y_train'], csa_config['cv_folds'])
            elif model_type == 'arimax':
                obj_fn = csa_objective_arimax(data['y_train'], exog_train, csa_config['cv_folds'])
            else:
                obj_fn = csa_objective_sarimax(data['y_train'], exog_train, cfg['seasonal_period'], csa_config['cv_folds'])
            csa_result = run_csa(model_type, obj_fn, csa_config['population_size'], csa_config['max_iterations'])
            best_params = csa_result.best_params
            if model_type in ('xgboost', 'random_forest'):
                model_csa = create_sklearn_model(model_type, best_params)
                model_csa.fit(data['X_train'], data['y_train'])
                y_pred_test  = model_csa.predict(data['X_test'])
                y_pred_train = model_csa.predict(data['X_train'])
                model_val = create_sklearn_model(model_type, best_params)
                model_val.fit(X_all_pre, y_all_pre)
                y_pred_val = (model_val.predict(data['X_val']) if len(data['X_val']) else np.array([]))
                csa_params = dict(best_params)
                save_model_artifacts(
                    model=model_val, model_type=model_type, scaler=data['scaler'],
                    feature_cols=feature_cols, exog_indices=list(exog_indices), params=csa_params,
                    save_dir=os.path.join(artifacts_dir, f'{model_type}_csa'),
                    gcs_bucket=os.environ.get('GCS_BUCKET'),
                    gcs_prefix=f'models/{SCRIPT_TAG}/{interval}/h{horizon}/{model_type}_csa',
                )
            else:
                order = (int(best_params.get('p', 1)), int(best_params.get('d', 1)), int(best_params.get('q', 1)))
                seasonal_order = ((int(best_params.get('P', 1)), int(best_params.get('D', 0)),
                                   int(best_params.get('Q', 1)), cfg['seasonal_period'])
                                  if model_type == 'sarimax' else (0, 0, 0, 0))
                fitted = train_statsmodels(model_type, data['y_train'], exog_train, order, seasonal_order)
                if fitted is not None:
                    y_pred_test  = predict_statsmodels(fitted, exog_test)
                    y_pred_train = np.array(fitted.fittedvalues)
                else:
                    y_pred_test  = test_preds[f'{model_type}_base'].copy()
                    y_pred_train = train_preds[f'{model_type}_base'].copy()
                fitted_val = train_statsmodels(model_type, y_all_pre, exog_all_pre, order, seasonal_order)
                y_pred_val = (predict_statsmodels(fitted_val, exog_val)
                              if fitted_val is not None and len(exog_val)
                              else np.full(len(data['y_val']), np.nan))
                csa_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}
                save_model_artifacts(
                    model=fitted_val, model_type=model_type, scaler=data['scaler'],
                    feature_cols=feature_cols, exog_indices=list(exog_indices), params=csa_params,
                    save_dir=os.path.join(artifacts_dir, f'{model_type}_csa'),
                    gcs_bucket=os.environ.get('GCS_BUCKET'),
                    gcs_prefix=f'models/{SCRIPT_TAG}/{interval}/h{horizon}/{model_type}_csa',
                )
            key = f'{model_type}_csa'
            train_preds[key] = y_pred_train; test_preds[key] = y_pred_test; val_preds[key] = y_pred_val
            train_results[key] = calculate_metrics(data['y_train'], y_pred_train, data['close_train'])
            test_results[key]  = calculate_metrics(data['y_test'],  y_pred_test,  data['close_test'])
            val_results[key]   = (calculate_metrics(data['y_val'], y_pred_val, data['close_val']) if len(data['y_val']) else {})
            all_params[key] = {**csa_params, 'csa_best_score': float(csa_result.best_score), 'csa_iterations': csa_result.total_iterations}
            m = test_results[key]
            print(f"    CSA   - MAPE: {m['MAPE']:.2f}%  RMSE: {m['RMSE']:.2f}  "
                  f"R²(price): {m['R2_Price']:.4f}  R²(lr): {m['R2_LogReturn']:.4f}  ({time.time()-t0:.1f}s)")

        # BAYESIAN
        if bayes_config and bayes_config.get('enabled', False):
            t0 = time.time()
            bcfg = _Cfg(seasonal_period=cfg['seasonal_period'], name=interval)
            bayes_opt = BayesianTimeSeriesOptimizer(
                model_type=model_type, X_train=data['X_train'], y_train=data['y_train'],
                config=bcfg, cv_folds=bayes_config['cv_folds'],
                n_calls=bayes_config['n_calls'], n_initial_points=bayes_config['n_initial_points'],
            )
            bayes_result = bayes_opt.optimize()
            best_params  = bayes_result.best_params
            if model_type in ('xgboost', 'random_forest'):
                model_bayes = create_sklearn_model(model_type, best_params)
                model_bayes.fit(data['X_train'], data['y_train'])
                y_pred_test  = model_bayes.predict(data['X_test'])
                y_pred_train = model_bayes.predict(data['X_train'])
                model_val = create_sklearn_model(model_type, best_params)
                model_val.fit(X_all_pre, y_all_pre)
                y_pred_val = (model_val.predict(data['X_val']) if len(data['X_val']) else np.array([]))
                bayes_params = dict(best_params)
                save_model_artifacts(
                    model=model_val, model_type=model_type, scaler=data['scaler'],
                    feature_cols=feature_cols, exog_indices=list(exog_indices), params=bayes_params,
                    save_dir=os.path.join(artifacts_dir, f'{model_type}_bayesian'),
                    gcs_bucket=os.environ.get('GCS_BUCKET'),
                    gcs_prefix=f'models/{SCRIPT_TAG}/{interval}/h{horizon}/{model_type}_bayesian',
                )
            else:
                order = (int(best_params.get('p', 1)), int(best_params.get('d', 1)), int(best_params.get('q', 1)))
                seasonal_order = ((int(best_params.get('P', 1)), int(best_params.get('D', 0)),
                                   int(best_params.get('Q', 1)), cfg['seasonal_period'])
                                  if model_type == 'sarimax' else (0, 0, 0, 0))
                fitted = train_statsmodels(model_type, data['y_train'], exog_train, order, seasonal_order)
                if fitted is not None:
                    y_pred_test  = predict_statsmodels(fitted, exog_test)
                    y_pred_train = np.array(fitted.fittedvalues)
                else:
                    y_pred_test  = test_preds[f'{model_type}_base'].copy()
                    y_pred_train = train_preds[f'{model_type}_base'].copy()
                fitted_val = train_statsmodels(model_type, y_all_pre, exog_all_pre, order, seasonal_order)
                y_pred_val = (predict_statsmodels(fitted_val, exog_val)
                              if fitted_val is not None and len(exog_val)
                              else np.full(len(data['y_val']), np.nan))
                bayes_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}
                save_model_artifacts(
                    model=fitted_val, model_type=model_type, scaler=data['scaler'],
                    feature_cols=feature_cols, exog_indices=list(exog_indices), params=bayes_params,
                    save_dir=os.path.join(artifacts_dir, f'{model_type}_bayesian'),
                    gcs_bucket=os.environ.get('GCS_BUCKET'),
                    gcs_prefix=f'models/{SCRIPT_TAG}/{interval}/h{horizon}/{model_type}_bayesian',
                )
            key = f'{model_type}_bayesian'
            train_preds[key] = y_pred_train; test_preds[key] = y_pred_test; val_preds[key] = y_pred_val
            train_results[key] = calculate_metrics(data['y_train'], y_pred_train, data['close_train'])
            test_results[key]  = calculate_metrics(data['y_test'],  y_pred_test,  data['close_test'])
            val_results[key]   = (calculate_metrics(data['y_val'], y_pred_val, data['close_val']) if len(data['y_val']) else {})
            all_params[key] = {**bayes_params, 'bayes_best_score': float(bayes_result.best_score), 'bayes_iterations': bayes_result.total_iterations}
            m = test_results[key]
            print(f"    BAYES - MAPE: {m['MAPE']:.2f}%  RMSE: {m['RMSE']:.2f}  "
                  f"R²(price): {m['R2_Price']:.4f}  R²(lr): {m['R2_LogReturn']:.4f}  ({time.time()-t0:.1f}s)")

    # Save
    horizon_dir = os.path.join(output_dir, interval, f'horizon_{horizon}')
    os.makedirs(horizon_dir, exist_ok=True)
    _save_split_outputs('training',   data['train_dates'], data['y_train'], data['close_train'],
                        train_preds, train_results, horizon_dir, interval, horizon)
    _save_split_outputs('testing',    data['test_dates'],  data['y_test'],  data['close_test'],
                        test_preds,  test_results,  horizon_dir, interval, horizon)
    if len(data['val_dates']):
        _save_split_outputs('validation', data['val_dates'], data['y_val'], data['close_val'],
                            val_preds, val_results, horizon_dir, interval, horizon)

    params_data = {
        'interval': interval, 'horizon': horizon, 'data_sources': ['cpo_variables'],
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_features': len(feature_cols), 'n_train': len(data['X_train']),
        'n_test': len(data['X_test']), 'n_val': len(data['X_val']),
        'val_start': str(data['val_dates'][0])  if len(data['val_dates']) else None,
        'val_end':   str(data['val_dates'][-1]) if len(data['val_dates']) else None,
        'models': all_params,
    }
    with open(os.path.join(horizon_dir, f'params_{interval}_h{horizon}.json'), 'w') as f:
        json.dump(params_data, f, indent=2, default=str)

    colors = {
        'xgboost_base': '#2E86AB', 'xgboost_csa': '#1B4965', 'xgboost_bayesian': '#5BA4CF',
        'random_forest_base': '#A23B72', 'random_forest_csa': '#7B2D5F', 'random_forest_bayesian': '#C96FA0',
        'arimax_base': '#F18F01', 'arimax_csa': '#C67200', 'arimax_bayesian': '#FFB84D',
        'sarimax_base': '#2CA58D', 'sarimax_csa': '#1E7A68', 'sarimax_bayesian': '#57C4A9',
    }
    actual_price = data['close_test'] * np.exp(data['y_test'])
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(data['test_dates'], actual_price, label='Actual', color='black', linewidth=2)
    for name, preds in test_preds.items():
        ls = '--' if name.endswith('_base') else '-'
        safe = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
        ax.plot(data['test_dates'], data['close_test'] * np.exp(safe),
                label=name.replace('_', ' ').title(), color=colors.get(name, '#999'),
                linewidth=1.1, linestyle=ls, alpha=0.8)
    ax.set_title(f'{interval} Forecast (CPO Only, Testing) - Horizon {horizon}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('CPO Price (MYR/tonne)')
    ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(horizon_dir, f'overlay_{interval}_h{horizon}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if len(data['val_dates']):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data['val_dates'], data['close_val'] * np.exp(data['y_val']),
                label='Actual', color='black', linewidth=2)
        for name, preds in val_preds.items():
            ls = '--' if name.endswith('_base') else '-'
            safe = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
            ax.plot(data['val_dates'], data['close_val'] * np.exp(safe),
                    label=name.replace('_', ' ').title(), color=colors.get(name, '#999'),
                    linewidth=1.1, linestyle=ls, alpha=0.8)
        ax.set_title(f'{interval} Forecast (CPO Only, Validation 2026) - Horizon {horizon}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date'); ax.set_ylabel('CPO Price (MYR/tonne)')
        ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(horizon_dir, f'validation_overlay_{interval}_h{horizon}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    test_results_df = pd.DataFrame(
        [{'Model': k.rsplit('_', 1)[0], 'Optimization': k.rsplit('_', 1)[1].upper(), **v}
         for k, v in test_results.items()])
    opt_palette = {'BASE': '#5DA5DA', 'CSA': '#FAA43A', 'BAYESIAN': '#57C4A9'}
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, metric in zip(axes.flatten(),
                          ['MAPE', 'sMAPE', 'RMSE', 'Directional_Accuracy', 'R2_Price', 'R2_LogReturn']):
        pivot = test_results_df.pivot(index='Model', columns='Optimization', values=metric)
        pivot.plot(kind='bar', ax=ax,
                   color=[opt_palette.get(c, '#999') for c in pivot.columns], edgecolor='white')
        ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_xlabel(''); ax.legend(title='Optimization')
        ax.tick_params(axis='x', rotation=30); ax.grid(True, alpha=0.3, axis='y')
    opt_names = ' vs '.join(sorted(test_results_df['Optimization'].unique()))
    fig.suptitle(f'{interval} Horizon {horizon} (CPO Only, Testing) - {opt_names}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(horizon_dir, f'metrics_{interval}_h{horizon}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Outputs saved to {horizon_dir}")
    val_results_df = pd.DataFrame(
        [{'Model': k.rsplit('_', 1)[0], 'Optimization': k.rsplit('_', 1)[1].upper(), **v}
         for k, v in val_results.items() if v]) if val_results else pd.DataFrame()
    return test_results_df, val_results_df


# =============================================================================
# Cross-Horizon Summary
# =============================================================================

def _horizon_summary_plots(summary_df: pd.DataFrame, interval: str,
                           interval_dir: str, tag: str):
    summary_df.to_csv(
        os.path.join(interval_dir, f'horizon_summary_{interval}_{tag}.csv'), index=False)
    horizons = sorted(summary_df['Horizon'].unique())
    for metric in ['RMSE', 'MAPE']:
        fig, ax = plt.subplots(figsize=(14, 7))
        for (model, opt), grp in summary_df.groupby(['Model', 'Optimization']):
            ax.plot(grp['Horizon'], grp[metric], marker='o',
                    linestyle='--' if opt == 'BASE' else '-',
                    label=f'{model} ({opt})', linewidth=1.5)
        ax.set_title(f'{interval} (CPO Only) - {metric} Across Horizons ({tag.title()})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Forecast Horizon'); ax.set_ylabel(metric)
        ax.set_xticks(horizons); ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(interval_dir,
                                 f'{metric.lower()}_across_horizons_{interval}_{tag}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 7))
    for (model, opt), grp in summary_df.groupby(['Model', 'Optimization']):
        ax.plot(grp['Horizon'], grp['R2_Price'], marker='o',
                linestyle='--' if opt == 'BASE' else '-',
                label=f'{model} ({opt})', linewidth=1.5)
    ax.set_title(f'{interval} (CPO Only) - R² (Price Space) Across Horizons ({tag.title()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Forecast Horizon'); ax.set_ylabel('R² (Price Space)')
    ax.set_xticks(horizons); ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(interval_dir, f'r2_across_horizons_{interval}_{tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_horizon_summary(interval: str,
                             all_test_results: Dict[int, pd.DataFrame],
                             all_val_results: Dict[int, pd.DataFrame],
                             output_dir: str):
    interval_dir = os.path.join(output_dir, interval)
    os.makedirs(interval_dir, exist_ok=True)

    def _build(results_by_horizon):
        rows = []
        for h, rdf in sorted(results_by_horizon.items()):
            if rdf is None or rdf.empty:
                continue
            for _, row in rdf.iterrows():
                rows.append({'Horizon': h, 'Model': row['Model'], 'Optimization': row['Optimization'],
                              'MAPE': row['MAPE'], 'sMAPE': row['sMAPE'], 'RMSE': row['RMSE'],
                              'Directional_Accuracy': row['Directional_Accuracy'],
                              'R2_Price': row['R2_Price'], 'R2_LogReturn': row['R2_LogReturn']})
        return pd.DataFrame(rows)

    test_s = _build(all_test_results)
    if not test_s.empty:
        _horizon_summary_plots(test_s, interval, interval_dir, 'testing')
        print(f"\n  Testing summary saved to {interval_dir}")
        print(test_s.to_string(index=False))

    val_s = _build(all_val_results)
    if not val_s.empty:
        _horizon_summary_plots(val_s, interval, interval_dir, 'validation')
        print(f"\n  Validation summary saved to {interval_dir}")
        print(val_s.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def run_interval(interval: str, output_dir: str, csa_config: Dict, bayes_config: Dict = None):
    """Run all horizons for a given interval."""
    print(f"\n{'#'*70}")
    print(f"  MULTI-HORIZON FORECAST (CPO ONLY) - {interval.upper()}")
    print(f"  Horizons: {HORIZONS}")
    print(f"{'#'*70}")

    merged_df = load_and_merge_data(interval)

    all_test_results = {}
    all_val_results  = {}
    for h in HORIZONS:
        test_df, val_df = run_single_horizon(interval, h, merged_df, output_dir, csa_config, bayes_config)
        all_test_results[h] = test_df
        all_val_results[h]  = val_df

    generate_horizon_summary(interval, all_test_results, all_val_results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Horizon CPO Price Forecasting - CPO Only')
    parser.add_argument('--interval', type=str, default='daily',
                        choices=['daily'],
                        help='Data interval')
    parser.add_argument('--optimizer', type=str, default='csa',
                        choices=['csa', 'bayesian', 'both'],
                        help='Optimization algorithm (default: csa)')
    parser.add_argument('--csa-population', type=int, default=50)
    parser.add_argument('--csa-iterations', type=int, default=50)
    parser.add_argument('--csa-cv-folds', type=int, default=3)
    parser.add_argument('--bayes-calls', type=int, default=50)
    parser.add_argument('--bayes-init', type=int, default=10)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_horizons_cpo_only')
    os.makedirs(output_dir, exist_ok=True)

    run_csa = args.optimizer in ('csa', 'both')
    run_bayes = args.optimizer in ('bayesian', 'both')

    csa_config = {
        'enabled': run_csa,
        'population_size': args.csa_population,
        'max_iterations': args.csa_iterations,
        'cv_folds': args.csa_cv_folds,
    }
    bayes_config = {
        'enabled': run_bayes,
        'n_calls': args.bayes_calls,
        'n_initial_points': args.bayes_init,
        'cv_folds': args.csa_cv_folds,
    }

    start = time.time()

    run_interval(args.interval.capitalize(), output_dir, csa_config, bayes_config)

    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {time.time()-start:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
