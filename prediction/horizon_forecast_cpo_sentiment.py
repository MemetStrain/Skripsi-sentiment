"""
Multi-Horizon CPO Price Forecasting with CSA Optimization (CPO + Sentiment)
============================================================================

Uses CPO price variables (cpo_variables_{interval}.csv) and
news sentiment (sentiment_aggregate_{interval}.csv).
No HMM data.

Forecasts CPO prices at multiple horizons per interval:
- Daily:   horizons 1, 2, 3, 4, 5, 6, 7
- Weekly:  horizons 1, 2, 3, 4
- Monthly: horizons 1, 2, 3, 4, 5, 6

Each horizon is preprocessed independently to prevent data leakage.
Uses 4 model types (XGBoost, Random Forest, ARIMAX, SARIMAX), each as
base and CSA-optimized variants.

Usage:
    python horizon_forecast_cpo_sentiment.py --interval daily
    python horizon_forecast_cpo_sentiment.py --interval weekly
    python horizon_forecast_cpo_sentiment.py --interval monthly
    python horizon_forecast_cpo_sentiment.py --interval all
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
    PROJECT_ROOT, HORIZONS, BASE_PARAMS,
    prepare_train_test, create_sklearn_model, select_top_exog,
    train_statsmodels, predict_statsmodels, calculate_metrics,
    csa_objective_sklearn, csa_objective_arimax, csa_objective_sarimax, run_csa,
)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Interval configurations (CPO + Sentiment)
INTERVAL_CONFIGS = {
    'Daily': {
        'cpo_file': os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Daily.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Daily.csv'),
        'seasonal_period': 5,
        'base_lag_periods': [1, 2, 3, 5, 10, 20],
        'min_samples': 100,
        'test_ratio': 0.2,
    },
    'Weekly': {
        'cpo_file': os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Weekly.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Weekly.csv'),
        'seasonal_period': 4,
        'base_lag_periods': [1, 2, 4, 8, 12],
        'min_samples': 50,
        'test_ratio': 0.2,
    },
    'Monthly': {
        'cpo_file': os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Monthly.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Monthly.csv'),
        'seasonal_period': 4,
        'base_lag_periods': [1, 2, 3, 6],
        'min_samples': 30,
        'test_ratio': 0.2,
    },
}


# =============================================================================
# Data Loading (CPO + Sentiment)
# =============================================================================

def load_and_merge_data(interval: str) -> pd.DataFrame:
    """Load CPO and sentiment data, merge into a single DataFrame."""
    cfg = INTERVAL_CONFIGS[interval]

    print(f"  Loading CPO data...")
    cpo = pd.read_csv(cfg['cpo_file'])
    cpo['Date'] = pd.to_datetime(cpo['Date'])

    print(f"  Loading sentiment data...")
    sentiment = pd.read_csv(cfg['sentiment_file'])
    if interval == 'Daily':
        sentiment['Date'] = pd.to_datetime(sentiment['Date'])
        rename_map = {
            'Article_Count': 'Article_Count',
            'Combined_Positive_Prob': 'Positive_Prob',
            'Combined_Negative_Prob': 'Negative_Prob',
            'Combined_Neutral_Prob': 'Neutral_Prob',
            'Combined_Confidence': 'Confidence',
        }
    elif interval == 'Monthly':
        sentiment['Date'] = pd.to_datetime(sentiment['YearMonth'] + '-01')
        rename_map = {
            'Total_Articles': 'Article_Count',
            'Combined_Avg_Positive_Prob': 'Positive_Prob',
            'Combined_Avg_Negative_Prob': 'Negative_Prob',
            'Combined_Avg_Neutral_Prob': 'Neutral_Prob',
            'Combined_Avg_Confidence': 'Confidence',
        }
    elif interval == 'Weekly':
        sentiment['Date'] = pd.to_datetime(sentiment['Week_Start'])
        rename_map = {
            'Total_Articles': 'Article_Count',
            'Combined_Avg_Positive_Prob': 'Positive_Prob',
            'Combined_Avg_Negative_Prob': 'Negative_Prob',
            'Combined_Avg_Neutral_Prob': 'Neutral_Prob',
            'Combined_Avg_Confidence': 'Confidence',
        }
    sentiment = sentiment.rename(columns=rename_map)
    keep_cols = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
                 'Neutral_Prob', 'Confidence', 'Sentiment_Score']
    sentiment = sentiment[[c for c in keep_cols if c in sentiment.columns]]

    # Merge
    print(f"  Merging datasets...")
    if interval == 'Monthly':
        cpo['_ym'] = cpo['Date'].dt.to_period('M')
        sentiment['_ym'] = sentiment['Date'].dt.to_period('M')
        merged = cpo.merge(sentiment.drop(columns=['Date']), on='_ym', how='inner', suffixes=('', '_sent'))
        merged = merged.drop(columns=['_ym'])
    elif interval == 'Weekly':
        for df in [cpo]:
            df['_yw'] = df['Date'].dt.isocalendar().year.astype(str) + '-W' + \
                        df['Date'].dt.isocalendar().week.astype(str).str.zfill(2)
        sentiment['_yw'] = sentiment['Date'].dt.isocalendar().year.astype(str) + '-W' + \
                           sentiment['Date'].dt.isocalendar().week.astype(str).str.zfill(2)
        merged = cpo.merge(sentiment.drop(columns=['Date']), on='_yw', how='inner', suffixes=('', '_sent'))
        merged = merged.drop(columns=['_yw'])
    else:
        merged = cpo.merge(sentiment, on='Date', how='inner', suffixes=('', '_sent'))

    merged = merged.sort_values('Date').reset_index(drop=True)

    print(f"  Merged: {len(merged)} rows, {merged['Date'].min()} to {merged['Date'].max()}")

    if len(merged) < cfg['min_samples']:
        raise ValueError(f"Only {len(merged)} rows, minimum required: {cfg['min_samples']}")

    return merged


# =============================================================================
# Feature Engineering (horizon-aware to prevent leakage)
# =============================================================================

def engineer_features_for_horizon(df: pd.DataFrame, interval: str, horizon: int
                                  ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features for a specific forecast horizon using CPO + sentiment data.

    To prevent data leakage / look-ahead bias:
    - Lag features use lags >= horizon (so we never peek into the forecast window)
    - Target is Close shifted by -horizon
    """
    cfg = INTERVAL_CONFIGS[interval]
    df = df.copy()

    # Temporal features (safe - derived from date, not target)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)

    if interval == 'Daily':
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
    elif interval == 'Weekly':
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)

    # Lag features - only use lags >= horizon to prevent look-ahead bias
    safe_lags = [lag for lag in cfg['base_lag_periods'] if lag >= horizon]
    if not safe_lags:
        safe_lags = [horizon]

    lag_cols = ['Close', 'Sentiment_Score']
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in safe_lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Interaction features (using current-period values that are known at prediction time)
    if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
        df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']

    # Target: h-step cumulative log return (stationary; inverse-transform for price-space errors)
    df['Target'] = np.log(df['Close'].shift(-horizon) / df['Close'])

    # Drop rows with NaN (from lags and target shift)
    df = df.dropna().reset_index(drop=True)

    # Feature columns
    exclude = ['Date', 'Target', 'Dominant_Sentiment']
    feature_cols = [c for c in df.columns
                    if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    return df, feature_cols



# =============================================================================
# Single Horizon Pipeline
# =============================================================================

def run_single_horizon(interval: str, horizon: int, merged_df: pd.DataFrame,
                       output_dir: str, csa_config: Dict) -> pd.DataFrame:
    """Run full prediction pipeline for one interval+horizon combination."""
    cfg = INTERVAL_CONFIGS[interval]
    model_types = ['xgboost', 'random_forest', 'arimax', 'sarimax']

    print(f"\n{'='*60}")
    print(f"  {interval} - Horizon {horizon} (CPO + Sentiment)")
    print(f"{'='*60}")

    # Preprocess from scratch for this horizon
    df, feature_cols = engineer_features_for_horizon(merged_df, interval, horizon)
    print(f"  Features: {len(feature_cols)}, Samples: {len(df)}")

    data = prepare_train_test(df, feature_cols, cfg['test_ratio'])
    print(f"  Train: {len(data['X_train'])}, Test: {len(data['X_test'])}")

    # Exog for ARIMAX/SARIMAX
    exog_train, exog_indices = select_top_exog(
        data['X_train'], data['y_train'], n=min(10, data['X_train'].shape[1]))
    exog_test = data['X_test'][:, exog_indices]

    all_results = {}
    all_predictions = {}
    all_params = {}

    for model_type in model_types:
        print(f"\n  {model_type.upper()}:")

        # --- BASE ---
        t0 = time.time()
        if model_type in ('xgboost', 'random_forest'):
            model = create_sklearn_model(model_type)
            model.fit(data['X_train'], data['y_train'])
            y_pred_base = model.predict(data['X_test'])
            base_params = BASE_PARAMS[model_type].copy()
        else:
            bp = BASE_PARAMS[model_type]
            order = bp['order']
            if model_type == 'sarimax':
                seasonal_order = (*bp['seasonal_order_pdq'], cfg['seasonal_period'])
            else:
                seasonal_order = (0, 0, 0, 0)
            fitted = train_statsmodels(model_type, data['y_train'], exog_train,
                                       order, seasonal_order)
            if fitted is not None:
                y_pred_base = predict_statsmodels(fitted, exog_test)
            else:
                y_pred_base = np.full(len(data['y_test']), np.mean(data['y_train']))
            base_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}

        metrics_base = calculate_metrics(data['y_test'], y_pred_base, data['close_test'])
        all_results[f'{model_type}_base'] = metrics_base
        all_predictions[f'{model_type}_base'] = y_pred_base
        all_params[f'{model_type}_base'] = base_params
        print(f"    BASE  - MAPE: {metrics_base['MAPE']:.2f}%  RMSE: {metrics_base['RMSE']:.2f}  "
              f"R²(price): {metrics_base['R2_Price']:.4f}  R²(lr): {metrics_base['R2_LogReturn']:.4f}  ({time.time()-t0:.1f}s)")

        # --- CSA ---
        t0 = time.time()
        if model_type in ('xgboost', 'random_forest'):
            obj_fn = csa_objective_sklearn(model_type, data['X_train'],
                                           data['y_train'], csa_config['cv_folds'])
        elif model_type == 'arimax':
            obj_fn = csa_objective_arimax(data['y_train'], exog_train,
                                          csa_config['cv_folds'])
        else:
            obj_fn = csa_objective_sarimax(data['y_train'], exog_train,
                                           cfg['seasonal_period'], csa_config['cv_folds'])

        csa_result = run_csa(model_type, obj_fn,
                             csa_config['population_size'], csa_config['max_iterations'])
        best_params = csa_result.best_params

        if model_type in ('xgboost', 'random_forest'):
            model_csa = create_sklearn_model(model_type, best_params)
            model_csa.fit(data['X_train'], data['y_train'])
            y_pred_csa = model_csa.predict(data['X_test'])
            csa_params = dict(best_params)
        else:
            order = (int(best_params.get('p', 1)), int(best_params.get('d', 1)),
                     int(best_params.get('q', 1)))
            if model_type == 'sarimax':
                seasonal_order = (int(best_params.get('P', 1)),
                                  int(best_params.get('D', 0)),
                                  int(best_params.get('Q', 1)),
                                  cfg['seasonal_period'])
            else:
                seasonal_order = (0, 0, 0, 0)
            fitted = train_statsmodels(model_type, data['y_train'], exog_train,
                                       order, seasonal_order)
            if fitted is not None:
                y_pred_csa = predict_statsmodels(fitted, exog_test)
            else:
                y_pred_csa = y_pred_base.copy()
            csa_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}

        metrics_csa = calculate_metrics(data['y_test'], y_pred_csa, data['close_test'])
        all_results[f'{model_type}_csa'] = metrics_csa
        all_predictions[f'{model_type}_csa'] = y_pred_csa
        all_params[f'{model_type}_csa'] = {
            **csa_params,
            'csa_best_score': float(csa_result.best_score),
            'csa_iterations': csa_result.total_iterations,
        }
        print(f"    CSA   - MAPE: {metrics_csa['MAPE']:.2f}%  RMSE: {metrics_csa['RMSE']:.2f}  "
              f"R²(price): {metrics_csa['R2_Price']:.4f}  R²(lr): {metrics_csa['R2_LogReturn']:.4f}  ({time.time()-t0:.1f}s)")

    # --- Save outputs ---
    horizon_dir = os.path.join(output_dir, interval, f'horizon_{horizon}')
    os.makedirs(horizon_dir, exist_ok=True)

    # Results CSV
    rows = []
    for model_name, metrics in all_results.items():
        parts = model_name.rsplit('_', 1)
        rows.append({'Model': parts[0], 'Optimization': parts[1].upper(), **metrics})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(horizon_dir, f'results_{interval}_h{horizon}.csv'), index=False)

    # Predictions CSV
    pred_df = pd.DataFrame({
        'Date':             data['test_dates'],
        'Close_Anchor':     data['close_test'],                              # raw Close_t
        'Actual_LogReturn': data['y_test'],
        'Actual_Price':     data['close_test'] * np.exp(data['y_test']),     # unclipped — honest reporting
    })
    for name, preds in all_predictions.items():
        pred_df[f'{name}_LogReturn'] = preds
        safe = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
        pred_df[f'{name}_Price'] = data['close_test'] * np.exp(safe)
    pred_df.to_csv(os.path.join(horizon_dir, f'predictions_{interval}_h{horizon}.csv'), index=False)

    # Params JSON
    params_data = {
        'interval': interval, 'horizon': horizon,
        'data_sources': ['cpo_variables', 'sentiment_aggregate'],
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_features': len(feature_cols), 'n_train': len(data['X_train']),
        'n_test': len(data['X_test']), 'models': all_params,
    }
    with open(os.path.join(horizon_dir, f'params_{interval}_h{horizon}.json'), 'w') as f:
        json.dump(params_data, f, indent=2, default=str)

    # --- Plots ---
    colors = {
        'xgboost_base': '#2E86AB', 'xgboost_csa': '#1B4965',
        'random_forest_base': '#A23B72', 'random_forest_csa': '#7B2D5F',
        'arimax_base': '#F18F01', 'arimax_csa': '#C67200',
        'sarimax_base': '#2CA58D', 'sarimax_csa': '#1E7A68',
    }

    actual_price = data['close_test'] * np.exp(data['y_test'])
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(data['test_dates'], actual_price, label='Actual', color='black', linewidth=2)
    for name, preds in all_predictions.items():
        ls = '--' if name.endswith('_base') else '-'
        safe = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
        price_preds = data['close_test'] * np.exp(safe)
        ax.plot(data['test_dates'], price_preds, label=name.replace('_', ' ').title(),
                color=colors.get(name, '#999'), linewidth=1.1, linestyle=ls, alpha=0.8)
    ax.set_title(f'{interval} Forecast (CPO + Sentiment) - Horizon {horizon}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('CPO Price (MYR/tonne)')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(horizon_dir, f'overlay_{interval}_h{horizon}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Metrics bar chart
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, metric in zip(axes.flatten(), ['MAPE', 'sMAPE', 'RMSE', 'Directional_Accuracy', 'R2_Price', 'R2_LogReturn']):
        pivot = results_df.pivot(index='Model', columns='Optimization', values=metric)
        pivot.plot(kind='bar', ax=ax, color=['#5DA5DA', '#FAA43A'], edgecolor='white')
        ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.legend(title='Optimization')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'{interval} Horizon {horizon} (CPO + Sentiment) - Base vs CSA', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(horizon_dir, f'metrics_{interval}_h{horizon}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Outputs saved to {horizon_dir}")
    return results_df


# =============================================================================
# Cross-Horizon Summary
# =============================================================================

def generate_horizon_summary(interval: str, all_horizon_results: Dict[int, pd.DataFrame],
                             output_dir: str):
    """Generate cross-horizon comparison plots and summary CSV."""
    summary_rows = []
    for h, rdf in sorted(all_horizon_results.items()):
        for _, row in rdf.iterrows():
            summary_rows.append({
                'Horizon': h, 'Model': row['Model'],
                'Optimization': row['Optimization'],
                'MAPE': row['MAPE'], 'sMAPE': row['sMAPE'], 'RMSE': row['RMSE'],
                'Directional_Accuracy': row['Directional_Accuracy'],
                'R2_Price': row['R2_Price'], 'R2_LogReturn': row['R2_LogReturn'],
            })
    summary_df = pd.DataFrame(summary_rows)

    interval_dir = os.path.join(output_dir, interval)
    os.makedirs(interval_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(interval_dir, f'horizon_summary_{interval}.csv'), index=False)

    for metric in ['RMSE', 'MAPE']:
        fig, ax = plt.subplots(figsize=(14, 7))
        for (model, opt), grp in summary_df.groupby(['Model', 'Optimization']):
            ls = '--' if opt == 'BASE' else '-'
            label = f'{model} ({opt})'
            ax.plot(grp['Horizon'], grp[metric], marker='o', linestyle=ls, label=label, linewidth=1.5)
        ax.set_title(f'{interval} (CPO + Sentiment) - {metric} Across Horizons', fontsize=14, fontweight='bold')
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(all_horizon_results.keys()))
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(interval_dir, f'{metric.lower()}_across_horizons_{interval}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    for (model, opt), grp in summary_df.groupby(['Model', 'Optimization']):
        ls = '--' if opt == 'BASE' else '-'
        ax.plot(grp['Horizon'], grp['R2_Price'], marker='o', linestyle=ls,
                label=f'{model} ({opt})', linewidth=1.5)
    ax.set_title(f'{interval} (CPO + Sentiment) - R² (Price Space) Across Horizons', fontsize=14, fontweight='bold')
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('R² (Price Space)')
    ax.set_xticks(sorted(all_horizon_results.keys()))
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(interval_dir, f'r2_across_horizons_{interval}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Summary saved to {interval_dir}")
    print(summary_df.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def run_interval(interval: str, output_dir: str, csa_config: Dict):
    """Run all horizons for a given interval."""
    print(f"\n{'#'*70}")
    print(f"  MULTI-HORIZON FORECAST (CPO + SENTIMENT) - {interval.upper()}")
    print(f"  Horizons: {HORIZONS[interval]}")
    print(f"{'#'*70}")

    merged_df = load_and_merge_data(interval)

    all_horizon_results = {}
    for h in HORIZONS[interval]:
        results_df = run_single_horizon(interval, h, merged_df, output_dir, csa_config)
        all_horizon_results[h] = results_df

    generate_horizon_summary(interval, all_horizon_results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Horizon CPO Price Forecasting - CPO + Sentiment')
    parser.add_argument('--interval', type=str, required=True,
                        choices=['daily', 'weekly', 'monthly', 'all'],
                        help='Data interval (or "all" for all intervals)')
    parser.add_argument('--csa-population', type=int, default=50)
    parser.add_argument('--csa-iterations', type=int, default=50)
    parser.add_argument('--csa-cv-folds', type=int, default=3)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_horizons_cpo_sentiment')
    os.makedirs(output_dir, exist_ok=True)

    csa_config = {
        'population_size': args.csa_population,
        'max_iterations': args.csa_iterations,
        'cv_folds': args.csa_cv_folds,
    }

    start = time.time()

    if args.interval == 'all':
        for interval in ['Daily', 'Weekly', 'Monthly']:
            run_interval(interval, output_dir, csa_config)
    else:
        run_interval(args.interval.capitalize(), output_dir, csa_config)

    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {time.time()-start:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
