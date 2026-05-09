"""
Multi-Horizon CPO Price Forecasting — C3 (Price + Sentiment)
============================================================

Ablation configuration C3: lagged price + news sentiment (no HMM).

Forecasts CPO prices at daily horizons 1–7. Each horizon is preprocessed
independently to prevent data leakage. XGBoost only, with `base` and `csa`
hyperparameter variants.

Splitting (matches horizon_forecast_configurable):
    pre-test (Date < VAL_CUTOFF)  → TimeSeriesSplit cross-validation
    test    (Date >= VAL_CUTOFF)  → final holdout

Usage:
    python horizon_forecast_C3_price_sentiment.py --interval daily
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
    PROJECT_ROOT, HORIZONS, BASE_PARAMS, VAL_CUTOFF, MODELS_DIR,
    CPO_VARS_DROP,
    prepare_cv_test_split, create_sklearn_model,
    calculate_metrics, csa_objective_sklearn, run_csa,
    save_model_artifacts, save_feature_importance,
)


SCRIPT_TAG = 'cpo_sentiment'


warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Interval configurations (CPO + Sentiment)
INTERVAL_CONFIGS = {
    'Daily': {
        'cpo_file':       os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Daily.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Daily.csv'),
        'min_samples': 100,
        'cv_folds':    5,
    },
}

# Per-source lag config (C3: price + sentiment).
# Lag 44 for sentiment is from sentiment-CPO correlation analysis.
LAG_CONFIG: List[Dict] = [
    {'source': 'Sentiment_Score', 'lags': [44]},
    {'source': 'Log_Return',      'lags': [1, 2, 3]},
]

USE_INTERACTIONS = True

EXCLUDE_COLS = {
    'Date', 'Target',
    'Close_Anchor', 'Close', 'Log_Return',
    'Dominant_Sentiment',
}


# =============================================================================
# Data Loading (CPO + Sentiment)
# =============================================================================

def load_and_merge_data(interval: str) -> pd.DataFrame:
    """Load CPO + sentiment data, dropping raw same-day OHLCV columns."""
    cfg = INTERVAL_CONFIGS[interval]

    print(f"  Loading CPO data...")
    cpo = pd.read_csv(cfg['cpo_file'])
    cpo['Date'] = pd.to_datetime(cpo['Date'])
    drop = [c for c in cpo.columns if c in CPO_VARS_DROP]
    cpo = cpo.drop(columns=drop).sort_values('Date').reset_index(drop=True)

    print(f"  Loading sentiment data...")
    sentiment = pd.read_csv(cfg['sentiment_file'])
    sentiment['Date'] = pd.to_datetime(sentiment['Date'])
    rename_map = {
        'Combined_Positive_Prob': 'Positive_Prob',
        'Combined_Negative_Prob': 'Negative_Prob',
        'Combined_Neutral_Prob':  'Neutral_Prob',
        'Combined_Confidence':    'Confidence',
    }
    sentiment = sentiment.rename(columns=rename_map)
    keep_cols = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
                 'Neutral_Prob', 'Confidence', 'Sentiment_Score']
    sentiment = sentiment[[c for c in keep_cols if c in sentiment.columns]]

    print(f"  Merging datasets...")
    merged = cpo.merge(sentiment, on='Date', how='inner', suffixes=('', '_sent'))
    merged = merged.sort_values('Date').reset_index(drop=True)

    print(f"  Merged: {len(merged)} rows, {merged['Date'].min()} to {merged['Date'].max()}")

    if len(merged) < cfg['min_samples']:
        raise ValueError(f"Only {len(merged)} rows, minimum required: {cfg['min_samples']}")

    return merged


# =============================================================================
# Feature Engineering (horizon-aware to prevent leakage)
# =============================================================================

def engineer_features_for_horizon(df: pd.DataFrame, horizon: int
                                  ) -> Tuple[pd.DataFrame, List[str]]:
    """Build features using CPO + sentiment data (no look-ahead)."""
    df = df.copy()

    df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)

    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)
    woy = df['Date'].dt.isocalendar().week.astype(int)
    df['WeekOfYear_Sin'] = np.sin(2 * np.pi * woy / 52)
    df['WeekOfYear_Cos'] = np.cos(2 * np.pi * woy / 52)

    for entry in LAG_CONFIG:
        col = entry['source']
        if col not in df.columns:
            continue
        for lag in entry['lags']:
            if lag < horizon:
                continue
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    if USE_INTERACTIONS:
        if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
            df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']

    df['Target'] = np.log(df['Close'].shift(-horizon) / df['Close'])

    df = df.dropna().reset_index(drop=True)

    numeric_dtypes = {'float64', 'float32', 'int64', 'int32'}
    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and str(df[c].dtype) in numeric_dtypes]

    return df, feature_cols


# =============================================================================
# Output helpers
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


def _save_cv_fold_predictions(cv_pred_rows: List[Dict], horizon_dir: str,
                              interval: str, horizon: int):
    if not cv_pred_rows:
        return
    pd.DataFrame(cv_pred_rows).to_csv(
        os.path.join(horizon_dir, f'cv_fold_predictions_{interval}_h{horizon}.csv'),
        index=False)


def _save_dataset(split: str, dates, X_raw, y, close, feature_cols: List[str],
                  horizon_dir: str, horizon: int):
    df = pd.DataFrame(X_raw, columns=feature_cols)
    df.insert(0, 'Date', dates)
    df['Close_Anchor'] = close
    df['Target_LogReturn'] = y
    df.to_csv(os.path.join(horizon_dir, f'dataset_{split}_h{horizon}.csv'), index=False)


# =============================================================================
# Single Horizon Pipeline
# =============================================================================

def run_single_horizon(interval: str, horizon: int, merged_df: pd.DataFrame,
                       output_dir: str, csa_config: Dict
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run XGBoost CV + test pipeline for one interval+horizon combination."""
    cfg = INTERVAL_CONFIGS[interval]
    model_types = ['xgboost']

    print(f"\n{'='*60}")
    print(f"  {interval} - Horizon {horizon} (C3: price + sentiment)")
    print(f"{'='*60}")

    df, feature_cols = engineer_features_for_horizon(merged_df, horizon)
    print(f"  Features: {len(feature_cols)}, Samples: {len(df)}")

    data = prepare_cv_test_split(df, feature_cols, cfg['cv_folds'], VAL_CUTOFF)
    print(f"  Pre-test: {len(data['X_pre'])}  "
          f"({pd.Timestamp(data['dates_pre'][0]).date()} → "
          f"{pd.Timestamp(data['dates_pre'][-1]).date()})")
    if len(data['X_test']):
        print(f"  Test    : {len(data['X_test'])}  "
              f"({pd.Timestamp(data['dates_test'][0]).date()} → "
              f"{pd.Timestamp(data['dates_test'][-1]).date()})")
    print(f"  CV folds: {cfg['cv_folds']}")

    artifacts_dir = os.path.join(MODELS_DIR, SCRIPT_TAG, interval, f'h{horizon}')
    os.makedirs(artifacts_dir, exist_ok=True)

    cv_results: Dict[str, Dict] = {}
    test_results: Dict[str, Dict] = {}
    test_preds: Dict[str, np.ndarray] = {}
    cv_pred_rows: List[Dict] = []
    all_params: Dict[str, Dict] = {}
    importance_models: Dict[str, object] = {}

    for model_type in model_types:
        print(f"\n  {model_type.upper()}:")

        # ------------------------------------------------------------------ BASE
        t0 = time.time()
        cv_metrics_list = []
        for fold in data['cv_splits']:
            m_fold = create_sklearn_model(model_type)
            m_fold.fit(fold['X_train'], fold['y_train'],
                       eval_set=[(fold['X_cv'], fold['y_cv'])], verbose=False)
            y_cv = m_fold.predict(fold['X_cv'])
            cv_metrics_list.append(calculate_metrics(fold['y_cv'], y_cv, fold['close_cv']))
            for i in range(len(fold['dates_cv'])):
                cv_pred_rows.append({
                    'Model':            model_type,
                    'Optimization':     'BASE',
                    'Fold':             fold['fold'],
                    'Date':              fold['dates_cv'][i],
                    'Close_Anchor':      fold['close_cv'][i],
                    'Actual_LogReturn':  fold['y_cv'][i],
                    'Actual_Price':      fold['close_cv'][i] * np.exp(np.clip(fold['y_cv'][i], -10, 10)),
                    f'{model_type}_LogReturn': y_cv[i],
                    f'{model_type}_Price':     fold['close_cv'][i] * np.exp(np.clip(y_cv[i], -10, 10)),
                })

        cv_avg = {k: round(float(np.mean([m[k] for m in cv_metrics_list])), 4)
                  for k in cv_metrics_list[0]}
        cv_results[f'{model_type}_base'] = cv_avg

        # Use last CV fold's val split as eval set for early stopping.
        last_fold = data['cv_splits'][-1]
        model_final = create_sklearn_model(model_type)
        model_final.fit(data['X_pre'], data['y_pre'],
                        eval_set=[(last_fold['X_cv'], last_fold['y_cv'])],
                        verbose=False)
        y_pred_test = (model_final.predict(data['X_test'])
                       if len(data['X_test']) else np.array([]))
        test_metrics = (calculate_metrics(data['y_test'], y_pred_test, data['close_test'])
                        if len(data['y_test']) else {})

        base_params = BASE_PARAMS[model_type].copy()
        save_model_artifacts(
            model=model_final, model_type=model_type, scaler=data['scaler'],
            feature_cols=feature_cols, params=base_params,
            save_dir=os.path.join(artifacts_dir, f'{model_type}_base'),
        )

        key = f'{model_type}_base'
        test_preds[key] = y_pred_test
        if test_metrics:
            test_results[key] = test_metrics
        all_params[key] = base_params
        importance_models[key] = model_final

        print(f"    BASE  - CV  MAPE: {cv_avg['MAPE']:.2f}%  RMSE: {cv_avg['RMSE']:.2f}  "
              f"R²(price): {cv_avg['R2_Price']:.4f}  R²(lr): {cv_avg['R2_LogReturn']:.4f}  "
              f"DirAcc: {cv_avg['Directional_Accuracy']:.2f}%  ({time.time()-t0:.1f}s)")
        if test_metrics:
            m = test_metrics
            print(f"          TEST MAPE: {m['MAPE']:.2f}%  RMSE: {m['RMSE']:.2f}  "
                  f"R²(price): {m['R2_Price']:.4f}  R²(lr): {m['R2_LogReturn']:.4f}  "
                  f"DirAcc: {m['Directional_Accuracy']:.2f}%")

        # ------------------------------------------------------------------ CSA
        if csa_config.get('enabled', True):
            t0 = time.time()
            obj_fn = csa_objective_sklearn(model_type, data['X_pre'],
                                           data['y_pre'], csa_config['cv_folds'])
            csa_result  = run_csa(model_type, obj_fn,
                                  csa_config['population_size'], csa_config['max_iterations'])
            best_params = csa_result.best_params

            cv_csa_metrics_list = []
            for fold in data['cv_splits']:
                m_fold_csa = create_sklearn_model(model_type, best_params)
                m_fold_csa.fit(fold['X_train'], fold['y_train'])
                y_cv_csa = m_fold_csa.predict(fold['X_cv'])
                cv_csa_metrics_list.append(
                    calculate_metrics(fold['y_cv'], y_cv_csa, fold['close_cv']))
                for i in range(len(fold['dates_cv'])):
                    cv_pred_rows.append({
                        'Model':            model_type,
                        'Optimization':     'CSA',
                        'Fold':             fold['fold'],
                        'Date':              fold['dates_cv'][i],
                        'Close_Anchor':      fold['close_cv'][i],
                        'Actual_LogReturn':  fold['y_cv'][i],
                        'Actual_Price':      fold['close_cv'][i] * np.exp(np.clip(fold['y_cv'][i], -10, 10)),
                        f'{model_type}_LogReturn': y_cv_csa[i],
                        f'{model_type}_Price':     fold['close_cv'][i] * np.exp(np.clip(y_cv_csa[i], -10, 10)),
                    })

            cv_csa_avg = {k: round(float(np.mean([m[k] for m in cv_csa_metrics_list])), 4)
                          for k in cv_csa_metrics_list[0]}
            cv_results[f'{model_type}_csa'] = cv_csa_avg

            model_csa_final = create_sklearn_model(model_type, best_params)
            model_csa_final.fit(data['X_pre'], data['y_pre'])
            y_pred_test_csa = (model_csa_final.predict(data['X_test'])
                               if len(data['X_test']) else np.array([]))
            csa_test_metrics = (calculate_metrics(data['y_test'], y_pred_test_csa,
                                                  data['close_test'])
                                if len(data['y_test']) else {})

            csa_params = dict(best_params)
            save_model_artifacts(
                model=model_csa_final, model_type=model_type, scaler=data['scaler'],
                feature_cols=feature_cols, params=csa_params,
                save_dir=os.path.join(artifacts_dir, f'{model_type}_csa'),
            )

            key = f'{model_type}_csa'
            test_preds[key] = y_pred_test_csa
            if csa_test_metrics:
                test_results[key] = csa_test_metrics
            all_params[key] = {**csa_params,
                               'csa_best_score': float(csa_result.best_score),
                               'csa_iterations': csa_result.total_iterations}
            importance_models[key] = model_csa_final

            print(f"    CSA   - CV  MAPE: {cv_csa_avg['MAPE']:.2f}%  RMSE: {cv_csa_avg['RMSE']:.2f}  "
                  f"R²(price): {cv_csa_avg['R2_Price']:.4f}  ({time.time()-t0:.1f}s)")
            if csa_test_metrics:
                m = csa_test_metrics
                print(f"          TEST MAPE: {m['MAPE']:.2f}%  RMSE: {m['RMSE']:.2f}  "
                      f"R²(price): {m['R2_Price']:.4f}  R²(lr): {m['R2_LogReturn']:.4f}  "
                      f"DirAcc: {m['Directional_Accuracy']:.2f}%")

    # ---------------------------------------------------------------------- Save
    horizon_dir = os.path.join(output_dir, interval, f'horizon_{horizon}')
    os.makedirs(horizon_dir, exist_ok=True)

    cv_rows = [{'Model': k.rsplit('_', 1)[0],
                'Optimization': k.rsplit('_', 1)[1].upper(), **v}
               for k, v in cv_results.items()]
    pd.DataFrame(cv_rows).to_csv(
        os.path.join(horizon_dir, f'cv_results_{interval}_h{horizon}.csv'), index=False)
    _save_cv_fold_predictions(cv_pred_rows, horizon_dir, interval, horizon)

    if len(data['dates_test']) and test_results:
        _save_split_outputs('testing', data['dates_test'], data['y_test'],
                            data['close_test'], test_preds, test_results,
                            horizon_dir, interval, horizon)

    _save_dataset('pretrain', data['dates_pre'], data['X_pre_raw'],
                  data['y_pre'], data['close_pre'], feature_cols, horizon_dir, horizon)
    if len(data['dates_test']):
        _save_dataset('test', data['dates_test'], data['X_test_raw'],
                      data['y_test'], data['close_test'], feature_cols,
                      horizon_dir, horizon)

    for key, mdl in importance_models.items():
        tag = key.rsplit('_', 1)[1].upper()
        save_feature_importance(mdl, feature_cols, horizon_dir, horizon, tag=tag)

    params_data = {
        'interval': interval, 'horizon': horizon,
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_features': len(feature_cols),
        'features':   feature_cols,
        'n_pre_test': int(len(data['X_pre'])),
        'n_test':     int(len(data['X_test'])),
        'n_cv_folds': cfg['cv_folds'],
        'val_cutoff': str(VAL_CUTOFF.date()),
        'test_start': str(data['dates_test'][0])  if len(data['dates_test']) else None,
        'test_end':   str(data['dates_test'][-1]) if len(data['dates_test']) else None,
        'lag_config': LAG_CONFIG,
        'use_interactions': USE_INTERACTIONS,
        'models':     all_params,
    }
    with open(os.path.join(horizon_dir, f'params_{interval}_h{horizon}.json'), 'w') as f:
        json.dump(params_data, f, indent=2, default=str)

    # ---------------------------------------------------------------------- Plots
    colors = {'xgboost_base': '#2E86AB', 'xgboost_csa': '#1B4965'}

    if len(data['dates_test']):
        actual_price = data['close_test'] * np.exp(np.clip(data['y_test'], -10, 10))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data['dates_test'], actual_price, label='Actual', color='black', linewidth=2)
        for name, preds in test_preds.items():
            if not len(preds):
                continue
            ls = '--' if name.endswith('_base') else '-'
            safe = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
            ax.plot(data['dates_test'], data['close_test'] * np.exp(safe),
                    label=name.replace('_', ' ').title(),
                    color=colors.get(name, '#999'), linewidth=1.1, linestyle=ls, alpha=0.8)
        ax.set_title(f'{interval} Forecast (Testing 2026+) - Horizon {horizon} (C3: price + sentiment)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date'); ax.set_ylabel('CPO Price (MYR/tonne)')
        ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(horizon_dir, f'overlay_{interval}_h{horizon}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    if test_results:
        test_results_df = pd.DataFrame(
            [{'Model': k.rsplit('_', 1)[0], 'Optimization': k.rsplit('_', 1)[1].upper(), **v}
             for k, v in test_results.items()])
        opt_palette = {'BASE': '#5DA5DA', 'CSA': '#FAA43A'}
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        for ax, metric in zip(axes.flatten(),
                              ['MAPE', 'sMAPE', 'RMSE', 'Directional_Accuracy', 'R2_Price', 'R2_LogReturn']):
            pivot = test_results_df.pivot(index='Model', columns='Optimization', values=metric)
            pivot.plot(kind='bar', ax=ax,
                       color=[opt_palette.get(c, '#999') for c in pivot.columns], edgecolor='white')
            ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.set_xlabel(''); ax.legend(title='Optimization')
            ax.tick_params(axis='x', rotation=30); ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle(f'{interval} Horizon {horizon} (Testing, C3) - BASE vs CSA',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(horizon_dir, f'metrics_{interval}_h{horizon}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        test_results_df = pd.DataFrame()

    cv_results_df = pd.DataFrame(cv_rows)
    print(f"  Outputs saved to {horizon_dir}")

    return cv_results_df, test_results_df


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
        ax.set_title(f'{interval} - {metric} Across Horizons (C3, {tag.title()})',
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
    ax.set_title(f'{interval} - R² (Price Space) Across Horizons (C3, {tag.title()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Forecast Horizon'); ax.set_ylabel('R² (Price Space)')
    ax.set_xticks(horizons); ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(interval_dir, f'r2_across_horizons_{interval}_{tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_horizon_summary(interval: str,
                             all_cv_results: Dict[int, pd.DataFrame],
                             all_test_results: Dict[int, pd.DataFrame],
                             output_dir: str):
    interval_dir = os.path.join(output_dir, interval)
    os.makedirs(interval_dir, exist_ok=True)

    def _build_summary(results_by_horizon):
        rows = []
        for h, rdf in sorted(results_by_horizon.items()):
            if rdf is None or rdf.empty:
                continue
            for _, row in rdf.iterrows():
                rows.append({'Horizon': h, 'Model': row['Model'],
                              'Optimization': row['Optimization'],
                              'MAPE': row['MAPE'], 'sMAPE': row['sMAPE'], 'RMSE': row['RMSE'],
                              'Directional_Accuracy': row['Directional_Accuracy'],
                              'R2_Price': row['R2_Price'], 'R2_LogReturn': row['R2_LogReturn']})
        return pd.DataFrame(rows)

    cv_summary = _build_summary(all_cv_results)
    if not cv_summary.empty:
        _horizon_summary_plots(cv_summary, interval, interval_dir, 'cv')
        print(f"\n  CV summary saved to {interval_dir}")
        print(cv_summary.to_string(index=False))

    test_summary = _build_summary(all_test_results)
    if not test_summary.empty:
        _horizon_summary_plots(test_summary, interval, interval_dir, 'testing')
        print(f"\n  Testing summary saved to {interval_dir}")
        print(test_summary.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def run_interval(interval: str, output_dir: str, csa_config: Dict,
                 horizons_filter=None):
    horizons = horizons_filter or HORIZONS
    print(f"\n{'#'*70}")
    print(f"  MULTI-HORIZON FORECAST - {interval.upper()}  (C3: price + sentiment)")
    print(f"  Horizons: {horizons}")
    print(f"{'#'*70}")

    merged_df = load_and_merge_data(interval)

    all_cv_results   = {}
    all_test_results = {}
    for h in horizons:
        cv_df, test_df = run_single_horizon(interval, h, merged_df, output_dir, csa_config)
        all_cv_results[h]   = cv_df
        all_test_results[h] = test_df

    if horizons_filter is None:
        generate_horizon_summary(interval, all_cv_results, all_test_results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Horizon CPO Price Forecasting — C3 (price + sentiment ablation)')
    parser.add_argument('--interval', type=str, default='daily', choices=['daily'])
    parser.add_argument('--csa-population', type=int, default=50)
    parser.add_argument('--csa-iterations', type=int, default=50)
    parser.add_argument('--csa-cv-folds', type=int, default=3)
    parser.add_argument('--no-csa', action='store_true')
    parser.add_argument('--horizons', type=str, default='',
                        help='Comma-separated horizons to run, e.g. "2,5,7" (default: all 7).')
    args = parser.parse_args()

    horizons_filter = None
    if args.horizons:
        horizons_filter = [int(x) for x in args.horizons.split(',') if x.strip()]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'output_horizons', SCRIPT_TAG)
    os.makedirs(output_dir, exist_ok=True)

    csa_config = {
        'enabled': not args.no_csa,
        'population_size': args.csa_population,
        'max_iterations': args.csa_iterations,
        'cv_folds': args.csa_cv_folds,
    }

    start = time.time()
    run_interval(args.interval.capitalize(), output_dir, csa_config,
                 horizons_filter=horizons_filter)
    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {time.time()-start:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
