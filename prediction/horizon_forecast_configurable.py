"""
Multi-Horizon CPO Price Forecasting — Configurable
====================================================

Edit the CONFIG block below to select data sources, features, lags, XGBoost
parameters, and which horizons to run.

Follows the exact same pipeline as horizon_forecast_C4_full.py:
  - Target  : h-step log return  = log(Close[t+h] / Close[t])
  - Scaler  : RobustScaler fitted on training set only
  - Splits  : train / test (ratio-based) / validation (date cutoff, 2026+)
  - Metrics : MAPE, sMAPE, RMSE, Directional_Accuracy, R2_Price, R2_LogReturn

Outputs per horizon  (output_horizons_{OUTPUT_TAG}/Daily/horizon_{h}/):
  training_predictions_Daily_h{n}.csv   — actual + predicted log-return & price
  testing_predictions_Daily_h{n}.csv
  validation_predictions_Daily_h{n}.csv
  training_results_Daily_h{n}.csv       — metrics table
  testing_results_Daily_h{n}.csv
  validation_results_Daily_h{n}.csv
  params_Daily_h{n}.json                — feature list, split sizes, XGB params
  dataset_train_h{n}.csv               — raw feature matrix + target (train)
  dataset_test_h{n}.csv                — raw feature matrix + target (test)
  dataset_val_h{n}.csv                 — raw feature matrix + target (validation)

Run:
    python horizon_forecast_configurable.py [--horizons 1,3,5]
"""

import json
import os
import sys
import time
import warnings
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.forecast_utils import (
    PROJECT_ROOT,
    calculate_metrics,
    save_model_artifacts,
    csa_objective_sklearn,
    run_csa,
)
from feature_engineering import build_unified_features

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def _xgb_mape_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE with epsilon guard — avoids inf when log-return target is near zero."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG — edit this block before running
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Data sources ──────────────────────────────────────────────────────────────
# Toggle which external datasets to merge in. At least one must be True.
USE_HMM:       bool = True   # HMM regime (State, Volatility, RSI, MACD)
USE_SENTIMENT: bool = True   # FinBERT title sentiment
USE_CPO_VARS:  bool = True   # CPO technical variables (spreads, SMAs, Bollinger, …)

# ── Horizons ──────────────────────────────────────────────────────────────────
# Days ahead to forecast. Any subset of positive integers.
HORIZONS: list[int] = [1, 2, 3, 4, 5, 6, 7]

# ── Ablation configuration ────────────────────────────────────────────────────
# The unified feature schema (prediction/master_features.py) is keyed by an
# ablation name. It is derived from the data-source toggles above so the merged
# frame and the emitted column schema can never disagree. The schema (which
# base features, which lag indices, calendar + interaction terms) is fixed by
# master_features.py — there is no per-script lag list to tune here.
ABLATION: str = (
    'C4_full'          if USE_HMM and USE_SENTIMENT else
    'C2_cpo_hmm'       if USE_HMM                   else
    'C3_cpo_sentiment' if USE_SENTIMENT             else
    'C1_cpo_only'
)

# ── Split config ─────────────────────────────────────────────────────────────
VAL_CUTOFF: str = '2026-01-01'  # data before this = CV; data from this onward = test
CV_FOLDS:   int = 5             # TimeSeriesSplit folds on pre-2026 data

# ── XGBoost parameters ────────────────────────────────────────────────────────
# Set to {} to use XGBoost library defaults.
XGB_PARAMS: dict = {
    'n_estimators':          20000,
    'max_depth':             9,
    'learning_rate':         0.01,
    'subsample':             0.6715220780746508,
    'colsample_bytree':      0.6,
    'min_child_weight':      8,
    'reg_alpha':             0.1,
    'reg_lambda':            0.5,
    'early_stopping_rounds': 5,
    'eval_metric':           _xgb_mape_metric,
}

# ── CSA (Crow Search Algorithm) hyperparameter optimisation ──────────────────
USE_CSA:        bool = True  # set False or pass --no-csa to skip
CSA_POPULATION: int  = 50
CSA_ITERATIONS: int  = 50
CSA_CV_FOLDS:   int  = 3

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_TAG:   str  = 'configurable'  # output goes to output_horizons_{OUTPUT_TAG}/
SAVE_DATASET: bool = True            # save raw feature matrices (ML input/output)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS  (derived from PROJECT_ROOT; do not edit unless your layout differs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HERE    = Path(os.path.abspath(__file__)).parent
_ROOT    = Path(PROJECT_ROOT)
_CPO_FILE       = _ROOT / 'cpo/output/cpo_variables_Daily.csv'
_SENTIMENT_FILE = _ROOT / 'news/output/sentiment_aggregate_Daily_title.csv'
_HMM_FILE       = _ROOT / 'markov/output/hmm_states_results_Daily.csv'

# Raw same-day OHLC / Change_Pct dropped from CPO vars. Volume is kept — the
# unified schema lags it (Volume_lag{k}), so there is no same-day leakage.
_CPO_VARS_DROP = {'Open', 'High', 'Low', 'Change_Pct'}


# =============================================================================
# Data loading
# =============================================================================

def load_and_merge() -> pd.DataFrame:
    """Load CPO vars (mandatory) + HMM + sentiment (conditional) and merge."""
    print("  Loading CPO variables...")
    cpo = pd.read_csv(_CPO_FILE, parse_dates=['Date'])
    drop = [c for c in cpo.columns if c in _CPO_VARS_DROP]
    cpo = cpo.drop(columns=drop).sort_values('Date').reset_index(drop=True)

    merged = cpo

    if USE_HMM:
        print("  Loading HMM states...")
        hmm = pd.read_csv(_HMM_FILE, parse_dates=['Date'])
        hmm = hmm.rename(columns={
            'Close':      'HMM_Close',
            'Log_Return': 'HMM_Log_Return',
            'Volatility': 'HMM_Volatility',
            'RSI':        'HMM_RSI',
            'MACD':       'HMM_MACD',
            'State':      'HMM_State',
            'State_Label':'HMM_State_Label',
        })
        merged = merged.merge(hmm, on='Date', how='inner')

    if USE_SENTIMENT:
        print("  Loading sentiment...")
        sent = pd.read_csv(_SENTIMENT_FILE, parse_dates=['Date'])
        rename = {
            'Combined_Positive_Prob': 'Positive_Prob',
            'Combined_Negative_Prob': 'Negative_Prob',
            'Combined_Neutral_Prob':  'Neutral_Prob',
            'Combined_Confidence':    'Confidence',
        }
        sent = sent.rename(columns=rename)
        keep = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
                'Neutral_Prob', 'Confidence', 'Sentiment_Score']
        sent = sent[[c for c in keep if c in sent.columns]]
        merged = merged.merge(sent, on='Date', how='inner')

    merged = merged.sort_values('Date').reset_index(drop=True)

    # One-hot encode HMM state label if present (drop the string column)
    if 'HMM_State_Label' in merged.columns:
        for state in merged['HMM_State_Label'].unique():
            col = f'HMM_{state.replace(" ", "_").replace("-", "_")}'
            merged[col] = (merged['HMM_State_Label'] == state).astype(int)
        merged = merged.drop(columns=['HMM_State_Label'])

    print(f"  Merged: {len(merged)} rows, "
          f"{merged['Date'].min().date()} → {merged['Date'].max().date()}")
    return merged


# =============================================================================
# Feature engineering (horizon-aware)
# =============================================================================

def build_features(df: pd.DataFrame, horizon: int):
    """
    Build the unified-schema feature matrix for one forecast horizon.

    Delegates to feature_engineering.build_unified_features (Formula A: every
    lag-k column is shifted by k + h - 1; calendar features are anchored at the
    forecast origin d - h), so the feature schema is identical across all 7
    horizons for the active ABLATION.

    Returns (df_clean, feature_cols) where df_clean carries `Target`
    (log(C[d]/C[d-h])) and `Close` (the forecast-origin anchor C[d-h], used to
    invert log-return predictions back to price) alongside the feature columns,
    with all NaN rows removed.
    """
    out, feature_cols = build_unified_features(df, horizon, ablation=ABLATION)
    out = out.rename(columns={'Target_LogReturn': 'Target',
                              'Close_Origin': 'Close'})
    return out, feature_cols


# =============================================================================
# Helpers
# =============================================================================

def _make_xgb(params: dict) -> XGBRegressor:
    valid = set(XGBRegressor().get_params().keys())
    filtered = {k: v for k, v in params.items() if k in valid}
    return XGBRegressor(**filtered, verbosity=1, random_state=42)


def _build_data(df: pd.DataFrame, feature_cols: list[str],
                n_cv_folds: int, val_cutoff: pd.Timestamp) -> dict:
    """
    Time-series cross-validation split.

    - pre-test rows (Date < val_cutoff) : TimeSeriesSplit into CV folds
    - test rows    (Date >= val_cutoff) : final holdout

    One global RobustScaler is fit on all pre-test data and reused across
    folds and the final model.
    """
    pre  = df[df['Date'] < val_cutoff].reset_index(drop=True)
    test = df[df['Date'] >= val_cutoff].reset_index(drop=True)

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

    scaler    = RobustScaler()
    X_pre_s   = scaler.fit_transform(X_pre_raw)
    X_test_s  = scaler.transform(X_test_raw) if len(X_test_raw) else X_test_raw

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

    return dict(
        X_pre=X_pre_s,   y_pre=y_pre,   dates_pre=dates_pre,   close_pre=close_pre,
        X_test=X_test_s, y_test=y_test, dates_test=dates_test, close_test=close_test,
        cv_splits=cv_splits,
        scaler=scaler,
        X_pre_raw=X_pre_raw, X_test_raw=X_test_raw,
    )


def _save_predictions(split: str, dates, y_true, close_anchor, y_pred,
                      out_dir: Path, horizon: int, optimization: str = 'BASE'):
    """Save prediction CSV matching C4's column format."""
    pred_df = pd.DataFrame({
        'Date':             dates,
        'Close_Anchor':     close_anchor,
        'Actual_LogReturn': y_true,
        'Actual_Price':     close_anchor * np.exp(np.clip(y_true, -10, 10)),
        'xgboost_LogReturn': y_pred,
        'xgboost_Price':     close_anchor * np.exp(np.clip(y_pred, -10, 10)),
    })
    suffix = '' if optimization == 'BASE' else f'_{optimization.lower()}'
    pred_df.to_csv(out_dir / f'{split}_predictions{suffix}_Daily_h{horizon}.csv', index=False)


def _save_results(split: str, metrics: dict, out_dir: Path, horizon: int,
                  optimization: str = 'BASE'):
    """Save results CSV matching C4's format (Model, Optimization, …metrics…)."""
    row = {'Model': 'xgboost', 'Optimization': optimization, **metrics}
    suffix = '' if optimization == 'BASE' else f'_{optimization.lower()}'
    pd.DataFrame([row]).to_csv(
        out_dir / f'{split}_results{suffix}_Daily_h{horizon}.csv', index=False)


def _save_dataset(split: str, dates, X_raw, y, close, feature_cols: list[str],
                  out_dir: Path, horizon: int):
    """Save the raw (unscaled) feature matrix + target for the given split."""
    df = pd.DataFrame(X_raw, columns=feature_cols)
    df.insert(0, 'Date', dates)
    df['Close_Anchor'] = close
    df['Target_LogReturn'] = y
    df.to_csv(out_dir / f'dataset_{split}_h{horizon}.csv', index=False)


# =============================================================================
# Single horizon runner
# =============================================================================

def run_horizon(horizon: int, merged: pd.DataFrame,
                output_dir: Path, val_cutoff: pd.Timestamp,
                csa_config: dict | None = None) -> dict:
    """Full train→test→validate pipeline for one horizon."""
    print(f"\n{'='*60}")
    print(f"  Horizon {horizon}")
    print(f"{'='*60}")

    df, feature_cols = build_features(merged, horizon)
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Samples   : {len(df)}  ({df['Date'].min().date()} → {df['Date'].max().date()})")

    data = _build_data(df, feature_cols, CV_FOLDS, val_cutoff)
    print(f"  Pre-test  : {len(data['X_pre'])}"
          f"  ({pd.Timestamp(data['dates_pre'][0]).date()} → "
          f"{pd.Timestamp(data['dates_pre'][-1]).date()})")
    if len(data['X_test']):
        print(f"  Test      : {len(data['X_test'])}"
              f"  ({pd.Timestamp(data['dates_test'][0]).date()} → "
              f"{pd.Timestamp(data['dates_test'][-1]).date()})")
    print(f"  CV folds  : {CV_FOLDS}")

    # ── Cross-validation (walk-forward) ──────────────────────────────────────
    t0 = time.time()
    cv_metrics_list = []
    cv_pred_rows    = []

    for fold in data['cv_splits']:
        m_fold = _make_xgb(XGB_PARAMS)
        m_fold.fit(fold['X_train'], fold['y_train'],
                   eval_set=[(fold['X_cv'], fold['y_cv'])], verbose=False)
        y_cv = m_fold.predict(fold['X_cv'])
        cv_metrics_list.append(calculate_metrics(fold['y_cv'], y_cv, fold['close_cv']))
        for i in range(len(fold['dates_cv'])):
            cv_pred_rows.append({
                'Split':             f"cv_fold{fold['fold']}",
                'Date':              fold['dates_cv'][i],
                'Close_Anchor':      fold['close_cv'][i],
                'Actual_LogReturn':  fold['y_cv'][i],
                'Actual_Price':      fold['close_cv'][i] * np.exp(np.clip(fold['y_cv'][i], -10, 10)),
                'xgboost_LogReturn': y_cv[i],
                'xgboost_Price':     fold['close_cv'][i] * np.exp(np.clip(y_cv[i], -10, 10)),
            })

    cv_metrics = {k: round(float(np.mean([m[k] for m in cv_metrics_list])), 4)
                  for k in cv_metrics_list[0]}

    # ── Final model: all pre-test → predict test ──────────────────────────────
    last_fold   = data['cv_splits'][-1]
    model_final = _make_xgb(XGB_PARAMS)
    model_final.fit(data['X_pre'], data['y_pre'],
                    eval_set=[(last_fold['X_cv'], last_fold['y_cv'])], verbose=100)
    y_pred_test  = model_final.predict(data['X_test']) if len(data['X_test']) else np.array([])
    test_metrics = (calculate_metrics(data['y_test'], y_pred_test, data['close_test'])
                    if len(data['y_test']) else {})

    elapsed = time.time() - t0
    print(f"\n  xgboost BASE  [{elapsed:.1f}s]")
    m = cv_metrics
    print(f"    CV    MAPE={m['MAPE']:.2f}%  RMSE={m['RMSE']:.2f}  "
          f"R²(price)={m['R2_Price']:.4f}  R²(lr)={m['R2_LogReturn']:.4f}  "
          f"DirAcc={m['Directional_Accuracy']:.2f}%  (avg {CV_FOLDS} folds)")
    if test_metrics:
        m = test_metrics
        print(f"    TEST  MAPE={m['MAPE']:.2f}%  RMSE={m['RMSE']:.2f}  "
              f"R²(price)={m['R2_Price']:.4f}  R²(lr)={m['R2_LogReturn']:.4f}  "
              f"DirAcc={m['Directional_Accuracy']:.2f}%")

    # ── CSA optimisation ──────────────────────────────────────────────────────
    csa_test_metrics = None
    csa_best_params  = None
    csa_pred_test    = np.array([])
    if csa_config and csa_config.get('enabled', False):
        t0 = time.time()
        print(f"\n  xgboost CSA  (pop={csa_config['population_size']}  "
              f"iter={csa_config['max_iterations']}  cv={csa_config['cv_folds']}) …")
        obj_fn      = csa_objective_sklearn('xgboost', data['X_pre'],
                                            data['y_pre'], csa_config['cv_folds'])
        csa_result  = run_csa('xgboost', obj_fn,
                              csa_config['population_size'], csa_config['max_iterations'])
        csa_best_params = csa_result.best_params

        model_csa_final = _make_xgb(csa_best_params)
        model_csa_final.fit(data['X_pre'], data['y_pre'],
                            eval_set=[(last_fold['X_cv'], last_fold['y_cv'])], verbose=100)
        csa_pred_test    = (model_csa_final.predict(data['X_test'])
                            if len(data['X_test']) else np.array([]))
        csa_test_metrics = (calculate_metrics(data['y_test'], csa_pred_test, data['close_test'])
                            if len(data['y_test']) else {})
        if csa_test_metrics:
            mc = csa_test_metrics
            print(f"    TEST  MAPE={mc['MAPE']:.2f}%  RMSE={mc['RMSE']:.2f}  "
                  f"R²(price)={mc['R2_Price']:.4f}  R²(lr)={mc['R2_LogReturn']:.4f}  "
                  f"DirAcc={mc['Directional_Accuracy']:.2f}%  [{time.time()-t0:.1f}s]")

    # ── Outputs ───────────────────────────────────────────────────────────────
    out_dir = output_dir / f'horizon_{horizon}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(data['dates_test']):
        _save_predictions('testing', data['dates_test'], data['y_test'],
                          data['close_test'], y_pred_test, out_dir, horizon)
    _save_results('cv',      cv_metrics,   out_dir, horizon)
    if test_metrics:
        _save_results('testing', test_metrics, out_dir, horizon)

    if csa_test_metrics:
        if len(data['dates_test']):
            _save_predictions('testing', data['dates_test'], data['y_test'],
                              data['close_test'], csa_pred_test, out_dir, horizon, 'CSA')
        _save_results('testing', csa_test_metrics, out_dir, horizon, 'CSA')

    _build_combined_predictions(data, cv_pred_rows, y_pred_test, out_dir, horizon)

    if SAVE_DATASET:
        _save_dataset('pretrain', data['dates_pre'], data['X_pre_raw'],
                      data['y_pre'], data['close_pre'], feature_cols, out_dir, horizon)
        if len(data['dates_test']):
            _save_dataset('test', data['dates_test'], data['X_test_raw'],
                          data['y_test'], data['close_test'], feature_cols, out_dir, horizon)

    _save_feature_importance(model_final, feature_cols, out_dir, horizon, tag='BASE')
    if csa_best_params is not None:
        _save_feature_importance(model_csa_final, feature_cols, out_dir, horizon, tag='CSA')

    params_data = {
        'horizon':        horizon,
        'timestamp':      pd.Timestamp.now().isoformat(),
        'n_features':     len(feature_cols),
        'features':       feature_cols,
        'n_pre_test':     int(len(data['X_pre'])),
        'n_test':         int(len(data['X_test'])),
        'n_cv_folds':     CV_FOLDS,
        'test_cutoff':    str(val_cutoff.date()),
        'xgb_params_base': {k: (v.__name__ if callable(v) else v) for k, v in XGB_PARAMS.items()},
        'xgb_params_csa':  ({k: (v.__name__ if callable(v) else v) for k, v in csa_best_params.items()}
                            if csa_best_params else None),
        'config': {
            'USE_HMM': USE_HMM, 'USE_SENTIMENT': USE_SENTIMENT,
            'USE_CPO_VARS': USE_CPO_VARS,
            'ABLATION': ABLATION,
        },
    }
    with open(out_dir / f'params_Daily_h{horizon}.json', 'w') as fh:
        json.dump(params_data, fh, indent=2)

    artifacts_dir = _HERE / 'saved_models' / OUTPUT_TAG / f'h{horizon}'
    save_model_artifacts(
        model=model_final, model_type='xgboost', scaler=data['scaler'],
        feature_cols=feature_cols, params=XGB_PARAMS,
        save_dir=str(artifacts_dir / 'xgboost_base'),
    )
    if csa_best_params is not None:
        save_model_artifacts(
            model=model_csa_final, model_type='xgboost', scaler=data['scaler'],
            feature_cols=feature_cols, params=csa_best_params,
            save_dir=str(artifacts_dir / 'xgboost_csa'),
        )

    if len(data['dates_test']):
        _plot_overlay(data, y_pred_test, 'Testing (2026+)', out_dir, horizon, split='test')

    print(f"  Saved → {out_dir}")
    return {'cv': cv_metrics, 'test': test_metrics, 'test_csa': csa_test_metrics}


def _build_combined_predictions(data: dict, cv_pred_rows: list, y_pred_test,
                                 out_dir: Path, horizon: int):
    """Combined predictions CSV: walk-forward CV rows (pre-2026) + test rows (2026+)."""
    rows = list(cv_pred_rows)
    if len(data['dates_test']):
        for i in range(len(data['dates_test'])):
            rows.append({
                'Split':             'test',
                'Date':              data['dates_test'][i],
                'Close_Anchor':      data['close_test'][i],
                'Actual_LogReturn':  data['y_test'][i],
                'Actual_Price':      data['close_test'][i] * np.exp(np.clip(data['y_test'][i], -10, 10)),
                'xgboost_LogReturn': y_pred_test[i],
                'xgboost_Price':     data['close_test'][i] * np.exp(np.clip(y_pred_test[i], -10, 10)),
            })
    pd.DataFrame(rows).to_csv(out_dir / f'predictions_Daily_h{horizon}.csv', index=False)


def _save_feature_importance(model: XGBRegressor, feature_cols: list[str],
                              out_dir: Path, horizon: int, top_n: int = 20,
                              tag: str = 'BASE'):
    """Save feature importance (gain + weight) CSV and bar chart for BASE or CSA model."""
    booster = model.get_booster()

    # When the model is trained on a numpy array XGBoost internally names features
    # f0, f1, f2, ... — remap to the actual column names before looking up scores.
    gain_raw   = booster.get_score(importance_type='gain')
    weight_raw = booster.get_score(importance_type='weight')
    gain   = {feature_cols[int(k[1:])]: v for k, v in gain_raw.items()}
    weight = {feature_cols[int(k[1:])]: v for k, v in weight_raw.items()}

    rows = []
    for feat in feature_cols:
        rows.append({
            'Feature':           feat,
            'Importance_Gain':   gain.get(feat, 0.0),
            'Importance_Weight': weight.get(feat, 0.0),
        })

    df_imp = (pd.DataFrame(rows)
              .sort_values('Importance_Gain', ascending=False)
              .reset_index(drop=True))
    df_imp.insert(0, 'Rank', df_imp.index + 1)

    tag_lower = tag.lower()
    df_imp.to_csv(out_dir / f'feature_importance_{tag_lower}_h{horizon}.csv', index=False)

    # Plot top-N by gain
    top = df_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    ax.barh(top['Feature'][::-1], top['Importance_Gain'][::-1], color='#2E86AB')
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'Feature Importance ({tag}) — Horizon {horizon} (top {len(top)} by gain)',
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


def _plot_overlay(data: dict, y_pred, title_tag: str, out_dir: Path,
                  horizon: int, split: str = 'test'):
    dates   = data[f'dates_{split}']
    y_true  = data[f'y_{split}']
    close   = data[f'close_{split}']
    actual  = close * np.exp(np.clip(y_true,  -10, 10))
    pred    = close * np.exp(np.clip(y_pred,  -10, 10))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(dates, actual, label='Actual',   color='black',   linewidth=2)
    ax.plot(dates, pred,   label='XGBoost',  color='#2E86AB', linewidth=1.2, linestyle='--')
    ax.set_title(f'Daily Forecast ({title_tag}) — Horizon {horizon}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('CPO Price (MYR/tonne)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    tag = 'validation_overlay' if split == 'val' else 'overlay'
    fig.savefig(out_dir / f'{tag}_Daily_h{horizon}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Cross-horizon summary
# =============================================================================

def generate_summary(all_metrics: dict[int, dict], output_dir: Path):
    """Summary CSVs and cross-horizon metric trend plots."""
    for split_tag in ('cv', 'test'):
        rows = []
        for h, m in sorted(all_metrics.items()):
            if m.get(split_tag):
                rows.append({'Horizon': h, 'Model': 'xgboost', 'Optimization': 'BASE',
                             **m[split_tag]})
            if split_tag == 'test' and m.get('test_csa'):
                rows.append({'Horizon': h, 'Model': 'xgboost', 'Optimization': 'CSA',
                             **m['test_csa']})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        label = 'cv' if split_tag == 'cv' else 'testing'
        df.to_csv(output_dir / f'horizon_summary_Daily_{label}.csv', index=False)

        for metric in ['RMSE', 'MAPE', 'R2_Price']:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Horizon'], df[metric], marker='o', color='#2E86AB', linewidth=1.5)
            ax.set_title(f'Daily — {metric} Across Horizons ({label.title()})',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Forecast Horizon')
            ax.set_ylabel(metric)
            ax.set_xticks(sorted(df['Horizon'].unique()))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f'{metric.lower().replace("2_", "2")}_across_horizons_Daily_{label}.png'
            fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"\n  {label.title()} summary:")
        print(df.to_string(index=False))
        print(f"  Saved → {output_dir / f'horizon_summary_Daily_{label}.csv'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Horizon CPO Forecasting — Configurable')
    parser.add_argument('--horizons', type=str, default='',
                        help='Comma-separated horizons, e.g. "1,3,5" (default: all in CONFIG)')
    parser.add_argument('--no-csa', action='store_true', help='Disable CSA optimisation')
    parser.add_argument('--csa-population', type=int, default=CSA_POPULATION)
    parser.add_argument('--csa-iterations', type=int, default=CSA_ITERATIONS)
    parser.add_argument('--csa-cv-folds',   type=int, default=CSA_CV_FOLDS)
    args = parser.parse_args()

    horizons = HORIZONS
    if args.horizons:
        horizons = [int(x.strip()) for x in args.horizons.split(',') if x.strip()]

    csa_config = {
        'enabled':         USE_CSA and not args.no_csa,
        'population_size': args.csa_population,
        'max_iterations':  args.csa_iterations,
        'cv_folds':        args.csa_cv_folds,
    }

    val_cutoff = pd.Timestamp(VAL_CUTOFF)
    output_dir = _HERE / f'output_horizons_{OUTPUT_TAG}' / 'Daily'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('#' * 70)
    print(f'  Multi-Horizon CPO Forecast — {OUTPUT_TAG.upper()}')
    print(f'  Horizons   : {horizons}')
    print(f'  Sources    : HMM={USE_HMM}  Sentiment={USE_SENTIMENT}  CPO_Vars={USE_CPO_VARS}')
    print(f'  Test cutoff: {VAL_CUTOFF}  |  CV folds: {CV_FOLDS}')
    print(f'  XGB params : {XGB_PARAMS}')
    print(f'  CSA        : {"enabled" if csa_config["enabled"] else "disabled"}  '
          f'(pop={csa_config["population_size"]}  iter={csa_config["max_iterations"]}  '
          f'cv={csa_config["cv_folds"]})')
    print('#' * 70)

    merged = load_and_merge()

    all_metrics: dict[int, dict] = {}
    t_start = time.time()
    for h in horizons:
        all_metrics[h] = run_horizon(h, merged, output_dir, val_cutoff, csa_config)

    if len(horizons) > 1:
        generate_summary(all_metrics, output_dir)

    print(f"\n{'='*70}")
    print(f'  DONE  ({time.time()-t_start:.1f}s)')
    print(f'  Output: {output_dir}')
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
