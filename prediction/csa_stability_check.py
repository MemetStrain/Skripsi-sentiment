"""
csa_stability_check.py — re-run CSA N times with different seeds (and optional
VAL_CUTOFF jitter) to test whether the optimizer is finding signal or noise.

If CSA is genuinely learning a hyperparameter region with directional skill:
    * chosen params should be reasonably stable across seeds
    * test-set DA / MAPE / RMSE should not swing wildly

If CSA is overfitting CV-MAPE noise:
    * params jump around the search space
    * test metrics have large coefficient of variation
    * different seeds pick "winners" with very different test DA

Mirrors the offline pipeline from horizon_forecast_C{1..4}.py:
    * pre-test  = Date < VAL_CUTOFF  →  TimeSeriesSplit n folds
    * objective = mean MAPE across CV folds
    * winner    = best CSA params, then refit on full pre-test, score on test

Usage:
    python prediction/csa_stability_check.py --tag cpo_hmm --horizon 6
    python prediction/csa_stability_check.py --tag cpo_hmm --horizon 6 \\
        --n-runs 10 --cutoff-jitter-weeks 4
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.forecast_utils import (  # noqa: E402
    VAL_CUTOFF, RANDOM_STATE,
    CSA_PARAM_SPACES, BASE_PARAMS,
    create_sklearn_model, csa_objective_sklearn,
)
from crow_search_optimizer import CrowSearchOptimizer  # noqa: E402

TAG_TO_MODULE: Dict[str, str] = {
    'cpo_only':      'horizon_forecast_C1_price_only',
    'cpo_hmm':       'horizon_forecast_C2_price_hmm',
    'cpo_sentiment': 'horizon_forecast_C3_price_sentiment',
    'full':          'horizon_forecast_C4_full',
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(tag: str, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    mod = importlib.import_module(TAG_TO_MODULE[tag])
    merged = mod.load_and_merge_data('Daily')
    return mod.engineer_features_for_horizon(merged, horizon)


def split_at(df: pd.DataFrame, cutoff: pd.Timestamp
             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pre  = df[df['Date'] <  cutoff].reset_index(drop=True)
    test = df[df['Date'] >= cutoff].reset_index(drop=True)
    return pre, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _da(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true != 0)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])) * 100)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# One CSA run + final fit
# ---------------------------------------------------------------------------

def run_one(X_pre: np.ndarray, y_pre: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            seed: int, population: int, iterations: int, cv_folds: int
            ) -> Dict:
    """Fit RobustScaler on pre-test, run CSA with `seed`, refit on full pre-test
    with the best params, score on test. Returns metrics + chosen params."""
    scaler = RobustScaler()
    Xp = scaler.fit_transform(X_pre)
    Xt = scaler.transform(X_test)

    obj = csa_objective_sklearn('xgboost', Xp, y_pre, cv_folds)

    optimizer = CrowSearchOptimizer(
        objective_function=obj,
        parameter_specs=CSA_PARAM_SPACES['xgboost'],
        population_size=population,
        max_iterations=iterations,
        awareness_probability=0.1,
        flight_length=2.0,
        early_stopping_patience=10,
        random_state=seed,
        verbose=False,
    )
    t0 = time.time()
    result = optimizer.optimize()
    elapsed = time.time() - t0

    params = dict(result.best_params)
    # Drop early-stopping plumbing — we fit without an eval set here.
    for k in ('early_stopping_rounds', 'verbose'):
        params.pop(k, None)

    model = create_sklearn_model('xgboost', params)
    model.fit(Xp, y_pre, verbose=False)
    y_hat = model.predict(Xt)

    return {
        'seed':           seed,
        'cv_best_score':  float(result.best_score),
        'cv_iterations':  int(result.total_iterations),
        'cv_evals':       int(result.total_evaluations),
        'elapsed_sec':    round(elapsed, 1),
        'test_da':        _da(y_test, y_hat),
        'test_mape':      _mape(y_test, y_hat),
        'test_rmse':      _rmse(y_test, y_hat),
        **{f'param_{k}': v for k, v in result.best_params.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summarize(runs: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in runs.columns:
            continue
        v = runs[c].astype(float)
        mean = v.mean()
        std  = v.std()
        cv   = (std / abs(mean) * 100) if mean else float('nan')
        rows.append({
            'metric': c,
            'mean':   round(mean, 4),
            'std':    round(std, 4),
            'min':    round(v.min(), 4),
            'max':    round(v.max(), 4),
            'cv_pct': round(cv, 2),
        })
    return pd.DataFrame(rows)


def run_one_pair(tag: str, horizon: int, args) -> Dict:
    """Run the stability sweep for one (tag, horizon). Returns a summary dict."""
    print(f'\n=== CSA stability check: tag={tag}  h={horizon}  '
          f'runs={args.n_runs} ===')
    print(f'CSA budget per run: pop={args.population}  iter={args.iterations}  '
          f'cv_folds={args.cv_folds}')

    df, feature_cols = load_dataset(tag, horizon)
    print(f'rows={len(df)}  features={len(feature_cols)}')

    configs: List[Tuple[str, pd.Timestamp, int]] = []
    for i in range(args.n_runs):
        configs.append((f'seed_{i}', VAL_CUTOFF, RANDOM_STATE + i))
    if args.cutoff_jitter_weeks > 0:
        delta = pd.Timedelta(weeks=args.cutoff_jitter_weeks)
        configs.append(('cutoff_minus', VAL_CUTOFF - delta, RANDOM_STATE))
        configs.append(('cutoff_plus',  VAL_CUTOFF + delta, RANDOM_STATE))

    rows: List[Dict] = []
    for label, cutoff, seed in configs:
        pre, test = split_at(df, cutoff)
        if len(test) < 10:
            print(f'\n[{label}] skipped — only {len(test)} test rows for cutoff '
                  f'{cutoff.date()}')
            continue

        print(f'\n[{label}] cutoff={cutoff.date()}  '
              f'pre={len(pre)}  test={len(test)}  seed={seed}')

        X_pre  = pre[feature_cols].values
        y_pre  = pre['Target'].values
        X_test = test[feature_cols].values
        y_test = test['Target'].values

        res = run_one(X_pre, y_pre, X_test, y_test,
                      seed=seed, population=args.population,
                      iterations=args.iterations, cv_folds=args.cv_folds)
        res['label']  = label
        res['cutoff'] = cutoff.date().isoformat()
        res['n_test'] = int(len(test))
        rows.append(res)

        print(f'   cv_score={res["cv_best_score"]:.4f}   '
              f'test DA={res["test_da"]:.2f}%   '
              f'MAPE={res["test_mape"]:.3f}   '
              f'RMSE={res["test_rmse"]:.2f}   '
              f'({res["elapsed_sec"]}s)')

    if not rows:
        return {'tag': tag, 'horizon': horizon, 'error': 'no runs completed'}

    runs = pd.DataFrame(rows)

    metric_cols = ['cv_best_score', 'test_da', 'test_mape', 'test_rmse']
    param_cols = [c for c in runs.columns if c.startswith('param_')]

    print('\n--- test-metric stability across runs ---')
    metric_summary = _summarize(runs, metric_cols)
    print(metric_summary.to_string(index=False))

    print('\n--- chosen-hyperparameter stability across runs ---')
    param_summary = _summarize(runs, param_cols)
    print(param_summary.to_string(index=False))

    da_std    = runs['test_da'].std()
    da_range  = runs['test_da'].max() - runs['test_da'].min()
    mape_cv   = runs['test_mape'].std() / runs['test_mape'].mean() * 100
    print('\n--- verdict heuristics ---')
    print(f'test-DA std       = {da_std:.2f} pp   (>10 pp → unstable)')
    print(f'test-DA range     = {da_range:.2f} pp   (>20 pp → very unstable)')
    print(f'test-MAPE CV%     = {mape_cv:.2f}%')
    if da_std > 10 or da_range > 20:
        verdict = 'overfit'
        print('→ CSA is likely overfitting CV noise. The headline DA is not '
              'reproducible across seeds.')
    elif da_std > 5:
        verdict = 'borderline'
        print('→ borderline — DA shifts noticeably with seed, but not catastrophically.')
    else:
        verdict = 'stable'
        print('→ CSA appears stable across seeds. The chosen region of '
              'hyperparam space is robust.')

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f'{tag}_h{horizon}_stability'
    runs.to_csv(os.path.join(args.out_dir, f'{stem}_runs.csv'), index=False)
    metric_summary.to_csv(os.path.join(args.out_dir, f'{stem}_metric_summary.csv'),
                          index=False)
    param_summary.to_csv(os.path.join(args.out_dir, f'{stem}_param_summary.csv'),
                         index=False)
    summary_payload = {
        'tag':      tag,
        'horizon':  horizon,
        'n_runs':   len(runs),
        'verdict':  verdict,
        'da_std':   float(da_std),
        'da_range': float(da_range),
        'mape_cv_pct': float(mape_cv),
        'da_min':   float(runs['test_da'].min()),
        'da_max':   float(runs['test_da'].max()),
        'da_mean':  float(runs['test_da'].mean()),
    }
    with open(os.path.join(args.out_dir, f'{stem}_summary.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2)
    print(f'\nArtefacts written to {args.out_dir}/{stem}_*')
    return summary_payload


def _load_winner_pairs() -> List[Tuple[str, int]]:
    path = os.path.join(HERE, 'winners.json')
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    return [(tag, int(h)) for h, tag in payload['winners_by_horizon'].items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', choices=list(TAG_TO_MODULE),
                    help='Required unless --all-winners.')
    ap.add_argument('--horizon', type=int,
                    help='Required unless --all-winners.')
    ap.add_argument('--all-winners', action='store_true',
                    help='Sweep every (tag, horizon) pair in winners.json '
                         'and write a combined summary CSV.')
    ap.add_argument('--n-runs', type=int, default=5,
                    help='Number of CSA runs with different seeds.')
    ap.add_argument('--population', type=int, default=25,
                    help='CSA population size (lower than the 50 used in '
                         'production to keep this script tractable).')
    ap.add_argument('--iterations', type=int, default=25,
                    help='CSA max iterations per run.')
    ap.add_argument('--cv-folds', type=int, default=3)
    ap.add_argument('--cutoff-jitter-weeks', type=int, default=0,
                    help='If >0, also run with VAL_CUTOFF shifted by ±W weeks '
                         '(adds 2 extra runs at the same base seed).')
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'output_diagnostics'))
    args = ap.parse_args()

    if args.all_winners:
        pairs = _load_winner_pairs()
        print(f'Sweeping {len(pairs)} winner pair(s) from winners.json')
    else:
        if not args.tag or args.horizon is None:
            ap.error('--tag and --horizon are required unless --all-winners is set.')
        pairs = [(args.tag, args.horizon)]

    summaries: List[Dict] = []
    for tag, horizon in sorted(pairs, key=lambda x: (x[1], x[0])):
        try:
            summaries.append(run_one_pair(tag, horizon, args))
        except Exception as e:  # noqa: BLE001
            print(f'\n[{tag} h{horizon}] FAILED: {e}')
            summaries.append({'tag': tag, 'horizon': horizon, 'error': str(e)})

    if args.all_winners:
        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, 'stability_check_all_winners.csv')
        pd.DataFrame(summaries).to_csv(out_csv, index=False)
        print(f'\nCombined summary: {out_csv}')


if __name__ == '__main__':
    main()
