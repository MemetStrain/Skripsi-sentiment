"""
csa_overfit_check.py — diagnose whether CSA's headline DA is real skill or noise.

Three diagnostics for one (tag, horizon) pair at a time:

1. Sliding-window DA       — rolling DA across the held-out test segment
                              (stability check; high std → DA is driven by a
                              few lucky days, not a learned signal)

2. Permutation test on DA  — shuffle the sign of actual log-returns N times,
                              recompute DA → null distribution. Report p-value
                              for the observed test-set DA. p > 0.05 means the
                              observed DA is not distinguishable from chance.

3. Walk-forward retrain    — slide the train/test cutoff forward in steps,
                              retrain at each step, score the next `step`
                              points. Reports DA distribution across folds.
                              Heaviest diagnostic; off by default.

Usage:
    python prediction/csa_overfit_check.py --tag cpo_hmm --horizon 6
    python prediction/csa_overfit_check.py --tag cpo_hmm --horizon 6 --walkforward
    python prediction/csa_overfit_check.py --tag full --horizon 1 \\
        --window 30 --n-perm 5000

If saved CSA artefacts (model.pkl + scaler.pkl) exist, they are loaded.
Otherwise, the script trains a fresh CSA model using the hyperparameters
recorded in meta.json (which is checked into the repo).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.forecast_utils import (  # noqa: E402
    VAL_CUTOFF, MODELS_DIR, BASE_PARAMS, RANDOM_STATE, create_sklearn_model,
)

TAG_TO_MODULE: Dict[str, str] = {
    'cpo_only':      'horizon_forecast_C1_price_only',
    'cpo_hmm':       'horizon_forecast_C2_price_hmm',
    'cpo_sentiment': 'horizon_forecast_C3_price_sentiment',
    'full':          'horizon_forecast_C4_full',
}


# ---------------------------------------------------------------------------
# Data loading via the same helpers the training scripts use
# ---------------------------------------------------------------------------

def load_dataset(tag: str, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    """Reuse the C-script's loader + horizon feature engineer for this tag."""
    mod = importlib.import_module(TAG_TO_MODULE[tag])
    merged = mod.load_and_merge_data('Daily')
    df, feature_cols = mod.engineer_features_for_horizon(merged, horizon)
    return df, feature_cols


def split_pre_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pre  = df[df['Date'] <  VAL_CUTOFF].reset_index(drop=True)
    test = df[df['Date'] >= VAL_CUTOFF].reset_index(drop=True)
    return pre, test


# ---------------------------------------------------------------------------
# Model: load saved CSA, or retrain from meta.json params
# ---------------------------------------------------------------------------

def _saved_dir(tag: str, horizon: int, variant: str) -> str:
    return os.path.join(MODELS_DIR, tag, 'Daily', f'h{horizon}', f'xgboost_{variant}')


def load_or_train(tag: str, horizon: int, variant: str,
                  X_pre: np.ndarray, y_pre: np.ndarray
                  ) -> Tuple[object, object]:
    """Return (model, scaler). Loads from disk if present, else retrains."""
    base = _saved_dir(tag, horizon, variant)
    model_path  = os.path.join(base, 'model.pkl')
    scaler_path = os.path.join(base, 'scaler.pkl')
    meta_path   = os.path.join(base, 'meta.json')

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'meta.json missing at {meta_path}')

    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f'  [retrain] {variant} — no .pkl found, refitting from meta.json params')
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X_pre)
    params = dict(meta.get('params', {}))
    if variant == 'base' and not params:
        params = dict(BASE_PARAMS['xgboost'])
    # No eval_set in this diagnostic path → drop early-stopping plumbing.
    for k in ('early_stopping_rounds', 'verbose'):
        params.pop(k, None)
    params.setdefault('random_state', RANDOM_STATE)
    model = create_sklearn_model('xgboost', params)
    model.fit(Xs, y_pre, verbose=False)
    return model, scaler


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _da(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy on log-returns: sign(y_true) == sign(y_pred)."""
    mask = (y_true != 0)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])) * 100)


def sliding_window_da(y_true: np.ndarray, y_pred: np.ndarray,
                      window: int, step: int = 1) -> pd.DataFrame:
    """DA on overlapping windows of `window` points, stepping by `step`."""
    rows = []
    for start in range(0, len(y_true) - window + 1, step):
        s = slice(start, start + window)
        rows.append({
            'start':   start,
            'end':     start + window,
            'da':      _da(y_true[s], y_pred[s]),
            'n':       window,
        })
    return pd.DataFrame(rows)


def permutation_test_da(y_true: np.ndarray, y_pred: np.ndarray,
                        n_perm: int, rng: np.random.Generator
                        ) -> Tuple[float, float, np.ndarray]:
    """
    Null hypothesis: predictions have no directional skill.
    Permute the *signs* of y_true (preserving magnitudes) and recompute DA.
    Returns (observed_da, p_value, null_distribution).
    """
    obs = _da(y_true, y_pred)
    pred_sign = np.sign(y_pred)
    null = np.empty(n_perm)
    n = len(y_true)
    for i in range(n_perm):
        perm_signs = rng.choice([-1.0, 1.0], size=n)
        shuffled_true = perm_signs * np.abs(y_true)
        null[i] = _da(shuffled_true, y_pred)
    p = float(np.mean(null >= obs))
    return obs, p, null


def walk_forward_retrain(df: pd.DataFrame, feature_cols: List[str],
                         tag: str, horizon: int,
                         step: int, min_train: int) -> pd.DataFrame:
    """
    Expanding-window walk-forward. At each cutoff t:
      - fit CSA-params model on rows [0:t]
      - predict rows [t:t+step]
      - score DA on that block
    Cutoffs slide by `step` from `min_train` to len(df) - step.
    """
    base = _saved_dir(tag, horizon, 'csa')
    with open(os.path.join(base, 'meta.json'), encoding='utf-8') as f:
        params = dict(json.load(f).get('params', {}))
    for k in ('early_stopping_rounds', 'verbose'):
        params.pop(k, None)

    X_all = df[feature_cols].values
    y_all = df['Target'].values
    dates = pd.to_datetime(df['Date']).values

    rows = []
    cutoffs = list(range(min_train, len(df) - step + 1, step))
    for c in cutoffs:
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_all[:c])
        X_te = scaler.transform(X_all[c:c + step])
        y_tr = y_all[:c]
        y_te = y_all[c:c + step]

        model = create_sklearn_model('xgboost', params)
        model.fit(X_tr, y_tr, verbose=False)
        y_hat = model.predict(X_te)

        rows.append({
            'cutoff_date': pd.Timestamp(dates[c - 1]).date().isoformat(),
            'test_start':  pd.Timestamp(dates[c]).date().isoformat(),
            'test_end':    pd.Timestamp(dates[c + step - 1]).date().isoformat(),
            'da':          _da(y_te, y_hat),
            'mape':        float(np.mean(np.abs((y_te - y_hat) / (np.abs(y_te) + 1e-9))) * 100),
            'n':           step,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, choices=list(TAG_TO_MODULE))
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--variant', default='csa', choices=['csa', 'base'],
                    help='Which trained variant to diagnose.')
    ap.add_argument('--window', type=int, default=30,
                    help='Sliding-window size in test-set days.')
    ap.add_argument('--n-perm', type=int, default=2000,
                    help='Permutation iterations for the DA null distribution.')
    ap.add_argument('--walkforward', action='store_true',
                    help='Run the walk-forward retrain diagnostic (slow).')
    ap.add_argument('--wf-step', type=int, default=20,
                    help='Walk-forward window size and stride.')
    ap.add_argument('--wf-min-train', type=int, default=0,
                    help='Minimum training rows before the first walk-forward '
                         'cutoff. 0 → use the pre-test segment length.')
    ap.add_argument('--seed', type=int, default=RANDOM_STATE)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'output_diagnostics'))
    args = ap.parse_args()

    print(f'\n=== CSA overfit check: tag={args.tag}  h={args.horizon}  '
          f'variant={args.variant} ===')

    df, feature_cols = load_dataset(args.tag, args.horizon)
    pre, test = split_pre_test(df)
    print(f'pre-test rows : {len(pre):>4}  '
          f'({pre.Date.min().date()} → {pre.Date.max().date()})')
    print(f'test rows     : {len(test):>4}  '
          f'({test.Date.min().date() if len(test) else "—"} → '
          f'{test.Date.max().date() if len(test) else "—"})')
    print(f'features      : {len(feature_cols)}')

    if len(test) == 0:
        raise SystemExit('No test rows (Date >= VAL_CUTOFF). Cannot diagnose.')

    X_pre  = pre[feature_cols].values
    y_pre  = pre['Target'].values
    X_test = test[feature_cols].values
    y_test = test['Target'].values

    model, scaler = load_or_train(args.tag, args.horizon, args.variant,
                                  X_pre, y_pre)
    y_pred = model.predict(scaler.transform(X_test))

    obs_da = _da(y_test, y_pred)
    print(f'\n[1] Test-set DA (single value): {obs_da:.2f}%  on n={len(y_test)}')

    # --- sliding window -----------------------------------------------------
    if args.window > len(y_test):
        print(f'[2] Sliding window: skipped — window {args.window} > '
              f'test length {len(y_test)}')
        slide = pd.DataFrame()
    else:
        slide = sliding_window_da(y_test, y_pred, args.window)
        print(f'\n[2] Sliding-window DA (window={args.window}, '
              f'n_windows={len(slide)}):')
        print(f'    mean   = {slide.da.mean():.2f}%')
        print(f'    std    = {slide.da.std():.2f} pp')
        print(f'    min    = {slide.da.min():.2f}%')
        print(f'    max    = {slide.da.max():.2f}%')
        below_50 = (slide.da < 50).mean() * 100
        print(f'    fraction of windows below 50%: {below_50:.1f}%')

    # --- permutation --------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    obs, p, null = permutation_test_da(y_test, y_pred, args.n_perm, rng)
    print(f'\n[3] Permutation test ({args.n_perm} iters):')
    print(f'    observed DA = {obs:.2f}%')
    print(f'    null mean   = {null.mean():.2f}%   null std = {null.std():.2f} pp')
    print(f'    p-value     = {p:.4f}   '
          f'({"NOT significant" if p > 0.05 else "significant"} at α=0.05)')

    # --- walk-forward retrain ----------------------------------------------
    wf = pd.DataFrame()
    if args.walkforward:
        min_train = args.wf_min_train or len(pre)
        print(f'\n[4] Walk-forward retrain '
              f'(step={args.wf_step}, min_train={min_train}) — this is slow')
        wf = walk_forward_retrain(df, feature_cols,
                                  args.tag, args.horizon,
                                  step=args.wf_step, min_train=min_train)
        if len(wf) == 0:
            print('    no folds produced — try a smaller --wf-step or '
                  'lower --wf-min-train')
        else:
            print(f'    folds  = {len(wf)}')
            print(f'    DA mean= {wf.da.mean():.2f}%   std={wf.da.std():.2f} pp')
            print(f'    DA min = {wf.da.min():.2f}%   max={wf.da.max():.2f}%')
            print(f'    fraction of folds below 50%: '
                  f'{(wf.da < 50).mean() * 100:.1f}%')

    # --- write artefacts ----------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f'{args.tag}_h{args.horizon}_{args.variant}'

    pd.DataFrame({
        'date':        pd.to_datetime(test['Date']).dt.date.astype(str),
        'y_true':      y_test,
        'y_pred':      y_pred,
        'sign_match':  (np.sign(y_test) == np.sign(y_pred)).astype(int),
    }).to_csv(os.path.join(args.out_dir, f'{stem}_test_predictions.csv'),
              index=False)

    if len(slide):
        slide.to_csv(os.path.join(args.out_dir, f'{stem}_sliding_window.csv'),
                     index=False)
    pd.DataFrame({'null_da': null}).to_csv(
        os.path.join(args.out_dir, f'{stem}_permutation_null.csv'), index=False)
    if len(wf):
        wf.to_csv(os.path.join(args.out_dir, f'{stem}_walkforward.csv'),
                  index=False)

    summary = {
        'tag':            args.tag,
        'horizon':        args.horizon,
        'variant':        args.variant,
        'n_test':         int(len(y_test)),
        'observed_da':    obs_da,
        'sliding_mean':   float(slide.da.mean()) if len(slide) else None,
        'sliding_std':    float(slide.da.std())  if len(slide) else None,
        'sliding_below50_pct': float((slide.da < 50).mean() * 100) if len(slide) else None,
        'permutation_p':  p,
        'permutation_null_mean': float(null.mean()),
        'walkforward_mean': float(wf.da.mean()) if len(wf) else None,
        'walkforward_std':  float(wf.da.std())  if len(wf) else None,
        'walkforward_below50_pct': float((wf.da < 50).mean() * 100) if len(wf) else None,
    }
    with open(os.path.join(args.out_dir, f'{stem}_summary.json'),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nArtefacts written to {args.out_dir}/{stem}_*.csv|json')


if __name__ == '__main__':
    main()
