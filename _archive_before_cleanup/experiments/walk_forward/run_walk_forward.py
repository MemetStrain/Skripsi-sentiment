"""
Walk-Forward Backtesting Evaluation
=====================================
Predicts CPO prices for Jan / Feb / Mar 2026 using two lead-time scenarios:
  - 1-month ahead : train cutoff = last day of prior month
  - 2-months ahead: train cutoff = last day of 2 months prior

Model hyperparameters are loaded from the saved params file (no re-optimization).
Data sources: CPO price variables + sentiment + HMM states (same as the main pipeline).

Usage:
    cd d:\\Skripsi1
    python experiments/walk_forward/run_walk_forward.py
"""

import os
import sys
import time

# -----------------------------------------------------------------------
# sys.path bootstrap — must come BEFORE any project imports
# -----------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT   = os.path.dirname(os.path.dirname(_HERE))
_PRED_DIR    = os.path.join(_PROJ_ROOT, 'prediction')
_UTILS_DIR   = os.path.join(_PRED_DIR, 'utils')

for _p in [_HERE, _PRED_DIR, _UTILS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -----------------------------------------------------------------------
# Project imports (after path setup)
# -----------------------------------------------------------------------
import pandas as pd

from config          import EXPERIMENT_GRID, MODEL_VARIANTS, OUTPUT_DIR
from data_loader     import load_full_dataset
from feature_builder import build_features, prepare_arrays
from model_runner    import load_saved_params, run_single_variant
from metrics_calculator import compute_metrics
from output_writer   import save_experiment_outputs, save_summary_table


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Walk-Forward CPO Price Prediction Evaluation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data and params once — reused across all experiments
    full_df      = load_full_dataset()
    saved_params = load_saved_params()

    # Feature engineering on the full dataset (lag boundary safety)
    featured_df, feature_cols = build_features(full_df)

    all_metrics_rows = []
    completed = 0

    for exp in EXPERIMENT_GRID:
        print(f"\n{'─'*60}")
        print(f"  Experiment: {exp['id']}")
        print(f"  Target: {exp['target_start']} → {exp['target_end']}")
        print(f"  Train cutoff: {exp['train_cutoff']}  (lead={exp['lead']} month(s))")
        print(f"{'─'*60}")

        train_cutoff  = pd.Timestamp(exp['train_cutoff'])
        target_start  = pd.Timestamp(exp['target_start'])
        target_end    = pd.Timestamp(exp['target_end'])

        # --- Check target availability ---
        target_rows = featured_df[
            (featured_df['Date'] >= target_start) &
            (featured_df['Date'] <= target_end)
        ]
        if len(target_rows) == 0:
            print(f"  SKIP — no target rows in dataset (data ends before {exp['target_start']})")
            continue

        # --- Prepare arrays ---
        arrays = prepare_arrays(
            featured_df, feature_cols,
            train_cutoff=train_cutoff,
            target_start=target_start,
            target_end=target_end,
        )
        print(f"  Train rows: {arrays['n_train']} | Target rows: {arrays['n_target']}")

        if arrays['n_train'] < 50:
            print(f"  SKIP — insufficient training rows ({arrays['n_train']})")
            continue

        # --- Run all 12 model variants ---
        predictions  = {}
        metrics_rows = []

        for variant_key in MODEL_VARIANTS:
            params = saved_params.get(variant_key)
            if params is None:
                print(f"  SKIP variant {variant_key} — not found in params file")
                continue

            print(f"  [{variant_key:30s}] ", end='', flush=True)
            t_v = time.time()
            y_pred = run_single_variant(variant_key, params, arrays)
            elapsed = time.time() - t_v

            n_valid = int((~__import__('numpy').isnan(y_pred)).sum())
            print(f"done ({elapsed:.1f}s)  valid={n_valid}/{arrays['n_target']}")

            predictions[variant_key] = y_pred
            row = compute_metrics(
                arrays['y_target'], y_pred, arrays['close_target'],
                variant_key, exp, arrays['n_train'],
            )
            metrics_rows.append(row)

        # --- Save experiment outputs ---
        save_experiment_outputs(
            exp,
            arrays['target_dates'],
            arrays['close_target'],
            arrays['y_target'],
            predictions,
            metrics_rows,
        )
        all_metrics_rows.extend(metrics_rows)
        completed += 1

        # Print best MAPE for this experiment
        valid_mapes = [r['MAPE'] for r in metrics_rows if r['MAPE'] < 1e9]
        if valid_mapes:
            best_mape    = min(valid_mapes)
            best_variant = metrics_rows[valid_mapes.index(best_mape)]['model_variant']
            print(f"  Best MAPE: {best_mape:.4f}% ({best_variant})")

    # --- Global summary ---
    print(f"\n{'='*60}")
    print(f"  Completed {completed}/{len(EXPERIMENT_GRID)} experiments")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print('='*60)

    save_summary_table(all_metrics_rows)


if __name__ == '__main__':
    main()
