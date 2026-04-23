"""
Save per-experiment predictions, metrics, comparison plots, and the global summary table.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List

from config import MODEL_VARIANTS, OUTPUT_DIR

# -----------------------------------------------------------------------
# Visual style: one colour family per model type, one line style per opt
# -----------------------------------------------------------------------
_COLORS = {
    'xgboost':       '#2196F3',   # blue
    'random_forest': '#9C27B0',   # purple
    'arimax':        '#FF9800',   # orange
    'sarimax':       '#009688',   # teal
}
_LINESTYLES = {
    'base':     '--',
    'csa':      '-',
    'bayesian': '-.',
}
_ALPHAS = {'base': 0.55, 'csa': 0.90, 'bayesian': 0.75}


def _variant_style(variant_key: str):
    for fam in ('xgboost', 'random_forest', 'arimax', 'sarimax'):
        if variant_key.startswith(fam):
            suffix = variant_key[len(fam) + 1:]
            color = _COLORS[fam]
            ls    = _LINESTYLES.get(suffix, '-')
            alpha = _ALPHAS.get(suffix, 0.7)
            return color, ls, alpha
    return 'grey', '-', 0.7


def save_experiment_outputs(
    exp: Dict,
    target_dates: np.ndarray,
    close_anchor: np.ndarray,
    y_true_lr: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics_rows: List[Dict],
) -> None:
    """
    Write per-experiment artefacts:
      output/{exp_id}/predictions.csv
      output/{exp_id}/metrics.csv
      output/{exp_id}/comparison_plot.png
    """
    exp_dir = os.path.join(OUTPUT_DIR, exp['id'])
    os.makedirs(exp_dir, exist_ok=True)

    # --- predictions.csv ---
    dates_series = pd.to_datetime(target_dates)
    pred_df = pd.DataFrame({
        'Date':             dates_series,
        'Close_Anchor':     close_anchor,
        'Actual_LogReturn': y_true_lr,
        'Actual_Price':     close_anchor * np.exp(np.clip(y_true_lr, -10, 10)),
    })
    for variant_key, preds in predictions.items():
        pred_df[f'{variant_key}_LogReturn'] = preds
        safe_preds = np.where(np.isnan(preds), np.nan, np.clip(preds, -10, 10))
        pred_df[f'{variant_key}_Price'] = close_anchor * np.exp(safe_preds)
    pred_df.to_csv(os.path.join(exp_dir, 'predictions.csv'), index=False)

    # --- metrics.csv ---
    pd.DataFrame(metrics_rows).to_csv(os.path.join(exp_dir, 'metrics.csv'), index=False)

    # --- comparison_plot.png ---
    _make_comparison_plot(exp, dates_series, pred_df, exp_dir)

    print(f"  Saved → {exp_dir}")


def _make_comparison_plot(exp: Dict, dates, pred_df: pd.DataFrame, exp_dir: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(dates, pred_df['Actual_Price'], color='black', linewidth=2,
                label='Actual Price', zorder=10)

        for variant_key in MODEL_VARIANTS:
            col = f'{variant_key}_Price'
            if col not in pred_df.columns:
                continue
            color, ls, alpha = _variant_style(variant_key)
            ax.plot(dates, pred_df[col], color=color, linestyle=ls, alpha=alpha,
                    linewidth=1.2, label=variant_key)

        ax.set_title(
            f"Walk-Forward Prediction: {exp['id']}\n"
            f"Lead {exp['lead']} month(s) — Train cutoff: {exp['train_cutoff']}",
            fontsize=11,
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('CPO Price (MYR/tonne)')
        ax.legend(loc='upper left', fontsize=6, ncol=2, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(exp_dir, 'comparison_plot.png'), dpi=120)
        plt.close(fig)


def save_summary_table(all_metrics_rows: List[Dict]) -> None:
    """Write summary_all_experiments.csv and print a MAPE pivot table."""
    if not all_metrics_rows:
        print("No completed experiments — summary table not written.")
        return

    summary_df = pd.DataFrame(all_metrics_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'summary_all_experiments.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")

    # Pivot: rows = experiment, cols = model_variant, values = MAPE
    metric_cols = ['MAPE', 'sMAPE', 'RMSE', 'Directional_Accuracy', 'R2_Price']
    for metric in metric_cols:
        if metric not in summary_df.columns:
            continue
        pivot = summary_df.pivot_table(
            index='experiment_id', columns='model_variant',
            values=metric, aggfunc='first',
        )
        print(f"\n{'='*60}")
        print(f"  {metric} by experiment × model variant")
        print('='*60)
        with pd.option_context('display.float_format', '{:.4f}'.format,
                               'display.max_columns', 20, 'display.width', 200):
            print(pivot.to_string())
