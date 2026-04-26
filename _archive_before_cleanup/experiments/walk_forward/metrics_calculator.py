"""
Wraps forecast_utils.calculate_metrics and adds experiment metadata.
"""
import numpy as np
from typing import Dict

from forecast_utils import calculate_metrics


def compute_metrics(
    y_true_lr: np.ndarray,
    y_pred_lr: np.ndarray,
    close_anchor: np.ndarray,
    variant_key: str,
    exp: Dict,
    n_train: int,
) -> Dict:
    """
    Compute price-space metrics and annotate with experiment metadata.

    Parameters
    ----------
    y_true_lr    : actual log returns for the target month
    y_pred_lr    : predicted log returns from the model variant
    close_anchor : Close price at each prediction row (t, not t+h)
    variant_key  : e.g. 'xgboost_csa'
    exp          : experiment config dict from EXPERIMENT_GRID
    n_train      : number of training rows used

    Returns
    -------
    Flat dict suitable for a pandas DataFrame row.
    """
    base = calculate_metrics(y_true_lr, y_pred_lr, close_anchor)

    n_total = len(y_pred_lr)
    n_valid = int(np.sum(~np.isnan(y_pred_lr)))

    return {
        'experiment_id':  exp['id'],
        'lead_months':    exp['lead'],
        'target_month':   exp['target_start'][:7],   # 'YYYY-MM'
        'train_cutoff':   exp['train_cutoff'],
        'model_variant':  variant_key,
        'n_train':        n_train,
        'n_predictions':  n_total,
        'n_valid':        n_valid,
        **base,
    }
