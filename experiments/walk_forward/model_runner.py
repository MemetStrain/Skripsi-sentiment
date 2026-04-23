"""
Load saved hyperparameters and run train+predict for each model variant.
"""
import json
import warnings
import numpy as np
from typing import Dict

from forecast_utils import create_sklearn_model, train_statsmodels, predict_statsmodels
from config import PARAMS_FILE, SEASONAL_PERIOD

# Keys stored in the params JSON that are optimizer metadata, not sklearn constructor args
_METADATA_KEYS = {'csa_best_score', 'csa_iterations', 'bayes_best_score', 'bayes_iterations'}

# sklearn model families
_SKLEARN_FAMILIES = {'xgboost', 'random_forest'}
_STATS_FAMILIES   = {'arimax', 'sarimax'}


def load_saved_params() -> Dict:
    """Return the 'models' section of params_Daily_h1.json."""
    with open(PARAMS_FILE, 'r') as f:
        data = json.load(f)
    return data['models']


def _model_family(variant_key: str) -> str:
    """Return 'xgboost', 'random_forest', 'arimax', or 'sarimax' from a variant key."""
    for fam in ('xgboost', 'random_forest', 'arimax', 'sarimax'):
        if variant_key.startswith(fam):
            return fam
    raise ValueError(f"Unknown model family for variant: {variant_key}")


def _clean_sklearn_params(params: Dict) -> Dict:
    """Strip optimizer metadata keys that are not sklearn constructor arguments."""
    return {k: v for k, v in params.items() if k not in _METADATA_KEYS}


def _run_sklearn(model_type: str, params: Dict, arrays: Dict) -> np.ndarray:
    clean = _clean_sklearn_params(params)
    model = create_sklearn_model(model_type, clean)
    model.fit(arrays['X_train'], arrays['y_train'])
    return model.predict(arrays['X_target'])


def _run_statsmodels(model_type: str, params: Dict, arrays: Dict) -> np.ndarray:
    n_target = arrays['n_target']
    if n_target == 0:
        return np.array([])

    order = tuple(int(x) for x in params['order'])
    seasonal_order = tuple(int(x) for x in params.get('seasonal_order', [0, 0, 0, 0]))

    # Always enforce the correct seasonal period for SARIMAX
    if model_type == 'sarimax':
        seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], SEASONAL_PERIOD)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = train_statsmodels(
            model_type,
            arrays['y_train'],
            arrays['X_exog_train'],
            order,
            seasonal_order,
        )

    if fitted is None:
        return np.full(n_target, np.nan)

    return predict_statsmodels(fitted, arrays['X_exog_target'])


def run_single_variant(variant_key: str, params: Dict, arrays: Dict) -> np.ndarray:
    """
    Train a model variant with saved hyperparameters and return predicted log returns.
    Returns np.full(n_target, np.nan) on any failure.
    """
    family = _model_family(variant_key)
    n_target = arrays['n_target']

    try:
        if family in _SKLEARN_FAMILIES:
            return _run_sklearn(family, params, arrays)
        else:
            return _run_statsmodels(family, params, arrays)
    except Exception as exc:
        print(f"    WARN {variant_key}: {exc}")
        return np.full(n_target, np.nan)
