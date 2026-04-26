"""
Walk-forward evaluation configuration.
All experiment definitions and path constants live here.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))   # d:\Skripsi1

PARAMS_FILE  = os.path.join(
    PROJECT_ROOT, 'prediction', 'output_horizons', 'Daily', 'horizon_1',
    'params_Daily_h1.json'
)
OUTPUT_DIR = os.path.join(_HERE, 'output')

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
INTERVAL        = 'Daily'
HORIZON         = 1
TOP_N_EXOG      = 10
SEASONAL_PERIOD = 5   # trading days in a week (for SARIMAX)

MODEL_VARIANTS = [
    'xgboost_base',         'xgboost_csa',         'xgboost_bayesian',
    'random_forest_base',   'random_forest_csa',   'random_forest_bayesian',
    'arimax_base',          'arimax_csa',           'arimax_bayesian',
    'sarimax_base',         'sarimax_csa',          'sarimax_bayesian',
]

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
# Each entry: experiment_id, lead_months, target_start, target_end, train_cutoff
EXPERIMENT_GRID = [
    {
        'id':           'lead1_jan2026',
        'lead':         1,
        'target_start': '2026-01-01',
        'target_end':   '2026-01-31',
        'train_cutoff': '2025-12-31',
    },
    {
        'id':           'lead1_feb2026',
        'lead':         1,
        'target_start': '2026-02-01',
        'target_end':   '2026-02-28',
        'train_cutoff': '2026-01-31',
    },
    {
        'id':           'lead1_mar2026',
        'lead':         1,
        'target_start': '2026-03-01',
        'target_end':   '2026-03-31',
        'train_cutoff': '2026-02-28',
    },
    {
        'id':           'lead2_jan2026',
        'lead':         2,
        'target_start': '2026-01-01',
        'target_end':   '2026-01-31',
        'train_cutoff': '2025-11-30',
    },
    {
        'id':           'lead2_feb2026',
        'lead':         2,
        'target_start': '2026-02-01',
        'target_end':   '2026-02-28',
        'train_cutoff': '2025-12-31',
    },
    {
        'id':           'lead2_mar2026',
        'lead':         2,
        'target_start': '2026-03-01',
        'target_end':   '2026-03-31',
        'train_cutoff': '2026-01-31',
    },
]
