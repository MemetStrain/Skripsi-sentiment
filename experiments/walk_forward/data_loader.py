"""
Load the full merged CPO + HMM dataset by reusing the original pipeline loader.
"""
import os
import sys

from config import PROJECT_ROOT

# Bootstrap sys.path so the original prediction modules are importable.
# prediction/ must come before prediction/utils/ so that bare `import crow_search_optimizer`
# inside forecast_utils.py resolves correctly.
for _p in [
    os.path.join(PROJECT_ROOT, 'prediction'),
    os.path.join(PROJECT_ROOT, 'prediction', 'utils'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from horizon_forecast_cpo_hmm import load_and_merge_data  # noqa: E402


def load_full_dataset() -> pd.DataFrame:
    """Return the full merged CPO + HMM DataFrame (all dates, no feature engineering yet)."""
    print("Loading and merging CPO + HMM data...")
    df = load_and_merge_data('Daily')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  Loaded {len(df)} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")
    return df
