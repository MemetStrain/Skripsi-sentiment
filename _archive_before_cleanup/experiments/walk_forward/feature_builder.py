"""
Feature engineering and train/target array preparation for walk-forward evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import List, Tuple, Dict

from horizon_forecast_cpo_hmm import engineer_features_for_horizon
from forecast_utils import select_top_exog
from config import INTERVAL, HORIZON, TOP_N_EXOG


def build_features(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run horizon-aware feature engineering on the FULL dataset.

    Feature engineering is applied before date filtering so that lag features
    at the train/target boundary are computed correctly.  Date filtering happens
    inside prepare_arrays.
    """
    print(f"  Engineering features (interval={INTERVAL}, horizon={HORIZON})...")
    featured_df, feature_cols = engineer_features_for_horizon(full_df, INTERVAL, HORIZON)
    print(f"  {len(feature_cols)} feature columns, {len(featured_df)} rows after dropna")
    return featured_df, feature_cols


def prepare_arrays(
    featured_df: pd.DataFrame,
    feature_cols: List[str],
    train_cutoff: pd.Timestamp,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> Dict:
    """
    Split featured_df into train and target sets, scale, and select exogenous features.

    Returns a dict with keys:
        X_train, y_train, train_dates, close_train,
        X_target, y_target, target_dates, close_target,
        X_exog_train, X_exog_target, exog_indices,
        scaler, n_train, n_target
    """
    train_mask  = featured_df['Date'] <= train_cutoff
    target_mask = (featured_df['Date'] >= target_start) & (featured_df['Date'] <= target_end)

    train_df  = featured_df[train_mask].reset_index(drop=True)
    target_df = featured_df[target_mask].reset_index(drop=True)

    # Leakage guard
    if len(train_df) > 0 and len(target_df) > 0:
        assert train_df['Date'].max() < target_df['Date'].min(), (
            f"Leakage: train max date {train_df['Date'].max()} >= "
            f"target min date {target_df['Date'].min()}"
        )

    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values
    close_train = train_df['Close'].values
    train_dates = train_df['Date'].values

    X_target = target_df[feature_cols].values
    y_target = target_df['Target'].values
    close_target = target_df['Close'].values
    target_dates = target_df['Date'].values

    # Fit scaler on training data only
    scaler = RobustScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_target_scaled = scaler.transform(X_target) if len(X_target) > 0 else X_target

    # Select top exog features using training data only
    X_exog_train, exog_indices = select_top_exog(
        X_train_scaled, y_train, n=min(TOP_N_EXOG, X_train_scaled.shape[1])
    )
    X_exog_target = X_target_scaled[:, exog_indices] if len(X_target_scaled) > 0 else X_target_scaled

    assert X_exog_train.shape[1] == X_exog_target.shape[1] if len(X_exog_target) > 0 else True, \
        "Exog feature count mismatch between train and target"

    return {
        'X_train':       X_train_scaled,
        'y_train':       y_train,
        'train_dates':   train_dates,
        'close_train':   close_train,
        'X_target':      X_target_scaled,
        'y_target':      y_target,
        'target_dates':  target_dates,
        'close_target':  close_target,
        'X_exog_train':  X_exog_train,
        'X_exog_target': X_exog_target,
        'exog_indices':  exog_indices,
        'scaler':        scaler,
        'n_train':       len(train_df),
        'n_target':      len(target_df),
    }
