"""
prediction/feature_engineering.py — dependency-light feature engineering
shared between the offline ablation training scripts and the live website
inference path.

The ablation training scripts (horizon_forecast_C{1..4}_*.py) historically
each defined their own `engineer_features_for_horizon`. This module extracts
the union of those recipes so a single function can be called from both
training and inference. Any divergence between the two would silently
produce wrong predictions at inference time, so we keep one source of truth.

Key design notes:
* The website does not read CSVs — it pulls price / sentiment / HMM data
  from Firestore as DataFrames. `merge_inputs` operates on DataFrames so
  both paths can call it.
* HMM one-hot dummies are derived from the `HMM_State_Label` column. At
  training time the top-5 most-frequent labels were dummied; at inference
  we replicate that selection deterministically from the same column.
* `engineer_all_features` adds the *superset* of all columns the four
  ablations could need. The trained model's `meta.json` records exactly
  which subset its `feature_cols` consumed, and the predictor selects by
  that list — extra columns in the superset are ignored, missing ones
  trigger a clear error.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Price features — mirror cpo/preprocess_cpo_variables.py (Wilder RSI, MACD, BB)
# =============================================================================

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered price-side columns produced by
    cpo/preprocess_cpo_variables.py to an OHLCV DataFrame.

    Required input columns: Date, Close, Open, High, Low, Volume.
    Output adds: Price_t-1..3, Return_t-1..2, Volume_t-1, Log_Return,
                 High_Low_Spread, Open_Close_Spread, SMA_3/6, EMA_3/6,
                 RSI (Wilder, 14), MACD, MACD_Signal, Bollinger_Band_Width.
    """
    df = df.copy()

    df['Price_t-1'] = df['Close'].shift(1)
    df['Price_t-2'] = df['Close'].shift(2)
    df['Price_t-3'] = df['Close'].shift(3)

    ret = df['Close'].pct_change()
    df['Return_t-1'] = ret.shift(1)
    df['Return_t-2'] = ret.shift(2)
    df['Volume_t-1'] = df['Volume'].shift(1)

    df['Log_Return']        = np.log(df['Close'] / df['Close'].shift(1))
    df['High_Low_Spread']   = df['High']  - df['Low']
    df['Open_Close_Spread'] = df['Open']  - df['Close']

    df['SMA_3'] = df['Close'].rolling(3, min_periods=3).mean()
    df['SMA_6'] = df['Close'].rolling(6, min_periods=6).mean()
    df['EMA_3'] = df['Close'].ewm(span=3, adjust=False).mean()
    df['EMA_6'] = df['Close'].ewm(span=6, adjust=False).mean()

    # RSI — Wilder smoothing
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std(ddof=1)
    df['Bollinger_Band_Width'] = 2.0 * 2.0 * std20  # (mid + 2σ) − (mid − 2σ)
    return df


# =============================================================================
# HMM-derived features — mirror markov/cpo_hmm_states.py (rolling-mean RSI etc.)
# =============================================================================

def add_hmm_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the HMM-side derived columns (HMM_Close, HMM_Log_Return, HMM_Volatility,
    HMM_RSI, HMM_MACD) computed with the same formulas as
    markov/cpo_hmm_states.py — i.e. simple rolling mean for RSI rather than
    Wilder smoothing — so the values match the offline CSV the C-scripts
    train on.
    """
    df = df.copy()
    close = df['Close'].astype(float)

    df['HMM_Close']      = close
    df['HMM_Log_Return'] = np.log(close / close.shift(1))
    df['HMM_Volatility'] = df['HMM_Log_Return'].rolling(20).std()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['HMM_RSI'] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['HMM_MACD'] = ema12 - ema26
    return df


# Lag periods used by the ablation scripts. The training-time filter
# `safe_lags = [l for l in BASE_LAG_PERIODS if l >= horizon]` happens
# in `engineer_features_for_horizon`; at inference we compute all lags
# and let `feature_cols` drive selection.
BASE_LAG_PERIODS = [1, 2, 3, 5, 10, 20]


# =============================================================================
# Merge — turn raw price / sentiment / HMM frames into a single feature frame
# =============================================================================

def merge_inputs(
    price_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    hmm_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge the three data sources into one DataFrame indexed by Date.

    Expects the same column names the offline preprocess scripts produce:

        price_df:      Date, Close, Open, High, Low, Volume, Change_Pct,
                       Price_t-1..3, Return_t-1..2, Volume_t-1, Log_Return,
                       High_Low_Spread, Open_Close_Spread, SMA_3/6, EMA_3/6,
                       RSI, MACD, MACD_Signal, Bollinger_Band_Width
        sentiment_df:  Date, Article_Count, Positive_Prob, Negative_Prob,
                       Neutral_Prob, Sentiment_Score, Confidence
                       (Combined_* aliases also accepted)
        hmm_df:        Date, Close, Log_Return, Volatility, RSI, MACD,
                       State, State_Label

    Either of the optional DataFrames may be None for the ablations that
    don't use them (C1 uses neither; C2 uses HMM only; C3 uses sentiment
    only; C4 uses both).
    """
    df = price_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if sentiment_df is not None and not sentiment_df.empty:
        s = sentiment_df.copy()
        s['Date'] = pd.to_datetime(s['Date'])
        s = s.rename(columns={
            'Combined_Positive_Prob': 'Positive_Prob',
            'Combined_Negative_Prob': 'Negative_Prob',
            'Combined_Neutral_Prob':  'Neutral_Prob',
            'Combined_Confidence':    'Confidence',
        })
        keep = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
                'Neutral_Prob', 'Confidence', 'Sentiment_Score']
        s = s[[c for c in keep if c in s.columns]]
        df = df.merge(s, on='Date', how='inner')

    if hmm_df is not None and not hmm_df.empty:
        h = hmm_df.copy()
        h['Date'] = pd.to_datetime(h['Date'])
        h = h.rename(columns={
            'Close':       'HMM_Close',
            'Log_Return':  'HMM_Log_Return',
            'Volatility':  'HMM_Volatility',
            'RSI':         'HMM_RSI',
            'MACD':        'HMM_MACD',
            'State':       'HMM_State',
            'State_Label': 'HMM_State_Label',
        })
        df = df.merge(h, on='Date', how='inner')

        if 'HMM_State_Label' in df.columns:
            top_states = df['HMM_State_Label'].value_counts().head(5).index.tolist()
            for state in top_states:
                col_name = f'HMM_{state.replace(" ", "_").replace("-", "_")}'
                df[col_name] = (df['HMM_State_Label'] == state).astype(int)
            df = df.drop(columns=['HMM_State_Label'])

    return df.sort_values('Date').reset_index(drop=True)


# =============================================================================
# Feature engineering — superset of columns any C{1..4} ablation may need
# =============================================================================

def engineer_all_features(
    df: pd.DataFrame,
    lag_periods: List[int] = BASE_LAG_PERIODS,
) -> pd.DataFrame:
    """
    Add cyclical seasonality, all lags, and all interaction terms.

    The output is the *superset* of columns any of the four ablations could
    have consumed at training time. The website inference path slices this
    superset by the trained model's `feature_cols` to get the exact input
    matrix shape the model expects.
    """
    df = df.copy()

    # Cyclical seasonality.
    df['Month_Sin']      = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_Cos']      = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    df['DayOfWeek_Sin']  = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['DayOfWeek_Cos']  = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
    df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)

    # Lags. Only generate columns for signals actually present in the frame
    # (so C1 — price-only — won't grow Sentiment / HMM lag columns).
    candidate_lag_cols = ['Close', 'Sentiment_Score', 'HMM_State']
    for col in candidate_lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_periods:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Interactions.
    if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
        df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']
    if 'HMM_Volatility' in df.columns and 'RSI' in df.columns:
        df['Volatility_x_RSI'] = df['HMM_Volatility'] * df['RSI']

    return df


def engineer_features_for_horizon(
    df: pd.DataFrame, horizon: int,
    lag_periods: List[int] = BASE_LAG_PERIODS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Training-time helper: add features, set Target = log(Close[t+h] / Close[t]),
    drop rows with NaNs, and return (df, feature_cols). Mirrors the per-script
    `engineer_features_for_horizon` that previously lived in each C-script.

    Horizon-aware lag filtering: only lags >= horizon are added, to prevent
    accidentally training on a lag shorter than the forecast distance.
    """
    safe_lags = [lag for lag in lag_periods if lag >= horizon] or [horizon]
    df = engineer_all_features(df, lag_periods=safe_lags)

    df['Target'] = np.log(df['Close'].shift(-horizon) / df['Close'])
    df = df.dropna().reset_index(drop=True)

    exclude = {'Date', 'Target', 'Dominant_Sentiment', 'HMM_Close'}
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in ('float64', 'int64', 'int32', 'float32')]
    return df, feature_cols


# =============================================================================
# Inference — slice the superset DF to a feature matrix matching feature_cols
# =============================================================================

def select_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Return df[feature_cols].values, raising a clear error if any column is
    missing. Used at inference time after `engineer_all_features` has built
    the superset.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f'Feature columns missing from inference frame: {missing}. '
            f'Did the data sources match what the model was trained on?'
        )
    return df[feature_cols].to_numpy()
