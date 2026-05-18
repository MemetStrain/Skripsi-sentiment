"""
prediction/feature_engineering.py — dependency-light feature engineering
shared between the offline ablation training scripts and the live website
inference path.

Unified lag schema (Formula A)
------------------------------
Every (ablation x horizon) pair emits the *same* set of feature columns; only
the underlying dates differ. `prediction/master_features.py` is the single
source of truth for which base features each ablation uses and which lag
indices each base exposes.

There are two entry points, both reading that one schema:

* `build_unified_features` — TRAINING. Rows are indexed by the target day `d`.
  A lag column `<base>_lag{k}` at horizon `h` takes `df[base].shift(k+h-1)`,
  so `_lag1` resolves to the forecast origin `d - h`. The target is
  `log(C[d] / C[d-h])`. Used by the C{1..4} ablation scripts.

* `engineer_all_features` — INFERENCE. Rows are indexed by the forecast
  origin `o`. A lag column `<base>_lag{k}` takes `df[base].shift(k-1)`, so the
  row's `_lag{k}` value equals `base[o-k+1]` — exactly what
  `build_unified_features` places at training row `d = o + h`. The website
  predictor slices the trained model's `feature_cols` from this superset.

Because both paths shift relative to the same anchor (the forecast origin),
their feature *values* agree by construction; the schema can never diverge.

Key design notes
----------------
* The website does not read CSVs — it pulls price / sentiment / HMM data
  from Firestore as DataFrames. `merge_inputs` operates on DataFrames so
  both paths can call it.
* HMM one-hot dummies are derived from the `HMM_State_Label` column. At
  training time the top-5 most-frequent labels were dummied; at inference
  we replicate that selection deterministically from the same column.
* `Price` / `Return` are same-day base columns derived from `Close`; the
  legacy `Price_t-N` / `Return_t-N` CSV columns are ignored (the unified
  `<base>_lag{k}` machinery re-derives the equivalent shifts).
"""

import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Make `master_features` importable regardless of how this module itself was
# imported (top-level `feature_engineering` from a script that put prediction/
# on sys.path, or `prediction.feature_engineering` from repo root).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from master_features import (  # noqa: E402  (after sys.path tweak)
    LAG_FEATURES, CALENDAR_FEATURE_NAMES, get_ablation_bases, lag_shift,
)


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
# Shared feature-engineering helpers (used by both train and inference paths)
# =============================================================================

def _build_base_columns(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    """
    Materialise the same-day CPO base columns the unified schema expects.

    `Price` / `Return` / `Log_Return` are derived from the close price when
    absent. The legacy `Price_t-N` / `Return_t-N` columns (if present) are left
    untouched — the unified `<base>_lag{k}` machinery re-derives the shifts.
    """
    df = df.copy()
    if 'Price' not in df.columns:
        df['Price'] = df[close_col]
    if 'Return' not in df.columns:
        df['Return'] = df[close_col].pct_change()
    if 'Log_Return' not in df.columns:
        df['Log_Return'] = np.log(df[close_col] / df[close_col].shift(1))
    return df


def _build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Sentiment_x_Return and Volatility_x_RSI as same-day (un-shifted)
    columns. They are later treated as base features and shifted into their
    `_lag{k}` form alongside every other base.
    """
    df = df.copy()
    if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
        df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']
    if 'HMM_Volatility' in df.columns and 'RSI' in df.columns:
        df['Volatility_x_RSI'] = df['HMM_Volatility'] * df['RSI']
    return df


def _build_calendar(df: pd.DataFrame, offset_rows: int) -> pd.DataFrame:
    """
    Add the 6 cyclical calendar columns, anchored at the forecast origin.

    The origin date is `offset_rows` trading rows before the row's own date,
    obtained with `Date.shift(offset_rows)` — the *same* row shift the target
    and lag features use. Anchoring all three at one point keeps a training
    row internally coherent and keeps the train / inference paths consistent
    on real, gap-containing trading calendars (a calendar-day subtraction
    would drift against the row-based lags whenever weekends/holidays fall in
    the window). Training passes `offset_rows = horizon`; inference passes
    `offset_rows = 0` because its rows are already indexed by the origin.
    """
    df = df.copy()
    src = df['Date'].shift(offset_rows)

    month = src.dt.month
    dow   = src.dt.dayofweek
    woy   = src.dt.isocalendar().week.astype('float64')  # float tolerates NaT

    df['Month_Sin']      = np.sin(2 * np.pi * month / 12)
    df['Month_Cos']      = np.cos(2 * np.pi * month / 12)
    df['DayOfWeek_Sin']  = np.sin(2 * np.pi * dow / 5)
    df['DayOfWeek_Cos']  = np.cos(2 * np.pi * dow / 5)
    df['WeekOfYear_Sin'] = np.sin(2 * np.pi * woy / 52)
    df['WeekOfYear_Cos'] = np.cos(2 * np.pi * woy / 52)
    return df


# =============================================================================
# Training feature builder — horizon-aware, target-day-indexed (Formula A)
# =============================================================================

def build_unified_features(
    df_raw: pd.DataFrame,
    horizon: int,
    ablation: str,
    *,
    target_close_col: str = 'Close',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the horizon-aware feature matrix for one ablation configuration.

    Schema invariant: for every (horizon, ablation) pair this function emits
    the same set of feature column names; only the underlying dates differ
    (Formula A).

    Convention:
        - Each row is indexed by the target day `d` (`Date` column).
        - For each base feature B and lag k in LAG_FEATURES[B], emit
          `B_lag{k}` = df[B].shift(k + h - 1); `_lag1` resolves to `d - h`.
        - Calendar columns anchored at the forecast origin (`Date` shifted
          back `h` trading rows).
        - `Close_Origin` = C[d-h] — the inverse-transform anchor (not a
          feature; carried for the training pipeline).
        - `Target_LogReturn` = log(C[d] / C[d-h]).
        - Rows with any NaN (head rows from shifts / target) are dropped.

    Args:
        df_raw: Merged frame with `Date`, the close column, and all raw
            feature columns the chosen ablation needs.
        horizon: Forecast horizon h, >= 1.
        ablation: One of 'C1_cpo_only', 'C2_cpo_hmm', 'C3_cpo_sentiment',
            'C4_full'.
        target_close_col: Name of the close-price column used for the target.

    Returns:
        (out, feature_cols) where `out` contains Date, Close_Origin, the 6
        calendar columns, every `<base>_lag{k}` column, and Target_LogReturn,
        with all-NaN rows removed. `feature_cols` lists only the model-input
        columns (calendar + lags), in deterministic order.

    Raises:
        ValueError: on bad horizon, unknown ablation, missing required
            columns, or an empty post-dropna result.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if 'Date' not in df_raw.columns:
        raise ValueError("df_raw must have a 'Date' column.")
    if target_close_col not in df_raw.columns:
        raise ValueError(
            f"target_close_col {target_close_col!r} not in df_raw."
        )

    df = df_raw.sort_values('Date').reset_index(drop=True).copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = _build_base_columns(df, target_close_col)
    df = _build_interactions(df)
    df = _build_calendar(df, horizon)

    bases = get_ablation_bases(ablation)

    out = pd.DataFrame({'Date': df['Date'].values})
    out['Close_Origin'] = df[target_close_col].shift(horizon).values

    feature_cols: List[str] = []

    # Calendar features (always included).
    for cal in CALENDAR_FEATURE_NAMES:
        out[cal] = df[cal].values
        feature_cols.append(cal)

    # Lag features.
    for base in bases:
        if base not in LAG_FEATURES:
            warnings.warn(
                f"[build_unified_features] Base feature {base!r} not in "
                f"LAG_FEATURES schema; skipping. (ablation={ablation})"
            )
            continue
        if base not in df.columns:
            warnings.warn(
                f"[build_unified_features] Source column {base!r} not in "
                f"df_raw; skipping all its lags. "
                f"(ablation={ablation}, h={horizon})"
            )
            continue
        for k in LAG_FEATURES[base]:
            col_name = f'{base}_lag{k}'
            out[col_name] = df[base].shift(lag_shift(k, horizon)).values
            feature_cols.append(col_name)

    # Target: log(C[d] / C[d-h]).
    close = df[target_close_col]
    out['Target_LogReturn'] = np.log(close / close.shift(horizon)).values

    # Drop rows with any NaN (head rows from .shift(), or target NaN at start).
    before = len(out)
    out = out.dropna().reset_index(drop=True)
    if len(out) == 0:
        emitted = [b for b in bases if b in LAG_FEATURES and b in df.columns]
        max_lag = max((max(LAG_FEATURES[b]) for b in emitted), default=0)
        raise ValueError(
            f"All {before} rows dropped after NaN removal. Check that df_raw "
            f"has enough history for horizon={horizon} and max lag={max_lag}."
        )

    return out, feature_cols


# =============================================================================
# Inference feature builder — origin-indexed unified superset
# =============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the inference-time unified feature superset.

    Each row is a forecast *origin* `o`. For every base feature any ablation
    could use, and every lag k in LAG_FEATURES[base], emit
    `<base>_lag{k}` = df[base].shift(k - 1) — so the row's lag-k value equals
    `base[o - k + 1]`, exactly what `build_unified_features` places at the
    training row `d = o + h`. Calendar columns come from the row's own date.

    The frame is NOT NaN-dropped (the predictor needs the most recent rows and
    skips any row whose sliced feature vector still contains NaN). The website
    predictor slices the trained model's `feature_cols` from this superset.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df = _build_base_columns(df, 'Close')
    df = _build_interactions(df)
    df = _build_calendar(df, 0)  # origin-indexed: calendar from the row's date

    # Union of every ablation's bases == C4_full's bases.
    for base in get_ablation_bases('C4_full'):
        if base not in LAG_FEATURES or base not in df.columns:
            continue
        for k in LAG_FEATURES[base]:
            df[f'{base}_lag{k}'] = df[base].shift(k - 1)

    return df


# =============================================================================
# Deprecation shim
# =============================================================================

def engineer_features_for_horizon(df: pd.DataFrame, horizon: int,
                                  lag_periods=None) -> Tuple[pd.DataFrame, List[str]]:
    """DEPRECATED: use build_unified_features(df, horizon, ablation)."""
    warnings.warn(
        "engineer_features_for_horizon is deprecated; "
        "use build_unified_features with an explicit ablation argument.",
        DeprecationWarning, stacklevel=2,
    )
    return build_unified_features(df, horizon, ablation='C4_full')


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
