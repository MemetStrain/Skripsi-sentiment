"""
hmm_updater.py — Fit Gaussian HMM on CPO price data and write states to Firestore.

Runs for Daily, Weekly, and Monthly frequencies.
Uses BIC to select the optimal number of hidden states (2–4).
Mirrors the logic in markov/cpo_hmm_states.py.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)

FREQUENCIES = ['Daily', 'Weekly', 'Monthly']
N_STATES_RANGE = range(2, 5)   # try 2, 3, 4 states
N_ITER = 200
N_INIT = 10                    # random restarts for Baum-Welch
ROLLING_WINDOW = 252           # ~1 trading year for normalisation


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HMM observation features from price data.
    Requires columns: date, close, (optional) open, high, low.
    Returns a DataFrame with normalised features.
    """
    df = df.sort_values('date').copy()
    close = df['close'].astype(float)

    # Log returns
    log_ret = np.log(close / close.shift(1))

    # Rolling volatility (std of log returns)
    volatility = log_ret.rolling(20).std()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # MACD (12, 26)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26

    # Bollinger Band width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = (2 * std20) / (sma20 + 1e-9)

    # Z-score normalise using rolling window
    def rolling_z(series):
        mu = series.rolling(ROLLING_WINDOW, min_periods=20).mean()
        sigma = series.rolling(ROLLING_WINDOW, min_periods=20).std()
        return (series - mu) / (sigma + 1e-9)

    feat = pd.DataFrame({
        'date': df['date'].values,
        'log_return_z': rolling_z(log_ret),
        'volatility_z': rolling_z(volatility),
        'rsi_norm': (rsi - 50) / 50,
        'macd_z': rolling_z(macd),
        'bb_width_z': rolling_z(bb_width),
    })
    feat = feat.dropna()
    return feat


# ---------------------------------------------------------------------------
# BIC model selection
# ---------------------------------------------------------------------------

def _fit_hmm_bic(X: np.ndarray) -> tuple:
    """
    Fit Gaussian HMM for each n_states in N_STATES_RANGE.
    Return (best_model, best_n_states, bic_scores).
    """
    bic_scores = {}
    best_model = None
    best_bic = np.inf

    for n in N_STATES_RANGE:
        best_for_n = None
        best_ll = -np.inf
        for _ in range(N_INIT):
            try:
                model = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type='full',
                    n_iter=N_ITER,
                    random_state=np.random.randint(0, 10000),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(X)
                ll = model.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_for_n = model
            except Exception:
                continue

        if best_for_n is None:
            continue

        # BIC = -2 * log_likelihood + n_params * log(T)
        T, k = X.shape
        n_params = n * k + n * k * (k + 1) // 2 + n * (n - 1)
        bic = -2 * best_ll + n_params * np.log(T)
        bic_scores[n] = bic
        if bic < best_bic:
            best_bic = bic
            best_model = best_for_n

    return best_model, bic_scores


# ---------------------------------------------------------------------------
# State labelling
# ---------------------------------------------------------------------------

def _label_states(model, n_states: int) -> dict:
    """
    Sort states by mean log-return and assign labels.
    2 states: Bullish / Bearish
    3 states: Bullish / Neutral / Bearish
    4 states: Bullish-1 / Bullish-2 / Bearish-1 / Bearish-2
    """
    means = model.means_[:, 0]  # first feature = log_return_z
    order = np.argsort(means)   # ascending: most bearish → most bullish

    if n_states == 2:
        labels_ordered = ['Bearish', 'Bullish']
    elif n_states == 3:
        labels_ordered = ['Bearish', 'Neutral', 'Bullish']
    else:
        labels_ordered = ['Bearish', 'Bearish-2', 'Bullish-2', 'Bullish']

    return {int(order[i]): labels_ordered[i] for i in range(n_states)}


# ---------------------------------------------------------------------------
# Resample price data for different frequencies
# ---------------------------------------------------------------------------

def _resample(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    rule = {'Daily': None, 'Weekly': 'W-FRI', 'Monthly': 'ME'}.get(frequency)
    if rule is None:
        return df.reset_index()

    agg_funcs = {
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }
    present = {k: v for k, v in agg_funcs.items() if k in df.columns}
    resampled = df.resample(rule).agg(present).dropna(subset=['close'])
    resampled.index = resampled.index.strftime('%Y-%m-%d')
    resampled.index.name = 'date'
    return resampled.reset_index()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def update_hmm_states(db) -> None:
    """
    Fetch all price data from `daily_prices`, compute HMM states for
    Daily / Weekly / Monthly frequencies, and write to `hmm_states`.
    """
    from firestore_writer import write_hmm_states_batch

    # Fetch all price data
    logger.info('Fetching all price data for HMM computation...')
    price_docs = db.collection('daily_prices').order_by('date').stream()
    price_rows = []
    for doc in price_docs:
        d = doc.to_dict()
        price_rows.append({
            'date': d.get('date'),
            'open': float(d.get('open', 0)),
            'high': float(d.get('high', 0)),
            'low': float(d.get('low', 0)),
            'close': float(d.get('close', 0)),
            'volume': float(d.get('volume', 0)),
        })

    if len(price_rows) < 50:
        logger.warning(f'Not enough price data for HMM ({len(price_rows)} rows)')
        return

    base_df = pd.DataFrame(price_rows)

    for freq in FREQUENCIES:
        logger.info(f'Computing HMM states for frequency={freq}')
        try:
            df = _resample(base_df, freq)
            feat_df = _compute_features(df)
            if len(feat_df) < 30:
                logger.warning(f'Skipping {freq}: too few samples ({len(feat_df)})')
                continue

            X = feat_df.drop('date', axis=1).values.astype(np.float64)
            model, bic_scores = _fit_hmm_bic(X)
            if model is None:
                logger.warning(f'HMM fitting failed for {freq}')
                continue

            n_states = model.n_components
            label_map = _label_states(model, n_states)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                state_seq = model.predict(X)

            # Build output records
            states_out = []
            for i, row in enumerate(feat_df.itertuples()):
                s = int(state_seq[i])
                states_out.append({
                    'date': row.date,
                    'frequency': freq,
                    'state': s,
                    'state_label': label_map.get(s, 'Neutral'),
                    'log_return': round(float(row.log_return_z), 4),
                    'volatility': round(float(row.volatility_z), 4),
                    'rsi': round(float(row.rsi_norm), 4),
                })

            write_hmm_states_batch(db, states_out)
            logger.info(f'HMM {freq}: {n_states} states, {len(states_out)} records written')

        except Exception as e:
            logger.error(f'HMM update failed for {freq}: {e}', exc_info=True)
