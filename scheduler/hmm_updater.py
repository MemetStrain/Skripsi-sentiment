"""
hmm_updater.py — Decode CPO Daily HMM states using FROZEN parameters.

This module never re-fits the HMM. Parameters are pinned during offline
training (markov/cpo_hmm_states.py) and persisted to Firestore in
`hmm_models/Daily`. At serve time we:

  1. Load OHLCV from `daily_prices`
  2. Recompute the same five HMM input features the offline pipeline used
     (Wilder RSI, MACD 12/26, absolute Bollinger width, rolling vol of
     log-return — then rolling Z-scores on a 252-day window)
  3. Reconstruct the fitted GaussianHMM from the persisted params
  4. Run an online forward filter (no Viterbi smoothing) to assign each
     row's state using only observations up to and including that row
  5. Upsert state docs to `hmm_states`

Why frozen-params + forward-filter rather than daily refit + Viterbi:
  - Training distribution stays identical to what the offline-trained
    XGBoost models in prediction/saved_models/ were calibrated against.
  - Historical states never change (refit + Viterbi mutates them every day).
  - Forward filter is milliseconds for ~3000 obs.

Refits happen only when you re-run markov/cpo_hmm_states.py offline and
re-publish hmm_models/Daily (see scheduler/migrate_hmm_to_firestore.py).
"""

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

FREQUENCY = 'Daily'


# ---------------------------------------------------------------------------
# Feature engineering — must match markov/cpo_hmm_states.py exactly.
# Upstream indicator formulas come from cpo/preprocess_cpo_variables.py;
# Z-score normalisation comes from markov/cpo_hmm_states.prepare_features.
# ---------------------------------------------------------------------------

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI — matches cpo/preprocess_cpo_variables.compute_rsi."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD line — fast EMA minus slow EMA, both adjust=False."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _bb_width_absolute(close: pd.Series, window: int = 20,
                       n_std: float = 2.0) -> pd.Series:
    """Bollinger Band absolute width = upper minus lower = 2 * n_std * rolling_std."""
    std = close.rolling(window=window).std(ddof=1)
    return 2.0 * n_std * std


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score; min_periods = max(window // 2, 2). Causal (no lookahead)."""
    min_p = max(window // 2, 2)
    mu    = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std()
    return (series - mu) / (sigma + 1e-8)


def _build_hmm_features(price_df: pd.DataFrame,
                        vol_window: int,
                        norm_window: int) -> pd.DataFrame:
    """Build the five HMM input features from raw OHLCV.

    Output columns: Date, Log_Return_Z, Volatility_Z, RSI_norm, MACD_Z, BB_Width_Z.
    Warmup rows with any NaN feature are dropped.
    """
    df = price_df.sort_values('date').reset_index(drop=True).copy()
    close = df['close'].astype(float)

    log_return = np.log(close / close.shift(1))
    volatility = log_return.rolling(vol_window, min_periods=2).std()
    rsi        = _wilder_rsi(close, period=14)
    macd       = _macd_line(close, fast=12, slow=26)
    bb_width   = _bb_width_absolute(close, window=20, n_std=2.0)

    feat = pd.DataFrame({
        'Date':         pd.to_datetime(df['date']),
        'Log_Return_Z': _rolling_zscore(log_return, norm_window),
        'Volatility_Z': _rolling_zscore(volatility, norm_window),
        'RSI_norm':     (rsi - 50.0) / 50.0,
        'MACD_Z':       _rolling_zscore(macd, norm_window),
        'BB_Width_Z':   _rolling_zscore(bb_width, norm_window),
    })
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat


# ---------------------------------------------------------------------------
# HMM reconstruction + online forward filter
# ---------------------------------------------------------------------------

def _hmm_from_params(params: dict) -> hmm.GaussianHMM:
    """Reconstruct a GaussianHMM from the persisted parameter dict.

    Uses init_params="" + params="stmc" so hmmlearn does not overwrite the
    seeded values. We never call .fit() on this object.
    """
    cov_type = params['covariance_type']
    n        = int(params['n_components'])

    model = hmm.GaussianHMM(
        n_components=n,
        covariance_type=cov_type,
        init_params="",
        params="stmc",
        min_covar=1e-3,
    )
    model.startprob_ = np.asarray(params['startprob_'], dtype=float)
    model.transmat_  = np.asarray(params['transmat_'],  dtype=float)
    model.means_     = np.asarray(params['means_'],     dtype=float)
    model.covars_    = np.asarray(params['covars_'],    dtype=float)
    return model


def _forward_filter(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Online forward filter — argmax_i P(q_t = i | O_1..O_t).

    Mirrors markov.cpo_hmm_states.forward_filter — see that function for the
    why (Viterbi smoothing would peek at observations past t).
    """
    log_emit  = model._compute_log_likelihood(X)
    log_start = np.log(np.maximum(model.startprob_, 1e-300))
    log_trans = np.log(np.maximum(model.transmat_, 1e-300))

    T, K = log_emit.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_emit[t]
    return np.argmax(log_alpha, axis=1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def update_hmm_states(db, write_existing: bool = False) -> None:
    """Decode HMM states for all `daily_prices` rows and write to `hmm_states`.

    Parameters
    ----------
    db
        Firestore client.
    write_existing
        If False (default), only write docs for dates not already in the
        `hmm_states` collection. Historical state values are deterministic
        under frozen params, so re-writing them is wasted Firestore writes.
        Set True if you've changed the params doc and want to overwrite.
    """
    from firestore_writer import write_hmm_states_batch, read_hmm_params

    # --- 1. Load frozen HMM parameters ----------------------------------------
    params = read_hmm_params(db, FREQUENCY)
    if params is None:
        logger.error(
            "hmm_models/%s missing — run scheduler/migrate_hmm_to_firestore.py "
            "first to publish params from markov/output/hmm_params_Daily.json",
            FREQUENCY,
        )
        return

    feat_cols   = params['feat_cols']
    vol_window  = int(params['volatility_window'])
    norm_window = int(params['norm_window'])
    state_to_label = {int(k): v for k, v in params['state_to_label'].items()}

    # --- 2. Pull all prices ---------------------------------------------------
    price_docs = db.collection('daily_prices').order_by('date').stream()
    price_rows = []
    for doc in price_docs:
        d = doc.to_dict()
        price_rows.append({
            'date':  d.get('date'),
            'close': float(d.get('close', 0)),
        })

    if len(price_rows) < 50:
        logger.warning(f'Not enough price data for HMM ({len(price_rows)} rows)')
        return

    price_df = pd.DataFrame(price_rows)

    # --- 3. Compute features (matching offline) -------------------------------
    feat_df = _build_hmm_features(price_df, vol_window, norm_window)
    if len(feat_df) < 30:
        logger.warning(f'Skipping HMM update: too few feature rows ({len(feat_df)})')
        return

    X = feat_df[feat_cols].values.astype(np.float64)

    # --- 4. Reconstruct model + forward-filter --------------------------------
    model = _hmm_from_params(params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        states = _forward_filter(model, X)

    # --- 5. Decide which rows to write ----------------------------------------
    existing_dates: set = set()
    if not write_existing:
        for doc in db.collection('hmm_states').stream():
            d = doc.to_dict() or {}
            if d.get('frequency') == FREQUENCY:
                existing_dates.add(str(d.get('date', '')))

    to_write = []
    for i, row in feat_df.iterrows():
        date_str = pd.Timestamp(row['Date']).strftime('%Y-%m-%d')
        if not write_existing and date_str in existing_dates:
            continue
        s = int(states[i])
        to_write.append({
            'date':        date_str,
            'frequency':   FREQUENCY,
            'state':       s,
            'state_label': state_to_label.get(s, 'Neutral'),
            'log_return':  round(float(row['Log_Return_Z']), 4),
            'volatility':  round(float(row['Volatility_Z']), 4),
            'rsi':         round(float(row['RSI_norm']),     4),
        })

    if not to_write:
        logger.info('HMM Daily: no new dates to write (all states up to date).')
        return

    write_hmm_states_batch(db, to_write)
    logger.info(
        f'HMM Daily: wrote {len(to_write)} new state docs '
        f'(frozen params, forward-filtered).'
    )
