"""
predictor.py — Live XGBoost inference for the dashboard.

Replaces the old `predictions` Firestore collection. For each horizon h:
* the auto-picked winning ablation config (`prediction/winners.json`)
* its CSA model (`prediction/saved_models/{tag}/Daily/h{h}/xgboost_csa/`)

is used to produce a rolling forecast trail across the 90-day visible
window plus one h-step-ahead future point.

The feature engineering exactly mirrors the offline training pipeline by
calling into `prediction/feature_engineering.py` (the single source of
truth shared with the C{1..4} ablation scripts).

Module-level model caches mean each gunicorn / Vercel function process
loads each pkl at most once.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_PRED_DIR     = os.path.join(_PROJECT_ROOT, 'prediction')

_WINNERS_PATH      = os.path.join(_PRED_DIR, 'winners.json')
_SAVED_MODELS_DIR  = os.path.join(_PRED_DIR, 'saved_models')

if _PRED_DIR not in sys.path:
    sys.path.insert(0, _PRED_DIR)

from feature_engineering import (  # noqa: E402  (after sys.path tweak)
    add_price_features, add_hmm_derived_features,
    merge_inputs, engineer_all_features, select_feature_matrix,
)


# ---------------------------------------------------------------------------
# Winners metadata
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_winners() -> dict:
    """Return the parsed winners.json payload (cached for the process lifetime)."""
    if not os.path.exists(_WINNERS_PATH):
        raise FileNotFoundError(
            f'winners.json not found at {_WINNERS_PATH}. '
            f'Run prediction/compute_winners.py after the C{{1..4}} training scripts.'
        )
    with open(_WINNERS_PATH, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model loader (per (tag, horizon, variant))
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def load_model(tag: str, horizon: int, variant: str = 'csa') -> Tuple[object, object, List[str]]:
    """
    Load (model, scaler, feature_cols) for the given ablation tag & horizon.
    Cached so each process pays the joblib.load cost at most once.
    """
    base = os.path.join(_SAVED_MODELS_DIR, tag, 'Daily', f'h{horizon}', f'xgboost_{variant}')
    model_path  = os.path.join(base, 'model.pkl')
    scaler_path = os.path.join(base, 'scaler.pkl')
    meta_path   = os.path.join(base, 'meta.json')

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f'Model artefacts missing in {base}')

    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    feature_cols = list(meta.get('feature_cols', []))
    if not feature_cols:
        raise ValueError(f'meta.json at {meta_path} has empty feature_cols')

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Firestore → DataFrame loaders
# ---------------------------------------------------------------------------

def _load_prices(db) -> pd.DataFrame:
    """Pull all daily_prices into a DataFrame. Sorted ascending by date."""
    rows = []
    for d in db.collection('daily_prices').order_by('date').stream():
        x = d.to_dict()
        rows.append({
            'Date':   x.get('date'),
            'Close':  float(x.get('close',  0)),
            'Open':   float(x.get('open',   0)),
            'High':   float(x.get('high',   0)),
            'Low':    float(x.get('low',    0)),
            'Volume': float(x.get('volume', 0)),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def _load_sentiment(db) -> pd.DataFrame:
    """Pull all sentiment_aggregates (Daily) into a DataFrame."""
    rows = []
    for d in db.collection('sentiment_aggregates').stream():
        x = d.to_dict()
        if x.get('frequency') and x['frequency'] != 'Daily':
            continue
        rows.append({
            'Date':            x.get('date'),
            'Article_Count':   int(x.get('article_count', 0)),
            'Positive_Prob':   float(x.get('positive_prob', 0.0)),
            'Negative_Prob':   float(x.get('negative_prob', 0.0)),
            'Neutral_Prob':    float(x.get('neutral_prob',  0.0)),
            'Sentiment_Score': float(x.get('sentiment_score', 0.0)),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def _load_hmm_state(db) -> pd.DataFrame:
    """
    Pull just (Date, HMM_State, HMM_State_Label) from hmm_states. The other
    HMM-prefixed features (HMM_RSI, HMM_MACD, HMM_Volatility, HMM_Log_Return,
    HMM_Close) are recomputed from price data by add_hmm_derived_features —
    that avoids the z-score-vs-raw mismatch between Firestore's stored values
    and the offline training CSV.
    """
    rows = []
    for d in db.collection('hmm_states').stream():
        x = d.to_dict()
        if x.get('frequency') and x['frequency'] != 'Daily':
            continue
        rows.append({
            'Date':            x.get('date'),
            'HMM_State':       int(x.get('state', 2)),
            'HMM_State_Label': x.get('state_label', 'Neutral'),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference frame
# ---------------------------------------------------------------------------

def build_inference_frame(db) -> pd.DataFrame:
    """
    Build the full superset DataFrame the inference path needs:
    price + price-derived technicals + HMM-derived technicals +
    HMM state dummies + sentiment + cyclical seasonality + lags + interactions.
    """
    price_df = _load_prices(db)
    if price_df.empty:
        raise RuntimeError('daily_prices is empty — cannot run inference.')

    sent_df = _load_sentiment(db)
    hmm_state_df = _load_hmm_state(db)

    # Price-side and HMM-side technicals computed locally from OHLCV.
    df = add_price_features(price_df)
    df = add_hmm_derived_features(df)

    # Change_Pct is not stored in Firestore (OHLCV only); derive from Close.
    # Training CSV had it in percentage units (0.92 for a 0.92% move).
    df['Change_Pct'] = df['Close'].pct_change() * 100

    # HMM_State + State_Label from Firestore.
    if not hmm_state_df.empty:
        df = df.merge(hmm_state_df, on='Date', how='left')
        df['HMM_State'] = df['HMM_State'].fillna(2).astype(int)
        df['HMM_State_Label'] = df['HMM_State_Label'].fillna('Neutral')
        # Top-5-most-common HMM dummies, matching the offline merge step.
        top_states = df['HMM_State_Label'].value_counts().head(5).index.tolist()
        for state in top_states:
            col_name = f'HMM_{state.replace(" ", "_").replace("-", "_")}'
            df[col_name] = (df['HMM_State_Label'] == state).astype(int)
        df = df.drop(columns=['HMM_State_Label'])

    # Sentiment (rename Combined_*-style to the names training-time uses).
    if not sent_df.empty:
        df = df.merge(sent_df, on='Date', how='left')
        for c in ['Article_Count', 'Positive_Prob', 'Negative_Prob',
                  'Neutral_Prob', 'Sentiment_Score']:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
        # Confidence = max sentiment probability per row.
        # Not stored in sentiment_aggregates — derived here to match training CSV.
        if 'Positive_Prob' in df.columns and 'Confidence' not in df.columns:
            df['Confidence'] = df[['Positive_Prob', 'Negative_Prob',
                                   'Neutral_Prob']].max(axis=1).fillna(0.0)

    # Final pass: cyclical + all lags + interactions.
    df = engineer_all_features(df)
    return df.sort_values('Date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecasts — rolling trails per horizon
# ---------------------------------------------------------------------------

def _next_trading_day_after(last_date: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekday after `last_date` (no holiday calendar)."""
    nxt = last_date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:  # Sat=5, Sun=6
        nxt = nxt + pd.Timedelta(days=1)
    return nxt


def compute_forecast_trails(
    db,
    max_horizon: int = 7,
    window_days: int = 90,
) -> Dict:
    """
    For each horizon h ∈ {1..max_horizon} produce a rolling forecast trail:
        for every anchor date d in the trailing `window_days` window,
            predict the price at `d + h trading days`.

    Returns a JSON-serialisable dict with structure:

        {
          "horizons": [1..max_horizon],
          "winners":  {1: "cpo_hmm", ...},
          "configs":  {1: "C2", ...},
          "trails": [
            {"horizon": 1, "config": "C2", "tag": "cpo_hmm",
             "points": [{"anchor_date":"2026-...","predicted_date":"...","predicted_price":...}, ...]},
            ...
          ],
          "metrics": {tag: {h: {"BASE": {...}, "CSA": {...}}}}  // copy from winners.json
        }
    """
    winners_payload = load_winners()
    winners = {int(h): tag for h, tag in winners_payload['winners_by_horizon'].items()}
    configs = {int(h): cfg for h, cfg in winners_payload['configs_by_horizon'].items()}

    df = build_inference_frame(db)
    if df.empty:
        return {'horizons': [], 'winners': {}, 'configs': {}, 'trails': [], 'metrics': {}}

    dates_sorted = df['Date'].tolist()
    last_idx = len(dates_sorted) - 1
    next_td  = _next_trading_day_after(dates_sorted[-1])

    cutoff = df['Date'].max() - pd.Timedelta(days=window_days)
    window_idx = df.index[df['Date'] >= cutoff].tolist()

    trails = []
    for h in range(1, max_horizon + 1):
        tag = winners.get(h)
        if not tag:
            logger.warning(f'No winner for horizon {h}; skipping trail.')
            continue
        try:
            model, scaler, feature_cols = load_model(tag, h, variant='csa')
        except Exception as e:
            logger.warning(f'Failed to load model for h={h} tag={tag}: {e}')
            continue

        # Per the staged spec, every horizon's last prediction lands on
        # `next_td` (one trading day past the last actual) so all 7 trails
        # converge at the same future point. Trails differ only in how far
        # back their first point is — h=1 starts at window_start+1,
        # h=7 starts at window_start+7.
        points = []
        _logged_missing: set = set()
        for i in window_idx:
            target_idx = i + h
            if target_idx > last_idx + 1:
                # Would extend past the single shared future point — skip.
                continue
            pred_date = (dates_sorted[target_idx]
                         if target_idx <= last_idx else next_td)

            row = df.iloc[[i]]
            try:
                X = select_feature_matrix(row, feature_cols)
            except ValueError as e:
                key = str(e)[:120]
                if key not in _logged_missing:
                    logger.warning(f'h={h} feature mismatch (logged once): {e}')
                    _logged_missing.add(key)
                continue
            if np.isnan(X).any():
                continue
            if scaler is not None:
                X = scaler.transform(X)
            log_ret = float(model.predict(X)[0])

            anchor_close = float(row['Close'].iloc[0])
            pred_price = float(anchor_close * np.exp(np.clip(log_ret, -10, 10)))

            points.append({
                'anchor_date':     dates_sorted[i].strftime('%Y-%m-%d'),
                'anchor_price':    round(anchor_close, 2),
                'predicted_date':  pred_date.strftime('%Y-%m-%d'),
                'predicted_price': round(pred_price, 2),
                'log_return':      round(log_ret, 6),
            })

        trails.append({
            'horizon': h,
            'tag':     tag,
            'config':  configs.get(h, ''),
            'points':  points,
        })

    return {
        'horizons':     list(range(1, max_horizon + 1)),
        'winners':      {str(h): t for h, t in winners.items()},
        'configs':      {str(h): c for h, c in configs.items()},
        'trails':       trails,
        'metrics':      winners_payload.get('metrics', {}),
        'tag_to_config': winners_payload.get('tag_to_config', {}),
        'generated_at': datetime.utcnow().isoformat() + 'Z',
    }
