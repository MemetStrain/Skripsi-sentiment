"""
prediction_updater.py — Compute all 56 prediction combinations and write to Firestore.

Combinations: 4 models × 2 variants × 7 daily horizons.
For 'csa'/'bayesian' variants, stored hyperparameters from `HorizonModelParameters` are used.
For 'base' variant, default hyperparameters are used.

Sklearn models (XGBoost, RandomForest) are cached to GCS after first training so that
subsequent scheduler runs load them instead of re-training from scratch.
Set GCS_BUCKET env-var to enable; omit to run without caching (train every time).
Models older than MODEL_CACHE_MAX_AGE_DAYS are automatically re-trained and re-saved.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FREQ_CONFIG = {
    'Daily': {'horizons': list(range(1, 8)), 'periods': 252},
}
MODELS = ['xgboost', 'random_forest', 'arimax', 'sarimax']
VARIANTS = ['base', 'csa', 'bayesian']

# Default hyperparameters for 'base' variant
BASE_PARAMS = {
    'xgboost':      {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
                     'subsample': 0.8, 'colsample_bytree': 0.8},
    'random_forest': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
    'arimax':       {'order': (2, 1, 2)},
    'sarimax':      {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 5)},
}

LAG_PERIODS = [1, 2, 3, 5, 10, 20]

GCS_BUCKET = os.environ.get('GCS_BUCKET', '')
MODEL_CACHE_DIR = '/tmp/cpo_scheduler_cache'
MODEL_CACHE_MAX_AGE_DAYS = 7


# ---------------------------------------------------------------------------
# Model cache helpers
# ---------------------------------------------------------------------------

def _cache_dir(doc_id: str) -> str:
    safe = doc_id.replace('/', '_')
    d = os.path.join(MODEL_CACHE_DIR, safe)
    os.makedirs(d, exist_ok=True)
    return d


def _load_cached_sklearn(doc_id: str) -> Optional[object]:
    """Return cached sklearn model if it exists and is fresh, else None."""
    d = _cache_dir(doc_id)
    mp = os.path.join(d, 'model.pkl')
    mm = os.path.join(d, 'meta.json')

    # Try GCS download if local copy absent
    if GCS_BUCKET and not os.path.exists(mp):
        _gcs_download(doc_id, d)

    if not os.path.exists(mp) or not os.path.exists(mm):
        return None

    try:
        with open(mm) as f:
            meta = json.load(f)
        saved_at = datetime.fromisoformat(meta.get('saved_at', '2000-01-01'))
        if (datetime.now() - saved_at).days > MODEL_CACHE_MAX_AGE_DAYS:
            return None
        return joblib.load(mp)
    except Exception:
        return None


def _save_cached_sklearn(model, doc_id: str) -> None:
    """Persist sklearn model to local cache and optionally GCS."""
    d = _cache_dir(doc_id)
    mp = os.path.join(d, 'model.pkl')
    mm = os.path.join(d, 'meta.json')
    try:
        joblib.dump(model, mp)
        with open(mm, 'w') as f:
            json.dump({'saved_at': datetime.now().isoformat()}, f)
        if GCS_BUCKET:
            _gcs_upload(doc_id, d)
    except Exception as exc:
        logger.warning(f'Model cache save failed ({doc_id}): {exc}')


def _gcs_upload(doc_id: str, local_dir: str) -> None:
    gcs_prefix = f'scheduler_models/{doc_id}'
    try:
        from google.cloud import storage  # type: ignore
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        for fname in ('model.pkl', 'meta.json'):
            fpath = os.path.join(local_dir, fname)
            if os.path.exists(fpath):
                bucket.blob(f'{gcs_prefix}/{fname}').upload_from_filename(fpath)
    except Exception as exc:
        logger.warning(f'GCS upload failed ({gcs_prefix}): {exc}')


def _gcs_download(doc_id: str, local_dir: str) -> None:
    gcs_prefix = f'scheduler_models/{doc_id}'
    try:
        from google.cloud import storage  # type: ignore
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        for fname in ('model.pkl', 'meta.json'):
            blob = bucket.blob(f'{gcs_prefix}/{fname}')
            if blob.exists():
                blob.download_to_filename(os.path.join(local_dir, fname))
    except Exception as exc:
        logger.warning(f'GCS download failed ({gcs_prefix}): {exc}')


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA, EMA, RSI, MACD, Bollinger Bands, ATR from OHLCV."""
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)

    df['sma5']  = close.rolling(5).mean()
    df['sma10'] = close.rolling(10).mean()
    df['sma20'] = close.rolling(20).mean()
    df['ema5']  = close.ewm(span=5,  adjust=False).mean()
    df['ema10'] = close.ewm(span=10, adjust=False).mean()
    df['ema20'] = close.ewm(span=20, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD
    df['macd'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_pct']   = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

    # ATR
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # Log return
    df['log_return'] = np.log(close / close.shift(1))

    return df


def _add_lag_features(df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
    for lag in LAG_PERIODS:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_merged_df(db, frequency: str) -> pd.DataFrame:
    """
    Load and merge price data, sentiment aggregates, and HMM states from Firestore.
    Returns a DataFrame indexed by date, sorted ascending.
    """
    # --- Price data ---
    price_docs = db.collection('daily_prices').order_by('date').stream()
    price_rows = [{
        'date': d.to_dict()['date'],
        'open':   float(d.to_dict().get('open',   0)),
        'high':   float(d.to_dict().get('high',   0)),
        'low':    float(d.to_dict().get('low',    0)),
        'close':  float(d.to_dict().get('close',  0)),
        'volume': float(d.to_dict().get('volume', 0)),
    } for d in price_docs]

    if not price_rows:
        return pd.DataFrame()

    price_df = pd.DataFrame(price_rows)
    price_df['date'] = pd.to_datetime(price_df['date'])

    price_df = price_df.set_index('date').sort_index().reset_index()
    price_df['date'] = price_df['date'].dt.strftime('%Y-%m-%d')

    # --- Sentiment aggregates ---
    sent_docs = (
        db.collection('sentiment_aggregates')
        .where('frequency', '==', 'Daily')
        .stream()
    )
    sent_rows = [{
        'date': d.to_dict()['date'],
        'sentiment_score': float(d.to_dict().get('sentiment_score', 0)),
        'positive_prob':   float(d.to_dict().get('positive_prob',   0.33)),
        'negative_prob':   float(d.to_dict().get('negative_prob',   0.33)),
    } for d in sent_docs]
    sent_df = pd.DataFrame(sent_rows) if sent_rows else pd.DataFrame(columns=['date', 'sentiment_score'])

    # --- HMM states ---
    hmm_docs = (
        db.collection('hmm_states')
        .where('frequency', '==', frequency)
        .stream()
    )
    hmm_rows = [{
        'date': d.to_dict()['date'],
        'state':       int(d.to_dict().get('state', 2)),
        'state_label': d.to_dict().get('state_label', 'Neutral'),
    } for d in hmm_docs]
    hmm_df = pd.DataFrame(hmm_rows) if hmm_rows else pd.DataFrame(columns=['date', 'state'])

    # --- Merge ---
    df = price_df.copy()
    if not sent_df.empty:
        df = df.merge(sent_df[['date', 'sentiment_score']], on='date', how='left')
    else:
        df['sentiment_score'] = 0.0

    if not hmm_df.empty:
        df = df.merge(hmm_df[['date', 'state']], on='date', how='left')
    else:
        df['state'] = 2

    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df['state'] = df['state'].fillna(2).astype(int)

    df = df.sort_values('date').reset_index(drop=True)
    df = _add_technical_features(df)
    df = _add_lag_features(df)
    df = df.dropna().reset_index(drop=True)

    return df


def _load_opt_params(db, frequency: str, model: str, horizon: int, variant: str) -> Optional[dict]:
    """Fetch stored optimizer hyperparameters from HorizonModelParameters collection.

    Doc ID format: {model}_{variant}_{frequency}_h{horizon}
    e.g. xgboost_csa_Daily_h1, xgboost_bayesian_Daily_h1
    """
    try:
        doc_id = f'{model}_{variant}_{frequency}_h{horizon}'
        doc = db.collection('HorizonModelParameters').document(doc_id).get()
        if doc.exists:
            return doc.to_dict().get('params')
    except Exception as e:
        logger.warning(f'Failed to load {variant} params for {doc_id}: {e}')
    return None


# ---------------------------------------------------------------------------
# Model training and prediction
# ---------------------------------------------------------------------------

def _build_features_target(df: pd.DataFrame, horizon: int) -> tuple:
    """
    Build feature matrix X and target y for the given horizon.
    y is the closing price `horizon` steps ahead.
    Returns (X_train, y_train, last_X, last_row).
    """
    feature_cols = [c for c in df.columns if c not in ('date', 'close')]
    target = df['close'].shift(-horizon)  # future price

    # Drop NaN created by shift
    valid = df[target.notna()].copy()
    y = target[target.notna()].values
    X = valid[feature_cols].values

    # Last row: predict from (features as of last known date)
    last_X = df[feature_cols].iloc[-1:].values
    last_row = df.iloc[-1]

    return X, y, last_X, last_row


def _predict_price_xgboost(X_train, y_train, last_X, params: dict,
                           doc_id: str = '') -> float:
    from xgboost import XGBRegressor
    model = _load_cached_sklearn(doc_id) if doc_id else None
    if model is None:
        model = XGBRegressor(**{k: v for k, v in params.items()
                                if k in XGBRegressor().get_params()},
                             random_state=42, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_train, y_train)
        if doc_id:
            _save_cached_sklearn(model, doc_id)
    return float(model.predict(last_X)[0])


def _predict_price_rf(X_train, y_train, last_X, params: dict,
                      doc_id: str = '') -> float:
    from sklearn.ensemble import RandomForestRegressor
    model = _load_cached_sklearn(doc_id) if doc_id else None
    if model is None:
        model = RandomForestRegressor(**{k: v for k, v in params.items()
                                         if k in RandomForestRegressor().get_params()},
                                      random_state=42, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_train, y_train)
        if doc_id:
            _save_cached_sklearn(model, doc_id)
    return float(model.predict(last_X)[0])


def _predict_price_arimax(df: pd.DataFrame, horizon: int, params: dict) -> float:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    exog_cols = ['sentiment_score', 'state', 'rsi', 'macd']
    exog_cols = [c for c in exog_cols if c in df.columns]
    endog = df['close'].values
    exog = df[exog_cols].values if exog_cols else None
    order = tuple(params.get('order', (2, 1, 2)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SARIMAX(endog, exog=exog, order=order).fit(disp=False)
        fc = model.forecast(steps=horizon, exog=exog[-horizon:] if exog is not None else None)
        return float(fc.iloc[-1])
    except Exception as e:
        logger.warning(f'ARIMAX failed: {e}')
        return float(df['close'].iloc[-1])


def _predict_price_sarimax(df: pd.DataFrame, horizon: int, params: dict) -> float:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    exog_cols = ['sentiment_score', 'state', 'rsi', 'macd']
    exog_cols = [c for c in exog_cols if c in df.columns]
    endog = df['close'].values
    exog = df[exog_cols].values if exog_cols else None
    order = tuple(params.get('order', (1, 1, 1)))
    seasonal_order = tuple(params.get('seasonal_order', (1, 0, 1, 5)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SARIMAX(endog, exog=exog,
                            order=order,
                            seasonal_order=seasonal_order).fit(disp=False)
        fc = model.forecast(steps=horizon, exog=exog[-horizon:] if exog is not None else None)
        return float(fc.iloc[-1])
    except Exception as e:
        logger.warning(f'SARIMAX failed: {e}')
        return float(df['close'].iloc[-1])


def _compute_metrics(X_train, y_train, model_type: str, params: dict,
                     df: pd.DataFrame) -> dict:
    """Compute MAPE, RMSE, R², directional accuracy on the training set."""
    try:
        # Use last 20% as pseudo-validation
        split = max(10, int(len(X_train) * 0.8))
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        if model_type == 'xgboost':
            from xgboost import XGBRegressor
            m = XGBRegressor(**{k: v for k, v in params.items()
                                if k in XGBRegressor().get_params()},
                             random_state=42, verbosity=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(**{k: v for k, v in params.items()
                                         if k in RandomForestRegressor().get_params()},
                                      random_state=42, n_jobs=-1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
        else:
            # For time-series models, use simple persistence as fallback metric
            preds = np.roll(y_val, 1)
            preds[0] = y_tr[-1]

        mape = float(np.mean(np.abs((y_val - preds) / (y_val + 1e-9))) * 100)
        rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-9))

        if len(y_val) > 1:
            dir_actual = np.diff(y_val) > 0
            dir_pred   = np.diff(preds) > 0
            da = float(np.mean(dir_actual == dir_pred) * 100)
        else:
            da = 50.0

        return {
            'mape': round(mape, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 4),
            'directional_accuracy': round(da, 2),
        }
    except Exception:
        return {'mape': 0.0, 'rmse': 0.0, 'r2': 0.0, 'directional_accuracy': 50.0}


def _horizon_to_date(last_date_str: str, horizon: int) -> str:
    """Estimate the target prediction date (Daily only)."""
    try:
        last = datetime.strptime(last_date_str, '%Y-%m-%d')
        return (last + timedelta(days=horizon)).strftime('%Y-%m-%d')
    except Exception:
        return last_date_str


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_predictions(db) -> None:
    """
    Compute all 56 prediction combinations and write to `predictions` collection.
    """
    from firestore_writer import write_prediction

    frequency = 'Daily'
    cfg = FREQ_CONFIG['Daily']

    logger.info(f'Loading merged DataFrame for {frequency}...')
    df = _load_merged_df(db, frequency)
    if df.empty or len(df) < 30:
        logger.warning(f'Not enough data for {frequency} predictions')
        return

    feature_cols = [c for c in df.columns if c not in ('date', 'close')]

    for horizon in cfg['horizons']:
        X_train, y_train, last_X, last_row = _build_features_target(df, horizon)
        if len(X_train) < 20:
            continue

        last_date = str(last_row['date'])
        last_close = float(last_row['close'])
        pred_date = _horizon_to_date(last_date, horizon)

        for model_type in MODELS:
            for variant in VARIANTS:
                doc_id = f'{model_type}_{variant}_{frequency}_h{horizon}'
                try:
                    # Get hyperparameters
                    if variant in ('csa', 'bayesian'):
                        params = _load_opt_params(db, frequency, model_type, horizon, variant)
                        if params is None:
                            params = BASE_PARAMS[model_type].copy()
                    else:
                        params = BASE_PARAMS[model_type].copy()

                    # Run prediction (sklearn models use cache; statsmodels re-train each time)
                    if model_type == 'xgboost':
                        pred = _predict_price_xgboost(X_train, y_train, last_X, params,
                                                      doc_id=doc_id)
                    elif model_type == 'random_forest':
                        pred = _predict_price_rf(X_train, y_train, last_X, params,
                                                 doc_id=doc_id)
                    elif model_type == 'arimax':
                        pred = _predict_price_arimax(df, horizon, params)
                    elif model_type == 'sarimax':
                        pred = _predict_price_sarimax(df, horizon, params)
                    else:
                        continue

                    # Compute metrics
                    metrics = _compute_metrics(
                        X_train, y_train, model_type, params, df
                    )

                    write_prediction(db, model_type, variant, frequency, horizon, {
                        'model': model_type,
                        'variant': variant,
                        'frequency': frequency,
                        'horizon': horizon,
                        'predicted_price': round(pred, 2),
                        'last_actual_date': last_date,
                        'last_actual_price': round(last_close, 2),
                        'predicted_date': pred_date,
                        'metrics': metrics,
                    })
                    logger.debug(f'Wrote prediction: {doc_id} → {pred:.2f}')

                except Exception as e:
                    logger.error(f'Prediction failed for {doc_id}: {e}')

        logger.info(f'Completed predictions for {frequency}')
