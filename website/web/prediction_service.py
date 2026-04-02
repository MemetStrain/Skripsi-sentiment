"""
Prediction Service - Real ML predictions using horizon forecasting.

Replicates the data merging, feature engineering, and model training logic
from prediction/horizon_forecast.py, adapted to read from Firestore.

Models are retrained on-the-fly using stored hyperparameters from Firestore.
This avoids storing large serialized models (Firestore 1MB doc limit) and
keeps predictions consistent with the latest stored data.
"""

import time
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from firebase_admin import firestore

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Daily interval config (matches horizon_forecast.py)
DAILY_CONFIG = {
    'seasonal_period': 5,
    'base_lag_periods': [1, 2, 3, 5, 10, 20],
    'min_samples': 100,
    'test_ratio': 0.2,
}

VALID_HORIZONS = [1, 2, 3, 4, 5, 6, 7]
VALID_MODELS = ['xgboost', 'random_forest', 'arimax', 'sarimax']
VALID_VARIANTS = ['base', 'csa']

# Default hyperparameters (matches horizon_forecast.py BASE_PARAMS)
BASE_PARAMS = {
    'xgboost': {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1,
    },
    'random_forest': {
        'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
        'min_samples_leaf': 2, 'max_features': 0.7,
    },
    'arimax': {'order': (2, 1, 2), 'seasonal_order': (0, 0, 0, 0)},
    'sarimax': {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 5)},
}

# Module-level cache for merged DataFrame
_cache = {
    'merged_df': None,
    'timestamp': 0,
}
CACHE_TTL = 300  # 5 minutes


# =============================================================================
# Data Fetching from Firestore
# =============================================================================

def _fetch_collection_as_df(collection_name: str, date_column: str = 'Date') -> pd.DataFrame:
    """Fetch all documents from a Firestore collection into a DataFrame."""
    db = firestore.client()
    docs = db.collection(collection_name).stream()

    rows = []
    for doc in docs:
        data = doc.to_dict()
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    return df


def fetch_model_inputs() -> Dict[str, pd.DataFrame]:
    """Fetch CPO variables, sentiment, and HMM data from Firestore."""
    cpo = _fetch_collection_as_df('CpoVariables')
    sentiment = _fetch_collection_as_df('SentimentAggregate')
    hmm = _fetch_collection_as_df('HmmStatesResults')
    return {'cpo': cpo, 'sentiment': sentiment, 'hmm': hmm}


# =============================================================================
# Data Merging (replicates horizon_forecast.py load_and_merge_data)
# =============================================================================

def merge_model_inputs(cpo: pd.DataFrame, sentiment: pd.DataFrame,
                       hmm: pd.DataFrame) -> pd.DataFrame:
    """Merge CPO, sentiment, and HMM DataFrames on Date (Daily interval)."""
    if cpo.empty or sentiment.empty or hmm.empty:
        raise ValueError("One or more input DataFrames are empty")

    # Rename sentiment columns to match horizon_forecast.py convention
    rename_map = {
        'Article_Count': 'Article_Count',
        'Combined_Positive_Prob': 'Positive_Prob',
        'Combined_Negative_Prob': 'Negative_Prob',
        'Combined_Neutral_Prob': 'Neutral_Prob',
        'Combined_Confidence': 'Confidence',
    }
    sentiment = sentiment.rename(columns=rename_map)
    keep_cols = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
                 'Neutral_Prob', 'Confidence', 'Sentiment_Score']
    sentiment = sentiment[[c for c in keep_cols if c in sentiment.columns]]

    # Rename HMM columns to avoid collision with CPO
    hmm = hmm.rename(columns={
        'Close': 'HMM_Close', 'Log_Return': 'HMM_Log_Return',
        'Volatility': 'HMM_Volatility', 'RSI': 'HMM_RSI',
        'MACD': 'HMM_MACD', 'State': 'HMM_State',
        'State_Label': 'HMM_State_Label',
    })

    # Inner join on Date (Daily)
    merged = cpo.merge(sentiment, on='Date', how='inner', suffixes=('', '_sent'))
    merged = merged.merge(hmm, on='Date', how='inner', suffixes=('', '_hmm'))
    merged = merged.sort_values('Date').reset_index(drop=True)

    # One-hot encode HMM states (top 5)
    if 'HMM_State_Label' in merged.columns:
        top_states = merged['HMM_State_Label'].value_counts().head(5).index.tolist()
        for state in top_states:
            col_name = f'HMM_{state.replace(" ", "_").replace("-", "_")}'
            merged[col_name] = (merged['HMM_State_Label'] == state).astype(int)
        merged = merged.drop(columns=['HMM_State_Label'])

    if len(merged) < DAILY_CONFIG['min_samples']:
        raise ValueError(
            f"Merged dataset has only {len(merged)} rows, "
            f"minimum required: {DAILY_CONFIG['min_samples']}")

    return merged


def get_merged_data() -> pd.DataFrame:
    """Get merged data with caching."""
    now = time.time()
    if _cache['merged_df'] is not None and (now - _cache['timestamp']) < CACHE_TTL:
        return _cache['merged_df'].copy()

    inputs = fetch_model_inputs()
    merged = merge_model_inputs(inputs['cpo'], inputs['sentiment'], inputs['hmm'])
    _cache['merged_df'] = merged
    _cache['timestamp'] = now
    return merged.copy()


# =============================================================================
# Feature Engineering (horizon-aware, replicates horizon_forecast.py)
# =============================================================================

def engineer_features(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features for a specific forecast horizon.
    Uses only safe lags (>= horizon) to prevent data leakage.
    """
    cfg = DAILY_CONFIG
    df = df.copy()

    # Temporal features
    df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)
    df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
    df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)

    # Lag features - only lags >= horizon to prevent look-ahead bias
    safe_lags = [lag for lag in cfg['base_lag_periods'] if lag >= horizon]
    if not safe_lags:
        safe_lags = [horizon]

    lag_cols = ['Close', 'Sentiment_Score', 'HMM_State']
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in safe_lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Interaction features
    if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
        df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']
    if 'HMM_Volatility' in df.columns and 'RSI' in df.columns:
        df['Volatility_x_RSI'] = df['HMM_Volatility'] * df['RSI']

    # Target: Close price h steps ahead
    df['Target'] = df['Close'].shift(-horizon)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    # Feature columns (exclude non-numeric and special columns)
    exclude = ['Date', 'Target', 'Dominant_Sentiment', 'HMM_Close']
    feature_cols = [c for c in df.columns
                    if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    return df, feature_cols


# =============================================================================
# Model Helpers (replicates horizon_forecast.py)
# =============================================================================

def create_sklearn_model(model_type: str, params: Dict):
    """Create an XGBoost or Random Forest model with given parameters."""
    p = dict(params)
    p.pop('random_state', None)
    p.pop('csa_best_score', None)
    p.pop('csa_iterations', None)

    if model_type == 'xgboost':
        valid_keys = {'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                      'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha',
                      'reg_lambda', 'max_delta_step'}
        filtered = {k: (int(v) if k in ('n_estimators', 'max_depth', 'min_child_weight') else v)
                    for k, v in p.items() if k in valid_keys}
        return XGBRegressor(**filtered, verbosity=0, random_state=RANDOM_STATE)
    elif model_type == 'random_forest':
        int_keys = ('n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf')
        filtered = {k: (int(v) if k in int_keys else v) for k, v in p.items()}
        return RandomForestRegressor(**filtered, random_state=RANDOM_STATE)


def select_top_exog(X: np.ndarray, y: np.ndarray, n: int = 10):
    """Select top-N features by absolute correlation with target."""
    correlations = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 0 else 0
        for i in range(X.shape[1])
    ])
    top_indices = np.argsort(correlations)[-n:]
    return X[:, top_indices], top_indices.tolist()


# =============================================================================
# Firestore Parameter/Metrics Fetching
# =============================================================================

def fetch_horizon_params(horizon: int) -> Dict:
    """Fetch model parameters for a specific horizon from Firestore."""
    db = firestore.client()
    doc_ref = db.collection('HorizonModelParameters').document(f'Daily_h{horizon}')
    doc = doc_ref.get()
    if not doc.exists:
        raise ValueError(f"No parameters found for horizon {horizon}")
    return doc.to_dict()


def fetch_horizon_metrics(horizon: int) -> Dict:
    """Fetch model metrics for a specific horizon from Firestore."""
    db = firestore.client()
    doc_ref = db.collection('HorizonModelMetrics').document(f'Daily_h{horizon}')
    doc = doc_ref.get()
    if not doc.exists:
        return {}
    return doc.to_dict()


def get_model_metrics(model_type: str, variant: str, horizon: int) -> Dict:
    """Get metrics for a specific model/variant/horizon combination."""
    data = fetch_horizon_metrics(horizon)
    metrics_list = data.get('metrics', [])
    opt_label = 'BASE' if variant == 'base' else 'CSA'
    for m in metrics_list:
        if m.get('model') == model_type and m.get('optimization') == opt_label:
            return {
                'mape': m.get('mape', 0),
                'rmse': m.get('rmse', 0),
                'r2': m.get('r2', 0),
                'directional_accuracy': m.get('directional_accuracy', 0),
            }
    return {'mape': 0, 'rmse': 0, 'r2': 0, 'directional_accuracy': 0}


def get_all_horizon_metrics(horizon: int) -> List[Dict]:
    """Get metrics for all models at a given horizon."""
    data = fetch_horizon_metrics(horizon)
    return data.get('metrics', [])


# =============================================================================
# Main Prediction Function
# =============================================================================

def run_prediction(model_type: str, variant: str, horizon: int) -> Dict:
    """
    Run a prediction for a given model type, variant, and horizon.

    Steps:
    1. Fetch merged data from Firestore (cached)
    2. Engineer horizon-aware features
    3. Fetch stored hyperparameters
    4. Train model on ALL data with stored hyperparams
    5. Predict the next value (horizon-steps ahead)

    Returns dict with prediction value and metrics.
    """
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of {VALID_HORIZONS}")
    if model_type not in VALID_MODELS:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {VALID_MODELS}")
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{variant}'. Must be one of {VALID_VARIANTS}")

    # 1. Get merged data
    merged = get_merged_data()

    # 2. Engineer features for this horizon
    df, feature_cols = engineer_features(merged, horizon)

    # 3. Fetch stored hyperparameters
    params_key = f'{model_type}_{variant}'
    try:
        horizon_params = fetch_horizon_params(horizon)
        model_params = horizon_params.get('models', {}).get(params_key)
    except ValueError:
        model_params = None

    if not model_params:
        model_params = BASE_PARAMS.get(model_type, {})

    # 4. Prepare data
    X = df[feature_cols].values
    y = df['Target'].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train and predict
    if model_type in ('xgboost', 'random_forest'):
        prediction = _predict_sklearn(model_type, model_params, X_scaled, y, scaler, df, feature_cols)
    else:
        prediction = _predict_statsmodels(model_type, model_params, X_scaled, y)

    # 6. Get stored metrics
    metrics = get_model_metrics(model_type, variant, horizon)

    # 7. Build response
    last_date = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]

    return {
        'model_type': model_type,
        'variant': variant,
        'horizon': horizon,
        'last_date': last_date.strftime('%Y-%m-%d'),
        'last_close': round(float(last_close), 2),
        'predicted_price': round(float(prediction), 2),
        'metrics': metrics,
    }


def _predict_sklearn(model_type, params, X_scaled, y, scaler, df, feature_cols):
    """Train sklearn/xgboost model and predict next value."""
    model = create_sklearn_model(model_type, params)
    model.fit(X_scaled, y)
    # Predict using the last row of features
    last_features = X_scaled[-1:, :]
    return model.predict(last_features)[0]


def _predict_statsmodels(model_type, params, X_scaled, y):
    """Train ARIMAX/SARIMAX and predict next value."""
    order = tuple(int(x) for x in params.get('order', (2, 1, 2)))
    seasonal_order = tuple(int(x) for x in params.get('seasonal_order', (0, 0, 0, 0)))

    # Select top exogenous features
    n_exog = min(10, X_scaled.shape[1])
    exog, _ = select_top_exog(X_scaled, y, n=n_exog)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model = SM_SARIMAX(
                endog=y, exog=exog,
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            # Forecast 1 step using the last exog row
            forecast = result.forecast(steps=1, exog=exog[-1:, :])
            return float(forecast.iloc[0]) if hasattr(forecast, 'iloc') else float(forecast[0])
        except Exception as e:
            logger.warning(f"Statsmodels prediction failed: {e}")
            return float(y[-1])  # fallback to last known value
