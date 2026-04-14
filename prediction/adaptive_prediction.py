"""
Adaptive CPO Price Prediction with CSA Optimization
=====================================================

Daily CPO price prediction with CSA Optimization.
Uses 4 model types (XGBoost, Random Forest, ARIMAX, SARIMAX), each as
base and CSA-optimized variants.

Data sources:
- CPO technical variables (cpo/output/cpo_variables_Daily.csv)
- News sentiment aggregates (news/output/sentiment_aggregate_Daily.csv)
- HMM market states (markov/output/hmm_states_results_Daily.csv)

Usage:
    python adaptive_prediction.py --interval daily
"""

import os
import sys
import json
import time
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX

# Add prediction directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crow_search_optimizer import CrowSearchOptimizer, ParameterSpec, CSAResult
from bayesian_optimizer import BayesianTimeSeriesOptimizer

warnings.filterwarnings('ignore')

# Plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

RANDOM_STATE = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 1. IntervalConfig
# =============================================================================

@dataclass
class IntervalConfig:
    """Configuration for a specific data interval."""
    name: str
    cpo_file: str
    sentiment_file: str
    hmm_file: str
    seasonal_period: int
    lag_periods: List[int]
    min_samples: int
    test_ratio: float = 0.2

    @staticmethod
    def get(interval: str) -> 'IntervalConfig':
        interval = interval.capitalize()
        configs = {
            'Daily': IntervalConfig(
                name='Daily',
                cpo_file=os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Daily.csv'),
                sentiment_file=os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Daily.csv'),
                hmm_file=os.path.join(PROJECT_ROOT, 'markov', 'output', 'hmm_states_results_Daily.csv'),
                seasonal_period=5,
                lag_periods=[1, 2, 3, 5, 10, 20],
                min_samples=100,
            ),
        }
        if interval not in configs:
            raise ValueError(f"Invalid interval '{interval}'. Only 'Daily' is supported.")
        return configs[interval]


# =============================================================================
# 2. DataLoader
# =============================================================================

class DataLoader:
    """Loads and merges CPO, sentiment, and HMM data for a given interval."""

    def __init__(self, config: IntervalConfig):
        self.config = config

    def load_cpo(self) -> pd.DataFrame:
        print(f"  Loading CPO data from {os.path.basename(self.config.cpo_file)}...")
        df = pd.read_csv(self.config.cpo_file)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"    Rows: {len(df)}, Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df

    def load_sentiment(self) -> pd.DataFrame:
        print(f"  Loading sentiment data from {os.path.basename(self.config.sentiment_file)}...")
        df = pd.read_csv(self.config.sentiment_file)

        df['Date'] = pd.to_datetime(df['Date'])
        rename_map = {
            'Article_Count': 'Article_Count',
            'Combined_Positive_Prob': 'Positive_Prob',
            'Combined_Negative_Prob': 'Negative_Prob',
            'Combined_Neutral_Prob': 'Neutral_Prob',
            'Combined_Confidence': 'Confidence',
        }

        df = df.rename(columns=rename_map)
        keep_cols = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
                     'Neutral_Prob', 'Confidence', 'Sentiment_Score']
        df = df[[c for c in keep_cols if c in df.columns]]
        print(f"    Rows: {len(df)}, Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df

    def load_hmm(self) -> pd.DataFrame:
        print(f"  Loading HMM data from {os.path.basename(self.config.hmm_file)}...")
        df = pd.read_csv(self.config.hmm_file)
        df['Date'] = pd.to_datetime(df['Date'])
        # Rename to avoid collision with CPO columns
        df = df.rename(columns={
            'Close': 'HMM_Close',
            'Log_Return': 'HMM_Log_Return',
            'Volatility': 'HMM_Volatility',
            'RSI': 'HMM_RSI',
            'MACD': 'HMM_MACD',
            'State': 'HMM_State',
            'State_Label': 'HMM_State_Label',
        })
        print(f"    Rows: {len(df)}, Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df

    def _merge_by_date(self, cpo: pd.DataFrame, sentiment: pd.DataFrame,
                       hmm: pd.DataFrame) -> pd.DataFrame:
        """Merge all three datasets on Date via inner join."""
        merged = cpo.merge(sentiment, on='Date', how='inner', suffixes=('', '_sent'))
        merged = merged.merge(hmm, on='Date', how='inner', suffixes=('', '_hmm'))
        return merged

    def _onehot_hmm_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode HMM state labels, keeping top-5 most frequent."""
        if 'HMM_State_Label' not in df.columns:
            return df
        top_states = df['HMM_State_Label'].value_counts().head(5).index.tolist()
        for state in top_states:
            col_name = f'HMM_{state.replace(" ", "_").replace("-", "_")}'
            df[col_name] = (df['HMM_State_Label'] == state).astype(int)
        df = df.drop(columns=['HMM_State_Label'])
        return df

    def merge_all(self) -> pd.DataFrame:
        """Load all sources and merge into a single DataFrame."""
        print(f"\n{'='*60}")
        print(f"Loading {self.config.name} data...")
        print(f"{'='*60}")

        cpo = self.load_cpo()
        sentiment = self.load_sentiment()
        hmm = self.load_hmm()

        print(f"\n  Merging datasets...")
        merged = self._merge_by_date(cpo, sentiment, hmm)

        merged = merged.sort_values('Date').reset_index(drop=True)
        merged = self._onehot_hmm_states(merged)

        print(f"  Merged rows: {len(merged)}")
        print(f"  Date range: {merged['Date'].min()} to {merged['Date'].max()}")
        print(f"  Columns: {len(merged.columns)}")

        if len(merged) < self.config.min_samples:
            raise ValueError(
                f"Merged dataset has only {len(merged)} rows, "
                f"minimum required: {self.config.min_samples}"
            )
        return merged


# =============================================================================
# 3. FeatureEngineer
# =============================================================================

class FeatureEngineer:
    """Builds model-ready feature matrix from merged data."""

    def __init__(self, config: IntervalConfig):
        self.config = config

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add lag, temporal, and interaction features. Returns (df, feature_cols)."""
        df = df.copy()

        # --- Temporal features ---
        df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)

        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)

        # --- Lag features ---
        lag_cols = ['Close', 'Sentiment_Score', 'HMM_State']
        for col in lag_cols:
            if col not in df.columns:
                continue
            for lag in self.config.lag_periods:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # --- Interaction features ---
        if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
            df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']
        if 'HMM_Volatility' in df.columns and 'RSI' in df.columns:
            df['Volatility_x_RSI'] = df['HMM_Volatility'] * df['RSI']

        # --- Target: next-period Close ---
        df['Target'] = df['Close'].shift(-1)

        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)

        # Define feature columns (exclude Date, Target, non-numeric)
        exclude = ['Date', 'Target', 'Dominant_Sentiment', 'HMM_Close']
        feature_cols = [c for c in df.columns
                        if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

        print(f"\n  Feature engineering complete:")
        print(f"    Features: {len(feature_cols)}")
        print(f"    Samples: {len(df)}")

        return df, feature_cols

    def prepare_train_test(self, df: pd.DataFrame, feature_cols: List[str]
                           ) -> Dict[str, any]:
        """Chronological train/test split with RobustScaler."""
        split_idx = int(len(df) * (1 - self.config.test_ratio))

        X = df[feature_cols].values
        y = df['Target'].values
        dates = df['Date'].values

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        train_dates, test_dates = dates[:split_idx], dates[split_idx:]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n  Train/Test split:")
        print(f"    Train: {len(X_train)} samples")
        print(f"    Test:  {len(X_test)} samples")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'scaler': scaler,
            'feature_names': feature_cols,
        }


# =============================================================================
# 4. ModelFactory
# =============================================================================

# Default hyperparameters
BASE_PARAMS = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'random_state': RANDOM_STATE,
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 0.7,
        'random_state': RANDOM_STATE,
    },
    'arimax': {
        'order': (2, 1, 2),
    },
    'sarimax': {
        'order': (1, 1, 1),
        'seasonal_order_pdq': (1, 0, 1),  # 's' added from config
    },
}


def create_sklearn_model(model_type: str, params: Optional[Dict] = None):
    """Create an XGBoost or RandomForest model with given params."""
    p = dict(params or BASE_PARAMS[model_type])
    # Remove random_state from params if present - we set it explicitly
    p.pop('random_state', None)
    if model_type == 'xgboost':
        valid_keys = set(XGBRegressor().get_params().keys())
        filtered = {k: v for k, v in p.items() if k in valid_keys}
        return XGBRegressor(**filtered, verbosity=0, random_state=RANDOM_STATE)
    elif model_type == 'random_forest':
        return RandomForestRegressor(**p, random_state=RANDOM_STATE)


def select_top_exog(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                    n: int = 10) -> Tuple[np.ndarray, List[int]]:
    """Select top-N features by absolute correlation with target."""
    correlations = np.array([abs(np.corrcoef(X[:, i], y)[0, 1])
                             if np.std(X[:, i]) > 0 else 0
                             for i in range(X.shape[1])])
    top_indices = np.argsort(correlations)[-n:]
    return X[:, top_indices], top_indices.tolist()


def train_statsmodels(model_type: str, y_train: np.ndarray, exog_train: np.ndarray,
                      order: tuple, seasonal_order: tuple,
                      verbose: bool = False) -> Optional[object]:
    """Fit ARIMAX or SARIMAX model. Returns fitted result or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SM_SARIMAX(
                endog=y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            return result
    except Exception as e:
        if verbose:
            print(f"    Statsmodels fit failed: {e}")
        return None


def predict_statsmodels(fitted, exog_test: np.ndarray) -> np.ndarray:
    """Forecast using a fitted statsmodels result."""
    try:
        forecast = fitted.forecast(steps=len(exog_test), exog=exog_test)
        return np.array(forecast)
    except Exception:
        return np.full(len(exog_test), np.nan)


# =============================================================================
# 5. CSATimeSeriesOptimizer
# =============================================================================

class CSATimeSeriesOptimizer:
    """CSA optimizer for all 4 model types using TimeSeriesSplit CV."""

    # Parameter spaces
    PARAM_SPACES = {
        'xgboost': [
            ParameterSpec('n_estimators', 50, 500, 'discrete'),
            ParameterSpec('max_depth', 3, 15, 'discrete'),
            ParameterSpec('learning_rate', 0.001, 0.3, 'continuous'),
            ParameterSpec('subsample', 0.6, 1.0, 'continuous'),
            ParameterSpec('colsample_bytree', 0.6, 1.0, 'continuous'),
            ParameterSpec('min_child_weight', 1, 10, 'discrete'),
        ],
        'random_forest': [
            ParameterSpec('n_estimators', 50, 500, 'discrete'),
            ParameterSpec('max_depth', 5, 30, 'discrete'),
            ParameterSpec('min_samples_split', 2, 20, 'discrete'),
            ParameterSpec('min_samples_leaf', 1, 10, 'discrete'),
            ParameterSpec('max_features', 0.3, 0.9, 'continuous'),
        ],
        'arimax': [
            ParameterSpec('p', 0, 5, 'discrete'),
            ParameterSpec('d', 0, 2, 'discrete'),
            ParameterSpec('q', 0, 5, 'discrete'),
        ],
        'sarimax': [
            ParameterSpec('p', 0, 3, 'discrete'),
            ParameterSpec('d', 0, 2, 'discrete'),
            ParameterSpec('q', 0, 3, 'discrete'),
            ParameterSpec('P', 0, 2, 'discrete'),
            ParameterSpec('D', 0, 1, 'discrete'),
            ParameterSpec('Q', 0, 2, 'discrete'),
        ],
    }

    def __init__(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                 config: IntervalConfig, cv_folds: int = 3,
                 population_size: int = 25, max_iterations: int = 50):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.cv_folds = cv_folds
        self.population_size = population_size
        self.max_iterations = max_iterations

        # For ARIMAX/SARIMAX, reduce exog to top features
        if model_type in ('arimax', 'sarimax'):
            self.exog_train, self.exog_indices = select_top_exog(
                X_train, y_train, [], n=min(10, X_train.shape[1]))
        else:
            self.exog_train = None
            self.exog_indices = None

    def _objective_sklearn(self, params: Dict) -> float:
        """CV objective for XGBoost / Random Forest."""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        model = create_sklearn_model(self.model_type, params)
        for train_idx, val_idx in tscv.split(self.X_train):
            try:
                model.fit(self.X_train[train_idx], self.y_train[train_idx])
                y_pred = model.predict(self.X_train[val_idx])
                rmse = np.sqrt(mean_squared_error(self.y_train[val_idx], y_pred))
                scores.append(rmse)
            except Exception:
                scores.append(np.inf)
        return np.mean(scores)

    def _objective_arimax(self, params: Dict) -> float:
        """CV objective for ARIMAX."""
        order = (int(params['p']), int(params['d']), int(params['q']))
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(self.exog_train):
            fitted = train_statsmodels(
                'arimax', self.y_train[train_idx], self.exog_train[train_idx],
                order=order, seasonal_order=(0, 0, 0, 0))
            if fitted is None:
                scores.append(np.inf)
                continue
            preds = predict_statsmodels(fitted, self.exog_train[val_idx])
            if np.any(np.isnan(preds)):
                scores.append(np.inf)
            else:
                scores.append(np.sqrt(mean_squared_error(self.y_train[val_idx], preds)))
        return np.mean(scores)

    def _objective_sarimax(self, params: Dict) -> float:
        """CV objective for SARIMAX."""
        order = (int(params['p']), int(params['d']), int(params['q']))
        seasonal_order = (int(params['P']), int(params['D']), int(params['Q']),
                          self.config.seasonal_period)
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        for train_idx, val_idx in tscv.split(self.exog_train):
            # Ensure training fold has enough data for seasonal period
            if len(train_idx) < self.config.seasonal_period * 2:
                scores.append(np.inf)
                continue
            fitted = train_statsmodels(
                'sarimax', self.y_train[train_idx], self.exog_train[train_idx],
                order=order, seasonal_order=seasonal_order)
            if fitted is None:
                scores.append(np.inf)
                continue
            preds = predict_statsmodels(fitted, self.exog_train[val_idx])
            if np.any(np.isnan(preds)):
                scores.append(np.inf)
            else:
                scores.append(np.sqrt(mean_squared_error(self.y_train[val_idx], preds)))
        return np.mean(scores)

    def optimize(self) -> CSAResult:
        """Run CSA optimization for the configured model type."""
        objectives = {
            'xgboost': self._objective_sklearn,
            'random_forest': self._objective_sklearn,
            'arimax': self._objective_arimax,
            'sarimax': self._objective_sarimax,
        }
        # Use fewer iterations for slower ARIMAX/SARIMAX
        max_iter = self.max_iterations
        pop_size = self.population_size
        if self.model_type in ('arimax', 'sarimax'):
            max_iter = min(max_iter, 30)
            pop_size = min(pop_size, 15)

        optimizer = CrowSearchOptimizer(
            objective_function=objectives[self.model_type],
            parameter_specs=self.PARAM_SPACES[self.model_type],
            population_size=pop_size,
            max_iterations=max_iter,
            awareness_probability=0.1,
            flight_length=2.0,
            early_stopping_patience=10,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        result = optimizer.optimize()
        return result


# =============================================================================
# 6. EvaluationEngine
# =============================================================================

class EvaluationEngine:
    """Calculates prediction quality metrics."""

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute MAPE, RMSE, Directional Accuracy, R-squared."""
        # Handle NaN predictions
        mask = ~np.isnan(y_pred)
        if mask.sum() < 2:
            return {'MAPE': np.inf, 'RMSE': np.inf, 'Directional_Accuracy': 0.0, 'R2': -np.inf}

        yt, yp = y_true[mask], y_pred[mask]

        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)

        # Directional accuracy
        if len(yt) > 1:
            true_dir = np.diff(yt) > 0
            pred_dir = np.diff(yp) > 0
            dir_acc = np.mean(true_dir == pred_dir) * 100
        else:
            dir_acc = 0.0

        return {
            'MAPE': round(mape, 4),
            'RMSE': round(rmse, 4),
            'Directional_Accuracy': round(dir_acc, 4),
            'R2': round(r2, 4),
        }

    @staticmethod
    def build_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Build comparison DataFrame from results dict."""
        rows = []
        for model_name, metrics in results.items():
            parts = model_name.rsplit('_', 1)
            model_type = parts[0]
            optimization = parts[1] if len(parts) > 1 else 'base'
            rows.append({
                'Model': model_type,
                'Optimization': optimization.upper(),
                **metrics,
            })
        df = pd.DataFrame(rows)
        return df


# =============================================================================
# 7. VisualizationEngine
# =============================================================================

class VisualizationEngine:
    """Generates all prediction visualizations."""

    COLORS = {
        'xgboost_base': '#2E86AB',
        'xgboost_csa': '#1B4965',
        'xgboost_bayesian': '#5BA4CF',
        'random_forest_base': '#A23B72',
        'random_forest_csa': '#7B2D5F',
        'random_forest_bayesian': '#C96FA0',
        'arimax_base': '#F18F01',
        'arimax_csa': '#C67200',
        'arimax_bayesian': '#FFB84D',
        'sarimax_base': '#2CA58D',
        'sarimax_csa': '#1E7A68',
        'sarimax_bayesian': '#57C4A9',
    }

    def __init__(self, output_dir: str, interval: str):
        self.output_dir = output_dir
        self.interval = interval

    def _save(self, fig, name: str):
        path = os.path.join(self.output_dir, f"{name}_{self.interval}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {os.path.basename(path)}")

    def plot_actual_vs_predicted(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                 dates: np.ndarray):
        """One plot per model: actual vs predicted."""
        for model_name, y_pred in predictions.items():
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(dates, y_true, label='Actual', color='black', linewidth=1.5)
            color = self.COLORS.get(model_name, '#999999')
            ax.plot(dates, y_pred, label=f'Predicted ({model_name})', color=color,
                    linewidth=1.2, alpha=0.85)
            ax.set_title(f'Actual vs Predicted - {model_name.replace("_", " ").title()} ({self.interval})',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('CPO Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            self._save(fig, f'adaptive_pred_{model_name}')

    def plot_all_predictions_overlay(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                     dates: np.ndarray):
        """All predictions on one plot."""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(dates, y_true, label='Actual', color='black', linewidth=2)
        for model_name, y_pred in predictions.items():
            color = self.COLORS.get(model_name, '#999999')
            ls = '--' if model_name.endswith('_base') else '-'
            ax.plot(dates, y_pred, label=model_name.replace('_', ' ').title(),
                    color=color, linewidth=1.1, linestyle=ls, alpha=0.8)
        ax.set_title(f'All Model Predictions ({self.interval})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('CPO Price')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, 'adaptive_all_predictions')

    def plot_metrics_comparison(self, results_df: pd.DataFrame):
        """Grouped bar chart: base vs CSA for each metric."""
        metrics = ['MAPE', 'RMSE', 'Directional_Accuracy', 'R2']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for ax, metric in zip(axes.flatten(), metrics):
            pivot = results_df.pivot(index='Model', columns='Optimization', values=metric)
            pivot.plot(kind='bar', ax=ax, color=['#5DA5DA', '#FAA43A'], edgecolor='white')
            ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric)
            ax.legend(title='Optimization')
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Metrics Comparison: Base vs CSA-Optimized ({self.interval})',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        self._save(fig, 'adaptive_metrics_comparison')

    def plot_csa_convergence(self, convergence_histories: Dict[str, List[float]]):
        """Convergence plot for optimized models.

        Keys may be '{model_type}' (legacy) or '{model_type}_{optimizer}'.
        """
        if not convergence_histories:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        opt_styles = {'csa': '-', 'bayesian': '--'}
        for key, history in convergence_histories.items():
            color = self.COLORS.get(key, '#999999')
            # Derive linestyle from optimizer suffix if present
            parts = key.rsplit('_', 1)
            ls = opt_styles.get(parts[-1], '-') if len(parts) == 2 else '-'
            ax.plot(history, label=key.replace('_', ' ').title(),
                    linewidth=2, color=color, linestyle=ls)
        ax.set_title(f'Optimizer Convergence History ({self.interval})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best RMSE (CV)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, 'adaptive_convergence')

    def plot_dashboard(self, results_df: pd.DataFrame):
        """Summary dashboard with all metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        metrics = ['MAPE', 'RMSE', 'Directional_Accuracy', 'R2']
        palette = sns.color_palette('Set2', n_colors=len(results_df))

        for ax, metric in zip(axes.flatten(), metrics):
            labels = results_df['Model'] + ' (' + results_df['Optimization'] + ')'
            values = results_df[metric].values
            bars = ax.barh(labels, values, color=palette, edgecolor='white')
            ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + abs(bar.get_width()) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}', va='center', fontsize=8)

        fig.suptitle(f'Model Performance Dashboard ({self.interval})',
                     fontsize=15, fontweight='bold')
        fig.tight_layout()
        self._save(fig, 'adaptive_dashboard')


# =============================================================================
# 8. ResultsManager
# =============================================================================

class ResultsManager:
    """Saves all outputs to disk."""

    def __init__(self, output_dir: str, interval: str):
        self.output_dir = output_dir
        self.interval = interval
        os.makedirs(output_dir, exist_ok=True)

    def save_results_csv(self, results_df: pd.DataFrame):
        path = os.path.join(self.output_dir, f'adaptive_prediction_results_{self.interval}.csv')
        results_df.to_csv(path, index=False)
        print(f"  Saved: {os.path.basename(path)}")

    def save_predictions_csv(self, dates: np.ndarray, y_true: np.ndarray,
                             predictions: Dict[str, np.ndarray]):
        pred_df = pd.DataFrame({'Date': dates, 'Actual': y_true})
        for name, preds in predictions.items():
            pred_df[name] = preds
        path = os.path.join(self.output_dir, f'adaptive_predictions_{self.interval}.csv')
        pred_df.to_csv(path, index=False)
        print(f"  Saved: {os.path.basename(path)}")

    def save_params_json(self, all_params: Dict):
        data = {
            'interval': self.interval,
            'timestamp': pd.Timestamp.now().isoformat(),
            'models': all_params,
        }
        path = os.path.join(self.output_dir, f'adaptive_prediction_params_{self.interval}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {os.path.basename(path)}")


# =============================================================================
# 9. AdaptivePredictionPipeline
# =============================================================================

class AdaptivePredictionPipeline:
    """Main orchestrator for the adaptive prediction workflow."""

    def __init__(self, interval: str, output_dir: str, csa_config: Dict,
                 skip_models: List[str] = None, verbose: bool = True,
                 optimizer: str = 'csa', bayes_n_calls: int = 50,
                 bayes_n_initial: int = 10):
        self.interval = interval.capitalize()
        self.config = IntervalConfig.get(self.interval)
        self.output_dir = output_dir
        self.csa_config = csa_config
        self.skip_models = [m.lower() for m in (skip_models or [])]
        self.verbose = verbose
        self.model_types = ['xgboost', 'random_forest', 'arimax', 'sarimax']
        # Which optimizers to run: 'csa', 'bayesian', or 'both'
        valid_opts = {'csa', 'bayesian', 'both'}
        if optimizer not in valid_opts:
            raise ValueError(f"optimizer must be one of {valid_opts}, got '{optimizer}'")
        self.optimizers = (['csa', 'bayesian'] if optimizer == 'both'
                           else [optimizer])
        self.bayes_n_calls = bayes_n_calls
        self.bayes_n_initial = bayes_n_initial

    def run(self):
        """Execute the full prediction pipeline."""
        pipeline_start = time.time()

        print("\n" + "=" * 70)
        print(f"  ADAPTIVE CPO PRICE PREDICTION - {self.interval.upper()} INTERVAL")
        print("=" * 70)
        print(f"  CSA Config: population={self.csa_config['population_size']}, "
              f"iterations={self.csa_config['max_iterations']}, "
              f"cv_folds={self.csa_config['cv_folds']}")

        # --- Step 1: Load & merge data ---
        loader = DataLoader(self.config)
        merged_df = loader.merge_all()

        # --- Step 2: Feature engineering ---
        feat_eng = FeatureEngineer(self.config)
        df, feature_cols = feat_eng.engineer_features(merged_df)
        data = feat_eng.prepare_train_test(df, feature_cols)

        # Prepare exog for ARIMAX/SARIMAX (top-10 features)
        exog_train_full, exog_indices = select_top_exog(
            data['X_train'], data['y_train'], data['feature_names'],
            n=min(10, data['X_train'].shape[1]))
        exog_test_full = data['X_test'][:, exog_indices]

        # --- Step 3: Train and evaluate all models ---
        all_results = {}
        all_predictions = {}
        all_params = {}
        all_convergence = {}

        for model_type in self.model_types:
            if model_type in self.skip_models:
                print(f"\n  Skipping {model_type} (--skip-models)")
                continue

            print(f"\n{'-'*50}")
            print(f"  Model: {model_type.upper()}")
            print(f"{'-'*50}")

            # ── BASE MODEL ──
            print(f"\n  [BASE] Training {model_type}...")
            t0 = time.time()

            if model_type in ('xgboost', 'random_forest'):
                model = create_sklearn_model(model_type)
                model.fit(data['X_train'], data['y_train'])
                y_pred_base = model.predict(data['X_test'])
                base_params = BASE_PARAMS[model_type].copy()
            else:
                # ARIMAX / SARIMAX
                bp = BASE_PARAMS[model_type]
                order = bp['order']
                if model_type == 'sarimax':
                    seasonal_order = (*bp['seasonal_order_pdq'], self.config.seasonal_period)
                else:
                    seasonal_order = (0, 0, 0, 0)

                fitted = train_statsmodels(model_type, data['y_train'], exog_train_full,
                                           order, seasonal_order, verbose=self.verbose)
                if fitted is not None:
                    y_pred_base = predict_statsmodels(fitted, exog_test_full)
                else:
                    print(f"    WARNING: {model_type} base model failed to converge. Using mean prediction.")
                    y_pred_base = np.full(len(data['y_test']), np.mean(data['y_train']))

                base_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}

            base_time = time.time() - t0
            metrics_base = EvaluationEngine.calculate_all_metrics(data['y_test'], y_pred_base)
            all_results[f'{model_type}_base'] = metrics_base
            all_predictions[f'{model_type}_base'] = y_pred_base
            all_params[f'{model_type}_base'] = base_params

            print(f"    Time: {base_time:.1f}s")
            print(f"    MAPE: {metrics_base['MAPE']:.2f}%  RMSE: {metrics_base['RMSE']:.2f}  "
                  f"Dir.Acc: {metrics_base['Directional_Accuracy']:.1f}%  R²: {metrics_base['R2']:.4f}")

            # ── OPTIMIZED VARIANTS ──
            for opt_name in self.optimizers:
                print(f"\n  [{opt_name.upper()}] Optimizing {model_type}...")
                t0 = time.time()

                if opt_name == 'csa':
                    opt = CSATimeSeriesOptimizer(
                        model_type=model_type,
                        X_train=data['X_train'],
                        y_train=data['y_train'],
                        config=self.config,
                        cv_folds=self.csa_config['cv_folds'],
                        population_size=self.csa_config['population_size'],
                        max_iterations=self.csa_config['max_iterations'],
                    )
                else:  # bayesian
                    opt = BayesianTimeSeriesOptimizer(
                        model_type=model_type,
                        X_train=data['X_train'],
                        y_train=data['y_train'],
                        config=self.config,
                        cv_folds=self.csa_config['cv_folds'],
                        n_calls=self.bayes_n_calls,
                        n_initial_points=self.bayes_n_initial,
                    )

                opt_result = opt.optimize()
                best_params = opt_result.best_params
                all_convergence[f'{model_type}_{opt_name}'] = opt_result.convergence_history

                print(f"    Best score: {opt_result.best_score:.4f} "
                      f"(converged at iter {opt_result.convergence_iteration})")

                if model_type in ('xgboost', 'random_forest'):
                    m_opt = create_sklearn_model(model_type, best_params)
                    m_opt.fit(data['X_train'], data['y_train'])
                    y_pred_opt = m_opt.predict(data['X_test'])
                    opt_params = {k: v for k, v in best_params.items()}
                else:
                    order = (int(best_params.get('p', 1)), int(best_params.get('d', 1)),
                             int(best_params.get('q', 1)))
                    if model_type == 'sarimax':
                        seasonal_order = (int(best_params.get('P', 1)),
                                          int(best_params.get('D', 0)),
                                          int(best_params.get('Q', 1)),
                                          self.config.seasonal_period)
                    else:
                        seasonal_order = (0, 0, 0, 0)

                    fitted = train_statsmodels(model_type, data['y_train'], exog_train_full,
                                               order, seasonal_order, verbose=self.verbose)
                    if fitted is not None:
                        y_pred_opt = predict_statsmodels(fitted, exog_test_full)
                    else:
                        print(f"    WARNING: {model_type} {opt_name} model failed. Using base predictions.")
                        y_pred_opt = y_pred_base.copy()

                    opt_params = {'order': list(order), 'seasonal_order': list(seasonal_order)}

                opt_time = time.time() - t0
                metrics_opt = EvaluationEngine.calculate_all_metrics(data['y_test'], y_pred_opt)
                variant_key = f'{model_type}_{opt_name}'
                all_results[variant_key] = metrics_opt
                all_predictions[variant_key] = y_pred_opt
                all_params[variant_key] = {
                    **opt_params,
                    f'{opt_name}_best_score': float(opt_result.best_score),
                    f'{opt_name}_iterations': opt_result.total_iterations,
                }

                print(f"    Time: {opt_time:.1f}s")
                print(f"    MAPE: {metrics_opt['MAPE']:.2f}%  RMSE: {metrics_opt['RMSE']:.2f}  "
                      f"Dir.Acc: {metrics_opt['Directional_Accuracy']:.1f}%  R²: {metrics_opt['R2']:.4f}")

                if metrics_base['RMSE'] > 0 and metrics_base['RMSE'] != np.inf:
                    improvement = ((metrics_base['RMSE'] - metrics_opt['RMSE'])
                                   / metrics_base['RMSE'] * 100)
                    print(f"    RMSE improvement vs base: {improvement:+.2f}%")

        # --- Step 4: Build comparison table ---
        results_df = EvaluationEngine.build_comparison_table(all_results)

        print(f"\n\n{'='*70}")
        print(f"  RESULTS SUMMARY - {self.interval.upper()}")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))

        # --- Step 5: Save outputs ---
        print(f"\n\nSaving outputs...")
        results_mgr = ResultsManager(self.output_dir, self.interval)
        results_mgr.save_results_csv(results_df)
        results_mgr.save_predictions_csv(data['test_dates'], data['y_test'], all_predictions)
        results_mgr.save_params_json(all_params)

        # --- Step 6: Generate visualizations ---
        print(f"\nGenerating visualizations...")
        viz = VisualizationEngine(self.output_dir, self.interval)
        viz.plot_actual_vs_predicted(data['y_test'], all_predictions, data['test_dates'])
        viz.plot_all_predictions_overlay(data['y_test'], all_predictions, data['test_dates'])
        viz.plot_metrics_comparison(results_df)
        viz.plot_csa_convergence(all_convergence)
        viz.plot_dashboard(results_df)

        total_time = time.time() - pipeline_start
        print(f"\n{'='*70}")
        print(f"  COMPLETE! Total time: {total_time:.1f}s")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

        return results_df


# =============================================================================
# 10. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive CPO Price Prediction with optimizer comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python adaptive_prediction.py --interval daily
  python adaptive_prediction.py --optimizer bayesian --bayes-calls 40
  python adaptive_prediction.py --optimizer both --skip-models sarimax
        """
    )
    parser.add_argument('--interval', type=str, default='daily',
                        choices=['daily'],
                        help='Data interval to use')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: prediction/output/)')
    parser.add_argument('--optimizer', type=str, default='csa',
                        choices=['csa', 'bayesian', 'both'],
                        help='Optimizer to use: csa, bayesian, or both (default: csa)')
    parser.add_argument('--csa-population', type=int, default=25,
                        help='CSA population size (default: 25)')
    parser.add_argument('--csa-iterations', type=int, default=50,
                        help='CSA max iterations (default: 50)')
    parser.add_argument('--csa-cv-folds', type=int, default=3,
                        help='Cross-validation folds (default: 3)')
    parser.add_argument('--bayes-calls', type=int, default=50,
                        help='Bayesian optimization total evaluations (default: 50)')
    parser.add_argument('--bayes-init', type=int, default=10,
                        help='Bayesian initial random points before GP guidance (default: 10)')
    parser.add_argument('--skip-models', type=str, nargs='*', default=[],
                        help='Models to skip (e.g., sarimax arimax)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed output')

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output')

    pipeline = AdaptivePredictionPipeline(
        interval=args.interval,
        output_dir=output_dir,
        csa_config={
            'population_size': args.csa_population,
            'max_iterations': args.csa_iterations,
            'cv_folds': args.csa_cv_folds,
        },
        skip_models=args.skip_models,
        verbose=args.verbose,
        optimizer=args.optimizer,
        bayes_n_calls=args.bayes_calls,
        bayes_n_initial=args.bayes_init,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
