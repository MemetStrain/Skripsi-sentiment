"""
ARIMAX & SARIMAX Model Validity Checker
========================================

Performs comprehensive diagnostics:
  1. Stationarity tests  : ADF, KPSS
  2. ACF / PACF analysis : original + differenced series
  3. Residual diagnostics: Ljung-Box, Jarque-Bera, ARCH-LM, Breusch-Godfrey
  4. Model summary       : AIC, BIC, HQIC, log-likelihood
  5. Visual report       : saved to prediction/output_validation/

Usage:
    python validate_arimax_sarimax.py --interval daily
    python validate_arimax_sarimax.py --interval weekly
    python validate_arimax_sarimax.py --interval monthly
    python validate_arimax_sarimax.py --interval all
    python validate_arimax_sarimax.py --interval daily --horizon 1 --no-plots
"""

import os
import sys
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
    acorr_breusch_godfrey,
)
from statsmodels.stats.stattools import jarque_bera
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'prediction', 'output_validation')
os.makedirs(OUT_DIR, exist_ok=True)

# ── same configs as horizon_forecast.py ──────────────────────────────────────
INTERVAL_CONFIGS = {
    'Daily': {
        'cpo_file':       os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Daily.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Daily.csv'),
        'hmm_file':       os.path.join(PROJECT_ROOT, 'markov', 'output', 'hmm_states_results_Daily.csv'),
        'seasonal_period': 5,
        'base_lag_periods': [1, 2, 3, 5, 10, 20],
        'min_samples': 100,
        'test_ratio': 0.2,
    },
    'Weekly': {
        'cpo_file':       os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Weekly.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Weekly.csv'),
        'hmm_file':       os.path.join(PROJECT_ROOT, 'markov', 'output', 'hmm_states_results_Weekly.csv'),
        'seasonal_period': 4,
        'base_lag_periods': [1, 2, 4, 8, 12],
        'min_samples': 50,
        'test_ratio': 0.2,
    },
    'Monthly': {
        'cpo_file':       os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Monthly.csv'),
        'sentiment_file': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Monthly.csv'),
        'hmm_file':       os.path.join(PROJECT_ROOT, 'markov', 'output', 'hmm_states_results_Monthly.csv'),
        'seasonal_period': 4,
        'base_lag_periods': [1, 2, 3, 6],
        'min_samples': 30,
        'test_ratio': 0.2,
    },
}

BASE_PARAMS = {
    'arimax':  {'order': (2, 1, 2)},
    'sarimax': {'order': (1, 1, 1), 'seasonal_order_pdq': (1, 0, 1)},
}

PASS = '\033[92m✓ PASS\033[0m'
FAIL = '\033[91m✗ FAIL\033[0m'
WARN = '\033[93m⚠ WARN\033[0m'


# =============================================================================
# Data helpers  (mirrors horizon_forecast.py)
# =============================================================================

def load_and_merge(interval: str) -> pd.DataFrame:
    cfg = INTERVAL_CONFIGS[interval]

    cpo = pd.read_csv(cfg['cpo_file'])
    cpo['Date'] = pd.to_datetime(cpo['Date'])

    sent = pd.read_csv(cfg['sentiment_file'])
    if interval == 'Daily':
        sent['Date'] = pd.to_datetime(sent['Date'])
        rename = {
            'Article_Count': 'Article_Count',
            'Combined_Positive_Prob': 'Positive_Prob',
            'Combined_Negative_Prob': 'Negative_Prob',
            'Combined_Neutral_Prob':  'Neutral_Prob',
            'Combined_Confidence':    'Confidence',
        }
    elif interval == 'Monthly':
        sent['Date'] = pd.to_datetime(sent['YearMonth'] + '-01')
        rename = {
            'Total_Articles':              'Article_Count',
            'Combined_Avg_Positive_Prob':  'Positive_Prob',
            'Combined_Avg_Negative_Prob':  'Negative_Prob',
            'Combined_Avg_Neutral_Prob':   'Neutral_Prob',
            'Combined_Avg_Confidence':     'Confidence',
        }
    else:  # Weekly
        sent['Date'] = pd.to_datetime(sent['Week_Start'])
        rename = {
            'Total_Articles':              'Article_Count',
            'Combined_Avg_Positive_Prob':  'Positive_Prob',
            'Combined_Avg_Negative_Prob':  'Negative_Prob',
            'Combined_Avg_Neutral_Prob':   'Neutral_Prob',
            'Combined_Avg_Confidence':     'Confidence',
        }
    sent = sent.rename(columns=rename)
    keep = ['Date', 'Article_Count', 'Positive_Prob', 'Negative_Prob',
            'Neutral_Prob', 'Confidence', 'Sentiment_Score']
    sent = sent[[c for c in keep if c in sent.columns]]

    hmm = pd.read_csv(cfg['hmm_file'])
    hmm['Date'] = pd.to_datetime(hmm['Date'])
    hmm = hmm.rename(columns={
        'Close': 'HMM_Close', 'Log_Return': 'HMM_Log_Return',
        'Volatility': 'HMM_Volatility', 'RSI': 'HMM_RSI',
        'MACD': 'HMM_MACD', 'State': 'HMM_State',
        'State_Label': 'HMM_State_Label',
    })

    if interval == 'Monthly':
        cpo['_k'] = cpo['Date'].dt.to_period('M')
        sent['_k'] = sent['Date'].dt.to_period('M')
        hmm['_k']  = hmm['Date'].dt.to_period('M')
        merged = (cpo
                  .merge(sent.drop(columns=['Date']), on='_k', how='inner', suffixes=('', '_s'))
                  .merge(hmm.drop(columns=['Date']),  on='_k', how='inner', suffixes=('', '_h'))
                  .drop(columns=['_k']))
    elif interval == 'Weekly':
        for df in [cpo, hmm]:
            df['_k'] = (df['Date'].dt.isocalendar().year.astype(str)
                        + '-W' + df['Date'].dt.isocalendar().week.astype(str).str.zfill(2))
        sent['_k'] = (sent['Date'].dt.isocalendar().year.astype(str)
                      + '-W' + sent['Date'].dt.isocalendar().week.astype(str).str.zfill(2))
        merged = (cpo
                  .merge(sent.drop(columns=['Date']), on='_k', how='inner', suffixes=('', '_s'))
                  .merge(hmm.drop(columns=['Date']),  on='_k', how='inner', suffixes=('', '_h'))
                  .drop(columns=['_k']))
    else:
        merged = (cpo
                  .merge(sent, on='Date', how='inner', suffixes=('', '_s'))
                  .merge(hmm,  on='Date', how='inner', suffixes=('', '_h')))

    merged = merged.sort_values('Date').reset_index(drop=True)

    if 'HMM_State_Label' in merged.columns:
        for state in merged['HMM_State_Label'].value_counts().head(5).index:
            col = f'HMM_{state.replace(" ", "_").replace("-", "_")}'
            merged[col] = (merged['HMM_State_Label'] == state).astype(int)
        merged = merged.drop(columns=['HMM_State_Label'])

    return merged


def build_features(df: pd.DataFrame, interval: str, horizon: int
                   ) -> Tuple[pd.DataFrame, List[str]]:
    cfg = INTERVAL_CONFIGS[interval]
    df = df.copy()

    df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    if interval in ('Daily', 'Weekly'):
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['Date'].dt.isocalendar().week.astype(int) / 52)
    if interval == 'Daily':
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 5)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 5)

    safe_lags = [lag for lag in cfg['base_lag_periods'] if lag >= horizon] or [horizon]
    for col in ['Close', 'Sentiment_Score', 'HMM_State']:
        if col not in df.columns:
            continue
        for lag in safe_lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    if 'Sentiment_Score' in df.columns and 'Log_Return' in df.columns:
        df['Sentiment_x_Return'] = df['Sentiment_Score'] * df['Log_Return']
    if 'HMM_Volatility' in df.columns and 'RSI' in df.columns:
        df['Volatility_x_RSI'] = df['HMM_Volatility'] * df['RSI']

    df['Target'] = df['Close'].shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    exclude = {'Date', 'Target', 'Dominant_Sentiment', 'HMM_Close'}
    feat_cols = [c for c in df.columns
                 if c not in exclude and df[c].dtype in ('float64', 'int64', 'int32', 'float32')]
    return df, feat_cols


def select_top_exog(X: np.ndarray, y: np.ndarray, n: int = 10
                    ) -> Tuple[np.ndarray, List[int]]:
    corr = np.array([abs(np.corrcoef(X[:, i], y)[0, 1])
                     if np.std(X[:, i]) > 0 else 0.0
                     for i in range(X.shape[1])])
    idx = np.argsort(corr)[-n:]
    return X[:, idx], idx.tolist()


# =============================================================================
# Stationarity tests
# =============================================================================

def _interpret(p: float, alpha: float = 0.05) -> str:
    return PASS if p < alpha else FAIL


def run_adf(series: np.ndarray, label: str) -> Dict:
    """Augmented Dickey-Fuller: H0 = unit root (non-stationary)."""
    result = adfuller(series, autolag='AIC')
    stat, p, lags, nobs, cvs = result[0], result[1], result[2], result[3], result[4]
    verdict = _interpret(p)   # p < 0.05 → reject H0 → stationary
    print(f"    ADF [{label}]: stat={stat:8.4f}  p={p:.4f}  lags={lags}  → {verdict} (stationary)")
    return {'test': 'ADF', 'label': label, 'stat': round(stat, 6),
            'p_value': round(p, 6), 'lags': lags, 'stationary': p < 0.05,
            'cv_1%': cvs['1%'], 'cv_5%': cvs['5%'], 'cv_10%': cvs['10%']}


def run_kpss(series: np.ndarray, label: str) -> Dict:
    """KPSS: H0 = stationary (reject → non-stationary)."""
    stat, p, lags, cvs = kpss(series, regression='c', nlags='auto')
    verdict = PASS if p >= 0.05 else FAIL  # want p ≥ 0.05 to keep H0
    print(f"    KPSS [{label}]: stat={stat:8.4f}  p={p:.4f}  lags={lags}  → {verdict} (stationary)")
    return {'test': 'KPSS', 'label': label, 'stat': round(stat, 6),
            'p_value': round(p, 6), 'lags': lags, 'stationary': p >= 0.05,
            'cv_1%': cvs['1%'], 'cv_5%': cvs['5%'], 'cv_10%': cvs['10%']}


def stationarity_report(series: np.ndarray, label: str) -> List[Dict]:
    results = []
    print(f"\n  [Stationarity] {label}")
    results.append(run_adf(series, label))
    results.append(run_kpss(series, label))

    # First difference
    d1 = np.diff(series)
    label_d1 = f'{label} (1st diff)'
    print(f"\n  [Stationarity] {label_d1}")
    results.append(run_adf(d1, label_d1))
    results.append(run_kpss(d1, label_d1))
    return results


# =============================================================================
# Residual diagnostics (whitebox)
# =============================================================================

def run_ljung_box(residuals: np.ndarray, lags: int = 20) -> List[Dict]:
    """H0: no autocorrelation up to lag k. Want p > 0.05 (good model)."""
    df_lb = acorr_ljungbox(residuals, lags=lags, return_df=True)
    rows = []
    for lag, (lb_stat, lb_p) in enumerate(zip(df_lb['lb_stat'], df_lb['lb_pvalue']), start=1):
        rows.append({'lag': lag, 'stat': round(lb_stat, 4), 'p_value': round(lb_p, 4),
                     'no_autocorr': lb_p > 0.05})
    # Print summary at lag 5, 10, 20
    for k in [5, 10, min(20, lags)]:
        if k <= len(rows):
            r = rows[k - 1]
            v = PASS if r['no_autocorr'] else FAIL
            print(f"    Ljung-Box lag={k:2d}: stat={r['stat']:8.4f}  p={r['p_value']:.4f}  → {v}")
    return rows


def run_jarque_bera(residuals: np.ndarray) -> Dict:
    """H0: residuals are normally distributed. Want p > 0.05."""
    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    verdict = PASS if jb_p > 0.05 else WARN
    print(f"    Jarque-Bera: stat={jb_stat:10.4f}  p={jb_p:.4f}  "
          f"skew={skew:.4f}  kurt={kurt:.4f}  → {verdict}")
    return {'test': 'Jarque-Bera', 'stat': round(jb_stat, 6), 'p_value': round(jb_p, 6),
            'skewness': round(skew, 6), 'kurtosis': round(kurt, 6), 'normal': jb_p > 0.05}


def run_arch_lm(residuals: np.ndarray, lags: int = 5) -> Dict:
    """ARCH-LM test: H0 = no ARCH effects. Want p > 0.05 (homoscedastic)."""
    lm_stat, lm_p, f_stat, f_p = het_arch(residuals, nlags=lags)
    verdict = PASS if lm_p > 0.05 else WARN
    print(f"    ARCH-LM (lag={lags}): LM={lm_stat:.4f}  p={lm_p:.4f}  → {verdict}")
    return {'test': 'ARCH-LM', 'lags': lags, 'lm_stat': round(lm_stat, 6),
            'lm_p': round(lm_p, 6), 'f_stat': round(f_stat, 6), 'f_p': round(f_p, 6),
            'homoscedastic': lm_p > 0.05}


def run_breusch_godfrey(fitted_model, lags: int = 5) -> Dict:
    """Breusch-Godfrey: H0 = no serial correlation. Want p > 0.05."""
    try:
        bg_stat, bg_p, f_stat, f_p = acorr_breusch_godfrey(fitted_model, nlags=lags)
        verdict = PASS if bg_p > 0.05 else FAIL
        print(f"    Breusch-Godfrey (lag={lags}): LM={bg_stat:.4f}  p={bg_p:.4f}  → {verdict}")
        return {'test': 'Breusch-Godfrey', 'lags': lags, 'lm_stat': round(bg_stat, 6),
                'lm_p': round(bg_p, 6), 'f_stat': round(f_stat, 6), 'f_p': round(f_p, 6),
                'no_serial_corr': bg_p > 0.05}
    except Exception as e:
        print(f"    Breusch-Godfrey: skipped ({e})")
        return {'test': 'Breusch-Godfrey', 'error': str(e)}


def run_shapiro_wilk(residuals: np.ndarray) -> Dict:
    """Shapiro-Wilk normality (recommended for n < 5000). Want p > 0.05."""
    n = len(residuals)
    if n > 5000:
        sample = np.random.default_rng(42).choice(residuals, size=5000, replace=False)
    else:
        sample = residuals
    sw_stat, sw_p = stats.shapiro(sample)
    verdict = PASS if sw_p > 0.05 else WARN
    print(f"    Shapiro-Wilk (n={n}): stat={sw_stat:.6f}  p={sw_p:.4f}  → {verdict}")
    return {'test': 'Shapiro-Wilk', 'n': n, 'stat': round(sw_stat, 6),
            'p_value': round(sw_p, 6), 'normal': sw_p > 0.05}


# =============================================================================
# Plots
# =============================================================================

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved → {os.path.relpath(path)}")


def plot_series_overview(series: np.ndarray, dates: pd.Series,
                         label: str, outfile: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Series Overview – {label}', fontsize=13, fontweight='bold')

    axes[0].plot(dates, series, color='steelblue', lw=0.8)
    axes[0].set_title('Original Series')
    axes[0].set_ylabel('Close Price')

    d1 = np.diff(series)
    axes[1].plot(dates.iloc[1:], d1, color='darkorange', lw=0.8)
    axes[1].set_title('1st Difference')
    axes[1].axhline(0, color='k', lw=0.5, ls='--')

    axes[2].plot(dates.iloc[2:], np.diff(d1), color='mediumseagreen', lw=0.8)
    axes[2].set_title('2nd Difference')
    axes[2].axhline(0, color='k', lw=0.5, ls='--')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, outfile)


def plot_acf_pacf(series: np.ndarray, label: str, outfile: str,
                  lags: int = 40) -> None:
    lags = min(lags, len(series) // 2 - 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'ACF / PACF – {label}', fontsize=13, fontweight='bold')

    plot_acf(series,          ax=axes[0, 0], lags=lags, title='ACF  – Original',        zero=False)
    plot_pacf(series,         ax=axes[0, 1], lags=lags, title='PACF – Original',        zero=False)
    d1 = np.diff(series)
    plot_acf(d1,              ax=axes[1, 0], lags=min(lags, len(d1)//2-1),
             title='ACF  – 1st Diff', zero=False)
    plot_pacf(d1,             ax=axes[1, 1], lags=min(lags, len(d1)//2-1),
             title='PACF – 1st Diff', zero=False)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, outfile)


def plot_residual_diagnostics(residuals: np.ndarray, label: str, outfile: str) -> None:
    lags = min(40, len(residuals) // 2 - 1)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Residual Diagnostics – {label}', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 1. Residuals over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(residuals, color='steelblue', lw=0.8, alpha=0.9)
    ax1.axhline(0, color='k', lw=0.8, ls='--')
    ax1.set_title('Residuals over Time')
    ax1.set_xlabel('Observation index')
    ax1.grid(True, alpha=0.3)

    # 2. ACF of residuals
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(residuals, ax=ax2, lags=lags, title='ACF of Residuals', zero=False)
    ax2.grid(True, alpha=0.3)

    # 3. PACF of residuals
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(residuals, ax=ax3, lags=lags, title='PACF of Residuals', zero=False)
    ax3.grid(True, alpha=0.3)

    # 4. Histogram + KDE
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(residuals, bins=30, density=True, color='steelblue',
             alpha=0.6, edgecolor='white', label='Residuals')
    x = np.linspace(residuals.min(), residuals.max(), 200)
    ax4.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
             'r--', lw=1.5, label='Normal fit')
    ax4.set_title('Residual Distribution')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Q-Q plot
    ax5 = fig.add_subplot(gs[2, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    ax5.scatter(osm, osr, s=5, color='steelblue', alpha=0.6)
    ax5.plot(osm, slope * np.array(osm) + intercept, 'r--', lw=1.5)
    ax5.set_title(f'Q-Q Plot (r={r:.4f})')
    ax5.set_xlabel('Theoretical Quantiles')
    ax5.set_ylabel('Sample Quantiles')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, outfile)


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray,
                              test_dates: pd.Series, label: str, outfile: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Actual vs Predicted – {label}', fontsize=13, fontweight='bold')

    axes[0].plot(test_dates, y_test, label='Actual',    color='steelblue',  lw=1.2)
    axes[0].plot(test_dates, y_pred, label='Predicted', color='darkorange', lw=1.0, ls='--')
    axes[0].set_title('Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test, y_pred, s=8, alpha=0.5, color='steelblue')
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=1, label='Perfect fit')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Scatter')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, outfile)


def plot_ljung_box_pvalues(lb_rows: List[Dict], label: str, outfile: str) -> None:
    lags   = [r['lag'] for r in lb_rows]
    pvals  = [r['p_value'] for r in lb_rows]
    colors = ['green' if p > 0.05 else 'red' for p in pvals]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, pvals, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(0.05, color='k', ls='--', lw=1, label='α = 0.05')
    ax.set_xlabel('Lag')
    ax.set_ylabel('p-value')
    ax.set_title(f'Ljung-Box p-values – {label}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, outfile)


# =============================================================================
# Model fitting + diagnostics
# =============================================================================

def _tag(interval: str, horizon: int, model_type: str) -> str:
    return f'{interval}_h{horizon}_{model_type}'


def validate_model(model_type: str, interval: str, horizon: int,
                   save_plots: bool = True) -> Dict:
    cfg = INTERVAL_CONFIGS[interval]
    tag = _tag(interval, horizon, model_type)
    out = {}

    print(f"\n{'='*60}")
    print(f"  Model: {model_type.upper()}  |  Interval: {interval}  |  Horizon: {horizon}")
    print(f"{'='*60}")

    # ── Load & prepare data ──────────────────────────────────────
    print("\n[1] Loading data …")
    raw = load_and_merge(interval)
    df, feat_cols = build_features(raw, interval, horizon)
    print(f"    Rows: {len(df)}  |  Features: {len(feat_cols)}")

    split = int(len(df) * (1 - cfg['test_ratio']))
    y = df['Target'].values
    y_train, y_test = y[:split], y[split:]
    dates = df['Date']
    dates_train, dates_test = dates.iloc[:split], dates.iloc[split:]

    X = df[feat_cols].values
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test  = scaler.transform(X[split:])

    exog_train, exog_idx = select_top_exog(X_train, y_train, n=min(10, X_train.shape[1]))
    exog_test = X_test[:, exog_idx]

    # ── Stationarity ─────────────────────────────────────────────
    print("\n[2] Stationarity tests (training series) …")
    stat_rows = stationarity_report(y_train, f'{interval} h{horizon} Close (train)')
    out['stationarity'] = stat_rows

    # ── ACF / PACF ───────────────────────────────────────────────
    print("\n[3] ACF / PACF …")
    if save_plots:
        plot_acf_pacf(
            y_train, f'{model_type.upper()} {interval} h{horizon}',
            os.path.join(OUT_DIR, f'{tag}_acf_pacf.png'))
        plot_series_overview(
            y_train, dates_train,
            f'{model_type.upper()} {interval} h{horizon}',
            os.path.join(OUT_DIR, f'{tag}_series_overview.png'))

    # ── Fit model ────────────────────────────────────────────────
    print("\n[4] Fitting model …")
    bp = BASE_PARAMS[model_type]
    order = bp['order']
    if model_type == 'sarimax':
        seasonal_order = (*bp['seasonal_order_pdq'], cfg['seasonal_period'])
    else:
        seasonal_order = (0, 0, 0, 0)

    print(f"    order={order}  seasonal_order={seasonal_order}")
    try:
        sm = SM_SARIMAX(
            endog=y_train, exog=exog_train,
            order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fitted = sm.fit(disp=False, maxiter=200)
    except Exception as e:
        print(f"    ERROR: could not fit model – {e}")
        return {'error': str(e)}

    # ── Model summary ─────────────────────────────────────────────
    print("\n[5] Information criteria …")
    ic = {
        'aic':  round(fitted.aic,  4),
        'bic':  round(fitted.bic,  4),
        'hqic': round(fitted.hqic, 4),
        'llf':  round(fitted.llf,  4),
    }
    for k, v in ic.items():
        print(f"    {k.upper():5s}: {v}")
    out['info_criteria'] = ic

    # ── Residuals ────────────────────────────────────────────────
    residuals = np.asarray(fitted.resid)
    residuals = residuals[~np.isnan(residuals)]
    print(f"\n[6] Residual stats  (n={len(residuals)}) …")
    print(f"    mean={residuals.mean():.4f}  std={residuals.std():.4f}  "
          f"min={residuals.min():.4f}  max={residuals.max():.4f}")

    # ── Whitebox tests ────────────────────────────────────────────
    print("\n[7] Whitebox tests …")

    print("  → Ljung-Box (autocorrelation in residuals):")
    lb_rows = run_ljung_box(residuals, lags=min(20, len(residuals) // 4))

    print("  → Jarque-Bera (normality of residuals):")
    jb = run_jarque_bera(residuals)

    print("  → Shapiro-Wilk (normality of residuals):")
    sw = run_shapiro_wilk(residuals)

    print("  → ARCH-LM (heteroscedasticity):")
    arch = run_arch_lm(residuals, lags=min(5, len(residuals) // 10))

    print("  → Breusch-Godfrey (serial correlation in model):")
    bg = run_breusch_godfrey(fitted, lags=min(5, len(residuals) // 10))

    out['whitebox'] = {
        'ljung_box':      lb_rows,
        'jarque_bera':    jb,
        'shapiro_wilk':   sw,
        'arch_lm':        arch,
        'breusch_godfrey': bg,
    }

    # ── Forecast on test set ──────────────────────────────────────
    print("\n[8] Test-set forecast …")
    try:
        y_pred = np.asarray(fitted.forecast(steps=len(y_test), exog=exog_test))
    except Exception as e:
        print(f"    Forecast failed: {e}")
        y_pred = np.full(len(y_test), np.nan)

    mask = ~np.isnan(y_pred)
    if mask.sum() >= 2:
        from sklearn.metrics import mean_squared_error, r2_score
        yt, yp = y_test[mask], y_pred[mask]
        mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        da   = np.mean((np.diff(yt) > 0) == (np.diff(yp) > 0)) * 100 if len(yt) > 1 else 0
        metrics = {'MAPE': round(mape, 4), 'RMSE': round(rmse, 4),
                   'R2': round(r2, 4), 'Directional_Accuracy': round(da, 4)}
        for k, v in metrics.items():
            print(f"    {k}: {v}")
        out['metrics'] = metrics

    # ── Plots ─────────────────────────────────────────────────────
    if save_plots:
        print("\n[9] Saving diagnostic plots …")
        plot_residual_diagnostics(
            residuals,
            f'{model_type.upper()} {interval} h{horizon}',
            os.path.join(OUT_DIR, f'{tag}_residuals.png'))

        if mask.sum() >= 2:
            plot_actual_vs_predicted(
                y_test[mask], y_pred[mask], dates_test.iloc[np.where(mask)[0]],
                f'{model_type.upper()} {interval} h{horizon}',
                os.path.join(OUT_DIR, f'{tag}_actual_vs_pred.png'))

        plot_ljung_box_pvalues(
            lb_rows,
            f'{model_type.upper()} {interval} h{horizon}',
            os.path.join(OUT_DIR, f'{tag}_ljungbox_pvalues.png'))

    return out


# =============================================================================
# Summary table
# =============================================================================

def build_summary_table(all_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in all_results:
        row = {
            'interval':  r['interval'],
            'horizon':   r['horizon'],
            'model':     r['model'],
            'AIC':       r.get('info_criteria', {}).get('aic', np.nan),
            'BIC':       r.get('info_criteria', {}).get('bic', np.nan),
            'HQIC':      r.get('info_criteria', {}).get('hqic', np.nan),
            'LLF':       r.get('info_criteria', {}).get('llf', np.nan),
            'MAPE':      r.get('metrics', {}).get('MAPE', np.nan),
            'RMSE':      r.get('metrics', {}).get('RMSE', np.nan),
            'R2':        r.get('metrics', {}).get('R2', np.nan),
            'DirAcc':    r.get('metrics', {}).get('Directional_Accuracy', np.nan),
        }
        wb = r.get('whitebox', {})
        # Ljung-Box: pass rate over all lags
        lb = wb.get('ljung_box', [])
        row['LB_pass_rate'] = (round(sum(x['no_autocorr'] for x in lb) / len(lb) * 100, 1)
                               if lb else np.nan)
        row['JB_normal']    = wb.get('jarque_bera', {}).get('normal', np.nan)
        row['SW_normal']    = wb.get('shapiro_wilk', {}).get('normal', np.nan)
        row['ARCH_homo']    = wb.get('arch_lm', {}).get('homoscedastic', np.nan)
        row['BG_no_serial'] = wb.get('breusch_godfrey', {}).get('no_serial_corr', np.nan)

        # Stationarity: ADF + KPSS on original series
        stat = r.get('stationarity', [])
        adf_orig  = next((s for s in stat if s['test'] == 'ADF'  and '1st diff' not in s['label']), {})
        kpss_orig = next((s for s in stat if s['test'] == 'KPSS' and '1st diff' not in s['label']), {})
        adf_d1    = next((s for s in stat if s['test'] == 'ADF'  and '1st diff' in s['label']),     {})
        kpss_d1   = next((s for s in stat if s['test'] == 'KPSS' and '1st diff' in s['label']),     {})
        row['ADF_stat_orig']  = adf_orig.get('stationary',  np.nan)
        row['KPSS_stat_orig'] = kpss_orig.get('stationary', np.nan)
        row['ADF_stat_d1']    = adf_d1.get('stationary',   np.nan)
        row['KPSS_stat_d1']   = kpss_d1.get('stationary',  np.nan)

        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate ARIMAX & SARIMAX models')
    parser.add_argument('--interval', default='daily',
                        choices=['daily', 'weekly', 'monthly', 'all'],
                        help='Interval to validate (default: daily)')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Specific horizon to validate (default: all for interval)')
    parser.add_argument('--model', default='all',
                        choices=['arimax', 'sarimax', 'all'],
                        help='Model type (default: all)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plot files')
    args = parser.parse_args()

    HORIZONS_MAP = {
        'Daily':   [1, 2, 3, 4, 5, 6, 7],
        'Weekly':  [1, 2, 3, 4],
        'Monthly': [1, 2, 3, 4, 5, 6],
    }

    intervals = ['Daily', 'Weekly', 'Monthly'] if args.interval == 'all' \
                else [args.interval.capitalize()]
    models = ['arimax', 'sarimax'] if args.model == 'all' else [args.model]

    all_results = []
    for interval in intervals:
        horizons = ([args.horizon] if args.horizon
                    else HORIZONS_MAP[interval])
        for horizon in horizons:
            for model_type in models:
                tag = _tag(interval, horizon, model_type)
                try:
                    result = validate_model(
                        model_type, interval, horizon,
                        save_plots=not args.no_plots,
                    )
                    result.update({'interval': interval, 'horizon': horizon,
                                   'model': model_type})
                    all_results.append(result)
                except Exception as e:
                    print(f"\n  [ERROR] {tag}: {e}")
                    all_results.append({'interval': interval, 'horizon': horizon,
                                        'model': model_type, 'error': str(e)})

    # ── Save summary ──────────────────────────────────────────────
    if all_results:
        summary_df = build_summary_table(all_results)
        summary_path = os.path.join(OUT_DIR, 'validation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Validation complete – {len(all_results)} model(s) checked")
        print(f"Summary  → {os.path.relpath(summary_path)}")
        print(f"Plots    → {os.path.relpath(OUT_DIR)}/")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
