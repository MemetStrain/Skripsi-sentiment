"""
ACF/PACF lag selection for CPO Log_Return.

Computes Autocorrelation Function (ACF) and Partial Autocorrelation Function
(PACF) for Log_Return up to MAX_LAGS, then prints the lags that exceed the
95 % confidence band and saves a combined plot.

Usage:
    python prediction/optimal_lag_acf.py
    python prediction/optimal_lag_acf.py --max-lags 60 --freq Daily
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
CPO_FILE = _ROOT / 'cpo' / 'output' / 'cpo_variables_Daily.csv'

MAX_LAGS = 60   # default; overridden by --max-lags
ALPHA    = 0.05  # significance level → 95 % CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log_return(freq: str = 'Daily') -> pd.Series:
    path = _ROOT / 'cpo' / 'output' / f'cpo_variables_{freq}.csv'
    if not path.exists():
        sys.exit(f"[ERROR] Data file not found: {path}\n"
                 "Run cpo/preprocess_cpo_variables.py first.")

    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if 'Log_Return' not in df.columns:
        sys.exit("[ERROR] 'Log_Return' column not found in the CPO data.")

    series = df['Log_Return'].dropna()
    print(f"Loaded {len(series)} Log_Return observations "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return series


def significant_lags(values: np.ndarray, ci: np.ndarray, skip_zero: bool = True):
    """Return lag indices where |value| exceeds the CI bound."""
    lags = np.arange(len(values))
    sig  = lags[np.abs(values) > ci]
    if skip_zero:
        sig = sig[sig > 0]
    return sig.tolist()


def ci_band(n: int, n_lags: int, alpha: float) -> np.ndarray:
    """Approximate 95 % pointwise CI: ±z / sqrt(n).  Constant width."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return np.full(n_lags + 1, z / np.sqrt(n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(max_lags: int, freq: str, save_plot: bool):
    series = load_log_return(freq)
    n = len(series)

    # --- compute ACF / PACF ---
    acf_vals,  acf_ci  = acf( series, nlags=max_lags, alpha=ALPHA, fft=True)
    pacf_vals, pacf_ci = pacf(series, nlags=max_lags, alpha=ALPHA, method='ywm')

    # statsmodels returns CI as (lower, upper) per lag — convert to half-width
    acf_band  = acf_ci[:, 1]  - acf_vals   # upper - point = half-width
    pacf_band = pacf_ci[:, 1] - pacf_vals

    # --- significant lags ---
    acf_sig  = significant_lags(acf_vals,  acf_band)
    pacf_sig = significant_lags(pacf_vals, pacf_band)

    print(f"\n{'─'*55}")
    print(f"  Max lags checked : {max_lags}")
    print(f"  Significance     : {int((1-ALPHA)*100)} %  (CI ≈ ±{1.96/np.sqrt(n):.4f})")
    print(f"{'─'*55}")
    print(f"  ACF  significant lags  (n={len(acf_sig):>2}) : {acf_sig}")
    print(f"  PACF significant lags  (n={len(pacf_sig):>2}) : {pacf_sig}")

    # Recommend: take all significant PACF lags up to the first gap.
    # Once the PACF drops inside the CI band, later spikes are likely spurious
    # (false positives accumulate at 5 % across many lags).
    if pacf_sig:
        cutoff_lags = []
        for lag in range(1, max_lags + 1):
            if lag in pacf_sig:
                cutoff_lags.append(lag)
            else:
                break   # first non-significant lag — stop here
        recommended = cutoff_lags if cutoff_lags else pacf_sig
        print(f"\n  Recommended Log_Return lags (PACF cutoff): {recommended}")
        print(f"  → Use in LAG_CONFIG: {{'source': 'Log_Return', 'lags': {recommended}}}")
    print(f"{'─'*55}\n")

    # --- plot ---
    lags_x = np.arange(max_lags + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'ACF & PACF of Log_Return — CPO {freq} prices', fontsize=13)

    for ax, vals, band, sig, label in [
        (axes[0], acf_vals,  acf_band,  acf_sig,  'ACF'),
        (axes[1], pacf_vals, pacf_band, pacf_sig, 'PACF'),
    ]:
        ax.bar(lags_x, vals, color='steelblue', width=0.4, label=label)
        ax.fill_between(lags_x, -band, band, alpha=0.15, color='orange',
                        label=f'{int((1-ALPHA)*100)} % CI')
        ax.axhline(0, color='black', linewidth=0.8)

        # mark significant lags
        for lag in sig:
            ax.annotate(str(lag),
                        xy=(lag, vals[lag]),
                        xytext=(0, 6 if vals[lag] >= 0 else -12),
                        textcoords='offset points',
                        fontsize=7, ha='center', color='darkred')

        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-0.5, max_lags + 0.5)

    axes[1].set_xlabel('Lag (days)')
    plt.tight_layout()

    if save_plot:
        out_path = _HERE / f'acf_pacf_log_return_{freq}.png'
        plt.savefig(out_path, dpi=150)
        print(f"  Plot saved → {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACF/PACF lag selector for CPO Log_Return')
    parser.add_argument('--max-lags', type=int, default=MAX_LAGS,
                        help=f'Maximum lag to compute (default: {MAX_LAGS})')
    parser.add_argument('--freq',     type=str, default='Daily',
                        help='Frequency label matching the CPO output file (default: Daily)')
    parser.add_argument('--no-save',  action='store_true',
                        help='Show plot interactively instead of saving to file')
    args = parser.parse_args()

    run(max_lags=args.max_lags, freq=args.freq, save_plot=not args.no_save)
