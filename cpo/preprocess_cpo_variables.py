"""
CPO Data Preprocessing Script
Loads Data_CPO_Daily.csv, engineers features, and saves to cpo/output/
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
DATA_FILES = {
    'Daily': os.path.join(os.path.dirname(__file__), 'Data_CPO_Daily.csv'),
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_id_number(val):
    """Convert Indonesian number format (1.234,56) to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('-', ''):
        return np.nan
    # Remove thousands separator '.', replace decimal ',' with '.'
    s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_volume(val):
    """Parse volume string with optional 'K' suffix (thousands)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('-', ''):
        return np.nan
    multiplier = 1.0
    if s.upper().endswith('K'):
        multiplier = 1_000.0
        s = s[:-1]
    return parse_id_number(s) * multiplier


def parse_change_pct(val):
    """Parse percentage string like '0,92%' or '-1,18%' to float (e.g. 0.92)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace('%', '')
    return parse_id_number(s)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_band_width(series: pd.Series,
                                 window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Upper band minus lower band (BB width)."""
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=1)
    return 2.0 * n_std * std   # (mid + n*std) - (mid - n*std)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_cpo(filepath: str, freq_label: str) -> pd.DataFrame:
    """
    Load one CPO CSV file, clean, and engineer all required features.

    Parameters
    ----------
    filepath   : Path to the raw CSV file.
    freq_label : 'Daily', 'Weekly', or 'Monthly' (for informational output only).

    Returns
    -------
    pd.DataFrame with all original and engineered columns, sorted ascending by Date.
    """
    print(f"\n[{freq_label}] Loading: {filepath}")
    raw = pd.read_csv(filepath)

    # --- Rename columns to English ---
    raw.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Change_Pct']

    # --- Parse data types ---
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y', dayfirst=True)
    for col in ['Close', 'Open', 'High', 'Low']:
        raw[col] = raw[col].apply(parse_id_number)
    raw['Volume']     = raw['Volume'].apply(parse_volume)
    raw['Change_Pct'] = raw['Change_Pct'].apply(parse_change_pct)

    # Sort chronologically (raw file is newest-first)
    df = raw.sort_values('Date').reset_index(drop=True)

    print(f"  Rows: {len(df)} | Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # -----------------------------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------------------------

    # --- Lagged prices ---
    df['Price_t-1'] = df['Close'].shift(1)
    df['Price_t-2'] = df['Close'].shift(2)
    df['Price_t-3'] = df['Close'].shift(3)

    # --- Simple returns (period-over-period) ---
    ret = df['Close'].pct_change()

    # --- Lagged returns ---
    df['Return_t-1'] = ret.shift(1)
    df['Return_t-2'] = ret.shift(2)

    # --- Lagged volume ---
    df['Volume_t-1'] = df['Volume'].shift(1)

    # --- Log return ---
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Price spreads ---
    df['High_Low_Spread']   = df['High'] - df['Low']
    df['Open_Close_Spread'] = df['Open'] - df['Close']

    # --- Simple Moving Averages (window = 3 and 6 periods) ---
    df['SMA_3'] = df['Close'].rolling(window=3,  min_periods=3).mean()
    df['SMA_6'] = df['Close'].rolling(window=6,  min_periods=6).mean()

    # --- Exponential Moving Averages (span = 3 and 6 periods) ---
    df['EMA_3'] = df['Close'].ewm(span=3, adjust=False).mean()
    df['EMA_6'] = df['Close'].ewm(span=6, adjust=False).mean()

    # --- RSI (14 periods) ---
    df['RSI'] = compute_rsi(df['Close'], period=14)

    # --- MACD (12/26/9) ---
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])

    # --- Bollinger Band Width (20 periods, ±2 std) ---
    df['Bollinger_Band_Width'] = compute_bollinger_band_width(df['Close'], window=20)

    # -----------------------------------------------------------------------
    # Column order
    # -----------------------------------------------------------------------
    base_cols  = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Pct']
    feat_cols  = [
        'Price_t-1', 'Price_t-2', 'Price_t-3',
        'Return_t-1', 'Return_t-2',
        'Volume_t-1',
        'Log_Return',
        'High_Low_Spread', 'Open_Close_Spread',
        'SMA_3', 'SMA_6',
        'EMA_3', 'EMA_6',
        'RSI',
        'MACD', 'MACD_Signal',
        'Bollinger_Band_Width',
    ]
    df = df[base_cols + feat_cols]

    print(f"  Columns: {len(df.columns)} | NaN rows (due to lag/window): "
          f"{df[feat_cols].isna().any(axis=1).sum()}")
    return df


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for freq, filepath in DATA_FILES.items():
        if not os.path.exists(filepath):
            print(f"[{freq}] File not found, skipping: {filepath}")
            continue

        df = preprocess_cpo(filepath, freq)

        out_path = os.path.join(OUTPUT_DIR, f'cpo_variables_{freq}.csv')
        df.to_csv(out_path, index=False, float_format='%.6f')
        print(f"  Saved -> {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
