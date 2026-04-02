"""
CPO Price Prediction Dataset Creator
Combines HMM states, news sentiment, and comprehensive CPO features
for price prediction modeling.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
# Data frequency: 'daily', 'weekly', or 'monthly'
DATA_FREQUENCY = 'daily'  # Change to 'weekly' or 'monthly' as needed

# File paths (automatically adjusted based on frequency)
freq_capitalized = DATA_FREQUENCY.capitalize()  # Daily, Weekly, Monthly
if DATA_FREQUENCY.lower() == 'daily':
    CPO_FILE = 'cpo/Data_CPO_Daily.csv'
    SENTIMENT_FILE = 'news/output/sentiment_aggregate_daily.csv'
    HMM_STATES_FILE = 'markov/output/hmm_states_results_daily.csv'
elif DATA_FREQUENCY.lower() == 'weekly':
    CPO_FILE = 'cpo/Data_CPO_Weekly.csv'
    SENTIMENT_FILE = 'news/output/sentiment_aggregate_weekly.csv'
    HMM_STATES_FILE = 'markov/output/hmm_states_results_weekly.csv'
elif DATA_FREQUENCY.lower() == 'monthly':
    CPO_FILE = 'cpo/Data_CPO_Monthly.csv'
    SENTIMENT_FILE = 'news/output/sentiment_aggregate_Monthly.csv'
    HMM_STATES_FILE = 'markov/output/hmm_states_results_monthly.csv'
else:
    CPO_FILE = 'cpo/Data_CPO_Daily.csv'  # default
    SENTIMENT_FILE = 'news/output/sentiment_aggregate_daily.csv'
    HMM_STATES_FILE = 'markov/output/hmm_states_results_daily.csv'

# Date range for dataset
START_YEAR = 2015
END_YEAR = 2025

# Feature engineering windows
SHORT_WINDOW = 5    # Short-term features (5 periods)
MEDIUM_WINDOW = 10  # Medium-term features (10 periods)
LONG_WINDOW = 20    # Long-term features (20 periods)
VERY_LONG_WINDOW = 60  # Very long-term features (60 periods)

# Lag features
MAX_PRICE_LAGS = 5  # Number of lagged price values
MAX_SENTIMENT_LAGS = 10  # Number of lagged sentiment values

# Output file
OUTPUT_FILE = f'markov/cpo_prediction_dataset_{DATA_FREQUENCY}.csv'
# =========================================================

def load_cpo_data(file_path):
    """Load and preprocess CPO price data."""
    print(f"Loading CPO price data...")
    df = pd.read_csv(file_path)
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
    
    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # Clean price data
    df['Close'] = df['Terakhir'].str.replace('.', '').str.replace(',', '.').astype(float)
    df['Open'] = df['Pembukaan'].str.replace('.', '').str.replace(',', '.').astype(float)
    df['High'] = df['Tertinggi'].str.replace('.', '').str.replace(',', '.').astype(float)
    df['Low'] = df['Terendah'].str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Clean volume
    if 'Vol.' in df.columns:
        df['Volume'] = df['Vol.'].replace('-', '0')
        df['Volume'] = df['Volume'].str.replace('K', '').str.replace(',', '.').astype(float) * 1000
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"CPO data loaded: {len(df)} records")
    return df

def load_hmm_states(file_path):
    """Load HMM state predictions."""
    print(f"Loading HMM states...")
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"HMM states loaded: {len(df)} records")
        return df[['Date', 'State']]
    except FileNotFoundError:
        print(f"WARNING: HMM states file not found: {file_path}")
        print("Continuing without HMM states...")
        return None

def load_sentiment_data(file_path):
    """Load sentiment data."""
    print(f"Loading sentiment data...")
    try:
        df = pd.read_csv(file_path)
        
        # Handle different date column names
        if 'Date_Str' in df.columns:
            df['Date'] = pd.to_datetime(df['Date_Str'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            print("WARNING: Could not find Date column in sentiment data")
            return None
        
        # Select relevant columns
        sentiment_cols = ['Date', 'Sentiment_Score']
        if 'Combined_Positive_Prob' in df.columns:
            sentiment_cols.extend(['Combined_Positive_Prob', 'Combined_Negative_Prob', 'Combined_Neutral_Prob'])
        if 'Dominant_Sentiment' in df.columns:
            sentiment_cols.append('Dominant_Sentiment')
        
        df_sentiment = df[sentiment_cols].copy()
        print(f"Sentiment data loaded: {len(df_sentiment)} records")
        return df_sentiment
        
    except FileNotFoundError:
        print(f"WARNING: Sentiment file not found: {file_path}")
        print("Continuing without sentiment data...")
        return None

def create_price_features(df):
    """
    Create comprehensive price-based features.
    """
    print(f"\nCreating price features...")
    
    df = df.copy()
    
    # ========== BASIC PRICE FEATURES ==========
    
    # 1. Returns (various periods)
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_3'] = df['Close'].pct_change(3)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    
    # Log returns (more stable for modeling)
    df['Log_Return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Return_5'] = np.log(df['Close'] / df['Close'].shift(5))
    
    # 2. Price momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
    
    # 3. Rate of Change (ROC)
    df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    df['ROC_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
    
    # ========== MOVING AVERAGES ==========
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=SHORT_WINDOW).mean()
    df['SMA_10'] = df['Close'].rolling(window=MEDIUM_WINDOW).mean()
    df['SMA_20'] = df['Close'].rolling(window=LONG_WINDOW).mean()
    df['SMA_60'] = df['Close'].rolling(window=VERY_LONG_WINDOW).mean()
    
    # Exponential Moving Averages (gives more weight to recent prices)
    df['EMA_5'] = df['Close'].ewm(span=SHORT_WINDOW, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=MEDIUM_WINDOW, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=LONG_WINDOW, adjust=False).mean()
    
    # ========== VOLATILITY FEATURES ==========
    
    # Rolling standard deviation (volatility)
    df['Volatility_5'] = df['Return_1'].rolling(window=SHORT_WINDOW).std()
    df['Volatility_10'] = df['Return_1'].rolling(window=MEDIUM_WINDOW).std()
    df['Volatility_20'] = df['Return_1'].rolling(window=LONG_WINDOW).std()
    
    # Parkinson's volatility (uses high-low range)
    df['Parkinson_Vol_10'] = np.sqrt(1/(4*np.log(2)) * ((np.log(df['High']/df['Low']))**2)).rolling(window=MEDIUM_WINDOW).mean()
    
    # ========== PRICE POSITION & MOMENTUM INDICATORS ==========
    
    # Price relative to moving averages
    df['Price_vs_SMA5'] = (df['Close'] - df['SMA_5']) / df['SMA_5']
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA60'] = (df['Close'] - df['SMA_60']) / df['SMA_60']
    
    # Moving average crossover signals
    df['SMA5_vs_SMA20'] = (df['SMA_5'] - df['SMA_20']) / df['SMA_20']
    df['SMA10_vs_SMA60'] = (df['SMA_10'] - df['SMA_60']) / df['SMA_60']
    
    # ========== RANGE & SPREAD FEATURES ==========
    
    # Daily trading range
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Low']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    
    # Open-Close relationship
    df['Open_Close_Diff'] = (df['Close'] - df['Open']) / df['Open']
    df['Close_to_High'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)
    df['Close_to_Low'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # ========== STATISTICAL FEATURES ==========
    
    # Rolling min and max
    df['Rolling_Min_20'] = df['Close'].rolling(window=LONG_WINDOW).min()
    df['Rolling_Max_20'] = df['Close'].rolling(window=LONG_WINDOW).max()
    
    # Price position in range
    df['Price_Position_20'] = (df['Close'] - df['Rolling_Min_20']) / (df['Rolling_Max_20'] - df['Rolling_Min_20'] + 1e-10)
    
    # Rolling quantiles
    df['Quantile_25_20'] = df['Close'].rolling(window=LONG_WINDOW).quantile(0.25)
    df['Quantile_75_20'] = df['Close'].rolling(window=LONG_WINDOW).quantile(0.75)
    
    # ========== TECHNICAL INDICATORS ==========
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle_20'] = df['Close'].rolling(window=LONG_WINDOW).mean()
    bb_std = df['Close'].rolling(window=LONG_WINDOW).std()
    df['BB_Upper_20'] = df['BB_Middle_20'] + (bb_std * 2)
    df['BB_Lower_20'] = df['BB_Middle_20'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper_20'] - df['BB_Lower_20']) / df['BB_Middle_20']
    df['BB_Position'] = (df['Close'] - df['BB_Lower_20']) / (df['BB_Upper_20'] - df['BB_Lower_20'] + 1e-10)
    
    # ========== VOLUME FEATURES (if available) ==========
    
    if 'Volume' in df.columns:
        # Volume moving averages
        df['Volume_MA_5'] = df['Volume'].rolling(window=SHORT_WINDOW).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=LONG_WINDOW).mean()
        
        # Volume changes
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Price-Volume relationship
        df['Price_Volume_Trend'] = df['Return_1'] * df['Volume']
    
    # ========== TREND FEATURES ==========
    
    # Count of consecutive up/down days
    df['Price_Direction'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
    
    # Acceleration (change in momentum)
    df['Acceleration_5'] = df['Momentum_5'] - df['Momentum_5'].shift(5)
    
    # ========== STATISTICAL AGGREGATIONS ==========
    
    # Rolling skewness and kurtosis
    df['Skew_20'] = df['Return_1'].rolling(window=LONG_WINDOW).skew()
    df['Kurt_20'] = df['Return_1'].rolling(window=LONG_WINDOW).kurt()
    
    # Rolling correlation between price and time
    df['Time_Index'] = range(len(df))
    df['Trend_Strength_20'] = df['Close'].rolling(window=LONG_WINDOW).apply(
        lambda x: np.corrcoef(x, range(len(x)))[0, 1] if len(x) == LONG_WINDOW else np.nan
    )
    
    print(f"Price features created!")
    return df

def create_lagged_features(df, price_lags=5, sentiment_lags=3):
    """
    Create lagged features for prices and sentiment.
    """
    print(f"\nCreating lagged features...")
    
    df = df.copy()
    
    # Lagged price features
    for lag in range(1, price_lags + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Return_1'].shift(lag)
        df[f'Volatility_Lag_{lag}'] = df['Volatility_5'].shift(lag)
    
    # Lagged sentiment features (if available)
    if 'Sentiment_Score' in df.columns:
        for lag in range(1, sentiment_lags + 1):
            df[f'Sentiment_Lag_{lag}'] = df['Sentiment_Score'].shift(lag)
    
    # Lagged HMM state (if available)
    if 'State' in df.columns:
        for lag in range(1, 3):  # 2 lags for HMM state
            df[f'HMM_State_Lag_{lag}'] = df['State'].shift(lag)
    
    print(f"Lagged features created!")
    return df

def create_target_variables(df, horizons=[1, 3, 5, 10]):
    """
    Create target variables for prediction at different time horizons.
    """
    print(f"\nCreating target variables...")
    
    df = df.copy()
    
    for h in horizons:
        # Future price
        df[f'Target_Price_{h}'] = df['Close'].shift(-h)
        
        # Future return
        df[f'Target_Return_{h}'] = df['Close'].pct_change(h).shift(-h)
        
        # Direction (Up/Down)
        df[f'Target_Direction_{h}'] = np.where(df[f'Target_Return_{h}'] > 0, 1, 0)
    
    print(f"Target variables created for horizons: {horizons}")
    return df

def merge_all_data(cpo_df, hmm_df, sentiment_df):
    """
    Merge CPO data with HMM states and sentiment data.
    """
    print(f"\nMerging all datasets...")
    
    # Start with CPO data
    df = cpo_df.copy()
    
    # Merge HMM states
    if hmm_df is not None:
        df = pd.merge(df, hmm_df, on='Date', how='left')
        print(f"  HMM states merged")
    
    # Merge sentiment data
    if sentiment_df is not None:
        df = pd.merge(df, sentiment_df, on='Date', how='left')
        print(f"  Sentiment data merged")
        
        # Forward fill sentiment for missing days (news doesn't come every day)
        sentiment_cols = [col for col in df.columns if 'Sentiment' in col or 'Positive_Prob' in col]
        df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')
    
    print(f"Merged dataset size: {len(df)} records")
    return df

def clean_and_finalize(df):
    """
    Clean the dataset and prepare for modeling.
    """
    print(f"\nCleaning and finalizing dataset...")
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Count missing values before dropping
    missing_before = df.isnull().sum().sum()
    print(f"  Total missing values: {missing_before}")
    
    # Drop rows with NaN in critical columns
    initial_len = len(df)
    df = df.dropna(subset=['Close', 'Target_Return_1'])  # Keep rows with valid target
    dropped = initial_len - len(df)
    print(f"  Dropped {dropped} rows with missing critical values")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove temporary columns
    cols_to_drop = ['Time_Index', 'Price_Direction', 'Tanggal', 'Terakhir', 'Pembukaan', 
                    'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print(f"Final dataset size: {len(df)} records")
    print(f"Total features: {len(df.columns)}")
    
    return df

def main():
    """Main function to create the prediction dataset."""
    print("="*70)
    print("CPO PRICE PREDICTION DATASET CREATOR")
    print(f"Frequency: {DATA_FREQUENCY.upper()}")
    print(f"Period: {START_YEAR} - {END_YEAR}")
    print("="*70)
    
    # Load all data sources
    cpo_df = load_cpo_data(CPO_FILE)
    hmm_df = load_hmm_states(HMM_STATES_FILE)
    sentiment_df = load_sentiment_data(SENTIMENT_FILE)
    
    # Filter date range
    cpo_df = cpo_df[(cpo_df['Year'] >= START_YEAR) & (cpo_df['Year'] <= END_YEAR)]
    print(f"\nFiltered to {START_YEAR}-{END_YEAR}: {len(cpo_df)} records")
    
    # Create price features
    df_features = create_price_features(cpo_df)
    
    # Merge with HMM and sentiment
    df_merged = merge_all_data(df_features, hmm_df, sentiment_df)
    
    # Create lagged features
    df_lagged = create_lagged_features(df_merged, 
                                       price_lags=MAX_PRICE_LAGS,
                                       sentiment_lags=MAX_SENTIMENT_LAGS)
    
    # Create target variables
    df_final = create_target_variables(df_lagged, horizons=[1, 3, 5, 10])
    
    # Clean and finalize
    df_clean = clean_and_finalize(df_final)
    
    # Save to CSV
    print("\n" + "="*70)
    print("SAVING DATASET")
    print("="*70)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to: {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total records: {len(df_clean)}")
    print(f"Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
    print(f"Total features: {len(df_clean.columns)}")
    
    # Feature categories
    feature_categories = {
        'Date/Time': [col for col in df_clean.columns if col in ['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'Week']],
        'Price (OHLC)': [col for col in df_clean.columns if col in ['Open', 'High', 'Low', 'Close']],
        'Returns': [col for col in df_clean.columns if 'Return' in col and 'Target' not in col],
        'Moving Averages': [col for col in df_clean.columns if 'MA' in col or 'EMA' in col],
        'Volatility': [col for col in df_clean.columns if 'Volatility' in col or 'Vol' in col],
        'Technical Indicators': [col for col in df_clean.columns if any(x in col for x in ['RSI', 'MACD', 'BB_', 'ROC'])],
        'HMM States': [col for col in df_clean.columns if 'State' in col or 'HMM' in col],
        'Sentiment': [col for col in df_clean.columns if 'Sentiment' in col or 'Positive_Prob' in col],
        'Lagged Features': [col for col in df_clean.columns if 'Lag' in col],
        'Target Variables': [col for col in df_clean.columns if 'Target' in col]
    }
    
    print("\nFeature breakdown by category:")
    for category, features in feature_categories.items():
        if features:
            print(f"  {category}: {len(features)} features")
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE!")
    print("="*70)
    
    return df_clean

if __name__ == "__main__":
    dataset = main()
