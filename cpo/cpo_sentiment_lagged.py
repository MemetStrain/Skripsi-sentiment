"""
Lagged Sentiment Analysis for CPO Price Prediction
Analyzes if news sentiment in previous periods predicts current price movements.
Supports Daily, Weekly, or Monthly aggregation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ===================== CONFIGURATION =====================
# Data frequency: 'Daily', 'Weekly', or 'Monthly'
DATA_FREQUENCY = 'Weekly'  # Change to 'Weekly' or 'Daily' as needed

# File paths
CPO_FILE = f'Data_CPO_{DATA_FREQUENCY}.csv'       # Change to Data_CPO_Weekly.csv or Data_CPO_Daily.csv
SENTIMENT_FILE = f'../news/output/sentiment_aggregate_{DATA_FREQUENCY}.csv'

# Date range for analysis
START_YEAR = 2015
END_YEAR = 2025

# Maximum lag periods to analyze
MAX_LAGS = 12  # For Monthly: 12 months, for Weekly: 12 weeks, for Daily: 12 days

# Output files
OUTPUT_CORRELATION_CSV = 'output/sentiment_lagged_correlation_results.csv'
OUTPUT_MATCH_CSV = 'output/sentiment_lagged_match_results.csv'
OUTPUT_DETAILED_CSV = 'output/sentiment_lagged_detailed_data.csv'
OUTPUT_PLOT = 'output/sentiment_lagged_analysis.png'
# =========================================================

def load_cpo_data(file_path, frequency='Daily'):
    """Load and process CPO price data."""
    print(f"Loading CPO price data ({frequency})...")
    df = pd.read_csv(file_path)
    
    # Parse date (format: DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
    
    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # Clean price data - remove commas and convert to float
    df['Price'] = df['Terakhir'].str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Clean percentage change - remove % and convert to float
    df['Change_Pct'] = df['Perubahan%'].str.replace('%', '').str.replace(',', '.').astype(float)
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Create merge keys based on frequency
    if frequency == 'Daily':
        df['MergeKey'] = df['Date'].dt.strftime('%Y-%m-%d')
    elif frequency == 'Weekly':
        df['YearWeek'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)
        df['MergeKey'] = df['YearWeek']
    elif frequency == 'Monthly':
        df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
        df['MergeKey'] = df['YearMonth']
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    # Select relevant columns
    cols = ['Year', 'Month', 'MergeKey', 'Date', 'Price', 'Change_Pct']
    if frequency == 'Daily':
        cols.insert(2, 'Day')
    elif frequency == 'Weekly':
        cols.insert(2, 'Week')
    
    cpo_data = df[cols].copy()
    
    print(f"CPO data loaded: {len(cpo_data)} records from {cpo_data['Year'].min()} to {cpo_data['Year'].max()}")
    return cpo_data

def load_sentiment_data(file_path, frequency='Daily'):
    """Load sentiment data and prepare for merging."""
    print(f"Loading sentiment data ({frequency})...")
    df = pd.read_csv(file_path)
    
    # Create merge keys based on frequency
    if frequency == 'Daily':
        if 'Date_Str' in df.columns:
            df['MergeKey'] = df['Date_Str']
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['MergeKey'] = df['Date'].dt.strftime('%Y-%m-%d')
        else:
            raise ValueError("Sentiment data must have 'Date' or 'Date_Str' column for Daily frequency")
    elif frequency == 'Weekly':
        if 'YearWeek' not in df.columns:
            raise ValueError("Sentiment data must have 'YearWeek' column for Weekly frequency")
        df['MergeKey'] = df['YearWeek']
    elif frequency == 'Monthly':
        if 'YearMonth' not in df.columns:
            if 'Year' in df.columns and 'Month' in df.columns:
                df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
            else:
                raise ValueError("Sentiment data must have 'YearMonth' or 'Year'+'Month' columns for Monthly frequency")
        df['MergeKey'] = df['YearMonth']
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    # Determine sentiment score column name (flexible naming)
    sentiment_col = None
    for col in ['Sentiment_Score', 'sentiment_score', 'Combined_Avg_Positive_Prob']:
        if col in df.columns:
            sentiment_col = col
            break
    
    # If we have probability columns, calculate sentiment score
    if sentiment_col is None:
        if 'Combined_Positive_Prob' in df.columns and 'Combined_Negative_Prob' in df.columns:
            df['Sentiment_Score'] = df['Combined_Positive_Prob'] - df['Combined_Negative_Prob']
        elif 'Combined_Avg_Positive_Prob' in df.columns and 'Combined_Avg_Negative_Prob' in df.columns:
            df['Sentiment_Score'] = df['Combined_Avg_Positive_Prob'] - df['Combined_Avg_Negative_Prob']
        else:
            raise ValueError("Could not find sentiment score column in data")
    elif sentiment_col != 'Sentiment_Score':
        df['Sentiment_Score'] = df[sentiment_col]
    
    # Get positive/negative percentages with flexible column names
    if 'Combined_Positive_Pct' in df.columns:
        df['Positive_Pct'] = df['Combined_Positive_Pct']
        df['Negative_Pct'] = df['Combined_Negative_Pct']
    elif 'Combined_Avg_Positive_Prob' in df.columns:
        df['Positive_Pct'] = df['Combined_Avg_Positive_Prob']
        df['Negative_Pct'] = df['Combined_Avg_Negative_Prob']
    elif 'Combined_Positive_Prob' in df.columns:
        df['Positive_Pct'] = df['Combined_Positive_Prob']
        df['Negative_Pct'] = df['Combined_Negative_Prob']
    
    print(f"Sentiment data loaded: {len(df)} records")
    return df

def calculate_price_movement(cpo_df):
    """Calculate price movement direction and metrics."""
    # Sort by Date if available, otherwise by MergeKey
    if 'Date' in cpo_df.columns:
        cpo_df = cpo_df.sort_values('Date').copy()
    else:
        cpo_df = cpo_df.sort_values('MergeKey').copy()
    
    # Calculate period-over-period price change
    cpo_df['Price_Change'] = cpo_df['Price'].diff()
    cpo_df['Price_Change_Pct'] = cpo_df['Price'].pct_change() * 100
    
    # Classify movement direction
    threshold = 0
    cpo_df['Price_Movement'] = cpo_df['Price_Change_Pct'].apply(
        lambda x: 'Up' if x > threshold else ('Down' if x < -threshold else 'Neutral')
    )
    
    return cpo_df

def merge_sentiment_price(sentiment_df, cpo_df):
    """Merge sentiment and price data."""
    # Prepare dataframes for merge - flexible column selection
    sentiment_cols = ['MergeKey', 'Sentiment_Score']
    if 'Positive_Pct' in sentiment_df.columns:
        sentiment_cols.extend(['Positive_Pct', 'Negative_Pct'])
    if 'Dominant_Sentiment' in sentiment_df.columns:
        sentiment_cols.append('Dominant_Sentiment')
    
    sentiment_merge = sentiment_df[sentiment_cols].copy()
    cpo_merge = cpo_df[['MergeKey', 'Price', 'Price_Change_Pct', 'Price_Movement', 'Date']].copy()
    
    # Merge on MergeKey
    merged = pd.merge(
        sentiment_merge,
        cpo_merge,
        on='MergeKey',
        how='inner'
    )
    
    # Extract date components from Date
    if 'Date' in merged.columns:
        merged['Date'] = pd.to_datetime(merged['Date'])
        merged['Year'] = merged['Date'].dt.year
        merged['Month'] = merged['Date'].dt.month
        merged['Day'] = merged['Date'].dt.day
    
    # Sort by date
    if 'Date' in merged.columns:
        merged = merged.sort_values('Date').reset_index(drop=True)
    else:
        merged = merged.sort_values('MergeKey').reset_index(drop=True)
    
    return merged

def create_lagged_features(df, max_lags=12, frequency='Monthly'):
    """
    Create lagged sentiment features.
    
    Parameters:
        df: Merged sentiment-price dataframe
        max_lags: Maximum number of lags to create
        frequency: 'Daily', 'Weekly', or 'Monthly'
    
    Returns:
        DataFrame with lagged features
    """
    df_lagged = df.copy()
    
    period_name = {'Daily': 'days', 'Weekly': 'weeks', 'Monthly': 'months'}[frequency]
    print(f"\nCreating lagged sentiment features (up to {max_lags} {period_name})...")
    
    for lag in range(1, max_lags + 1):
        # Lag sentiment score
        df_lagged[f'Sentiment_Score_Lag{lag}'] = df_lagged['Sentiment_Score'].shift(lag)
        
        # Lag positive/negative percentages if available
        if 'Positive_Pct' in df_lagged.columns:
            df_lagged[f'Positive_Pct_Lag{lag}'] = df_lagged['Positive_Pct'].shift(lag)
            df_lagged[f'Negative_Pct_Lag{lag}'] = df_lagged['Negative_Pct'].shift(lag)
    
    # Drop rows with NaN values introduced by lagging
    df_lagged = df_lagged.dropna()
    
    print(f"Created {max_lags} lagged features")
    print(f"Analysis dataset size: {len(df_lagged)} {period_name}")
    
    return df_lagged

def calculate_lagged_correlations(df_lagged, max_lags=12):
    """
    Calculate correlation between lagged sentiment and current price movement.
    
    Returns:
        DataFrame with correlation statistics for each lag
    """
    print("\nCalculating lagged correlations...")
    
    correlations = []
    
    for lag in range(1, max_lags + 1):
        sentiment_col = f'Sentiment_Score_Lag{lag}'
        pos_col = f'Positive_Pct_Lag{lag}'
        neg_col = f'Negative_Pct_Lag{lag}'
        
        # Correlations with price change %
        corr_sentiment = df_lagged['Price_Change_Pct'].corr(df_lagged[sentiment_col])
        corr_positive = df_lagged['Price_Change_Pct'].corr(df_lagged[pos_col])
        corr_negative = df_lagged['Price_Change_Pct'].corr(df_lagged[neg_col])
        
        # Calculate Spearman correlation (non-parametric)
        from scipy.stats import spearmanr, pearsonr
        spearman_corr, spearman_p = spearmanr(df_lagged['Price_Change_Pct'], df_lagged[sentiment_col])
        pearson_corr, pearson_p = pearsonr(df_lagged['Price_Change_Pct'], df_lagged[sentiment_col])
        
        correlations.append({
            'Lag_Months': lag,
            'Pearson_Correlation': pearson_corr,
            'Pearson_P_Value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_Value': spearman_p,
            'Positive_Pct_Corr': corr_positive,
            'Negative_Pct_Corr': corr_negative,
            'Significant': 'Yes' if pearson_p < 0.05 else 'No'
        })
    
    corr_df = pd.DataFrame(correlations)
    return corr_df

def analyze_lagged_matches(df_lagged, max_lags=12):
    """
    Analyze match rates for different lag periods.
    
    Match logic:
    - Positive lagged sentiment should lead to price Up
    - Negative lagged sentiment should lead to price Down
    """
    print("\nAnalyzing lagged match rates...")
    
    match_results = []
    
    for lag in range(1, max_lags + 1):
        pos_col = f'Positive_Pct_Lag{lag}'
        neg_col = f'Negative_Pct_Lag{lag}'
        
        # Calculate sentiment strength from positive/negative percentages
        df_lagged[f'Sentiment_Strength_Lag{lag}'] = (
            df_lagged[pos_col] - df_lagged[neg_col]
        )
        
        # Predict direction based on lagged sentiment
        df_lagged[f'Predicted_Direction_Lag{lag}'] = df_lagged[f'Sentiment_Strength_Lag{lag}'].apply(
            lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Neutral')
        )
        
        # Compare with actual price movement
        matches = (df_lagged[f'Predicted_Direction_Lag{lag}'] == df_lagged['Price_Movement']).sum()
        total = len(df_lagged)
        match_rate = (matches / total * 100) if total > 0 else 0
        
        # Separate analysis for Up predictions
        up_predictions = df_lagged[df_lagged[f'Predicted_Direction_Lag{lag}'] == 'Up']
        up_correct = (up_predictions['Price_Movement'] == 'Up').sum() if len(up_predictions) > 0 else 0
        up_accuracy = (up_correct / len(up_predictions) * 100) if len(up_predictions) > 0 else 0
        
        # Separate analysis for Down predictions
        down_predictions = df_lagged[df_lagged[f'Predicted_Direction_Lag{lag}'] == 'Down']
        down_correct = (down_predictions['Price_Movement'] == 'Down').sum() if len(down_predictions) > 0 else 0
        down_accuracy = (down_correct / len(down_predictions) * 100) if len(down_predictions) > 0 else 0
        
        match_results.append({
            'Lag_Months': lag,
            'Total_Predictions': total,
            'Correct_Predictions': matches,
            'Match_Rate_%': match_rate,
            'Up_Predictions': len(up_predictions),
            'Up_Accuracy_%': up_accuracy,
            'Down_Predictions': len(down_predictions),
            'Down_Accuracy_%': down_accuracy
        })
    
    match_df = pd.DataFrame(match_results)
    return match_df

def create_lagged_visualizations(corr_df, match_df, df_lagged, stats):
    """Create visualizations for lagged analysis."""
    print("\nGenerating lagged analysis visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Correlation by lag
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(corr_df['Lag_Months'], corr_df['Pearson_Correlation'], 
             marker='o', linewidth=2, markersize=8, label='Pearson')
    ax1.plot(corr_df['Lag_Months'], corr_df['Spearman_Correlation'], 
             marker='s', linewidth=2, markersize=8, label='Spearman')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.05, color='r', linestyle=':', alpha=0.3, label='Significance threshold')
    ax1.set_xlabel('Lag (Months)', fontsize=10)
    ax1.set_ylabel('Correlation Coefficient', fontsize=10)
    ax1.set_title('Sentiment Correlation with Price Change by Lag', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Match rate by lag
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(match_df['Lag_Months'], match_df['Match_Rate_%'], 
             marker='o', linewidth=2, markersize=8, color='green', label='Overall Match Rate')
    ax2.axhline(y=33.33, color='r', linestyle='--', alpha=0.5, label='Random Chance (33.3%)')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Baseline')
    ax2.set_xlabel('Lag (Months)', fontsize=10)
    ax2.set_ylabel('Match Rate (%)', fontsize=10)
    ax2.set_title('Prediction Accuracy by Lag Period', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    # 3. Up/Down prediction accuracy by lag
    ax3 = plt.subplot(3, 2, 3)
    width = 0.35
    x = match_df['Lag_Months']
    ax3.bar(x - width/2, match_df['Up_Accuracy_%'], width, label='Up Prediction Accuracy', alpha=0.8)
    ax3.bar(x + width/2, match_df['Down_Accuracy_%'], width, label='Down Prediction Accuracy', alpha=0.8)
    ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Lag (Months)', fontsize=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=10)
    ax3.set_title('Up vs Down Prediction Accuracy by Lag', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    ax3.set_ylim([0, 100])
    
    # 4. P-values by lag
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(corr_df['Lag_Months'], corr_df['Pearson_P_Value'], 
             marker='o', linewidth=2, markersize=8, color='red')
    ax4.axhline(y=0.05, color='g', linestyle='--', linewidth=2, label='Significance level (p=0.05)')
    ax4.set_xlabel('Lag (Months)', fontsize=10)
    ax4.set_ylabel('P-Value', fontsize=10)
    ax4.set_title('Statistical Significance (P-Value) by Lag', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    
    # 5. Heatmap of correlations
    ax5 = plt.subplot(3, 2, 5)
    corr_heatmap_data = corr_df[['Lag_Months', 'Pearson_Correlation', 
                                   'Positive_Pct_Corr', 'Negative_Pct_Corr']].set_index('Lag_Months').T
    sns.heatmap(corr_heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Correlation'}, ax=ax5, vmin=-0.5, vmax=0.5)
    ax5.set_title('Correlation Heatmap by Lag', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Lag (Months)', fontsize=10)
    
    # 6. Number of predictions by lag
    ax6 = plt.subplot(3, 2, 6)
    width = 0.35
    x = match_df['Lag_Months']
    ax6.bar(x - width/2, match_df['Up_Predictions'], width, label='Up Predictions', alpha=0.8)
    ax6.bar(x + width/2, match_df['Down_Predictions'], width, label='Down Predictions', alpha=0.8)
    ax6.set_xlabel('Lag (Months)', fontsize=10)
    ax6.set_ylabel('Number of Predictions', fontsize=10)
    ax6.set_title('Distribution of Predictions by Lag', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print("Visualization saved to: sentiment_lagged_analysis.png")
    
    return fig

def print_summary_report(corr_df, match_df):
    """Print detailed summary report."""
    print("\n" + "="*70)
    print("LAGGED SENTIMENT ANALYSIS SUMMARY REPORT")
    print("="*70)
    
    # Find best lag for correlation
    best_corr_idx = corr_df['Pearson_Correlation'].abs().idxmax()
    best_corr_lag = corr_df.loc[best_corr_idx]
    
    print(f"\n{'CORRELATION ANALYSIS':^70}")
    print("-"*70)
    print(f"Best lag for correlation: {int(best_corr_lag['Lag_Months'])} months")
    print(f"  Pearson Correlation: {best_corr_lag['Pearson_Correlation']:.4f}")
    print(f"  P-Value: {best_corr_lag['Pearson_P_Value']:.4f}")
    print(f"  Significant: {best_corr_lag['Significant']}")
    
    # Significant lags
    sig_lags = corr_df[corr_df['Pearson_P_Value'] < 0.05]
    if len(sig_lags) > 0:
        print(f"\nStatistically significant lags (p < 0.05):")
        for idx, row in sig_lags.iterrows():
            print(f"  - {int(row['Lag_Months'])} months: correlation = {row['Pearson_Correlation']:.4f}")
    else:
        print(f"\nNo statistically significant lags found (all p > 0.05)")
    
    # Find best lag for match rate
    best_match_idx = match_df['Match_Rate_%'].idxmax()
    best_match_lag = match_df.loc[best_match_idx]
    
    print(f"\n{'PREDICTION ACCURACY ANALYSIS':^70}")
    print("-"*70)
    print(f"Best lag for prediction: {int(best_match_lag['Lag_Months'])} months")
    print(f"  Match Rate: {best_match_lag['Match_Rate_%']:.2f}%")
    print(f"  Correct Predictions: {int(best_match_lag['Correct_Predictions'])} / {int(best_match_lag['Total_Predictions'])}")
    print(f"  Up Prediction Accuracy: {best_match_lag['Up_Accuracy_%']:.2f}%")
    print(f"  Down Prediction Accuracy: {best_match_lag['Down_Accuracy_%']:.2f}%")
    
    # Top 3 lags by match rate
    print(f"\nTop 3 lags by prediction accuracy:")
    top_3 = match_df.nlargest(3, 'Match_Rate_%')
    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {idx}. {int(row['Lag_Months'])} months: {row['Match_Rate_%']:.2f}% " +
              f"({int(row['Correct_Predictions'])}/{int(row['Total_Predictions'])} correct)")
    
    print("\n" + "="*70)

def main():
    """Main function to run lagged sentiment analysis."""
    print("="*70)
    print("LAGGED SENTIMENT ANALYSIS FOR CPO PRICE PREDICTION")
    print(f"Frequency: {DATA_FREQUENCY.upper()}")
    print(f"Period: {START_YEAR} - {END_YEAR}")
    print(f"Max Lags: {MAX_LAGS}")
    print("="*70)
    
    # Load data with specified frequency
    cpo_df = load_cpo_data(CPO_FILE, frequency=DATA_FREQUENCY)
    sentiment_df = load_sentiment_data(SENTIMENT_FILE, frequency=DATA_FREQUENCY)
    
    # Filter to specified date range
    cpo_df = cpo_df[(cpo_df['Year'] >= START_YEAR) & (cpo_df['Year'] <= END_YEAR)]
    if 'Year' in sentiment_df.columns:
        sentiment_df = sentiment_df[(sentiment_df['Year'] >= START_YEAR) & (sentiment_df['Year'] <= END_YEAR)]
    
    # Calculate price movements
    cpo_df = calculate_price_movement(cpo_df)
    
    # Merge data
    merged_df = merge_sentiment_price(sentiment_df, cpo_df)
    
    # Create lagged features
    df_lagged = create_lagged_features(merged_df, max_lags=MAX_LAGS, frequency=DATA_FREQUENCY)
    
    # Calculate correlations
    corr_df = calculate_lagged_correlations(df_lagged, max_lags=MAX_LAGS)
    
    # Calculate match rates
    match_df = analyze_lagged_matches(df_lagged, max_lags=MAX_LAGS)
    
    # Create visualizations
    fig = create_lagged_visualizations(corr_df, match_df, df_lagged, {})
    
    # Print summary report
    print_summary_report(corr_df, match_df)
    
    # Save results to CSV
    corr_df.to_csv(OUTPUT_CORRELATION_CSV, index=False)
    print(f"\nCorrelation results saved to: {OUTPUT_CORRELATION_CSV}")
    
    match_df.to_csv(OUTPUT_MATCH_CSV, index=False)
    print(f"Match results saved to: {OUTPUT_MATCH_CSV}")
    
    # Save detailed dataset
    base_cols = ['MergeKey', 'Price', 'Price_Change_Pct', 'Price_Movement', 'Sentiment_Score']
    if 'Year' in df_lagged.columns:
        base_cols.insert(0, 'Year')
    if 'Month' in df_lagged.columns:
        base_cols.insert(1, 'Month')
    if 'Date' in df_lagged.columns:
        base_cols.insert(0, 'Date')
    
    save_cols = base_cols + [col for col in df_lagged.columns if 'Lag' in col]
    df_lagged[[col for col in save_cols if col in df_lagged.columns]].to_csv(OUTPUT_DETAILED_CSV, index=False)
    print(f"Detailed data saved to: {OUTPUT_DETAILED_CSV}")
    
    print("\nAnalysis complete!")
    
    return corr_df, match_df, df_lagged

if __name__ == "__main__":
    corr_results, match_results, lagged_data = main()
