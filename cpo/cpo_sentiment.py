"""
Code to check if sentiment of news matches the movement of CPO prices.
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
DATA_FREQUENCY = 'Daily'  # Change to 'Weekly' or 'Monthly' as needed

# File paths # Ensure consistency with data frequency
CPO_FILE = f'Data_CPO_{DATA_FREQUENCY}.csv'       # Change to Data_CPO_Weekly.csv or Data_CPO_Monthly.csv
SENTIMENT_FILE = f'../news/output/sentiment_aggregate_{DATA_FREQUENCY}.csv'

# Date range for analysis
START_YEAR = 2015
END_YEAR = 2025

# Output files
OUTPUT_CSV = 'output/sentiment_price_matching_results.csv'
OUTPUT_PLOT = 'output/sentiment_price_analysis.png'
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
        # Expect Date or Date_Str column
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
            sentiment_col = 'Sentiment_Score'
        elif 'Combined_Avg_Positive_Prob' in df.columns and 'Combined_Avg_Negative_Prob' in df.columns:
            df['Sentiment_Score'] = df['Combined_Avg_Positive_Prob'] - df['Combined_Avg_Negative_Prob']
            sentiment_col = 'Sentiment_Score'
        else:
            raise ValueError("Could not find sentiment score column in data")
    
    # Rename to standard name if needed
    if sentiment_col != 'Sentiment_Score':
        df['Sentiment_Score'] = df[sentiment_col]
    
    print(f"Sentiment data loaded: {len(df)} records")
    return df

def calculate_price_movement(cpo_df):
    """Calculate price movement direction (up/down/neutral)."""
    # Sort by Date if available, otherwise by MergeKey
    if 'Date' in cpo_df.columns:
        cpo_df = cpo_df.sort_values('Date').copy()
    else:
        cpo_df = cpo_df.sort_values('MergeKey').copy()
    
    # Calculate period-over-period price change
    cpo_df['Price_Change'] = cpo_df['Price'].diff()
    cpo_df['Price_Change_Pct'] = cpo_df['Price'].pct_change() * 100
    
    # Classify movement direction
    # Using a threshold to avoid noise (e.g., < 0.5% change is considered neutral)
    threshold = 0  # 0.5% threshold
    cpo_df['Price_Movement'] = cpo_df['Price_Change_Pct'].apply(
        lambda x: 'Up' if x > threshold else ('Down' if x < -threshold else 'Neutral')
    )
    
    return cpo_df

def calculate_sentiment_movement(sentiment_df):
    """Calculate sentiment movement direction."""
    # Sort by Date/MergeKey
    if 'Date' in sentiment_df.columns:
        sentiment_df = sentiment_df.sort_values('Date').copy()
    else:
        sentiment_df = sentiment_df.sort_values('MergeKey').copy()
    
    # Calculate period-over-period sentiment score change
    sentiment_df['Sentiment_Change'] = sentiment_df['Sentiment_Score'].diff()
    
    # Classify sentiment movement
    threshold = 0.05  # Threshold for sentiment change
    sentiment_df['Sentiment_Movement'] = sentiment_df['Sentiment_Change'].apply(
        lambda x: 'Positive' if x > threshold else ('Negative' if x < -threshold else 'Neutral')
    )
    
    return sentiment_df

def match_sentiment_price_movement(sentiment_df, cpo_df):
    """Merge sentiment and price data and check if movements match."""
    # Select relevant columns for merging (flexible based on what's available)
    sentiment_cols = ['MergeKey', 'Sentiment_Score', 'Sentiment_Movement', 'Sentiment_Change']
    # Add optional columns if they exist
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
    
    # Extract date components from MergeKey or Date
    if 'Date' in merged.columns:
        merged['Date'] = pd.to_datetime(merged['Date'])
        merged['Year'] = merged['Date'].dt.year
        merged['Month'] = merged['Date'].dt.month
        merged['Day'] = merged['Date'].dt.day
    
    # Sort by date
    if 'Date' in merged.columns:
        merged = merged.sort_values('Date')
    else:
        merged = merged.sort_values('MergeKey')
    
    # Check if movements match
    # Positive sentiment should correlate with price increase
    # Negative sentiment should correlate with price decrease
    
    def check_match(row):
        sentiment_move = row['Sentiment_Movement']
        price_move = row['Price_Movement']
        
        if sentiment_move == 'Positive' and price_move == 'Up':
            return 'Match'
        elif sentiment_move == 'Negative' and price_move == 'Down':
            return 'Match'
        elif sentiment_move == 'Neutral' and price_move == 'Neutral':
            return 'Match'
        elif sentiment_move == 'Positive' and price_move == 'Down':
            return 'Mismatch'
        elif sentiment_move == 'Negative' and price_move == 'Up':
            return 'Mismatch'
        else:
            return 'Partial'  # One is neutral, other is not
    
    merged['Match_Status'] = merged.apply(check_match, axis=1)
    
    return merged

def analyze_correlation(merged_df):
    """Analyze correlation between sentiment and price movements."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Basic statistics
    total_months = len(merged_df)
    matches = len(merged_df[merged_df['Match_Status'] == 'Match'])
    mismatches = len(merged_df[merged_df['Match_Status'] == 'Mismatch'])
    partial = len(merged_df[merged_df['Match_Status'] == 'Partial'])
    
    print(f"\nTotal months analyzed: {total_months}")
    print(f"Matches: {matches} ({matches/total_months*100:.2f}%)")
    print(f"Mismatches: {mismatches} ({mismatches/total_months*100:.2f}%)")
    print(f"Partial matches: {partial} ({partial/total_months*100:.2f}%)")
    
    # Correlation coefficient
    correlation = merged_df['Sentiment_Score'].corr(merged_df['Price_Change_Pct'])
    print(f"\nCorrelation coefficient (Sentiment Score vs Price Change %): {correlation:.4f}")
    
    # More detailed analysis by year (if Year column exists)
    if 'Year' in merged_df.columns:
        print("\n" + "-"*60)
        print("Year-by-Year Analysis:")
        print("-"*60)
        yearly_stats = merged_df.groupby('Year').agg({
            'Match_Status': lambda x: (x == 'Match').sum() / len(x) * 100,
            'Sentiment_Score': 'mean',
            'Price_Change_Pct': 'mean'
        }).round(2)
        yearly_stats.columns = ['Match_Rate_%', 'Avg_Sentiment', 'Avg_Price_Change_%']
        print(yearly_stats)
    
    return {
        'total_months': total_months,
        'matches': matches,
        'mismatches': mismatches,
        'partial': partial,
        'correlation': correlation,
        'match_rate': matches/total_months*100
    }

def create_visualizations(merged_df, stats):
    """Create visualizations of the analysis."""
    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time series of sentiment score and price change
    ax1 = plt.subplot(3, 2, 1)
    ax1_twin = ax1.twinx()
    
    # Sort and prepare dates for plotting
    if 'Date' in merged_df.columns:
        merged_df_sorted = merged_df.sort_values('Date')
        dates = pd.to_datetime(merged_df_sorted['Date'])
    elif 'Year' in merged_df.columns and 'Month' in merged_df.columns:
        merged_df_sorted = merged_df.sort_values(['Year', 'Month'])
        dates = pd.to_datetime(merged_df_sorted['Year'].astype(str) + '-' + 
                              merged_df_sorted['Month'].astype(str).str.zfill(2) + '-01')
    else:
        merged_df_sorted = merged_df.sort_values('MergeKey')
        dates = range(len(merged_df_sorted))  # Use index if no date available
    
    line1 = ax1.plot(dates, merged_df_sorted['Sentiment_Score'], 'b-', label='Sentiment Score', linewidth=2)
    line2 = ax1_twin.plot(dates, merged_df_sorted['Price_Change_Pct'], 'r-', label='Price Change %', linewidth=2)
    
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Sentiment Score', color='b', fontsize=10)
    ax1_twin.set_ylabel('Price Change %', color='r', fontsize=10)
    ax1.set_title('Sentiment Score vs Price Change Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 2. Scatter plot: Sentiment Score vs Price Change
    ax2 = plt.subplot(3, 2, 2)
    scatter = ax2.scatter(merged_df['Sentiment_Score'], merged_df['Price_Change_Pct'], 
                         c=merged_df['Match_Status'].map({'Match': 'green', 'Mismatch': 'red', 'Partial': 'orange'}),
                         alpha=0.6, s=50)
    ax2.set_xlabel('Sentiment Score', fontsize=10)
    ax2.set_ylabel('Price Change %', fontsize=10)
    ax2.set_title('Sentiment Score vs Price Change', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add legend for scatter colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Match'),
                      Patch(facecolor='red', label='Mismatch'),
                      Patch(facecolor='orange', label='Partial')]
    ax2.legend(handles=legend_elements, loc='best')
    
    # 3. Match status distribution
    ax3 = plt.subplot(3, 2, 3)
    match_counts = merged_df['Match_Status'].value_counts()
    colors = {'Match': 'green', 'Mismatch': 'red', 'Partial': 'orange'}
    bars = ax3.bar(match_counts.index, match_counts.values, 
                   color=[colors[x] for x in match_counts.index], alpha=0.7)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Match Status Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(merged_df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Yearly match rate (if Year column exists)
    ax4 = plt.subplot(3, 2, 4)
    if 'Year' in merged_df.columns:
        yearly_match = merged_df.groupby('Year').apply(
            lambda x: (x['Match_Status'] == 'Match').sum() / len(x) * 100
        )
        ax4.plot(yearly_match.index, yearly_match.values, marker='o', linewidth=2, markersize=6)
        ax4.axhline(y=stats['match_rate'], color='r', linestyle='--', 
                    label=f'Overall Average: {stats["match_rate"]:.1f}%')
        ax4.set_xlabel('Year', fontsize=10)
        ax4.set_ylabel('Match Rate (%)', fontsize=10)
        ax4.set_title('Yearly Match Rate', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim([0, 100])
    else:
        # Show overall match rate as text if no yearly data
        ax4.text(0.5, 0.5, f'Overall Match Rate\n{stats["match_rate"]:.1f}%', 
                ha='center', va='center', fontsize=20, transform=ax4.transAxes)
        ax4.set_title('Match Rate', fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    # 5. Price movement distribution
    ax5 = plt.subplot(3, 2, 5)
    price_movements = merged_df['Price_Movement'].value_counts()
    ax5.bar(price_movements.index, price_movements.values, color='steelblue', alpha=0.7)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('Price Movement Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Sentiment movement distribution
    ax6 = plt.subplot(3, 2, 6)
    sentiment_movements = merged_df['Sentiment_Movement'].value_counts()
    ax6.bar(sentiment_movements.index, sentiment_movements.values, color='purple', alpha=0.7)
    ax6.set_ylabel('Count', fontsize=10)
    ax6.set_title('Sentiment Movement Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print("Visualization saved to: sentiment_price_analysis.png")
    
    return fig

def main():
    """Main function to run the analysis."""
    print("="*60)
    print("CPO PRICE AND NEWS SENTIMENT MATCHING ANALYSIS")
    print(f"Frequency: {DATA_FREQUENCY.upper()}")
    print(f"Period: {START_YEAR} - {END_YEAR}")
    print("="*60)
    
    # Load data with specified frequency
    cpo_df = load_cpo_data(CPO_FILE, frequency=DATA_FREQUENCY)
    sentiment_df = load_sentiment_data(SENTIMENT_FILE, frequency=DATA_FREQUENCY)
    
    # Filter to specified date range
    cpo_df = cpo_df[(cpo_df['Year'] >= START_YEAR) & (cpo_df['Year'] <= END_YEAR)]
    if 'Year' in sentiment_df.columns:
        sentiment_df = sentiment_df[(sentiment_df['Year'] >= START_YEAR) & (sentiment_df['Year'] <= END_YEAR)]
    
    # Calculate movements
    cpo_df = calculate_price_movement(cpo_df)
    sentiment_df = calculate_sentiment_movement(sentiment_df)
    
    # Match sentiment and price movements
    merged_df = match_sentiment_price_movement(sentiment_df, cpo_df)
    
    # Analyze correlation
    stats = analyze_correlation(merged_df)
    
    # Create visualizations
    fig = create_visualizations(merged_df, stats)
    
    # Save detailed results to CSV
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed results saved to: {OUTPUT_CSV}")
    
    # Summary report
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Overall Match Rate: {stats['match_rate']:.2f}%")
    print(f"Correlation Coefficient: {stats['correlation']:.4f}")
    
    if stats['correlation'] > 0.3:
        print("\n✓ Strong positive correlation found!")
    elif stats['correlation'] > 0.1:
        print("\n→ Moderate positive correlation found.")
    elif stats['correlation'] > -0.1:
        print("\n→ Weak correlation found.")
    else:
        print("\n✗ Negative correlation found.")
    
    print("\nAnalysis complete!")
    
    return merged_df, stats

if __name__ == "__main__":
    merged_data, statistics = main()

