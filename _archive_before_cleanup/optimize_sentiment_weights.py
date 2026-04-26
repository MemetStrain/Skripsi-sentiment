"""
Script to optimize sentiment probability weights to achieve >80% match rate
with CPO price movements using grid search.
"""

import pandas as pd
import numpy as np
from itertools import product
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import pickle
warnings.filterwarnings('ignore')

def load_cpo_data(file_path):
    """Load and process CPO price data."""
    print("Loading CPO price data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CPO data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading CPO data from {file_path}: {e}")
    
    if df.empty:
        raise ValueError(f"CPO data file is empty: {file_path}")
    
    # Parse date (format: DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y', errors='coerce')
    
    # Check for invalid dates
    if df['Date'].isna().any():
        print(f"Warning: {df['Date'].isna().sum()} rows with invalid dates will be dropped")
        df = df.dropna(subset=['Date'])
    
    # Extract year and month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Clean price data - remove commas and convert to float
    df['Price'] = df['Terakhir'].str.replace('.', '').str.replace(',', '.').astype(float, errors='ignore')
    df = df.dropna(subset=['Price'])  # Remove rows with invalid prices
    
    # Clean percentage change - remove % and convert to float
    df['Change_Pct'] = df['Perubahan%'].str.replace('%', '').str.replace(',', '.').astype(float, errors='ignore')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Create YearMonth column for merging
    df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    
    # Select relevant columns
    cpo_monthly = df[['Year', 'Month', 'YearMonth', 'Date', 'Price', 'Change_Pct']].copy()
    
    if cpo_monthly.empty:
        raise ValueError(f"No valid CPO data after processing: {file_path}")
    
    return cpo_monthly

def load_sentiment_data(file_path):
    """Load monthly sentiment data."""
    print("Loading sentiment data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Sentiment data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading sentiment data from {file_path}: {e}")
    
    if df.empty:
        raise ValueError(f"Sentiment data file is empty: {file_path}")
    
    # Validate required columns exist
    required_cols = ['Year', 'Month', 'Title_Avg_Positive_Prob', 'Title_Avg_Negative_Prob', 
                     'Title_Avg_Neutral_Prob', 'Content_Avg_Positive_Prob', 'Content_Avg_Negative_Prob',
                     'Content_Avg_Neutral_Prob', 'Combined_Avg_Positive_Prob', 'Combined_Avg_Negative_Prob',
                     'Combined_Avg_Neutral_Prob']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in sentiment data: {missing_cols}")
    
    # Create YearMonth column for merging
    df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    
    return df

def calculate_price_movement(cpo_df, threshold=0):
    """Calculate price movement direction (up/down/neutral)."""
    cpo_df = cpo_df.sort_values(['Year', 'Month']).copy()
    
    # Calculate month-over-month price change
    cpo_df['Price_Change_Pct'] = cpo_df['Price'].pct_change() * 100
    # Handle NaN from first row (no previous value)
    cpo_df['Price_Change_Pct'] = cpo_df['Price_Change_Pct'].fillna(0)
    
    # Classify movement direction
    cpo_df['Price_Movement'] = cpo_df['Price_Change_Pct'].apply(
        lambda x: 'Up' if pd.notna(x) and x > threshold else ('Down' if pd.notna(x) and x < -threshold else 'Neutral')
    )
    
    return cpo_df

def calculate_weighted_sentiment_score(row, prob_type, w_pos, w_neg, w_neu):
    """
    Calculate normalized weighted sentiment score.
    
    Normalized score: (w_pos*Positive - w_neg*Negative) / (|w_pos| + |w_neg| + |w_neu|)
    This ensures the score is normalized between -1 and 1.
    """
    try:
        if prob_type == 'Title':
            pos_prob = row['Title_Avg_Positive_Prob']
            neg_prob = row['Title_Avg_Negative_Prob']
            neu_prob = row['Title_Avg_Neutral_Prob']
        elif prob_type == 'Content':
            pos_prob = row['Content_Avg_Positive_Prob']
            neg_prob = row['Content_Avg_Negative_Prob']
            neu_prob = row['Content_Avg_Neutral_Prob']
        elif prob_type == 'Combined':
            pos_prob = row['Combined_Avg_Positive_Prob']
            neg_prob = row['Combined_Avg_Negative_Prob']
            neu_prob = row['Combined_Avg_Neutral_Prob']
        else:
            raise ValueError(f"Unknown probability type: {prob_type}")
        
        # Handle NaN values in probabilities
        if pd.isna(pos_prob) or pd.isna(neg_prob) or pd.isna(neu_prob):
            return 0.0
        
        # Normalized weighted score
        numerator = w_pos * pos_prob - w_neg * neg_prob
        denominator = abs(w_pos) + abs(w_neg) + abs(w_neu)
        
        if denominator == 0:
            return 0.0
        
        score = numerator / denominator
        
        # Handle any NaN or inf results
        if pd.isna(score) or np.isinf(score):
            return 0.0
        
        return score
    except (KeyError, TypeError) as e:
        # Return 0.0 if there's an error accessing the data
        return 0.0

def calculate_sentiment_movement(sentiment_df, prob_type, w_pos, w_neg, w_neu, threshold=0.05):
    """Calculate sentiment movement direction using weighted probabilities."""
    sentiment_df = sentiment_df.sort_values(['Year', 'Month']).copy()
    
    # Calculate weighted sentiment score
    sentiment_df['Weighted_Sentiment_Score'] = sentiment_df.apply(
        lambda row: calculate_weighted_sentiment_score(row, prob_type, w_pos, w_neg, w_neu),
        axis=1
    )
    
    # Fill any NaN scores with 0
    sentiment_df['Weighted_Sentiment_Score'] = sentiment_df['Weighted_Sentiment_Score'].fillna(0)
    
    # Calculate month-over-month sentiment score change
    sentiment_df['Sentiment_Change'] = sentiment_df['Weighted_Sentiment_Score'].diff()
    # Handle NaN from first row (no previous value)
    sentiment_df['Sentiment_Change'] = sentiment_df['Sentiment_Change'].fillna(0)
    
    # Classify sentiment movement
    sentiment_df['Sentiment_Movement'] = sentiment_df['Sentiment_Change'].apply(
        lambda x: 'Positive' if pd.notna(x) and x > threshold else ('Negative' if pd.notna(x) and x < -threshold else 'Neutral')
    )
    
    return sentiment_df

def match_sentiment_price_movement(sentiment_df, cpo_df):
    """Merge sentiment and price data and check if movements match."""
    # Prepare dataframes for merge
    sentiment_merge = sentiment_df[['YearMonth', 'Weighted_Sentiment_Score', 'Sentiment_Movement', 
                                     'Sentiment_Change']].copy()
    cpo_merge = cpo_df[['YearMonth', 'Price', 'Price_Change_Pct', 'Price_Movement']].copy()
    
    # Remove any NaN YearMonth values before merge
    sentiment_merge = sentiment_merge.dropna(subset=['YearMonth'])
    cpo_merge = cpo_merge.dropna(subset=['YearMonth'])
    
    # Merge on YearMonth
    merged = pd.merge(
        sentiment_merge,
        cpo_merge,
        on='YearMonth',
        how='inner'
    )
    
    # Return empty dataframe if no matches
    if merged.empty:
        return pd.DataFrame()
    
    # Add Year and Month back from YearMonth
    # Handle potential NaN values in YearMonth
    merged = merged.dropna(subset=['YearMonth'])
    if merged.empty:
        return pd.DataFrame()
    
    try:
        year_month_split = merged['YearMonth'].str.split('-', expand=True)
        merged['Year'] = year_month_split[0].astype(int, errors='ignore')
        merged['Month'] = year_month_split[1].astype(int, errors='ignore')
        
        # Remove rows where Year or Month couldn't be converted
        merged = merged.dropna(subset=['Year', 'Month'])
    except (ValueError, KeyError, AttributeError) as e:
        # If splitting fails, return empty dataframe
        return pd.DataFrame()
    
    if merged.empty:
        return pd.DataFrame()
    
    # Sort by date
    merged = merged.sort_values(['Year', 'Month'])
    
    # Check if movements match
    def check_match(row):
        try:
            sentiment_move = row['Sentiment_Movement']
            price_move = row['Price_Movement']
            
            if pd.isna(sentiment_move) or pd.isna(price_move):
                return 'Partial'
            
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
        except (KeyError, TypeError):
            return 'Partial'
    
    merged['Match_Status'] = merged.apply(check_match, axis=1)
    
    return merged

def evaluate_weights(sentiment_df, cpo_df, prob_type, w_pos, w_neg, w_neu):
    """Evaluate a specific weight combination."""
    # Calculate sentiment movement with these weights
    sentiment_with_movement = calculate_sentiment_movement(sentiment_df, prob_type, w_pos, w_neg, w_neu)
    
    # Match with price movements
    merged = match_sentiment_price_movement(sentiment_with_movement, cpo_df)
    
    if len(merged) == 0:
        return {
            'match_rate': 0.0,
            'mismatch_rate': 1.0,
            'total_months': 0,
            'matches': 0,
            'mismatches': 0,
            'partial': 0,
            'score': -999  # Penalty for no data
        }
    
    # Calculate statistics
    total_months = len(merged)
    matches = len(merged[merged['Match_Status'] == 'Match'])
    mismatches = len(merged[merged['Match_Status'] == 'Mismatch'])
    partial = len(merged[merged['Match_Status'] == 'Partial'])
    
    match_rate = matches / total_months * 100
    mismatch_rate = mismatches / total_months * 100
    
    # Combined score: maximize match rate, minimize mismatch rate
    # Score = match_rate - mismatch_rate (higher is better)
    score = match_rate - mismatch_rate
    
    return {
        'match_rate': match_rate,
        'mismatch_rate': mismatch_rate,
        'total_months': total_months,
        'matches': matches,
        'mismatches': mismatches,
        'partial': partial,
        'score': score
    }

def _init_worker(sentiment_df, cpo_df):
    """Initialize worker process with dataframes."""
    global _global_sentiment_df, _global_cpo_df
    _global_sentiment_df = sentiment_df
    _global_cpo_df = cpo_df

# Global variables for multiprocessing (set before creating pool)
_global_sentiment_df = None
_global_cpo_df = None

def evaluate_single_combination(args):
    """Worker function for parallel processing."""
    global _global_sentiment_df, _global_cpo_df
    prob_type, w_pos, w_neg, w_neu = args
    try:
        result = evaluate_weights(_global_sentiment_df, _global_cpo_df, prob_type, w_pos, w_neg, w_neu)
        result['prob_type'] = prob_type
        result['w_pos'] = w_pos
        result['w_neg'] = w_neg
        result['w_neu'] = w_neu
        return result
    except Exception as e:
        # Return a default result on error
        return {
            'prob_type': prob_type,
            'w_pos': w_pos,
            'w_neg': w_neg,
            'w_neu': w_neu,
            'match_rate': 0.0,
            'mismatch_rate': 1.0,
            'total_months': 0,
            'matches': 0,
            'mismatches': 0,
            'partial': 0,
            'score': -999,
            'error': str(e)
        }

def grid_search_optimization(sentiment_df, cpo_df, prob_type, step=0.1, use_parallel=True, n_jobs=None):
    """
    Perform grid search to find optimal weights.
    
    Args:
        sentiment_df: Sentiment dataframe
        cpo_df: CPO price dataframe
        prob_type: 'Title', 'Content', or 'Combined'
        step: Step size for grid search (default 0.1 for weights from -1 to 1)
        use_parallel: Whether to use parallel processing (default True)
        n_jobs: Number of parallel jobs (None = use all CPUs)
    """
    print(f"\n{'='*70}")
    print(f"Grid Search for {prob_type} Probabilities")
    print(f"{'='*70}")
    
    # Generate weight combinations from -1 to 1
    weight_range = np.arange(-1.0, 1.1, step)
    
    # Filter out combinations where all weights are zero
    combinations = []
    for w_pos, w_neg, w_neu in product(weight_range, weight_range, weight_range):
        if not (w_pos == 0 and w_neg == 0 and w_neu == 0):
            combinations.append((w_pos, w_neg, w_neu))
    
    print(f"Testing {len(combinations)} weight combinations...")
    print(f"Step size: {step}")
    
    # Prepare arguments for parallel processing
    if use_parallel:
        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)  # Use all but one CPU
        print(f"Using parallel processing with {n_jobs} workers...")
        
        # Prepare arguments for each combination (only weights, not dataframes)
        args_list = [(prob_type, w_pos, w_neg, w_neu) 
                     for w_pos, w_neg, w_neu in combinations]
        
        # Process in parallel (dataframes passed via initializer)
        with Pool(n_jobs, initializer=_init_worker, initargs=(sentiment_df.copy(), cpo_df.copy())) as pool:
            # Process in chunks to show progress
            chunk_size = max(100, len(combinations) // 100)  # Show progress every ~1%
            results = []
            best_score = -999
            best_result = None
            solutions_over_80 = []
            
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i+chunk_size]
                chunk_results = pool.map(evaluate_single_combination, chunk)
                
                for result in chunk_results:
                    # Skip results with errors (score = -999)
                    if result.get('error'):
                        continue
                    
                    results.append(result)
                    
                    # Track best solution
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_result = result.copy()
                    
                    # Track solutions with >80% match rate
                    if result['match_rate'] >= 80.0:
                        solutions_over_80.append(result)
                
                # Show progress
                progress = min(i + len(chunk), len(combinations))
                print(f"  Progress: {progress}/{len(combinations)} combinations tested... "
                      f"({progress/len(combinations)*100:.1f}%)")
    else:
        # Sequential processing (original method)
        print("Using sequential processing...")
        results = []
        best_score = -999
        best_result = None
        solutions_over_80 = []
        
        for idx, (w_pos, w_neg, w_neu) in enumerate(combinations):
            if (idx + 1) % 1000 == 0:
                print(f"  Progress: {idx + 1}/{len(combinations)} combinations tested... "
                      f"({(idx+1)/len(combinations)*100:.1f}%)")
            
            try:
                # Evaluate this combination
                result = evaluate_weights(sentiment_df, cpo_df, prob_type, w_pos, w_neg, w_neu)
                
                result['prob_type'] = prob_type
                result['w_pos'] = w_pos
                result['w_neg'] = w_neg
                result['w_neu'] = w_neu
                
                results.append(result)
                
                # Track best solution
                if result['score'] > best_score:
                    best_score = result['score']
                    best_result = result
                
                # Track solutions with >80% match rate
                if result['match_rate'] >= 80.0:
                    solutions_over_80.append(result)
            except Exception as e:
                # Skip this combination if there's an error
                print(f"  Warning: Error with weights ({w_pos:.2f}, {w_neg:.2f}, {w_neu:.2f}): {e}")
                continue
    
    print(f"\nCompleted testing {len(combinations)} combinations")
    print(f"Found {len(solutions_over_80)} solutions with >=80% match rate")
    
    return results, best_result, solutions_over_80

def main():
    """Main function to run the optimization."""
    print("="*70)
    print("SENTIMENT WEIGHT OPTIMIZATION FOR CPO PRICE MATCHING")
    print("Target: >80% Match Rate with Minimized Mismatches")
    print("="*70)
    
    # File paths
    cpo_file = 'cpo/Data_CPO_Monthly.csv'
    sentiment_file_tone = 'news/output/monthly_sentiment_aggregate_tone.csv'
    sentiment_file_regular = 'news/output/monthly_sentiment_aggregate.csv'
    
    # Load data
    cpo_df = load_cpo_data(cpo_file)
    sentiment_tone = load_sentiment_data(sentiment_file_tone)
    sentiment_regular = load_sentiment_data(sentiment_file_regular)
    
    # Filter to 2007-2024
    cpo_df = cpo_df[(cpo_df['Year'] >= 2007) & (cpo_df['Year'] <= 2024)]
    sentiment_tone = sentiment_tone[(sentiment_tone['Year'] >= 2007) & (sentiment_tone['Year'] <= 2024)]
    sentiment_regular = sentiment_regular[(sentiment_regular['Year'] >= 2007) & (sentiment_regular['Year'] <= 2024)]
    
    # Calculate price movements
    cpo_df = calculate_price_movement(cpo_df)
    
    # Probability types to test
    prob_types = ['Title', 'Content', 'Combined']
    
    # Sentiment files to test
    sentiment_files = {
        'Tone': sentiment_tone,
        'Regular': sentiment_regular
    }
    
    # Grid search step size (smaller = more precise but slower)
    # Options:
    #   0.2 = 11 values per weight = 1,331 combinations per type (fast)
    #   0.1 = 21 values per weight = 9,261 combinations per type (recommended with parallel)
    #   0.05 = 41 values per weight = 68,921 combinations per type (thorough)
    #   0.02 = 101 values per weight = 1,030,301 combinations per type (very thorough, slow)
    #   0.01 = 201 values per weight = 8,120,601 combinations per type (extremely thorough, very slow)
    step_size = 0.1
    
    # Calculate total combinations
    weight_count = len(np.arange(-1.0, 1.1, step_size))
    combinations_per_type = weight_count ** 3 - 1  # -1 for all zeros
    total_combinations = combinations_per_type * len(prob_types) * len(sentiment_files)
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION PARAMETERS")
    print(f"{'='*70}")
    print(f"Step size: {step_size}")
    print(f"Weight values per dimension: {weight_count}")
    print(f"Combinations per probability type: {combinations_per_type:,}")
    print(f"Total combinations to test: {total_combinations:,}")
    print(f"Probability types: {len(prob_types)}")
    print(f"Sentiment files: {len(sentiment_files)}")
    
    # Use parallel processing for faster execution
    use_parallel = True
    n_jobs = None  # None = use all available CPUs minus 1
    
    if use_parallel:
        available_cpus = cpu_count()
        workers = max(1, available_cpus - 1) if n_jobs is None else n_jobs
        print(f"Parallel processing: Enabled ({workers} workers)")
    else:
        print(f"Parallel processing: Disabled (sequential)")
    print(f"{'='*70}\n")
    
    all_results = []
    all_best_results = []
    all_solutions_over_80 = []
    
    # Run optimization for each sentiment file and probability type
    for file_name, sentiment_df in sentiment_files.items():
        print(f"\n\n{'#'*70}")
        print(f"Processing {file_name} Sentiment File")
        print(f"{'#'*70}")
        
        for prob_type in prob_types:
            results, best_result, solutions_over_80 = grid_search_optimization(
                sentiment_df, cpo_df, prob_type, step=step_size, 
                use_parallel=use_parallel, n_jobs=n_jobs
            )
            
            # Add file name to results
            for r in results:
                r['sentiment_file'] = file_name
            for r in solutions_over_80:
                r['sentiment_file'] = file_name
            if best_result:
                best_result['sentiment_file'] = file_name
            
            all_results.extend(results)
            all_best_results.append(best_result)
            all_solutions_over_80.extend(solutions_over_80)
            
            # Print best result for this combination
            if best_result:
                print(f"\nBest result for {file_name} - {prob_type}:")
                print(f"  Weights: Pos={best_result['w_pos']:.2f}, Neg={best_result['w_neg']:.2f}, Neu={best_result['w_neu']:.2f}")
                print(f"  Match Rate: {best_result['match_rate']:.2f}%")
                print(f"  Mismatch Rate: {best_result['mismatch_rate']:.2f}%")
                print(f"  Score: {best_result['score']:.2f}")
    
    # Validate results before converting to DataFrame
    if len(all_results) == 0:
        print("\n⚠ Warning: No results to save! Check data files and filters.")
        return None, None, None
    
    # Convert to DataFrame for easier analysis
    try:
        results_df = pd.DataFrame(all_results)
        solutions_over_80_df = pd.DataFrame(all_solutions_over_80) if all_solutions_over_80 else pd.DataFrame()
    except Exception as e:
        print(f"\n⚠ Error converting results to DataFrame: {e}")
        return None, None, None
    
    # Save all results
    try:
        results_df.to_csv('output/sentiment_weight_optimization_all_results.csv', index=False)
        print(f"\nAll results saved to: output/sentiment_weight_optimization_all_results.csv")
    except Exception as e:
        print(f"\n⚠ Error saving results to CSV: {e}")
    
    # Save solutions with >=80% match rate
    if len(solutions_over_80_df) > 0:
        try:
            solutions_over_80_df = solutions_over_80_df.sort_values('score', ascending=False)
            solutions_over_80_df.to_csv('output/sentiment_weight_optimization_solutions_over_80.csv', index=False)
            print(f"Solutions with >=80% match rate saved to: output/sentiment_weight_optimization_solutions_over_80.csv")
            print(f"\nTotal solutions with >=80% match rate: {len(solutions_over_80_df)}")
        except Exception as e:
            print(f"\n⚠ Error saving solutions to CSV: {e}")
    else:
        print("\n⚠ No solutions found with >=80% match rate")
        print("Consider:")
        print("  1. Reducing the step size for finer grid search")
        print("  2. Adjusting the threshold for sentiment movement")
        print("  3. Using different probability combinations")
    
    # Always save the best models (one per combination: file + prob_type)
    try:
        if all_best_results:
            best_models_df = pd.DataFrame(all_best_results)
            best_models_df = best_models_df.sort_values('score', ascending=False)
            best_models_df.to_csv('output/sentiment_weight_optimization_best_models.csv', index=False)
            print(f"\nBest models (one per combination) saved to: output/sentiment_weight_optimization_best_models.csv")
            print(f"Total best models saved: {len(best_models_df)}")
            
            # Also save top N best models overall (e.g., top 20)
            top_n = min(20, len(best_models_df))
            top_models_df = best_models_df.head(top_n)
            top_models_df.to_csv('output/sentiment_weight_optimization_top_models.csv', index=False)
            print(f"Top {top_n} best models saved to: output/sentiment_weight_optimization_top_models.csv")
        else:
            print("\n⚠ No best models to save")
    except Exception as e:
        print(f"\n⚠ Error saving best models to CSV: {e}")
    
    # Print summary of best solutions
    print("\n" + "="*70)
    print("SUMMARY OF BEST SOLUTIONS")
    print("="*70)
    
    if len(solutions_over_80_df) > 0:
        print("\nTop 10 Solutions with >=80% Match Rate:")
        print("-"*70)
        try:
            top_10 = solutions_over_80_df.head(10)
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                print(f"\n{idx}. {row['sentiment_file']} - {row['prob_type']} Probabilities")
                print(f"   Weights: Pos={row['w_pos']:.2f}, Neg={row['w_neg']:.2f}, Neu={row['w_neu']:.2f}")
                print(f"   Match Rate: {row['match_rate']:.2f}%")
                print(f"   Mismatch Rate: {row['mismatch_rate']:.2f}%")
                print(f"   Score: {row['score']:.2f}")
                print(f"   Total Months: {int(row['total_months'])}")
        except Exception as e:
            print(f"Error displaying top solutions: {e}")
    else:
        print("\nBest solutions overall (even if <80%):")
        print("-"*70)
        try:
            if all_best_results:
                best_overall = pd.DataFrame(all_best_results).sort_values('score', ascending=False).head(10)
                for idx, (_, row) in enumerate(best_overall.iterrows(), 1):
                    print(f"\n{idx}. {row['sentiment_file']} - {row['prob_type']} Probabilities")
                    print(f"   Weights: Pos={row['w_pos']:.2f}, Neg={row['w_neg']:.2f}, Neu={row['w_neu']:.2f}")
                    print(f"   Match Rate: {row['match_rate']:.2f}%")
                    print(f"   Mismatch Rate: {row['mismatch_rate']:.2f}%")
                    print(f"   Score: {row['score']:.2f}")
            else:
                print("No results available to display.")
        except Exception as e:
            print(f"Error displaying best solutions: {e}")
    
    print("\n" + "="*70)
    print("Optimization complete!")
    print("="*70)
    
    # Return best models as well
    best_models_df = pd.DataFrame(all_best_results) if all_best_results else pd.DataFrame()
    
    return results_df, solutions_over_80_df, best_models_df

if __name__ == "__main__":
    all_results, solutions_over_80, best_models = main()

