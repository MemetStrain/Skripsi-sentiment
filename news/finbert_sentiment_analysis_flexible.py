import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_CSV = 'mpob_news_preprocessed.csv'
OUTPUT_CSV = 'mpob_news_with_sentiment.csv'
BATCH_SIZE = 16  # Process articles in batches (will auto-adjust for GPU)
MAX_LENGTH = 512  # Maximum token length for FinBERT
USE_HALF_PRECISION = True  # Use float16 for faster GPU inference (recommended for CUDA)
FORCE_CPU = False  # Set to True to force CPU even if CUDA is available

# Aggregation Configuration
AGGREGATION_MODE = 'Daily'  # Daily aggregation only
AGGREGATION_OUTPUT_CSV = f'output/sentiment_aggregate_{AGGREGATION_MODE}.csv'  # Output file for aggregation

# CUDA Configuration
def get_device():
    """Get the best available device with CUDA information."""
    if FORCE_CPU:
        print("CPU mode forced by configuration.")
        return 'cpu', None
    
    if torch.cuda.is_available():
        device = 'cuda'
        device_id = 0  # Use first GPU (can be changed if multiple GPUs)
        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
        
        print(f"\n{'='*70}")
        print("CUDA DETECTED - GPU MODE ENABLED")
        print(f"{'='*70}")
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"{'='*70}\n")
        
        # Set default GPU device
        torch.cuda.set_device(device_id)
        return f'cuda:{device_id}', gpu_memory
    else:
        print("\nCUDA not available. Using CPU mode.")
        print("For GPU acceleration, ensure you have:")
        print("  1. NVIDIA GPU with CUDA support")
        print("  2. CUDA toolkit installed")
        print("  3. PyTorch with CUDA support (install: pip install torch --index-url https://download.pytorch.org/whl/cu118)")
        return 'cpu', None

DEVICE, GPU_MEMORY = get_device()

# Auto-adjust batch size based on GPU memory
if DEVICE != 'cpu' and GPU_MEMORY:
    if GPU_MEMORY < 4:
        BATCH_SIZE = min(BATCH_SIZE, 8)  # Small GPU memory
    elif GPU_MEMORY < 8:
        BATCH_SIZE = min(BATCH_SIZE, 16)  # Medium GPU memory
    else:
        BATCH_SIZE = min(BATCH_SIZE, 32)  # Large GPU memory

# FinBERT labels (Araci's FinBERT uses lowercase labels)
LABELS = ['Negative', 'Neutral', 'Positive']

def load_finbert_model():
    """Load FinBERT model and tokenizer (Araci's FinBERT) with CUDA optimization."""
    print(f"\nLoading FinBERT model (Araci/ProsusAI)...")
    print(f"Device: {DEVICE}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Use half precision (float16) for faster inference on GPU
    if USE_HALF_PRECISION and DEVICE != 'cpu':
        try:
            model = model.half()  # Convert to float16
            print("Using half precision (float16) for faster GPU inference")
        except Exception as e:
            print(f"Warning: Could not enable half precision: {e}")
            print("Falling back to full precision (float32)")
    
    model.eval()  # Set to evaluation mode
    
    # Clear GPU cache if using CUDA
    if DEVICE != 'cpu':
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
    
    print("FinBERT model loaded successfully!")
    
    if DEVICE != 'cpu':
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU Memory allocated: {allocated:.2f} GB")
    
    return model, tokenizer

def analyze_sentiment_finbert(text, model, tokenizer):
    """
    Analyze sentiment using FinBERT.
    Returns: (sentiment_label, confidence_score, probabilities_dict)
    """
    if not text or pd.isna(text) or str(text).strip() == '':
        return 'Neutral', 0.0, {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 1.0}
    
    # Clean and prepare text
    text = str(text).strip()
    
    # Truncate if too long (FinBERT max length is 512 tokens)
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_LENGTH,
        padding=True
    )
    
    # Move inputs to the same device as model
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Handle half precision conversion if needed
        if logits.dtype == torch.float16:
            # Convert to float32 for softmax (more stable)
            logits = logits.float()
        
        probabilities = torch.softmax(logits, dim=1)
        
        # Move to CPU and convert to numpy
        if DEVICE != 'cpu':
            probabilities = probabilities.cpu().numpy()[0]
        else:
            probabilities = probabilities.numpy()[0]
    
    # Get predicted label and confidence
    max_index = np.argmax(probabilities)
    sentiment = LABELS[max_index]
    confidence = float(probabilities[max_index])
    
    # Create probabilities dictionary (Araci's FinBERT order: Negative, Neutral, Positive)
    prob_dict = {
        'Negative': float(probabilities[0]),
        'Neutral': float(probabilities[1]),
        'Positive': float(probabilities[2])
    }
    
    return sentiment, confidence, prob_dict

def analyze_article_sentiment(row, model, tokenizer):
    """
    Analyze sentiment for both title and content of an article.
    Returns a dictionary with sentiment scores.
    """
    results = {}
    
    # Analyze title sentiment
    title_text = row.get('Title', '')
    title_sentiment, title_confidence, title_probs = analyze_sentiment_finbert(title_text, model, tokenizer)
    
    results['Title_Sentiment'] = title_sentiment
    results['Title_Confidence'] = title_confidence
    results['Title_Positive_Prob'] = title_probs['Positive']
    results['Title_Negative_Prob'] = title_probs['Negative']
    results['Title_Neutral_Prob'] = title_probs['Neutral']
    
    # Analyze content sentiment
    content_text = row.get('Content', '')
    # Skip error messages
    if '[Error:' in str(content_text):
        results['Content_Sentiment'] = 'Neutral'
        results['Content_Confidence'] = 0.0
        results['Content_Positive_Prob'] = 0.0
        results['Content_Negative_Prob'] = 0.0
        results['Content_Neutral_Prob'] = 1.0
        content_probs = {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 1.0}
    else:
        content_sentiment, content_confidence, content_probs = analyze_sentiment_finbert(content_text, model, tokenizer)
        results['Content_Sentiment'] = content_sentiment
        results['Content_Confidence'] = content_confidence
        results['Content_Positive_Prob'] = content_probs['Positive']
        results['Content_Negative_Prob'] = content_probs['Negative']
        results['Content_Neutral_Prob'] = content_probs['Neutral']
    
    # Combined sentiment (weighted: 30% title, 70% content)
    # Calculate weighted probabilities
    combined_pos = 0.3 * title_probs['Positive'] + 0.7 * content_probs['Positive']
    combined_neg = 0.3 * title_probs['Negative'] + 0.7 * content_probs['Negative']
    combined_neu = 0.3 * title_probs['Neutral'] + 0.7 * content_probs['Neutral']
    
    combined_probs = {
        'Positive': combined_pos,
        'Negative': combined_neg,
        'Neutral': combined_neu
    }
    
    combined_sentiment = max(combined_probs, key=combined_probs.get)
    combined_confidence = combined_probs[combined_sentiment]
    
    results['Combined_Sentiment'] = combined_sentiment
    results['Combined_Confidence'] = combined_confidence
    results['Combined_Positive_Prob'] = combined_pos
    results['Combined_Negative_Prob'] = combined_neg
    results['Combined_Neutral_Prob'] = combined_neu
    
    return results

def process_batch(df_batch, model, tokenizer):
    """Process a batch of articles with GPU memory management."""
    results_list = []
    for idx, row in df_batch.iterrows():
        try:
            sentiment_results = analyze_article_sentiment(row, model, tokenizer)
            results_list.append(sentiment_results)
        except RuntimeError as e:
            # Handle CUDA out of memory errors
            if 'out of memory' in str(e) and DEVICE != 'cpu':
                print(f"\nCUDA out of memory error on article {idx}. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                # Retry once
                try:
                    sentiment_results = analyze_article_sentiment(row, model, tokenizer)
                    results_list.append(sentiment_results)
                except Exception as retry_e:
                    print(f"Retry failed: {retry_e}")
                    default_results = get_default_sentiment_results()
                    results_list.append(default_results)
            else:
                print(f"\nError processing article {idx}: {e}")
                default_results = get_default_sentiment_results()
                results_list.append(default_results)
        except Exception as e:
            print(f"\nError processing article {idx}: {e}")
            default_results = get_default_sentiment_results()
            results_list.append(default_results)
    
    # Periodically clear GPU cache to prevent memory buildup
    if DEVICE != 'cpu' and len(results_list) % 10 == 0:
        torch.cuda.empty_cache()
    
    return results_list

def get_default_sentiment_results():
    """Return default Neutral sentiment results."""
    return {
        'Title_Sentiment': 'Neutral', 'Title_Confidence': 0.0,
        'Title_Positive_Prob': 0.0, 'Title_Negative_Prob': 0.0, 'Title_Neutral_Prob': 1.0,
        'Content_Sentiment': 'Neutral', 'Content_Confidence': 0.0,
        'Content_Positive_Prob': 0.0, 'Content_Negative_Prob': 0.0, 'Content_Neutral_Prob': 1.0,
        'Combined_Sentiment': 'Neutral', 'Combined_Confidence': 0.0,
        'Combined_Positive_Prob': 0.0, 'Combined_Negative_Prob': 0.0, 'Combined_Neutral_Prob': 1.0
    }

def aggregate_sentiment(df):
    """
    Aggregate sentiment scores by daily periods.

    Parameters:
    - df: DataFrame with sentiment scores
    - mode: must be 'Daily'

    Returns: DataFrame with aggregated sentiment
    """
    print("\n" + "=" * 70)
    print(f"AGGREGATING DAILY SENTIMENT SCORES")
    print("=" * 70)

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # Remove rows with invalid dates
    df_valid = df[df['Date'].notna()].copy()

    if len(df_valid) == 0:
        print(f"Warning: No valid dates found. Cannot aggregate.")
        return None

    # Sort by date
    df_valid = df_valid.sort_values('Date').reset_index(drop=True)

    return aggregate_Daily_sentiment(df_valid)

def aggregate_Daily_sentiment(df):
    """
    Aggregate sentiment scores by day (working days, Mon-Fri).
    For days without news, forward-fill from the previous working day.
    """
    # Ensure dates are sorted
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Get date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Create complete date range for working days (Mon-Fri)
    all_dates = pd.date_range(start=min_date, end=max_date, freq='B')  # 'B' = business days
    print(f"Total working days in range: {len(all_dates)}")
    
    # Group by date and aggregate
    Daily_agg = df.groupby('Date').agg({
        'Title_Positive_Prob': 'mean',
        'Title_Negative_Prob': 'mean',
        'Title_Neutral_Prob': 'mean',
        'Title_Confidence': 'mean',
        'Content_Positive_Prob': 'mean',
        'Content_Negative_Prob': 'mean',
        'Content_Neutral_Prob': 'mean',
        'Content_Confidence': 'mean',
        'Combined_Positive_Prob': 'mean',
        'Combined_Negative_Prob': 'mean',
        'Combined_Neutral_Prob': 'mean',
        'Combined_Confidence': 'mean',
        'Title': 'count'  # Count articles per day
    }).rename(columns={'Title': 'Article_Count'})
    
    # Create DataFrame with all working days
    complete_df = pd.DataFrame({'Date': all_dates})
    complete_df = complete_df.merge(Daily_agg, on='Date', how='left')
    
    # Forward fill missing days (use previous day's sentiment)
    fill_cols = [col for col in complete_df.columns if col not in ['Date', 'Article_Count']]
    complete_df[fill_cols] = complete_df[fill_cols].fillna(method='ffill')
    
    # Fill article count with 0 for days without news
    complete_df['Article_Count'] = complete_df['Article_Count'].fillna(0).astype(int)
    
    # Calculate sentiment score (Positive - Negative)
    complete_df['Sentiment_Score'] = (
        complete_df['Combined_Positive_Prob'] - complete_df['Combined_Negative_Prob']
    )
    
    # Determine dominant sentiment
    def get_dominant_sentiment(row):
        probs = {
            'Positive': row['Combined_Positive_Prob'],
            'Negative': row['Combined_Negative_Prob'],
            'Neutral': row['Combined_Neutral_Prob']
        }
        return max(probs, key=probs.get)
    
    complete_df['Dominant_Sentiment'] = complete_df.apply(get_dominant_sentiment, axis=1)
    
    # Add date components
    complete_df['Year'] = complete_df['Date'].dt.year
    complete_df['Month'] = complete_df['Date'].dt.month
    complete_df['Day'] = complete_df['Date'].dt.day
    complete_df['Weekday'] = complete_df['Date'].dt.day_name()
    complete_df['Date_Str'] = complete_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Reorder columns
    cols = ['Date', 'Date_Str', 'Year', 'Month', 'Day', 'Weekday', 'Article_Count']
    other_cols = [col for col in complete_df.columns if col not in cols]
    complete_df = complete_df[cols + other_cols]
    
    # Round numeric columns
    numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Month', 'Day', 'Article_Count']]
    complete_df[numeric_cols] = complete_df[numeric_cols].round(4)
    
    print(f"Created Daily aggregation for {len(complete_df)} working days")
    print(f"Days with news: {(complete_df['Article_Count'] > 0).sum()}")
    print(f"Days without news (forward-filled): {(complete_df['Article_Count'] == 0).sum()}")
    
    return complete_df

def _aggregate_Monthly_sentiment_UNUSED(df):
    # kept for reference only — not called
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')
    
    # Group by Year-Month
    Monthly_data = []
    
    for year_month, group in df.groupby('YearMonth'):
        year = year_month.year
        month = year_month.month
        month_name = group['MonthName'].iloc[0]
        total_articles = len(group)
        
        agg = {
            'Year': year,
            'Month': month,
            'MonthName': month_name,
            'YearMonth': str(year_month),
            'Total_Articles': total_articles,
            # Title metrics
            'Title_Avg_Positive_Prob': group['Title_Positive_Prob'].mean(),
            'Title_Avg_Negative_Prob': group['Title_Negative_Prob'].mean(),
            'Title_Avg_Neutral_Prob': group['Title_Neutral_Prob'].mean(),
            'Title_Avg_Confidence': group['Title_Confidence'].mean(),
            # Content metrics
            'Content_Avg_Positive_Prob': group['Content_Positive_Prob'].mean(),
            'Content_Avg_Negative_Prob': group['Content_Negative_Prob'].mean(),
            'Content_Avg_Neutral_Prob': group['Content_Neutral_Prob'].mean(),
            'Content_Avg_Confidence': group['Content_Confidence'].mean(),
            # Combined metrics
            'Combined_Avg_Positive_Prob': group['Combined_Positive_Prob'].mean(),
            'Combined_Avg_Negative_Prob': group['Combined_Negative_Prob'].mean(),
            'Combined_Avg_Neutral_Prob': group['Combined_Neutral_Prob'].mean(),
            'Combined_Avg_Confidence': group['Combined_Confidence'].mean(),
        }
        
        # Calculate sentiment score
        agg['Sentiment_Score'] = (
            agg['Combined_Avg_Positive_Prob'] - agg['Combined_Avg_Negative_Prob']
        )
        
        # Dominant sentiment
        probs = {
            'Positive': agg['Combined_Avg_Positive_Prob'],
            'Negative': agg['Combined_Avg_Negative_Prob'],
            'Neutral': agg['Combined_Avg_Neutral_Prob']
        }
        agg['Dominant_Sentiment'] = max(probs, key=probs.get)
        
        Monthly_data.append(agg)
    
    # Create DataFrame
    Monthly_df = pd.DataFrame(Monthly_data)
    Monthly_df = Monthly_df.sort_values(['Year', 'Month']).reset_index(drop=True)
    
    # Round numeric columns
    numeric_cols = Monthly_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Month', 'Total_Articles']]
    Monthly_df[numeric_cols] = Monthly_df[numeric_cols].round(4)
    
    print(f"Aggregated sentiment for {len(Monthly_df)} months")
    print(f"Date range: {Monthly_df['YearMonth'].min()} to {Monthly_df['YearMonth'].max()}")
    
    return Monthly_df

def main():
    print("=" * 70)
    print("FinBERT Sentiment Analysis for MPOB News (Flexible Aggregation)")
    print("=" * 70)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Aggregation Mode: {AGGREGATION_MODE if AGGREGATION_MODE else 'None (No aggregation)'}")
    if AGGREGATION_MODE:
        print(f"  Aggregation Output: {AGGREGATION_OUTPUT_CSV}")
    
    # Load data
    print(f"\nLoading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} articles")
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found!")
        return
    
    # Check if sentiment columns already exist
    sentiment_cols = ['Title_Sentiment', 'Content_Sentiment', 'Combined_Sentiment']
    if all(col in df.columns for col in sentiment_cols):
        print("\nSentiment columns already exist. Processing only missing rows...")
        # Process only rows with missing sentiment
        mask = df['Combined_Sentiment'].isna()
        df_to_process = df[mask].copy()
        df_processed = df[~mask].copy()
        
        if len(df_to_process) == 0:
            print("All articles already have sentiment scores!")
            df_final = df.copy()
            
            # Perform aggregation if requested
            if AGGREGATION_MODE:
                agg_df = aggregate_sentiment(df_final)
                if agg_df is not None and len(agg_df) > 0:
                    print(f"\nSaving {AGGREGATION_MODE} aggregation to {AGGREGATION_OUTPUT_CSV}...")
                    agg_df.to_csv(AGGREGATION_OUTPUT_CSV, index=False)
                    print(f"Saved {AGGREGATION_MODE} sentiment aggregation!")
                    
                    # Display sample
                    print("\n" + "=" * 70)
                    print("DAILY SENTIMENT AGGREGATION SAMPLE")
                    print("=" * 70)
                    display_cols = ['Date_Str', 'Article_Count', 'Sentiment_Score',
                                    'Combined_Positive_Prob', 'Combined_Negative_Prob', 'Dominant_Sentiment']
                    print(agg_df[display_cols].tail(10).to_string(index=False))

            return
        
        print(f"Processing {len(df_to_process)} remaining articles...")
    else:
        df_to_process = df.copy()
        df_processed = pd.DataFrame()
    
    # Load model
    model, tokenizer = load_finbert_model()
    
    # Process articles in batches
    print(f"\nProcessing articles in batches of {BATCH_SIZE}...")
    if DEVICE != 'cpu':
        print(f"Using GPU acceleration with batch size: {BATCH_SIZE}")
    all_results = []
    
    # Split into batches
    num_batches = (len(df_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(df_to_process), BATCH_SIZE), desc="Processing batches"):
        batch = df_to_process.iloc[i:i+BATCH_SIZE]
        batch_results = process_batch(batch, model, tokenizer)
        all_results.extend(batch_results)
        
        # Clear GPU cache every 5 batches to prevent memory issues
        if DEVICE != 'cpu' and (i // BATCH_SIZE + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Final GPU cache clear
    if DEVICE != 'cpu':
        torch.cuda.empty_cache()
        # Print peak memory usage
        if hasattr(torch.cuda, 'max_memory_allocated'):
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"\nPeak GPU memory usage: {peak_memory:.2f} GB")
    
    # Add sentiment results to dataframe
    sentiment_df = pd.DataFrame(all_results)
    df_to_process = pd.concat([df_to_process.reset_index(drop=True), sentiment_df], axis=1)
    
    # Combine with already processed data if exists
    if len(df_processed) > 0:
        df_final = pd.concat([df_processed, df_to_process], ignore_index=True)
    else:
        df_final = df_to_process
    
    # Save results
    print(f"\nSaving results to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_final)} articles with sentiment scores!")
    
    # Perform aggregation if requested
    if AGGREGATION_MODE:
        agg_df = aggregate_sentiment(df_final)
        if agg_df is not None and len(agg_df) > 0:
            print(f"\nSaving {AGGREGATION_MODE} aggregation to {AGGREGATION_OUTPUT_CSV}...")
            agg_df.to_csv(AGGREGATION_OUTPUT_CSV, index=False)
            print(f"Saved {AGGREGATION_MODE} sentiment aggregation!")
            
            # Display sample
            print("\n" + "=" * 70)
            print("DAILY SENTIMENT AGGREGATION SAMPLE")
            print("=" * 70)
            display_cols = ['Date_Str', 'Article_Count', 'Sentiment_Score',
                            'Combined_Positive_Prob', 'Combined_Negative_Prob', 'Dominant_Sentiment']
            print(agg_df[display_cols].tail(10).to_string(index=False))
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\nCombined Sentiment Distribution:")
    print(df_final['Combined_Sentiment'].value_counts())
    print(f"\n  Positive: {(df_final['Combined_Sentiment'] == 'Positive').sum()} ({(df_final['Combined_Sentiment'] == 'Positive').sum()/len(df_final)*100:.2f}%)")
    print(f"  Negative: {(df_final['Combined_Sentiment'] == 'Negative').sum()} ({(df_final['Combined_Sentiment'] == 'Negative').sum()/len(df_final)*100:.2f}%)")
    print(f"  Neutral:  {(df_final['Combined_Sentiment'] == 'Neutral').sum()} ({(df_final['Combined_Sentiment'] == 'Neutral').sum()/len(df_final)*100:.2f}%)")
    
    print(f"\nAverage Confidence Scores:")
    print(f"  Title:    {df_final['Title_Confidence'].mean():.3f}")
    print(f"  Content:  {df_final['Content_Confidence'].mean():.3f}")
    print(f"  Combined: {df_final['Combined_Confidence'].mean():.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
