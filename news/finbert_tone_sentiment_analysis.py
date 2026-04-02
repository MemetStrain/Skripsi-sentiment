import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_CSV = 'mpob_news_fast.csv'
OUTPUT_CSV = 'mpob_news_with_sentiment_tone.csv'
MONTHLY_AGGREGATE_CSV = 'output/monthly_sentiment_aggregate_tone.csv'
BATCH_SIZE = 16  # Process articles in batches (will auto-adjust for GPU)
MAX_LENGTH = 512  # Maximum token length for FinBERT
USE_HALF_PRECISION = True  # Use float16 for faster GPU inference (recommended for CUDA)
FORCE_CPU = False  # Set to True to force CPU even if CUDA is available

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
        print("  3. PyTorch with CUDA support (install: pip install torch --index-url https://download.pytorch.org/whl/cu124)")
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

# FinBERT Tone labels (order: Positive, Negative, Neutral)
LABELS = ['Positive', 'Negative', 'Neutral']

def load_finbert_model():
    """Load FinBERT Tone model and tokenizer with CUDA optimization."""
    print(f"\nLoading FinBERT Tone model (yiyanghkust/finbert-tone)...")
    print(f"Device: {DEVICE}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    
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
    
    print("FinBERT Tone model loaded successfully!")
    
    if DEVICE != 'cpu':
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU Memory allocated: {allocated:.2f} GB")
    
    return model, tokenizer

def analyze_sentiment_finbert(text, model, tokenizer):
    """
    Analyze sentiment using FinBERT Tone.
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
    
    # Create probabilities dictionary (FinBERT Tone order: Positive, Negative, Neutral)
    prob_dict = {
        'Positive': float(probabilities[0]),
        'Negative': float(probabilities[1]),
        'Neutral': float(probabilities[2])
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

def aggregate_monthly_sentiment(df):
    """
    Aggregate sentiment scores by month and create summary statistics.
    Returns a DataFrame with monthly sentiment aggregations.
    """
    print("\n" + "=" * 70)
    print("AGGREGATING MONTHLY SENTIMENT SCORES")
    print("=" * 70)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    
    # Remove rows with invalid dates
    df_valid = df[df['Date'].notna()].copy()
    
    if len(df_valid) == 0:
        print("Warning: No valid dates found. Cannot aggregate by month.")
        return None
    
    # Create Year-Month column
    df_valid['YearMonth'] = df_valid['Date'].dt.to_period('M')
    df_valid['Year'] = df_valid['Date'].dt.year
    df_valid['Month'] = df_valid['Date'].dt.month
    df_valid['MonthName'] = df_valid['Date'].dt.strftime('%B')
    
    # Group by Year-Month
    monthly_data = []
    
    for year_month, group in df_valid.groupby('YearMonth'):
        year = year_month.year
        month = year_month.month
        month_name = group['MonthName'].iloc[0]
        total_articles = len(group)
        
        # Initialize aggregation dictionary
        agg = {
            'Year': year,
            'Month': month,
            'MonthName': month_name,
            'YearMonth': str(year_month),
            'Total_Articles': total_articles
        }
        
        # Aggregate for Title Sentiment
        title_counts = group['Title_Sentiment'].value_counts()
        agg['Title_Positive_Count'] = title_counts.get('Positive', 0)
        agg['Title_Negative_Count'] = title_counts.get('Negative', 0)
        agg['Title_Neutral_Count'] = title_counts.get('Neutral', 0)
        agg['Title_Positive_Pct'] = (agg['Title_Positive_Count'] / total_articles) * 100
        agg['Title_Negative_Pct'] = (agg['Title_Negative_Count'] / total_articles) * 100
        agg['Title_Neutral_Pct'] = (agg['Title_Neutral_Count'] / total_articles) * 100
        agg['Title_Avg_Confidence'] = group['Title_Confidence'].mean()
        agg['Title_Avg_Positive_Prob'] = group['Title_Positive_Prob'].mean()
        agg['Title_Avg_Negative_Prob'] = group['Title_Negative_Prob'].mean()
        agg['Title_Avg_Neutral_Prob'] = group['Title_Neutral_Prob'].mean()
        
        # Aggregate for Content Sentiment
        content_counts = group['Content_Sentiment'].value_counts()
        agg['Content_Positive_Count'] = content_counts.get('Positive', 0)
        agg['Content_Negative_Count'] = content_counts.get('Negative', 0)
        agg['Content_Neutral_Count'] = content_counts.get('Neutral', 0)
        agg['Content_Positive_Pct'] = (agg['Content_Positive_Count'] / total_articles) * 100
        agg['Content_Negative_Pct'] = (agg['Content_Negative_Count'] / total_articles) * 100
        agg['Content_Neutral_Pct'] = (agg['Content_Neutral_Count'] / total_articles) * 100
        agg['Content_Avg_Confidence'] = group['Content_Confidence'].mean()
        agg['Content_Avg_Positive_Prob'] = group['Content_Positive_Prob'].mean()
        agg['Content_Avg_Negative_Prob'] = group['Content_Negative_Prob'].mean()
        agg['Content_Avg_Neutral_Prob'] = group['Content_Neutral_Prob'].mean()
        
        # Aggregate for Combined Sentiment
        combined_counts = group['Combined_Sentiment'].value_counts()
        agg['Combined_Positive_Count'] = combined_counts.get('Positive', 0)
        agg['Combined_Negative_Count'] = combined_counts.get('Negative', 0)
        agg['Combined_Neutral_Count'] = combined_counts.get('Neutral', 0)
        agg['Combined_Positive_Pct'] = (agg['Combined_Positive_Count'] / total_articles) * 100
        agg['Combined_Negative_Pct'] = (agg['Combined_Negative_Count'] / total_articles) * 100
        agg['Combined_Neutral_Pct'] = (agg['Combined_Neutral_Count'] / total_articles) * 100
        agg['Combined_Avg_Confidence'] = group['Combined_Confidence'].mean()
        agg['Combined_Avg_Positive_Prob'] = group['Combined_Positive_Prob'].mean()
        agg['Combined_Avg_Negative_Prob'] = group['Combined_Negative_Prob'].mean()
        agg['Combined_Avg_Neutral_Prob'] = group['Combined_Neutral_Prob'].mean()
        
        # Calculate sentiment score (Positive - Negative, normalized to -1 to 1)
        # Using Combined sentiment as primary metric
        agg['Sentiment_Score'] = (agg['Combined_Positive_Count'] - agg['Combined_Negative_Count']) / total_articles
        
        # Dominant sentiment (most common)
        agg['Dominant_Sentiment'] = combined_counts.index[0] if len(combined_counts) > 0 else 'Neutral'
        
        monthly_data.append(agg)
    
    # Create DataFrame and sort by date
    monthly_df = pd.DataFrame(monthly_data)
    monthly_df = monthly_df.sort_values(['Year', 'Month']).reset_index(drop=True)
    
    # Round numeric columns for readability
    numeric_cols = monthly_df.select_dtypes(include=[np.number]).columns
    monthly_df[numeric_cols] = monthly_df[numeric_cols].round(4)
    
    print(f"\nAggregated sentiment for {len(monthly_df)} months")
    print(f"Date range: {monthly_df['YearMonth'].min()} to {monthly_df['YearMonth'].max()}")
    
    return monthly_df

def main():
    print("=" * 70)
    print("FinBERT Tone Sentiment Analysis for MPOB News")
    print("=" * 70)
    
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
            # Still aggregate monthly sentiment even if no new processing needed
            df_final = df.copy()
            monthly_df = aggregate_monthly_sentiment(df_final)
            if monthly_df is not None and len(monthly_df) > 0:
                print(f"\nSaving monthly aggregation to {MONTHLY_AGGREGATE_CSV}...")
                monthly_df.to_csv(MONTHLY_AGGREGATE_CSV, index=False)
                print(f"Saved monthly sentiment aggregation for {len(monthly_df)} months!")
                display_cols = ['YearMonth', 'Total_Articles', 'Combined_Positive_Pct', 
                               'Combined_Negative_Pct', 'Combined_Neutral_Pct', 'Sentiment_Score', 
                               'Dominant_Sentiment']
                print("\n" + "=" * 70)
                print("MONTHLY SENTIMENT AGGREGATION SAMPLE (Last 5 months)")
                print("=" * 70)
                print(monthly_df[display_cols].tail(5).to_string(index=False))
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
    
    # Aggregate monthly sentiment
    monthly_df = aggregate_monthly_sentiment(df_final)
    if monthly_df is not None and len(monthly_df) > 0:
        print(f"\nSaving monthly aggregation to {MONTHLY_AGGREGATE_CSV}...")
        monthly_df.to_csv(MONTHLY_AGGREGATE_CSV, index=False)
        print(f"Saved monthly sentiment aggregation for {len(monthly_df)} months!")
        
        # Display sample of monthly aggregation
        print("\n" + "=" * 70)
        print("MONTHLY SENTIMENT AGGREGATION SAMPLE (Last 5 months)")
        print("=" * 70)
        display_cols = ['YearMonth', 'Total_Articles', 'Combined_Positive_Pct', 
                       'Combined_Negative_Pct', 'Combined_Neutral_Pct', 'Sentiment_Score', 
                       'Dominant_Sentiment']
        print(monthly_df[display_cols].tail(5).to_string(index=False))
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\nTitle Sentiment Distribution:")
    print(df_final['Title_Sentiment'].value_counts())
    print(f"\n  Positive: {(df_final['Title_Sentiment'] == 'Positive').sum()} ({(df_final['Title_Sentiment'] == 'Positive').sum()/len(df_final)*100:.2f}%)")
    print(f"  Negative: {(df_final['Title_Sentiment'] == 'Negative').sum()} ({(df_final['Title_Sentiment'] == 'Negative').sum()/len(df_final)*100:.2f}%)")
    print(f"  Neutral:  {(df_final['Title_Sentiment'] == 'Neutral').sum()} ({(df_final['Title_Sentiment'] == 'Neutral').sum()/len(df_final)*100:.2f}%)")
    
    print("\nContent Sentiment Distribution:")
    print(df_final['Content_Sentiment'].value_counts())
    print(f"\n  Positive: {(df_final['Content_Sentiment'] == 'Positive').sum()} ({(df_final['Content_Sentiment'] == 'Positive').sum()/len(df_final)*100:.2f}%)")
    print(f"  Negative: {(df_final['Content_Sentiment'] == 'Negative').sum()} ({(df_final['Content_Sentiment'] == 'Negative').sum()/len(df_final)*100:.2f}%)")
    print(f"  Neutral:  {(df_final['Content_Sentiment'] == 'Neutral').sum()} ({(df_final['Content_Sentiment'] == 'Neutral').sum()/len(df_final)*100:.2f}%)")
    
    print("\nCombined Sentiment Distribution:")
    print(df_final['Combined_Sentiment'].value_counts())
    print(f"\n  Positive: {(df_final['Combined_Sentiment'] == 'Positive').sum()} ({(df_final['Combined_Sentiment'] == 'Positive').sum()/len(df_final)*100:.2f}%)")
    print(f"  Negative: {(df_final['Combined_Sentiment'] == 'Negative').sum()} ({(df_final['Combined_Sentiment'] == 'Negative').sum()/len(df_final)*100:.2f}%)")
    print(f"  Neutral:  {(df_final['Combined_Sentiment'] == 'Neutral').sum()} ({(df_final['Combined_Sentiment'] == 'Neutral').sum()/len(df_final)*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"\nAverage Confidence Scores:")
    print(f"  Title:   {df_final['Title_Confidence'].mean():.3f}")
    print(f"  Content: {df_final['Content_Confidence'].mean():.3f}")
    print(f"  Combined: {df_final['Combined_Confidence'].mean():.3f}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

