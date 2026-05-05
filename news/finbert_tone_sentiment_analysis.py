import sys
import warnings

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings('ignore')

# Ensure NLTK punkt_tab tokenizer is downloaded (no-op if already cached locally).
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Configuration
INPUT_CSV = 'mpob_news_preprocessed.csv'
OUTPUT_CSV = 'mpob_news_with_sentiment_tone.csv'
DAILY_AGGREGATE_CSV = 'output/sentiment_aggregate_Daily.csv'
BATCH_SIZE = 16  # Articles per batch (sentences within are flattened for inference)
MAX_LENGTH = 512  # Maximum token length for FinBERT (per sentence)
SENTENCE_CONFIDENCE_THRESHOLD = 0.5  # Drop sentences whose max-class probability is below this
TITLE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.7
USE_HALF_PRECISION = True  # Use float16 for faster GPU inference (recommended for CUDA)
FORCE_CPU = False  # Set to True to force CPU even if CUDA is available

# CUDA Configuration
def get_device():
    """Get the best available device with CUDA information."""
    if FORCE_CPU:
        print("CPU mode forced by configuration.")
        return 'cpu', None

    if torch.cuda.is_available():
        device_id = 0
        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

        print(f"\n{'='*70}")
        print("CUDA DETECTED - GPU MODE ENABLED")
        print(f"{'='*70}")
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"{'='*70}\n")

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

# Auto-adjust batch size based on GPU memory.
if DEVICE != 'cpu' and GPU_MEMORY:
    if GPU_MEMORY < 4:
        BATCH_SIZE = min(BATCH_SIZE, 8)
    elif GPU_MEMORY < 8:
        BATCH_SIZE = min(BATCH_SIZE, 16)
    else:
        BATCH_SIZE = min(BATCH_SIZE, 32)

# yiyanghkust/finbert-tone label order is {0: Neutral, 1: Positive, 2: Negative} —
# DIFFERENT from ProsusAI/finbert. Hardcoded here to match the model's id2label.
LABEL_BY_INDEX = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
LABELS = ['Neutral', 'Positive', 'Negative']  # index-aligned with model output


def load_finbert_model():
    """Load FinBERT-Tone model and tokenizer with CUDA optimization."""
    print(f"\nLoading FinBERT-Tone model (yiyanghkust/finbert-tone)...")
    print(f"Device: {DEVICE}")

    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

    model = model.to(DEVICE)

    if USE_HALF_PRECISION and DEVICE != 'cpu':
        try:
            model = model.half()
            print("Using half precision (float16) for faster GPU inference")
        except Exception as e:
            print(f"Warning: Could not enable half precision: {e}")
            print("Falling back to full precision (float32)")

    model.eval()

    if DEVICE != 'cpu':
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

    print("FinBERT-Tone model loaded successfully!")

    if DEVICE != 'cpu':
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU Memory allocated: {allocated:.2f} GB")

    return model, tokenizer


def _split_into_sentences(content):
    """Split article content into a list of trimmed, non-empty sentence strings."""
    if content is None or (isinstance(content, float) and pd.isna(content)):
        return []
    text = str(content).strip()
    if not text or '[Error:' in text:
        return []
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception as exc:
        print(f"WARN: sentence tokenization failed; treating whole content as one sentence: {exc}")
        return [text]


def _score_texts(texts, model, tokenizer):
    """Run FinBERT-Tone on a flat list of texts.

    Returns Nx3 ndarray with columns aligned to LABELS = [Neutral, Positive, Negative].
    """
    if not texts:
        return np.zeros((0, 3), dtype=np.float32)
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.dtype == torch.float16:
            logits = logits.float()
        probs = torch.softmax(logits, dim=1)
        if DEVICE != 'cpu':
            probs = probs.cpu().numpy()
        else:
            probs = probs.numpy()
    return probs


def _aggregate_content(probs_arr):
    """Mean probabilities across content sentences with the confidence filter.

    Args:
        probs_arr: Nx3 ndarray, columns [Neutral, Positive, Negative].

    Returns:
        Tuple (mean_neu, mean_pos, mean_neg, n_total, n_used, fallback_triggered).
        For empty input, returns degenerate neutral (1, 0, 0, 0, 0, False).
    """
    n_total = int(probs_arr.shape[0])
    if n_total == 0:
        return (1.0, 0.0, 0.0, 0, 0, False)

    max_per_row = probs_arr.max(axis=1)
    keep_mask = max_per_row >= SENTENCE_CONFIDENCE_THRESHOLD
    n_kept = int(keep_mask.sum())

    fallback = False
    if n_kept == 0:
        # Fallback: keep all sentences so the article still gets a label.
        kept = probs_arr
        n_used = n_total
        fallback = True
    else:
        kept = probs_arr[keep_mask]
        n_used = n_kept

    mean = kept.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]), n_total, n_used, fallback)


def _assemble_result(probs_arr, title_idx, content_start, content_end):
    """Build a per-article result dict from a slice of the flat probabilities array."""
    if title_idx is None:
        title_label = 'Neutral'
        title_conf = 0.0
        title_neu, title_pos, title_neg = 1.0, 0.0, 0.0
        title_passed = False
    else:
        tp = probs_arr[title_idx]
        title_neu, title_pos, title_neg = float(tp[0]), float(tp[1]), float(tp[2])
        idx = int(np.argmax(tp))
        title_label = LABELS[idx]
        title_conf = float(tp[idx])
        title_passed = title_conf >= SENTENCE_CONFIDENCE_THRESHOLD

    content_slice = probs_arr[content_start:content_end]
    content_neu, content_pos, content_neg, n_total, n_used, fallback = _aggregate_content(content_slice)

    if n_total == 0:
        content_label = 'Neutral'
        content_conf = 0.0
    else:
        content_probs = (content_neu, content_pos, content_neg)
        idx = int(np.argmax(content_probs))
        content_label = LABELS[idx]
        content_conf = float(content_probs[idx])

    # Combined: title_weight * title + content_weight * content when title passes filter,
    # otherwise content-only (or title-only if content is missing entirely).
    if title_passed and n_total > 0:
        combined_neu = TITLE_WEIGHT * title_neu + CONTENT_WEIGHT * content_neu
        combined_pos = TITLE_WEIGHT * title_pos + CONTENT_WEIGHT * content_pos
        combined_neg = TITLE_WEIGHT * title_neg + CONTENT_WEIGHT * content_neg
    elif n_total > 0:
        combined_neu, combined_pos, combined_neg = content_neu, content_pos, content_neg
    elif title_idx is not None:
        combined_neu, combined_pos, combined_neg = title_neu, title_pos, title_neg
    else:
        combined_neu, combined_pos, combined_neg = 1.0, 0.0, 0.0

    combined_probs = (combined_neu, combined_pos, combined_neg)
    idx = int(np.argmax(combined_probs))
    combined_label = LABELS[idx]
    combined_conf = float(combined_probs[idx])

    return {
        'Title_Sentiment': title_label,
        'Title_Confidence': title_conf,
        'Title_Positive_Prob': title_pos,
        'Title_Negative_Prob': title_neg,
        'Title_Neutral_Prob': title_neu,
        'Content_Sentiment': content_label,
        'Content_Confidence': content_conf,
        'Content_Positive_Prob': content_pos,
        'Content_Negative_Prob': content_neg,
        'Content_Neutral_Prob': content_neu,
        'Combined_Sentiment': combined_label,
        'Combined_Confidence': combined_conf,
        'Combined_Positive_Prob': combined_pos,
        'Combined_Negative_Prob': combined_neg,
        'Combined_Neutral_Prob': combined_neu,
        'Content_Sentence_Count': n_total,
        'Content_Sentence_Used': n_used,
        'Content_Filter_Fallback': fallback,
    }


def process_batch(df_batch, model, tokenizer):
    """Score a batch of articles using cross-article flat-list sentence batching."""
    flat_texts = []
    article_indices = []  # (title_flat_idx_or_none, content_start, content_end) per row

    for _, row in df_batch.iterrows():
        title = str(row.get('Title', '') or '').strip()
        title_idx = None
        if title:
            title_idx = len(flat_texts)
            flat_texts.append(title)

        sentences = _split_into_sentences(row.get('Content', ''))
        content_start = len(flat_texts)
        flat_texts.extend(sentences)
        content_end = len(flat_texts)
        article_indices.append((title_idx, content_start, content_end))

    try:
        probs_arr = _score_texts(flat_texts, model, tokenizer)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and DEVICE != 'cpu':
            print(f"\nCUDA OOM on batch ({len(df_batch)} articles, {len(flat_texts)} sentences). "
                  "Clearing cache and falling back to per-article inference.")
            torch.cuda.empty_cache()
            return _process_batch_per_article(df_batch, model, tokenizer)
        raise
    except Exception as e:
        print(f"\nBatch inference failed: {e}. Falling back to per-article inference.")
        return _process_batch_per_article(df_batch, model, tokenizer)

    results = [_assemble_result(probs_arr, t, cs, ce) for (t, cs, ce) in article_indices]

    if DEVICE != 'cpu':
        torch.cuda.empty_cache()
    return results


def _process_batch_per_article(df_batch, model, tokenizer):
    """Per-article fallback when cross-article batched inference fails."""
    results = []
    for idx in df_batch.index:
        try:
            single = process_batch(df_batch.loc[[idx]], model, tokenizer)
            results.append(single[0])
        except Exception as e:
            print(f"\nError processing article {idx}: {e}")
            results.append(get_default_sentiment_results())
    return results


def get_default_sentiment_results():
    """Return default Neutral sentiment results (used as the failure fallback)."""
    return {
        'Title_Sentiment': 'Neutral', 'Title_Confidence': 0.0,
        'Title_Positive_Prob': 0.0, 'Title_Negative_Prob': 0.0, 'Title_Neutral_Prob': 1.0,
        'Content_Sentiment': 'Neutral', 'Content_Confidence': 0.0,
        'Content_Positive_Prob': 0.0, 'Content_Negative_Prob': 0.0, 'Content_Neutral_Prob': 1.0,
        'Combined_Sentiment': 'Neutral', 'Combined_Confidence': 0.0,
        'Combined_Positive_Prob': 0.0, 'Combined_Negative_Prob': 0.0, 'Combined_Neutral_Prob': 1.0,
        'Content_Sentence_Count': 0, 'Content_Sentence_Used': 0, 'Content_Filter_Fallback': False,
    }


def aggregate_daily_sentiment(df):
    """
    Aggregate sentiment scores by working day (Mon-Fri).
    Days without news are forward-filled from the previous working day so the
    output schema matches the existing ``output/sentiment_aggregate_Daily.csv``.
    """
    print("\n" + "=" * 70)
    print("AGGREGATING DAILY SENTIMENT SCORES")
    print("=" * 70)

    # Date may already be parsed by aggregate_monthly_sentiment; re-parse defensively.
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    df_valid = df[df['Date'].notna()].copy().sort_values('Date').reset_index(drop=True)
    if len(df_valid) == 0:
        print("Warning: No valid dates found. Cannot aggregate daily.")
        return None

    min_date = df_valid['Date'].min()
    max_date = df_valid['Date'].max()
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    all_dates = pd.date_range(start=min_date, end=max_date, freq='B')  # business days
    print(f"Total working days in range: {len(all_dates)}")

    daily_agg = df_valid.groupby('Date').agg({
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
        'Title': 'count',
    }).rename(columns={'Title': 'Article_Count'})

    complete_df = pd.DataFrame({'Date': all_dates}).merge(daily_agg, on='Date', how='left')

    # Forward-fill probability/confidence columns; article count defaults to 0.
    fill_cols = [c for c in complete_df.columns if c not in ('Date', 'Article_Count')]
    complete_df[fill_cols] = complete_df[fill_cols].ffill()
    complete_df['Article_Count'] = complete_df['Article_Count'].fillna(0).astype(int)

    complete_df['Sentiment_Score'] = (
        complete_df['Combined_Positive_Prob'] - complete_df['Combined_Negative_Prob']
    )

    def _dominant(row):
        probs = {
            'Positive': row['Combined_Positive_Prob'],
            'Negative': row['Combined_Negative_Prob'],
            'Neutral': row['Combined_Neutral_Prob'],
        }
        return max(probs, key=probs.get)

    complete_df['Dominant_Sentiment'] = complete_df.apply(_dominant, axis=1)

    complete_df['Year'] = complete_df['Date'].dt.year
    complete_df['Month'] = complete_df['Date'].dt.month
    complete_df['Day'] = complete_df['Date'].dt.day
    complete_df['Weekday'] = complete_df['Date'].dt.day_name()
    complete_df['Date_Str'] = complete_df['Date'].dt.strftime('%Y-%m-%d')

    cols = ['Date', 'Date_Str', 'Year', 'Month', 'Day', 'Weekday', 'Article_Count']
    other_cols = [c for c in complete_df.columns if c not in cols]
    complete_df = complete_df[cols + other_cols]

    numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ('Year', 'Month', 'Day', 'Article_Count')]
    complete_df[numeric_cols] = complete_df[numeric_cols].round(4)

    print(f"Created Daily aggregation for {len(complete_df)} working days")
    print(f"Days with news: {(complete_df['Article_Count'] > 0).sum()}")

    return complete_df


def _save_aggregations(df_final):
    """Run the daily aggregation and write the CSV."""
    daily_df = aggregate_daily_sentiment(df_final)
    if daily_df is not None and len(daily_df) > 0:
        print(f"\nSaving daily aggregation to {DAILY_AGGREGATE_CSV}...")
        daily_df.to_csv(DAILY_AGGREGATE_CSV, index=False)
        print(f"Saved daily sentiment aggregation for {len(daily_df)} working days!")

        print("\n" + "=" * 70)
        print("DAILY SENTIMENT AGGREGATION SAMPLE (Last 5 days)")
        print("=" * 70)
        display_cols = ['Date_Str', 'Article_Count', 'Combined_Positive_Prob',
                        'Combined_Negative_Prob', 'Combined_Neutral_Prob', 'Sentiment_Score',
                        'Dominant_Sentiment']
        print(daily_df[display_cols].tail(5).to_string(index=False))


def main():
    print("=" * 70)
    print("FinBERT-Tone Sentiment Analysis for MPOB News (sentence-level)")
    print("=" * 70)

    print(f"\nLoading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} articles")
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found!")
        return

    sentiment_cols = ['Title_Sentiment', 'Content_Sentiment', 'Combined_Sentiment']
    if all(col in df.columns for col in sentiment_cols):
        print("\nSentiment columns already exist. Processing only missing rows...")
        mask = df['Combined_Sentiment'].isna()
        df_to_process = df[mask].copy()
        df_processed = df[~mask].copy()

        if len(df_to_process) == 0:
            print("All articles already have sentiment scores!")
            df_final = df.copy()
            _save_aggregations(df_final)
            return
        print(f"Processing {len(df_to_process)} remaining articles...")
    else:
        df_to_process = df.copy()
        df_processed = pd.DataFrame()

    model, tokenizer = load_finbert_model()

    print(f"\nProcessing articles in batches of {BATCH_SIZE} (sentences flattened across batch)...")
    if DEVICE != 'cpu':
        print(f"Using GPU acceleration with batch size: {BATCH_SIZE}")
    all_results = []

    for i in tqdm(range(0, len(df_to_process), BATCH_SIZE), desc="Processing batches"):
        batch = df_to_process.iloc[i:i + BATCH_SIZE]
        batch_results = process_batch(batch, model, tokenizer)
        all_results.extend(batch_results)

        if DEVICE != 'cpu' and (i // BATCH_SIZE + 1) % 5 == 0:
            torch.cuda.empty_cache()

    if DEVICE != 'cpu':
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'max_memory_allocated'):
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"\nPeak GPU memory usage: {peak_memory:.2f} GB")

    sentiment_df = pd.DataFrame(all_results)
    df_to_process = pd.concat([df_to_process.reset_index(drop=True), sentiment_df], axis=1)

    if len(df_processed) > 0:
        df_final = pd.concat([df_processed, df_to_process], ignore_index=True)
    else:
        df_final = df_to_process

    print(f"\nSaving results to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_final)} articles with sentiment scores!")

    _save_aggregations(df_final)

    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 70)

    for column in ['Title_Sentiment', 'Content_Sentiment', 'Combined_Sentiment']:
        print(f"\n{column} Distribution:")
        print(df_final[column].value_counts())
        for label in LABELS:
            count = (df_final[column] == label).sum()
            pct = count / len(df_final) * 100
            print(f"  {label:<8}: {count} ({pct:.2f}%)")

    print("\n" + "=" * 70)
    print("\nAverage Confidence Scores:")
    print(f"  Title:    {df_final['Title_Confidence'].mean():.3f}")
    print(f"  Content:  {df_final['Content_Confidence'].mean():.3f}")
    print(f"  Combined: {df_final['Combined_Confidence'].mean():.3f}")

    fallback_count = int(df_final['Content_Filter_Fallback'].sum())
    print(f"\nContent confidence-filter fallback fired on {fallback_count} of {len(df_final)} articles "
          f"({fallback_count / len(df_final) * 100:.2f}%).")
    print(f"Avg sentences per article (total / kept): "
          f"{df_final['Content_Sentence_Count'].mean():.2f} / {df_final['Content_Sentence_Used'].mean():.2f}")
    print("\nAnalysis complete!")


def smoke_test():
    """Print-based sanity check on three short hardcoded articles."""
    test_articles = [
        {
            'Title': 'CPO prices surge to record high on strong demand',
            'Content': (
                'Crude palm oil futures rallied 5 percent today on the back of strong export demand. '
                'Analysts upgraded their year-end target after the rally. '
                'Buyers continue to chase supply.'
            ),
        },
        {
            'Title': 'CPO prices plunge as inventories hit multi-year high',
            'Content': (
                'Palm oil futures collapsed 4 percent as Malaysian stockpiles surged to a five-year peak. '
                'Traders warned of further downside this quarter. '
                'Plantation stocks fell sharply.'
            ),
        },
        {
            'Title': 'MPOB releases monthly palm oil bulletin',
            'Content': (
                'The Malaysian Palm Oil Board issued its monthly statistical bulletin on Friday. '
                'The report covers production, exports and inventory data. '
                'The next bulletin is scheduled for May.'
            ),
        },
    ]

    print("=" * 70)
    print("SMOKE TEST: sentence-level FinBERT-Tone pipeline")
    print("=" * 70)

    model, tokenizer = load_finbert_model()
    df_test = pd.DataFrame(test_articles)
    results = process_batch(df_test, model, tokenizer)

    for i, (article, res) in enumerate(zip(test_articles, results)):
        print(f"\n--- Article {i + 1}: {article['Title'][:60]} ---")
        print(f"  Content sentences (total)  : {res['Content_Sentence_Count']}")
        print(f"  Content sentences (kept)   : {res['Content_Sentence_Used']}")
        print(f"  Filter fallback triggered  : {res['Content_Filter_Fallback']}")
        print(
            f"  Title    -> {res['Title_Sentiment']:<8} conf={res['Title_Confidence']:.3f}  "
            f"P/N/Ne = {res['Title_Positive_Prob']:.3f}/{res['Title_Negative_Prob']:.3f}/{res['Title_Neutral_Prob']:.3f}"
        )
        print(
            f"  Content  -> {res['Content_Sentiment']:<8} conf={res['Content_Confidence']:.3f}  "
            f"P/N/Ne = {res['Content_Positive_Prob']:.3f}/{res['Content_Negative_Prob']:.3f}/{res['Content_Neutral_Prob']:.3f}"
        )
        print(
            f"  Combined -> {res['Combined_Sentiment']:<8} conf={res['Combined_Confidence']:.3f}  "
            f"P/N/Ne = {res['Combined_Positive_Prob']:.3f}/{res['Combined_Negative_Prob']:.3f}/{res['Combined_Neutral_Prob']:.3f}"
        )


if __name__ == "__main__":
    if '--smoke-test' in sys.argv:
        smoke_test()
    else:
        main()
