# FinBERT Sentiment Analysis for MPOB News

This script analyzes sentiment for news articles in `mpob_news_fast.csv` using FinBERT, a financial domain-specific BERT model.

## Features

- **Title Sentiment Analysis**: Analyzes sentiment of article titles
- **Content Sentiment Analysis**: Analyzes sentiment of article content
- **Combined Sentiment**: Weighted combination (30% title, 70% content)
- **Confidence Scores**: Probability scores for each sentiment class
- **Batch Processing**: Processes articles in batches for efficiency
- **Resume Capability**: Can resume processing if interrupted (only processes missing rows)

## Requirements

Make sure you have the required packages installed:

```bash
pip install pandas numpy torch transformers tqdm
```

## Usage

1. Place your CSV file (`mpob_news_fast.csv`) in the same directory as the script.

2. Run the script:
   ```bash
   python finbert_sentiment_analysis.py
   ```

3. The script will:
   - Load the FinBERT model (first run will download ~500MB)
   - Process all articles in the CSV
   - Add sentiment columns to the dataframe
   - Save results to `mpob_news_with_sentiment.csv`

## Output Columns

The output CSV will include the following new columns:

### Title Sentiment
- `Title_Sentiment`: Sentiment label (Positive/Negative/Neutral)
- `Title_Confidence`: Confidence score (0-1)
- `Title_Positive_Prob`: Probability of Positive sentiment
- `Title_Negative_Prob`: Probability of Negative sentiment
- `Title_Neutral_Prob`: Probability of Neutral sentiment

### Content Sentiment
- `Content_Sentiment`: Sentiment label (Positive/Negative/Neutral)
- `Content_Confidence`: Confidence score (0-1)
- `Content_Positive_Prob`: Probability of Positive sentiment
- `Content_Negative_Prob`: Probability of Negative sentiment
- `Content_Neutral_Prob`: Probability of Neutral sentiment

### Combined Sentiment
- `Combined_Sentiment`: Weighted sentiment label
- `Combined_Confidence`: Weighted confidence score
- `Combined_Positive_Prob`: Weighted Positive probability
- `Combined_Negative_Prob`: Weighted Negative probability
- `Combined_Neutral_Prob`: Weighted Neutral probability

## Configuration

You can modify these variables at the top of the script:

- `INPUT_CSV`: Input CSV filename (default: 'mpob_news_fast.csv')
- `OUTPUT_CSV`: Output CSV filename (default: 'mpob_news_with_sentiment.csv')
- `BATCH_SIZE`: Number of articles to process at once (default: 8, reduce if you run out of memory)
- `MAX_LENGTH`: Maximum token length (default: 512, FinBERT's max)

## Performance

- **First Run**: Downloads FinBERT model (~500MB) - takes a few minutes
- **Processing Speed**: 
  - CPU: ~2-5 seconds per article
  - GPU: ~0.1-0.5 seconds per article
- **Memory**: Requires ~2-4GB RAM, more with GPU

## Notes

- The script automatically uses GPU if available (CUDA), otherwise falls back to CPU
- Articles with error messages in content are automatically set to Neutral
- Empty or missing text fields are handled gracefully
- Processing can be resumed if interrupted (re-run the script)

## Troubleshooting

**Out of Memory Error**: Reduce `BATCH_SIZE` (e.g., change to 4 or 2)

**Model Download Issues**: The first run downloads the model from Hugging Face. Ensure you have internet connection and enough disk space (~500MB).

**Processing Too Slow**: 
- Use GPU if available (automatically detected)
- Reduce `BATCH_SIZE` if using CPU
- Consider processing a subset of data first

