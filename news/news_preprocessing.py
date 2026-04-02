"""
News Text Preprocessing Module
Cleans and prepares news articles for sentiment analysis
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_CSV = 'mpob_news_fast.csv'
OUTPUT_CSV = 'mpob_news_preprocessed.csv'

class NewsPreprocessor:
    """Preprocess news text for sentiment analysis"""
    
    def __init__(self):
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s\.\,\!\?\-\'\"]')
        
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_html_tags(self, text):
        """Remove HTML tags"""
        return re.sub(r'<[^>]+>', '', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra spaces and normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()
    
    def remove_special_characters(self, text, keep_punctuation=True):
        """Remove special characters but keep basic punctuation"""
        if keep_punctuation:
            # Keep letters, numbers, common punctuation, and spaces
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\'\"\&\%\(\)]', '', text)
        else:
            # Keep only letters, numbers, and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def normalize_unicode(self, text):
        """Normalize unicode characters"""
        text = unicodedata.normalize('NFKD', text)
        return text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def remove_date_patterns(self, text):
        """Remove common date patterns"""
        # Remove dates like 26/11/2025 or 26-11-2025
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', text)
        # Remove dates like (Jakarta, 26/11/2025)
        text = re.sub(r'\(\w+,\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\)', '', text)
        return text
    
    def remove_source_attributions(self, text):
        """Remove source attributions like (Jakarta Globe)"""
        text = re.sub(r'\([^)]*\)', '', text)
        return text
    
    def remove_repeated_words(self, text):
        """Remove repeated words (e.g., 'the the' -> 'the')"""
        text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
        return text
    
    def normalize_numbers(self, text):
        """Normalize numbers and percentages"""
        # Keep percentages but normalize
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        # Normalize large numbers (e.g., 1,000 -> 1000)
        text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)
        return text
    
    def preprocess_text(self, text, remove_stopwords=False):
        """
        Main preprocessing pipeline
        
        Args:
            text: Raw text to preprocess
            remove_stopwords: Whether to remove common stopwords (not recommended for sentiment analysis)
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Remove URLs
        text = self.remove_urls(text)
        
        # Step 3: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 4: Remove date patterns
        text = self.remove_date_patterns(text)
        
        # Step 5: Remove source attributions
        text = self.remove_source_attributions(text)
        
        # Step 6: Normalize unicode
        text = self.normalize_unicode(text)
        
        # Step 7: Remove special characters (keep basic punctuation)
        text = self.remove_special_characters(text, keep_punctuation=True)
        
        # Step 8: Normalize numbers
        text = self.normalize_numbers(text)
        
        # Step 9: Remove repeated words
        text = self.remove_repeated_words(text)
        
        # Step 10: Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        return text
    
    def preprocess_title(self, title):
        """Preprocess title with minimal changes to preserve meaning"""
        if not isinstance(title, str) or pd.isna(title):
            return ""
        
        title = title.lower()
        title = self.remove_urls(title)
        title = self.remove_html_tags(title)
        title = self.normalize_unicode(title)
        title = self.remove_special_characters(title, keep_punctuation=True)
        title = self.remove_extra_whitespace(title)
        
        return title
    
    def get_text_statistics(self, text):
        """Get statistics about the text"""
        if not text:
            return {'words': 0, 'chars': 0, 'sentences': 0}
        
        words = len(text.split())
        chars = len(text)
        sentences = len(re.split(r'[.!?]+', text))
        
        return {
            'words': words,
            'chars': chars,
            'sentences': sentences
        }


def main():
    """Main preprocessing pipeline"""
    
    print("=" * 70)
    print("NEWS TEXT PREPROCESSING")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} articles")
    
    # Initialize preprocessor
    preprocessor = NewsPreprocessor()
    
    print("\nPreprocessing news articles...")
    
    # Preprocess content
    print("  - Processing content...")
    df['Content'] = df['Content'].apply(
        lambda x: preprocessor.preprocess_text(x),
        1
    )
    
    # Preprocess title
    print("  - Processing titles...")
    df['Title'] = df['Title'].apply(
        lambda x: preprocessor.preprocess_title(x),
        1
    )
    
    # Remove empty rows after preprocessing
    initial_count = len(df)
    df = df[(df['Content'].str.len() > 0) | (df['Title'].str.len() > 0)].reset_index(drop=True)
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"  - Removed {removed_count} empty articles")
    
    # Handle missing snippets
    print("  - Handling missing values...")
    df['Snippet'] = df['Snippet'].fillna(df['Content'].str[:200])
    
    # Keep only processed columns
    columns_order = ['Date', 'Title', 'Category', 'Content', 'Snippet', 'URL']
    df = df[columns_order]
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} preprocessed articles")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("PREPROCESSING STATISTICS")
    print("=" * 70)
    print(f"\nTotal articles processed: {len(df)}")
    print(f"Articles removed (empty): {removed_count}")
    print(f"\nContent statistics:")
    content_words = df['Content'].apply(lambda x: len(x.split()) if x else 0)
    print(f"  - Average words per article: {content_words.mean():.1f}")
    print(f"  - Min words: {content_words.min()}")
    print(f"  - Max words: {content_words.max()}")
    print(f"  - Median words: {content_words.median():.0f}")
    
    print(f"\nTitle statistics:")
    title_words = df['Title'].apply(lambda x: len(x.split()) if x else 0)
    print(f"  - Average words per title: {title_words.mean():.1f}")
    print(f"  - Min words: {title_words.min()}")
    print(f"  - Max words: {title_words.max()}")
    
    print(f"\nSample preprocessed article:")
    print("-" * 70)
    sample_idx = 0
    print(f"Title: {df['Title'].iloc[sample_idx][:100]}...")
    print(f"\nContent (first 200 chars): {df['Content'].iloc[sample_idx][:200]}...")
    print("-" * 70)
    
    print("\n✓ Preprocessing complete!")
    print(f"Output file: {OUTPUT_CSV}")
    
    return df


if __name__ == "__main__":
    df = main()
