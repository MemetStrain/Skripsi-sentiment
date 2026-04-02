"""
Services.py - Business Logic Layer for CPO Price Prediction System
==================================================================
This file contains all data processing functions aligned with the Data Flow Diagram (DFD).
Each function represents a specific "Process" bubble in the DFD.

Pattern: Pure Procedural/Functional Programming
- No classes, only pure functions
- Each function has a single responsibility
- Functions are independent and testable
- All data storage uses Firestore (NoSQL), not Django ORM
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import csv
import io
import pandas as pd
import numpy as np
from django.core.exceptions import ValidationError


# ============================================================================
# DFD PROCESS 1: DATA INPUT & PARSING
# ============================================================================

def parse_indonesian_csv(file) -> List[Dict]:
    """
    Process 1.1: Parse CSV file from investing.com (Indonesian format)
    
    Input: CSV file with headers: Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol.
    Output: List of dictionaries with cleaned and mapped data
    
    Maps Indonesian headers to English:
    - Tanggal -> date
    - Terakhir -> close
    - Pembukaan -> open
    - Tertinggi -> high
    - Terendah -> low
    - Vol. -> volume
    """
    try:
        # Read file content
        file_content = file.read()
        
        # Try to decode with common encodings
        try:
            decoded_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                decoded_content = file_content.decode('latin-1')
            except UnicodeDecodeError:
                decoded_content = file_content.decode('cp1252')
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(decoded_content))
        
        # Map Indonesian headers to English
        header_mapping = {
            'Tanggal': 'date',
            'Terakhir': 'close',
            'Pembukaan': 'open',
            'Tertinggi': 'high',
            'Terendah': 'low',
            'Vol.': 'volume',
            # English alternatives (in case CSV is already in English)
            'Date': 'date',
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume',
        }
        
        parsed_data = []
        
        for row in csv_reader:
            try:
                # Map headers
                mapped_row = {}
                for key, value in row.items():
                    if key in header_mapping:
                        mapped_row[header_mapping[key]] = value
                
                # Clean and convert data
                cleaned_row = clean_price_row(mapped_row)
                if cleaned_row:
                    parsed_data.append(cleaned_row)
                    
            except Exception as e:
                # Skip problematic rows but continue processing
                print(f"Warning: Skipping row due to error: {e}")
                continue
        
        return parsed_data
    
    except Exception as e:
        raise ValidationError(f"Error parsing CSV file: {str(e)}")


def clean_price_row(row: Dict) -> Optional[Dict]:
    """
    Process 1.2: Clean and validate a single price data row
    
    Handles:
    - Date format conversion (DD/MM/YYYY or DD.MM.YYYY to YYYY-MM-DD)
    - Remove thousands separators (. or ,)
    - Convert strings to floats
    - Validate ranges
    """
    try:
        # Clean date
        date_str = row.get('date', '').strip()
        if not date_str:
            return None
        
        # Parse date (handle multiple formats)
        date_obj = parse_date_string(date_str)
        if not date_obj:
            return None
        
        # Clean numeric values (remove thousands separators)
        close = clean_numeric_value(row.get('close', '0'))
        open_price = clean_numeric_value(row.get('open', '0'))
        high = clean_numeric_value(row.get('high', '0'))
        low = clean_numeric_value(row.get('low', '0'))
        volume = clean_numeric_value(row.get('volume', '0'))
        
        # Validate ranges
        if close <= 0 or open_price <= 0 or high <= 0 or low <= 0:
            return None
        
        if low > high or low > close or low > open_price:
            return None
        
        if high < close or high < open_price:
            return None
        
        return {
            'date': date_obj,
            'close': close,
            'open': open_price,
            'high': high,
            'low': low,
            'volume': volume if volume > 0 else 0.0
        }
    
    except Exception as e:
        print(f"Error cleaning row: {e}")
        return None


def parse_date_string(date_str: str) -> Optional[datetime.date]:
    """
    Process 1.3: Parse date string in various formats
    
    Supports:
    - DD/MM/YYYY
    - DD.MM.YYYY
    - DD-MM-YYYY
    - YYYY-MM-DD
    """
    date_formats = [
        '%d/%m/%Y',  # 31/12/2023
        '%d.%m.%Y',  # 31.12.2023
        '%d-%m-%Y',  # 31-12-2023
        '%Y-%m-%d',  # 2023-12-31
        '%m/%d/%Y',  # 12/31/2023 (US format)
    ]
    
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format).date()
        except ValueError:
            continue
    
    return None


def clean_numeric_value(value_str: str) -> float:
    """
    Process 1.4: Clean and convert numeric string to float
    
    Handles:
    - Thousands separators: 1.234,56 or 1,234.56
    - Percentage signs
    - Currency symbols
    """
    if not value_str or value_str == '-':
        return 0.0
    
    # Remove whitespace
    value_str = str(value_str).strip()
    
    # Remove currency symbols
    value_str = value_str.replace('Rp', '').replace('$', '').replace('€', '')
    
    # Remove percentage signs
    value_str = value_str.replace('%', '')
    
    # Detect format: Indonesian (1.234,56) vs English (1,234.56)
    if ',' in value_str and '.' in value_str:
        # Has both separators
        if value_str.rfind(',') > value_str.rfind('.'):
            # Indonesian format: 1.234,56
            value_str = value_str.replace('.', '').replace(',', '.')
        else:
            # English format: 1,234.56
            value_str = value_str.replace(',', '')
    elif ',' in value_str:
        # Only comma - could be decimal or thousands
        if value_str.count(',') == 1 and len(value_str.split(',')[1]) <= 2:
            # Likely decimal: 1234,56
            value_str = value_str.replace(',', '.')
        else:
            # Likely thousands: 1,234,567
            value_str = value_str.replace(',', '')
    elif '.' in value_str:
        # Only dot - could be decimal or thousands
        if value_str.count('.') == 1 and len(value_str.split('.')[1]) > 2:
            # Likely thousands: 1.234
            value_str = value_str.replace('.', '')
        # Otherwise assume decimal: 1234.56
    
    try:
        return float(value_str)
    except ValueError:
        return 0.0


# ============================================================================
# DFD PROCESS 2: DATA STORAGE
# ============================================================================

def save_price_data_batch(parsed_data: List[Dict]) -> Tuple[int, int, List[str]]:
    """
    Process 2.1: Save batch of price data to Firestore
    
    Returns: (success_count, skip_count, error_messages)
    """
    from firebase_admin import firestore
    
    db = firestore.client()
    success_count = 0
    skip_count = 0
    error_messages = []
    
    for row in parsed_data:
        try:
            # Use date as document ID
            doc_id = row['date'].isoformat()
            doc_ref = db.collection('DailyMarketData').document(doc_id)
            
            # Check if document exists
            doc_snapshot = doc_ref.get()
            
            # Prepare data
            data = {
                'date': row['date'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'updated_at': datetime.now().isoformat()
            }
            
            if doc_snapshot.exists:
                # Update existing document
                doc_ref.update(data)
                skip_count += 1
            else:
                # Create new document
                data['created_at'] = datetime.now().isoformat()
                doc_ref.set(data)
                success_count += 1
        
        except Exception as e:
            error_messages.append(f"Error on {row['date']}: {str(e)}")
            continue
    
    return success_count, skip_count, error_messages


# ============================================================================
# DFD PROCESS 3: DATA RETRIEVAL
# ============================================================================

def fetch_price_data(days: int = 90, order_by: str = 'date') -> List[Dict]:
    """
    Process 3.1: Fetch price history data from Firestore
    
    Args:
        days: Number of days to fetch (default 90)
        order_by: Field to order by (default 'date')
    
    Returns: List of price data dictionaries
    """
    from firebase_admin import firestore
    
    db = firestore.client()
    cutoff_date = datetime.now().date() - timedelta(days=days)
    cutoff_date_str = cutoff_date.isoformat()
    
    # Fetch from Firestore
    docs = db.collection('DailyMarketData').where('date', '>=', cutoff_date_str).order_by('date').stream()
    
    price_list = []
    for doc in docs:
        data = doc.to_dict()
        price_list.append({
            'date': data.get('date'),
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'close': float(data.get('close', 0)),
            'volume': float(data.get('volume', 0))
        })
    
    return price_list


def fetch_market_states(days: int = 90) -> List[Dict]:
    """
    Process 3.2: Fetch market state predictions from Firestore
    
    Returns: List of market state dictionaries
    """
    from firebase_admin import firestore
    
    db = firestore.client()
    cutoff_date = datetime.now().date() - timedelta(days=days)
    cutoff_date_str = cutoff_date.isoformat()
    
    # Fetch from Firestore
    docs = db.collection('MarketStates').where('date', '>=', cutoff_date_str).order_by('date').stream()
    
    state_list = []
    for doc in docs:
        data = doc.to_dict()
        state_list.append({
            'date': data.get('date'),
            'state': int(data.get('state', 2)),
            'probability': float(data.get('probability', 0.5))
        })
    
    return state_list


def fetch_news_data(sentiment_filter: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Process 3.3: Fetch news data from Firestore with optional sentiment filter
    
    Args:
        sentiment_filter: 'Positive', 'Negative', 'Neutral', or None
        limit: Maximum number of records to return
    
    Returns: List of news dictionaries
    """
    from firebase_admin import firestore
    
    db = firestore.client()
    
    # Build query
    query = db.collection('NewsData').order_by('date', direction=firestore.Query.DESCENDING)
    
    if sentiment_filter and sentiment_filter in ['Positive', 'Negative', 'Neutral']:
        query = query.where('sentiment_label', '==', sentiment_filter)
    
    # Fetch documents
    docs = query.stream()
    
    news_list = []
    for doc in docs:
        data = doc.to_dict()
        news_list.append({
            'id': doc.id,
            'date': data.get('date'),
            'title': data.get('title', ''),
            'snippet': data.get('snippet', ''),
            'url': data.get('url', ''),
            'sentiment_score': float(data.get('sentiment_score', 0)),
            'sentiment_label': data.get('sentiment_label', 'Neutral')
        })
        
        # Apply limit
        if limit and len(news_list) >= limit:
            break
    
    return news_list


# ============================================================================
# DFD PROCESS 4: MACHINE LEARNING - HMM STATES (stored in Firestore)
# ============================================================================
# HMM states are pre-computed by markov/cpo_hmm_states.py and stored in
# the HmmStatesResults Firestore collection. The website reads them directly.
# See prediction_service.py for real ML predictions.


def save_market_states_batch(states_data: List[Dict]) -> int:
    """
    Process 4.3: Save calculated HMM states to Firestore
    
    Returns: Number of records created/updated
    """
    from firebase_admin import firestore
    
    db = firestore.client()
    batch = db.batch()
    count = 0
    
    for state_info in states_data:
        # Use date as document ID
        if isinstance(state_info['date'], str):
            doc_id = state_info['date']
        else:
            doc_id = state_info['date'].isoformat()
        
        doc_ref = db.collection('MarketStates').document(doc_id)
        
        data = {
            'date': doc_id,
            'state': int(state_info['state']),
            'probability': float(state_info['probability']),
            'updated_at': datetime.now().isoformat()
        }
        
        batch.set(doc_ref, data, merge=True)
        count += 1
    
    # Commit batch
    batch.commit()
    
    return count


# ============================================================================
# DFD PROCESS 5: PREDICTION GENERATION
# ============================================================================
# Real predictions are handled by prediction_service.py using horizon forecasting
# with trained XGBoost, Random Forest, ARIMAX, and SARIMAX models.

# ============================================================================
# DFD PROCESS 6: SENTIMENT ANALYSIS
# ============================================================================
# Sentiment analysis is pre-computed by news/finbert_sentiment_analysis_flexible.py
# and stored in the SentimentAggregate Firestore collection.


# ============================================================================
# DFD PROCESS 7: STATISTICS CALCULATION
# ============================================================================

def calculate_statistics(price_data: List[Dict]) -> Dict:
    """
    Process 7.1: Calculate statistical metrics from price data
    
    Args:
        price_data: List of dictionaries with price information
    
    Returns: Dictionary with various statistics
    """
    if not price_data:
        return {
            'current_price': 0,
            'avg_price': 0,
            'max_price': 0,
            'min_price': 0,
            'total_days': 0,
            'price_change': 0,
            'price_change_pct': 0,
            'volatility': 0
        }
    
    prices = [float(p.get('close', 0)) for p in price_data]
    
    current_price = prices[-1] if prices else 0
    avg_price = sum(prices) / len(prices) if prices else 0
    max_price = max(prices) if prices else 0
    min_price = min(prices) if prices else 0
    
    # Calculate price change
    if len(prices) >= 2:
        price_change = prices[-1] - prices[0]
        price_change_pct = (price_change / prices[0]) * 100 if prices[0] != 0 else 0
    else:
        price_change = 0
        price_change_pct = 0
    
    # Calculate volatility (standard deviation)
    if len(prices) > 1:
        volatility = np.std(prices)
    else:
        volatility = 0
    
    return {
        'current_price': round(current_price, 2),
        'avg_price': round(avg_price, 2),
        'max_price': round(max_price, 2),
        'min_price': round(min_price, 2),
        'total_days': len(price_data),
        'price_change': round(price_change, 2),
        'price_change_pct': round(price_change_pct, 2),
        'volatility': round(volatility, 2)
    }


def calculate_model_metrics(actual_prices: List[float], predicted_prices: List[float]) -> Dict:
    """
    Process 7.2: Calculate ML model performance metrics
    
    Metrics:
    - MAPE (Mean Absolute Percentage Error)
    - R² (R-squared / Coefficient of Determination)
    - Directional Accuracy
    
    Returns: Dictionary with metric values
    """
    if not actual_prices or not predicted_prices or len(actual_prices) != len(predicted_prices):
        return {
            'mape': 0.0,
            'r2': 0.0,
            'accuracy': 0.0
        }
    
    # Convert to numpy arrays
    actual = np.array(actual_prices)
    predicted = np.array(predicted_prices)
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate Directional Accuracy
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        accuracy = 0
    
    return {
        'mape': round(mape, 2),
        'r2': round(r2, 2),
        'accuracy': round(accuracy, 2)
    }


# ============================================================================
# DFD PROCESS 8: CHART DATA PREPARATION
# ============================================================================

def prepare_chart_data(price_data: List[Dict], market_states: List[Dict], 
                       predictions: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Process 8.1: Prepare data structure for Chart.js visualization
    
    Args:
        price_data: List of price dictionaries
        market_states: List of market state dictionaries
        predictions: Optional list of prediction dictionaries
    
    Combines:
    - Historical prices
    - Market states (HMM)
    - Predictions (optional)
    
    Returns: List of dicts ready for JSON serialization
    """
    # Create state lookup dictionary
    state_dict = {state.get('date'): state for state in market_states}
    
    # State labels mapping
    state_labels = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
    
    chart_data = []
    
    for price in price_data:
        price_date = price.get('date')
        
        # Get market state
        market_state = state_dict.get(price_date)
        state_value = int(market_state.get('state', 2)) if market_state else 2
        state_label = state_labels.get(state_value, 'Unknown')
        state_prob = float(market_state.get('probability', 0.5)) if market_state else 0.5
        
        actual_price = float(price.get('close', 0))

        chart_data.append({
            'date': price_date,
            'actual': actual_price,
            'open': float(price.get('open', 0)),
            'high': float(price.get('high', 0)),
            'low': float(price.get('low', 0)),
            'volume': float(price.get('volume', 0)),
            'state': state_value,
            'state_label': state_label,
            'state_probability': round(state_prob, 2)
        })
    
    return chart_data


# ============================================================================
# DFD PROCESS 9: NEWS AGGREGATION
# ============================================================================

def get_sentiment_counts() -> Dict:
    """
    Process 9.1: Count news by sentiment category from Firestore
    
    Returns: Dictionary with counts by sentiment
    """
    from firebase_admin import firestore
    
    db = firestore.client()
    
    # Fetch all news documents
    docs = db.collection('NewsData').stream()
    
    # Count by sentiment
    counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'total': 0
    }
    
    for doc in docs:
        data = doc.to_dict()
        label = data.get('sentiment_label', 'Neutral')
        counts['total'] += 1
        
        if label == 'Positive':
            counts['positive'] += 1
        elif label == 'Negative':
            counts['negative'] += 1
        elif label == 'Neutral':
            counts['neutral'] += 1
    
    return counts
