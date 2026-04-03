"""
price_fetcher.py — Fetch CPO price data and write to Firestore `daily_prices`.

Initial load: reads Data_CPO_Daily.csv (Indonesian format) → writes all rows.
Daily update: fetches the latest trading day from Investing.com via investiny.
"""

import csv
import io
import logging
import os
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Investing.com asset ID for CPO (Palm Oil Futures)
CPO_INVESTING_ID = 49764  # FCPO continuous contract


# ---------------------------------------------------------------------------
# Date / number parsing helpers (Indonesian CSV format)
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> Optional[date]:
    for fmt in ('%d/%m/%Y', '%d.%m.%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_number(value_str: str) -> float:
    """Parse Indonesian number format: 4.720,00 → 4720.0"""
    if not value_str or value_str.strip() in ('-', ''):
        return 0.0
    s = str(value_str).strip()
    s = s.replace('Rp', '').replace('$', '').replace('%', '').strip()

    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            # Indonesian: 1.234,56
            s = s.replace('.', '').replace(',', '.')
        else:
            # English: 1,234.56
            s = s.replace(',', '')
    elif ',' in s:
        # Could be decimal (1234,56) or thousands (1,234)
        if s.count(',') == 1 and len(s.split(',')[1]) <= 2:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif '.' in s:
        if s.count('.') == 1 and len(s.split('.')[1]) > 2:
            s = s.replace('.', '')

    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Initial load: read local CSV
# ---------------------------------------------------------------------------

def load_prices_from_csv(csv_path: str) -> list[dict]:
    """
    Read the Indonesian-format CPO CSV file and return a list of price dicts.
    Expected columns (Indonesian): Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol.
    """
    if not os.path.exists(csv_path):
        logger.warning(f'CSV file not found: {csv_path}')
        return []

    header_map = {
        'Tanggal': 'date', 'Terakhir': 'close', 'Pembukaan': 'open',
        'Tertinggi': 'high', 'Terendah': 'low', 'Vol.': 'volume',
        # English fallbacks
        'Date': 'date', 'Close': 'close', 'Open': 'open',
        'High': 'high', 'Low': 'low', 'Volume': 'volume',
    }

    rows = []
    try:
        with open(csv_path, encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                mapped = {header_map[k]: v for k, v in raw.items() if k in header_map}
                d = _parse_date(mapped.get('date', ''))
                if d is None:
                    continue
                close = _parse_number(mapped.get('close', '0'))
                open_ = _parse_number(mapped.get('open', '0'))
                high = _parse_number(mapped.get('high', '0'))
                low = _parse_number(mapped.get('low', '0'))
                vol = _parse_number(mapped.get('volume', '0'))
                if close <= 0:
                    continue
                rows.append({
                    'date': d.isoformat(),
                    'open': open_ or close,
                    'high': high or close,
                    'low': low or close,
                    'close': close,
                    'volume': vol,
                })
    except Exception as e:
        logger.error(f'Error reading CSV {csv_path}: {e}')

    logger.info(f'Loaded {len(rows)} rows from {csv_path}')
    return rows


# ---------------------------------------------------------------------------
# Daily update: fetch latest price from Investing.com
# ---------------------------------------------------------------------------

def fetch_latest_price() -> Optional[dict]:
    """
    Fetch the most recent CPO price from Investing.com using investiny.
    Returns a single price dict or None on failure.
    """
    try:
        from investiny import historical_data

        # Fetch the last 5 trading days to ensure we get the latest
        end = datetime.now()
        start = end - timedelta(days=7)

        data = historical_data(
            investing_id=CPO_INVESTING_ID,
            from_date=start.strftime('%m/%d/%Y'),
            to_date=end.strftime('%m/%d/%Y'),
        )

        if not data or not data.get('t'):
            logger.warning('investiny returned no data')
            return None

        timestamps = data['t']
        closes = data['c']
        opens = data.get('o', closes)
        highs = data.get('h', closes)
        lows = data.get('l', closes)
        volumes = data.get('v', [0] * len(timestamps))

        # Take the last (most recent) row
        idx = -1
        trade_date = date.fromtimestamp(timestamps[idx]).isoformat()
        return {
            'date': trade_date,
            'open': float(opens[idx]),
            'high': float(highs[idx]),
            'low': float(lows[idx]),
            'close': float(closes[idx]),
            'volume': float(volumes[idx]) if volumes[idx] else 0.0,
            'change_pct': None,
        }

    except Exception as e:
        logger.error(f'Failed to fetch latest price: {e}')
        return None


def is_price_stored(db, trade_date: str) -> bool:
    """Check if a price document already exists in `daily_prices`."""
    doc = db.collection('daily_prices').document(trade_date).get()
    return doc.exists
