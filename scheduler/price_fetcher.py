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

import pandas as pd

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

    investiny.historical_data returns a dict keyed by column names
    (date, open, high, low, close, volume) — each value is a list aligned by row.
    """
    try:
        from investiny import historical_data

        # Investing.com requires MM/DD/YYYY. A narrow window (e.g. 7 days)
        # can hit holidays/weekends and return no_data, so we widen on retry.
        end = datetime.now()
        data = None
        last_err: Optional[Exception] = None
        for lookback_days in (90, 365, 365 * 3):
            start = end - timedelta(days=lookback_days)
            try:
                data = historical_data(
                    investing_id=CPO_INVESTING_ID,
                    from_date=start.strftime('%m/%d/%Y'),
                    to_date=end.strftime('%m/%d/%Y'),
                    interval='D',
                )
                if data:
                    break
            except Exception as e:
                last_err = e
                logger.warning(f'investiny call failed (lookback={lookback_days}d): {e}')
                continue

        if not data:
            if last_err:
                logger.error(f'investiny returned no data after retries; last error: {last_err}')
            else:
                logger.warning('investiny returned no data')
            return None

        # Normalise keys (investiny may return capitalised column names)
        norm = {k.lower(): v for k, v in data.items()}

        dates = norm.get('date')
        closes = norm.get('close')
        if not dates or not closes:
            logger.warning(f'investiny payload missing date/close: keys={list(data.keys())}')
            return None

        opens = norm.get('open', closes)
        highs = norm.get('high', closes)
        lows = norm.get('low', closes)
        volumes = norm.get('volume', [0] * len(dates))

        # Take the last (most recent) row
        idx = -1
        raw_date = dates[idx]
        if isinstance(raw_date, (int, float)):
            trade_date = date.fromtimestamp(raw_date).isoformat()
        else:
            # investiny returns ISO strings like '2026-05-06'
            trade_date = str(raw_date)[:10]

        vol = volumes[idx] if idx < len(volumes) else 0
        return {
            'date': trade_date,
            'open': float(opens[idx]),
            'high': float(highs[idx]),
            'low': float(lows[idx]),
            'close': float(closes[idx]),
            'volume': float(vol) if vol else 0.0,
            'change_pct': None,
        }

    except Exception as e:
        logger.error(f'Failed to fetch latest price: {e}')
        return None


def is_price_stored(db, trade_date: str) -> bool:
    """Check if a price document already exists in `daily_prices`."""
    doc = db.collection('daily_prices').document(trade_date).get()
    return doc.exists


# ---------------------------------------------------------------------------
# Trading-day helper — used by the daily flow to decide if data is stale
# ---------------------------------------------------------------------------

def most_recent_trading_day(today: Optional[date] = None) -> str:
    """
    Return the most recent weekday on or before `today` as YYYY-MM-DD.

    This is the cutoff the scheduler compares against when deciding whether
    the local price/news CSVs are up to date. We don't keep a Malaysian
    holiday calendar — weekday-only is the practical floor; on a Monday
    public holiday the scheduler will harmlessly try once and find no new
    data, which is fine for an ad-hoc local run.
    """
    today = today or date.today()
    # Saturday=5, Sunday=6 → step back to Friday.
    while today.weekday() >= 5:
        today = today - timedelta(days=1)
    return today.isoformat()


# ---------------------------------------------------------------------------
# Preprocessing — bridge to cpo/preprocess_cpo_variables.py
# ---------------------------------------------------------------------------

def preprocess_price_csv(csv_path: str, output_path: str) -> None:
    """
    Run the offline preprocessing pipeline on the local price CSV and write
    the engineered features to `output_path` (typically cpo/output/cpo_variables_Daily.csv).

    The website's prediction code reads the engineered CSV; the scheduler
    re-runs this whenever a new daily price row is appended.
    """
    import sys
    cpo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cpo'))
    if cpo_dir not in sys.path:
        sys.path.insert(0, cpo_dir)
    from preprocess_cpo_variables import preprocess_cpo  # type: ignore

    df: pd.DataFrame = preprocess_cpo(csv_path, 'Daily')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f'Wrote engineered price CSV: {output_path} ({len(df)} rows)')
