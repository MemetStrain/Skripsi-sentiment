"""
local_csv_writer.py — append-with-dedup helpers for the local CSV files
that act as the source of truth for the scheduler.

The scheduler now writes to three news CSVs (raw scrape, preprocessed,
sentiment-tone) and one price CSV. These helpers do an in-place merge
that is idempotent across runs.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# News CSVs — keyed by URL
# ---------------------------------------------------------------------------

def append_news_rows(csv_path: str, new_rows: Iterable[dict],
                     fieldnames: Optional[list] = None) -> int:
    """
    Append `new_rows` to `csv_path`, dropping any whose URL is already present.
    Creates the file with a header if it does not exist.

    Returns the number of rows actually written.
    """
    new_rows = list(new_rows)
    if not new_rows:
        return 0

    if fieldnames is None:
        fieldnames = list(new_rows[0].keys())

    existing_urls = _read_csv_column(csv_path, 'URL')
    deduped = [r for r in new_rows if r.get('URL') and r['URL'] not in existing_urls]
    if not deduped:
        return 0

    file_existed = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_existed:
            writer.writeheader()
        for row in deduped:
            writer.writerow(row)

    logger.info(f'Appended {len(deduped)} rows to {csv_path}')
    return len(deduped)


def latest_news_date(csv_path: str) -> Optional[str]:
    """Return the most recent date (YYYY-MM-DD) in the news CSV, or None."""
    if not os.path.exists(csv_path):
        return None

    latest = None
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = _normalise_date(row.get('Date', row.get('date', '')))
            if d and (latest is None or d > latest):
                latest = d
    return latest


# ---------------------------------------------------------------------------
# Price CSV — keyed by Date
# ---------------------------------------------------------------------------

def latest_price_date(csv_path: str) -> Optional[str]:
    """Return the most recent date (YYYY-MM-DD) in the CPO price CSV, or None."""
    if not os.path.exists(csv_path):
        return None

    latest = None
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First column is "Tanggal" (Indonesian) or "Date"
            raw = row.get('Tanggal') or row.get('Date') or ''
            d = _normalise_date(raw)
            if d and (latest is None or d > latest):
                latest = d
    return latest


def is_price_date_in_csv(csv_path: str, iso_date: str) -> bool:
    """True if `iso_date` (YYYY-MM-DD) already appears in the CPO price CSV."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get('Tanggal') or row.get('Date') or ''
            if _normalise_date(raw) == iso_date:
                return True
    return False


def append_price_row_indonesian(csv_path: str, price: dict) -> bool:
    """
    Append one price row to the Indonesian-format CPO CSV. Skips if the date
    already exists. Expects price as a dict with ISO date and float OHLCV.

    Returns True if a row was appended.
    """
    iso_date = price['date']
    if is_price_date_in_csv(csv_path, iso_date):
        return False

    # Convert ISO date back to Indonesian display format (DD/MM/YYYY).
    dt = datetime.strptime(iso_date, '%Y-%m-%d')
    id_date = dt.strftime('%d/%m/%Y')

    def fmt_num(v: float) -> str:
        # Indonesian number format: thousands '.', decimal ','. Two decimals.
        s = f'{v:,.2f}'  # English: 1,234.56
        return s.replace(',', '_').replace('.', ',').replace('_', '.')

    def fmt_volume(v: float) -> str:
        if v <= 0:
            return '-'
        if v >= 1000:
            return fmt_num(v / 1000.0) + 'K'
        return fmt_num(v)

    def fmt_pct(v: Optional[float]) -> str:
        if v is None:
            return '-'
        return fmt_num(v) + '%'

    row = {
        'Tanggal':   id_date,
        'Terakhir':  fmt_num(price['close']),
        'Pembukaan': fmt_num(price['open']),
        'Tertinggi': fmt_num(price['high']),
        'Terendah':  fmt_num(price['low']),
        'Vol.':      fmt_volume(price.get('volume', 0)),
        'Perubahan%': fmt_pct(price.get('change_pct')),
    }

    file_existed = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)

    fieldnames = ['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi',
                  'Terendah', 'Vol.', 'Perubahan%']
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_existed:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f'Appended price row {iso_date} to {csv_path}')
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_column(csv_path: str, column: str) -> set:
    """Return a set of all values in `column` of the given CSV (lookup case-insensitive)."""
    if not os.path.exists(csv_path):
        return set()
    out = set()
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        # Resolve column case-insensitively against actual header
        header_map = {h.lower(): h for h in (reader.fieldnames or [])}
        actual = header_map.get(column.lower())
        if not actual:
            return set()
        for row in reader:
            v = row.get(actual)
            if v:
                out.add(v.strip())
    return out


def _normalise_date(raw: str) -> Optional[str]:
    """Return YYYY-MM-DD or None. Accepts a few common formats."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
                '%d %b %Y', '%B %d, %Y'):
        try:
            return datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None
