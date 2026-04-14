"""
firestore_writer.py — centralised Firestore write helpers for the scheduler.

All writes use batch commits (max 450 docs per batch per Firestore limits).
News deduplication uses md5(url) as document ID so set() is idempotent.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

from firebase_admin import firestore

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # smaller batches to avoid quota exhaustion


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batch_write(db, operations: List[tuple]) -> int:
    """
    Execute a list of (doc_ref, data, merge) tuples in batches.
    Returns total docs written.
    """
    total = 0
    for i in range(0, len(operations), _BATCH_SIZE):
        batch = db.batch()
        chunk = operations[i:i + _BATCH_SIZE]
        for doc_ref, data, merge in chunk:
            if merge:
                batch.set(doc_ref, data, merge=True)
            else:
                batch.set(doc_ref, data)
        batch.commit()
        total += len(chunk)
        logger.info(f'  Batch committed: {total} docs so far...')
        time.sleep(1)  # 1s pause between batches to stay within quota
    return total


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def write_price(db, price_data: Dict) -> None:
    """Write a single price document to `daily_prices`. Doc ID = YYYY-MM-DD."""
    doc_id = price_data['date']
    doc_ref = db.collection('daily_prices').document(doc_id)
    doc_ref.set({**price_data, 'updated_at': _now_iso()}, merge=True)


def write_prices_batch(db, prices: List[Dict]) -> int:
    """Batch-write a list of price dicts to `daily_prices`."""
    ops = []
    for p in prices:
        doc_ref = db.collection('daily_prices').document(p['date'])
        ops.append((doc_ref, {**p, 'updated_at': _now_iso()}, True))
    count = _batch_write(db, ops)
    logger.info(f'Wrote {count} price documents.')
    return count


# ---------------------------------------------------------------------------
# HMM states
# ---------------------------------------------------------------------------

def write_hmm_states_batch(db, states: List[Dict]) -> int:
    """Batch-write HMM state documents to `hmm_states`.
    Doc ID = {frequency}_{YYYY-MM-DD}.
    """
    ops = []
    for s in states:
        doc_id = f"{s['frequency']}_{s['date']}"
        doc_ref = db.collection('hmm_states').document(doc_id)
        ops.append((doc_ref, {**s, 'updated_at': _now_iso()}, False))
    count = _batch_write(db, ops)
    logger.info(f'Wrote {count} HMM state documents.')
    return count


# ---------------------------------------------------------------------------
# News articles
# ---------------------------------------------------------------------------

def url_to_doc_id(url: str) -> str:
    """Stable doc ID based on URL hash — prevents duplicate inserts."""
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def write_news_articles(db, articles: List[Dict]) -> int:
    """Batch-write news articles to `news_articles`. Idempotent via URL hash."""
    ops = []
    for a in articles:
        url = a.get('url', '')
        if not url:
            continue
        doc_id = url_to_doc_id(url)
        doc_ref = db.collection('news_articles').document(doc_id)
        ops.append((doc_ref, {**a, 'scraped_at': _now_iso()}, False))
    count = _batch_write(db, ops)
    logger.info(f'Wrote {count} news article documents.')
    return count


def get_latest_article_date(db) -> str | None:
    """Return the most recent `date` field in `news_articles`, or None."""
    docs = (
        db.collection('news_articles')
        .order_by('date', direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        return doc.to_dict().get('date')
    return None


# ---------------------------------------------------------------------------
# Sentiment aggregates
# ---------------------------------------------------------------------------

def write_sentiment_aggregates(db, aggregates: List[Dict]) -> int:
    """Write sentiment aggregate documents to `sentiment_aggregates`.
    Doc ID = {frequency}_{date_key}.
    """
    ops = []
    for agg in aggregates:
        freq = agg.get('frequency', 'Daily')
        date = agg.get('date', '')
        doc_id = f'{freq}_{date}'
        doc_ref = db.collection('sentiment_aggregates').document(doc_id)
        ops.append((doc_ref, agg, False))
    count = _batch_write(db, ops)
    logger.info(f'Wrote {count} sentiment aggregate documents.')
    return count


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def write_prediction(db, model: str, variant: str, frequency: str, horizon: int, data: Dict) -> None:
    """Write / overwrite a single prediction document."""
    doc_id = f'{model}_{variant}_{frequency}_h{horizon}'
    doc_ref = db.collection('predictions').document(doc_id)
    doc_ref.set({**data, 'computed_at': _now_iso()})
