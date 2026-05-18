"""
firestore_writer.py — centralised Firestore write helpers for the scheduler.

All writes use batch commits (max 450 docs per batch per Firestore limits).
News deduplication uses md5(url) as document ID so set() is idempotent.
"""

import hashlib
import json
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


def wipe_hmm_states(db, frequency: str = 'Daily') -> int:
    """Delete every doc in `hmm_states` for the given frequency."""
    ops = []
    deleted = 0
    docs = db.collection('hmm_states').stream()
    for doc in docs:
        d = doc.to_dict() or {}
        if d.get('frequency') == frequency or doc.id.startswith(f'{frequency}_'):
            ops.append(doc.reference)
    for i in range(0, len(ops), _BATCH_SIZE):
        batch = db.batch()
        for ref in ops[i:i + _BATCH_SIZE]:
            batch.delete(ref)
        batch.commit()
        deleted += len(ops[i:i + _BATCH_SIZE])
        time.sleep(1)
    logger.info(f'Deleted {deleted} HMM state documents (frequency={frequency}).')
    return deleted


# ---------------------------------------------------------------------------
# HMM model parameters (frozen — written by offline training, read by daily
# scheduler so it never re-fits at serve time).
# ---------------------------------------------------------------------------

def write_hmm_params(db, frequency: str, params: Dict) -> None:
    """Persist fitted HMM parameters to `hmm_models/{frequency}`.

    Firestore disallows nested arrays (rejects 2D+ lists like transmat_,
    means_, covars_). We sidestep that by bundling the full params dict as
    a JSON string in `payload_json`, while still surfacing a few scalars
    as native fields for Firestore-console readability.

    The scheduler loads these on every run and reconstructs the GaussianHMM
    rather than refitting. Keeps train/serve consistent.
    """
    doc_ref = db.collection('hmm_models').document(frequency)
    doc_ref.set({
        # Native scalars — handy for inspection & queries.
        'frequency':       frequency,
        'n_components':    int(params.get('n_components', 0)),
        'covariance_type': str(params.get('covariance_type', '')),
        'fit_cutoff':      str(params.get('fit_cutoff', '')),
        'fit_seed':        int(params.get('fit_seed', 0)),
        'fit_timestamp':   str(params.get('fit_timestamp', '')),
        'training_n_obs':  int(params.get('training_n_obs', 0)),
        'updated_at':      _now_iso(),
        # Full params (incl. nested arrays) bundled as one JSON string.
        'payload_json':    json.dumps(params),
    })
    logger.info(f'Wrote HMM params doc: hmm_models/{frequency}')


def read_hmm_params(db, frequency: str = 'Daily') -> Dict[str, Any] | None:
    """Read HMM parameters from `hmm_models/{frequency}`. Returns None if absent.

    New-schema docs store the params dict in a single `payload_json` field
    (Firestore rejects nested arrays). Falls back to the raw doc for any
    legacy docs written field-by-field before that rule was hit.
    """
    snap = db.collection('hmm_models').document(frequency).get()
    if not snap.exists:
        return None
    raw = snap.to_dict() or {}
    if 'payload_json' in raw:
        return json.loads(raw['payload_json'])
    return raw


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
