"""
reconcile.py — full CSV→Firestore reconciliation for the scheduler.

The daily scheduler appends new rows to the local CSVs (the source of truth)
and then calls these helpers to make Firestore an exact mirror.

Unlike the old incremental writes — which only pushed the single row the
scheduler itself fetched/scraped — reconciliation diffs the *entire* CSV
against the Firestore collection every run. Any drift heals on the next run
instead of accumulating:

  * a daily run that was skipped or failed midway,
  * a CSV updated by another offline script (cpo preprocessing, the news
    pipeline) without the scheduler ever seeing it,
  * a partially-committed batch.

Each helper streams the target collection once (reads are cheap), computes
the set of docs that are missing or stale, and writes only that diff (writes
are the costly, quota-limited operation).
"""

import logging

from firestore_writer import (
    write_prices_batch, write_news_articles, write_sentiment_aggregates,
    url_to_doc_id,
)
from price_fetcher import load_prices_from_csv
from news_extractor import load_news_from_csv
from sentiment_runner import compute_sentiment_aggregates

logger = logging.getLogger(__name__)

# Two price rows count as equal when every OHLCV field matches within this
# tolerance (CSVs store 2 decimals; Firestore stores full floats).
_PRICE_TOL = 0.01
# Two aggregates count as equal when the mean sentiment score matches within
# this tolerance.
_AGG_TOL = 1e-4


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def _prices_differ(csv_row: dict, fs_doc: dict) -> bool:
    for field in ('open', 'high', 'low', 'close', 'volume'):
        a = float(csv_row.get(field, 0) or 0)
        b = float(fs_doc.get(field, 0) or 0)
        if abs(a - b) > _PRICE_TOL:
            return True
    return False


def reconcile_prices(db, csv_prices: list[dict]) -> int:
    """Mirror every row of the CPO price CSV into `daily_prices`.

    Writes only rows absent from Firestore or whose OHLCV differs.
    Returns the number of documents written.
    """
    if not csv_prices:
        logger.info('  Reconcile prices: CSV empty or unreadable; nothing to do.')
        return 0

    existing = {doc.id: (doc.to_dict() or {})
                for doc in db.collection('daily_prices').stream()}

    stale = [
        r for r in csv_prices
        if r['date'] not in existing or _prices_differ(r, existing[r['date']])
    ]
    if not stale:
        logger.info(f'  Reconcile prices: in sync ({len(csv_prices)} rows).')
        return 0

    logger.info(f'  Reconcile prices: {len(stale)} of {len(csv_prices)} '
                f'rows missing/stale -> writing.')
    return write_prices_batch(db, stale)


# ---------------------------------------------------------------------------
# News articles
# ---------------------------------------------------------------------------

def reconcile_news(db, articles: list[dict]) -> int:
    """Mirror every article in the news CSVs into `news_articles`.

    A news record is immutable once published, so this only writes articles
    whose URL hash is not yet a document. Returns the number written.
    """
    if not articles:
        logger.info('  Reconcile news: CSV empty or unreadable; nothing to do.')
        return 0

    existing_ids = {doc.id for doc in db.collection('news_articles').stream()}
    missing = [a for a in articles
               if a.get('url') and url_to_doc_id(a['url']) not in existing_ids]
    if not missing:
        logger.info(f'  Reconcile news: in sync ({len(articles)} articles).')
        return 0

    logger.info(f'  Reconcile news: {len(missing)} of {len(articles)} '
                f'articles missing -> writing.')
    return write_news_articles(db, missing)


# ---------------------------------------------------------------------------
# Sentiment aggregates
# ---------------------------------------------------------------------------

def _aggs_differ(csv_agg: dict, fs_doc: dict) -> bool:
    if int(csv_agg.get('article_count', 0)) != int(fs_doc.get('article_count', 0)):
        return True
    return abs(float(csv_agg.get('sentiment_score', 0)) -
               float(fs_doc.get('sentiment_score', 0))) > _AGG_TOL


def reconcile_aggregates(db, articles: list[dict]) -> int:
    """Recompute Daily sentiment aggregates from the news CSVs and mirror them
    into `sentiment_aggregates`.

    Writes only dates that are missing or whose article_count / mean score
    changed (a date gets fresh aggregates whenever it receives a new article).
    Returns the number of documents written.
    """
    aggregates = compute_sentiment_aggregates(articles)
    if not aggregates:
        logger.info('  Reconcile aggregates: nothing computed; skipping.')
        return 0

    existing = {doc.id: (doc.to_dict() or {})
                for doc in db.collection('sentiment_aggregates').stream()}

    stale = []
    for agg in aggregates:
        doc_id = f"{agg.get('frequency', 'Daily')}_{agg['date']}"
        if doc_id not in existing or _aggs_differ(agg, existing[doc_id]):
            stale.append(agg)
    if not stale:
        logger.info(f'  Reconcile aggregates: in sync ({len(aggregates)} dates).')
        return 0

    logger.info(f'  Reconcile aggregates: {len(stale)} of {len(aggregates)} '
                f'dates missing/stale -> writing.')
    return write_sentiment_aggregates(db, stale)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def reconcile_all(db, paths: dict) -> int:
    """Run all three CSV→Firestore reconciliations.

    Returns the total number of documents written across prices, news and
    aggregates — 0 means Firestore was already an exact mirror of the CSVs.
    """
    logger.info('Reconciling local CSVs -> Firestore...')
    total = 0

    total += reconcile_prices(db, load_prices_from_csv(paths['cpo_raw']))

    # News + aggregates derive from the same CSV; load it once.
    articles = load_news_from_csv(paths['news_raw'], paths['news_sent'])
    total += reconcile_news(db, articles)
    total += reconcile_aggregates(db, articles)

    logger.info(f'Reconcile complete: {total} document(s) written.')
    return total
