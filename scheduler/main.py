"""
main.py — CPO Prediction Scheduler entry point (local-only).

Usage:
  python main.py --mode initial    # one-time bulk load from local CSVs to Firestore
  python main.py --mode daily      # incremental: fetch price, scrape+score news,
                                   #              recompute aggregates, refresh HMM

Local CSVs are the source of truth. Firestore is a downstream mirror that the
website reads. Predictions are no longer computed here — the website performs
live inference using the offline-trained weights under prediction/saved_models/.

Environment variables required:
  FIREBASE_CREDENTIALS_JSON       Full JSON string of Firebase service account key
  OR
  GOOGLE_APPLICATION_CREDENTIALS  Path to the JSON file (for local runs)

Optional (paths):
  CPO_CSV_PATH            Default: ../cpo/Data_CPO_Daily.csv
  CPO_PREPROC_CSV_PATH    Default: ../cpo/output/cpo_variables_Daily.csv
  NEWS_RAW_CSV_PATH       Default: ../news/mpob_news_fast.csv
  NEWS_PREPROC_CSV_PATH   Default: ../news/mpob_news_preprocessed.csv
  NEWS_SENT_CSV_PATH      Default: ../news/mpob_news_with_sentiment_tone.csv
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger('scheduler')


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_BASE = os.path.dirname(os.path.abspath(__file__))


def _path(env_var: str, default_relative: str) -> str:
    """Resolve a CSV path from env var, falling back to a path relative to scheduler/."""
    return os.environ.get(env_var) or os.path.abspath(os.path.join(_BASE, default_relative))


def _paths() -> dict:
    return {
        'cpo_raw':       _path('CPO_CSV_PATH',           '../cpo/Data_CPO_Daily.csv'),
        'cpo_preproc':   _path('CPO_PREPROC_CSV_PATH',   '../cpo/output/cpo_variables_Daily.csv'),
        'news_raw':      _path('NEWS_RAW_CSV_PATH',      '../news/mpob_news_fast.csv'),
        'news_preproc':  _path('NEWS_PREPROC_CSV_PATH',  '../news/mpob_news_preprocessed.csv'),
        'news_sent':     _path('NEWS_SENT_CSV_PATH',     '../news/mpob_news_with_sentiment_tone.csv'),
    }


# ---------------------------------------------------------------------------
# Firebase init
# ---------------------------------------------------------------------------

def init_firebase():
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return firebase_admin.get_app()

    creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON', '').strip()
    if creds_json:
        cred = credentials.Certificate(json.loads(creds_json))
        return firebase_admin.initialize_app(cred)

    gac = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if gac and os.path.exists(gac):
        return firebase_admin.initialize_app(credentials.Certificate(gac))

    for candidate in [
        os.path.join(_BASE, 'firebase-credentials.json'),
        os.path.join(_BASE, '..', 'website', 'firebase-credentials.json'),
    ]:
        if os.path.exists(candidate):
            return firebase_admin.initialize_app(credentials.Certificate(os.path.abspath(candidate)))

    return firebase_admin.initialize_app()  # ADC fallback


# ---------------------------------------------------------------------------
# Step-checkpoint helpers (initial load only)
# ---------------------------------------------------------------------------

_PROGRESS_FILE = os.path.join(_BASE, 'initial_load_progress.json')


def _load_progress() -> dict:
    if os.path.exists(_PROGRESS_FILE):
        try:
            with open(_PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _mark_done(progress: dict, step: str) -> None:
    progress[step] = True
    with open(_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def _reset_progress() -> None:
    if os.path.exists(_PROGRESS_FILE):
        os.remove(_PROGRESS_FILE)
        logger.info(f'Progress file removed: {_PROGRESS_FILE}')


# ---------------------------------------------------------------------------
# Initial load — bulk-mirror local CSVs into Firestore
# ---------------------------------------------------------------------------

def run_initial_load(db):
    """
    1. CPO prices  →  daily_prices
    2. MPOB news   →  news_articles  (uses pre-computed sentiment in the tone CSV)
    3. Aggregates  →  sentiment_aggregates  (recomputed from full news CSV)
    4. HMM states  →  hmm_states

    Each step is checkpointed; delete initial_load_progress.json (or run with
    --reset-progress) to start fresh.
    """
    logger.info('=== INITIAL LOAD START ===')
    progress = _load_progress()
    if progress:
        logger.info(f'  Resuming — already done: {[k for k, v in progress.items() if v]}')

    paths = _paths()

    # Step 1: CPO prices.
    if progress.get('step1'):
        logger.info('Step 1: SKIPPED (already done)')
    else:
        logger.info('Step 1: Loading CPO price data from CSV...')
        from price_fetcher import load_prices_from_csv
        from firestore_writer import write_prices_batch
        prices = load_prices_from_csv(paths['cpo_raw'])
        if prices:
            write_prices_batch(db, prices)
        _mark_done(progress, 'step1')

    # Step 2: News articles (the tone CSV already has Combined_* sentiment).
    if progress.get('step2'):
        logger.info('Step 2: SKIPPED (already done)')
        from news_extractor import load_news_from_csv
        articles = load_news_from_csv(paths['news_raw'], paths['news_sent'])
    else:
        logger.info('Step 2: Loading MPOB news from CSV...')
        from news_extractor import load_news_from_csv
        from firestore_writer import write_news_articles
        articles = load_news_from_csv(paths['news_raw'], paths['news_sent'])
        if articles:
            write_news_articles(db, articles)
        _mark_done(progress, 'step2')

    # Step 3: Sentiment aggregates (rebuild from full CSV).
    if progress.get('step3'):
        logger.info('Step 3: SKIPPED (already done)')
    else:
        logger.info('Step 3: Computing sentiment aggregates from full news CSV...')
        from sentiment_runner import compute_sentiment_aggregates
        from firestore_writer import write_sentiment_aggregates
        aggregates = compute_sentiment_aggregates(articles)
        if aggregates:
            write_sentiment_aggregates(db, aggregates)
        _mark_done(progress, 'step3')

    # Step 4: HMM states.
    if progress.get('step4'):
        logger.info('Step 4: SKIPPED (already done)')
    else:
        logger.info('Step 4: Computing HMM states...')
        from hmm_updater import update_hmm_states
        update_hmm_states(db)
        _mark_done(progress, 'step4')

    logger.info('=== INITIAL LOAD COMPLETE ===')
    logger.info(f'  (Progress file: {_PROGRESS_FILE} — delete it to re-run from scratch)')


# ---------------------------------------------------------------------------
# Daily update — incremental, CSV-as-source-of-truth
# ---------------------------------------------------------------------------

def _is_aggregates_empty(db) -> bool:
    """True if the Firestore `sentiment_aggregates` collection has no documents."""
    return not list(db.collection('sentiment_aggregates').limit(1).stream())


def _step_price(db, paths: dict) -> bool:
    """
    Fetch the latest price; if newer than the local CSV, append the CSV,
    re-run preprocessing, and upsert Firestore. Returns True if anything changed.
    """
    from price_fetcher import fetch_latest_price, most_recent_trading_day, preprocess_price_csv
    from firestore_writer import write_price
    from local_csv_writer import latest_price_date, append_price_row_indonesian

    cutoff = most_recent_trading_day()
    have   = latest_price_date(paths['cpo_raw'])
    if have and have >= cutoff:
        logger.info(f'  Price CSV is current (latest={have}, cutoff={cutoff}). Skip fetch.')
        return False

    logger.info(f'  Price CSV is stale (latest={have}, cutoff={cutoff}). Fetching...')
    price = fetch_latest_price()
    if not price:
        logger.warning('  fetch_latest_price returned None.')
        return False

    if have and price['date'] <= have:
        logger.info(f'  Fetched price {price["date"]} not newer than CSV ({have}). Skip.')
        return False

    if not append_price_row_indonesian(paths['cpo_raw'], price):
        logger.info('  Price already present in CSV.')
        return False

    logger.info('  Re-running CPO preprocessing pipeline...')
    preprocess_price_csv(paths['cpo_raw'], paths['cpo_preproc'])

    write_price(db, price)
    logger.info(f'  Mirrored price {price["date"]} to Firestore.')
    return True


def _step_news(paths: dict) -> tuple[list, bool]:
    """
    If the local tone CSV is older than the most recent trading day, scrape
    new articles, preprocess, score, and append all three local CSVs.
    Returns (new_articles_list, did_change).
    """
    from price_fetcher import most_recent_trading_day
    from news_extractor import (
        scrape_new_articles, preprocess_articles,
        article_to_raw_row, article_to_sentiment_row, RAW_FIELDS, SENTIMENT_FIELDS,
    )
    from sentiment_runner import run_sentiment_on_articles
    from local_csv_writer import latest_news_date, append_news_rows

    cutoff = most_recent_trading_day()
    have   = latest_news_date(paths['news_sent'])
    if have and have >= cutoff:
        logger.info(f'  News CSV is current (latest={have}, cutoff={cutoff}). Skip scrape.')
        return [], False

    logger.info(f'  News CSV is stale (latest={have}, cutoff={cutoff}). Scraping...')
    raw = scrape_new_articles(have)
    if not raw:
        logger.info('  No new articles scraped.')
        return [], False

    # Append RAW first so we have a record of what was scraped before any
    # downstream step can lose it (e.g. preprocessing dropping empty articles).
    append_news_rows(paths['news_raw'], [article_to_raw_row(a) for a in raw], RAW_FIELDS)

    cleaned = preprocess_articles(raw)  # mutates in place, drops empty
    if not cleaned:
        logger.info('  All scraped articles dropped during preprocessing.')
        return [], False
    append_news_rows(paths['news_preproc'], [article_to_raw_row(a) for a in cleaned], RAW_FIELDS)

    scored = run_sentiment_on_articles(cleaned)
    append_news_rows(paths['news_sent'],
                     [article_to_sentiment_row(a) for a in scored], SENTIMENT_FIELDS)

    return scored, True


def run_daily_update(db):
    """
    Incremental update against the local CSVs and Firestore mirror:

    1. Price: fetch, dedup against local CSV, append + preprocess if new.
    2. News:  scrape since latest tone-CSV date, preprocess, score with FinBERT-Tone,
              append to all three news CSVs.
    3. Mirror new news articles into Firestore.
    4. Recompute sentiment aggregates from the full local tone CSV → Firestore.
    5. Recompute HMM states.

    Predictions are NOT computed here — the website handles inference live.
    """
    logger.info('=== DAILY UPDATE START ===')
    paths = _paths()

    # 1. Price.
    logger.info('Step 1: Price update')
    _step_price(db, paths)

    # 2. News.
    logger.info('Step 2: News update')
    new_articles, news_changed = _step_news(paths)

    # 3. Mirror new news to Firestore.
    if news_changed and new_articles:
        from firestore_writer import write_news_articles
        write_news_articles(db, new_articles)
        logger.info(f'  Mirrored {len(new_articles)} new articles to Firestore.')

    # 4. Sentiment aggregates — incremental, with one-shot rebuild on re-init.
    logger.info('Step 4: Updating sentiment aggregates...')
    from news_extractor import load_news_from_csv
    from sentiment_runner import compute_sentiment_aggregates
    from firestore_writer import write_sentiment_aggregates

    if news_changed and new_articles:
        # Recompute only the dates that received new articles, but include ALL
        # articles on those dates (avoids undercounting when a date already
        # had partial aggregates from a prior run).
        affected_dates = {a['date'] for a in new_articles if a.get('date')}
        all_articles = load_news_from_csv(paths['news_raw'], paths['news_sent'])
        affected = [a for a in all_articles if a.get('date') in affected_dates]
        aggregates = compute_sentiment_aggregates(affected)
        if aggregates:
            write_sentiment_aggregates(db, aggregates)
            logger.info(f'  Updated aggregates for {len(aggregates)} affected date(s).')
    elif _is_aggregates_empty(db):
        logger.info('  Firestore aggregates empty (post-reinit). Rebuilding from full CSV...')
        all_articles = load_news_from_csv(paths['news_raw'], paths['news_sent'])
        aggregates = compute_sentiment_aggregates(all_articles)
        if aggregates:
            write_sentiment_aggregates(db, aggregates)
    else:
        logger.info('  No news change; aggregates already up to date.')

    # 5. HMM states.
    logger.info('Step 5: Updating HMM states...')
    from hmm_updater import update_hmm_states
    update_hmm_states(db)

    logger.info('=== DAILY UPDATE COMPLETE ===')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CPO Prediction Scheduler (local)')
    parser.add_argument('--mode', choices=['initial', 'daily', 'rebuild-hmm'],
                        default='daily',
                        help=('initial      = bulk-load CSVs to Firestore; '
                              'daily        = incremental update; '
                              'rebuild-hmm  = publish frozen HMM params + '
                              'wipe & rebuild hmm_states from the offline CSV '
                              '(run after re-training the offline HMM).'))
    parser.add_argument('--reset-progress', action='store_true',
                        help='Delete the initial-load checkpoint file and exit')
    args = parser.parse_args()

    if args.reset_progress:
        _reset_progress()
        return

    if args.mode == 'rebuild-hmm':
        # Delegate to the dedicated migration script so its CLI args stay
        # discoverable. Reset argv so argparse inside migrate_hmm sees no leftovers.
        import migrate_hmm_to_firestore
        sys.argv = ['migrate_hmm_to_firestore.py']
        migrate_hmm_to_firestore.main()
        return

    try:
        init_firebase()
    except Exception as e:
        logger.error(f'Firebase initialisation failed: {e}')
        sys.exit(1)

    from firebase_admin import firestore
    db = firestore.client()

    if args.mode == 'initial':
        run_initial_load(db)
    else:
        run_daily_update(db)


if __name__ == '__main__':
    main()
