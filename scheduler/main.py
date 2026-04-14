"""
main.py — CPO Prediction Scheduler entry point.

Usage:
  python main.py --mode initial    # one-time historical data load
  python main.py --mode daily      # incremental daily update (default)

Environment variables required:
  FIREBASE_CREDENTIALS_JSON   Full JSON string of Firebase service account key
  OR
  GOOGLE_APPLICATION_CREDENTIALS  Path to the JSON file (for local/GCP runs)

Optional:
  CPO_CSV_PATH        Path to Data_CPO_Daily.csv  (default: /cpo/Data_CPO_Daily.csv)
  NEWS_CSV_PATH       Path to mpob_news_fast.csv   (default: /news/mpob_news_fast.csv)
  NEWS_SENT_CSV_PATH  Path to mpob_news_with_sentiment.csv
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
# Firebase init
# ---------------------------------------------------------------------------

def init_firebase():
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return firebase_admin.get_app()

    creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON', '').strip()
    if creds_json:
        creds_dict = json.loads(creds_json)
        cred = credentials.Certificate(creds_dict)
        return firebase_admin.initialize_app(cred)

    # GCP default credentials (when running as a Cloud Run service account)
    gac = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if gac and os.path.exists(gac):
        cred = credentials.Certificate(gac)
        return firebase_admin.initialize_app(cred)

    # Local fallback — firebase-credentials.json next to main.py or in website/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, 'firebase-credentials.json'),
        os.path.join(script_dir, '..', 'website', 'firebase-credentials.json'),
    ]:
        if os.path.exists(candidate):
            cred = credentials.Certificate(os.path.abspath(candidate))
            return firebase_admin.initialize_app(cred)

    # Application Default Credentials (e.g., on GCP Compute/Cloud Run)
    return firebase_admin.initialize_app()


# ---------------------------------------------------------------------------
# Initial load
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step checkpoint helpers  (initial load only)
# ---------------------------------------------------------------------------

_PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'initial_load_progress.json')


def _load_progress() -> dict:
    """Return dict of completed steps, e.g. {'step1': True, 'step2': True}."""
    if os.path.exists(_PROGRESS_FILE):
        try:
            with open(_PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _mark_done(progress: dict, step: str) -> None:
    """Mark a step as done and persist to disk."""
    progress[step] = True
    with open(_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def _reset_progress() -> None:
    """Delete the progress file so all steps re-run next time."""
    if os.path.exists(_PROGRESS_FILE):
        os.remove(_PROGRESS_FILE)
        logger.info(f'Progress file removed: {_PROGRESS_FILE}')


# ---------------------------------------------------------------------------
# Initial load
# ---------------------------------------------------------------------------

def run_initial_load(db):
    """
    Load ALL historical data into Firestore:
    1. CPO prices from CSV → daily_prices
    2. MPOB news (with sentiment if available) → news_articles
    3. Sentiment aggregates → sentiment_aggregates
    4. HMM states → hmm_states
    5. All 56 predictions → predictions

    Each step is checkpointed to initial_load_progress.json next to main.py.
    Re-running the script skips any already-completed steps.
    Delete that file (or run with --reset-progress) to start fresh.
    """
    logger.info('=== INITIAL LOAD START ===')
    progress = _load_progress()
    if progress:
        done = [k for k, v in progress.items() if v]
        logger.info(f'  Resuming — already done: {done}')

    _base = os.path.dirname(os.path.abspath(__file__))
    cpo_csv = os.environ.get('CPO_CSV_PATH', os.path.join(_base, '..', 'cpo', 'Data_CPO_Daily.csv'))
    news_csv = os.environ.get('NEWS_CSV_PATH', os.path.join(_base, '..', 'news', 'mpob_news_fast.csv'))
    news_sent_csv = os.environ.get('NEWS_SENT_CSV_PATH', os.path.join(_base, '..', 'news', 'mpob_news_with_sentiment.csv'))

    # Step 1: CPO prices
    if progress.get('step1'):
        logger.info('Step 1: SKIPPED (already done)')
    else:
        logger.info('Step 1: Loading CPO price data from CSV...')
        from price_fetcher import load_prices_from_csv
        from firestore_writer import write_prices_batch
        prices = load_prices_from_csv(cpo_csv)
        if prices:
            write_prices_batch(db, prices)
            logger.info(f'  {len(prices)} price records written.')
        else:
            logger.warning('  No price data loaded.')
        _mark_done(progress, 'step1')

    # Step 2: News articles (use pre-computed sentiment CSV to skip FinBERT)
    if progress.get('step2'):
        logger.info('Step 2: SKIPPED (already done)')
        # Re-load articles in memory so Step 3 can use them if needed
        from news_extractor import load_news_from_csv
        articles = load_news_from_csv(news_csv, news_sent_csv)
    else:
        logger.info('Step 2: Loading MPOB news from CSV...')
        from news_extractor import load_news_from_csv
        from firestore_writer import write_news_articles
        articles = load_news_from_csv(news_csv, news_sent_csv)
        if articles:
            write_news_articles(db, articles)
            logger.info(f'  {len(articles)} articles written.')
        else:
            logger.warning('  No articles loaded.')
        _mark_done(progress, 'step2')

    # Step 3: Sentiment aggregates
    if progress.get('step3'):
        logger.info('Step 3: SKIPPED (already done)')
    else:
        logger.info('Step 3: Computing sentiment aggregates...')
        from sentiment_runner import compute_sentiment_aggregates
        from firestore_writer import write_sentiment_aggregates
        aggregates = compute_sentiment_aggregates(articles)
        if aggregates:
            write_sentiment_aggregates(db, aggregates)
            logger.info(f'  {len(aggregates)} aggregate records written.')
        _mark_done(progress, 'step3')

    # Step 4: HMM states
    if progress.get('step4'):
        logger.info('Step 4: SKIPPED (already done)')
    else:
        logger.info('Step 4: Computing HMM states...')
        from hmm_updater import update_hmm_states
        update_hmm_states(db)
        _mark_done(progress, 'step4')

    # Step 5: Predictions
    if progress.get('step5'):
        logger.info('Step 5: SKIPPED (already done)')
    else:
        logger.info('Step 5: Computing all 56 predictions...')
        from prediction_updater import run_all_predictions
        run_all_predictions(db)
        _mark_done(progress, 'step5')

    logger.info('=== INITIAL LOAD COMPLETE ===')
    logger.info(f'  (Progress file: {_PROGRESS_FILE} — delete it to re-run from scratch)')


# ---------------------------------------------------------------------------
# Daily update
# ---------------------------------------------------------------------------

def run_daily_update(db):
    """
    Incremental daily update:
    1. Fetch latest CPO price → daily_prices
    2. Scrape new MPOB articles → news_articles
    3. Run FinBERT on new articles
    4. Update sentiment aggregates
    5. Re-run HMM states
    6. Recompute all 136 predictions
    """
    logger.info('=== DAILY UPDATE START ===')

    # Step 1: CPO price
    logger.info('Step 1: Fetching latest CPO price...')
    from price_fetcher import fetch_latest_price, is_price_stored
    from firestore_writer import write_price
    price = fetch_latest_price()
    if price:
        if not is_price_stored(db, price['date']):
            write_price(db, price)
            logger.info(f"  New price: {price['date']} close={price['close']}")
        else:
            logger.info(f"  Price for {price['date']} already stored, skipping.")
    else:
        logger.warning('  Failed to fetch latest price.')

    # Step 2: Scrape new news
    logger.info('Step 2: Scraping new MPOB articles...')
    from firestore_writer import get_latest_article_date, write_news_articles
    from news_extractor import scrape_new_articles
    cutoff = get_latest_article_date(db)
    new_articles = scrape_new_articles(cutoff)
    logger.info(f'  {len(new_articles)} new articles scraped.')

    if new_articles:
        # Step 3: FinBERT on new articles
        logger.info('Step 3: Running FinBERT sentiment on new articles...')
        from sentiment_runner import run_sentiment_on_articles
        new_articles = run_sentiment_on_articles(new_articles)
        write_news_articles(db, new_articles)
        logger.info(f'  {len(new_articles)} articles with sentiment written.')

        # Step 4: Update sentiment aggregates for new dates
        logger.info('Step 4: Updating sentiment aggregates...')
        from sentiment_runner import compute_sentiment_aggregates
        from firestore_writer import write_sentiment_aggregates
        new_aggregates = compute_sentiment_aggregates(new_articles)
        if new_aggregates:
            write_sentiment_aggregates(db, new_aggregates)
            logger.info(f'  {len(new_aggregates)} aggregate records updated.')
    else:
        logger.info('Step 3–4: No new articles, skipping sentiment.')

    # Step 5: HMM states
    logger.info('Step 5: Updating HMM states...')
    from hmm_updater import update_hmm_states
    update_hmm_states(db)

    # Step 6: Predictions
    logger.info('Step 6: Recomputing all 56 predictions...')
    from prediction_updater import run_all_predictions
    run_all_predictions(db)

    logger.info('=== DAILY UPDATE COMPLETE ===')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CPO Prediction Scheduler')
    parser.add_argument(
        '--mode',
        choices=['initial', 'daily'],
        default='daily',
        help='initial = first-run historical load; daily = incremental update',
    )
    parser.add_argument(
        '--reset-progress',
        action='store_true',
        help='Delete the initial load checkpoint file and exit (next --mode initial runs all steps)',
    )
    args = parser.parse_args()

    if args.reset_progress:
        _reset_progress()
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
