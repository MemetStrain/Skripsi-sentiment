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

    # Application Default Credentials (e.g., on GCP Compute/Cloud Run)
    return firebase_admin.initialize_app()


# ---------------------------------------------------------------------------
# Initial load
# ---------------------------------------------------------------------------

def run_initial_load(db):
    """
    Load ALL historical data into Firestore:
    1. CPO prices from CSV → daily_prices
    2. MPOB news (with sentiment if available) → news_articles
    3. Sentiment aggregates → sentiment_aggregates
    4. HMM states for all frequencies → hmm_states
    5. All 136 predictions → predictions
    """
    logger.info('=== INITIAL LOAD START ===')

    cpo_csv = os.environ.get('CPO_CSV_PATH', '/cpo/Data_CPO_Daily.csv')
    news_csv = os.environ.get('NEWS_CSV_PATH', '/news/mpob_news_fast.csv')
    news_sent_csv = os.environ.get('NEWS_SENT_CSV_PATH', '/news/mpob_news_with_sentiment.csv')

    # Step 1: CPO prices
    logger.info('Step 1: Loading CPO price data from CSV...')
    from price_fetcher import load_prices_from_csv
    from firestore_writer import write_prices_batch
    prices = load_prices_from_csv(cpo_csv)
    if prices:
        write_prices_batch(db, prices)
        logger.info(f'  {len(prices)} price records written.')
    else:
        logger.warning('  No price data loaded.')

    # Step 2: News articles (use pre-computed sentiment CSV to skip FinBERT)
    logger.info('Step 2: Loading MPOB news from CSV...')
    from news_extractor import load_news_from_csv
    from firestore_writer import write_news_articles
    articles = load_news_from_csv(news_csv, news_sent_csv)
    if articles:
        write_news_articles(db, articles)
        logger.info(f'  {len(articles)} articles written.')
    else:
        logger.warning('  No articles loaded.')

    # Step 3: Sentiment aggregates from all loaded articles
    logger.info('Step 3: Computing sentiment aggregates...')
    from sentiment_runner import compute_sentiment_aggregates
    from firestore_writer import write_sentiment_aggregates
    aggregates = compute_sentiment_aggregates(articles)
    if aggregates:
        write_sentiment_aggregates(db, aggregates)
        logger.info(f'  {len(aggregates)} aggregate records written.')

    # Step 4: HMM states
    logger.info('Step 4: Computing HMM states...')
    from hmm_updater import update_hmm_states
    update_hmm_states(db)

    # Step 5: Predictions
    logger.info('Step 5: Computing all 136 predictions...')
    from prediction_updater import run_all_predictions
    run_all_predictions(db)

    logger.info('=== INITIAL LOAD COMPLETE ===')


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
    logger.info('Step 6: Recomputing all 136 predictions...')
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
    args = parser.parse_args()

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
