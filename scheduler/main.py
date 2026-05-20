"""
main.py — CPO Prediction Scheduler entry point (local-only).

Usage:
  python main.py --mode initial    # one-time bulk load from local CSVs to Firestore
  python main.py --mode daily      # incremental: fetch price, scrape+score news,
                                   #              reconcile Firestore, refresh HMM
                                   #              + forecasts

Local CSVs are the source of truth. Firestore is a downstream mirror that the
website reads.

Every daily run finishes by RECONCILING the full CSVs against Firestore
(see reconcile.py) — so a skipped run or a CSV edited by another offline
script can no longer leave Firestore permanently behind; the next run heals
the drift.

The run then refreshes the rolling forecast trail by calling
scheduler/precompute_forecasts.py in-process — it imports
prediction/inference.py and writes per-(horizon, anchor) docs to the
`forecasts` collection plus `forecast_meta/Daily`. XGBoost inference is
too heavy for Vercel's serverless functions, so it is precomputed here
and the site only reads those Firestore docs.

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
    5. Forecasts   →  forecasts/* + forecast_meta/Daily  (precompute_forecasts.precompute_and_write)

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

    # Step 5: Forecasts (precomputed from the freshly-loaded Firestore data).
    # Must run AFTER HMM (step 4): inference reads daily_prices, sentiment_aggregates,
    # and hmm_states, so all three must be committed first.
    if progress.get('step5'):
        logger.info('Step 5: SKIPPED (already done)')
    else:
        logger.info('Step 5: Precomputing forecasts...')
        if _step_precompute(db):
            _mark_done(progress, 'step5')
        else:
            logger.warning('Step 5: forecast precompute failed — rerun to retry.')

    logger.info('=== INITIAL LOAD COMPLETE ===')
    logger.info(f'  (Progress file: {_PROGRESS_FILE} — delete it to re-run from scratch)')


# ---------------------------------------------------------------------------
# Daily update — incremental, CSV-as-source-of-truth
# ---------------------------------------------------------------------------

def _step_price(paths: dict) -> bool:
    """
    Fetch the latest price; if newer than the local CSV, append the CSV and
    re-run preprocessing. Returns True if the CSV changed.

    Firestore is NOT written here — `reconcile_all` mirrors the full CSV into
    Firestore afterwards, so a single place owns CSV->Firestore sync.
    """
    from price_fetcher import fetch_latest_price, most_recent_trading_day, preprocess_price_csv
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

    logger.info(f'  Appended price {price["date"]} to local CSV.')
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


def _step_precompute(db) -> bool:
    """Refresh the rolling forecast trail by calling the in-process orchestrator.

    Wrapped in try/except: forecast failure must NOT roll back the
    already-committed price / news / reconcile / HMM updates earlier in
    the run. The orchestrator imports prediction/inference.py and writes
    per-(horizon, anchor) docs to `forecasts` + `forecast_meta/Daily`.

    Returns True if at least one forecast document was written.
    """
    try:
        import precompute_forecasts
        n = precompute_forecasts.precompute_and_write(db)
    except Exception as e:
        logger.exception(f'  Forecast precompute failed: {e}')
        return False
    return n > 0


def run_daily_update(db):
    """
    Incremental update against the local CSVs, then a full reconcile so the
    Firestore mirror can never drift out of sync with the CSVs:

    1. Price:     fetch, dedup against local CSV, append + preprocess if new.
    2. News:      scrape since latest tone-CSV date, preprocess, score with
                  FinBERT-Tone, append to all three news CSVs.
    3. Reconcile: diff the FULL CSVs against Firestore and write any missing /
                  stale prices, news and sentiment aggregates. Self-healing —
                  a skipped run or an externally-edited CSV is corrected here.
    4. HMM:       decode states from the now-current daily_prices.
    5. Forecasts: refresh forecasts/* + forecast_meta/Daily by calling
                  precompute_forecasts.precompute_and_write in-process.

    Steps 1-2 only acquire data into the CSVs (the source of truth); step 3 is
    the single place that writes CSV data to Firestore.
    """
    logger.info('=== DAILY UPDATE START ===')
    paths = _paths()

    # 1. Price acquisition → local CSV.
    logger.info('Step 1: Price acquisition')
    price_changed = _step_price(paths)

    # 2. News acquisition → local CSVs.
    logger.info('Step 2: News acquisition')
    _, news_changed = _step_news(paths)  # articles mirrored by reconcile, below

    # 3. Reconcile full CSVs → Firestore (prices, news, sentiment aggregates).
    logger.info('Step 3: Reconcile CSVs -> Firestore')
    from reconcile import reconcile_all
    recon_writes = reconcile_all(db, paths)

    # 4. HMM states — decoded from the freshly-reconciled daily_prices.
    logger.info('Step 4: Updating HMM states...')
    from hmm_updater import update_hmm_states
    update_hmm_states(db)

    # 5. Forecasts — refresh the rolling trail so the dashboard tracks the new
    #    data. Skipped when nothing changed upstream (deterministic doc IDs
    #    mean a re-run would just overwrite the same docs). Isolated in
    #    try/except inside _step_precompute so a forecast failure cannot
    #    roll back the already-committed price/news/HMM updates above.
    logger.info('Step 5: Refreshing forecasts...')
    if price_changed or news_changed or recon_writes > 0:
        _step_precompute(db)
    else:
        logger.info('  No data changes this run; forecasts already current. Skip.')

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
