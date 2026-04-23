"""
cleanup_old_articles.py — Firestore cleanup utilities.

Commands:
    # Delete news_articles dated before 2014
    python cleanup_old_articles.py old-articles [--dry-run]

    # Delete the entire sentiment_aggregates collection
    python cleanup_old_articles.py sentiment-aggregates [--dry-run]
"""

import argparse
import logging
import sys
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger('cleanup')

BATCH_SIZE = 400  # stay well under Firestore's 500-op batch limit


def _init_firebase():
    import json
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON', '').strip()
    if creds_json:
        firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_json)))
        return

    gac = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if gac and os.path.exists(gac):
        firebase_admin.initialize_app(credentials.Certificate(gac))
        return

    for candidate in [
        os.path.join(script_dir, 'firebase-credentials.json'),
        os.path.join(script_dir, '..', 'website', 'firebase-credentials.json'),
    ]:
        if os.path.exists(candidate):
            firebase_admin.initialize_app(credentials.Certificate(os.path.abspath(candidate)))
            return

    firebase_admin.initialize_app()


def _delete_refs(db, refs: list, dry_run: bool, label: str) -> None:
    logger.info(f'Found {len(refs)} {label} documents to delete.')
    if dry_run:
        logger.info('Dry-run — no documents deleted.')
        return
    if not refs:
        logger.info('Nothing to delete.')
        return

    deleted = 0
    for i in range(0, len(refs), BATCH_SIZE):
        batch = db.batch()
        for ref in refs[i:i + BATCH_SIZE]:
            batch.delete(ref)
        batch.commit()
        deleted += len(refs[i:i + BATCH_SIZE])
        logger.info(f'  Deleted {deleted}/{len(refs)}...')
        time.sleep(0.5)

    logger.info(f'Done. Deleted {deleted} {label} documents.')


def delete_old_articles(db, dry_run: bool) -> None:
    cutoff = '2014-01-01'
    logger.info(f'Querying news_articles with date < {cutoff}...')
    refs = [
        doc.reference for doc in
        db.collection('news_articles').where('date', '<', cutoff).stream()
    ]
    _delete_refs(db, refs, dry_run, 'news_articles')


def delete_sentiment_aggregates(db, dry_run: bool) -> None:
    logger.info('Querying all sentiment_aggregates documents...')
    refs = [doc.reference for doc in db.collection('sentiment_aggregates').stream()]
    _delete_refs(db, refs, dry_run, 'sentiment_aggregates')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['old-articles', 'sentiment-aggregates'],
                        help='Which cleanup to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count documents without deleting them')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _init_firebase()

    from firebase_admin import firestore
    db = firestore.client()

    if args.command == 'old-articles':
        delete_old_articles(db, args.dry_run)
    elif args.command == 'sentiment-aggregates':
        delete_sentiment_aggregates(db, args.dry_run)


if __name__ == '__main__':
    main()
