"""
Management command to upload model input data and parameters to Firestore.

Uploads:
- CpoVariables: Technical features from cpo_variables_Daily.csv
- SentimentAggregate: Sentiment scores from sentiment_aggregate_Daily.csv
- HmmStatesResults: HMM market states from hmm_states_results_Daily.csv
- HorizonModelParameters: Per-horizon model hyperparameters from params JSON files
- HorizonModelMetrics: Per-horizon model metrics from results CSV files

Usage:
    python manage.py upload_model_data
    python manage.py upload_model_data --collection cpo_variables
    python manage.py upload_model_data --clear
"""

import os
import json
import time
import math
import pandas as pd
from django.core.management.base import BaseCommand
from firebase_admin import firestore


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

COLLECTIONS = {
    'cpo_variables': {
        'firestore_name': 'CpoVariables',
        'csv_path': os.path.join(PROJECT_ROOT, 'cpo', 'output', 'cpo_variables_Daily.csv'),
        'date_column': 'Date',
    },
    'sentiment_aggregate': {
        'firestore_name': 'SentimentAggregate',
        'csv_path': os.path.join(PROJECT_ROOT, 'news', 'output', 'sentiment_aggregate_Daily.csv'),
        'date_column': 'Date',
    },
    'hmm_states_results': {
        'firestore_name': 'HmmStatesResults',
        'csv_path': os.path.join(PROJECT_ROOT, 'markov', 'output', 'hmm_states_results_Daily.csv'),
        'date_column': 'Date',
    },
}

HORIZONS_DIR = os.path.join(PROJECT_ROOT, 'prediction', 'output_horizons', 'Daily')
DAILY_HORIZONS = [1, 2, 3, 4, 5, 6, 7]


class Command(BaseCommand):
    help = 'Upload model input data and parameters to Firestore'

    def add_arguments(self, parser):
        parser.add_argument(
            '--collection',
            type=str,
            choices=list(COLLECTIONS.keys()) + ['horizon_params', 'horizon_metrics', 'all'],
            default='all',
            help='Which collection to upload (default: all)',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Delete existing documents before uploading',
        )

    def handle(self, *args, **options):
        db = firestore.client()
        collection = options['collection']
        clear = options['clear']

        if collection == 'all':
            targets = list(COLLECTIONS.keys()) + ['horizon_params', 'horizon_metrics']
        else:
            targets = [collection]

        for target in targets:
            if target in COLLECTIONS:
                self._upload_csv(db, target, clear)
            elif target == 'horizon_params':
                self._upload_horizon_params(db, clear)
            elif target == 'horizon_metrics':
                self._upload_horizon_metrics(db, clear)

        self.stdout.write(self.style.SUCCESS('Upload complete.'))

    def _upload_csv(self, db, collection_key, clear):
        config = COLLECTIONS[collection_key]
        collection_name = config['firestore_name']
        csv_path = config['csv_path']
        date_col = config['date_column']

        self.stdout.write(f'\nUploading {collection_key} -> {collection_name}')
        self.stdout.write(f'  Source: {csv_path}')

        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'  File not found: {csv_path}'))
            return

        df = pd.read_csv(csv_path)
        self.stdout.write(f'  Rows: {len(df)}')

        if clear:
            self._clear_collection(db, collection_name)

        # Upload in batches of 450 (Firestore limit is 500)
        batch_size = 450
        total = len(df)
        uploaded = 0
        start_time = time.time()

        for chunk_start in range(0, total, batch_size):
            chunk_end = min(chunk_start + batch_size, total)
            batch = db.batch()

            for idx in range(chunk_start, chunk_end):
                row = df.iloc[idx]
                date_str = str(row[date_col])[:10]  # YYYY-MM-DD
                doc_ref = db.collection(collection_name).document(date_str)

                doc_data = {}
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        doc_data[col] = None
                    elif isinstance(val, (int, float)):
                        if math.isnan(val) or math.isinf(val):
                            doc_data[col] = None
                        else:
                            doc_data[col] = float(val)
                    else:
                        doc_data[col] = str(val)

                batch.set(doc_ref, doc_data)

            batch.commit()
            uploaded += (chunk_end - chunk_start)
            self.stdout.write(f'  Uploaded {uploaded}/{total}')

        elapsed = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f'  Done: {uploaded} docs in {elapsed:.1f}s'))

    def _upload_horizon_params(self, db, clear):
        collection_name = 'HorizonModelParameters'
        self.stdout.write(f'\nUploading horizon parameters -> {collection_name}')

        if clear:
            self._clear_collection(db, collection_name)

        for h in DAILY_HORIZONS:
            params_path = os.path.join(
                HORIZONS_DIR, f'horizon_{h}', f'params_Daily_h{h}.json')

            if not os.path.exists(params_path):
                self.stdout.write(self.style.WARNING(
                    f'  Skipping h{h}: {params_path} not found'))
                continue

            with open(params_path, 'r') as f:
                params = json.load(f)

            doc_id = f'Daily_h{h}'
            doc_ref = db.collection(collection_name).document(doc_id)
            doc_ref.set(params)
            self.stdout.write(f'  Uploaded {doc_id}')

        self.stdout.write(self.style.SUCCESS('  Done: horizon parameters'))

    def _upload_horizon_metrics(self, db, clear):
        collection_name = 'HorizonModelMetrics'
        self.stdout.write(f'\nUploading horizon metrics -> {collection_name}')

        if clear:
            self._clear_collection(db, collection_name)

        for h in DAILY_HORIZONS:
            results_path = os.path.join(
                HORIZONS_DIR, f'horizon_{h}', f'results_Daily_h{h}.csv')

            if not os.path.exists(results_path):
                self.stdout.write(self.style.WARNING(
                    f'  Skipping h{h}: {results_path} not found'))
                continue

            df = pd.read_csv(results_path)
            metrics_list = []
            for _, row in df.iterrows():
                metrics_list.append({
                    'model': row['Model'],
                    'optimization': row['Optimization'],
                    'mape': float(row['MAPE']),
                    'rmse': float(row['RMSE']),
                    'directional_accuracy': float(row['Directional_Accuracy']),
                    'r2': float(row['R2']),
                })

            doc_id = f'Daily_h{h}'
            doc_ref = db.collection(collection_name).document(doc_id)
            doc_ref.set({
                'interval': 'Daily',
                'horizon': h,
                'metrics': metrics_list,
            })
            self.stdout.write(f'  Uploaded {doc_id} ({len(metrics_list)} models)')

        self.stdout.write(self.style.SUCCESS('  Done: horizon metrics'))

    def _clear_collection(self, db, collection_name):
        self.stdout.write(f'  Clearing {collection_name}...')
        docs = db.collection(collection_name).limit(500).stream()
        deleted = 0
        while True:
            batch = db.batch()
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            if count == 0:
                break
            batch.commit()
            deleted += count
            docs = db.collection(collection_name).limit(500).stream()
        if deleted:
            self.stdout.write(f'  Deleted {deleted} existing docs')
