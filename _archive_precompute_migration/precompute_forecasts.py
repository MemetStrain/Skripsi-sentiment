"""
precompute_forecasts.py — offline forecast precomputation.

Runs the live XGBoost inference pipeline (predictor.compute_forecast_trails)
and writes the result to Firestore at `forecasts/latest` as a compact JSON
string. The Vercel-hosted website then serves forecasts by simply reading
that document — it never bundles or runs the ~930 MB ML stack.

Run locally (or on a schedule) after daily_prices / hmm_states /
sentiment_aggregates have been refreshed:

    # one-time: install the offline-only ML deps
    pip install -r requirements.txt -r requirements-ml.txt

    python precompute_forecasts.py

Credentials: FIREBASE_CREDENTIALS_JSON env var, else firebase-credentials.json.
"""
import json
import os
import sys

import firebase_admin
from firebase_admin import credentials, firestore

_HERE = os.path.dirname(os.path.abspath(__file__))   # website/
sys.path.insert(0, os.path.join(_HERE, 'web'))       # so `import predictor` resolves

from predictor import compute_forecast_trails        # noqa: E402


def _init_firebase() -> None:
    if firebase_admin._apps:
        return
    creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON', '').strip()
    if creds_json:
        firebase_admin.initialize_app(credentials.Certificate(json.loads(creds_json)))
        return
    cred_path = os.path.join(_HERE, 'firebase-credentials.json')
    if os.path.exists(cred_path):
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
        return
    raise SystemExit(
        'Firebase credentials not found — set FIREBASE_CREDENTIALS_JSON '
        'or place firebase-credentials.json next to this script.'
    )


def main() -> None:
    _init_firebase()
    db = firestore.client()

    print('Running XGBoost inference (compute_forecast_trails)…')
    payload = compute_forecast_trails(db, max_horizon=7, window_days=90)

    trails = payload.get('trails', [])
    n_points = sum(len(t.get('points', [])) for t in trails)
    print(f'  -> {len(trails)} trails, {n_points} forecast points')

    doc = {
        'payload': json.dumps(payload, separators=(',', ':')),
        'generated_at': payload.get('generated_at'),
    }
    size_kb = len(doc['payload']) / 1024
    if size_kb > 1000:
        raise SystemExit(
            f'Payload is {size_kb:.0f} KB — exceeds the 1 MB Firestore '
            f'document limit. Reduce window_days or store per-horizon docs.'
        )

    db.collection('forecasts').document('latest').set(doc)
    print(f'Wrote forecasts/latest to Firestore ({size_kb:.0f} KB).')


if __name__ == '__main__':
    main()
