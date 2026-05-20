"""
winners.py — lightweight reader for prediction/winners.json.

Split out of predictor.py so the Vercel-hosted site can read the
metrics table without importing joblib / numpy / pandas / xgboost.
"""
import json
import os
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_WINNERS_PATH = os.path.abspath(
    os.path.join(_HERE, '..', '..', 'prediction', 'winners.json')
)


@lru_cache(maxsize=1)
def load_winners() -> dict:
    """Return the parsed winners.json payload (cached for the process lifetime)."""
    if not os.path.exists(_WINNERS_PATH):
        raise FileNotFoundError(
            f'winners.json not found at {_WINNERS_PATH}. '
            f'Run prediction/compute_winners.py after the C{{1..4}} training scripts.'
        )
    with open(_WINNERS_PATH, encoding='utf-8') as f:
        return json.load(f)
