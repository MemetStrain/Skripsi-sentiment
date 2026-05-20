"""
precompute_forecasts.py — final scheduler phase that materialises the rolling
XGBoost forecast trail into Firestore so the Vercel-hosted site can serve
forecasts by reading docs instead of running joblib/numpy/xgboost at request
time.

Design note — full recompute, not incremental
---------------------------------------------
Every run regenerates the entire trailing PRECOMPUTE_WINDOW_DAYS window from
scratch rather than appending only the new anchor. This is intentional:

* It is a once-daily LOCAL batch — ~365 anchors x 7 horizons is seconds of
  XGBoost inference, so the per-request efficiency argument that motivated
  incremental updates in the old live path no longer applies.
* Deterministic doc IDs (`Daily_h{h}_{anchor}`) make the recompute idempotent
  and self-healing: a skipped run or an externally-mutated Firestore doc is
  corrected on the next run with no special-case logic.

Incremental append is explicitly out of scope.

This module is a thin orchestrator: it CALLS `inference.compute_forecast_trails`
verbatim (the same function the old website used) and only persists the result.
No inference math lives here.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List

from firestore_writer import write_forecasts, write_forecast_meta

logger = logging.getLogger(__name__)

PRECOMPUTE_WINDOW_DAYS = 365
PRECOMPUTE_MAX_HORIZON = 7
FREQUENCY = 'Daily'

# Legacy single-doc forecast store written by the previous
# website/precompute_forecasts.py path. Superseded by the per-point
# `forecasts` collection + `forecast_meta/Daily`; deleted after a
# successful write so it cannot serve a stale payload to the site.
_LEGACY_DOC = ('forecasts', 'latest')


def _import_inference():
    """Import `inference` from the sibling prediction/ package.

    Adds prediction/ to sys.path so `inference.py`'s top-level
    `from feature_engineering import ...` resolves the same way the
    C{1..4} training scripts do.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.abspath(os.path.join(here, '..', 'prediction'))
    if pred_dir not in sys.path:
        sys.path.insert(0, pred_dir)
    import inference  # noqa: E402
    return inference


def _flatten_trails(trails: List[Dict]) -> List[Dict]:
    """Flatten compute_forecast_trails' nested trails -> per-point scalar dicts.

    The shape `compute_forecast_trails` returns groups points under each
    trail (one trail per horizon). Firestore docs are scalar-only, so we
    pull the trail's horizon/tag/config onto each point and produce a flat
    list ready for `write_forecasts`.
    """
    points: List[Dict] = []
    for trail in trails:
        horizon = int(trail['horizon'])
        tag     = str(trail.get('tag', ''))
        config  = str(trail.get('config', ''))
        for p in trail.get('points', []):
            points.append({
                'frequency':       FREQUENCY,
                'horizon':         horizon,
                'tag':             tag,
                'config':          config,
                'anchor_date':     str(p['anchor_date']),
                'anchor_price':    float(p.get('anchor_price', 0.0)),
                'predicted_date':  str(p['predicted_date']),
                'predicted_price': float(p['predicted_price']),
                'log_return':      float(p.get('log_return', 0.0)),
            })
    return points


def _delete_legacy_doc(db) -> None:
    """Remove the old single-document forecast store, if present.

    Safe to call repeatedly; missing doc is a no-op.
    """
    coll, doc_id = _LEGACY_DOC
    try:
        ref = db.collection(coll).document(doc_id)
        if ref.get().exists:
            ref.delete()
            logger.info(f'  Deleted legacy doc {coll}/{doc_id}.')
    except Exception as e:
        logger.warning(f'  Could not delete legacy {coll}/{doc_id}: {e}')


def precompute_and_write(db) -> int:
    """Run XGBoost inference and persist the trail to Firestore.

    Returns the number of forecast point documents written (0 if the
    inference frame was empty or inference produced no trails).
    """
    inference = _import_inference()

    logger.info(
        f'  Running compute_forecast_trails(max_horizon={PRECOMPUTE_MAX_HORIZON}, '
        f'window_days={PRECOMPUTE_WINDOW_DAYS})...'
    )
    payload = inference.compute_forecast_trails(
        db,
        max_horizon=PRECOMPUTE_MAX_HORIZON,
        window_days=PRECOMPUTE_WINDOW_DAYS,
    )

    trails = payload.get('trails', [])
    if not trails:
        logger.warning('  compute_forecast_trails returned no trails; nothing written.')
        return 0

    points = _flatten_trails(trails)
    if not points:
        logger.warning('  Trails contained no points; nothing written.')
        return 0

    n = write_forecasts(db, points)

    write_forecast_meta(db, {
        'frequency':          FREQUENCY,
        'generated_at':       payload.get('generated_at', ''),
        'max_horizon':        PRECOMPUTE_MAX_HORIZON,
        'window_days':        PRECOMPUTE_WINDOW_DAYS,
        'winners_by_horizon': payload.get('winners', {}),
        'configs_by_horizon': payload.get('configs', {}),
        'metrics':            payload.get('metrics', {}),
        'tag_to_config':      payload.get('tag_to_config', {}),
        'horizons':           payload.get('horizons', []),
    })

    _delete_legacy_doc(db)

    logger.info(f'  Forecast precompute complete: {n} point docs + forecast_meta/{FREQUENCY}.')
    return n
