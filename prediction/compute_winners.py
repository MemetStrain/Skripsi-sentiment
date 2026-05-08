"""
compute_winners.py — pick the best ablation config per horizon and emit
prediction/winners.json for the website to consume.

For each horizon h ∈ {1..7}, this script:
  1. Reads testing_results_Daily_h{h}.csv from each ablation's output dir
     (prediction/output_horizons/{tag}/Daily/horizon_{h}/), where
     tag ∈ {cpo_only, cpo_hmm, cpo_sentiment, full}.
  2. Picks the tag with the lowest BASE-variant MAPE — that is the winner.
  3. The website then loads the winner's CSA model
     (prediction/saved_models/{winner_tag}/Daily/h{h}/xgboost_csa/) for
     live inference.

The full 4×7 metrics matrix (BASE + CSA) is also embedded in the JSON so
the dashboard can render the comparison table without re-reading every CSV.

Usage:
    python prediction/compute_winners.py
"""

import csv
import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE   = os.path.join(PROJECT_ROOT, 'output_horizons')
WINNERS_PATH  = os.path.join(PROJECT_ROOT, 'winners.json')

TAGS     = ['cpo_only', 'cpo_hmm', 'cpo_sentiment', 'full']
HORIZONS = [1, 2, 3, 4, 5, 6, 7]
INTERVAL = 'Daily'

# Map ablation tag → user-facing config name.
TAG_TO_CONFIG = {
    'cpo_only':      'C1',
    'cpo_hmm':       'C2',
    'cpo_sentiment': 'C3',
    'full':          'C4',
}


def _testing_results_path(tag: str, horizon: int) -> str:
    return os.path.join(
        OUTPUT_BASE, tag, INTERVAL, f'horizon_{horizon}',
        f'testing_results_{INTERVAL}_h{horizon}.csv',
    )


def _read_metrics(tag: str, horizon: int) -> Optional[Dict[str, Dict[str, float]]]:
    """Return {'BASE': {...}, 'CSA': {...}} or None if file missing."""
    path = _testing_results_path(tag, horizon)
    if not os.path.exists(path):
        logger.warning(f'Missing: {path}')
        return None

    out: Dict[str, Dict[str, float]] = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Model') != 'xgboost':
                continue
            opt = row.get('Optimization', '').upper()
            if opt not in ('BASE', 'CSA'):
                continue
            out[opt] = {
                'mape':    _to_float(row.get('MAPE')),
                'smape':   _to_float(row.get('sMAPE')),
                'rmse':    _to_float(row.get('RMSE')),
                'da':      _to_float(row.get('Directional_Accuracy')),
                'r2_price':     _to_float(row.get('R2_Price')),
                'r2_logreturn': _to_float(row.get('R2_LogReturn')),
            }
    return out or None


def _to_float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, '') else None
    except (TypeError, ValueError):
        return None


def compute() -> dict:
    """
    Build the winners payload.

    Returns dict with keys:
      generated_at        ISO timestamp string
      horizons            [1..7]
      tags                ablation tag list
      winners_by_horizon  {"1": "cpo_hmm", ...} (winning tag)
      configs_by_horizon  {"1": "C2", ...}      (user-facing label)
      metrics             nested: {tag: {horizon: {'BASE': {...}, 'CSA': {...}}}}
    """
    from datetime import datetime, timezone

    metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}
    for tag in TAGS:
        metrics[tag] = {}
        for h in HORIZONS:
            row = _read_metrics(tag, h)
            if row is not None:
                metrics[tag][h] = row

    winners: Dict[int, str] = {}
    configs: Dict[int, str] = {}
    for h in HORIZONS:
        candidates = []
        for tag in TAGS:
            base = metrics.get(tag, {}).get(h, {}).get('BASE')
            if base and base.get('mape') is not None:
                candidates.append((tag, base['mape']))
        if not candidates:
            logger.warning(f'No candidates for horizon {h}; skipping winner pick.')
            continue
        candidates.sort(key=lambda x: x[1])
        winners[h] = candidates[0][0]
        configs[h] = TAG_TO_CONFIG[candidates[0][0]]

    return {
        'generated_at':        datetime.now(timezone.utc).isoformat(),
        'horizons':            HORIZONS,
        'tags':                TAGS,
        'tag_to_config':       TAG_TO_CONFIG,
        'winners_by_horizon':  {str(h): t for h, t in winners.items()},
        'configs_by_horizon':  {str(h): c for h, c in configs.items()},
        'metrics':             {
            tag: {str(h): m for h, m in by_h.items()}
            for tag, by_h in metrics.items()
        },
    }


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)-8s  %(message)s')
    payload = compute()

    with open(WINNERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'Wrote {WINNERS_PATH}')
    print(f'Winners by horizon: {payload["winners_by_horizon"]}')
    print(f'Configs by horizon: {payload["configs_by_horizon"]}')


if __name__ == '__main__':
    main()
