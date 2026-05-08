"""
train_winners_csa.py — Phase C of the staged training workflow.

After Phase A (running all four C-scripts with --no-csa to get base-only
metrics) and Phase B (compute_winners.py picking the lowest-base-MAPE
config per horizon), this script CSA-optimises *only* the winning
ablation per horizon. That trims the CSA workload from 28 (4 × 7) runs
down to 7.

Usage:
    python prediction/train_winners_csa.py
        [--csa-population 50]
        [--csa-iterations 50]
        [--csa-cv-folds 3]
        [--dry-run]

After this finishes, re-run `compute_winners.py` so the dashboard's
metrics matrix picks up the new CSA cells.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

HERE         = os.path.dirname(os.path.abspath(__file__))
WINNERS_PATH = os.path.join(HERE, 'winners.json')

TAG_TO_SCRIPT: Dict[str, str] = {
    'cpo_only':      'horizon_forecast_C1_price_only.py',
    'cpo_hmm':       'horizon_forecast_C2_price_hmm.py',
    'cpo_sentiment': 'horizon_forecast_C3_price_sentiment.py',
    'full':          'horizon_forecast_C4_full.py',
}


def _load_winners() -> Dict[int, str]:
    if not os.path.exists(WINNERS_PATH):
        raise FileNotFoundError(
            f'{WINNERS_PATH} not found. Run prediction/compute_winners.py first '
            f'(after the four C-scripts have produced base-only outputs).'
        )
    with open(WINNERS_PATH, encoding='utf-8') as f:
        payload = json.load(f)
    raw = payload.get('winners_by_horizon', {})
    return {int(h): tag for h, tag in raw.items()}


def _group_winners_by_tag(winners: Dict[int, str]) -> Dict[str, List[int]]:
    by_tag: Dict[str, List[int]] = {}
    for h, tag in winners.items():
        by_tag.setdefault(tag, []).append(h)
    for tag in by_tag:
        by_tag[tag].sort()
    return by_tag


def main():
    parser = argparse.ArgumentParser(
        description='CSA-optimise only the winning ablation per horizon.')
    parser.add_argument('--csa-population', type=int, default=50)
    parser.add_argument('--csa-iterations', type=int, default=50)
    parser.add_argument('--csa-cv-folds',   type=int, default=3)
    parser.add_argument('--interval', type=str, default='daily', choices=['daily'])
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the planned commands without running them.')
    args = parser.parse_args()

    winners = _load_winners()
    if not winners:
        print('No winners recorded — nothing to do.')
        return

    by_tag = _group_winners_by_tag(winners)

    print('Phase C plan — CSA-optimise winning (tag, horizon) pairs:')
    for tag, hs in by_tag.items():
        print(f'  {tag:>14} → horizons {hs}')
    print()

    for tag, hs in by_tag.items():
        script = TAG_TO_SCRIPT.get(tag)
        if not script:
            print(f'!! Unknown tag {tag!r}, skipping.')
            continue

        cmd = [
            sys.executable, os.path.join(HERE, script),
            '--interval', args.interval,
            '--horizons', ','.join(map(str, hs)),
            '--csa-population', str(args.csa_population),
            '--csa-iterations', str(args.csa_iterations),
            '--csa-cv-folds',   str(args.csa_cv_folds),
        ]

        print(f'>>> {tag}: {" ".join(cmd)}')
        if args.dry_run:
            continue
        subprocess.check_call(cmd, cwd=HERE)
        print()

    print('Phase C complete.')
    print('Tip: re-run `python prediction/compute_winners.py` to refresh the '
          'metrics matrix with CSA cells included.')


if __name__ == '__main__':
    main()
