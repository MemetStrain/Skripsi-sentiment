"""
migrate_hmm_to_firestore.py — one-shot migration to the frozen-params HMM.

Runs the wipe + rebuild flow:
  1. Load fitted HMM parameters from markov/output/hmm_params_Daily.json and
     publish them to Firestore `hmm_models/Daily`.
  2. Wipe every existing `hmm_states` doc (those came from the old refit-daily
     pipeline that used Viterbi smoothing).
  3. Re-import all states from markov/output/hmm_states_results_Daily.csv,
     which was produced by the causal offline pipeline.

After this runs, the daily scheduler (hmm_updater.update_hmm_states) only
appends new dates and never touches historical rows.

Usage:
    python migrate_hmm_to_firestore.py
    python migrate_hmm_to_firestore.py --params-json ../markov/output/hmm_params_Daily.json \
                                       --states-csv  ../markov/output/hmm_states_results_Daily.csv
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger('migrate_hmm')


_BASE = os.path.dirname(os.path.abspath(__file__))


def _default(rel_path: str) -> str:
    return os.path.abspath(os.path.join(_BASE, rel_path))


def _load_params(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'HMM params JSON not found at {path}. Run markov/cpo_hmm_states.py first.'
        )
    with open(path) as fh:
        params = json.load(fh)
    required = {'feat_cols', 'startprob_', 'transmat_', 'means_', 'covars_',
                'state_to_label', 'covariance_type', 'n_components',
                'volatility_window', 'norm_window'}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f'HMM params missing keys: {sorted(missing)}')
    return params


def _states_from_csv(path: str, frequency: str = 'Daily') -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'HMM states CSV not found at {path}. Run markov/cpo_hmm_states.py first.'
        )
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'date':        r['Date'].strftime('%Y-%m-%d'),
            'frequency':   frequency,
            'state':       int(r['State']),
            'state_label': str(r['State_Label']),
            'log_return':  round(float(r['Log_Return']), 6) if 'Log_Return' in df.columns else 0.0,
            'volatility':  round(float(r['Volatility']),  6) if 'Volatility' in df.columns else 0.0,
            'rsi':         round(float(r['RSI']),         4) if 'RSI'        in df.columns else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Migrate to frozen-params HMM: publish params + wipe + rebuild states.'
    )
    parser.add_argument('--params-json', default=_default('../markov/output/hmm_params_Daily.json'),
                        help='Path to offline-trained HMM params JSON.')
    parser.add_argument('--states-csv',  default=_default('../markov/output/hmm_states_results_Daily.csv'),
                        help='Path to offline-trained HMM states CSV.')
    parser.add_argument('--frequency',   default='Daily', choices=['Daily'])
    parser.add_argument('--skip-wipe',   action='store_true',
                        help='Do not delete existing hmm_states docs before writing.')
    args = parser.parse_args()

    # Defer Firebase imports so --help works without creds.
    try:
        from main import init_firebase
        init_firebase()
    except Exception as e:
        logger.error(f'Firebase initialisation failed: {e}')
        sys.exit(1)

    from firebase_admin import firestore
    db = firestore.client()

    from firestore_writer import (
        write_hmm_params, wipe_hmm_states, write_hmm_states_batch,
    )

    # --- 1. Publish params ----------------------------------------------------
    logger.info(f'Loading HMM params from {args.params_json}')
    params = _load_params(args.params_json)
    write_hmm_params(db, args.frequency, params)

    # --- 2. Wipe existing states ---------------------------------------------
    if args.skip_wipe:
        logger.info('Skipping wipe — existing hmm_states docs left in place.')
    else:
        logger.info(f'Wiping existing hmm_states docs (frequency={args.frequency})...')
        wipe_hmm_states(db, args.frequency)

    # --- 3. Re-import states from offline CSV --------------------------------
    logger.info(f'Loading offline states from {args.states_csv}')
    states = _states_from_csv(args.states_csv, args.frequency)
    logger.info(f'Writing {len(states)} state docs...')
    write_hmm_states_batch(db, states)

    logger.info('=== HMM migration complete ===')
    logger.info('  hmm_models/Daily      → frozen params written')
    logger.info(f'  hmm_states (Daily)    → {len(states)} docs')
    logger.info('Daily scheduler (hmm_updater.update_hmm_states) will now only')
    logger.info('append new dates — historical states are immutable under frozen params.')


if __name__ == '__main__':
    main()
