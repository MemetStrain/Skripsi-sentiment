"""Show the n_components and state_to_label currently in Firestore's
hmm_models/Daily, and compare to the local markov/output/hmm_params_Daily.json.
"""
import json
import os

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from firebase_admin import firestore
db = firestore.client()

snap = db.collection('hmm_models').document('Daily').get()
if not snap.exists:
    print('hmm_models/Daily does NOT exist in Firestore.')
else:
    raw = snap.to_dict() or {}
    if 'payload_json' in raw:
        params = json.loads(raw['payload_json'])
    else:
        params = raw
    print('Firestore hmm_models/Daily:')
    print(f'  n_components   = {params.get("n_components")}')
    print(f'  state_to_label = {params.get("state_to_label")}')
    print(f'  fit_timestamp  = {params.get("fit_timestamp")}')

print()

local = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                     'markov', 'output', 'hmm_params_Daily.json'))
if os.path.exists(local):
    with open(local) as f:
        lp = json.load(f)
    print(f'Local {local}:')
    print(f'  n_components   = {lp.get("n_components")}')
    print(f'  state_to_label = {lp.get("state_to_label")}')
    print(f'  fit_timestamp  = {lp.get("fit_timestamp")}')
else:
    print(f'Local file not found: {local}')
