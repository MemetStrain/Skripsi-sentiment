"""Print the distinct hmm_states labels in Firestore (Daily) and a sample
of dates per label, so we can see whether stale N>3 docs are mixed in with
the current 3-state run.

Run with:  python debug_hmm_labels.py
"""
import os
from collections import defaultdict

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from firebase_admin import firestore
db = firestore.client()

by_label = defaultdict(list)
by_state = defaultdict(int)
for doc in db.collection('hmm_states').stream():
    d = doc.to_dict() or {}
    if d.get('frequency') not in (None, 'Daily'):
        continue
    label = d.get('state_label', '?')
    state = d.get('state', '?')
    by_label[label].append(d.get('date'))
    by_state[state] += 1

print('Distinct state_label values + count + first/last date:')
for label, dates in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
    dates_sorted = sorted([d for d in dates if d])
    first = dates_sorted[0] if dates_sorted else '?'
    last  = dates_sorted[-1] if dates_sorted else '?'
    print(f'  {label:<20s}  n={len(dates):>5}  first={first}  last={last}')

print('\nDistinct state integers + count:')
for s, n in sorted(by_state.items()):
    print(f'  state={s!r}  n={n}')
