"""One-shot diagnostic: print the latest 10 daily_prices docs as the dashboard
view sees them. Run with: python debug_dailyprices.py"""
import os
import sys
from datetime import datetime, timedelta

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from firebase_admin import firestore
db = firestore.client()

three_months_ago = (datetime.now().date() - timedelta(days=90)).isoformat()
print(f'Lower bound (>= {three_months_ago})')
print(f'Today: {datetime.now().date().isoformat()}')
print('-' * 60)

# Same query the dashboard view uses, then take the last 10 by date.
docs = list(
    db.collection('daily_prices')
    .where('date', '>=', three_months_ago)
    .order_by('date')
    .stream()
)
print(f'Total docs returned by view query: {len(docs)}')
print('Last 10 by date:')
for d in docs[-10:]:
    data = d.to_dict()
    print(f'  doc_id={d.id}  date={data.get("date")!r}  '
          f'close={data.get("close")}  updated_at={data.get("updated_at")}')

# Also show top-5 by raw doc-id ordering, in case some recent docs have a
# missing/typed-wrong `date` field that excludes them from the where+order_by.
print('\nTop 5 by doc-id descending (sanity check, no filter):')
all_docs = list(db.collection('daily_prices').stream())
all_docs.sort(key=lambda d: d.id, reverse=True)
for d in all_docs[:5]:
    data = d.to_dict()
    print(f'  doc_id={d.id}  date_field={data.get("date")!r}  '
          f'date_type={type(data.get("date")).__name__}  close={data.get("close")}')
