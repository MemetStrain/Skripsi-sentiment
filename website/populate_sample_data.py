r"""
populate_sample_data.py - Populate Firestore with Sample Data
==============================================================
Script untuk mengisi Firestore dengan data sample untuk testing.

Cara Menjalankan:
-----------------
1. Pastikan virtual environment sudah aktif:
   venv\Scripts\activate

2. Jalankan script:
   python populate_sample_data.py

Note: Script ini akan populate 3 collections di Firestore:
- DailyMarketData (90 hari data harga CPO)
- MarketStates (90 hari prediksi HMM states)
- NewsData (30 artikel berita dengan sentiment analysis)
"""

import os
import django
from datetime import datetime, timedelta
import random

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Import Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate('firebase-credentials.json')
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("🚀 Starting sample data population to Firestore...")
print("=" * 60)

# ============================================================================
# 1. POPULATE DAILY MARKET DATA (Price History)
# ============================================================================
print("\n📊 Populating DailyMarketData collection...")

start_date = datetime.now().date() - timedelta(days=90)
prices_created = 0

batch = db.batch()
batch_count = 0

for i in range(90):
    date = start_date + timedelta(days=i)
    base_price = 3500 + random.uniform(-200, 200)
    
    doc_ref = db.collection('DailyMarketData').document(date.strftime('%Y-%m-%d'))
    
    batch.set(doc_ref, {
        'date': date.strftime('%Y-%m-%d'),
        'open': round(base_price + random.uniform(-20, 20), 2),
        'high': round(base_price + random.uniform(0, 50), 2),
        'low': round(base_price - random.uniform(0, 50), 2),
        'close': round(base_price, 2),
        'volume': round(random.uniform(1000000, 2000000), 2),
        'created_at': firestore.SERVER_TIMESTAMP
    })
    
    prices_created += 1
    batch_count += 1
    
    # Firestore batch limit is 500, commit every 400 to be safe
    if batch_count >= 400:
        batch.commit()
        batch = db.batch()
        batch_count = 0

# Commit remaining
if batch_count > 0:
    batch.commit()

print(f"✅ Created {prices_created} price records in DailyMarketData collection")

# ============================================================================
# 2. POPULATE MARKET STATES (HMM Predictions)
# ============================================================================
print("\n🎯 Populating MarketStates collection...")

states_created = 0
batch = db.batch()
batch_count = 0

for i in range(90):
    date = start_date + timedelta(days=i)
    # Change state every ~15 days to simulate market regime changes
    # 0 = Bearish, 1 = Bullish, 2 = Neutral
    state = (i // 15) % 3
    
    doc_ref = db.collection('MarketStates').document(date.strftime('%Y-%m-%d'))
    
    batch.set(doc_ref, {
        'date': date.strftime('%Y-%m-%d'),
        'state': state,
        'probability': round(random.uniform(0.7, 0.95), 2),
        'created_at': firestore.SERVER_TIMESTAMP
    })
    
    states_created += 1
    batch_count += 1
    
    if batch_count >= 400:
        batch.commit()
        batch = db.batch()
        batch_count = 0

# Commit remaining
if batch_count > 0:
    batch.commit()

print(f"✅ Created {states_created} market state records in MarketStates collection")

# ============================================================================
# 3. POPULATE NEWS DATA (News with Sentiment Analysis)
# ============================================================================
print("\n📰 Populating NewsData collection...")

news_titles = [
    "CPO Prices Rise Amid Strong Demand from China",
    "Palm Oil Export Regulations Tightened by Indonesian Government",
    "Weather Concerns Impact CPO Production Forecast",
    "Global Vegetable Oil Market Shows Positive Outlook",
    "CPO Futures Hit Monthly High on Supply Concerns",
    "Indonesia Plans to Increase Biodiesel Blend Mandate",
    "Malaysia Reports Lower Palm Oil Stock Levels",
    "Trade Tensions Affect CPO Export to Major Markets",
    "Sustainable Palm Oil Certification Gains Momentum",
    "CPO Market Stabilizes After Volatile Trading Session",
    "New Technology Boosts Palm Oil Mill Efficiency",
    "Environmental Groups Raise Concerns Over Deforestation",
    "CPO Prices Drop on Weaker Crude Oil Market",
    "India Increases Palm Oil Import Quotas",
    "ASEAN Countries Discuss CPO Trade Cooperation",
    "Weather Improves, CPO Production Expected to Increase",
    "CPO Spot Prices Show Upward Trend This Week",
    "European Union Reviews Palm Oil Import Policies",
    "CPO Industry Invests in Renewable Energy Projects",
    "Market Analysis: CPO Price Outlook for Next Quarter",
    "Indonesia Records Strong CPO Export Growth",
    "Palm Oil Plantation Expansion Slows Down",
    "CPO Price Volatility Concerns Industry Stakeholders",
    "New CPO Processing Plant Opens in Sumatra",
    "Global Food Crisis Boosts Demand for Palm Oil",
    "CPO Market Faces Headwinds from Currency Fluctuations",
    "Sustainable Palm Oil Initiative Launched by Industry Leaders",
    "CPO Prices Supported by Strong Biodiesel Demand",
    "Heavy Rainfall Affects Palm Oil Harvesting Operations",
    "CPO Trading Volume Reaches New Record High"
]

sentiments = ['Positive', 'Negative', 'Neutral']
news_created = 0

batch = db.batch()
batch_count = 0

for i in range(30):
    date = datetime.now() - timedelta(days=i*2)
    sentiment = random.choice(sentiments)
    
    # Generate realistic sentiment scores based on label
    if sentiment == 'Positive':
        score = round(random.uniform(0.3, 0.9), 3)
    elif sentiment == 'Negative':
        score = round(random.uniform(-0.9, -0.3), 3)
    else:  # Neutral
        score = round(random.uniform(-0.2, 0.2), 3)
    
    title = news_titles[i]
    
    # Use auto-generated ID for news
    doc_ref = db.collection('NewsData').document()
    
    batch.set(doc_ref, {
        'title': title,
        'date': date.strftime('%Y-%m-%d'),
        'snippet': f"This article discusses recent developments in the CPO market. {title}. Market analysts provide insights on the implications for commodity prices and trading activities.",
        'sentiment_score': score,
        'sentiment_label': sentiment,
        'url': f"https://example.com/news/cpo-{i+1}",
        'created_at': firestore.SERVER_TIMESTAMP
    })
    
    news_created += 1
    batch_count += 1
    
    if batch_count >= 400:
        batch.commit()
        batch = db.batch()
        batch_count = 0

# Commit remaining
if batch_count > 0:
    batch.commit()

print(f"✅ Created {news_created} news articles in NewsData collection")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("🎉 Sample data population complete!")
print("=" * 60)
print(f"📊 Total records created in Firestore:")
print(f"   - DailyMarketData: {prices_created} documents")
print(f"   - MarketStates: {states_created} documents")
print(f"   - NewsData: {news_created} documents")
print("\n🌐 You can now view the dashboard at: http://localhost:8000/")
print("=" * 60)
