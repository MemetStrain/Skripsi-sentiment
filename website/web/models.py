"""
Models.py - Django Models
========================
This file only contains Django's built-in User model for authentication.

All business data (prices, news, market states) are stored directly in Firestore,
not in Django ORM models.

Note: Django's User model is still used for authentication as designed.
"""
from django.db import models

# Note: Django's built-in User model (django.contrib.auth.models.User) 
# is used for authentication and doesn't need to be defined here.
# 
# All other data (PriceHistory, News, MarketState) are stored in Firestore:
# - DailyMarketData collection (price history)
# - NewsData collection (news with sentiment)
# - MarketStates collection (HMM predictions)
