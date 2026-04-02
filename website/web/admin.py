"""
Admin.py - Django Admin Configuration
=====================================
This file only registers Django's built-in User model for admin interface.

All business data (prices, news, market states) are managed through:
- Custom views (admin_upload_price)
- Firestore console
- Management commands

Note: Django admin is only used for user management.
"""
from django.contrib import admin

# Django's built-in User model is automatically registered by default
# No custom models need to be registered here since all data is in Firestore
