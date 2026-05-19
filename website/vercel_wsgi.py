"""
Vercel entry point — explicitly adds website/ to sys.path so that
`config.settings` (website/config/settings.py) is importable.
"""
import os
import sys

# Ensure the website/ directory is on the path so 'config' module is found
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Vercel's @vercel/python builder auto-detects a WSGI callable named `app`.
app = application
