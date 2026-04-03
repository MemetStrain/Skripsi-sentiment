"""
Vercel entry point — @vercel/python adds this file's directory (website/)
to sys.path, making `config.settings` importable as website/config/settings.py.
"""
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
