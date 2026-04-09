"""
Django settings — CPO Prediction (Vercel + Firestore edition)

- No database: auth lives in Firestore, sessions in signed cookies
- Firebase credentials loaded from FIREBASE_CREDENTIALS_JSON env var
  (falls back to firebase-credentials.json for local dev)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Security
# ------------------------------------------------------------------
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-%2a7sen4$v%o38t$gzw#kq3gvx(ct(usn95bd-qvta_jo9hg3a'
)

DEBUG = os.environ.get('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost 127.0.0.1').split()
ALLOWED_HOSTS += ['.vercel.app']

# ------------------------------------------------------------------
# Application definition
# ------------------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'web',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'web.auth_backend.FirestoreAuthMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.messages.context_processors.messages',
                'web.auth_backend.user_context_processor',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# ------------------------------------------------------------------
# Database — none (Firestore is used for all data)
# ------------------------------------------------------------------
DATABASES = {}

# ------------------------------------------------------------------
# Sessions — stored entirely in a signed cookie (no DB / file needed)
# ------------------------------------------------------------------
SESSION_ENGINE = 'django.contrib.sessions.backends.signed_cookies'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# ------------------------------------------------------------------
# Auth redirects (used by firestore_login_required decorator)
# ------------------------------------------------------------------
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'login'

# ------------------------------------------------------------------
# Internationalization
# ------------------------------------------------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kuala_Lumpur'
USE_I18N = True
USE_TZ = True

# ------------------------------------------------------------------
# Static files — served via CDN (Tailwind, Chart.js) so this is
# only needed for any local assets (favicon etc.)
# ------------------------------------------------------------------
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# ------------------------------------------------------------------
# Message storage — uses session (signed cookie), no DB needed
# ------------------------------------------------------------------
MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'
