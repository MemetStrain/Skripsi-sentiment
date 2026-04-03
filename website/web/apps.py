import os
import json
from django.apps import AppConfig


class WebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'web'

    def ready(self):
        """Initialize Firebase on startup. Credentials come from:
        1. FIREBASE_CREDENTIALS_JSON env var (production / Vercel)
        2. firebase-credentials.json file (local dev fallback)
        """
        import firebase_admin
        from firebase_admin import credentials
        from django.conf import settings

        if firebase_admin._apps:
            return  # already initialized

        creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON', '').strip()
        if creds_json:
            try:
                creds_dict = json.loads(creds_json)
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred)
                return
            except Exception as e:
                print(f'Firebase init from env var failed: {e}')

        cred_path = os.path.join(settings.BASE_DIR, 'firebase-credentials.json')
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            return

        print('WARNING: Firebase credentials not found. Firestore features will not work.')
