from django.apps import AppConfig


class WebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = "web"
    
    def ready(self):
        """
        Initialize Firebase when Django app starts
        This ensures Firestore is available in all views
        """
        import firebase_admin
        from firebase_admin import credentials
        from django.conf import settings
        import os
        
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            # Get credentials path
            cred_path = os.path.join(settings.BASE_DIR, 'firebase-credentials.json')
            
            if os.path.exists(cred_path):
                # Initialize with credentials file
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized successfully")
            else:
                # Try to initialize with default credentials (for production)
                try:
                    firebase_admin.initialize_app()
                    print("✅ Firebase initialized with default credentials")
                except Exception as e:
                    print(f"⚠️ Firebase initialization failed: {e}")
                    print("Note: Firebase features will not work without proper credentials")
