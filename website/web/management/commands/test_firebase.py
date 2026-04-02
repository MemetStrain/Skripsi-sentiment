"""
Django Management Command: Test Firebase Connection
Usage: python manage.py test_firebase
"""
from django.core.management.base import BaseCommand
from web.firebase_backend import FirebaseConnection, FirestorePriceHistory
from datetime import date
import os


class Command(BaseCommand):
    help = 'Test koneksi ke Firebase Firestore'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🔥 Testing Firebase Connection...\n'))
        
        # Check credentials file
        cred_path = os.path.join('firebase-credentials.json')
        if not os.path.exists(cred_path):
            self.stdout.write(self.style.ERROR(
                '❌ Firebase credentials not found!\n\n'
                'Follow these steps:\n'
                '1. Go to Firebase Console: https://console.firebase.google.com/\n'
                '2. Select your project\n'
                '3. Settings → Service Accounts → Generate New Private Key\n'
                '4. Save as firebase-credentials.json in root folder\n'
            ))
            return
        
        self.stdout.write(self.style.SUCCESS('✓ Credentials file found'))
        
        # Test connection
        try:
            db = FirebaseConnection.get_db()
            self.stdout.write(self.style.SUCCESS('✓ Connected to Firestore'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Connection failed: {str(e)}'))
            return
        
        # Test write operation
        try:
            self.stdout.write('\n📝 Testing write operation...')
            
            test_price = FirestorePriceHistory(
                date=date.today(),
                open=5000.0,
                high=5100.0,
                low=4900.0,
                close=5050.0,
                volume=1000.0
            )
            test_price.save()
            
            self.stdout.write(self.style.SUCCESS('✓ Write test successful'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Write test failed: {str(e)}'))
            return
        
        # Test read operation
        try:
            self.stdout.write('📖 Testing read operation...')
            
            read_price = FirestorePriceHistory.get_by_date(date.today())
            
            if read_price:
                self.stdout.write(self.style.SUCCESS('✓ Read test successful'))
                self.stdout.write(f'   Data: {read_price.date} - Close: {read_price.close}')
            else:
                self.stdout.write(self.style.WARNING('⚠️  No data found'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Read test failed: {str(e)}'))
            return
        
        # List collections
        try:
            self.stdout.write('\n📚 Available collections:')
            collections = db.collections()
            for collection in collections:
                doc_count = len(list(collection.stream()))
                self.stdout.write(f'   - {collection.id}: {doc_count} documents')
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'⚠️  Could not list collections: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS('\n✅ All tests passed! Firebase is ready to use.'))
