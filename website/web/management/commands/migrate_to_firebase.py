"""
Django Management Command: Migrate data dari SQLite ke Firebase Firestore
Usage: python manage.py migrate_to_firebase
"""
from django.core.management.base import BaseCommand
from django.db import connection
from web.models import PriceHistory, News, MarketState, USE_FIREBASE
from web.firebase_backend import FirestorePriceHistory, FirestoreNews, FirestoreMarketState
import os


class Command(BaseCommand):
    help = 'Migrate data dari SQLite ke Firebase Firestore'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            help='Specify model to migrate: price, news, state, or all (default)',
            default='all'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Simulate migration without actually writing to Firebase',
        )

    def handle(self, *args, **options):
        model_choice = options['model']
        dry_run = options['dry_run']
        
        if dry_run:
            self.stdout.write(self.style.WARNING('🔄 DRY RUN MODE - No data will be written to Firebase'))
        
        # Check Firebase credentials
        cred_path = os.path.join('firebase-credentials.json')
        if not os.path.exists(cred_path):
            self.stdout.write(self.style.ERROR(
                '❌ Firebase credentials not found!\n'
                'Download firebase-credentials.json dari Firebase Console dan taruh di root folder.'
            ))
            return
        
        self.stdout.write(self.style.SUCCESS('✓ Firebase credentials found'))
        
        # Check if USE_FIREBASE is enabled
        if not USE_FIREBASE:
            self.stdout.write(self.style.WARNING(
                '⚠️  USE_FIREBASE is set to False\n'
                'Set environment variable: USE_FIREBASE=true'
            ))
        
        # Migrate based on choice
        if model_choice in ['price', 'all']:
            self.migrate_price_history(dry_run)
        
        if model_choice in ['news', 'all']:
            self.migrate_news(dry_run)
        
        if model_choice in ['state', 'all']:
            self.migrate_market_state(dry_run)
        
        self.stdout.write(self.style.SUCCESS('\n✅ Migration completed!'))
    
    def migrate_price_history(self, dry_run=False):
        """Migrate PriceHistory to Firestore"""
        self.stdout.write('\n📊 Migrating Price History...')
        
        prices = PriceHistory.objects.all().order_by('date')
        total = prices.count()
        
        if total == 0:
            self.stdout.write(self.style.WARNING('  ⚠️  No price data found in SQLite'))
            return
        
        self.stdout.write(f'  Found {total} price records')
        
        success_count = 0
        error_count = 0
        
        for i, price in enumerate(prices, 1):
            try:
                if not dry_run:
                    firestore_obj = FirestorePriceHistory(
                        date=price.date,
                        open=price.open,
                        high=price.high,
                        low=price.low,
                        close=price.close,
                        volume=price.volume or 0.0
                    )
                    firestore_obj.save()
                
                success_count += 1
                
                # Progress indicator
                if i % 10 == 0 or i == total:
                    self.stdout.write(f'  Progress: {i}/{total}', ending='\r')
            
            except Exception as e:
                error_count += 1
                self.stdout.write(self.style.ERROR(f'  Error on {price.date}: {str(e)}'))
        
        self.stdout.write('')  # New line
        self.stdout.write(self.style.SUCCESS(f'  ✓ Migrated {success_count} price records'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'  ✗ Failed: {error_count} records'))
    
    def migrate_news(self, dry_run=False):
        """Migrate News to Firestore"""
        self.stdout.write('\n📰 Migrating News...')
        
        news_list = News.objects.all().order_by('date')
        total = news_list.count()
        
        if total == 0:
            self.stdout.write(self.style.WARNING('  ⚠️  No news data found in SQLite'))
            return
        
        self.stdout.write(f'  Found {total} news records')
        
        success_count = 0
        error_count = 0
        
        for i, news in enumerate(news_list, 1):
            try:
                if not dry_run:
                    firestore_obj = FirestoreNews(
                        date=news.date,
                        title=news.title,
                        snippet=news.snippet,
                        sentiment_score=news.sentiment_score,
                        sentiment_label=news.sentiment_label,
                        url=news.url
                    )
                    firestore_obj.save()
                
                success_count += 1
                
                # Progress indicator
                if i % 10 == 0 or i == total:
                    self.stdout.write(f'  Progress: {i}/{total}', ending='\r')
            
            except Exception as e:
                error_count += 1
                self.stdout.write(self.style.ERROR(f'  Error on {news.title[:30]}: {str(e)}'))
        
        self.stdout.write('')  # New line
        self.stdout.write(self.style.SUCCESS(f'  ✓ Migrated {success_count} news records'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'  ✗ Failed: {error_count} records'))
    
    def migrate_market_state(self, dry_run=False):
        """Migrate MarketState to Firestore"""
        self.stdout.write('\n📈 Migrating Market States...')
        
        states = MarketState.objects.all().order_by('date')
        total = states.count()
        
        if total == 0:
            self.stdout.write(self.style.WARNING('  ⚠️  No market state data found in SQLite'))
            return
        
        self.stdout.write(f'  Found {total} market state records')
        
        success_count = 0
        error_count = 0
        
        for i, state in enumerate(states, 1):
            try:
                if not dry_run:
                    firestore_obj = FirestoreMarketState(
                        date=state.date,
                        state=state.state,
                        probability=state.probability
                    )
                    firestore_obj.save()
                
                success_count += 1
                
                # Progress indicator
                if i % 10 == 0 or i == total:
                    self.stdout.write(f'  Progress: {i}/{total}', ending='\r')
            
            except Exception as e:
                error_count += 1
                self.stdout.write(self.style.ERROR(f'  Error on {state.date}: {str(e)}'))
        
        self.stdout.write('')  # New line
        self.stdout.write(self.style.SUCCESS(f'  ✓ Migrated {success_count} market state records'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'  ✗ Failed: {error_count} records'))
