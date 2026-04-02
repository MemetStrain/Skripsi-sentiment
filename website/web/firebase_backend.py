"""
Firebase Backend untuk Django Models
Menggantikan SQLite dengan Firestore sebagai database utama
"""
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from datetime import datetime, date
from decimal import Decimal
from django.conf import settings
import os


class FirebaseConnection:
    """Singleton connection ke Firebase"""
    _db = None
    _initialized = False
    
    @classmethod
    def get_db(cls):
        if not cls._initialized:
            # Cek apakah Firebase credentials tersedia
            cred_path = os.path.join(settings.BASE_DIR, 'firebase-credentials.json')
            
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                # Gunakan default credentials atau environment variable
                try:
                    firebase_admin.initialize_app()
                except ValueError:
                    # Already initialized
                    pass
            
            cls._db = firestore.client()
            cls._initialized = True
        
        return cls._db


class FirebaseModelManager:
    """Manager untuk operasi CRUD ke Firestore"""
    
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.db = FirebaseConnection.get_db()
    
    def _serialize_value(self, value):
        """Convert Python types ke Firestore compatible types"""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, date):
            return value.isoformat()
        elif isinstance(value, Decimal):
            return float(value)
        return value
    
    def _deserialize_value(self, value, field_type):
        """Convert Firestore types ke Python types"""
        if field_type == 'date':
            if isinstance(value, str):
                return datetime.fromisoformat(value).date()
            return value
        elif field_type == 'datetime':
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            return value
        elif field_type == 'decimal':
            return Decimal(str(value))
        return value
    
    def create(self, data):
        """Create document baru di Firestore"""
        # Serialize data
        serialized = {k: self._serialize_value(v) for k, v in data.items()}
        serialized['created_at'] = datetime.now().isoformat()
        
        # Generate ID atau gunakan ID yang diberikan
        if 'id' in serialized:
            doc_id = str(serialized['id'])
            doc_ref = self.db.collection(self.collection_name).document(doc_id)
        else:
            doc_ref = self.db.collection(self.collection_name).document()
            serialized['id'] = doc_ref.id
        
        doc_ref.set(serialized)
        return serialized
    
    def get(self, doc_id):
        """Get single document by ID"""
        doc_ref = self.db.collection(self.collection_name).document(str(doc_id))
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        return None
    
    def filter(self, **kwargs):
        """Filter documents"""
        query = self.db.collection(self.collection_name)
        
        for key, value in kwargs.items():
            serialized_value = self._serialize_value(value)
            query = query.where(key, '==', serialized_value)
        
        docs = query.stream()
        return [doc.to_dict() for doc in docs]
    
    def all(self):
        """Get all documents"""
        docs = self.db.collection(self.collection_name).stream()
        return [doc.to_dict() for doc in docs]
    
    def update(self, doc_id, data):
        """Update document"""
        serialized = {k: self._serialize_value(v) for k, v in data.items()}
        serialized['updated_at'] = datetime.now().isoformat()
        
        doc_ref = self.db.collection(self.collection_name).document(str(doc_id))
        doc_ref.update(serialized)
        return self.get(doc_id)
    
    def delete(self, doc_id):
        """Delete document"""
        doc_ref = self.db.collection(self.collection_name).document(str(doc_id))
        doc_ref.delete()
        return True
    
    def order_by(self, field, direction='DESCENDING'):
        """Order documents"""
        from google.cloud.firestore import Query
        
        dir_map = {
            'DESCENDING': Query.DESCENDING,
            'ASCENDING': Query.ASCENDING
        }
        
        query = self.db.collection(self.collection_name).order_by(
            field, 
            direction=dir_map.get(direction, Query.DESCENDING)
        )
        
        docs = query.stream()
        return [doc.to_dict() for doc in docs]
    
    def count(self):
        """Count documents (expensive operation)"""
        return len(self.all())


class FirestorePriceHistory:
    """Wrapper untuk PriceHistory menggunakan Firestore"""
    collection = 'price_history'
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.date = kwargs.get('date')
        self.open = kwargs.get('open', 0.0)
        self.high = kwargs.get('high', 0.0)
        self.low = kwargs.get('low', 0.0)
        self.close = kwargs.get('close', 0.0)
        self.volume = kwargs.get('volume', 0.0)
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')
        
        self.manager = FirebaseModelManager(self.collection)
    
    def save(self):
        """Save to Firestore"""
        data = {
            'date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }
        
        if self.id:
            # Update existing
            result = self.manager.update(self.id, data)
        else:
            # Create new - use date as ID
            data['id'] = str(self.date)
            result = self.manager.create(data)
            self.id = result['id']
        
        return self
    
    @classmethod
    def get_by_date(cls, date_obj):
        """Get price by date"""
        manager = FirebaseModelManager(cls.collection)
        data = manager.get(str(date_obj))
        
        if data:
            return cls(**data)
        return None
    
    @classmethod
    def all_prices(cls):
        """Get all prices"""
        manager = FirebaseModelManager(cls.collection)
        docs = manager.order_by('date', 'DESCENDING')
        
        return [cls(**doc) for doc in docs]
    
    @classmethod
    def latest(cls, limit=100):
        """Get latest prices"""
        prices = cls.all_prices()
        return prices[:limit]


class FirestoreNews:
    """Wrapper untuk News menggunakan Firestore"""
    collection = 'news'
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.date = kwargs.get('date')
        self.title = kwargs.get('title', '')
        self.snippet = kwargs.get('snippet', '')
        self.url = kwargs.get('url', '')
        self.sentiment_score = kwargs.get('sentiment_score', 0.0)
        self.sentiment_label = kwargs.get('sentiment_label', 'neutral')
        self.created_at = kwargs.get('created_at')
        
        self.manager = FirebaseModelManager(self.collection)
    
    def save(self):
        """Save to Firestore"""
        data = {
            'date': self.date,
            'title': self.title,
            'snippet': self.snippet,
            'url': self.url,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
        }
        
        if self.id:
            result = self.manager.update(self.id, data)
        else:
            result = self.manager.create(data)
            self.id = result['id']
        
        return self
    
    @classmethod
    def all_news(cls):
        """Get all news"""
        manager = FirebaseModelManager(cls.collection)
        docs = manager.order_by('date', 'DESCENDING')
        
        return [cls(**doc) for doc in docs]


class FirestoreMarketState:
    """Wrapper untuk MarketState menggunakan Firestore"""
    collection = 'market_states'
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.date = kwargs.get('date')
        self.state = kwargs.get('state', 2)
        self.probability = kwargs.get('probability', 0.0)
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')
        
        self.manager = FirebaseModelManager(self.collection)
    
    @property
    def state_color(self):
        """Get color based on state"""
        colors = {
            0: '#ef4444',  # bearish
            1: '#22c55e',  # bullish
            2: '#6b7280',  # neutral
        }
        return colors.get(self.state, '#6b7280')
    
    def get_state_display(self):
        """Get state label"""
        labels = {
            0: 'Bearish',
            1: 'Bullish',
            2: 'Neutral',
        }
        return labels.get(self.state, 'Unknown')
    
    def save(self):
        """Save to Firestore"""
        data = {
            'date': self.date,
            'state': self.state,
            'probability': self.probability,
        }
        
        if self.id:
            result = self.manager.update(self.id, data)
        else:
            data['id'] = str(self.date)
            result = self.manager.create(data)
            self.id = result['id']
        
        return self
    
    @classmethod
    def get_by_date(cls, date_obj):
        """Get market state by date"""
        manager = FirebaseModelManager(cls.collection)
        data = manager.get(str(date_obj))
        
        if data:
            return cls(**data)
        return None
    
    @classmethod
    def all_states(cls):
        """Get all market states"""
        manager = FirebaseModelManager(cls.collection)
        docs = manager.order_by('date', 'DESCENDING')
        
        return [cls(**doc) for doc in docs]
