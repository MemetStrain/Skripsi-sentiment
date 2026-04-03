"""
Custom Firestore Authentication Backend
========================================
Replaces Django's SQLite-based auth with Firestore-backed user storage.
Designed for Vercel deployment where no persistent filesystem is available.

Provides:
- FirestoreUser     : User object (no Django ORM dependency)
- AnonymousUser     : Sentinel for unauthenticated requests
- authenticate()    : Email + password check against Firestore
- create_user()     : Registration — writes to Firestore `users` collection
- get_user()        : Fetch user by UID (used on each request)
- login()           : Store user in signed-cookie session
- logout()          : Clear session
- FirestoreAuthMiddleware  : Sets request.user on every request
- user_context_processor   : Makes `user` available in all templates
- firestore_login_required : Decorator to protect views
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from django.shortcuts import redirect
from django.conf import settings


# ---------------------------------------------------------------------------
# User objects
# ---------------------------------------------------------------------------

@dataclass
class FirestoreUser:
    uid: str
    username: str
    email: str
    is_active: bool = True

    is_authenticated = True
    is_anonymous = False

    @property
    def pk(self):
        return self.uid

    def __str__(self):
        return self.username


class AnonymousUser:
    uid = None
    username = ''
    email = ''
    is_authenticated = False
    is_anonymous = True
    is_active = False
    pk = None

    def __str__(self):
        return 'AnonymousUser'


# ---------------------------------------------------------------------------
# Auth operations
# ---------------------------------------------------------------------------

def authenticate(email: str, password: str):
    """
    Verify email + password against Firestore `users` collection.
    Returns FirestoreUser on success, None on failure.
    """
    from firebase_admin import firestore
    from django.contrib.auth.hashers import check_password as django_check_password

    try:
        db = firestore.client()
        docs = list(
            db.collection('users')
            .where('email', '==', email.strip().lower())
            .limit(1)
            .stream()
        )
        if not docs:
            return None
        data = docs[0].to_dict()
        if not data.get('is_active', True):
            return None
        if not django_check_password(password, data.get('password_hash', '')):
            return None
        # Update last_login
        docs[0].reference.update({'last_login': datetime.now(timezone.utc).isoformat()})
        return FirestoreUser(
            uid=data['uid'],
            username=data['username'],
            email=data['email'],
            is_active=data.get('is_active', True),
        )
    except Exception:
        return None


def create_user(username: str, email: str, password: str) -> FirestoreUser:
    """
    Create a new user document in Firestore `users` collection.
    Raises ValueError if username or email already exists.
    """
    from firebase_admin import firestore
    from django.contrib.auth.hashers import make_password

    db = firestore.client()
    email_lower = email.strip().lower()
    username = username.strip()

    # Uniqueness checks
    if list(db.collection('users').where('email', '==', email_lower).limit(1).stream()):
        raise ValueError('Email already registered.')
    if list(db.collection('users').where('username', '==', username).limit(1).stream()):
        raise ValueError('Username already taken.')

    doc_ref = db.collection('users').document()
    uid = doc_ref.id
    doc_ref.set({
        'uid': uid,
        'username': username,
        'email': email_lower,
        'password_hash': make_password(password),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_login': None,
        'is_active': True,
    })
    return FirestoreUser(uid=uid, username=username, email=email_lower)


def get_user(uid: str):
    """
    Fetch a FirestoreUser by UID. Returns None if not found.
    Called on each request by FirestoreAuthMiddleware.
    """
    from firebase_admin import firestore

    try:
        db = firestore.client()
        doc = db.collection('users').document(uid).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        if not data.get('is_active', True):
            return None
        return FirestoreUser(
            uid=data['uid'],
            username=data['username'],
            email=data['email'],
            is_active=data.get('is_active', True),
        )
    except Exception:
        return None


def login(request, user: FirestoreUser):
    """Store authenticated user info in the signed-cookie session."""
    request.session['_uid'] = user.uid
    request.session['_username'] = user.username
    request.session['_email'] = user.email
    request.user = user


def logout(request):
    """Clear session and reset request.user."""
    request.session.flush()
    request.user = AnonymousUser()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class FirestoreAuthMiddleware:
    """
    Sets request.user on every request by reading the UID from the
    signed-cookie session and fetching the matching Firestore user.

    Uses a simple in-memory UID-to-user cache keyed on the UID string
    (one entry per Django process/worker) to avoid a Firestore read on
    every request. The cache entry is invalidated when UID changes.
    """

    _user_cache: dict = {}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        uid = request.session.get('_uid')
        if uid:
            cached = FirestoreAuthMiddleware._user_cache.get(uid)
            if cached is None:
                cached = get_user(uid)
                if cached:
                    FirestoreAuthMiddleware._user_cache[uid] = cached
            request.user = cached if cached else AnonymousUser()
        else:
            request.user = AnonymousUser()
        return self.get_response(request)


# ---------------------------------------------------------------------------
# Context processor
# ---------------------------------------------------------------------------

def user_context_processor(request):
    """Makes `user` available in every Django template."""
    return {'user': getattr(request, 'user', AnonymousUser())}


# ---------------------------------------------------------------------------
# Login-required decorator
# ---------------------------------------------------------------------------

def firestore_login_required(view_func):
    """Redirect unauthenticated users to the login page."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            login_url = getattr(settings, 'LOGIN_URL', '/login/')
            return redirect(f'{login_url}?next={request.path}')
        return view_func(request, *args, **kwargs)
    return wrapper
