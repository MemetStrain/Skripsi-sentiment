"""
services.py — lightweight utilities for the web app.
All data reads go directly through Firestore in views.py.
Heavy ML logic lives in the scheduler/ package.
"""
# No heavy imports (numpy/pandas belong in the scheduler, not the web app)
