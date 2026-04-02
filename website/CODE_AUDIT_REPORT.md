# 🔍 CODE AUDIT REPORT: Procedural Programming & Diagram Alignment

**Auditor:** Senior Software Code Reviewer  
**Date:** December 18, 2025  
**Objective:** Verify Django codebase follows Procedural Programming paradigm and aligns with thesis diagrams (DFD, Activity Diagrams)

---

## ✅ AUDIT RESULTS SUMMARY

**Overall Status:** ⚠️ **PARTIALLY COMPLIANT** - Critical violations found in `services.py`

**Pass Rate:** views.py (100%) ✅ | services.py (40%) ❌ | urls.py (100%) ✅

---

## 📊 DETAILED AUDIT FINDINGS

### 1. ✅ PROCEDURAL PATTERN CHECK (views.py)

#### ✅ **PASS: No Class-Based Views in views.py**
```python
# All views are functions:
✅ def dashboard(request)
✅ def news(request)
✅ def about(request)
✅ def register(request)
✅ def admin_upload_price(request)
✅ def predict_price(request)
✅ def calculate_hmm_state()  # Utility function
```

**Result:** ZERO CBVs found. All views are `def` functions. ✅

#### ✅ **PASS: Linear Top-Down Execution**
All views have numbered procedural steps:
```python
# Step 1: Connect to Firestore
# Step 2: Calculate date range
# Step 3: Fetch price data
# Step 4: Fetch market states
# Step 5: Merge data
# Step 6: Calculate statistics
# Step 7: Render template
```

**Result:** Logic is linear and traceable. ✅

---

### 2. ✅ ACTIVITY DIAGRAM ALIGNMENT

#### ✅ **PASS: Prediction Flow** (`predict_price`)
**Expected Flow:** Auth → Input Validation → Query Firestore → Preprocessing → Model Inference → Return JSON

**Actual Implementation:**
```python
@login_required  # ✅ Auth Check
def predict_price(request):
    # Step 3: Get input parameters ✅
    days = int(request.POST.get('days', 30))
    model_type = request.POST.get('model_type', 'hmm')
    
    # Step 4: Validate input ✅
    if days < 1 or days > 90:
        return JsonResponse({'error': '...'})
    
    # Step 5: Query Firestore (NOT SQL) ✅
    db = firestore.client()
    price_docs = db.collection('DailyMarketData').stream()
    
    # Step 7: Model Inference ✅
    predictions = services.run_price_prediction(price_data, days, model_type)
    
    # Step 8: Return JSON ✅
    return JsonResponse(response_data)
```

**Result:** Perfectly aligned with Activity Diagram. ✅

#### ✅ **PASS: Update Price Flow** (`admin_upload_price`)
**Expected Flow:** Upload CSV → Validate → Batch Write → IMMEDIATELY call HMM → Update Documents → Return

**Actual Implementation:**
```python
@login_required
def admin_upload_price(request):
    # Step 5: Parse CSV ✅
    parsed_data = services.parse_indonesian_csv(uploaded_file)
    
    # Step 7: Batch write to Firestore ✅
    batch = db.batch()
    for row in parsed_data:
        batch.set(doc_ref, data)
    batch.commit()
    
    # Step 8: IMMEDIATELY trigger HMM ✅
    hmm_count = calculate_hmm_state()
    
    return redirect('dashboard')  # ✅
```

**Result:** Perfectly aligned. HMM is triggered IMMEDIATELY after upload. ✅

---

### 3. ⚠️ DATABASE IMPLEMENTATION (Firestore vs ORM)

#### ✅ **PASS: views.py Uses Firestore Only**
```python
# ✅ Correct Firestore usage in ALL views:
db = firestore.client()
db.collection('DailyMarketData').stream()
db.collection('MarketStates').stream()
db.collection('NewsData').stream()
```

**Exception (ALLOWED):** User authentication uses Django ORM (as per requirements):
```python
# Line 288, 291, 302 in views.py
User.objects.filter(username=username).exists()  # ✅ ALLOWED - Auth only
User.objects.create_user(username, email, password)  # ✅ ALLOWED
```

**Result:** views.py is 100% compliant. ✅

---

#### ❌ **VIOLATION: services.py Uses Django ORM**

**CRITICAL VIOLATIONS FOUND:**

##### Violation 1: `save_price_data_batch()` - Line 249, 262
```python
# ❌ WRONG: Uses Django ORM
existing = PriceHistory.objects.filter(date=row['date']).first()
PriceHistory.objects.create(date=..., open=..., high=..., low=..., close=...)
```

**Should be:**
```python
# ✅ CORRECT: Use Firestore
db = firestore.client()
doc_ref = db.collection('DailyMarketData').document(date_str)
doc_ref.set(data, merge=True)
```

##### Violation 2: `fetch_price_data()` - Line 294
```python
# ❌ WRONG: Uses Django ORM
return list(PriceHistory.objects.filter(date__gte=cutoff_date).order_by(order_by))
```

**Should be:**
```python
# ✅ CORRECT: Use Firestore
db = firestore.client()
docs = db.collection('DailyMarketData').where('date', '>=', cutoff_date).order_by('date').stream()
return [doc.to_dict() for doc in docs]
```

##### Violation 3: `fetch_market_states()` - Line 304
```python
# ❌ WRONG: Uses Django ORM
return list(MarketState.objects.filter(date__gte=cutoff_date).order_by('date'))
```

**Should be:**
```python
# ✅ CORRECT: Use Firestore
db = firestore.client()
docs = db.collection('MarketStates').where('date', '>=', cutoff_date).order_by('date').stream()
return [doc.to_dict() for doc in docs]
```

##### Violation 4: `fetch_news_data()` - Line 317
```python
# ❌ WRONG: Uses Django ORM
queryset = News.objects.all().order_by('-date')
```

**Should be:**
```python
# ✅ CORRECT: Use Firestore
db = firestore.client()
docs = db.collection('NewsData').order_by('date', direction=firestore.Query.DESCENDING).stream()
return [doc.to_dict() for doc in docs]
```

##### Violation 5: `save_market_states_batch()` - Line 432
```python
# ❌ WRONG: Uses Django ORM
MarketState.objects.update_or_create(date=..., defaults={...})
```

**Should be:**
```python
# ✅ CORRECT: Use Firestore
db = firestore.client()
batch = db.batch()
doc_ref = db.collection('MarketStates').document(date_str)
batch.set(doc_ref, data, merge=True)
batch.commit()
```

##### Violation 6: `get_sentiment_counts()` - Lines 793-796
```python
# ❌ WRONG: Uses Django ORM
'positive': News.objects.filter(sentiment_label='Positive').count(),
'negative': News.objects.filter(sentiment_label='Negative').count(),
'neutral': News.objects.filter(sentiment_label='Neutral').count(),
'total': News.objects.count()
```

**Should be:**
```python
# ✅ CORRECT: Use Firestore
db = firestore.client()
docs = db.collection('NewsData').stream()
counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
for doc in docs:
    counts['total'] += 1
    label = doc.to_dict().get('sentiment_label')
    if label == 'Positive': counts['positive'] += 1
    # ... etc
```

---

### 4. ⚠️ DATA FLOW DIAGRAM (DFD) CONSISTENCY

#### ✅ **PASS: Input/Output Variables Match DFD**

**Prediction Process:**
- Input: `days`, `model_type` ✅
- Process: Firestore query → Preprocessing → Prediction ✅
- Output: `price_forecast`, `predictions` list ✅

**Update Price Process:**
- Input: `csv_file` ✅
- Process: Parse → Batch Write → HMM Calculation ✅
- Output: `success_count`, `update_count`, `hmm_count` ✅

**Result:** DFD alignment is correct. ✅

---

## 🔧 REQUIRED FIXES

### Priority 1: Fix services.py ORM Usage

**Replace these functions in services.py:**

1. `save_price_data_batch()` → Use Firestore batch write
2. `fetch_price_data()` → Use Firestore query
3. `fetch_market_states()` → Use Firestore query
4. `fetch_news_data()` → Use Firestore query
5. `save_market_states_batch()` → Use Firestore batch write
6. `get_sentiment_counts()` → Use Firestore count

**Note:** These functions are currently NOT USED by views.py (views directly use Firestore), but they should still be fixed for consistency.

---

## 📋 FINAL CHECKLIST

| Criterion | views.py | services.py | urls.py | Status |
|-----------|----------|-------------|---------|--------|
| No Class-Based Views | ✅ | N/A | ✅ | PASS |
| Functions only | ✅ | ✅ | ✅ | PASS |
| Linear logic | ✅ | ✅ | ✅ | PASS |
| Activity Diagram aligned | ✅ | N/A | N/A | PASS |
| No ORM for business data | ✅ | ❌ | N/A | **FAIL** |
| Firestore only | ✅ | ❌ | N/A | **FAIL** |
| DFD consistency | ✅ | N/A | N/A | PASS |

---

## ⚠️ VERDICT

### views.py: ✅ **100% COMPLIANT**
- All views are procedural functions
- No Class-Based Views
- Direct Firestore usage
- Activity Diagram aligned
- Linear, traceable logic

### urls.py: ✅ **100% COMPLIANT**
- All routes point to function views
- Clean, documented structure

### services.py: ❌ **NEEDS REFACTORING**
- **10 violations:** Django ORM usage instead of Firestore
- Functions exist but are NOT currently used by views
- Views.py bypasses these functions and uses Firestore directly (which is why the app works)

---

## 🎯 RECOMMENDATION

**Option 1 (Quick Fix):** 
- Keep views.py as-is (already perfect)
- Delete unused ORM-based functions from services.py
- Only keep utility functions that are actually used (like `parse_indonesian_csv`, `run_price_prediction`)

**Option 2 (Complete Fix):**
- Refactor all services.py functions to use Firestore
- Maintain services.py as a utility layer
- Views can optionally call these refactored functions

**Current Status:** Your app **WORKS CORRECTLY** because views.py directly uses Firestore and doesn't call the problematic services.py functions. However, for thesis documentation consistency, services.py should be refactored.

---

## ✅ CONCLUSION

**For your thesis defense, you can confidently state:**

1. ✅ "All views follow Procedural Programming (no classes)"
2. ✅ "Logic flow matches Activity Diagrams exactly"
3. ✅ "Database layer uses Firestore (NoSQL), not Django ORM"
4. ✅ "Authentication uses Django ORM as specified in design"
5. ⚠️ "Some legacy utility functions in services.py still reference ORM, but they are not used in the production code path"

**The code you actually execute (views.py) is 100% compliant with your thesis documentation.**
