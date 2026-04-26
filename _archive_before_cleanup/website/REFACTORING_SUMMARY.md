# Django Refactoring Complete: CBV to FBV (Procedural Style)

## Summary of Changes

The Django codebase has been successfully refactored from Class-Based Views (CBV) to Function-Based Views (FBV) with a **Procedural Programming** style that aligns with your thesis documentation (Activity Diagrams).

---

## 🎯 Key Achievements

### ✅ All Views Converted to Procedural Functions

**Before (CBV - Hidden Logic):**
```python
class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard.html"
    def get_context_data(self, **kwargs):
        # Hidden logic inside methods
```

**After (FBV - Explicit Procedural Logic):**
```python
@login_required
def dashboard(request):
    # Step 1: Connect to Firestore
    db = firestore.client()
    
    # Step 2: Fetch data explicitly
    price_docs = db.collection('DailyMarketData').stream()
    
    # Step 3: Process data
    # ... explicit procedural steps
    
    # Step 4: Render template
    return render(request, 'dashboard.html', context)
```

---

## 📁 Files Modified

### 1. **`web/views.py`** (Complete Rewrite)

All views now follow procedural pattern with explicit steps:

#### **Dashboard View** (`dashboard`)
- **Purpose:** Display CPO price data with HMM states
- **Flow:**
  1. Connect to Firestore explicitly
  2. Fetch `DailyMarketData` collection (last 90 days)
  3. Fetch `MarketStates` collection (HMM results)
  4. Merge price data with state data
  5. Calculate statistics
  6. Prepare chart data
  7. Render dashboard

#### **News View** (`news`)
- **Purpose:** Display CPO news with sentiment analysis
- **Flow:**
  1. Connect to Firestore
  2. Get filter parameters
  3. Fetch `NewsData` collection
  4. Apply sentiment filter
  5. Implement manual pagination (Firestore-compatible)
  6. Count by sentiment
  7. Render news page

#### **Admin Upload View** (`admin_upload_price`) - **NEW**
- **Purpose:** Upload CSV price data and trigger HMM calculation
- **Flow (CRITICAL for thesis):**
  1. Handle GET (show form) / POST (process upload)
  2. Validate CSV file
  3. Parse CSV using `services.parse_indonesian_csv()`
  4. **Batch write to Firestore** (`DailyMarketData` collection)
  5. **CRUCIAL: Call `calculate_hmm_state()` immediately**
  6. Update `MarketStates` collection with HMM results
  7. Show success message
  8. Redirect to dashboard

#### **Prediction View** (`predict_price`) - **NEW**
- **Purpose:** Generate ML-based price predictions
- **Flow:**
  1. Handle GET (show form) / POST (process prediction)
  2. Receive input (days, model_type)
  3. Validate input
  4. Fetch historical data from Firestore
  5. Run prediction algorithm via `services.run_price_prediction()`
  6. Return JSON response (for AJAX)

#### **About View** (`about`)
- **Purpose:** Static page, no database interaction

#### **Register View** (`register`)
- **Purpose:** User registration (Django Auth - SQLite)
- **Note:** Auth uses Django ORM as per requirements, only data storage uses Firestore

---

### 2. **`web/urls.py`** (Updated)

New URL patterns added:

```python
urlpatterns = [
    # Main Views
    path('', views.dashboard, name='dashboard'),
    path('news/', views.news, name='news'),
    path('about/', views.about, name='about'),
    
    # Admin Features
    path('admin/upload-price/', views.admin_upload_price, name='admin_upload_price'),
    
    # ML Features
    path('predict/', views.predict_price, name='predict_price'),
    
    # Auth
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='dashboard'), name='logout'),
    path('register/', views.register, name='register'),
]
```

---

### 3. **`web/services.py`** (Updated)

Modified functions to work with Firestore data structures:

- `run_price_prediction()` - Now accepts `List[Dict]` from Firestore instead of Django ORM objects
- All prediction generators (`generate_hmm_predictions`, `generate_rf_predictions`, etc.) - Date serialization added

---

### 4. **New Utility Function in `views.py`**

#### `calculate_hmm_state()`
**Critical function that implements the "Update Price" feature from thesis:**

```python
def calculate_hmm_state():
    """
    UTILITY FUNCTION: Calculate HMM States and Save to Firestore
    
    Procedural Steps:
    1. Fetch all price data from Firestore
    2. Calculate daily returns
    3. Calculate volatility
    4. Determine market state (Bearish=0, Bullish=1, Neutral=2)
    5. Calculate confidence probability
    6. Batch write to MarketStates collection
    7. Return count of processed states
    """
```

**Business Rules Implemented:**
- **Bullish:** return > 0.5% AND volatility < 2%
- **Bearish:** return < -0.5% AND volatility > 1%
- **Neutral:** Otherwise

---

### 5. **New Templates Created**

#### `web/templates/upload_price.html`
- Beautiful upload form with instructions
- Shows CSV format requirements
- Displays post-upload process flow
- File drag-and-drop support

#### `web/templates/predict.html`
- Prediction form (days, model type)
- Real-time prediction via AJAX
- Chart.js visualization
- Prediction results table

---

## 🔥 Key Features Implemented

### 1. **Explicit Firestore Integration**
- Every view explicitly connects to Firestore: `db = firestore.client()`
- Direct collection queries: `db.collection('DailyMarketData').stream()`
- No hidden ORM abstractions

### 2. **Procedural Flow with Comments**
- Each step numbered and documented
- Linear, easy-to-trace logic
- Aligned with Activity Diagrams

### 3. **CSV Upload with Automatic HMM Calculation**
- Admin uploads CSV → Data saved → **HMM automatically calculated** → Dashboard updated
- This implements the critical "Update Price" feature from thesis

### 4. **Machine Learning Prediction**
- Support for 4 model types: HMM, Random Forest, XGBoost, ARIMAX
- AJAX-based prediction generation
- JSON response with chart data

### 5. **Authentication with Decorators**
- Replaced `LoginRequiredMixin` with `@login_required` decorator
- Consistent with procedural style

---

## 📊 Firestore Collections Used

| Collection Name | Purpose | Document ID | Fields |
|----------------|---------|-------------|--------|
| `DailyMarketData` | Price history | date (ISO string) | date, open, high, low, close, volume |
| `MarketStates` | HMM states | date (ISO string) | date, state (0/1/2), probability |
| `NewsData` | News with sentiment | auto-generated | date, title, snippet, url, sentiment_score, sentiment_label |

---

## 🚀 How to Use

### 1. **View Dashboard**
```
http://localhost:8000/
```
- Shows price chart with HMM states
- Last 90 days of data
- Statistics and metrics

### 2. **Upload New Price Data**
```
http://localhost:8000/admin/upload-price/
```
- Upload CSV from investing.com
- Data is validated and saved to Firestore
- **HMM states are calculated automatically**
- Redirects to dashboard with updated data

### 3. **Generate Predictions**
```
http://localhost:8000/predict/
```
- Select number of days (1-90)
- Choose model type (HMM, RF, XGBoost, ARIMAX)
- Get JSON response with predictions
- View chart and table

### 4. **View News**
```
http://localhost:8000/news/
```
- Filter by sentiment (Positive, Negative, Neutral)
- Paginated results (10 per page)
- Shows sentiment counts

---

## 🎓 Alignment with Thesis Documentation

### Activity Diagrams Match
Every view now has explicit steps that can be traced directly to activity diagram boxes:

```
[Start] → [Connect to Firestore] → [Fetch Data] → [Process Data] → 
[Calculate Statistics] → [Prepare Chart Data] → [Render Template] → [End]
```

### Data Flow Diagram (DFD) Match
- **Process 1:** CSV Upload (Data Input & Parsing)
- **Process 2:** Firestore Write (Data Storage)
- **Process 3:** Data Retrieval (Fetch Collections)
- **Process 4:** HMM Calculation (Machine Learning)
- **Process 5:** Prediction Generation (ML Models)

---

## 🔧 Technical Details

### Authentication
- **Django ORM (SQLite)** for user authentication (as per requirements)
- `@login_required` decorator on protected views

### Data Storage
- **Google Cloud Firestore (NoSQL)** for all application data
- Direct `firebase_admin` API calls
- No Django ORM models used for data storage

### CSV Parsing
- Uses `services.parse_indonesian_csv()` function
- Supports Indonesian and English headers
- Multiple date formats (DD/MM/YYYY, DD.MM.YYYY, etc.)

### HMM Calculation
- Triggered automatically after CSV upload
- Calculates daily returns and volatility
- Determines market state (Bearish/Bullish/Neutral)
- Saves to Firestore `MarketStates` collection

---

## ✅ Checklist Completed

- ✅ **No Classes:** All views are `def` functions, not classes
- ✅ **No `get_context_data`:** Data fetching is explicit in function body
- ✅ **Authentication Decorator:** `@login_required` instead of `LoginRequiredMixin`
- ✅ **Explicit GET/POST Handling:** `if request.method == 'POST':`
- ✅ **Firestore Direct Access:** `db = firestore.client()` in every view
- ✅ **Linear Control Flow:** Step-by-step numbered comments
- ✅ **Update Price Feature:** CSV upload → Firestore save → HMM calculation
- ✅ **Prediction Feature:** Input → Fetch data → ML predict → JSON response

---

## 📝 Example: Before vs After

### Dashboard View

**Before (CBV):**
```python
class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard.html"
    def get_context_data(self, **kwargs):
        # Hidden logic
        context = super().get_context_data(**kwargs)
        # ORM queries hidden in methods
        return context
```

**After (FBV - Procedural):**
```python
@login_required
def dashboard(request):
    # Step 1: Connect to Firestore
    db = firestore.client()
    
    # Step 2: Calculate date range
    three_months_ago = datetime.now().date() - timedelta(days=90)
    
    # Step 3: Fetch price data
    price_docs = db.collection('DailyMarketData').where('date', '>=', three_months_ago_str).stream()
    price_list = []
    for doc in price_docs:
        data = doc.to_dict()
        price_list.append({...})
    
    # Step 4: Fetch market states
    state_docs = db.collection('MarketStates').stream()
    
    # Step 5: Merge data
    # ... explicit merging logic
    
    # Step 6: Calculate statistics
    # ... explicit calculation
    
    # Step 7: Render
    return render(request, 'dashboard.html', context)
```

---

## 🎉 Success!

The codebase is now:
- ✅ **Procedural** (no classes for views)
- ✅ **Explicit** (every step documented)
- ✅ **Linear** (easy to trace execution flow)
- ✅ **Firestore-native** (direct API calls)
- ✅ **Thesis-aligned** (matches Activity Diagrams)

All views are ready for demonstration and documentation in your thesis!
