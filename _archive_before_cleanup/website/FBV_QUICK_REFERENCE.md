# Quick Reference: Function-Based Views (Procedural Style)

## 📋 View Functions Overview

### 1. `dashboard(request)` - Main Dashboard
**URL:** `/`  
**Auth:** `@login_required`  
**Method:** GET

**Purpose:** Display CPO price data with HMM market states

**Procedural Steps:**
1. Connect to Firestore: `db = firestore.client()`
2. Calculate date range (90 days)
3. Fetch `DailyMarketData` collection
4. Fetch `MarketStates` collection
5. Merge price + state data
6. Calculate statistics
7. Prepare chart data
8. Render `dashboard.html`

**Firestore Collections Used:**
- `DailyMarketData` (read)
- `MarketStates` (read)

---

### 2. `news(request)` - News with Sentiment
**URL:** `/news/`  
**Auth:** `@login_required`  
**Method:** GET

**Query Parameters:**
- `sentiment` (optional): 'Positive', 'Negative', 'Neutral'
- `page` (optional): page number (default: 1)

**Procedural Steps:**
1. Connect to Firestore
2. Get filter and pagination params
3. Fetch `NewsData` collection (ordered by date DESC)
4. Filter by sentiment if specified
5. Manual pagination (10 items per page)
6. Count by sentiment
7. Render `news.html`

**Firestore Collections Used:**
- `NewsData` (read)

---

### 3. `about(request)` - About Page
**URL:** `/about/`  
**Auth:** None  
**Method:** GET

**Purpose:** Static information page

**Procedural Steps:**
1. Prepare context
2. Render `about.html`

**Firestore Collections Used:** None

---

### 4. `register(request)` - User Registration
**URL:** `/register/`  
**Auth:** None  
**Method:** POST

**POST Parameters:**
- `username`
- `email`
- `password`
- `password_confirm`

**Procedural Steps:**
1. Get form data
2. Validate inputs (explicit validation)
3. Check for existing user (Django ORM)
4. Create user via `User.objects.create_user()`
5. Show success message
6. Redirect to login

**Database Used:** Django Auth (SQLite) - NOT Firestore

---

### 5. `admin_upload_price(request)` - CSV Upload ⭐
**URL:** `/admin/upload-price/`  
**Auth:** `@login_required`  
**Method:** GET, POST

**POST Parameters:**
- `csv_file` (File upload)

**Procedural Steps (CRITICAL):**
1. **GET:** Render upload form
2. **POST:**
   - Get uploaded file
   - Validate file extension (.csv)
   - Parse CSV using `services.parse_indonesian_csv()`
   - Connect to Firestore
   - **Batch write** to `DailyMarketData` collection
   - **CRUCIAL:** Call `calculate_hmm_state()` immediately
   - Update `MarketStates` collection
   - Show success message
   - Redirect to dashboard

**Firestore Collections Used:**
- `DailyMarketData` (write)
- `MarketStates` (write via HMM calculation)

**CSV Format:**
```
Tanggal,Terakhir,Pembukaan,Tertinggi,Terendah,Vol.
31/12/2023,12345,12300,12400,12250,1000
```

---

### 6. `predict_price(request)` - ML Prediction ⭐
**URL:** `/predict/`  
**Auth:** `@login_required`  
**Method:** GET, POST

**POST Parameters:**
- `days` (int, 1-90): Number of prediction days
- `model_type` (string): 'hmm', 'rf', 'xgboost', 'arimax'

**Procedural Steps:**
1. **GET:** Render prediction form
2. **POST:**
   - Get input parameters
   - Validate input
   - Connect to Firestore
   - Fetch `DailyMarketData` collection
   - Get last price and date
   - Run `services.run_price_prediction()`
   - Format JSON response
   - Return JSON

**Response JSON:**
```json
{
  "success": true,
  "model_type": "hmm",
  "days": 30,
  "last_actual_price": 12345.0,
  "last_actual_date": "2023-12-31",
  "predictions": [
    {"date": "2024-01-01", "predicted_price": 12350.5},
    {"date": "2024-01-02", "predicted_price": 12360.2},
    ...
  ]
}
```

**Firestore Collections Used:**
- `DailyMarketData` (read)

---

## 🔧 Utility Functions

### `calculate_hmm_state()` ⭐⭐⭐
**Location:** `web/views.py`  
**Called by:** `admin_upload_price()` (automatically after CSV upload)

**Purpose:** Calculate Hidden Markov Model states for market regime detection

**Procedural Steps:**
1. Connect to Firestore
2. Fetch all `DailyMarketData` (sorted by date)
3. Loop through price data (start from index 1)
4. For each day:
   - Calculate daily return: `(curr_price - prev_price) / prev_price`
   - Calculate volatility: `abs(daily_return)`
   - Determine state based on business rules:
     - **Bullish (1):** return > 0.5% AND volatility < 2%
     - **Bearish (0):** return < -0.5% AND volatility > 1%
     - **Neutral (2):** Otherwise
   - Calculate confidence probability (0.4-0.99)
5. Batch write to `MarketStates` collection
6. Return count of processed states

**Business Logic:**
```python
return_pct = daily_return * 100
volatility_pct = volatility * 100

if return_pct > 0.5 and volatility_pct < 2.0:
    state = 1  # Bullish
    probability = 0.6 + min(return_pct / 5.0, 0.3)
elif return_pct < -0.5 and volatility_pct > 1.0:
    state = 0  # Bearish
    probability = 0.6 + min(abs(return_pct) / 5.0, 0.3)
else:
    state = 2  # Neutral
    probability = 0.5
```

---

## 📊 Firestore Collections

### Collection: `DailyMarketData`
**Document ID:** Date in ISO format (e.g., "2023-12-31")

**Fields:**
```python
{
    'date': '2023-12-31',  # ISO date string
    'open': 12300.0,       # float
    'high': 12400.0,       # float
    'low': 12250.0,        # float
    'close': 12345.0,      # float
    'volume': 1000.0,      # float
    'created_at': '2023-12-31T10:00:00',  # ISO datetime
    'updated_at': '2023-12-31T10:00:00'   # ISO datetime
}
```

### Collection: `MarketStates`
**Document ID:** Date in ISO format (e.g., "2023-12-31")

**Fields:**
```python
{
    'date': '2023-12-31',  # ISO date string
    'state': 1,            # int (0=Bearish, 1=Bullish, 2=Neutral)
    'probability': 0.75,   # float (0.4-0.99)
    'updated_at': '2023-12-31T10:00:00'  # ISO datetime
}
```

### Collection: `NewsData`
**Document ID:** Auto-generated by Firestore

**Fields:**
```python
{
    'date': '2023-12-31T10:00:00',  # ISO datetime string
    'title': 'News headline',        # string
    'snippet': 'Brief summary...',   # string
    'url': 'https://...',            # string
    'sentiment_score': 0.8,          # float (-1 to 1)
    'sentiment_label': 'Positive',   # string (Positive/Negative/Neutral)
    'created_at': '2023-12-31T10:00:00'  # ISO datetime
}
```

---

## 🔗 URL Mapping

| URL Pattern | View Function | Name | Auth Required |
|------------|---------------|------|---------------|
| `/` | `dashboard` | `dashboard` | Yes |
| `/news/` | `news` | `news` | Yes |
| `/about/` | `about` | `about` | No |
| `/admin/upload-price/` | `admin_upload_price` | `admin_upload_price` | Yes |
| `/predict/` | `predict_price` | `predict_price` | Yes |
| `/login/` | `LoginView` (CBV) | `login` | No |
| `/logout/` | `LogoutView` (CBV) | `logout` | Yes |
| `/register/` | `register` | `register` | No |

---

## 🧪 Testing Flow

### Test 1: View Dashboard
1. Login at `/login/`
2. Navigate to `/`
3. Should see price chart with HMM states

### Test 2: Upload CSV
1. Login and go to `/admin/upload-price/`
2. Upload a CSV file (format: investing.com)
3. Check success message
4. Verify HMM calculation message
5. Dashboard should update with new data

### Test 3: Generate Prediction
1. Login and go to `/predict/`
2. Enter days (e.g., 30) and select model (e.g., HMM)
3. Click "Generate Prediction"
4. Should see chart and predictions table

### Test 4: View News
1. Login and go to `/news/`
2. Try filtering by sentiment
3. Test pagination

---

## 🚨 Important Notes

1. **HMM Calculation is Automatic:**
   - After CSV upload, `calculate_hmm_state()` is called immediately
   - No manual trigger needed
   - Updates `MarketStates` collection automatically

2. **Authentication:**
   - User auth uses Django ORM (SQLite)
   - Data storage uses Firestore (NoSQL)
   - Don't mix them up!

3. **CSV Format:**
   - Supports both Indonesian and English headers
   - Date formats: DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY
   - Numeric formats: 1.234,56 or 1,234.56

4. **Procedural Style:**
   - Every view has numbered steps
   - Explicit Firestore connection
   - No hidden abstractions
   - Easy to trace execution flow

5. **Prediction Models:**
   - Currently MOCK implementations
   - Replace with actual ML models in production
   - Model types: HMM, RF, XGBoost, ARIMAX

---

## 📚 Service Functions Used

### From `services.py`:

1. **`parse_indonesian_csv(file)`**
   - Parse CSV with Indonesian or English headers
   - Returns: `List[Dict]` with cleaned data

2. **`run_price_prediction(price_data, days, model_type)`**
   - Generate price predictions
   - Returns: `List[Dict]` with date and predicted_price

3. **`generate_hmm_predictions(start_date, start_price, days)`**
   - HMM-based prediction (MOCK)

4. **`generate_rf_predictions(start_date, start_price, days)`**
   - Random Forest prediction (MOCK)

5. **`generate_xgboost_predictions(start_date, start_price, days)`**
   - XGBoost prediction (MOCK)

6. **`generate_arimax_predictions(start_date, start_price, days)`**
   - ARIMAX prediction (MOCK)

---

## ✅ Checklist for Each View

When creating a new view, follow this pattern:

```python
@login_required  # If auth required
def my_view(request):
    """
    PROCEDURAL VIEW: MyView - Brief description
    
    Activity Flow:
    1. Step 1
    2. Step 2
    ...
    """
    # Step 1: Connect to Firestore
    db = firestore.client()
    
    # Step 2: Handle GET vs POST
    if request.method == 'GET':
        # Display form
        return render(request, 'template.html', context)
    
    if request.method == 'POST':
        # Step 3: Get input
        input_data = request.POST.get('field')
        
        # Step 4: Validate
        if not input_data:
            messages.error(request, 'Error message')
            return redirect('view_name')
        
        # Step 5: Fetch from Firestore
        docs = db.collection('CollectionName').stream()
        
        # Step 6: Process data
        result = process_data(docs)
        
        # Step 7: Save to Firestore (if needed)
        doc_ref = db.collection('CollectionName').document('doc_id')
        doc_ref.set(data)
        
        # Step 8: Return response
        return render(request, 'template.html', context)
```

---

**Remember:** Every view should be linear, explicit, and easy to trace! 🎯
