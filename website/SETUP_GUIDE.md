# 🚀 Complete Setup Guide - CPO Price Prediction System

## 📋 Project Structure
```
website/
├── config/              # Django project settings
│   ├── settings.py      # ✅ Updated with 'web' and 'import_export'
│   └── urls.py          # ✅ Configured to include web app URLs
├── web/                 # Main Django app
│   ├── models.py        # ✅ PriceHistory, News, MarketState
│   ├── admin.py         # ✅ Indonesian CSV import logic
│   ├── views.py         # ✅ Dashboard, News, About views
│   ├── urls.py          # ✅ URL routing
│   └── templates/       # ✅ All HTML templates
│       ├── base.html
│       ├── dashboard.html
│       ├── news.html
│       └── about.html
└── requirements.txt     # ✅ All dependencies
```

---

## 🔧 Step-by-Step Installation

### 1️⃣ Install Dependencies

```bash
pip install django-import-export openpyxl
```

**Or install everything:**
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 3️⃣ Create Superuser (Admin)

```bash
python manage.py createsuperuser
```

Follow the prompts to create your admin account.

### 4️⃣ Run Development Server

```bash
python manage.py runserver
```

### 5️⃣ Access the Application

- **Main Dashboard:** http://localhost:8000/
- **News Page:** http://localhost:8000/news/
- **About Page:** http://localhost:8000/about/
- **Django Admin:** http://localhost:8000/admin/

---

## 📊 Importing Sample Data

### Method 1: Using Django Admin (Recommended)

1. **Login to Admin Panel:** http://localhost:8000/admin/
2. **Go to Price Histories**
3. **Click "IMPORT" button** (top right)
4. **Select your CSV file** with Indonesian format
5. **Preview and Confirm**

### Method 2: Using Django Shell

```bash
python manage.py shell
```

```python
from web.models import PriceHistory, News, MarketState
from datetime import datetime, timedelta
import random

# Create sample price data
start_date = datetime.now().date() - timedelta(days=90)
for i in range(90):
    date = start_date + timedelta(days=i)
    base_price = 3500 + random.uniform(-200, 200)
    
    PriceHistory.objects.get_or_create(
        date=date,
        defaults={
            'open': base_price + random.uniform(-20, 20),
            'high': base_price + random.uniform(0, 50),
            'low': base_price - random.uniform(0, 50),
            'close': base_price,
            'volume': random.uniform(1000000, 2000000)
        }
    )

# Create sample market states
for i in range(90):
    date = start_date + timedelta(days=i)
    # Change state every ~15 days
    state = (i // 15) % 3
    
    MarketState.objects.get_or_create(
        date=date,
        defaults={
            'state': state,
            'probability': random.uniform(0.7, 0.95)
        }
    )

# Create sample news
sentiments = ['Positive', 'Negative', 'Neutral']
for i in range(20):
    date = datetime.now() - timedelta(days=i*2)
    sentiment = random.choice(sentiments)
    score = random.uniform(-1, 1) if sentiment == 'Negative' else random.uniform(0, 1)
    
    News.objects.get_or_create(
        title=f"Sample CPO News Article {i+1}",
        date=date,
        defaults={
            'snippet': f"This is a sample news snippet about CPO market conditions. Article {i+1} discusses various market factors.",
            'sentiment_score': score,
            'sentiment_label': sentiment,
            'url': f"https://example.com/news/{i+1}"
        }
    )

print("✅ Sample data created successfully!")
```

---

## 📁 CSV Format for Price Data

### Indonesian Format (Investing.com):

```csv
Tanggal,Terakhir,Pembukaan,Tertinggi,Terendah,Vol.,Perubahan%
17.12.2024,"3.500,00","3.450,00","3.520,00","3.440,00",1.50M,+1.45%
16.12.2024,"3.450,50","3.400,00","3.460,00","3.390,00",1.20M,+1.48%
15.12.2024,"3.400,00","3.380,00","3.410,00","3.370,00",1.10M,+0.59%
```

**Save this as `sample_cpo_data.csv` and import via Django Admin!**

---

## ✨ Key Features Implemented

### 🎨 **Frontend (Django Templates + Chart.js)**

- ✅ **Dashboard Page** ([dashboard.html](web/templates/dashboard.html))
  - Interactive Line Chart with actual & predicted prices
  - HMM State background overlay (Green=Bullish, Red=Bearish, Gray=Neutral)
  - 4 Metric cards (Current Price, MAPE, R², Accuracy)
  - Price statistics (Max, Min, Average)
  - Responsive tooltips showing OHLC data

- ✅ **News Page** ([news.html](web/templates/news.html))
  - Bootstrap cards with sentiment badges
  - Color-coded sentiment (🟢 Positive, 🔴 Negative, ⚪ Neutral)
  - Pagination (10 news per page)
  - Filter by sentiment
  - Sentiment score progress bars

- ✅ **About Page** ([about.html](web/templates/about.html))
  - Project information
  - Technologies used
  - System architecture

- ✅ **Base Template** ([base.html](web/templates/base.html))
  - Professional navbar with active states
  - Admin dropdown menu
  - Bootstrap 5 styling
  - Responsive design

### 🔧 **Backend (Django)**

- ✅ **Models** ([models.py](web/models.py))
  - `PriceHistory`: OHLC data with volume
  - `News`: Sentiment analysis results
  - `MarketState`: HMM predictions

- ✅ **Views** ([views.py](web/views.py))
  - `dashboard_view`: 3 months data + mock predictions
  - `news_view`: Pagination + sentiment filter
  - `about_view`: Static info page

- ✅ **Admin** ([admin.py](web/admin.py))
  - **Indonesian CSV Parser** 🇮🇩
  - Auto number format conversion (`3.500,00` → `3500.00`)
  - Date format support (`17.12.2024` → `2024-12-17`)
  - Import/Export functionality
  - Color-coded market states

---

## 🎯 Chart.js Implementation Details

### HMM State Overlay (Main Feature!)

The dashboard uses **Chart.js Annotation Plugin** to create colored background boxes based on HMM states:

```javascript
// Green background when State = 1 (Bullish)
// Red background when State = 0 (Bearish)
// Gray background when State = 2 (Neutral)
```

The annotations are dynamically generated based on your `MarketState` model data!

### Interactive Features:

- 📊 **Hover tooltips** showing OHLC + Volume + State Probability
- 📈 **Smooth line tension** for better visualization
- 🎨 **Gradient fills** under actual price line
- ➖ **Dashed line** for predictions
- 📱 **Responsive** chart resizing

---

## 🗂️ Database Schema

### PriceHistory Table
| Field | Type | Description |
|-------|------|-------------|
| date | Date | Unique, indexed |
| open | Float | Opening price |
| high | Float | Highest price |
| low | Float | Lowest price |
| close | Float | Closing price |
| volume | Float | Trading volume |

### News Table
| Field | Type | Description |
|-------|------|-------------|
| date | DateTime | Publication date |
| title | Text | News headline |
| snippet | Text | News excerpt |
| sentiment_score | Float | -1 to 1 |
| sentiment_label | Char | Positive/Negative/Neutral |
| url | URL | Source link |

### MarketState Table
| Field | Type | Description |
|-------|------|-------------|
| date | Date | Unique, indexed |
| state | Integer | 0=Bearish, 1=Bullish, 2=Neutral |
| probability | Float | Confidence (0-1) |

---

## 🚨 Troubleshooting

### Error: "No module named 'import_export'"
```bash
pip install django-import-export
```

### Error: "TemplateDoesNotExist"
- Make sure `'web'` is in `INSTALLED_APPS`
- Check that `APP_DIRS: True` in `TEMPLATES` settings

### Error: CSV import fails
- Ensure headers are: `Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol.`
- Check number format: `3.500,00` (dot for thousands, comma for decimal)

### Empty Dashboard Chart
- Import price data via Admin
- Run the sample data script above
- Check that dates are within last 90 days

---

## 🔄 Next Steps

### 🎯 **To Replace Mock Predictions:**

Edit [views.py](web/views.py) line ~30:

```python
# Replace this:
mock_prediction = price.close + random.uniform(-50, 50)

# With real ML model prediction:
from your_ml_module import predict_price
mock_prediction = predict_price(price.date, price.close)
```

### 🤖 **To Add Real ML Model:**

1. Train your HMM/XGBoost/RF model
2. Save model to `web/ml_models/`
3. Create `web/ml_utils.py`:
   ```python
   import joblib
   
   def load_model():
       return joblib.load('web/ml_models/hmm_model.pkl')
   
   def predict_next_prices(current_data, days=7):
       model = load_model()
       # Your prediction logic
       return predictions
   ```
4. Import in `views.py` and use instead of mock data

### 📰 **To Add Real News:**

1. Use News API or web scraping
2. Integrate FinBERT for sentiment:
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   import torch
   
   tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
   model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
   
   def analyze_sentiment(text):
       inputs = tokenizer(text, return_tensors="pt")
       outputs = model(**inputs)
       probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
       # Return sentiment
   ```

---

## 📞 Support

If you encounter issues:
1. Check [DJANGO_SETUP.md](DJANGO_SETUP.md) for detailed Django configuration
2. Verify all dependencies are installed: `pip list`
3. Check Django version: `python -m django --version`
4. Run migrations again: `python manage.py migrate`

---

## ✅ Checklist Before Running

- [ ] `pip install django-import-export openpyxl`
- [ ] `python manage.py makemigrations`
- [ ] `python manage.py migrate`
- [ ] `python manage.py createsuperuser`
- [ ] Import sample data via Admin or shell script
- [ ] `python manage.py runserver`
- [ ] Open http://localhost:8000/

---

**Created:** December 17, 2025  
**Status:** ✅ Fully Functional with Mock Data  
**Ready for:** Real ML Model Integration

🎉 **Enjoy your CPO Price Prediction Dashboard!**
