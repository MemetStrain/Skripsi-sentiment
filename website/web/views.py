"""
Views.py - Function-Based Views with Procedural Programming Style
=================================================================
All views are explicit, linear functions that directly interact with Firestore.
No class-based views, no hidden logic, no ORM queries.

Pattern: Procedural Programming
- Each view is a standalone function
- Explicit Firestore connection in each function
- Clear step-by-step data flow
- Aligned with Activity Diagrams in thesis documentation
"""
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from datetime import datetime, timedelta, date
from firebase_admin import firestore
import json
import csv
import io
from . import services
from . import prediction_service


@login_required
def dashboard(request):
    """
    PROCEDURAL VIEW: Dashboard - Display CPO Price Data with HMM States
    
    Activity Flow (Aligned with Activity Diagram):
    1. Connect to Firestore
    2. Fetch DailyMarketData collection (last 90 days)
    3. Fetch MarketStates collection (HMM results)
    4. Merge data (price + state)
    5. Calculate statistics
    6. Prepare chart data for visualization
    7. Render dashboard.html
    """
    # Step 1: Connect to Firestore explicitly
    try:
        db = firestore.client()
    except ValueError as e:
        messages.error(request, 'Firebase is not initialized. Please check your firebase-credentials.json file.')
        return render(request, 'dashboard.html', {
            'chart_data': json.dumps([]),
            'metrics': {'mape': 0, 'r2': 0, 'accuracy': 0},
            'stats': {'current_price': 0, 'avg_price': 0, 'max_price': 0, 'min_price': 0, 'total_days': 0},
            'latest_date': 'N/A',
            'page_title': 'CPO Price Prediction Dashboard'
        })
    
    # Step 2: Calculate date range (3 months)
    three_months_ago = datetime.now().date() - timedelta(days=90)
    three_months_ago_str = three_months_ago.isoformat()
    
    # Step 3: Fetch price data from Firestore DailyMarketData collection
    price_docs = db.collection('DailyMarketData').where('date', '>=', three_months_ago_str).order_by('date').stream()
    price_list = []
    for doc in price_docs:
        data = doc.to_dict()
        price_list.append({
            'date': data.get('date'),
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'close': float(data.get('close', 0)),
            'volume': float(data.get('volume', 0))
        })
    
    # Step 4: Fetch market states from Firestore MarketStates collection
    state_docs = db.collection('MarketStates').where('date', '>=', three_months_ago_str).order_by('date').stream()
    state_dict = {}
    for doc in state_docs:
        data = doc.to_dict()
        state_dict[data.get('date')] = {
            'state': int(data.get('state', 2)),
            'probability': float(data.get('probability', 0.5))
        }
    
    # Step 5: Merge price data with market states
    chart_data = []
    for price_row in price_list:
        price_date = price_row['date']
        
        # Get HMM state for this date
        state_info = state_dict.get(price_date, {'state': 2, 'probability': 0.5})
        state_value = state_info['state']
        state_prob = state_info['probability']
        
        # Map state code to label
        state_labels = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        state_label = state_labels.get(state_value, 'Unknown')
        
        chart_data.append({
            'date': price_date,
            'actual': price_row['close'],
            'open': price_row['open'],
            'high': price_row['high'],
            'low': price_row['low'],
            'volume': price_row['volume'],
            'state': state_value,
            'state_label': state_label,
            'state_probability': round(state_prob, 2)
        })
    
    # Step 6: Calculate statistics
    if price_list:
        prices = [p['close'] for p in price_list]
        stats = {
            'current_price': prices[-1] if prices else 0,
            'avg_price': sum(prices) / len(prices),
            'max_price': max(prices),
            'min_price': min(prices),
            'total_days': len(prices)
        }
        latest_date = price_list[-1]['date']
    else:
        stats = {
            'current_price': 0,
            'avg_price': 0,
            'max_price': 0,
            'min_price': 0,
            'total_days': 0
        }
        latest_date = 'N/A'
    
    # Step 7: Fetch real model metrics for horizon 1 (best model)
    try:
        all_metrics = prediction_service.get_all_horizon_metrics(horizon=1)
        # Find best model by lowest MAPE
        if all_metrics:
            best = min(all_metrics, key=lambda m: m.get('mape', float('inf')))
            real_metrics = {
                'mape': round(best.get('mape', 0), 2),
                'r2': round(best.get('r2', 0), 4),
                'accuracy': round(best.get('directional_accuracy', 0), 2),
                'best_model': f"{best.get('model', '')} ({best.get('optimization', '')})",
            }
        else:
            real_metrics = {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'}
    except Exception:
        real_metrics = {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'}
    
    # Step 8: Prepare context and render
    context = {
        'chart_data': json.dumps(chart_data),
        'metrics': real_metrics,
        'stats': stats,
        'latest_date': latest_date,
        'page_title': 'CPO Price Prediction Dashboard'
    }
    
    return render(request, 'dashboard.html', context)


@login_required
def news(request):
    """
    PROCEDURAL VIEW: News - Display CPO News with Sentiment Analysis
    
    Activity Flow:
    1. Connect to Firestore
    2. Get sentiment filter from query parameters
    3. Fetch NewsData collection from Firestore
    4. Filter by sentiment if specified
    5. Implement manual pagination
    6. Count news by sentiment
    7. Render news.html
    """
    # Step 1: Connect to Firestore
    db = firestore.client()
    
    # Step 2: Get filter and pagination parameters
    sentiment_filter = request.GET.get('sentiment', None)
    page_number = int(request.GET.get('page', 1))
    items_per_page = 9
    
    # Step 3: Fetch all news from Firestore NewsData collection
    # Note: We fetch all and filter/sort in Python to avoid needing composite index
    news_docs = db.collection('NewsData').stream()
    
    # Step 4: Convert to list and apply filter
    news_list = []
    for doc in news_docs:
        data = doc.to_dict()
        # Apply sentiment filter if specified
        if sentiment_filter and sentiment_filter in ['Positive', 'Negative', 'Neutral']:
            if data.get('sentiment_label') != sentiment_filter:
                continue  # Skip this document
        
        news_list.append({
            'id': doc.id,
            'date': data.get('date'),
            'title': data.get('title', ''),
            'snippet': data.get('snippet', ''),
            'url': data.get('url', ''),
            'sentiment_score': float(data.get('sentiment_score', 0)),
            'sentiment_label': data.get('sentiment_label', 'Neutral')
        })
    
    # Step 5: Sort by date (descending) in Python
    news_list.sort(key=lambda x: x['date'] if x['date'] else '', reverse=True)
    
    # Step 6: Manual pagination
    total_news = len(news_list)
    total_pages = (total_news + items_per_page - 1) // items_per_page  # Ceiling division
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    news_page_list = news_list[start_idx:end_idx]
    
    # Step 7: Count by sentiment (fetch all to count)
    all_news_docs = db.collection('NewsData').stream()
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
    for doc in all_news_docs:
        data = doc.to_dict()
        label = data.get('sentiment_label', 'Neutral')
        sentiment_counts['total'] += 1
        if label == 'Positive':
            sentiment_counts['positive'] += 1
        elif label == 'Negative':
            sentiment_counts['negative'] += 1
        elif label == 'Neutral':
            sentiment_counts['neutral'] += 1
    
    # Step 8: Prepare pagination info
    pagination = {
        'current_page': page_number,
        'total_pages': total_pages,
        'has_previous': page_number > 1,
        'has_next': page_number < total_pages,
        'previous_page': page_number - 1 if page_number > 1 else None,
        'next_page': page_number + 1 if page_number < total_pages else None,
        'page_range': range(1, total_pages + 1)  # For template loop
    }
    
    # Step 9: Render template
    context = {
        'news_list': news_page_list,
        'news_page': news_page_list,  # Template uses news_page
        'sentiment_counts': sentiment_counts,
        'current_filter': sentiment_filter,
        'pagination': pagination,
        'page_title': 'CPO News & Sentiment Analysis'
    }
    
    return render(request, 'news.html', context)


def about(request):
    """
    PROCEDURAL VIEW: About - Display Project Information
    
    Simple static page, no database interaction needed.
    """
    context = {
        'page_title': 'About This Project'
    }
    return render(request, 'about.html', context)


@require_http_methods(["GET", "POST"])
def register(request):
    """
    PROCEDURAL VIEW: Register - Create New User Account

    Activity Flow:
    1. If GET: Display registration form
    2. If POST: Receive POST data (username, email, password)
    3. Validate input fields
    4. Check for existing user
    5. Create user in Django Auth system (SQLite)
    6. Redirect to login page

    Note: User authentication uses Django ORM (SQLite) as per requirements.
    Only data storage uses Firestore.
    """
    # Step 0: Handle GET request (display form)
    if request.method == 'GET':
        return render(request, 'register.html', {'page_title': 'Register'})

    # Step 1: Get form data
    username = request.POST.get('username', '').strip()
    email = request.POST.get('email', '').strip()
    password = request.POST.get('password', '')
    password_confirm = request.POST.get('password_confirm', '')
    
    # Step 2: Validate inputs (explicit validation logic)
    errors = []
    
    if not username or not email or not password or not password_confirm:
        errors.append('All fields are required.')
    
    if password != password_confirm:
        errors.append('Passwords do not match.')
    
    if len(password) < 8:
        errors.append('Password must be at least 8 characters long.')
    
    # Step 3: Check for existing user
    if User.objects.filter(username=username).exists():
        errors.append('Username already exists.')
    
    if User.objects.filter(email=email).exists():
        errors.append('Email already registered.')
    
    # Step 4: If validation errors, show messages and redirect
    if errors:
        for error in errors:
            messages.error(request, error)
        return redirect('register')
    
    # Step 5: Create user in Django Auth system
    try:
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        messages.success(request, 'Account created successfully! You can now login.')
        return redirect('login')
    except Exception as e:
        messages.error(request, f'Error creating account: {str(e)}')
        return redirect('register')


@login_required
def admin_upload_price(request):
    """
    PROCEDURAL VIEW: Admin Upload - Update CPO Price Data from CSV
    
    This is the CRITICAL "Update Price" feature mentioned in the thesis.
    
    Activity Flow (EXPLICIT PROCEDURAL STEPS):
    1. Check if request method is GET or POST
    2. If GET: Display upload form
    3. If POST: Handle CSV upload
    4. Validate CSV file (check headers)
    5. Parse CSV data (use services.parse_indonesian_csv)
    6. Batch write to Firestore DailyMarketData collection
    7. CRUCIAL: Call calculate_hmm_state() utility function immediately
    8. Update Firestore MarketStates collection with HMM results
    9. Show success message with statistics
    10. Redirect to dashboard
    """
    # Step 1: Handle GET request (display form)
    if request.method == 'GET':
        context = {
            'page_title': 'Upload CPO Price Data'
        }
        return render(request, 'upload_price.html', context)
    
    # Step 2: Handle POST request (process upload)
    if request.method == 'POST':
        # Step 3: Get uploaded file
        uploaded_file = request.FILES.get('csv_file')
        
        if not uploaded_file:
            messages.error(request, 'No file uploaded.')
            return redirect('admin_upload_price')
        
        # Step 4: Validate file extension
        if not uploaded_file.name.endswith('.csv'):
            messages.error(request, 'File must be a CSV file.')
            return redirect('admin_upload_price')
        
        # Step 5: Parse CSV using services module
        try:
            parsed_data = services.parse_indonesian_csv(uploaded_file)
        except Exception as e:
            messages.error(request, f'Error parsing CSV: {str(e)}')
            return redirect('admin_upload_price')
        
        if not parsed_data:
            messages.error(request, 'No valid data found in CSV.')
            return redirect('admin_upload_price')
        
        # Step 6: Connect to Firestore
        db = firestore.client()
        
        # Step 7: Batch write to DailyMarketData collection
        batch = db.batch()
        success_count = 0
        update_count = 0
        
        for row in parsed_data:
            # Use date as document ID
            doc_id = row['date'].isoformat()
            doc_ref = db.collection('DailyMarketData').document(doc_id)
            
            # Check if document exists
            doc_snapshot = doc_ref.get()
            
            # Prepare data
            data = {
                'date': row['date'].isoformat(),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'updated_at': datetime.now().isoformat()
            }
            
            if doc_snapshot.exists:
                # Update existing document
                batch.update(doc_ref, data)
                update_count += 1
            else:
                # Create new document
                data['created_at'] = datetime.now().isoformat()
                batch.set(doc_ref, data)
                success_count += 1
        
        # Commit batch write
        try:
            batch.commit()
        except Exception as e:
            messages.error(request, f'Error writing to Firestore: {str(e)}')
            return redirect('admin_upload_price')
        
        # Step 8: CRUCIAL - Trigger HMM Calculation immediately after upload
        try:
            hmm_count = calculate_hmm_state()
            messages.success(
                request,
                f'Upload successful! Added: {success_count}, Updated: {update_count}. '
                f'HMM states calculated for {hmm_count} days.'
            )
        except Exception as e:
            messages.warning(
                request,
                f'Data uploaded successfully ({success_count} added, {update_count} updated), '
                f'but HMM calculation failed: {str(e)}'
            )
        
        return redirect('dashboard')


def calculate_hmm_state():
    """
    UTILITY FUNCTION: Calculate HMM States and Save to Firestore
    
    This function is called immediately after price data upload.
    
    Procedural Steps:
    1. Fetch all price data from Firestore (sorted by date)
    2. Calculate daily returns
    3. Calculate volatility
    4. Determine market state (Bearish=0, Bullish=1, Neutral=2)
    5. Calculate confidence probability
    6. Batch write to MarketStates collection
    7. Return count of processed states
    """
    # Step 1: Connect to Firestore
    db = firestore.client()
    
    # Step 2: Fetch all price data (sorted by date ascending)
    price_docs = db.collection('DailyMarketData').order_by('date').stream()
    price_list = []
    for doc in price_docs:
        data = doc.to_dict()
        price_list.append({
            'date': data.get('date'),
            'close': float(data.get('close', 0))
        })
    
    # Step 3: Calculate HMM states
    if len(price_list) < 2:
        return 0
    
    states_to_save = []
    
    for i in range(1, len(price_list)):
        # Step 4: Calculate daily return
        prev_close = price_list[i - 1]['close']
        curr_close = price_list[i]['close']
        
        if prev_close == 0:
            continue
        
        daily_return = (curr_close - prev_close) / prev_close
        volatility = abs(daily_return)
        
        # Step 5: Determine state based on business rules
        return_pct = daily_return * 100
        volatility_pct = volatility * 100
        
        # Initialize with neutral
        state = 2
        probability = 0.5
        
        # Bullish detection
        if return_pct > 0.5 and volatility_pct < 2.0:
            state = 1
            confidence_boost = min(return_pct / 5.0, 0.3)
            probability = 0.6 + confidence_boost
        
        # Bearish detection
        elif return_pct < -0.5 and volatility_pct > 1.0:
            state = 0
            confidence_boost = min(abs(return_pct) / 5.0, 0.3)
            probability = 0.6 + confidence_boost
        
        # Neutral
        else:
            state = 2
            probability = 0.5
        
        # Clamp probability
        probability = max(0.4, min(probability, 0.99))
        
        states_to_save.append({
            'date': price_list[i]['date'],
            'state': state,
            'probability': round(probability, 2)
        })
    
    # Step 6: Batch write to MarketStates collection
    batch = db.batch()
    
    for state_data in states_to_save:
        doc_id = state_data['date']
        doc_ref = db.collection('MarketStates').document(doc_id)
        
        data = {
            'date': state_data['date'],
            'state': state_data['state'],
            'probability': state_data['probability'],
            'updated_at': datetime.now().isoformat()
        }
        
        batch.set(doc_ref, data, merge=True)
    
    # Commit batch
    batch.commit()
    
    return len(states_to_save)


@login_required
def predict_price(request):
    """
    PROCEDURAL VIEW: Predict - Horizon-based ML Price Prediction

    Activity Flow:
    1. GET: Display prediction form with model/variant/horizon options
    2. POST: Run real ML prediction via prediction_service
    3. Return JSON with predicted price and model metrics
    """
    model_types = ['xgboost', 'random_forest', 'arimax', 'sarimax']
    variants = ['base', 'csa']
    horizons = [1, 2, 3, 4, 5, 6, 7]

    # Step 1: Handle GET request
    if request.method == 'GET':
        context = {
            'page_title': 'CPO Price Prediction',
            'model_types': model_types,
            'variants': variants,
            'horizons': horizons,
        }
        return render(request, 'predict.html', context)

    # Step 2: Handle POST request
    if request.method == 'POST':
        try:
            model_type = request.POST.get('model_type', 'xgboost')
            variant = request.POST.get('variant', 'csa')
            horizon = int(request.POST.get('horizon', 1))
        except (ValueError, TypeError):
            return JsonResponse({'error': 'Invalid input parameters'}, status=400)

        # Validate
        if model_type not in model_types:
            return JsonResponse({'error': f'Invalid model type. Choose from: {model_types}'}, status=400)
        if variant not in variants:
            return JsonResponse({'error': f'Invalid variant. Choose from: {variants}'}, status=400)
        if horizon not in horizons:
            return JsonResponse({'error': f'Invalid horizon. Choose from: {horizons}'}, status=400)

        # Step 3: Run prediction
        try:
            result = prediction_service.run_prediction(
                model_type=model_type,
                variant=variant,
                horizon=horizon,
            )
        except Exception as e:
            return JsonResponse({'error': f'Prediction failed: {str(e)}'}, status=500)

        # Step 4: Format response
        response_data = {
            'success': True,
            'model_type': model_type,
            'variant': variant,
            'horizon': horizon,
            'last_actual_price': result['last_close'],
            'last_actual_date': result['last_date'],
            'predicted_price': result['predicted_price'],
            'metrics': result['metrics'],
        }

        return JsonResponse(response_data)
