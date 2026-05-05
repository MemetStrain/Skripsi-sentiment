"""
Views — CPO Prediction Dashboard
=================================
All views are explicit, linear functions reading from Firestore.
Public-facing read-only site; no authentication required.
"""
from django.shortcuts import render
from django.contrib import messages
from django.http import JsonResponse
from datetime import datetime, timedelta
from firebase_admin import firestore
import json


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def dashboard(request):
    """Display CPO price chart with HMM market states."""
    try:
        db = firestore.client()
    except ValueError:
        messages.error(request, 'Firebase is not initialized.')
        return render(request, 'dashboard.html', _empty_dashboard_ctx())

    three_months_ago = (datetime.now().date() - timedelta(days=90)).isoformat()

    # Fetch price data from new `daily_prices` collection
    price_docs = (
        db.collection('daily_prices')
        .where('date', '>=', three_months_ago)
        .order_by('date')
        .stream()
    )
    price_list = []
    for doc in price_docs:
        d = doc.to_dict()
        price_list.append({
            'date': d.get('date'),
            'open': float(d.get('open', 0)),
            'high': float(d.get('high', 0)),
            'low': float(d.get('low', 0)),
            'close': float(d.get('close', 0)),
            'volume': float(d.get('volume', 0)),
        })

    # Fetch HMM states from `hmm_states` (Daily frequency).
    # Doc IDs are `Daily_YYYY-MM-DD` so we filter by date in Python
    # to avoid a composite Firestore index requirement.
    state_docs = (
        db.collection('hmm_states')
        .where('date', '>=', three_months_ago)
        .order_by('date')
        .stream()
    )
    state_dict = {}
    for doc in state_docs:
        d = doc.to_dict()
        if d.get('frequency') not in (None, 'Daily'):
            continue
        state_dict[d.get('date')] = {
            'state_label': d.get('state_label', 'Neutral'),
            'state': d.get('state', 2),
        }

    # Build chart data
    chart_data = []
    for row in price_list:
        state_info = state_dict.get(row['date'], {'state': 2, 'state_label': 'Neutral'})
        state_label = state_info['state_label']
        state_value = state_info['state']
        # Map label to numeric for colour coding (0=Bearish,1=Bullish,2=Neutral)
        label_to_int = {'Bearish': 0, 'Bullish': 1, 'Neutral': 2}
        chart_data.append({
            'date': row['date'],
            'actual': row['close'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'state': label_to_int.get(state_label, state_value),
            'state_label': state_label,
        })

    # Stats
    if price_list:
        prices = [p['close'] for p in price_list]
        stats = {
            'current_price': prices[-1],
            'avg_price': sum(prices) / len(prices),
            'max_price': max(prices),
            'min_price': min(prices),
            'total_days': len(prices),
        }
        latest_date = price_list[-1]['date']
    else:
        stats = {'current_price': 0, 'avg_price': 0, 'max_price': 0, 'min_price': 0, 'total_days': 0}
        latest_date = 'N/A'

    # Best metric from pre-computed predictions (horizon 1 Daily, lowest MAPE).
    # Fetch by constructing known doc IDs to avoid composite index requirements.
    metrics = {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'}
    try:
        best_mape = float('inf')
        for model in ('xgboost',):
            for variant in ('base', 'csa'):
                doc_id = f'{model}_{variant}_Daily_h1'
                doc = db.collection('predictions').document(doc_id).get()
                if not doc.exists:
                    continue
                d = doc.to_dict()
                m = d.get('metrics', {})
                mape = m.get('mape', float('inf'))
                if mape < best_mape:
                    best_mape = mape
                    metrics = {
                        'mape': round(mape, 2),
                        'r2': round(m.get('r2', 0), 4),
                        'accuracy': round(m.get('directional_accuracy', 0), 2),
                        'best_model': f"{d.get('model', '')} ({d.get('variant', '')})",
                    }
    except Exception:
        pass

    return render(request, 'dashboard.html', {
        'chart_data': json.dumps(chart_data),
        'metrics': metrics,
        'stats': stats,
        'latest_date': latest_date,
        'page_title': 'CPO Price Prediction Dashboard',
    })


def _empty_dashboard_ctx():
    return {
        'chart_data': json.dumps([]),
        'metrics': {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'},
        'stats': {'current_price': 0, 'avg_price': 0, 'max_price': 0, 'min_price': 0, 'total_days': 0},
        'latest_date': 'N/A',
        'page_title': 'CPO Price Prediction Dashboard',
    }


# ---------------------------------------------------------------------------
# Prediction API  (called by dashboard JS)
# ---------------------------------------------------------------------------

def prediction_api(request):
    """
    GET /api/prediction/?model=xgboost&variant=csa&frequency=Daily&horizon=1
    Returns pre-computed prediction from Firestore `predictions` collection.
    """
    model = request.GET.get('model', 'xgboost').lower()
    variant = request.GET.get('variant', 'csa').lower()
    frequency = request.GET.get('frequency', 'Daily')
    try:
        horizon = int(request.GET.get('horizon', 1))
    except (ValueError, TypeError):
        return JsonResponse({'error': 'Invalid horizon'}, status=400)

    valid_models = {'xgboost'}
    valid_variants = {'base', 'csa'}
    valid_freqs = {'Daily'}
    if model not in valid_models or variant not in valid_variants or frequency not in valid_freqs:
        return JsonResponse(
            {'error': f"Invalid parameters. Supported: model={sorted(valid_models)}, "
                      f"variant={sorted(valid_variants)}, frequency={sorted(valid_freqs)}"},
            status=400,
        )

    doc_id = f'{model}_{variant}_{frequency}_h{horizon}'
    try:
        db = firestore.client()
        doc = db.collection('predictions').document(doc_id).get()
        if not doc.exists:
            return JsonResponse({'error': 'Prediction not available yet'}, status=404)
        data = doc.to_dict()
        return JsonResponse({
            'success': True,
            'model': data.get('model'),
            'variant': data.get('variant'),
            'frequency': data.get('frequency'),
            'horizon': data.get('horizon'),
            'last_actual_date': data.get('last_actual_date'),
            'last_actual_price': data.get('last_actual_price'),
            'predicted_date': data.get('predicted_date'),
            'predicted_price': data.get('predicted_price'),
            'metrics': data.get('metrics', {}),
            'computed_at': data.get('computed_at'),
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def news(request):
    """Display news articles from `news_articles` collection."""
    db = firestore.client()

    sentiment_filter = request.GET.get('sentiment')
    page_number = int(request.GET.get('page', 1))
    items_per_page = 9

    # Fetch all and sort in Python (avoids composite index requirement)
    news_docs = db.collection('news_articles').stream()
    news_list = []
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}

    for doc in news_docs:
        d = doc.to_dict()
        label = d.get('sentiment_label', 'Neutral')

        # Count totals before filtering
        sentiment_counts['total'] += 1
        if label == 'Positive':
            sentiment_counts['positive'] += 1
        elif label == 'Negative':
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1

        if sentiment_filter and label != sentiment_filter:
            continue

        # First paragraph: use the snippet field (pre-computed by scheduler)
        snippet = d.get('snippet', '')
        if not snippet and d.get('content'):
            # Fallback: first 200 chars of content
            snippet = d['content'][:200].rsplit(' ', 1)[0] + '…'

        news_list.append({
            'date': d.get('date', ''),
            'title': d.get('title', ''),
            'category': d.get('category', ''),
            'snippet': snippet,
            'url': d.get('url', '#'),
            'sentiment_label': label,
            'sentiment_score': float(d.get('sentiment_score', 0)),
        })

    news_list.sort(key=lambda x: x['date'], reverse=True)

    total_news = len(news_list)
    total_pages = max(1, (total_news + items_per_page - 1) // items_per_page)
    start = (page_number - 1) * items_per_page
    news_page = news_list[start:start + items_per_page]

    pagination = {
        'current_page': page_number,
        'total_pages': total_pages,
        'has_previous': page_number > 1,
        'has_next': page_number < total_pages,
        'previous_page': page_number - 1 if page_number > 1 else None,
        'next_page': page_number + 1 if page_number < total_pages else None,
        'page_range': range(1, total_pages + 1),
    }

    return render(request, 'news.html', {
        'news_page': news_page,
        'sentiment_counts': sentiment_counts,
        'current_filter': sentiment_filter,
        'pagination': pagination,
        'page_title': 'CPO News & Sentiment',
    })


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def about(request):
    return render(request, 'about.html', {'page_title': 'About'})
