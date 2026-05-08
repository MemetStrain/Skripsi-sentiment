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
        state_info = state_dict.get(row['date'], {'state': None, 'state_label': None})
        state_label = state_info['state_label']
        # Map label to numeric for colour coding (0=Bearish,1=Bullish,2=Neutral).
        # None for dates with no HMM entry → no background shading (white).
        label_to_int = {'Bearish': 0, 'Bullish': 1, 'Neutral': 2}
        chart_data.append({
            'date': row['date'],
            'actual': row['close'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'state': label_to_int.get(state_label) if state_label else None,
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

    # Predictions are computed live by /api/forecasts/. We render the static
    # 4×7 metrics-comparison table server-side from prediction/winners.json
    # and a small summary badge for h=1's winner; the chart's rolling-forecast
    # overlay is fetched async by the dashboard JS.
    metrics = {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'}
    winners_payload = {}
    try:
        from .predictor import load_winners
        winners_payload = load_winners()
        h1_tag = winners_payload.get('winners_by_horizon', {}).get('1')
        if h1_tag:
            h1_metrics = (
                winners_payload.get('metrics', {})
                .get(h1_tag, {})
                .get('1', {})
                .get('CSA', {})
            )
            if h1_metrics:
                metrics = {
                    'mape':       round(h1_metrics.get('mape', 0), 2),
                    'r2':         round(h1_metrics.get('r2_price', 0), 4),
                    'accuracy':   round(h1_metrics.get('da', 0), 2),
                    'best_model': f"XGBoost CSA ({winners_payload.get('configs_by_horizon', {}).get('1', '?')})",
                }
    except FileNotFoundError:
        # winners.json not produced yet — page still renders with N/A metrics.
        pass

    # Sentiment trend from `sentiment_aggregates` (Daily, 90-day window).
    sentiment_list = []
    try:
        sent_docs = (
            db.collection('sentiment_aggregates')
            .where('date', '>=', three_months_ago)
            .order_by('date')
            .stream()
        )
        for doc in sent_docs:
            d = doc.to_dict()
            if d.get('frequency') != 'Daily':
                continue
            sentiment_list.append({
                'date':            d.get('date'),
                'positive_prob':   round(float(d.get('positive_prob', 0)), 4),
                'negative_prob':   round(float(d.get('negative_prob', 0)), 4),
                'neutral_prob':    round(float(d.get('neutral_prob',  0)), 4),
                'sentiment_score': round(float(d.get('sentiment_score', 0)), 4),
            })
    except Exception:
        pass

    return render(request, 'dashboard.html', {
        'chart_data':      json.dumps(chart_data),
        'metrics':         metrics,
        'winners_data':    json.dumps(winners_payload),
        'sentiment_data':  json.dumps(sentiment_list),
        'stats':           stats,
        'latest_date':     latest_date,
        'page_title':      'CPO Price Prediction Dashboard',
    })


def _empty_dashboard_ctx():
    return {
        'chart_data':     json.dumps([]),
        'metrics':        {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'},
        'winners_data':   json.dumps({}),
        'sentiment_data': json.dumps([]),
        'stats':          {'current_price': 0, 'avg_price': 0, 'max_price': 0, 'min_price': 0, 'total_days': 0},
        'latest_date':    'N/A',
        'page_title':     'CPO Price Prediction Dashboard',
    }


# ---------------------------------------------------------------------------
# Forecasts API  (called by dashboard JS for the rolling-trail chart overlay)
# ---------------------------------------------------------------------------

def forecasts_api(request):
    """
    GET /api/forecasts/?max_horizon=7&window_days=90

    Runs live XGBoost inference for h ∈ {1..max_horizon} using the
    auto-picked winning ablation config per horizon (lowest base-MAPE
    from prediction/winners.json), and returns rolling forecast trails
    over the trailing `window_days` window.
    """
    try:
        max_horizon = int(request.GET.get('max_horizon', 7))
        window_days = int(request.GET.get('window_days', 90))
    except (ValueError, TypeError):
        return JsonResponse({'error': 'Invalid integer parameter'}, status=400)
    max_horizon = max(1, min(max_horizon, 7))
    window_days = max(7, min(window_days, 365))

    try:
        from .predictor import compute_forecast_trails
        db = firestore.client()
        payload = compute_forecast_trails(db, max_horizon=max_horizon,
                                          window_days=window_days)
        return JsonResponse(payload)
    except FileNotFoundError as e:
        return JsonResponse({'error': str(e)}, status=503)
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
    page_number = max(1, min(page_number, total_pages))
    start = (page_number - 1) * items_per_page
    news_page = news_list[start:start + items_per_page]

    window_size = 5
    window_start = max(1, page_number - window_size // 2)
    window_end = min(total_pages, window_start + window_size - 1)
    window_start = max(1, window_end - window_size + 1)

    pagination = {
        'current_page': page_number,
        'total_pages': total_pages,
        'has_previous': page_number > 1,
        'has_next': page_number < total_pages,
        'previous_page': page_number - 1 if page_number > 1 else None,
        'next_page': page_number + 1 if page_number < total_pages else None,
        'page_range': range(window_start, window_end + 1),
        'show_left_ellipsis': window_start > 1,
        'show_right_ellipsis': window_end < total_pages,
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
