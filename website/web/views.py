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

    # Forecasts (and the winners/metrics tree the metrics-comparison table
    # renders from) are precomputed offline by the scheduler and live in
    # forecast_meta/Daily — we never read the filesystem winners.json on
    # Vercel. The metrics summary badge for h=1 is rendered server-side here;
    # the 4×7 table and the chart overlay are fetched/built client-side from
    # `winners_data` and /api/forecasts/.
    metrics = {'mape': 0, 'r2': 0, 'accuracy': 0, 'best_model': 'N/A'}
    winners_payload = {}
    try:
        meta_doc = db.collection('forecast_meta').document('Daily').get()
        if meta_doc.exists:
            meta_raw = meta_doc.to_dict() or {}
            winners_payload = json.loads(meta_raw.get('payload_json', '{}'))
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
    except Exception:
        # forecast_meta/Daily not produced yet — page still renders with N/A metrics.
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

    Returns the rolling forecast trails precomputed by the local scheduler
    (scheduler/precompute_forecasts.py) and stored in Firestore as
    per-(horizon, anchor) docs in the `forecasts` collection plus the
    `forecast_meta/Daily` summary. XGBoost inference is too heavy to run on
    Vercel's serverless functions, so it runs offline and the site only reads
    the result here.

    Response shape (unchanged from the old live path, so the dashboard JS
    needs no changes):
        {horizons, winners, configs, trails:[{horizon,tag,config,points:[...]}],
         metrics, tag_to_config, generated_at}
    """
    try:
        max_horizon = int(request.GET.get('max_horizon', 7))
    except (TypeError, ValueError):
        max_horizon = 7
    max_horizon = max(1, min(7, max_horizon))

    try:
        window_days = int(request.GET.get('window_days', 90))
    except (TypeError, ValueError):
        window_days = 90
    window_days = max(7, min(365, window_days))

    try:
        db = firestore.client()

        meta_doc = db.collection('forecast_meta').document('Daily').get()
        if not meta_doc.exists:
            return JsonResponse(
                {'error': 'forecast_meta/Daily not found. '
                          'Run scheduler/main.py to populate forecasts.'},
                status=503,
            )
        meta_raw = meta_doc.to_dict() or {}
        meta_payload = json.loads(meta_raw.get('payload_json', '{}'))

        cutoff = (datetime.now().date() - timedelta(days=window_days)).isoformat()

        # Mirrors how `news` filters in Python instead of using composite
        # Firestore indexes: stream once, filter on the fly.
        trails_by_h: dict[int, list[dict]] = {}
        for d in db.collection('forecasts').stream():
            x = d.to_dict() or {}
            if (x.get('frequency') or 'Daily') != 'Daily':
                continue
            h = int(x.get('horizon', 0))
            if h < 1 or h > max_horizon:
                continue
            pred_date = x.get('predicted_date', '')
            if not pred_date or pred_date < cutoff:
                continue
            trails_by_h.setdefault(h, []).append({
                'anchor_date':     x.get('anchor_date', ''),
                'anchor_price':    round(float(x.get('anchor_price', 0.0)), 2),
                'predicted_date':  pred_date,
                'predicted_price': round(float(x.get('predicted_price', 0.0)), 2),
                'log_return':      round(float(x.get('log_return', 0.0)), 6),
            })

        if not trails_by_h:
            return JsonResponse(
                {'error': 'forecasts collection empty. '
                          'Run scheduler/main.py to populate forecasts.'},
                status=503,
            )

        # Stored horizon keys are JSON strings (forecast_meta.payload_json) —
        # the old payload also used string keys here, so no int-cast needed.
        winners = meta_payload.get('winners_by_horizon', {})
        configs = meta_payload.get('configs_by_horizon', {})

        trails = []
        for h in sorted(trails_by_h.keys()):
            points = sorted(trails_by_h[h], key=lambda p: p['anchor_date'])
            trails.append({
                'horizon': h,
                'tag':     winners.get(str(h), ''),
                'config':  configs.get(str(h), ''),
                'points':  points,
            })

        return JsonResponse({
            'horizons':      meta_payload.get('horizons', list(range(1, max_horizon + 1))),
            'winners':       winners,
            'configs':       configs,
            'trails':        trails,
            'metrics':       meta_payload.get('metrics', {}),
            'tag_to_config': meta_payload.get('tag_to_config', {}),
            'generated_at':  meta_raw.get('generated_at', ''),
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
