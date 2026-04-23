"""
news_extractor.py — Scrape MPOB news articles and load them to Firestore.

Initial load: reads mpob_news_with_sentiment.csv (already has sentiment).
Incremental: scrapes MPOB website for articles newer than the last stored date.
"""

import csv
import logging
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MPOB_SEARCH_URL = (
    'https://prestasisawit.mpob.gov.my/en/palmnews/advance-search/result'
)
SEARCH_KEYWORDS = ['cpo', 'crude palm oil']
REQUEST_TIMEOUT = 15
MAX_WORKERS = 4
RETRY_LIMIT = 3


# ---------------------------------------------------------------------------
# Initial load: read the pre-processed CSV
# ---------------------------------------------------------------------------

def load_news_from_csv(csv_path: str, sentiment_csv_path: Optional[str] = None) -> list[dict]:
    """
    Read MPOB news from the pre-processed CSV.
    If sentiment_csv_path is given (mpob_news_with_sentiment.csv), use that
    so we have sentiment fields without re-running FinBERT.
    """
    path = sentiment_csv_path if sentiment_csv_path and os.path.exists(sentiment_csv_path) else csv_path
    if not os.path.exists(path):
        logger.warning(f'News CSV not found: {path}')
        return []

    articles = []
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalise date to YYYY-MM-DD
                raw_date = row.get('Date', row.get('date', ''))
                norm_date = _normalise_date(raw_date)
                if not norm_date or norm_date < '2014-01-01':
                    continue

                content = row.get('Content', row.get('content', ''))
                snippet = row.get('Snippet', row.get('snippet', ''))
                if not snippet and content:
                    snippet = content[:200].rsplit(' ', 1)[0] + '…'

                article = {
                    'date': norm_date,
                    'title': row.get('Title', row.get('title', '')).strip(),
                    'category': row.get('Category', row.get('category', '')).strip(),
                    'content': content,
                    'snippet': snippet,
                    'url': row.get('URL', row.get('url', '')).strip(),
                }

                # Attach pre-computed sentiment if available.
                # The sentiment CSV uses Combined_* column names; map them to
                # the same field names the daily pipeline produces so Firestore
                # documents are consistent regardless of load mode.
                _csv_col_map = {
                    'Combined_Sentiment':     'sentiment_label',
                    'Combined_Positive_Prob': 'positive_prob',
                    'Combined_Negative_Prob': 'negative_prob',
                    'Combined_Neutral_Prob':  'neutral_prob',
                    # Also accept already-normalised names (e.g. reprocessed CSVs)
                    'sentiment_label':        'sentiment_label',
                    'positive_prob':          'positive_prob',
                    'negative_prob':          'negative_prob',
                    'neutral_prob':           'neutral_prob',
                }
                for csv_col, field in _csv_col_map.items():
                    if csv_col not in row:
                        continue
                    val = row[csv_col].strip() if isinstance(row[csv_col], str) else row[csv_col]
                    if field == 'sentiment_label':
                        article[field] = val
                    else:
                        try:
                            article[field] = round(float(val), 4)
                        except (ValueError, TypeError):
                            pass

                # Compute sentiment_score consistent with the daily pipeline:
                # score = positive_prob - negative_prob  (range -1 to +1)
                if 'positive_prob' in article and 'negative_prob' in article:
                    article['sentiment_score'] = round(
                        article['positive_prob'] - article['negative_prob'], 4
                    )

                if article['url']:
                    articles.append(article)

    except Exception as e:
        logger.error(f'Error reading news CSV: {e}')

    logger.info(f'Loaded {len(articles)} articles from {path}')
    return articles


def _normalise_date(raw: str) -> Optional[str]:
    """Return YYYY-MM-DD or None."""
    for fmt in ('%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d %b %Y', '%B %d, %Y'):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Incremental scrape: articles newer than a cutoff date
# ---------------------------------------------------------------------------

def scrape_new_articles(cutoff_date: Optional[str]) -> list[dict]:
    """
    Scrape MPOB news articles published after `cutoff_date` (YYYY-MM-DD).
    Returns a list of dicts (without sentiment — caller runs FinBERT on these).

    If cutoff_date is None (initial scrape), this function returns [] because
    the initial load uses the CSV files directly.
    """
    if cutoff_date is None:
        return []

    logger.info(f'Scraping MPOB news newer than {cutoff_date}')
    articles = []

    for keyword in SEARCH_KEYWORDS:
        page = 1
        while True:
            batch = _scrape_page(keyword, page, cutoff_date)
            if batch is None:
                break  # error or end of results
            if len(batch) == 0:
                break  # no more new articles on this page
            articles.extend(batch)
            page += 1
            time.sleep(random.uniform(0.5, 1.5))

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for a in articles:
        if a['url'] not in seen_urls:
            seen_urls.add(a['url'])
            unique.append(a)

    logger.info(f'Scraped {len(unique)} new articles')
    return unique


def _scrape_page(keyword: str, page: int, cutoff_date: str) -> Optional[list[dict]]:
    """
    Scrape one result page for the given keyword.
    Returns [] if all articles on this page are older than cutoff_date.
    Returns None on error.
    """
    params = {
        'q': keyword,
        'page': page,
        'per_page': 20,
    }
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(MPOB_SEARCH_URL, params=params, headers=headers,
                                timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return []  # no more pages
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == RETRY_LIMIT - 1:
                logger.warning(f'Failed to fetch page {page} for "{keyword}": {e}')
                return None
            time.sleep(2 ** attempt)

    articles = []
    try:
        soup = BeautifulSoup(resp.text, 'html.parser')
        result_items = soup.select('.news-item, .result-item, article.news')

        if not result_items:
            return []  # no articles found

        for item in result_items:
            date_el = item.select_one('.date, time, .news-date')
            title_el = item.select_one('h2, h3, .title, .news-title')
            link_el = item.select_one('a[href]')
            category_el = item.select_one('.category, .tag, .news-category')
            snippet_el = item.select_one('p, .snippet, .excerpt')

            if not (date_el and title_el and link_el):
                continue

            raw_date = date_el.get_text(strip=True)
            norm_date = _normalise_date(raw_date)
            if not norm_date:
                continue

            # Stop if article is not newer than cutoff
            if norm_date <= cutoff_date:
                return []  # earlier articles won't be newer either

            url = link_el['href']
            if not url.startswith('http'):
                url = 'https://prestasisawit.mpob.gov.my' + url

            content = _fetch_article_content(url, headers)
            snippet = snippet_el.get_text(strip=True)[:250] if snippet_el else ''

            articles.append({
                'date': norm_date,
                'title': title_el.get_text(strip=True),
                'category': category_el.get_text(strip=True) if category_el else 'General',
                'content': content,
                'snippet': snippet or (content[:200] + '…' if content else ''),
                'url': url,
            })

    except Exception as e:
        logger.warning(f'Parse error on page {page}: {e}')
        return []

    return articles


def _fetch_article_content(url: str, headers: dict) -> str:
    """Fetch the full text of a single article."""
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Look for common article body selectors
            body = (
                soup.select_one('.article-body, .news-content, .entry-content, article')
                or soup.select_one('main')
            )
            if body:
                for tag in body.select('script, style, nav, footer'):
                    tag.decompose()
                return body.get_text(' ', strip=True)
            return ''
        except Exception:
            time.sleep(1)
    return ''
