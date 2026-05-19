"""
news_extractor.py — Scrape MPOB news articles and load them to Firestore.

Initial load: reads mpob_news_with_sentiment_tone.csv (already has sentiment).
Incremental: scrapes MPOB website for articles newer than the last stored date,
preprocesses them, and exposes helpers for the scheduler to append the full
article record to all three local CSVs (raw / preprocessed / sentiment).
"""

import csv
import logging
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# MPOB's advance-search result page. The `_token` is required by the site;
# pagination is followed via the `rel="next"` link on each page rather than a
# page-number param. The selectors below target the live markup — keep them in
# sync with news/scrap_fast.py, the standalone scraper they were ported from.
MPOB_BASE = 'https://prestasisawit.mpob.gov.my'
_MPOB_TOKEN = 'YjvdaKzN9niwad2HbnwQZYgeEmK92dphtzmqeNuU'
MPOB_SEARCH_URLS = [
    f'{MPOB_BASE}/en/palmnews/advance-search/result'
    f'?_token={_MPOB_TOKEN}&keyword=cpo',
    f'{MPOB_BASE}/en/palmnews/advance-search/result'
    f'?_token={_MPOB_TOKEN}&keyword=crude%20palm%20oil',
]
REQUEST_TIMEOUT = (10, 30)   # (connect, read) seconds
MAX_WORKERS = 8
RETRY_LIMIT = 3

# MPOB serves an incomplete cert chain; scrap_fast.py uses verify=False too.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;q=0.9,'
        'image/avif,image/webp,*/*;q=0.8'
    ),
}

# Make the standalone news_preprocessing module importable from `news/`.
_NEWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'news'))
if _NEWS_DIR not in sys.path:
    sys.path.insert(0, _NEWS_DIR)


# ---------------------------------------------------------------------------
# Preprocessing — bridge to news/news_preprocessing.py NewsPreprocessor
# ---------------------------------------------------------------------------

_preprocessor = None


def _get_preprocessor():
    """Lazy-load NewsPreprocessor so import cost is paid only when used."""
    global _preprocessor
    if _preprocessor is None:
        from news_preprocessing import NewsPreprocessor  # type: ignore
        _preprocessor = NewsPreprocessor()
    return _preprocessor


def preprocess_articles(articles: list[dict]) -> list[dict]:
    """
    Apply NewsPreprocessor to title and content of each article in-place,
    drop any whose title and content are both empty after cleaning.
    Returns the surviving list.
    """
    if not articles:
        return articles
    pp = _get_preprocessor()
    kept = []
    for a in articles:
        a['title'] = pp.preprocess_title(a.get('title', ''))
        a['content'] = pp.preprocess_text(a.get('content', ''))
        if not a.get('snippet'):
            a['snippet'] = (a['content'][:200].rsplit(' ', 1)[0] + '…') if a['content'] else ''
        if a['title'] or a['content']:
            kept.append(a)
    logger.info(f'Preprocessed {len(kept)}/{len(articles)} articles (dropped empty)')
    return kept


# ---------------------------------------------------------------------------
# CSV row mappers — convert article dict (snake_case) ↔ CSV row (PascalCase)
# ---------------------------------------------------------------------------

RAW_FIELDS = ['Date', 'Title', 'Category', 'Content', 'Snippet', 'URL']

SENTIMENT_FIELDS = RAW_FIELDS + [
    'Title_Sentiment', 'Title_Confidence',
    'Title_Positive_Prob', 'Title_Negative_Prob', 'Title_Neutral_Prob',
    'Content_Sentiment', 'Content_Confidence',
    'Content_Positive_Prob', 'Content_Negative_Prob', 'Content_Neutral_Prob',
    'Combined_Sentiment', 'Combined_Confidence',
    'Combined_Positive_Prob', 'Combined_Negative_Prob', 'Combined_Neutral_Prob',
    'Content_Sentence_Count', 'Content_Sentence_Used', 'Content_Filter_Fallback',
]


def article_to_raw_row(article: dict) -> dict:
    """Map an article dict to a raw / preprocessed CSV row (PascalCase)."""
    return {
        'Date':     article.get('date', ''),
        'Title':    article.get('title', ''),
        'Category': article.get('category', ''),
        'Content':  article.get('content', ''),
        'Snippet':  article.get('snippet', ''),
        'URL':      article.get('url', ''),
    }


def article_to_sentiment_row(article: dict) -> dict:
    """Map an article (with sentiment fields) to a tone-CSV row.

    Per-title and per-content fields are written when present (the sentiment
    runner exposes them); otherwise left blank. Combined_* mirrors the
    in-memory snake_case fields (sentiment_label, positive_prob, ...).
    """
    row = article_to_raw_row(article)
    row.update({
        'Title_Sentiment':       article.get('title_sentiment', ''),
        'Title_Confidence':      _fmt_float(article.get('title_confidence')),
        'Title_Positive_Prob':   _fmt_float(article.get('title_positive_prob')),
        'Title_Negative_Prob':   _fmt_float(article.get('title_negative_prob')),
        'Title_Neutral_Prob':    _fmt_float(article.get('title_neutral_prob')),
        'Content_Sentiment':     article.get('content_sentiment', ''),
        'Content_Confidence':    _fmt_float(article.get('content_confidence')),
        'Content_Positive_Prob': _fmt_float(article.get('content_positive_prob')),
        'Content_Negative_Prob': _fmt_float(article.get('content_negative_prob')),
        'Content_Neutral_Prob':  _fmt_float(article.get('content_neutral_prob')),
        'Combined_Sentiment':       article.get('sentiment_label', ''),
        'Combined_Confidence':      _fmt_float(article.get('combined_confidence')),
        'Combined_Positive_Prob':   _fmt_float(article.get('positive_prob')),
        'Combined_Negative_Prob':   _fmt_float(article.get('negative_prob')),
        'Combined_Neutral_Prob':    _fmt_float(article.get('neutral_prob')),
        'Content_Sentence_Count':   article.get('content_sentence_count', ''),
        'Content_Sentence_Used':    article.get('content_sentence_used', ''),
        'Content_Filter_Fallback':  article.get('content_filter_fallback', ''),
    })
    return row


def _fmt_float(v) -> str:
    if v is None or v == '':
        return ''
    try:
        return f'{float(v):.4f}'
    except (TypeError, ValueError):
        return ''


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

def _make_session() -> requests.Session:
    """A session with retry/backoff and a connection pool sized for workers."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    retry = Retry(
        total=RETRY_LIMIT,
        backoff_factor=1,                      # wait 1s, 2s, 4s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS * 2,
        pool_maxsize=MAX_WORKERS * 2,
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


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
    session = _make_session()
    articles: list[dict] = []
    try:
        for start_url in MPOB_SEARCH_URLS:
            articles.extend(_scrape_keyword(session, start_url, cutoff_date))
    finally:
        session.close()

    # Deduplicate by URL (the two keyword searches overlap heavily).
    seen: set[str] = set()
    unique = []
    for a in articles:
        if a['url'] and a['url'] not in seen:
            seen.add(a['url'])
            unique.append(a)

    logger.info(f'Scraped {len(unique)} new articles')
    return unique


def _scrape_keyword(session: requests.Session, start_url: str,
                    cutoff_date: str) -> list[dict]:
    """
    Walk the result pages for one keyword URL, newest-first, collecting
    articles dated after `cutoff_date`. Stops at the first article that is
    not newer than the cutoff (the listing is ordered, so the rest are older).
    """
    articles: list[dict] = []
    current_url: Optional[str] = start_url
    page = 1

    while current_url:
        try:
            resp = session.get(current_url, timeout=REQUEST_TIMEOUT, verify=False)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f'Failed to fetch list page {page}: {e}')
            break

        soup = BeautifulSoup(resp.text, 'html.parser')
        cards = soup.select('a > div.rounded.shadow-md')
        if not cards:
            break

        batch, reached_cutoff = _parse_cards(cards, cutoff_date)
        if batch:
            _fetch_contents(batch)
            articles.extend(batch)

        if reached_cutoff:
            break

        next_btn = soup.find('a', attrs={'rel': 'next'})
        if next_btn and next_btn.has_attr('href'):
            current_url = next_btn['href']
            page += 1
            time.sleep(random.uniform(0.3, 0.8))
        else:
            current_url = None

    return articles


def _parse_cards(cards, cutoff_date: str) -> tuple[list[dict], bool]:
    """
    Extract article stubs (no content yet) from a page's news cards.
    Returns (articles, reached_cutoff) — reached_cutoff is True once a card
    dated on/before `cutoff_date` is seen, signalling the caller to stop.
    """
    batch: list[dict] = []
    for card in cards:
        try:
            date_el = card.select_one('span.text-sm.text-black')
            norm_date = _normalise_date(date_el.get_text(strip=True)) if date_el else None
            if not norm_date:
                continue
            if norm_date <= cutoff_date:
                return batch, True   # listing is newest-first — stop here

            link_tag = card.parent
            if not link_tag or not link_tag.has_attr('href'):
                continue
            url = link_tag['href']
            if not url.startswith('http'):
                url = MPOB_BASE + url

            title_el = card.select_one('span.font-bold')
            snippet_el = card.select_one('p.line-clamp-3')
            cat_el = card.select_one('span.divider-right')

            batch.append({
                'date': norm_date,
                'title': title_el.get_text(strip=True) if title_el else '',
                'category': cat_el.get_text(strip=True) if cat_el else 'General',
                'content': '',
                'snippet': snippet_el.get_text(strip=True) if snippet_el else '',
                'url': url,
            })
        except Exception as e:
            logger.warning(f'Error parsing news card: {e}')
            continue
    return batch, False


def _fetch_contents(articles: list[dict]) -> None:
    """Fetch full article text for each stub in parallel, updating in place."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_article_content, a['url']): a
                   for a in articles}
        for future in as_completed(futures):
            article = futures[future]
            try:
                content = future.result()
            except Exception as e:
                logger.warning(f'Content fetch failed for {article["url"]}: {e}')
                content = ''
            article['content'] = content
            if not article['snippet'] and content:
                article['snippet'] = content[:200].rsplit(' ', 1)[0] + '…'


def _fetch_article_content(url: str) -> str:
    """Fetch and return the full text of a single article (own session)."""
    session = _make_session()
    try:
        time.sleep(0.1)   # small courtesy delay between worker requests
        resp = session.get(url, timeout=REQUEST_TIMEOUT, verify=False)
        if resp.status_code != 200:
            return ''
        soup = BeautifulSoup(resp.text, 'html.parser')
        body = soup.select_one('div.w-full.h-full.text-lg')
        if not body:
            return ''
        for tag in body.select('script, style, nav, footer'):
            tag.decompose()
        return body.get_text(separator='\n\n', strip=True)
    except Exception:
        return ''
    finally:
        session.close()
