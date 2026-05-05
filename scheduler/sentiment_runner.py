"""
sentiment_runner.py — Run FinBERT-Tone sentiment analysis on new MPOB articles.

Sentence-level pipeline: each article's content is split with NLTK, FinBERT-Tone
scores each sentence, low-confidence sentences are dropped, and the remaining
probabilities are averaged. Title is scored as a single sentence and combined
with the content mean (0.3 / 0.7 weighting) when the title clears the same
confidence threshold.

Operates on in-memory list of dicts (no CSV I/O).
GPU is used automatically if available; falls back to CPU.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = 'yiyanghkust/finbert-tone'
MAX_LENGTH = 512
BATCH_SIZE = 16  # Articles per batch (sentences within are flattened for inference)
SENTENCE_CONFIDENCE_THRESHOLD = 0.5  # Drop sentences whose max-class probability is below this
TITLE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.7

# yiyanghkust/finbert-tone id2label is {0: Neutral, 1: Positive, 2: Negative} —
# DIFFERENT from ProsusAI/finbert. Hardcoded to match the model's id2label.
LABELS = ['Neutral', 'Positive', 'Negative']  # index-aligned with model output


def _ensure_nltk_punkt():
    """Ensure NLTK's punkt_tab tokenizer data is available; download once if missing."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info('Downloading NLTK punkt_tab tokenizer (one-time)')
        nltk.download('punkt_tab', quiet=True)


def _split_into_sentences(text: str) -> list[str]:
    """Split a content string into sentences via NLTK. Returns [] for empty input."""
    if not text:
        return []
    from nltk.tokenize import sent_tokenize
    text = text.strip()
    if not text:
        return []
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception as exc:
        logger.warning('NLTK sent_tokenize failed; treating whole text as one sentence: %s', exc)
        return [text]


def _load_model():
    """Load FinBERT-Tone model and tokenizer. Returns (model, tokenizer, device)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading %s on %s', MODEL_NAME, device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _score_texts(texts, model, tokenizer, device):
    """Run FinBERT-Tone on a flat list of texts.

    Returns Nx3 ndarray with columns aligned to LABELS = [Neutral, Positive, Negative].
    """
    if not texts:
        return np.zeros((0, 3), dtype=np.float32)
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs


def _aggregate_content(probs_arr):
    """Mean probabilities across content sentences with the confidence filter.

    Returns (mean_neu, mean_pos, mean_neg, n_total, n_used, fallback_triggered).
    For empty input returns (None, 0, 0, False) — caller decides how to handle that.
    """
    n_total = int(probs_arr.shape[0])
    if n_total == 0:
        return (None, 0, 0, False)

    max_per_row = probs_arr.max(axis=1)
    keep_mask = max_per_row >= SENTENCE_CONFIDENCE_THRESHOLD
    n_kept = int(keep_mask.sum())

    fallback = False
    if n_kept == 0:
        kept = probs_arr
        n_used = n_total
        fallback = True
    else:
        kept = probs_arr[keep_mask]
        n_used = n_kept

    return (kept.mean(axis=0), n_total, n_used, fallback)


def run_sentiment_on_articles(articles: list[dict]) -> list[dict]:
    """
    Add sentiment fields to each article dict in-place.

    Fields added: sentiment_label, sentiment_score, positive_prob,
                  negative_prob, neutral_prob.

    Internally uses sentence-level FinBERT-Tone scoring, mean-probability
    aggregation with a confidence filter, and a 0.3/0.7 title/content weighted
    combine. Returns the same list with sentiment fields added.
    """
    if not articles:
        return articles

    _ensure_nltk_punkt()
    model, tokenizer, device = _load_model()

    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]

        # Build flat sentence list across the whole batch with index tracking.
        flat_texts = []
        meta = []  # one tuple per article: (title_idx_or_none, content_start, content_end)
        for a in batch:
            title = (a.get('title') or '').strip()
            title_idx = None
            if title:
                title_idx = len(flat_texts)
                flat_texts.append(title)
            sentences = _split_into_sentences(a.get('content') or '')
            content_start = len(flat_texts)
            flat_texts.extend(sentences)
            content_end = len(flat_texts)
            meta.append((title_idx, content_start, content_end))

        try:
            probs = _score_texts(flat_texts, model, tokenizer, device)
        except Exception as e:
            logger.warning('Sentiment batch %d failed: %s', i // BATCH_SIZE, e)
            for a in batch:
                a.setdefault('sentiment_label', 'Neutral')
                a.setdefault('sentiment_score', 0.0)
                a.setdefault('positive_prob', 0.33)
                a.setdefault('negative_prob', 0.33)
                a.setdefault('neutral_prob', 0.34)
            continue

        for a, (title_idx, content_start, content_end) in zip(batch, meta):
            # Title: single sentence; keep if confidence clears threshold.
            if title_idx is None:
                title_probs = None
                title_passed = False
            else:
                title_probs = probs[title_idx]
                title_passed = float(title_probs.max()) >= SENTENCE_CONFIDENCE_THRESHOLD

            # Content: aggregate with filter (and fallback to all-sentences if all filtered).
            content_mean, _, _, _ = _aggregate_content(probs[content_start:content_end])

            # Combine. Order: [Neutral, Positive, Negative].
            if content_mean is not None and title_passed:
                combined = TITLE_WEIGHT * title_probs + CONTENT_WEIGHT * content_mean
            elif content_mean is not None:
                combined = content_mean
            elif title_probs is not None:
                combined = title_probs
            else:
                combined = np.array([1.0, 0.0, 0.0])  # degenerate neutral

            label_idx = int(np.argmax(combined))
            label = LABELS[label_idx]
            neu, pos, neg = float(combined[0]), float(combined[1]), float(combined[2])
            score = pos - neg

            a.update({
                'sentiment_label': label,
                'sentiment_score': round(score, 4),
                'positive_prob': round(pos, 4),
                'negative_prob': round(neg, 4),
                'neutral_prob': round(neu, 4),
            })

        if (i // BATCH_SIZE) % 10 == 0:
            logger.info('Sentiment progress: %d/%d', min(i + BATCH_SIZE, len(articles)), len(articles))

    return articles


def compute_sentiment_aggregates(articles: list[dict]) -> list[dict]:
    """
    Aggregate article sentiment by date (Daily granularity).
    Returns list of dicts ready for `firestore_writer.write_sentiment_aggregates`.
    """
    from collections import defaultdict

    daily: dict = defaultdict(lambda: {
        'positive_sum': 0.0, 'negative_sum': 0.0, 'neutral_sum': 0.0,
        'score_sum': 0.0, 'count': 0,
    })

    for a in articles:
        d = a.get('date', '')
        if not d or 'sentiment_score' not in a or 'positive_prob' not in a:
            continue
        bucket = daily[d]
        bucket['positive_sum'] += a['positive_prob']
        bucket['negative_sum'] += a['negative_prob']
        bucket['neutral_sum'] += a['neutral_prob']
        bucket['score_sum'] += a['sentiment_score']
        bucket['count'] += 1

    aggregates = []
    for date_str, b in sorted(daily.items()):
        n = b['count']
        pos = b['positive_sum'] / n
        neg = b['negative_sum'] / n
        neu = b['neutral_sum'] / n
        score = b['score_sum'] / n
        dominant = max(
            [('Positive', pos), ('Negative', neg), ('Neutral', neu)],
            key=lambda x: x[1]
        )[0]
        aggregates.append({
            'date': date_str,
            'frequency': 'Daily',
            'article_count': n,
            'positive_prob': round(pos, 4),
            'negative_prob': round(neg, 4),
            'neutral_prob': round(neu, 4),
            'sentiment_score': round(score, 4),
            'dominant_sentiment': dominant,
        })

    return aggregates
