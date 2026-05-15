"""
sentiment_runner.py — Run FinBERT-Tone sentiment analysis on new MPOB articles.

Title-only pipeline: each article's headline is scored by FinBERT-Tone and the
result is used as the article's sentiment. Mirrors the offline reference
news/finbert_tone_sentiment_analysis.py running with --mode title.

Operates on in-memory list of dicts (no CSV I/O).
GPU is used automatically if available; falls back to CPU.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = 'yiyanghkust/finbert-tone'
MAX_LENGTH = 512
BATCH_SIZE = 16  # Articles per batch

# yiyanghkust/finbert-tone id2label is {0: Neutral, 1: Positive, 2: Negative} —
# DIFFERENT from ProsusAI/finbert. Hardcoded to match the model's id2label.
LABELS = ['Neutral', 'Positive', 'Negative']  # index-aligned with model output


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


def run_sentiment_on_articles(articles: list[dict]) -> list[dict]:
    """
    Add sentiment fields to each article dict in-place, scoring the title only.

    Fields added: sentiment_label, sentiment_score, positive_prob,
                  negative_prob, neutral_prob, combined_confidence,
                  title_sentiment, title_confidence, title_{neutral,positive,negative}_prob.

    Combined_* mirrors Title_* — the body text is no longer scored. Articles
    with an empty title get a degenerate Neutral result.
    """
    if not articles:
        return articles

    model, tokenizer, device = _load_model()

    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]

        flat_titles = []
        meta = []  # one entry per article: title_idx_or_none
        for a in batch:
            title = (a.get('title') or '').strip()
            if title:
                meta.append(len(flat_titles))
                flat_titles.append(title)
            else:
                meta.append(None)

        try:
            probs = _score_texts(flat_titles, model, tokenizer, device)
        except Exception as e:
            logger.warning('Sentiment batch %d failed: %s', i // BATCH_SIZE, e)
            for a in batch:
                a.setdefault('sentiment_label', 'Neutral')
                a.setdefault('sentiment_score', 0.0)
                a.setdefault('positive_prob', 0.33)
                a.setdefault('negative_prob', 0.33)
                a.setdefault('neutral_prob', 0.34)
            continue

        for a, title_idx in zip(batch, meta):
            if title_idx is None:
                combined = np.array([1.0, 0.0, 0.0])  # degenerate neutral
            else:
                combined = probs[title_idx]

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
                'combined_confidence': round(float(combined.max()), 4),
            })

            if title_idx is not None:
                a.update({
                    'title_sentiment':     label,
                    'title_confidence':    round(float(combined.max()), 4),
                    'title_neutral_prob':  round(neu, 4),
                    'title_positive_prob': round(pos, 4),
                    'title_negative_prob': round(neg, 4),
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
