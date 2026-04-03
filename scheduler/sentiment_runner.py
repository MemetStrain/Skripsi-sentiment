"""
sentiment_runner.py — Run FinBERT sentiment analysis on new MPOB articles.

Operates on in-memory list of dicts (no CSV I/O).
GPU is used automatically if available; falls back to CPU.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

MODEL_NAME = 'ProsusAI/finbert'
MAX_LENGTH = 512
BATCH_SIZE = 16  # adjusted down for CPU safety; GPU can handle larger


def _load_model():
    """Load FinBERT model and tokenizer. Returns (model, tokenizer, device)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Loading FinBERT on {device}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def run_sentiment_on_articles(articles: list[dict]) -> list[dict]:
    """
    Add sentiment fields to each article dict in-place.
    Fields added: sentiment_label, sentiment_score, positive_prob,
                  negative_prob, neutral_prob.
    Returns the same list with sentiment added.
    """
    if not articles:
        return articles

    model, tokenizer, device = _load_model()
    label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}  # FinBERT label order

    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]
        texts = []
        for a in batch:
            # Combine title + first 400 chars of content for inference
            text = a.get('title', '') + ' ' + a.get('content', '')[:400]
            texts.append(text.strip() or 'N/A')

        try:
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

            for j, article in enumerate(batch):
                pos_p, neg_p, neu_p = float(probs[j][0]), float(probs[j][1]), float(probs[j][2])
                label_idx = int(probs[j].argmax())
                label = label_map[label_idx]
                # Sentiment score: positive=+1, negative=-1, neutral=0
                score = pos_p - neg_p
                article.update({
                    'sentiment_label': label,
                    'sentiment_score': round(score, 4),
                    'positive_prob': round(pos_p, 4),
                    'negative_prob': round(neg_p, 4),
                    'neutral_prob': round(neu_p, 4),
                })

        except Exception as e:
            logger.warning(f'Sentiment batch {i//BATCH_SIZE} failed: {e}')
            # Assign neutral as fallback
            for article in batch:
                article.setdefault('sentiment_label', 'Neutral')
                article.setdefault('sentiment_score', 0.0)
                article.setdefault('positive_prob', 0.33)
                article.setdefault('negative_prob', 0.33)
                article.setdefault('neutral_prob', 0.34)

        if (i // BATCH_SIZE) % 10 == 0:
            logger.info(f'Sentiment progress: {min(i + BATCH_SIZE, len(articles))}/{len(articles)}')

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
        if not d:
            continue
        bucket = daily[d]
        bucket['positive_sum'] += a.get('positive_prob', 0.33)
        bucket['negative_sum'] += a.get('negative_prob', 0.33)
        bucket['neutral_sum'] += a.get('neutral_prob', 0.34)
        bucket['score_sum'] += a.get('sentiment_score', 0.0)
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
