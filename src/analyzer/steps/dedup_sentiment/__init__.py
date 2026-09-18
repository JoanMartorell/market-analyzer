"""Paso 8: cascada de deduplicación (hash, MinHash, embeddings) y FinBERT local."""

from analyzer.steps.dedup_sentiment.cascade import Grouping, cluster_articles
from analyzer.steps.dedup_sentiment.models import Sentiment, load_embedder, load_sentiment
from analyzer.steps.dedup_sentiment.service import DedupReport, run_dedup_sentiment
from analyzer.steps.dedup_sentiment.step import DedupSentiment

__all__ = [
    "DedupReport",
    "DedupSentiment",
    "Grouping",
    "Sentiment",
    "cluster_articles",
    "load_embedder",
    "load_sentiment",
    "run_dedup_sentiment",
]
