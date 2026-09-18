"""Modelos opcionales: embeddings para el nivel 2 de dedup y FinBERT para sentimiento.

Viven en el extra ``nlp`` (uv sync --extra nlp) porque arrastran torch. Sin
él el pipeline sigue funcionando: la deduplicación se queda en hash y
MinHash, y los temas van sin sentimiento. Por eso los cargadores devuelven
``None`` en vez de fallar, y por eso la importación es perezosa: quien no
use el extra no paga ni el tiempo de importar transformers.

Cargar un modelo la primera vez lo descarga de HuggingFace. Si no hay red o
el modelo no existe, también se degrada: no hay señal que dependa de esto.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger(__name__)

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
MAX_TOKENS = 512  # ventana de BERT; el resumen de una noticia cabe de sobra


@dataclass(frozen=True)
class Sentiment:
    label: str
    score: float  # firmado: +confianza si positivo, -confianza si negativo, 0.0 si neutro


class Embedder(Protocol):
    def encode(
        self, texts: Sequence[str]
    ) -> NDArray[np.float64]: ...  # una fila normalizada por texto


class SentimentModel(Protocol):
    def score(self, texts: Sequence[str]) -> list[Sentiment]: ...


@dataclass(frozen=True)
class SentenceTransformerEmbedder:
    model: Any

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float64")


@dataclass(frozen=True)
class FinbertSentiment:
    classifier: Any

    def score(self, texts: Sequence[str]) -> list[Sentiment]:
        if not texts:
            return []
        results = self.classifier(list(texts))
        return [Sentiment(label=_label(r), score=_signed(r)) for r in results]


def load_embedder(model_name: str) -> Embedder | None:
    """Modelo de embeddings, o ``None`` si el extra ``nlp`` no está instalado."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.info("dedup.embeddings_unavailable", detail="falta el extra nlp")
        return None
    try:
        return SentenceTransformerEmbedder(SentenceTransformer(model_name))
    except (OSError, ValueError) as exc:  # modelo inexistente o sin red la primera vez
        log.warning("dedup.embeddings_failed", model=model_name, detail=str(exc))
        return None


def load_sentiment(model_name: str) -> SentimentModel | None:
    """FinBERT, o ``None`` si el extra ``nlp`` no está instalado."""
    try:
        from transformers import pipeline
    except ImportError:
        log.info("dedup.sentiment_unavailable", detail="falta el extra nlp")
        return None
    try:
        classifier = pipeline(
            "text-classification", model=model_name, truncation=True, max_length=MAX_TOKENS
        )
    except (OSError, ValueError) as exc:
        log.warning("dedup.sentiment_failed", model=model_name, detail=str(exc))
        return None
    return FinbertSentiment(classifier)


def _label(result: Any) -> str:
    return str(result["label"]).strip().casefold()


def _signed(result: Any) -> float:
    confidence = float(result["score"])
    label = _label(result)
    if label == POSITIVE:
        return confidence
    if label == NEGATIVE:
        return -confidence
    return 0.0
