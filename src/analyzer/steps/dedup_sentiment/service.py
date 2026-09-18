"""De artículos sueltos a temas con sentimiento.

Las noticias de las últimas horas llegan repetidas: agencias, agregadores y
portales republican la misma nota. Mandarlas todas al LLM es pagar tres
veces por el mismo hecho y, peor, darle la impresión de que un hecho
repetido es un hecho importante. Aquí se agrupan en temas (ver ``cascade``),
cada tema elige el artículo que mejor lo cuenta y solo ese pasa por FinBERT.

El representante se elige con los pesos de ``representative_weights``:
fiabilidad de la fuente, ser el primero en publicar y longitud del texto.
El primero suele ser el teletipo original y el más largo el que trae
contexto; la fiabilidad rompe el empate hacia la fuente conocida.

El resultado se guarda point-in-time por región y día, y se deja en el
contexto para el paso del LLM.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import structlog

from analyzer.steps.dedup_sentiment.cascade import (
    EMBEDDING,
    EXACT,
    LEVELS,
    MINHASH,
    Grouping,
    cluster_articles,
)
from analyzer.steps.dedup_sentiment.models import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    Embedder,
    Sentiment,
    SentimentModel,
    load_embedder,
    load_sentiment,
)
from analyzer.steps.dedup_sentiment.text import article_text
from analyzer.storage.clusters import (
    CLUSTER_COLUMNS,
    MEMBER_COLUMNS,
    NewsClusterMemberStore,
    NewsClusterStore,
)
from core.config import NewsSettings, RepresentativeWeights

log = structlog.get_logger(__name__)

# Singular y plural, que un resumen que dice "1 duplicados" se lee como un error.
ARTICLE = ("noticia", "noticias")
TOPIC = ("tema", "temas")
DUPLICATE = ("duplicado", "duplicados")
LEVEL_LABELS = {
    EXACT: ("exacto", "exactos"),
    MINHASH: ("por minhash", "por minhash"),
    EMBEDDING: ("por embeddings", "por embeddings"),
}
SENTIMENT_LABELS = {
    POSITIVE: ("positivo", "positivos"),
    NEGATIVE: ("negativo", "negativos"),
    NEUTRAL: ("neutro", "neutros"),
}

EmbedderFactory = Callable[[str], Embedder | None]
SentimentFactory = Callable[[str], SentimentModel | None]


@dataclass(frozen=True)
class DedupReport:
    region_id: str
    as_of: date
    articles: int
    merges: Mapping[str, int]  # nivel de la cascada -> artículos absorbidos
    sentiment: Mapping[str, int]  # etiqueta -> temas
    embeddings_used: bool
    sentiment_used: bool
    clusters: pd.DataFrame = field(repr=False, compare=False)

    @property
    def duplicates(self) -> int:
        return self.articles - len(self.clusters)

    def summary(self) -> str:
        text = f"{_count(self.articles, ARTICLE)} -> {_count(len(self.clusters), TOPIC)}"
        if self.duplicates:
            detail = ", ".join(
                _count(self.merges[level], LEVEL_LABELS[level])
                for level in LEVELS
                if self.merges.get(level)
            )
            text += f", {_count(self.duplicates, DUPLICATE)} ({detail})"
        else:
            text += ", sin duplicados"
        if self.sentiment:
            counts = ", ".join(
                _count(self.sentiment[label], names)
                for label, names in SENTIMENT_LABELS.items()
                if self.sentiment.get(label)
            )
            text += f"; {counts}"
        missing = [
            name
            for name, used in (
                ("embeddings", self.embeddings_used),
                ("sentimiento", self.sentiment_used),
            )
            if not used
        ]
        if missing:
            text += f"; sin {' ni '.join(missing)} (falta el extra nlp)"
        return text


def run_dedup_sentiment(
    articles: pd.DataFrame,
    as_of: date,
    *,
    region_id: str,
    settings: NewsSettings,
    store: NewsClusterStore,
    members: NewsClusterMemberStore,
    make_embedder: EmbedderFactory | None = None,
    make_sentiment: SentimentFactory | None = None,
) -> DedupReport:
    """Los modelos se cargan aquí y no antes: sin artículos no se paga su arranque."""
    dedup = settings.dedup
    embed = make_embedder or load_embedder
    embedder = embed(dedup.embedding_model) if not articles.empty else None
    grouping = cluster_articles(
        articles,
        shingle_size=dedup.shingle_size,
        num_perm=dedup.minhash_num_perm,
        minhash_threshold=dedup.minhash_threshold,
        window_hours=dedup.cluster_window_hours,
        embedder=embedder,
        cosine_threshold=dedup.embedding_cosine_threshold,
    )
    clusters, member_rows = build_clusters(articles, grouping, settings.representative_weights)

    classify = make_sentiment or load_sentiment
    model = classify(settings.sentiment_model) if not clusters.empty else None
    clusters = add_sentiment(clusters, model)

    with store.transaction():
        store.replace_day(region_id, as_of, clusters)
        members.replace_day(region_id, as_of, member_rows)

    report = DedupReport(
        region_id=region_id,
        as_of=as_of,
        articles=len(articles),
        merges=grouping.merges,
        sentiment=_sentiment_counts(clusters),
        embeddings_used=embedder is not None,
        sentiment_used=model is not None,
        clusters=clusters,
    )
    log.info("dedup_sentiment.done", detail=report.summary())
    return report


def build_clusters(
    articles: pd.DataFrame, grouping: Grouping, weights: RepresentativeWeights
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Una fila por tema con su representante, y una fila por artículo con su tema."""
    if articles.empty:
        return _empty(CLUSTER_COLUMNS), _empty(MEMBER_COLUMNS)

    frame = articles.copy()
    frame["text"] = [
        article_text(t, s) for t, s in zip(frame["title"], frame["summary"], strict=True)
    ]
    frame["cluster"] = frame["id"].astype(str).map(grouping.labels)

    rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for _, group in frame.groupby("cluster", sort=True):
        ids = sorted(str(i) for i in group["id"])
        cluster_id = _cluster_id(ids)
        chief = representative(group, weights)
        rows.append(
            {
                "cluster_id": cluster_id,
                "ticker": chief["ticker"],
                "mic": chief["mic"],
                "representative_id": str(chief["id"]),
                "size": len(ids),
                "method": grouping.level_of(ids),
                "sources": ", ".join(sorted(set(group["source"].astype(str)))),
                "reliability": float(group["reliability"].max()),
                "language": str(chief["language"]),
                "first_published_at": group["published_at"].min(),
                "last_published_at": group["published_at"].max(),
                "title": str(chief["title"]),
                "summary": chief["summary"],
                "url": str(chief["url"]),
                "sentiment_label": None,
                "sentiment_score": float("nan"),
            }
        )
        member_rows.extend(
            {
                "cluster_id": cluster_id,
                "news_id": article_id,
                "method": grouping.methods[article_id],
                "is_representative": article_id == str(chief["id"]),
            }
            for article_id in ids
        )

    clusters = pd.DataFrame(rows, columns=[*CLUSTER_COLUMNS])
    clusters = clusters.sort_values(
        ["ticker", "last_published_at", "cluster_id"], ascending=[True, False, True]
    )
    return clusters.reset_index(drop=True), pd.DataFrame(member_rows, columns=[*MEMBER_COLUMNS])


def representative(group: pd.DataFrame, weights: RepresentativeWeights) -> pd.Series:
    """El artículo que mejor cuenta el tema, según los pesos de la configuración."""
    lengths = group["text"].str.len()
    longest = lengths.max()
    oldest = group["published_at"] == group["published_at"].min()
    score = (
        weights.source_reliability * group["reliability"].astype("float64")
        + weights.is_oldest * oldest.astype("float64")
        + weights.normalized_length * (lengths / longest if longest else 0.0)
    )
    ordered = group.assign(_score=score).sort_values(
        ["_score", "id"],
        ascending=[False, True],  # el id desempata: mismo tema, mismo elegido
    )
    return ordered.iloc[0]


def add_sentiment(clusters: pd.DataFrame, model: SentimentModel | None) -> pd.DataFrame:
    """Sentimiento del representante de cada tema; sin modelo, columnas vacías."""
    if model is None or clusters.empty:
        return clusters
    texts = [
        article_text(t, s) for t, s in zip(clusters["title"], clusters["summary"], strict=True)
    ]
    scored: list[Sentiment] = model.score(texts)
    if len(scored) != len(clusters):
        raise ValueError(f"el modelo de sentimiento devolvió {len(scored)} de {len(clusters)}")
    result = clusters.copy()
    result["sentiment_label"] = [s.label for s in scored]
    result["sentiment_score"] = [s.score for s in scored]
    return result


def _sentiment_counts(clusters: pd.DataFrame) -> dict[str, int]:
    if clusters.empty or clusters["sentiment_label"].isna().all():
        return {}
    counts = clusters["sentiment_label"].value_counts()
    return {str(label): int(n) for label, n in counts.items()}


def _cluster_id(ids: list[str]) -> str:
    """Determinista: los mismos artículos dan el mismo tema aunque se repita el día."""
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*columns])


def _count(number: int, names: tuple[str, str]) -> str:
    singular, plural = names
    return f"{number} {singular if number == 1 else plural}"
