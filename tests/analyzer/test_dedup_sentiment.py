"""Paso dedup_sentiment: texto, cascada de deduplicación, representante, sentimiento y paso."""

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from analyzer.engine import StepContext, StepError, StepStatus
from analyzer.steps.dedup_sentiment import (
    DedupSentiment,
    Grouping,
    Sentiment,
    cluster_articles,
    load_embedder,
    load_sentiment,
    run_dedup_sentiment,
)
from analyzer.steps.dedup_sentiment.cascade import EMBEDDING, EXACT, MINHASH, SINGLE
from analyzer.steps.dedup_sentiment.models import FinbertSentiment
from analyzer.steps.dedup_sentiment.service import build_clusters, representative
from analyzer.steps.dedup_sentiment.text import article_text, normalize, shingles, text_hash
from analyzer.storage import NewsClusterMemberStore, NewsClusterStore, NewsStore, connect
from core.config import AppConfig, Env, NewsSettings, RepresentativeWeights

AS_OF = date(2026, 9, 17)
NOW = datetime(2026, 9, 18, 6, 0, tzinfo=UTC)

LONG = (
    "La compañía presentó resultados por encima de lo que esperaba el consenso "
    "de analistas y elevó su previsión de ingresos para el conjunto del año, "
    "apoyada en la demanda de su división de servicios en Europa y Asia"
)
LONG_EDITED = LONG + " segun el sector"  # la misma nota con una coletilla de más

Stores = tuple[NewsClusterStore, NewsClusterMemberStore]


def _article(
    article_id: str,
    *,
    ticker: str | None = "AAPL",
    title: str = "Titular",
    summary: str = "Resumen",
    source: str = "yahoo",
    reliability: float = 0.6,
    hours_ago: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": article_id,
        "ticker": ticker,
        "mic": "XNAS" if ticker else None,
        "source": source,
        "reliability": reliability,
        "language": "es",
        "published_at": NOW - timedelta(hours=hours_ago),
        "title": title,
        "summary": summary,
        "url": f"https://n.example/{article_id}",
    }


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    frame["text"] = [
        article_text(t, s) for t, s in zip(frame["title"], frame["summary"], strict=True)
    ]
    return frame


def _cluster(frame: pd.DataFrame, **overrides: Any) -> Grouping:
    options: dict[str, Any] = {
        "shingle_size": 3,
        "num_perm": 128,
        "minhash_threshold": 0.85,
        "window_hours": 48,
    }
    options.update(overrides)
    return cluster_articles(frame, **options)


# --- texto ------------------------------------------------------------------------------


def test_normalize_ignores_case_accents_and_punctuation() -> None:
    assert normalize("¡Telefónica SUBE un 4%!") == "telefonica sube un 4"
    assert text_hash("Apple sube.") == text_hash("APPLE  sube")
    assert article_text("Titular", "Resumen") == "Titular. Resumen"
    assert article_text("Titular", None) == "Titular"


def test_shingles_keeps_short_texts_whole() -> None:
    assert shingles("Apple sube hoy mucho", 3) == {"apple sube hoy", "sube hoy mucho"}
    assert shingles("Apple sube", 3) == {"apple sube"}
    assert shingles("  ", 3) == frozenset()


# --- cascada ----------------------------------------------------------------------------


def test_exact_duplicates_merge_only_within_ticker_and_window() -> None:
    frame = _frame(
        _article("a", title="Apple sube", hours_ago=1),
        _article("b", title="APPLE  sube!", hours_ago=2, source="cnbc"),
        _article("c", title="Apple sube", ticker="MSFT", hours_ago=1),
        _article("d", title="Apple sube", hours_ago=400),  # otro trimestre, otro evento
    )

    grouping = _cluster(frame)

    assert grouping.labels["a"] == grouping.labels["b"]
    assert grouping.labels["c"] != grouping.labels["a"]
    assert grouping.labels["d"] != grouping.labels["a"]
    assert grouping.merges == {EXACT: 1}
    assert grouping.methods["b"] == EXACT
    assert grouping.methods["c"] == SINGLE
    assert grouping.level_of(["a", "b"]) == EXACT


def test_minhash_merges_a_retouched_copy() -> None:
    frame = _frame(
        _article("a", title="Resultados", summary=LONG),
        _article("b", title="Resultados", summary=LONG_EDITED, source="cnbc", hours_ago=3),
        _article("c", title="Dimite el consejero delegado", summary="Sin relación con lo demás"),
    )

    grouping = _cluster(frame)

    assert grouping.labels["a"] == grouping.labels["b"]
    assert grouping.labels["c"] != grouping.labels["a"]
    assert grouping.merges == {MINHASH: 1}
    assert grouping.clusters == 2


def test_global_articles_form_their_own_group() -> None:
    frame = _frame(
        _article("a", title="Apple sube"),
        _article("b", title="Apple sube", ticker=None, source="cnbc"),
    )

    grouping = _cluster(frame)

    assert grouping.labels["a"] != grouping.labels["b"]


@dataclass
class FakeEmbedder:
    """Dos direcciones: lo que habla de resultados y lo que no."""

    calls: int = 0

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls += 1
        rows = [[1.0, 0.0] if "resultado" in t.casefold() else [0.0, 1.0] for t in texts]
        return np.asarray(rows, dtype="float64")


def test_embeddings_merge_what_minhash_cannot() -> None:
    frame = _frame(
        _article("a", title="La empresa bate previsiones", summary="Buenos resultados"),
        _article("b", title="Beneficio récord", summary="El resultado supera lo previsto"),
        _article("c", title="Cambio en el consejo", summary="Nuevo consejero"),
    )
    embedder = FakeEmbedder()

    grouping = _cluster(frame, embedder=embedder, cosine_threshold=0.85)

    assert grouping.labels["a"] == grouping.labels["b"]
    assert grouping.labels["c"] != grouping.labels["a"]
    assert grouping.merges == {EMBEDDING: 1}
    assert grouping.level_of(["a", "b"]) == EMBEDDING
    assert embedder.calls == 1


def test_embeddings_skip_what_a_cheaper_level_already_merged() -> None:
    frame = _frame(
        _article("a", title="Buenos resultados"),
        _article("b", title="BUENOS resultados", source="cnbc"),
    )
    embedder = FakeEmbedder()

    grouping = _cluster(frame, embedder=embedder, cosine_threshold=0.85)

    assert grouping.merges == {EXACT: 1}
    assert embedder.calls == 0  # nada suelto que embeber


def test_cluster_result_does_not_depend_on_the_order_of_the_rows() -> None:
    rows = [
        _article("a", title="Apple sube"),
        _article("b", title="Apple sube", source="cnbc", hours_ago=2),
        _article("c", title="Otra cosa distinta del todo"),
    ]

    first = _cluster(_frame(*rows))
    backwards = _cluster(_frame(*reversed(rows)))

    assert first.labels == backwards.labels
    assert first.methods == backwards.methods


# --- representante y temas ---------------------------------------------------------------


@pytest.fixture
def weights(cfg: AppConfig) -> RepresentativeWeights:
    return cfg.settings.news.representative_weights


def test_representative_prefers_the_first_and_most_complete(
    weights: RepresentativeWeights,
) -> None:
    group = _frame(
        _article("a", summary="Corto", hours_ago=1, reliability=0.9),
        _article("b", summary=LONG, hours_ago=2, reliability=0.6),
    )

    assert representative(group, weights)["id"] == "b"


def test_representative_breaks_a_tie_with_the_better_source(
    weights: RepresentativeWeights,
) -> None:
    group = _frame(
        _article("a", summary="Mismo texto", reliability=0.5),
        _article("b", summary="Mismo texto", reliability=0.9, source="cnbc"),
    )

    assert representative(group, weights)["id"] == "b"


def test_build_clusters_describes_the_topic_and_its_members(
    weights: RepresentativeWeights,
) -> None:
    frame = _frame(
        _article("a", title="Apple sube", summary=LONG, hours_ago=2),
        _article("b", title="APPLE sube", summary=LONG, source="cnbc", reliability=0.9),
    )

    clusters, members = build_clusters(frame, _cluster(frame), weights)

    assert len(clusters) == 1
    assert clusters.loc[0, "representative_id"] == "a"  # el primero en publicar
    assert clusters.loc[0, "size"] == 2
    assert clusters.loc[0, "method"] == EXACT
    assert clusters.loc[0, "sources"] == "cnbc, yahoo"
    assert clusters.loc[0, "reliability"] == 0.9  # la mejor fuente del tema
    assert members["news_id"].tolist() == ["a", "b"]
    assert members["is_representative"].tolist() == [True, False]


def test_cluster_id_depends_only_on_its_articles(weights: RepresentativeWeights) -> None:
    rows = [_article("a", title="Apple sube"), _article("b", title="Apple sube", source="cnbc")]
    frame = _frame(*rows)
    backwards = _frame(*reversed(rows))

    first, _ = build_clusters(frame, _cluster(frame), weights)
    second, _ = build_clusters(backwards, _cluster(backwards), weights)

    assert first.loc[0, "cluster_id"] == second.loc[0, "cluster_id"]


# --- sentimiento -------------------------------------------------------------------------


@dataclass
class FakeSentiment:
    seen: list[list[str]] = field(default_factory=list)

    def score(self, texts: Sequence[str]) -> list[Sentiment]:
        self.seen.append(list(texts))
        return [
            Sentiment("positive", 0.9) if "sube" in t.casefold() else Sentiment("neutral", 0.0)
            for t in texts
        ]


def test_finbert_signs_the_confidence() -> None:
    labels = ["positive", "Negative", "neutral"]
    model = FinbertSentiment(lambda texts: [{"label": name, "score": 0.8} for name in labels])

    scored = model.score(["a", "b", "c"])

    assert [s.label for s in scored] == ["positive", "negative", "neutral"]
    assert [round(s.score, 2) for s in scored] == [0.8, -0.8, 0.0]


def test_models_degrade_without_the_nlp_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un módulo a ``None`` en sys.modules hace que su import falle: el extra sin instalar."""
    for module in ("sentence_transformers", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)

    assert load_embedder("cualquiera") is None
    assert load_sentiment("cualquiera") is None


# --- servicio ----------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


@pytest.fixture
def stores(cfg_tmp: AppConfig) -> Iterator[Stores]:
    with connect(cfg_tmp) as con:
        yield NewsClusterStore(con, "prod"), NewsClusterMemberStore(con, "prod")


def _news(cfg: AppConfig) -> NewsSettings:
    return cfg.settings.news


def test_run_dedup_sentiment_writes_topics_and_is_repeatable(
    cfg_tmp: AppConfig, stores: Stores
) -> None:
    store, members = stores
    frame = _frame(
        _article("a", title="Apple sube", summary=LONG, hours_ago=2),
        _article("b", title="APPLE sube", summary=LONG, source="cnbc", reliability=0.8),
        _article("c", title="Cambio en el consejo", summary="Nuevo consejero", hours_ago=0.5),
        _article("d", title="Titular global", ticker=None, source="cnbc", hours_ago=3),
    )
    model = FakeSentiment()

    def run() -> Any:
        return run_dedup_sentiment(
            frame,
            AS_OF,
            region_id="americas",
            settings=_news(cfg_tmp),
            store=store,
            members=members,
            make_embedder=lambda name: None,
            make_sentiment=lambda name: model,
        )

    report = run()

    assert report.articles == 4
    assert report.duplicates == 1
    assert report.summary() == (
        "4 noticias -> 3 temas, 1 duplicado (1 exacto); "
        "1 positivo, 2 neutros; sin embeddings (falta el extra nlp)"
    )
    assert model.seen == [  # un texto por tema, el del representante, y en orden
        [
            "Cambio en el consejo. Nuevo consejero",
            "Apple sube. " + LONG,
            "Titular global. Resumen",
        ]
    ]

    stored = store.load("americas", AS_OF)
    assert len(stored) == 3
    apple = stored.loc[stored["representative_id"] == "a"].iloc[0]
    assert apple["size"] == 2
    assert apple["sources"] == "cnbc, yahoo"
    assert apple["sentiment_label"] == "positive"
    assert apple["sentiment_score"] == pytest.approx(0.9)
    assert apple["first_published_at"] == pd.Timestamp(NOW - timedelta(hours=2))
    assert stored["ticker"].fillna("-").tolist() == ["AAPL", "AAPL", "-"]  # el global, el último
    assert len(members.load("americas", AS_OF)) == 4

    again = run()
    assert sorted(again.clusters["cluster_id"]) == sorted(report.clusters["cluster_id"])
    assert len(store.load("americas", AS_OF)) == 3  # se reescribe el día, no se acumula


def test_run_dedup_sentiment_without_models_leaves_sentiment_empty(
    cfg_tmp: AppConfig, stores: Stores
) -> None:
    store, members = stores

    report = run_dedup_sentiment(
        _frame(_article("a", title="Apple sube")),
        AS_OF,
        region_id="americas",
        settings=_news(cfg_tmp),
        store=store,
        members=members,
        make_embedder=lambda name: None,
        make_sentiment=lambda name: None,
    )

    assert report.summary() == (
        "1 noticia -> 1 tema, sin duplicados; sin embeddings ni sentimiento (falta el extra nlp)"
    )
    assert store.load("americas", AS_OF)["sentiment_label"].isna().all()


def test_run_dedup_sentiment_rejects_a_model_that_answers_short(
    cfg_tmp: AppConfig, stores: Stores
) -> None:
    store, members = stores

    class Short:
        def score(self, texts: Sequence[str]) -> list[Sentiment]:
            return []

    with pytest.raises(ValueError, match="devolvió 0 de 1"):
        run_dedup_sentiment(
            _frame(_article("a")),
            AS_OF,
            region_id="americas",
            settings=_news(cfg_tmp),
            store=store,
            members=members,
            make_embedder=lambda name: None,
            make_sentiment=lambda name: Short(),
        )


# --- paso --------------------------------------------------------------------------------


def _ctx(cfg_tmp: AppConfig, targets: pd.DataFrame | None) -> StepContext:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    if targets is not None:
        ctx.data["news_targets"] = targets
    return ctx


def _targets(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers), "mic": ["XNAS"] * len(tickers)})


def test_step_needs_the_news_step_before(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="el paso news debe ir antes"):
        DedupSentiment().run(_ctx(cfg_tmp, None))


def test_step_skips_without_targets(cfg_tmp: AppConfig) -> None:
    ctx = _ctx(cfg_tmp, pd.DataFrame(columns=["ticker", "mic"]))

    outcome = DedupSentiment().run(ctx)

    assert outcome.status is StepStatus.SKIPPED
    assert ctx.data["clusters"].empty


def test_step_skips_when_prod_news_is_empty(cfg_tmp: AppConfig) -> None:
    outcome = DedupSentiment().run(_ctx(cfg_tmp, _targets("AAPL")))

    assert outcome.status is StepStatus.SKIPPED
    assert "sin noticias" in outcome.message


def test_step_groups_what_the_news_step_saved(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("load_embedder", "load_sentiment"):
        monkeypatch.setattr(f"analyzer.steps.dedup_sentiment.service.{name}", lambda model: None)
    now = datetime.now(UTC)
    frame = _frame(
        _article("a", title="Apple sube"),
        _article("b", title="Apple sube", source="cnbc"),
        _article("c", title="Otra noticia sin relación"),
    )
    frame["published_at"] = [now - timedelta(hours=1), now, now]
    with connect(cfg_tmp) as con:
        NewsStore(con, "prod").upsert(frame, AS_OF)

    ctx = _ctx(cfg_tmp, _targets("AAPL"))
    outcome = DedupSentiment().run(ctx)

    assert outcome.status is StepStatus.OK
    assert outcome.message.startswith("3 noticias -> 2 temas, 1 duplicado (1 exacto)")
    assert len(ctx.data["clusters"]) == 2
    assert ctx.data["dedup"].region_id == "americas"
    with connect(cfg_tmp) as con:
        assert len(NewsClusterStore(con, "prod").load("americas", AS_OF)) == 2
