"""Descarga de noticias para un conjunto de valores y su escritura en ``prod.news``.

Pide a cada fuente los artículos de las últimas ``lookback_hours`` para los
valores objetivo (candidatos del screener y posiciones abiertas). Una fuente
que falla se anota y no detiene a las demás; que fallen todas sí es un
error, porque entonces el día no tiene contexto. Las fuentes globales
(portada de un medio) se bajan una vez y sus artículos no llevan valor.

Deja los artículos de esta ejecución, ya con id, para el paso de
deduplicación. Repetir el ciclo no duplica filas: el id es determinista.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd
import structlog

from analyzer.steps.news.sources import Article, ArticleSource
from analyzer.storage import NEWS_COLUMNS, NewsStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SourceResult:
    name: str
    articles: int
    reason: str | None = None  # "sin clave", "sin respuesta"... si no se usó
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class NewsReport:
    as_of: date
    since: datetime
    targets: int  # valores para los que se pidieron noticias
    fetched: int  # artículos únicos dentro de la ventana
    new: int  # que no estaban ya guardados
    sources: tuple[SourceResult, ...]
    articles: pd.DataFrame = field(repr=False, compare=False)

    def summary(self) -> str:
        head = {0: "sin noticias", 1: "1 noticia"}.get(self.fetched, f"{self.fetched} noticias")
        text = f"{head} de {self.targets} valores"
        if self.fetched:
            text += f", {self.new} nuevas"
        ok = [f"{s.name} {s.articles}" for s in self.sources if s.ok]
        unused = [f"{s.name} {s.reason}" for s in self.sources if not s.ok]
        parts = [", ".join(ok)] if ok else []
        parts.extend(unused)
        return f"{text} ({'; '.join(parts)})" if parts else text


def run_news(
    targets: pd.DataFrame,
    as_of: date,
    *,
    sources: Sequence[ArticleSource],
    store: NewsStore,
    http: httpx.Client,
    lookback_hours: int,
    now: datetime | None = None,
    unavailable: Sequence[SourceResult] = (),
) -> NewsReport:
    """``unavailable``: fuentes configuradas que no se pudieron construir (sin clave...)."""
    if not sources:
        raise ValueError("no hay fuentes de noticias disponibles")
    moment = now or datetime.now(UTC)
    since = moment - timedelta(hours=lookback_hours)
    keys = targets.loc[:, ["ticker", "mic"]].astype(str).drop_duplicates("ticker")
    tickers = sorted(keys["ticker"])
    mics = dict(zip(keys["ticker"], keys["mic"], strict=True))

    results: list[SourceResult] = list(unavailable)
    collected: list[Article] = []
    for source in sources:
        try:
            articles = source.fetch(http, tickers, since)
        except (httpx.HTTPError, ValueError, OSError) as exc:
            log.warning("news.source_failed", source=source.name, detail=str(exc))
            results.append(SourceResult(source.name, 0, reason="sin respuesta", error=str(exc)))
            continue
        results.append(SourceResult(source.name, len(articles)))
        collected.extend(articles)
    if not any(r.ok for r in results):
        raise ValueError(
            "ninguna fuente de noticias respondió: "
            + "; ".join(f"{r.name}: {r.error or r.reason}" for r in results)
        )

    frame = to_frame(collected, sources, mics, since=since, until=moment)
    existing = store.existing_ids(frame["id"]) if not frame.empty else set()
    store.upsert(frame, as_of)
    report = NewsReport(
        as_of=as_of,
        since=since,
        targets=len(tickers),
        fetched=len(frame),
        new=int((~frame["id"].isin(existing)).sum()) if not frame.empty else 0,
        sources=tuple(results),
        articles=frame,
    )
    log.info("news.fetched", detail=report.summary())
    return report


def to_frame(
    articles: Sequence[Article],
    sources: Sequence[ArticleSource],
    mics: dict[str, str],
    *,
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    """Artículos como filas de ``prod.news``: con id, fiabilidad e idioma de su fuente."""
    meta = {s.name: s for s in sources}
    rows = []
    for a in articles:
        if not (since <= a.published_at <= until + timedelta(hours=1)):  # relojes adelantados
            continue
        rows.append(
            {
                "id": article_id(a),
                "ticker": a.ticker,
                "mic": mics.get(a.ticker) if a.ticker else None,
                "source": a.source,
                "reliability": meta[a.source].reliability,
                "language": meta[a.source].language,
                "published_at": a.published_at.astimezone(UTC),
                "title": a.title,
                "summary": a.summary,
                "url": a.url,
            }
        )
    frame = pd.DataFrame(rows, columns=list(NEWS_COLUMNS))
    frame = frame.drop_duplicates("id").sort_values(["published_at", "id"], ascending=[False, True])
    return frame.reset_index(drop=True)


def article_id(article: Article) -> str:
    raw = f"{article.source}|{article.ticker or ''}|{article.url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
