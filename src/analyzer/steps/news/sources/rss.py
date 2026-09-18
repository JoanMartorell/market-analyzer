"""Fuente RSS genérica: cualquier entrada ``kind: rss`` de ``sources.yaml``.

Por valor (``url_template`` con ``{ticker}``) o global (``url``). El símbolo
en la URL sigue la convención de Yahoo, con guion en las clases (``BRK-B``),
que es la de las únicas fuentes RSS por valor que hay. Una entrada sin
fecha o sin enlace se descarta: sin fecha no hay ventana que aplicar.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime

import feedparser
import httpx
import structlog

from analyzer.steps.news.sources.base import Article, clean_text, get
from core.config import NewsSource as NewsSourceConfig

log = structlog.get_logger(__name__)


def feed_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(" ", "-").replace(".", "-")


class RssSource:
    def __init__(self, name: str, config: NewsSourceConfig) -> None:
        if config.kind != "rss":
            raise ValueError(f"fuente {name!r} no es RSS")
        self._name = name
        self._config = config

    @property
    def name(self) -> str:
        return self._name

    @property
    def reliability(self) -> float:
        return self._config.reliability

    @property
    def language(self) -> str:
        return self._config.language

    @property
    def per_ticker(self) -> bool:
        return self._config.per_ticker

    def fetch(self, http: httpx.Client, tickers: Sequence[str], since: datetime) -> list[Article]:
        if not self.per_ticker:
            url = self._config.url or ""
            return parse_feed(get(http, url).text, source=self.name, ticker=None, since=since)

        template = self._config.url_template or ""
        articles: list[Article] = []
        failures: list[str] = []
        for ticker in tickers:
            url = template.format(ticker=feed_symbol(ticker))
            try:
                text = get(http, url).text
            except httpx.HTTPError as exc:
                failures.append(f"{ticker}: {exc}")
                log.warning("news.feed_failed", source=self.name, ticker=ticker, detail=str(exc))
                continue
            articles.extend(parse_feed(text, source=self.name, ticker=ticker, since=since))
        if tickers and len(failures) == len(tickers):
            raise httpx.HTTPError(f"{self.name}: ningún feed respondió ({failures[0]})")
        return articles


def parse_feed(text: str, *, source: str, ticker: str | None, since: datetime) -> list[Article]:
    """Entradas del feed publicadas desde ``since`` (inclusive), como artículos limpios."""
    feed = feedparser.parse(text)
    articles: list[Article] = []
    for entry in feed.entries:
        published = _published(entry)
        url = clean_text(entry.get("link"))
        title = clean_text(entry.get("title"))
        if published is None or published < since or not url or not title:
            continue
        articles.append(
            Article(
                source=source,
                ticker=ticker,
                title=title,
                summary=clean_text(entry.get("summary")),
                url=url,
                published_at=published,
            )
        )
    return articles


def _published(entry: feedparser.FeedParserDict) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not isinstance(parsed, time.struct_time):
        return None
    return datetime(*parsed[:6], tzinfo=UTC)  # feedparser ya lo pasa a UTC
