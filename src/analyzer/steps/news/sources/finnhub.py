"""Fuente Finnhub: ``GET /company-news`` por ticker, con clave y límite de peticiones.

Devuelve una lista JSON con ``headline``, ``summary``, ``url``, ``source`` y
``datetime`` en segundos Unix. El free tier admite 60 llamadas por minuto:
se espacian las peticiones según ``rate_limit_per_minute``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import httpx

from analyzer.steps.news.sources.base import Article, clean_text, get
from core.config import NewsSource as NewsSourceConfig


class FinnhubSource:
    def __init__(
        self,
        name: str,
        config: NewsSourceConfig,
        api_key: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.kind != "api" or not config.base_url:
            raise ValueError(f"fuente {name!r} no es una API con base_url")
        self._name = name
        self._config = config
        self._api_key = api_key
        self._sleep = sleep
        limit = config.rate_limit_per_minute
        self._interval = 60.0 / limit if limit else 0.0

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
        return True

    def fetch(self, http: httpx.Client, tickers: Sequence[str], since: datetime) -> list[Article]:
        url = f"{(self._config.base_url or '').rstrip('/')}/company-news"
        window = {"from": since.date().isoformat(), "to": datetime.now(UTC).date().isoformat()}
        articles: list[Article] = []
        for i, ticker in enumerate(tickers):
            if i and self._interval:
                self._sleep(self._interval)
            params = {"symbol": ticker, "token": self._api_key, **window}
            payload = get(http, url, params=params).json()
            if not isinstance(payload, list):
                raise ValueError(f"finnhub devolvió {type(payload).__name__} para {ticker}")
            articles.extend(self._parse(payload, ticker, since))
        return articles

    def _parse(self, payload: list[object], ticker: str, since: datetime) -> list[Article]:
        articles: list[Article] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            stamp = item.get("datetime")
            url = clean_text(item.get("url"))
            title = clean_text(item.get("headline"))
            if not isinstance(stamp, int | float) or not url or not title:
                continue
            published = datetime.fromtimestamp(float(stamp), tz=UTC)
            if published < since:
                continue
            articles.append(
                Article(
                    source=self.name,
                    ticker=ticker,
                    title=title,
                    summary=clean_text(item.get("summary")),
                    url=url,
                    published_at=published,
                )
            )
        return articles
