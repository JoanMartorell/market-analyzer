"""Fuentes de noticias, por id de ``config/sources.yaml``.

Las de ``kind: rss`` se resuelven con la fuente RSS genérica; las de
``kind: api`` necesitan implementación propia y clave en ``.env``. Añadir
una API es crear su módulo y registrarla en ``API_SOURCES``.
"""

from __future__ import annotations

from collections.abc import Callable

from analyzer.steps.news.sources.base import (
    USER_AGENT,
    Article,
    ArticleSource,
    clean_text,
    get,
    new_client,
)
from analyzer.steps.news.sources.finnhub import FinnhubSource
from analyzer.steps.news.sources.rss import RssSource, feed_symbol, parse_feed
from core.config import Env
from core.config import NewsSource as NewsSourceConfig

API_SOURCES: dict[str, Callable[[str, NewsSourceConfig, str], ArticleSource]] = {
    "finnhub": FinnhubSource,
}


def build_source(source_id: str, config: NewsSourceConfig, env: Env) -> ArticleSource:
    if config.kind == "rss":
        return RssSource(source_id, config)
    factory = API_SOURCES.get(source_id)
    if factory is None:
        raise ValueError(
            f"fuente de noticias {source_id!r} no implementada; "
            f"APIs disponibles: {', '.join(sorted(API_SOURCES))}"
        )
    key_name = config.api_key_env or ""
    api_key = env.get(key_name) if key_name else None
    if key_name and api_key is None:
        raise ValueError(f"fuente de noticias {source_id!r} necesita {key_name} en .env")
    return factory(source_id, config, api_key or "")


__all__ = [
    "API_SOURCES",
    "USER_AGENT",
    "Article",
    "ArticleSource",
    "FinnhubSource",
    "RssSource",
    "build_source",
    "clean_text",
    "feed_symbol",
    "get",
    "new_client",
    "parse_feed",
]
