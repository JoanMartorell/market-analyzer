"""Contrato de una fuente de noticias y utilidades comunes.

Una fuente recibe los tickers que interesan y desde cuándo, y devuelve
artículos ya limpios: título y resumen sin HTML, url y fecha de publicación
en UTC. Las fuentes por valor (``per_ticker``) hacen una petición por
ticker y etiquetan cada artículo con él; las globales traen la portada del
medio y dejan ``ticker`` a ``None``. Una fuente que no puede responder
lanza ``httpx.HTTPError``; el servicio la cuenta como fallida y sigue con
las demás.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

USER_AGENT = "market-analyzer/0.1 (+https://github.com/joanmartorell/market-analyzer)"
_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Article:
    source: str
    ticker: str | None  # None en fuentes globales
    title: str
    summary: str
    url: str
    published_at: datetime  # consciente, en UTC


class ArticleSource(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def reliability(self) -> float: ...

    @property
    def language(self) -> str: ...

    @property
    def per_ticker(self) -> bool: ...

    def fetch(
        self, http: httpx.Client, tickers: Sequence[str], since: datetime
    ) -> list[Article]: ...


def clean_text(raw: object) -> str:
    """Quita etiquetas HTML, deshace entidades y colapsa espacios."""
    if not isinstance(raw, str):
        return ""
    return _SPACES.sub(" ", html.unescape(_TAGS.sub(" ", raw))).strip()


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def get(http: httpx.Client, url: str, params: dict[str, str] | None = None) -> httpx.Response:
    """GET con tres intentos ante fallos de red; un 4xx/5xx es ``HTTPStatusError``."""
    response = http.get(url, params=params)
    response.raise_for_status()
    return response


def new_client() -> httpx.Client:
    return httpx.Client(timeout=20.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})
