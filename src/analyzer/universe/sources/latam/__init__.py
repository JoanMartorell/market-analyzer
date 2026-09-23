"""Latinoamérica: unión del Ibovespa (Brasil) y el S&P/BMV IPC (México) según Wikipedia.

Sin histórico gratuito: ``history=False``, como Europa y APAC. Los tickers se
comprueban contra Yahoo (``.SA`` y ``.MX``) y se corrigen o se descartan.
"""

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.latam.tables import IBOVESPA, IPC
from analyzer.universe.sources.wikipedia_table import WikipediaSnapshot
from analyzer.universe.sources.yahoo_symbols import YahooSymbols

LATAM_MARKETS: dict[str, str] = {"BVMF": "BRL", "XMEX": "MXN"}

LATAM = UniverseSpec(
    id="latam_large_cap",
    currency="USD",  # no hay divisa común; cada fila lleva la suya
    source=WikipediaSnapshot("wikipedia:latam", [IBOVESPA, IPC]),
    enrichers=(YahooSymbols(markets=LATAM_MARKETS),),
    history=False,
)

__all__ = ["IBOVESPA", "IPC", "LATAM", "LATAM_MARKETS"]
