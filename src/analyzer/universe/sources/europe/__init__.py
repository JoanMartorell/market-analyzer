"""Europa: STOXX Europe 600 según la tabla de componentes de Wikipedia.

Sin histórico gratuito: ``history=False``, el fichero acumula cambios desde
el primer build. La bolsa y la divisa se derivan del país de la tabla. La
tabla trae bastantes códigos Reuters en vez de símbolos de bolsa, así que
después se pasan por Yahoo para dejar el ticker con el que hay precios.
"""

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.europe.stoxx600 import COUNTRY_MARKETS, STOXX600
from analyzer.universe.sources.wikipedia_table import WikipediaSnapshot
from analyzer.universe.sources.yahoo_symbols import YahooSymbols

EUROPE_MARKETS: dict[str, str] = dict(COUNTRY_MARKETS.values())  # mic -> divisa

EUROPE = UniverseSpec(
    id="stoxx600",
    currency="EUR",
    source=WikipediaSnapshot("wikipedia:stoxx600", [STOXX600]),
    enrichers=(YahooSymbols(markets=EUROPE_MARKETS),),
    history=False,
)

__all__ = ["EUROPE", "STOXX600"]
