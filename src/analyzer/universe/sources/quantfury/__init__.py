"""Universos del catálogo de Quantfury: lo que de verdad se puede negociar.

Tres universos, uno por región, porque cada región cierra a una hora:

- ``quantfury_us``: NYSE y NASDAQ (acciones y ETF, en USD).
- ``quantfury_latam``: B3 (BRL) y BMV (MXN).
- ``quantfury_europe``: acciones europeas que el bróker negocia en Cboe
  Europe en EUR, colocadas en su bolsa de origen.

Todos son instantáneas (``history=False``) del mismo ``instruments.csv``.
"""

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.quantfury.catalogue import CATALOGUE_FILE, QuantfuryCatalogue
from analyzer.universe.sources.yahoo_symbols import YahooSymbols

US_MARKETS: dict[str, str] = {"XNYS": "USD", "XNAS": "USD"}
LATAM_MARKETS: dict[str, str] = {"BVMF": "BRL", "XMEX": "MXN"}
EUROPE_EUR_MARKETS: dict[str, str] = {
    mic: "EUR"
    for mic in ("XETR", "XPAR", "XAMS", "XMAD", "XMIL", "XBRU", "XHEL", "XWBO", "XDUB", "XLIS")
}

QUANTFURY_US = UniverseSpec(
    id="quantfury_us",
    currency="USD",
    source=QuantfuryCatalogue(["NYSE", "NASDAQ"]),
    enrichers=(YahooSymbols(markets=US_MARKETS),),
    history=False,
)

QUANTFURY_LATAM = UniverseSpec(
    id="quantfury_latam",
    currency="USD",  # no hay divisa común; cada fila lleva la suya
    source=QuantfuryCatalogue(["B3", "BMV"]),
    enrichers=(YahooSymbols(markets=LATAM_MARKETS),),
    history=False,
)

QUANTFURY_EUROPE = UniverseSpec(
    id="quantfury_europe",
    currency="EUR",
    source=QuantfuryCatalogue(["Cboe Europe"]),
    enrichers=(YahooSymbols(markets=EUROPE_EUR_MARKETS),),
    history=False,
)

__all__ = [
    "CATALOGUE_FILE",
    "EUROPE_EUR_MARKETS",
    "LATAM_MARKETS",
    "QUANTFURY_EUROPE",
    "QUANTFURY_LATAM",
    "QUANTFURY_US",
    "US_MARKETS",
    "QuantfuryCatalogue",
]
