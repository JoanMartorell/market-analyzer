"""Tabla de componentes del STOXX Europe 600 en Wikipedia.

La tabla trae ticker, empresa, sector ICB y país. No trae bolsa, así que se
infiere del país: la cotización principal de una empresa suele estar en su
mercado doméstico. Países sin bolsa en la región (Bermuda, Grecia, Israel,
Luxemburgo) quedan con ``mic`` vacío.
"""

from analyzer.universe.sources.wikipedia_table import WikipediaTable

COUNTRY_MARKETS: dict[str, tuple[str, str]] = {
    "United Kingdom": ("XLON", "GBP"),
    "Germany": ("XETR", "EUR"),
    "France": ("XPAR", "EUR"),
    "Switzerland": ("XSWX", "CHF"),
    "Sweden": ("XSTO", "SEK"),
    "Netherlands": ("XAMS", "EUR"),
    "Spain": ("XMAD", "EUR"),
    "Italy": ("XMIL", "EUR"),
    "Belgium": ("XBRU", "EUR"),
    "Finland": ("XHEL", "EUR"),
    "Norway": ("XOSL", "NOK"),
    "Denmark": ("XCSE", "DKK"),
    "Austria": ("XWBO", "EUR"),
    "Ireland": ("XDUB", "EUR"),
    "Portugal": ("XLIS", "EUR"),
    "Poland": ("XWAR", "PLN"),
}

STOXX600 = WikipediaTable(
    index_id="stoxx600",
    url="https://en.wikipedia.org/wiki/STOXX_Europe_600",
    ticker_column="Ticker",
    name_column="Company",
    sector_column="ICB Sector",
    market_column="Country",
    market_map=COUNTRY_MARKETS,
)
