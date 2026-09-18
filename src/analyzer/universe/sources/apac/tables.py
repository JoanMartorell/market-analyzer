"""Tablas de componentes de los índices de Asia-Pacífico en Wikipedia."""

from analyzer.universe.sources.wikipedia_table import WikipediaTable, code_ticker

# La Wikipedia inglesa no lista los componentes del Nikkei; la japonesa sí,
# repartidos en una tabla por sector con columnas 証券コード (código) y 銘柄 (valor).
NIKKEI225 = WikipediaTable(
    index_id="nikkei225",
    url="https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1",
    ticker_column="証券コード",
    name_column="銘柄",
    mic="XTKS",
    currency="JPY",
    clean_ticker=code_ticker(4),
    concat_all=True,
)

HSI = WikipediaTable(
    index_id="hsi",
    url="https://en.wikipedia.org/wiki/Hang_Seng_Index",
    ticker_column="Ticker",  # "SEHK: 5" -> "0005"
    name_column="Name",
    sector_column="Sub-index",
    mic="XHKG",
    currency="HKD",
    clean_ticker=code_ticker(4),
)

ASX200 = WikipediaTable(
    index_id="asx200",
    url="https://en.wikipedia.org/wiki/S%26P/ASX_200",
    ticker_column="Code",
    name_column="Company",
    sector_column="Sector",
    mic="XASX",
    currency="AUD",
)

KOSPI200 = WikipediaTable(
    index_id="kospi200",
    url="https://en.wikipedia.org/wiki/KOSPI_200",
    ticker_column="Symbol",
    name_column="Company",
    sector_column="GICS Sector",
    mic="XKRX",
    currency="KRW",
    clean_ticker=code_ticker(6),
)
