"""Tablas de componentes de los índices de Latinoamérica en Wikipedia.

Ibovespa (B3, São Paulo) y S&P/BMV IPC (Bolsa Mexicana). El artículo inglés del
Ibovespa no lista la composición (solo una caja de navegación); la tabla vive en
la Wikipedia en portugués, con cabeceras en portugués y filas vacías entre
valores. El IPC está en la Wikipedia inglesa bajo su nombre en castellano. Las
cabeceras se dan con varios candidatos: son páginas que se editan a menudo.
"""

from analyzer.universe.sources.wikipedia_table import WikipediaTable, exchange_code_ticker

IBOVESPA = WikipediaTable(
    index_id="ibovespa",
    url="https://pt.wikipedia.org/wiki/Lista_de_companhias_citadas_no_Ibovespa",
    ticker_column=("Ticker", "Symbol", "Code", "Código"),
    name_column=("Company", "Name", "Stock", "Empresa", "Ação"),
    sector_column=("Sector", "Industry", "Setor"),
    mic="BVMF",
    currency="BRL",
    clean_ticker=exchange_code_ticker,
)

IPC = WikipediaTable(
    index_id="sp_bmv_ipc",
    url="https://en.wikipedia.org/wiki/%C3%8Dndice_de_Precios_y_Cotizaciones",
    ticker_column=("Ticker", "Symbol", "Ticker symbol", "Code", "Clave de pizarra"),
    name_column=("Company", "Name", "Issuer", "Empresa", "Emisora"),
    sector_column=("Sector", "Industry", "GICS Sector"),
    mic="XMEX",
    currency="MXN",
    clean_ticker=exchange_code_ticker,
)
