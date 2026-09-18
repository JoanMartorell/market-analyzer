"""Enriquecedor: lista actual del S&P 500 en Wikipedia.

Aporta nombre, sector GICS, subindustria y CIK de los miembros actuales, y
añade como intervalo abierto cualquier ticker que Wikipedia liste y la
fuente aún no tenga (altas muy recientes). Los abiertos que Wikipedia ya
no lista se reportan sin tocarlos.
"""

from __future__ import annotations

import io
from datetime import date

import httpx
import pandas as pd

from analyzer.universe.base import FetchContext, Notes
from analyzer.universe.helpers import DEFAULT_USER_AGENT, canonical_ticker, merge_fill

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SOURCE_NAME = "wikipedia:current"


class WikipediaCurrentList:
    name = SOURCE_NAME

    def enrich(
        self, table: pd.DataFrame, client: httpx.Client, ctx: FetchContext
    ) -> tuple[pd.DataFrame, Notes]:
        response = client.get(
            WIKIPEDIA_URL, headers={"User-Agent": ctx.user_agent or DEFAULT_USER_AGENT}
        )
        response.raise_for_status()
        return apply_wikipedia(table, parse_wikipedia_html(response.text), ctx.today)


def parse_wikipedia_html(html: str) -> pd.DataFrame:
    """Tabla de componentes -> DataFrame[ticker, name, sector, sub_industry, cik, date_added]."""
    wanted = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Date added", "CIK"}
    for table in pd.read_html(io.StringIO(html)):
        if wanted <= {str(c) for c in table.columns}:
            break
    else:
        raise ValueError("Wikipedia: no se encontró la tabla de componentes actuales")

    date_added = table["Date added"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].map(canonical_ticker),
            "name": table["Security"].astype(str).str.strip(),
            "sector": table["GICS Sector"].astype(str).str.strip(),
            "sub_industry": table["GICS Sub-Industry"].astype(str).str.strip(),
            "cik": pd.to_numeric(table["CIK"], errors="coerce").astype("Int64"),
            "date_added": pd.to_datetime(date_added, errors="coerce"),
        }
    )
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def apply_wikipedia(
    table: pd.DataFrame, wikipedia: pd.DataFrame, today: date
) -> tuple[pd.DataFrame, Notes]:
    """Lógica pura: altas recientes + enriquecimiento por ticker."""
    open_tickers = set(table.loc[table["end"].isna(), "ticker"])
    fresh = wikipedia.loc[~wikipedia["ticker"].isin(open_tickers)]
    if not fresh.empty:
        additions = pd.DataFrame(
            {
                "ticker": fresh["ticker"].to_numpy(),
                "start": fresh["date_added"].fillna(pd.Timestamp(today)).to_numpy(),
                "end": pd.NaT,
                "source": SOURCE_NAME,
            }
        )
        table = pd.concat([table, additions], ignore_index=True)

    table = merge_fill(table, wikipedia, ["name", "sector", "sub_industry", "cik"])
    notes: Notes = {
        "added": sorted(fresh["ticker"]),
        "open_not_listed": sorted(open_tickers - set(wikipedia["ticker"])),
    }
    return table, notes
