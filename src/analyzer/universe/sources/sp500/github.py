"""Fuente: intervalos de pertenencia al S&P 500 desde 1996.

Dataset fja05680/sp500 en GitHub, derivado de los datos del libro
"Trading Evolved" de Clenow y mantenido a mano contra Wikipedia. Es lo más
completo que existe gratis; la tabla de cambios de Wikipedia ya no está y
de todos modos era parcial.
"""

from __future__ import annotations

import io

import httpx
import pandas as pd

from analyzer.universe.base import FetchContext
from analyzer.universe.helpers import DEFAULT_USER_AGENT, canonical_ticker

GITHUB_INTERVALS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)


class GithubIntervals:
    name = "github:fja05680/sp500"

    def fetch(self, client: httpx.Client, ctx: FetchContext) -> pd.DataFrame:
        response = client.get(GITHUB_INTERVALS_URL, headers={"User-Agent": DEFAULT_USER_AGENT})
        response.raise_for_status()
        return parse_intervals_csv(response.text)


def parse_intervals_csv(text: str) -> pd.DataFrame:
    """CSV ``ticker,start_date,end_date`` -> DataFrame[ticker, start, end]."""
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    expected = {"ticker", "start_date", "end_date"}
    if not expected <= set(raw.columns):
        raise ValueError(f"CSV de intervalos sin columnas {expected}: {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "ticker": raw["ticker"].map(canonical_ticker),
            "start": pd.to_datetime(raw["start_date"], errors="coerce"),
            "end": pd.to_datetime(raw["end_date"], errors="coerce"),
        }
    )
    return out.dropna(subset=["ticker"]).sort_values(["ticker", "start"]).reset_index(drop=True)
