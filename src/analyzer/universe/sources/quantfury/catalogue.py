"""Fuente: catálogo de instrumentos negociables en Quantfury, versionado en el repo.

Quantfury no publica su catálogo en una API abierta, así que se guarda una
foto en ``instruments.csv`` junto a este módulo. Una fila por instrumento
negociable (acciones y ETF), con la bolsa tal como la nombra el bróker
(``exchange``), su símbolo (``symbol``) y el ticker y la bolsa (``ticker``,
``mic``) con los que hay precios en la cotización local.

En casi todo ``ticker`` coincide con ``symbol``. Las acciones que el bróker
negocia en Cboe Europe se liquidan en EUR pero no traen bolsa de origen: el
fichero les pone la de su cotización principal (Xetra, París, Ámsterdam...),
que es la que tiene velas diarias en Yahoo, y el ticker local cuando el del
bróker no coincide (RYAAY es RYA en Dublín, SHEL es SHELL en Ámsterdam).
El enriquecedor de Yahoo corrige lo que aún no cuadre.

Sin histórico: ``history=False``. Actualizar el catálogo es editar el CSV;
el build siguiente cierra lo que desaparece y abre lo que entra.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import httpx
import pandas as pd

from analyzer.universe.base import FetchContext
from analyzer.universe.helpers import canonical_ticker

CATALOGUE_FILE = Path(__file__).with_name("instruments.csv")
CATALOGUE_COLUMNS: tuple[str, ...] = (
    "exchange",
    "symbol",
    "ticker",
    "mic",
    "currency",
    "type",
    "name",
)
ETF_TYPE = "Etfs"
ETF_SECTOR = "ETF"


class QuantfuryCatalogue:
    """Instrumentos del catálogo en las bolsas ``exchanges`` (nombres del bróker)."""

    def __init__(self, exchanges: Sequence[str], path: Path = CATALOGUE_FILE) -> None:
        if not exchanges:
            raise ValueError("QuantfuryCatalogue necesita al menos una bolsa")
        self.exchanges = tuple(exchanges)
        self.path = path
        self.name = "quantfury:" + "+".join(e.lower().replace(" ", "_") for e in self.exchanges)

    def fetch(self, client: httpx.Client, ctx: FetchContext) -> pd.DataFrame:
        del client, ctx  # el catálogo es local: ni red ni fecha
        return read_catalogue(self.path, self.exchanges)


def read_catalogue(path: Path, exchanges: Sequence[str]) -> pd.DataFrame:
    """CSV -> DataFrame[ticker, name, sector, mic, currency, start, end, source].

    Falla si faltan columnas, si una bolsa pedida no tiene ninguna fila (un
    nombre mal escrito dejaría la región vacía sin avisar) o si hay claves
    (ticker, mic) repetidas.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in CATALOGUE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {missing}")

    unknown = sorted(set(exchanges) - set(raw["exchange"]))
    if unknown:
        available = ", ".join(sorted(set(raw["exchange"])))
        raise ValueError(f"{path.name}: bolsas sin instrumentos {unknown}; hay: {available}")

    rows = raw.loc[raw["exchange"].isin(list(exchanges))]
    out = pd.DataFrame(
        {
            "ticker": rows["ticker"].map(canonical_ticker),
            "name": rows["name"].str.strip(),
            "sector": rows["type"].map(lambda t: ETF_SECTOR if t == ETF_TYPE else None),
            "mic": rows["mic"].str.strip().str.upper(),
            "currency": rows["currency"].str.strip().str.upper(),
        }
    ).dropna(subset=["ticker"])

    duplicated = out.loc[out.duplicated(["ticker", "mic"]), ["ticker", "mic"]]
    if not duplicated.empty:
        keys = sorted(f"{t}@{m}" for t, m in duplicated.itertuples(index=False))
        raise ValueError(f"{path.name}: claves repetidas {keys}")

    out["start"] = pd.NaT
    out["end"] = pd.NaT
    out["source"] = "quantfury:" + rows["exchange"].str.lower().str.replace(" ", "_")
    return out.reset_index(drop=True)
