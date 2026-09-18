"""Contrato de un proveedor de precios diarios.

Recibe claves (ticker, mic) y un rango de fechas; devuelve un DataFrame con
``FETCH_COLUMNS`` y, si el proveedor las conoce, ``OPTIONAL_FETCH_COLUMNS``
(``dividend`` por acción y ``split`` como acciones nuevas por antigua; 0 o
NaN cuando no hay evento). La columna ``source`` la añade el servicio de
ingesta con ``provider.name``. Las claves que el proveedor no cubre
simplemente no aparecen en la salida.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

FETCH_COLUMNS = ("ticker", "mic", "date", "open", "high", "low", "close", "adj_close", "volume")
OPTIONAL_FETCH_COLUMNS = ("dividend", "split")
ALL_FETCH_COLUMNS = FETCH_COLUMNS + OPTIONAL_FETCH_COLUMNS


class PriceProvider(Protocol):
    @property
    def name(self) -> str: ...

    def fetch(self, keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame: ...


def empty_fetch() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ALL_FETCH_COLUMNS})
