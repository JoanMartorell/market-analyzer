"""Funciones permitidas en las expresiones de las reglas.

Cada función recibe series indexadas por (ticker, mic, date), ordenadas por
fecha dentro de cada clave. "Una sesión atrás" es un desplazamiento dentro
de la clave: nunca se mezcla el último día de un valor con el primero del
siguiente. ``lookback`` dice cuántas sesiones anteriores a la ventana de la
regla necesita la función; el screener carga esas sesiones de más para que
la primera sesión de la ventana también pueda evaluarse.

Añadir una función es añadir una entrada a ``FUNCTIONS``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

KEY_LEVELS = ["ticker", "mic"]


@dataclass(frozen=True)
class Function:
    name: str
    call: Callable[..., Any]
    arity: int  # argumentos posicionales exactos
    lookback: int  # sesiones previas a la ventana que necesita


def prev(series: pd.Series, sessions: int = 1) -> pd.Series:
    """La serie ``sessions`` sesiones atrás, dentro de cada clave (NaN al principio)."""
    return series.groupby(level=KEY_LEVELS, sort=False).shift(sessions)


def bullish_cross(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Cierto en la sesión en que ``fast`` pasa de no superar a ``slow`` a estar por encima."""
    return (fast > slow) & (prev(fast) <= prev(slow))


def bearish_cross(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Cierto en la sesión en que ``fast`` pasa de no estar bajo ``slow`` a estar por debajo."""
    return (fast < slow) & (prev(fast) >= prev(slow))


FUNCTIONS: dict[str, Function] = {
    f.name: f
    for f in (
        Function("bullish_cross", bullish_cross, arity=2, lookback=1),
        Function("bearish_cross", bearish_cross, arity=2, lookback=1),
    )
}
