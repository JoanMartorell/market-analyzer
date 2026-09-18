"""Catálogo de indicadores. Añadir uno es añadir una entrada a ``INDICATORS``.

Cada indicador recibe las velas de un solo valor ordenadas por fecha (las
columnas de ``prod.prices``) y devuelve un DataFrame con sus columnas,
alineado por índice. Todos trabajan sobre el cierre ajustado: un split o un
dividendo no debe aparecer como movimiento. Las columnas base (``close``,
``adj_close``, ``volume``) se copian tal cual a la tabla de indicadores para
que las reglas las usen como variables.

``warmup`` es cuántas velas previas necesita un indicador para dar un valor
completo. Por debajo de eso el resultado es un artefacto de la ventana, no
del mercado, y el servicio no lo persiste.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

Compute = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class Indicator:
    name: str
    columns: tuple[str, ...]
    warmup: int  # velas previas necesarias para un valor completo
    compute: Compute


# --- implementaciones -----------------------------------------------------------


def _rsi(length: int) -> Compute:
    def compute(bars: pd.DataFrame) -> pd.DataFrame:
        return _frame(ta.rsi(bars["adj_close"], length=length), [f"rsi_{length}"], bars.index)

    return compute


def _sma(length: int) -> Compute:
    def compute(bars: pd.DataFrame) -> pd.DataFrame:
        return _frame(ta.sma(bars["adj_close"], length=length), [f"sma_{length}"], bars.index)

    return compute


def _macd(bars: pd.DataFrame) -> pd.DataFrame:
    # pandas-ta devuelve MACD, histograma y señal, en ese orden.
    result = ta.macd(bars["adj_close"], fast=12, slow=26, signal=9)
    return _frame(result, ["macd", "macd_hist", "macd_signal"], bars.index)


def _atr(length: int) -> Compute:
    def compute(bars: pd.DataFrame) -> pd.DataFrame:
        factor = bars["adj_close"] / bars["close"]  # lleva máximo y mínimo a la escala ajustada
        result = ta.atr(
            bars["high"] * factor, bars["low"] * factor, bars["adj_close"], length=length
        )
        return _frame(result, [f"atr_{length}"], bars.index)

    return compute


def _avg_volume(length: int) -> Compute:
    def compute(bars: pd.DataFrame) -> pd.DataFrame:
        mean = bars["volume"].astype("float64").rolling(length).mean()
        return pd.DataFrame({f"avg_volume_{length}": mean})

    return compute


def _returns(*lengths: int) -> Compute:
    def compute(bars: pd.DataFrame) -> pd.DataFrame:
        adj = bars["adj_close"]
        return pd.DataFrame({f"ret_{n}d": adj / adj.shift(n) - 1 for n in lengths})

    return compute


def _frame(result: object, columns: list[str], index: pd.Index) -> pd.DataFrame:
    """Normaliza la salida de pandas-ta: ``None`` (serie más corta que el periodo) pasa a NaN."""
    if result is None:
        return pd.DataFrame(float("nan"), index=index, columns=columns)
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"pandas-ta devolvió {type(result).__name__}")
    return result.set_axis(columns, axis=1)


# --- catálogo -------------------------------------------------------------------

INDICATORS: tuple[Indicator, ...] = (
    Indicator("rsi_14", ("rsi_14",), 14, _rsi(14)),
    Indicator("sma_20", ("sma_20",), 19, _sma(20)),
    Indicator("sma_50", ("sma_50",), 49, _sma(50)),
    Indicator("sma_200", ("sma_200",), 199, _sma(200)),
    Indicator("macd", ("macd", "macd_hist", "macd_signal"), 33, _macd),
    Indicator("atr_14", ("atr_14",), 14, _atr(14)),
    Indicator("avg_volume_20", ("avg_volume_20",), 19, _avg_volume(20)),
    Indicator("returns", ("ret_1d", "ret_5d", "ret_20d"), 20, _returns(1, 5, 20)),
)

INDICATOR_COLUMNS: tuple[str, ...] = tuple(c for ind in INDICATORS for c in ind.columns)
WARMUP = max(ind.warmup for ind in INDICATORS)

if len(set(INDICATOR_COLUMNS)) != len(INDICATOR_COLUMNS):
    raise RuntimeError("catálogo de indicadores con columnas repetidas")


def compute_for_key(bars: pd.DataFrame) -> pd.DataFrame:
    """Todos los indicadores del catálogo para las velas de un valor, ya ordenadas por fecha."""
    parts: list[pd.DataFrame] = []
    for indicator in INDICATORS:
        values = indicator.compute(bars)
        if tuple(values.columns) != indicator.columns:
            raise ValueError(
                f"indicador {indicator.name!r} devolvió {list(values.columns)}, "
                f"se esperaba {list(indicator.columns)}"
            )
        parts.append(values)
    return pd.concat(parts, axis=1)
