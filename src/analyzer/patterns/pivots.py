"""Pivotes del precio: un zigzag causal con umbral en ATR.

Un máximo pasa a ser pivote cuando el precio ha caído desde él al menos
``reversal`` veces el ATR del día; un mínimo, cuando ha subido otro tanto.
El pivote se conoce en la vela que confirma el giro, no antes: la detección
de patrones solo usa pivotes ya confirmados, así que lo que se calcula para
un día no cambia cuando llegan los siguientes.

El ATR es una media simple del rango verdadero, no la suavizada de Wilder:
solo recuerda ``length`` velas, y con eso el resultado deja de depender de
dónde empieza la ventana en cuanto el zigzag confirma un pivote.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analyzer.patterns.model import Pivot, PivotKind

Array = NDArray[np.float64]


def average_true_range(high: Array, low: Array, close: Array, length: int) -> Array:
    """Media simple del rango verdadero; NaN hasta tener ``length`` velas."""
    previous = np.concatenate(([np.nan], close[:-1]))
    spread = np.fmax(high - low, np.fmax(np.abs(high - previous), np.abs(low - previous)))
    return pd.Series(spread).rolling(length).mean().to_numpy(dtype="float64")


class ZigZag:
    """Recorre la serie vela a vela y va dejando en ``pivots`` los giros confirmados."""

    def __init__(self, high: Array, low: Array, atr: Array, reversal: float) -> None:
        self._high = high
        self._low = low
        self._atr = atr
        self._reversal = reversal
        self._direction = 0  # +1 tramo alcista (se sigue el máximo), -1 bajista, 0 sin decidir
        self._extreme = -1  # vela del extremo del tramo en curso
        self._first_high = -1  # extremos provisionales mientras no hay dirección
        self._first_low = -1
        self.pivots: list[Pivot] = []

    def step(self, t: int) -> Pivot | None:
        """Procesa la vela ``t``; devuelve el pivote que confirma, si confirma alguno."""
        high, low = self._high[t], self._low[t]
        threshold = self._reversal * self._atr[t]
        if math.isnan(high) or math.isnan(low) or not threshold > 0:
            return None  # vela incompleta o sin ATR todavía: ni mueve extremos ni confirma
        if self._direction == 0:
            return self._start(t, high, low, threshold)
        if self._direction > 0:
            if high > self._high[self._extreme]:
                self._extreme = t
            elif self._high[self._extreme] - low >= threshold:
                return self._confirm(self._extreme, "H", t)
            return None
        if low < self._low[self._extreme]:
            self._extreme = t
        elif high - self._low[self._extreme] >= threshold:
            return self._confirm(self._extreme, "L", t)
        return None

    def _start(self, t: int, high: float, low: float, threshold: float) -> Pivot | None:
        if self._first_high < 0:
            self._first_high = self._first_low = t
            return None
        if high > self._high[self._first_high]:
            self._first_high = t
        if low < self._low[self._first_low]:
            self._first_low = t
        up = self._first_low < t and high - self._low[self._first_low] >= threshold
        down = self._first_high < t and self._high[self._first_high] - low >= threshold
        if up and (not down or self._first_low < self._first_high):
            return self._confirm(self._first_low, "L", t)
        if down:
            return self._confirm(self._first_high, "H", t)
        return None

    def _confirm(self, index: int, kind: PivotKind, t: int) -> Pivot:
        """Fija el pivote y abre el tramo contrario desde su extremo entre el pivote y ``t``."""
        series = self._high if kind == "H" else self._low
        pivot = Pivot(index=index, price=float(series[index]), kind=kind, confirmed_at=t)
        self.pivots.append(pivot)
        if kind == "H":
            self._direction = -1
            self._extreme = index + 1 + int(np.nanargmin(self._low[index + 1 : t + 1]))
        else:
            self._direction = 1
            self._extreme = index + 1 + int(np.nanargmax(self._high[index + 1 : t + 1]))
        return pivot
