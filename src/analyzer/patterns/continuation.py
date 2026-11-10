"""Figuras de continuación: triángulo ascendente y bandera alcista.

Como las de giro, se escriben una vez en alcista; el triángulo descendente
y la bandera bajista son las mismas con el precio cambiado de signo.

- Triángulo ascendente: dos máximos a la misma altura y tres mínimos cada
  vez más altos. Se rompe la resistencia. Sale de los pivotes, como las
  figuras de giro. Los tres mínimos crecientes lo separan del
  hombro-cabeza-hombro invertido, que también termina en un mínimo más alto
  bajo una línea plana pero viene de un hombro más alto que la cabeza.

- Bandera alcista: un mástil (subida fuerte en pocas sesiones), una pausa
  corta que devuelve como mucho la mitad del mástil, y el cierre que supera
  el máximo del mástil. No sale de los pivotes: la pausa de una bandera es
  demasiado pequeña para el zigzag, así que se mira directamente en las
  velas anteriores a la que rompe. Romper el máximo del mástil es la
  definición exigente (la laxa es romper el canal de la pausa); a cambio no
  depende de trazar ninguna línea a ojo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from analyzer.patterns.model import PatternParams
from analyzer.patterns.pivots import Array
from analyzer.patterns.reversal import Point, Shape


def ascending_triangle(points: Sequence[Point], atr: float, params: PatternParams) -> Shape | None:
    if len(points) < 5 or not points[-1].is_low:
        return None
    origin, first_top, first_low, second_top, second_low = points[-5:]
    tolerance = params.equal_tolerance_atr * atr
    if abs(first_top.value - second_top.value) > tolerance:
        return None
    # Mínimos crecientes de verdad: por encima de lo que en un doble suelo sería "igual".
    if not (
        first_low.value - origin.value > tolerance
        and second_low.value - first_low.value > tolerance
    ):
        return None
    resistance = max(first_top.value, second_top.value)
    height = resistance - first_low.value
    if height < params.min_height_atr * atr:
        return None
    if second_low.index - first_top.index < params.min_separation:
        return None
    return Shape(
        pivots=5,
        start=first_top.index,
        anchor=(second_top.index, resistance),
        slope=0.0,
        height=height,
        stop=second_low.value - params.stop_buffer_atr * atr,
    )


@dataclass(frozen=True)
class Flag:
    """Bandera completada en una vela, en el espacio normalizado (alcista)."""

    base: tuple[int, float]  # arranque del mástil
    top: tuple[int, float]  # máximo del mástil: la línea que se rompe
    low: tuple[int, float]  # mínimo de la pausa
    stop: float
    target: float


def bull_flag(
    high: Array, low: Array, close: Array, atr: Array, t: int, params: PatternParams
) -> Flag | None:
    """La bandera que el cierre de ``t`` completa, si la hay. Solo mira velas hasta ``t``."""
    start = t - params.flag_max_bars
    if start - params.pole_max_bars < 0:
        return None
    top_index = _extreme(high, start, t, np.nanargmax)
    if top_index is None or t - top_index < params.flag_min_bars:
        return None
    top = float(high[top_index])
    if not close[t] > top:
        return None

    pole_start = top_index - params.pole_max_bars
    base_index = _extreme(low, pole_start, top_index, np.nanargmin)
    if base_index is None or float(np.nanmax(high[pole_start:top_index])) > top:
        return None  # el máximo no es el final de una subida: es un techo anterior
    pole = top - float(low[base_index])
    reference = float(atr[top_index])
    if not pole >= params.pole_min_atr * reference:
        return None

    low_index = _extreme(low, top_index + 1, t, np.nanargmin)
    if low_index is None:
        return None
    flag_low = float(low[low_index])
    if top - flag_low > params.flag_max_retrace * pole:
        return None  # devolvió demasiado: ya no es una pausa
    target = flag_low + pole  # el mástil, proyectado desde el mínimo de la pausa
    if not close[t] < target:
        return None
    return Flag(
        base=(base_index, float(low[base_index])),
        top=(top_index, top),
        low=(low_index, flag_low),
        stop=flag_low - params.stop_buffer_atr * reference,
        target=target,
    )


def _extreme(series: Array, start: int, stop: int, pick: Callable[[Array], Any]) -> int | None:
    """Posición del extremo de ``series[start:stop]``; ``None`` si el tramo no tiene datos."""
    window = series[start:stop]
    if len(window) == 0 or bool(np.isnan(window).all()):
        return None
    return start + int(pick(window))
