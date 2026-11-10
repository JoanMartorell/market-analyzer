"""Figuras de giro sobre los últimos pivotes: doble suelo y hombro-cabeza-hombro invertido.

Las dos figuras se escriben una sola vez, en su versión alcista. Las
bajistas (doble techo, hombro-cabeza-hombro) son las mismas con el precio
cambiado de signo: quien llama pasa los pivotes ya normalizados (``Point``)
y deshace el signo al leer los niveles.

Cada función mira solo los últimos pivotes confirmados y devuelve la figura
*armada*: la línea que hay que romper, la altura y el stop. Que el cierre la
rompa o no es cosa de ``detect``.

- Doble suelo: máximo previo, suelo, rebote y segundo suelo a la misma
  altura. Se rompe el máximo del rebote.
- HCH invertido: máximo previo, hombro, rebote, cabeza más baja, rebote y
  segundo hombro. Se rompe la línea que une los dos rebotes.

Los umbrales de cabeza y de igualdad de suelos se excluyen entre sí: los
mismos pivotes no pueden ser a la vez un doble suelo y el final de un HCH.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from analyzer.patterns.model import PatternParams


@dataclass(frozen=True)
class Point:
    """Un pivote en el espacio normalizado (alcista): precio con signo y si hace de mínimo."""

    index: int
    value: float
    is_low: bool


@dataclass(frozen=True)
class Shape:
    """Figura armada, en el espacio normalizado."""

    pivots: int  # cuántos pivotes del final la forman
    start: int  # vela desde la que se dibuja la línea de ruptura
    anchor: tuple[int, float]  # un punto de la línea de ruptura
    slope: float  # pendiente de esa línea por vela
    height: float
    stop: float

    def level(self, t: int) -> float:
        """La línea de ruptura en la vela ``t``."""
        return self.anchor[1] + self.slope * (t - self.anchor[0])


Detector = Callable[[Sequence[Point], float, PatternParams], Shape | None]


def double_bottom(points: Sequence[Point], atr: float, params: PatternParams) -> Shape | None:
    if len(points) < 4 or not points[-1].is_low:
        return None
    before, first, bounce, second = points[-4:]
    floor = min(first.value, second.value)
    height = bounce.value - floor
    if abs(first.value - second.value) > params.equal_tolerance_atr * atr:
        return None
    if height < params.min_height_atr * atr:
        return None
    if second.index - first.index < params.min_separation:
        return None
    if before.value - bounce.value < params.prior_move * height:
        return None  # sin caída previa no hay nada que girar
    return Shape(
        pivots=4,
        start=first.index,
        anchor=(bounce.index, bounce.value),
        slope=0.0,
        height=height,
        stop=floor - params.stop_buffer_atr * atr,
    )


def inverse_head_shoulders(
    points: Sequence[Point], atr: float, params: PatternParams
) -> Shape | None:
    if len(points) < 6 or not points[-1].is_low:
        return None
    before, left, first_bounce, head, second_bounce, right = points[-6:]
    if head.value >= min(left.value, right.value) - params.head_margin_atr * atr:
        return None
    if abs(left.value - right.value) > params.shoulder_tolerance_atr * atr:
        return None
    slope = (second_bounce.value - first_bounce.value) / (second_bounce.index - first_bounce.index)
    neckline_at_head = first_bounce.value + slope * (head.index - first_bounce.index)
    height = neckline_at_head - head.value
    if height < params.min_height_atr * atr:
        return None
    if abs(second_bounce.value - first_bounce.value) > params.max_neckline_tilt * height:
        return None
    left_side, right_side = head.index - left.index, right.index - head.index
    if left_side + right_side < 2 * params.min_separation:
        return None
    if max(left_side, right_side) > params.max_shoulder_time_ratio * min(left_side, right_side):
        return None
    if before.value - max(first_bounce.value, second_bounce.value) < params.prior_move * height:
        return None
    return Shape(
        pivots=6,
        start=left.index,
        anchor=(second_bounce.index, second_bounce.value),
        slope=slope,
        height=height,
        stop=right.value - params.stop_buffer_atr * atr,
    )
