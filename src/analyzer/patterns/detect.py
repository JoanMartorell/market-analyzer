"""Detección de patrones sobre las velas de un valor, sin mirar al futuro.

La serie se recorre vela a vela. Cada vez que el zigzag confirma un pivote
se comprueba qué figuras forman los últimos pivotes y quedan *armadas*; a
partir de ahí, el primer cierre que rompe la línea de una figura armada la
completa, y esa vela es la del patrón. Un pivote nuevo desarma lo anterior:
la figura ya no es la última del gráfico.

Lo que se decide en la vela ``t`` usa datos hasta ``t`` y nada más. Por eso
vale lo mismo calcular un día suelto que recalcular el histórico entero
tras un split, y por eso el resultado se puede evaluar después contra lo
que pasó.

Todo se calcula en escala ajustada (un dividendo no es un movimiento). Los
niveles que van a las columnas ``pat_*`` se devuelven a la escala del
precio real del día de la ruptura, que es contra la que se compara el
cierre en las reglas y lo que ve quien opera.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.patterns.model import (
    DEFAULT_PARAMS,
    DOUBLE_BOTTOM,
    DOUBLE_TOP,
    ENTRY_COLUMN,
    HEAD_SHOULDERS,
    INV_HEAD_SHOULDERS,
    PATTERN_COLUMNS,
    PATTERNS,
    STOP_COLUMN,
    TARGET_COLUMN,
    PatternMatch,
    PatternParams,
    PatternSpec,
    Pivot,
)
from analyzer.patterns.pivots import Array, ZigZag, average_true_range
from analyzer.patterns.reversal import (
    Detector,
    Point,
    Shape,
    double_bottom,
    inverse_head_shoulders,
)

ADJUSTED_COLUMNS = ("open", "high", "low", "close")
FACTOR = "factor"  # adj_close / close de cada vela

# Cada figura con su signo: +1 se busca tal cual, -1 con el precio invertido.
_DETECTORS: tuple[tuple[PatternSpec, int, Detector], ...] = (
    (DOUBLE_BOTTOM, 1, double_bottom),
    (DOUBLE_TOP, -1, double_bottom),
    (INV_HEAD_SHOULDERS, 1, inverse_head_shoulders),
    (HEAD_SHOULDERS, -1, inverse_head_shoulders),
)


class _Armed:
    """Figura a la espera de ruptura: la forma normalizada, su signo y sus pivotes reales."""

    def __init__(self, spec: PatternSpec, sign: int, shape: Shape, pivots: list[Pivot]) -> None:
        self.spec = spec
        self.sign = sign
        self.shape = shape
        self.pivots = tuple(pivots[-shape.pivots :])

    def match(self, t: int) -> PatternMatch:
        level = self.shape.level(t)
        start = self.shape.start
        return PatternMatch(
            name=self.spec.name,
            direction=self.spec.direction,
            bar=t,
            entry=self.sign * level,
            stop=self.sign * self.shape.stop,
            target=self.sign * (level + self.shape.height),
            pivots=self.pivots,
            neckline=((start, self.sign * self.shape.level(start)), (t, self.sign * level)),
        )


def adjusted_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    """Apertura, máximo, mínimo y cierre en escala ajustada, más el factor de cada vela."""
    factor = bars["adj_close"].astype("float64") / bars["close"].astype("float64")
    out = pd.DataFrame({c: bars[c].astype("float64") * factor for c in ADJUSTED_COLUMNS})
    out[FACTOR] = factor
    return out


def find_patterns(bars: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> list[PatternMatch]:
    """Patrones completados en las velas de un valor (ordenadas por fecha), en escala ajustada.

    ``bar`` y los índices de los pivotes son posiciones dentro de ``bars``.
    """
    ohlc = adjusted_ohlc(bars)
    return scan(
        ohlc["high"].to_numpy(dtype="float64"),
        ohlc["low"].to_numpy(dtype="float64"),
        ohlc["close"].to_numpy(dtype="float64"),
        params,
    )


def pattern_columns(bars: pd.DataFrame, params: PatternParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Columnas ``pat_*`` alineadas con ``bars``: 1 en la vela de la ruptura y sus niveles.

    Los niveles van en la escala del precio real de esa vela y son NaN en
    los días sin patrón.
    """
    ohlc = adjusted_ohlc(bars)
    factor = ohlc[FACTOR].to_numpy(dtype="float64")
    matches = scan(
        ohlc["high"].to_numpy(dtype="float64"),
        ohlc["low"].to_numpy(dtype="float64"),
        ohlc["close"].to_numpy(dtype="float64"),
        params,
    )
    size = len(bars)
    values: dict[str, Array] = {p.column: np.zeros(size) for p in PATTERNS}
    values |= {c: np.full(size, np.nan) for c in (ENTRY_COLUMN, STOP_COLUMN, TARGET_COLUMN)}
    taken: set[int] = set()
    for match in matches:
        if match.bar in taken:
            continue  # dos figuras el mismo día: se queda la primera del catálogo
        taken.add(match.bar)
        values[match.spec.column][match.bar] = 1.0
        values[ENTRY_COLUMN][match.bar] = match.entry / factor[match.bar]
        values[STOP_COLUMN][match.bar] = match.stop / factor[match.bar]
        values[TARGET_COLUMN][match.bar] = match.target / factor[match.bar]
    return pd.DataFrame(values, index=bars.index).loc[:, list(PATTERN_COLUMNS)]


def scan(high: Array, low: Array, close: Array, params: PatternParams) -> list[PatternMatch]:
    """Recorre la serie y devuelve los patrones en el orden en que se completan."""
    atr = average_true_range(high, low, close, params.atr_length)
    zigzag = ZigZag(high, low, atr, params.reversal_atr)
    armed: list[_Armed] = []
    matches: list[PatternMatch] = []
    for t in range(len(close)):
        if zigzag.step(t) is not None:
            armed = _arm(zigzag.pivots, close, atr, t, params)
        if not armed:
            continue
        price = close[t]
        waiting: list[_Armed] = []
        for figure in armed:
            level = figure.shape.level(t)
            normalized = figure.sign * price
            if not normalized > level:  # un cierre NaN tampoco rompe
                waiting.append(figure)
            elif (
                t - figure.pivots[0].index <= params.max_span
                and normalized < level + figure.shape.height  # objetivo sin consumir
            ):
                matches.append(figure.match(t))
        armed = waiting
    return matches


def _arm(
    pivots: list[Pivot], close: Array, atr: Array, t: int, params: PatternParams
) -> list[_Armed]:
    """Figuras que forman los últimos pivotes y que ningún cierre anterior a ``t`` ha roto ya."""
    last = pivots[-1]
    reference = float(atr[last.index])
    if not reference > 0:
        return []
    armed: list[_Armed] = []
    views = {sign: _normalized(pivots, sign) for sign in (1, -1)}
    since = np.arange(last.index + 1, t)
    for spec, sign, detector in _DETECTORS:
        shape = detector(views[sign], reference, params)
        if shape is None:
            continue
        levels = shape.anchor[1] + shape.slope * (since - shape.anchor[0])
        if bool(np.any(sign * close[since] > levels)):
            continue  # rompió antes de que el último pivote estuviera confirmado: tarde
        armed.append(_Armed(spec, sign, shape, pivots))
    return armed


def _normalized(pivots: list[Pivot], sign: int) -> list[Point]:
    """Los pivotes como los vería una figura alcista: con ``sign`` -1, el gráfico boca abajo."""
    recent = pivots[-6:]
    return [
        Point(index=p.index, value=sign * p.price, is_low=(p.kind == "L") == (sign > 0))
        for p in recent
    ]
