"""Vocabulario de los patrones chartistas: catálogo, parámetros, pivotes y hallazgos.

Un patrón es una figura sobre los últimos pivotes del precio (máximos y
mínimos relevantes) que se da por completada el día en que el cierre rompe
su línea de referencia. De la figura salen tres niveles, los mismos que se
dibujan en cualquier manual: el nivel roto (entrada), el stop al otro lado
de la figura y el objetivo, que proyecta la altura de la figura desde la
ruptura.

Los parámetros viven aquí y no en YAML por la misma razón que los periodos
de los indicadores: las columnas ``pat_*`` se guardan día a día, y un
umbral que cambiara sin pasar por el código dejaría un histórico calculado
con dos criterios distintos sin rastro de cuándo cambió.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Direction = Literal["long", "short"]
PivotKind = Literal["H", "L"]


@dataclass(frozen=True)
class PatternSpec:
    name: str
    direction: Direction
    label: str  # como se nombra en mensajes y gráficos

    @property
    def column(self) -> str:
        return f"pat_{self.name}"


DOUBLE_BOTTOM = PatternSpec("double_bottom", "long", "doble suelo")
DOUBLE_TOP = PatternSpec("double_top", "short", "doble techo")
INV_HEAD_SHOULDERS = PatternSpec("inv_head_shoulders", "long", "hombro-cabeza-hombro invertido")
HEAD_SHOULDERS = PatternSpec("head_shoulders", "short", "hombro-cabeza-hombro")

ASCENDING_TRIANGLE = PatternSpec("ascending_triangle", "long", "triángulo ascendente")
DESCENDING_TRIANGLE = PatternSpec("descending_triangle", "short", "triángulo descendente")
BULL_FLAG = PatternSpec("bull_flag", "long", "bandera alcista")
BEAR_FLAG = PatternSpec("bear_flag", "short", "bandera bajista")

# El orden decide qué figura se queda el día si dos rompen a la vez: primero las de giro.
PATTERNS: tuple[PatternSpec, ...] = (
    DOUBLE_BOTTOM,
    DOUBLE_TOP,
    INV_HEAD_SHOULDERS,
    HEAD_SHOULDERS,
    ASCENDING_TRIANGLE,
    DESCENDING_TRIANGLE,
    BULL_FLAG,
    BEAR_FLAG,
)
PATTERNS_BY_NAME: dict[str, PatternSpec] = {p.name: p for p in PATTERNS}

ENTRY_COLUMN, STOP_COLUMN, TARGET_COLUMN = "pat_entry", "pat_stop", "pat_target"
LEVEL_COLUMNS: tuple[str, ...] = (ENTRY_COLUMN, STOP_COLUMN, TARGET_COLUMN)
PATTERN_COLUMNS: tuple[str, ...] = (*(p.column for p in PATTERNS), *LEVEL_COLUMNS)


@dataclass(frozen=True)
class PatternParams:
    """Umbrales de la detección. Las distancias de precio van en ATR para no depender del valor."""

    atr_length: int = 14
    reversal_atr: float = 3.0  # giro que convierte un extremo en pivote
    equal_tolerance_atr: float = 1.0  # diferencia máxima entre dos suelos (o techos) "iguales"
    head_margin_atr: float = 1.0  # lo que la cabeza debe superar a cada hombro
    shoulder_tolerance_atr: float = 2.0  # diferencia máxima entre los dos hombros
    min_height_atr: float = 3.0  # altura mínima de la figura
    min_separation: int = 8  # sesiones mínimas entre los dos suelos (el doble entre hombros)
    max_span: int = 120  # sesiones máximas del primer pivote a la ruptura
    prior_move: float = 0.5  # tramo previo: cuánto supera a la línea de ruptura, en alturas
    max_neckline_tilt: float = 0.5  # desnivel máximo de la línea clavicular, en alturas
    max_shoulder_time_ratio: float = 3.0  # asimetría temporal máxima entre los dos lados
    stop_buffer_atr: float = 0.25  # holgura del stop más allá de la figura
    # Banderas: se miran en las velas, no en los pivotes.
    pole_min_atr: float = 6.0  # subida mínima del mástil
    pole_max_bars: int = 15  # sesiones máximas en las que el mástil hace esa subida
    flag_min_bars: int = 4  # sesiones mínimas de pausa entre el máximo del mástil y la ruptura
    flag_max_bars: int = 20
    flag_max_retrace: float = 0.5  # parte del mástil que la pausa puede devolver


DEFAULT_PARAMS = PatternParams()
# Sesiones previas que necesita una fila para no depender de dónde empieza la
# ventana: la figura más larga y margen para que el zigzag se sincronice.
PATTERN_WARMUP = DEFAULT_PARAMS.max_span + 30


@dataclass(frozen=True)
class Pivot:
    index: int  # posición de la vela dentro de la serie
    price: float
    kind: PivotKind
    confirmed_at: int  # vela en la que el giro lo confirmó; antes no se conocía


@dataclass(frozen=True)
class PatternMatch:
    """Un patrón completado: la vela de la ruptura, sus niveles y la figura para dibujarla."""

    name: str
    direction: Direction
    bar: int  # vela de la ruptura
    entry: float  # nivel roto
    stop: float
    target: float
    pivots: tuple[Pivot, ...]
    neckline: tuple[tuple[int, float], tuple[int, float]]  # dos puntos de la línea rota

    @property
    def spec(self) -> PatternSpec:
        return PATTERNS_BY_NAME[self.name]

    def scaled(self, factor: float) -> PatternMatch:
        """El mismo hallazgo con los precios multiplicados por ``factor``."""
        (i1, v1), (i2, v2) = self.neckline
        return replace(
            self,
            entry=self.entry * factor,
            stop=self.stop * factor,
            target=self.target * factor,
            pivots=tuple(replace(p, price=p.price * factor) for p in self.pivots),
            neckline=((i1, v1 * factor), (i2, v2 * factor)),
        )
