"""Universo de valores: quién formaba parte del índice cada día.

Fichero de constituyentes point-in-time por región, una fila por intervalo
de pertenencia (ticker, start, end). Preguntar "¿quién estaba el día D?" es
filtrar por ``start <= D < end``. Sin esto el backtest solo ve a los
supervivientes y se infla un 1-2% anual.

Cada índice vive en ``sources/<id>/`` con su fuente y enriquecedores; el
registro los expone por id y ``build`` los ejecuta de forma genérica.
"""

from analyzer.universe.base import Enricher, FetchContext, Notes, Source, UniverseSpec
from analyzer.universe.build import BuildReport, build_universe
from analyzer.universe.constituents import (
    COLUMNS,
    load_constituents,
    members_as_of,
    members_between,
    write_constituents,
)
from analyzer.universe.registry import UNIVERSES, get_universe

__all__ = [
    "COLUMNS",
    "UNIVERSES",
    "BuildReport",
    "Enricher",
    "FetchContext",
    "Notes",
    "Source",
    "UniverseSpec",
    "build_universe",
    "get_universe",
    "load_constituents",
    "members_as_of",
    "members_between",
    "write_constituents",
]
