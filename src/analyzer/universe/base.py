"""Contratos del universo.

Un universo se define con una ``UniverseSpec``: una fuente que devuelve los
intervalos de pertenencia y cero o más enriquecedores que añaden columnas.
Añadir un índice nuevo es crear una carpeta en ``sources/`` con su fuente,
sus enriquecedores y su spec, y registrarla en ``registry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd

Notes = dict[str, Any]

SOURCE_COLUMNS: tuple[str, ...] = ("ticker", "start", "end")  # mínimo que debe devolver una fuente


@dataclass(frozen=True)
class FetchContext:
    today: date
    user_agent: str | None = None  # contacto identificable; la SEC lo exige
    universe_dir: Path | None = None  # carpeta del universo, por si un enriquecedor guarda memoria


class Source(Protocol):
    """Devuelve DataFrame[ticker, start, end] y, si los conoce, mic y currency."""

    @property
    def name(self) -> str: ...

    def fetch(self, client: httpx.Client, ctx: FetchContext) -> pd.DataFrame: ...


class Enricher(Protocol):
    """Recibe la tabla, añade o rellena columnas y devuelve notas para el informe."""

    @property
    def name(self) -> str: ...

    def enrich(
        self, table: pd.DataFrame, client: httpx.Client, ctx: FetchContext
    ) -> tuple[pd.DataFrame, Notes]: ...


@dataclass(frozen=True)
class UniverseSpec:
    id: str
    currency: str  # divisa por defecto si la fuente no la indica por fila
    source: Source
    enrichers: tuple[Enricher, ...] = ()
    # True: la fuente ya trae start/end históricos.
    # False: la fuente solo da la composición actual; cada build se compara con
    # el fichero anterior y cierra o abre intervalos. El histórico se acumula
    # desde el primer build.
    history: bool = True
