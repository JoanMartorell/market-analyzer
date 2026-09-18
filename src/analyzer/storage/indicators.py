"""Tabla ``<schema>.indicators``: una fila por (ticker, mic, date) con los indicadores del día.

Es una tabla ancha: las columnas base (cierre, cierre ajustado, volumen) van
en el DDL y cada indicador del catálogo es una columna DOUBLE que se añade
como migración al instanciar. Añadir un indicador al catálogo añade su
columna sin tocar nada aquí. El catálogo vive en el paso ``indicators``; el
almacén solo recibe la lista de columnas para no depender de él.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd

from analyzer.storage.sql import check_identifier
from analyzer.storage.table import Table

KEY_COLUMNS = ("ticker", "mic", "date")
BASE_COLUMNS = ("close", "adj_close", "volume")  # las reglas los usan como variables


class IndicatorStore(Table):
    NAME = "indicators"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        ticker      VARCHAR   NOT NULL,
        mic         VARCHAR   NOT NULL,
        date        DATE      NOT NULL,
        close       DOUBLE    NOT NULL,
        adj_close   DOUBLE,
        volume      BIGINT,
        computed_at TIMESTAMP NOT NULL,
        PRIMARY KEY (ticker, mic, date)
    )
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, schema: str, columns: Sequence[str]) -> None:
        self.indicator_columns = tuple(check_identifier(c) for c in columns)
        clash = set(self.indicator_columns) & set(KEY_COLUMNS + BASE_COLUMNS + ("computed_at",))
        if clash:
            raise ValueError(f"indicadores con nombre reservado: {', '.join(sorted(clash))}")
        super().__init__(con, schema)

    @property
    def columns(self) -> tuple[str, ...]:
        return KEY_COLUMNS + BASE_COLUMNS + self.indicator_columns + ("computed_at",)

    def _migrations(self) -> Mapping[str, str]:
        return dict.fromkeys(self.indicator_columns, "DOUBLE")

    def upsert(self, frame: pd.DataFrame, computed_at: datetime | None = None) -> int:
        """Inserta o reemplaza filas. Las columnas del catálogo deben venir todas."""
        wanted = KEY_COLUMNS + BASE_COLUMNS + self.indicator_columns
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise ValueError(f"faltan columnas de indicadores: {', '.join(missing)}")
        if frame.empty:
            return 0
        incoming = frame.loc[:, list(wanted)].copy()
        incoming["date"] = pd.to_datetime(incoming["date"]).dt.date
        for column in ("close", "adj_close", *self.indicator_columns):
            incoming[column] = incoming[column].astype("float64")
        incoming["volume"] = incoming["volume"].round().astype("Int64")
        stamp = computed_at if computed_at is not None else datetime.now(UTC)
        incoming["computed_at"] = stamp.replace(tzinfo=None)
        return self._insert_or_replace(incoming, self.columns)

    def load(
        self,
        start: Any | None = None,
        end: Any | None = None,
        keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Filas entre ``start`` y ``end`` (inclusive), ordenadas por clave y fecha."""
        return self._select(self.columns, start=start, end=end, keys=keys)
