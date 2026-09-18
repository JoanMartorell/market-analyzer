"""Base de las tablas DuckDB: nombre validado, DDL, migraciones y filtro por clave."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

import duckdb
import pandas as pd

from analyzer.storage.sql import check_identifier


class Table:
    """Tabla ``<schema>.<NAME>`` creada al instanciar; ``COLUMNS_ADDED`` son migraciones."""

    NAME: ClassVar[str]
    DDL: ClassVar[str]  # con ``{table}`` como marcador
    COLUMNS_ADDED: ClassVar[Mapping[str, str]] = {}  # columna -> tipo, añadidas tras el DDL inicial

    def __init__(self, con: duckdb.DuckDBPyConnection, schema: str) -> None:
        self._con = con
        self.schema = check_identifier(schema)
        self.table = f'"{self.schema}".{check_identifier(self.NAME)}'
        con.execute(self.DDL.format(table=self.table))
        for column, kind in self.COLUMNS_ADDED.items():
            name = check_identifier(column)
            con.execute(f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS {name} {kind}")

    def _insert_or_replace(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
        """Upsert de ``frame`` por la clave primaria de la tabla."""
        if frame.empty:
            return 0
        view = f"incoming_{self.NAME}"
        self._con.register(view, frame.loc[:, list(columns)])
        try:
            self._con.execute(
                f"INSERT OR REPLACE INTO {self.table} BY NAME "  # noqa: S608 - tabla validada
                f"SELECT {', '.join(columns)} FROM {view}"
            )
        finally:
            self._con.unregister(view)
        return len(frame)

    def _select(
        self,
        columns: tuple[str, ...],
        *,
        start: Any | None = None,
        end: Any | None = None,
        keys: pd.DataFrame | None = None,
        order: str = "ticker, mic, date",
    ) -> pd.DataFrame:
        """``SELECT columns`` con filtros opcionales: rango de ``date`` y claves (ticker, mic)."""
        clauses: list[str] = []
        params: list[Any] = []
        if start is not None:
            clauses.append("date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("date <= ?")
            params.append(end)
        with self._key_filter(keys) as key_clause:
            if key_clause:
                clauses.append(key_clause)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            frame = self._con.execute(
                f"SELECT {', '.join(columns)} FROM {self.table}{where} ORDER BY {order}",  # noqa: S608
                params,
            ).df()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    @contextmanager
    def _key_filter(self, keys: pd.DataFrame | None) -> Iterator[str]:
        """Registra ``keys`` como vista temporal y devuelve la condición SQL que la usa."""
        if keys is None:
            yield ""
            return
        wanted = keys.loc[:, ["ticker", "mic"]].astype(str).drop_duplicates()
        view = f"wanted_keys_{self.NAME}"
        self._con.register(view, wanted)
        try:
            yield f"(ticker, mic) IN (SELECT (ticker, mic) FROM {view})"  # noqa: S608
        finally:
            self._con.unregister(view)
