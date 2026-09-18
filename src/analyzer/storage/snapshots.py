"""Tablas ``<schema>.snapshots`` y ``<schema>.snapshot_runs``: el panel point-in-time.

``snapshots`` guarda una fila por valor y día con lo que vio el screener: los
atributos del universo (sector, divisa...) y los indicadores del día. Las
tablas de precios e indicadores se reescriben (splits, dividendos que
reajustan el histórico, cuarentenas); esta no. ``snapshot_runs`` es la
cabecera de cada panel: región, día, hash y cobertura. Con el hash, una
señal apunta a los datos exactos que la generaron.

Como en ``indicators``, las columnas del catálogo se migran al instanciar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

import duckdb
import pandas as pd

from analyzer.storage.sql import check_identifier
from analyzer.storage.table import Table

KEY_COLUMNS = ("ticker", "mic", "date")
UNIVERSE_COLUMNS = ("name", "sector", "sub_industry", "currency")  # atributos del universo
BASE_COLUMNS = ("close", "adj_close", "volume")
RESERVED = ("region", *KEY_COLUMNS, *UNIVERSE_COLUMNS, *BASE_COLUMNS, "created_at")


class SnapshotStore(Table):
    NAME = "snapshots"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        region       VARCHAR   NOT NULL,
        ticker       VARCHAR   NOT NULL,
        mic          VARCHAR   NOT NULL,
        date         DATE      NOT NULL,
        name         VARCHAR,
        sector       VARCHAR,
        sub_industry VARCHAR,
        currency     VARCHAR,
        close        DOUBLE    NOT NULL,
        adj_close    DOUBLE,
        volume       BIGINT,
        created_at   TIMESTAMP NOT NULL,
        PRIMARY KEY (ticker, mic, date)
    )
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, schema: str, columns: Sequence[str]) -> None:
        self.indicator_columns = tuple(check_identifier(c) for c in columns)
        clash = set(self.indicator_columns) & set(RESERVED)
        if clash:
            raise ValueError(f"indicadores con nombre reservado: {', '.join(sorted(clash))}")
        super().__init__(con, schema)

    @property
    def panel_columns(self) -> tuple[str, ...]:
        """Columnas del panel tal como lo ve el screener (sin región ni marca de tiempo)."""
        return KEY_COLUMNS + UNIVERSE_COLUMNS + BASE_COLUMNS + self.indicator_columns

    @property
    def columns(self) -> tuple[str, ...]:
        return ("region", *self.panel_columns, "created_at")

    def _migrations(self) -> Mapping[str, str]:
        return dict.fromkeys(self.indicator_columns, "DOUBLE")

    def replace_day(self, region: str, as_of: date, panel: pd.DataFrame) -> int:
        """Sustituye el panel de ``region`` en ``as_of`` por ``panel``.

        Borra e inserta; quien llama decide la transacción (ver ``Table.transaction``).
        """
        missing = [c for c in self.panel_columns if c not in panel.columns]
        if missing:
            raise ValueError(f"faltan columnas del panel: {', '.join(missing)}")
        incoming = panel.loc[:, list(self.panel_columns)].copy()
        incoming["region"] = region
        incoming["date"] = pd.to_datetime(incoming["date"]).dt.date
        for column in ("close", "adj_close", *self.indicator_columns):
            incoming[column] = incoming[column].astype("float64")
        incoming["volume"] = incoming["volume"].round().astype("Int64")
        incoming["created_at"] = datetime.now(UTC).replace(tzinfo=None)
        self._con.execute(
            f"DELETE FROM {self.table} WHERE region = ? AND date = ?",  # noqa: S608
            [region, as_of],
        )
        return self._insert_or_replace(incoming, self.columns)

    def load(self, region: str, as_of: date) -> pd.DataFrame:
        """Panel de ``region`` en ``as_of`` con las columnas del panel, ordenado por clave."""
        frame = self._con.execute(
            f"SELECT {', '.join(self.panel_columns)} FROM {self.table} "  # noqa: S608
            "WHERE region = ? AND date = ? ORDER BY ticker, mic",
            [region, as_of],
        ).df()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame


class SnapshotRunStore(Table):
    NAME = "snapshot_runs"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        region        VARCHAR   NOT NULL,
        date          DATE      NOT NULL,
        snapshot_hash VARCHAR   NOT NULL,
        rows          INTEGER   NOT NULL,
        expected      INTEGER   NOT NULL,
        coverage      DOUBLE    NOT NULL,
        created_at    TIMESTAMP NOT NULL,
        PRIMARY KEY (region, date)
    )
    """
    COLUMNS = ("region", "date", "snapshot_hash", "rows", "expected", "coverage", "created_at")

    def upsert(
        self, region: str, as_of: date, *, snapshot_hash: str, rows: int, expected: int
    ) -> None:
        coverage = rows / expected if expected else 0.0
        frame = pd.DataFrame(
            [
                {
                    "region": region,
                    "date": as_of,
                    "snapshot_hash": snapshot_hash,
                    "rows": rows,
                    "expected": expected,
                    "coverage": coverage,
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                }
            ]
        )
        self._insert_or_replace(frame, self.COLUMNS)

    def get(self, region: str, as_of: date) -> dict[str, Any] | None:
        """Cabecera del panel de ``region`` en ``as_of`` o ``None`` si no se ha generado."""
        row = self._con.execute(
            f"SELECT {', '.join(self.COLUMNS)} FROM {self.table} "  # noqa: S608
            "WHERE region = ? AND date = ?",
            [region, as_of],
        ).fetchone()
        if row is None:
            return None
        return dict(zip(self.COLUMNS, row, strict=True))
