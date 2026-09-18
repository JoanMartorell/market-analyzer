"""Tabla ``<schema>.llm_calls``: una fila por llamada al modelo.

Guarda lo que costó y lo que respondió, no solo el resultado. Sirve para
tres cosas: vigilar el presupuesto mensual sin depender de la consola de
facturación, reutilizar la respuesta si el ciclo se repite con la misma
entrada, y poder explicar meses después por qué una señal decía lo que decía.

Una fila por región y día (``max_calls_per_cycle`` es 1): repetir el ciclo
sustituye la fila, no acumula. ``payload_hash`` es el sha256 de la entrada
que se mandó; si no ha cambiado, no hay que pagar otra llamada.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from analyzer.storage.table import Table

CALL_COLUMNS = (
    "region",
    "date",
    "model",
    "payload_hash",
    "batch",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_eur",
    "response",
    "created_at",
)


class LlmCallStore(Table):
    NAME = "llm_calls"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        region            VARCHAR   NOT NULL,
        date              DATE      NOT NULL,
        model             VARCHAR   NOT NULL,
        payload_hash      VARCHAR   NOT NULL,
        batch             BOOLEAN   NOT NULL,
        input_tokens      INTEGER   NOT NULL,
        output_tokens     INTEGER   NOT NULL,
        cache_read_tokens INTEGER   NOT NULL,
        cache_write_tokens INTEGER  NOT NULL,
        cost_eur          DOUBLE    NOT NULL,
        response          VARCHAR   NOT NULL,
        created_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (region, date)
    )
    """

    def upsert(self, row: dict[str, Any]) -> None:
        """Guarda (o sustituye) la llamada de una región y día."""
        missing = [c for c in CALL_COLUMNS if c not in row and c != "created_at"]
        if missing:
            raise ValueError(f"faltan campos de la llamada: {', '.join(missing)}")
        record = dict(row)
        record["created_at"] = datetime.now(UTC).replace(tzinfo=None)
        self._insert_or_replace(pd.DataFrame([record]), CALL_COLUMNS)

    def get(self, region: str, as_of: date) -> dict[str, Any] | None:
        """Llamada guardada de ``region`` en ``as_of``, o ``None`` si no la hay."""
        row = self._con.execute(
            f"SELECT {', '.join(CALL_COLUMNS)} FROM {self.table} "  # noqa: S608 - tabla validada
            "WHERE region = ? AND date = ?",
            [region, as_of],
        ).fetchone()
        return dict(zip(CALL_COLUMNS, row, strict=True)) if row is not None else None

    def month_cost(self, as_of: date) -> float:
        """Coste estimado acumulado en el mes natural de ``as_of``, todas las regiones."""
        total = self._con.execute(
            f"SELECT coalesce(sum(cost_eur), 0.0) FROM {self.table} "  # noqa: S608
            "WHERE year(date) = ? AND month(date) = ?",
            [as_of.year, as_of.month],
        ).fetchone()
        return float(total[0]) if total is not None else 0.0
