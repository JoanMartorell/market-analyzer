"""Tabla ``<schema>.signals``: lo que el sistema dijo cada día, con su procedencia.

Una señal es un candidato del screener tal como salió del ciclo: la regla
que lo marcó (con el hash de su YAML), el panel del que salió (con su hash)
y, si el modelo lo analizó, su veredicto. Con eso una señal se puede volver
a explicar meses después aunque las reglas hayan cambiado y el histórico se
haya reescrito.

Se guardan todos los candidatos, también los que el modelo descarta: sin
ellos no se puede medir si el modelo aporta algo. Repetir el ciclo sustituye
las señales del día de esa región, no las acumula.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from analyzer.storage.table import Table

SIGNAL_COLUMNS = (
    "region",
    "ticker",
    "mic",
    "date",
    "rule_id",
    "rule_hash",
    "snapshot_hash",
    "direction",
    "score",
    "conditions_met",
    "close",
    "currency",
    "execute_on",
    "verdict",
    "confidence",
    "rationale",
    "risks",
    "llm_model",
    "created_at",
)
TEXT_LIST_COLUMNS = ("risks",)  # DuckDB las devuelve como arrays; aquí, listas


class SignalStore(Table):
    NAME = "signals"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        region         VARCHAR   NOT NULL,
        ticker         VARCHAR   NOT NULL,
        mic            VARCHAR   NOT NULL,
        date           DATE      NOT NULL,
        rule_id        VARCHAR   NOT NULL,
        rule_hash      VARCHAR   NOT NULL,
        snapshot_hash  VARCHAR   NOT NULL,
        direction      VARCHAR   NOT NULL,
        score          DOUBLE    NOT NULL,
        conditions_met VARCHAR   NOT NULL,
        close          DOUBLE    NOT NULL,
        currency       VARCHAR,
        execute_on     DATE      NOT NULL,
        verdict        VARCHAR,
        confidence     DOUBLE,
        rationale      VARCHAR,
        risks          VARCHAR[],
        llm_model      VARCHAR,
        created_at     TIMESTAMP NOT NULL,
        PRIMARY KEY (ticker, mic, date, rule_id)
    )
    """

    def replace_day(self, region: str, as_of: date, signals: pd.DataFrame) -> tuple[int, int]:
        """Sustituye las señales de ``region`` en ``as_of``: devuelve (borradas, guardadas).

        Borra e inserta dentro de una transacción: un ciclo repetido con
        menos candidatos no puede dejar señales huérfanas del anterior.
        """
        missing = [c for c in SIGNAL_COLUMNS if c not in signals.columns and c != "created_at"]
        if missing:
            raise ValueError(f"faltan columnas de las señales: {', '.join(missing)}")
        incoming = signals.copy()
        incoming["created_at"] = datetime.now(UTC).replace(tzinfo=None)
        with self.transaction():
            deleted = self._con.execute(
                f"DELETE FROM {self.table} WHERE region = ? AND date = ?",  # noqa: S608
                [region, as_of],
            ).fetchone()
            inserted = self._insert_or_replace(incoming, SIGNAL_COLUMNS)
        return (int(deleted[0]) if deleted else 0, inserted)

    def load(self, region: str, as_of: date) -> pd.DataFrame:
        """Señales de ``region`` en ``as_of``, de mayor a menor score."""
        frame = self._con.execute(
            f"SELECT {', '.join(SIGNAL_COLUMNS)} FROM {self.table} "  # noqa: S608
            "WHERE region = ? AND date = ? ORDER BY score DESC, ticker, mic, rule_id",
            [region, as_of],
        ).df()
        for column in ("date", "execute_on"):
            frame[column] = pd.to_datetime(frame[column])
        for column in TEXT_LIST_COLUMNS:
            # Un NULL llega como pd.NA, no como None; una lista, como array.
            frame[column] = pd.Series(
                [list(v) if isinstance(v, np.ndarray | list) else None for v in frame[column]],
                index=frame.index,
                dtype=object,
            )
        return frame
