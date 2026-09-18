"""Tabla ``<schema>.signals``: lo que el sistema dijo cada día, con su procedencia.

Una señal es un candidato del screener tal como salió del ciclo: la regla
que lo marcó (con el hash de su YAML), el panel del que salió (con su hash)
y, si el modelo lo analizó, su veredicto. Con eso una señal se puede volver
a explicar meses después aunque las reglas hayan cambiado y el histórico se
haya reescrito.

Se guardan todos los candidatos, también los que el modelo descarta: sin
ellos no se puede medir si el modelo aporta algo. Repetir el ciclo sustituye
las señales del día de esa región, no las acumula.

Las columnas de conciliación las rellena el paso 12 al día siguiente, con
la apertura real de la sesión de ejecución: son lo único que se actualiza
en una señal ya escrita, y sustituir el día las deja otra vez vacías.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import ClassVar

import numpy as np
import pandas as pd

from analyzer.storage.table import Table

SIGNAL_KEY = ("ticker", "mic", "date", "rule_id")
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
RECONCILE_COLUMNS = ("execution_open", "open_gap", "reconcile_status", "reconciled_at")
LOADED_SIGNAL_COLUMNS = SIGNAL_COLUMNS + RECONCILE_COLUMNS
TEXT_LIST_COLUMNS = ("risks",)  # DuckDB las devuelve como arrays; aquí, listas
DATE_COLUMNS = ("date", "execute_on")


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
    COLUMNS_ADDED: ClassVar[dict[str, str]] = {
        "execution_open": "DOUBLE",  # apertura real de la sesión execute_on
        "open_gap": "DOUBLE",  # execution_open / close - 1
        "reconcile_status": "VARCHAR",  # conciliada | desviada | sin_precio
        "reconciled_at": "TIMESTAMP",
    }

    # --- escritura ---------------------------------------------------------

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

    def reconcile(self, reconciled: pd.DataFrame) -> int:
        """Escribe apertura real, gap y estado en las señales de ``reconciled``, por clave.

        Devuelve las señales actualizadas. Una clave que no exista no se
        crea: la conciliación solo completa señales que ya estaban.
        """
        needed = (*SIGNAL_KEY, "execution_open", "open_gap", "reconcile_status")
        missing = [c for c in needed if c not in reconciled.columns]
        if missing:
            raise ValueError(f"faltan columnas de la conciliación: {', '.join(missing)}")
        if reconciled.empty:
            return 0
        incoming = reconciled.loc[:, list(needed)].copy()
        incoming["ticker"] = incoming["ticker"].astype(str)
        incoming["mic"] = incoming["mic"].astype(str)
        incoming["rule_id"] = incoming["rule_id"].astype(str)
        incoming["date"] = pd.to_datetime(incoming["date"]).dt.date
        incoming["execution_open"] = incoming["execution_open"].astype("float64")
        incoming["open_gap"] = incoming["open_gap"].astype("float64")
        incoming["reconciled_at"] = datetime.now(UTC).replace(tzinfo=None)
        view = f"reconciled_{self.NAME}"
        self._con.register(view, incoming)
        try:
            updated = self._con.execute(
                f"UPDATE {self.table} AS s SET "  # noqa: S608 - tabla validada
                "execution_open = r.execution_open, open_gap = r.open_gap, "
                "reconcile_status = r.reconcile_status, reconciled_at = r.reconciled_at "
                f"FROM {view} AS r WHERE s.ticker = r.ticker AND s.mic = r.mic "
                "AND s.date = r.date AND s.rule_id = r.rule_id"
            ).fetchone()
        finally:
            self._con.unregister(view)
        return int(updated[0]) if updated else 0

    # --- lectura -----------------------------------------------------------

    def load(self, region: str, as_of: date) -> pd.DataFrame:
        """Señales de ``region`` en ``as_of``, de mayor a menor score."""
        return self._frame(
            self._con.execute(
                f"SELECT {', '.join(LOADED_SIGNAL_COLUMNS)} FROM {self.table} "  # noqa: S608
                "WHERE region = ? AND date = ? ORDER BY score DESC, ticker, mic, rule_id",
                [region, as_of],
            ).df()
        )

    def pending_reconciliation(self, region: str, as_of: date) -> pd.DataFrame:
        """Señales de ``region`` que se ejecutaban en ``as_of`` o antes y hay que conciliar.

        Las de ``as_of`` entran siempre, conciliadas o no: repetir el día las
        recalcula. Las anteriores solo si siguen sin estado, que es lo que
        deja un día en que el ciclo no llegó al final.
        """
        return self._frame(
            self._con.execute(
                f"SELECT {', '.join(LOADED_SIGNAL_COLUMNS)} FROM {self.table} "  # noqa: S608
                "WHERE region = ? AND execute_on <= ? "
                "AND (reconcile_status IS NULL OR execute_on = ?) "
                "ORDER BY execute_on, score DESC, ticker, mic, rule_id",
                [region, as_of, as_of],
            ).df()
        )

    @staticmethod
    def _frame(frame: pd.DataFrame) -> pd.DataFrame:
        for column in DATE_COLUMNS:
            frame[column] = pd.to_datetime(frame[column])
        for column in TEXT_LIST_COLUMNS:
            # Un NULL llega como pd.NA, no como None; una lista, como array.
            frame[column] = pd.Series(
                [list(v) if isinstance(v, np.ndarray | list) else None for v in frame[column]],
                index=frame.index,
                dtype=object,
            )
        return frame
