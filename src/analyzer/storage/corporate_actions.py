"""Tabla ``<schema>.corporate_actions``: un evento por (ticker, mic, date, kind).

``kind`` es ``split`` (``value`` = acciones nuevas por antigua; 0.5 es un
contrasplit 1:2) o ``dividend`` (``value`` = importe por acción en la divisa
de cotización). ``source`` dice quién lo detectó: el feed del proveedor o la
heurística de ratio.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from analyzer.storage.table import Table

ACTION_COLUMNS = ("ticker", "mic", "date", "kind", "value", "source")
LOADED_ACTION_COLUMNS = (*ACTION_COLUMNS, "detected_at")


class CorporateActionStore(Table):
    NAME = "corporate_actions"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        ticker      VARCHAR   NOT NULL,
        mic         VARCHAR   NOT NULL,
        date        DATE      NOT NULL,
        kind        VARCHAR   NOT NULL,
        value       DOUBLE    NOT NULL,
        source      VARCHAR   NOT NULL,
        detected_at TIMESTAMP NOT NULL,
        PRIMARY KEY (ticker, mic, date, kind)
    )
    """

    def upsert(self, frame: pd.DataFrame) -> int:
        missing = [c for c in ACTION_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"faltan columnas de eventos: {', '.join(missing)}")
        if frame.empty:
            return 0
        incoming = frame.loc[:, list(ACTION_COLUMNS)].copy()
        incoming["date"] = pd.to_datetime(incoming["date"]).dt.date
        incoming["value"] = incoming["value"].astype("float64")
        incoming["detected_at"] = datetime.now(UTC).replace(tzinfo=None)
        return self._insert_or_replace(incoming, LOADED_ACTION_COLUMNS)

    def load(
        self,
        start: Any | None = None,
        end: Any | None = None,
        keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self._select(LOADED_ACTION_COLUMNS, start=start, end=end, keys=keys)
