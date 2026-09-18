"""Tabla ``<schema>.prices``: una vela diaria por (ticker, mic, date).

Se guardan el cierre real y el ajustado por dividendos y splits. Los
indicadores usan ``adj_close``; comparar ambos permite detectar operaciones
corporativas sin depender de otro feed. ``dividend`` y ``split`` son
opcionales: las rellena el proveedor que las conozca (yfinance las trae en
la misma descarga) y quedan NULL en los demás.

La escritura es un upsert por clave: relanzar el mismo día no duplica filas.
La clave no admite nulos, así que un valor sin bolsa conocida se guarda con
``mic = UNKNOWN_MIC`` en vez de NULL. ``ingested_at`` dice cuándo llegó cada
vela: es lo que permite saber si el histórico de un valor es anterior a un
evento corporativo y hay que volver a bajarlo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import pandas as pd

from analyzer.storage.table import Table

PRICE_COLUMNS = (
    "ticker",
    "mic",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
)
OPTIONAL_PRICE_COLUMNS = ("dividend", "split")  # split = acciones nuevas por antigua (10 = 10:1)
LOADED_COLUMNS = PRICE_COLUMNS + OPTIONAL_PRICE_COLUMNS + ("ingested_at",)
UNKNOWN_MIC = "UNKNOWN"


class PriceStore(Table):
    NAME = "prices"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        ticker      VARCHAR   NOT NULL,
        mic         VARCHAR   NOT NULL,
        date        DATE      NOT NULL,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE    NOT NULL,
        adj_close   DOUBLE,
        volume      BIGINT,
        source      VARCHAR   NOT NULL,
        ingested_at TIMESTAMP NOT NULL,
        PRIMARY KEY (ticker, mic, date)
    )
    """
    COLUMNS_ADDED: ClassVar[dict[str, str]] = {"dividend": "DOUBLE", "split": "DOUBLE"}

    # --- escritura ---------------------------------------------------------

    def upsert(self, frame: pd.DataFrame, ingested_at: datetime | None = None) -> int:
        """Inserta o reemplaza velas. Devuelve filas escritas (sin cierre no se guarda nada).

        ``ingested_at`` solo se fija a mano para reproducir cargas antiguas (tests).
        """
        incoming = normalize_prices(frame)
        if incoming.empty:
            return 0
        stamp = ingested_at if ingested_at is not None else datetime.now(UTC)
        incoming["ingested_at"] = stamp.replace(tzinfo=None)
        incoming["date"] = incoming["date"].dt.date
        return self._insert_or_replace(incoming, LOADED_COLUMNS)

    # --- lectura -----------------------------------------------------------

    def last_dates(self, keys: pd.DataFrame | None = None) -> pd.DataFrame:
        """Última fecha guardada por clave: columnas ticker, mic, last_date (``date``).

        Con ``keys`` (ticker, mic) se limita a esas claves.
        """
        with self._key_filter(keys) as key_clause:
            where = f" WHERE {key_clause}" if key_clause else ""
            frame = self._con.execute(
                f"SELECT ticker, mic, max(date) AS last_date FROM {self.table}{where} "  # noqa: S608
                "GROUP BY ticker, mic ORDER BY ticker, mic"
            ).df()
        frame["last_date"] = pd.to_datetime(frame["last_date"]).dt.date
        return frame

    def load(
        self,
        start: Any | None = None,
        end: Any | None = None,
        keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Velas entre ``start`` y ``end`` (ambos inclusive), ordenadas por clave y fecha.

        Con ``keys`` (ticker, mic) se limita a esas claves.
        """
        return self._select(LOADED_COLUMNS, start=start, end=end, keys=keys)

    def stats(self, keys: pd.DataFrame | None = None) -> dict[str, Any]:
        """Filas, claves distintas y rango de fechas, opcionalmente solo para ``keys``."""
        with self._key_filter(keys) as key_clause:
            where = f" WHERE {key_clause}" if key_clause else ""
            row = self._con.execute(
                f"SELECT count(*), count(DISTINCT (ticker, mic)), min(date), max(date) "  # noqa: S608
                f"FROM {self.table}{where}"
            ).fetchone()
        assert row is not None
        rows, n_keys, first, last = row
        return {"rows": rows, "keys": n_keys, "first_date": first, "last_date": last}


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Deja el DataFrame con las columnas y tipos de la tabla; descarta velas sin cierre."""
    missing = [c for c in PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"faltan columnas de precios: {', '.join(missing)}")
    out = frame.copy()
    for column in OPTIONAL_PRICE_COLUMNS:
        if column not in out.columns:
            out[column] = float("nan")
    out = out.loc[:, list(PRICE_COLUMNS + OPTIONAL_PRICE_COLUMNS)].dropna(subset=["close"])
    if out.empty:
        return out
    out["ticker"] = out["ticker"].astype(str)
    out["mic"] = out["mic"].fillna(UNKNOWN_MIC).astype(str)
    dates = pd.to_datetime(out["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates.dt.normalize()
    for column in ("open", "high", "low", "close", "adj_close", *OPTIONAL_PRICE_COLUMNS):
        out[column] = out[column].astype("float64")
    out["volume"] = out["volume"].round().astype("Int64")
    out["source"] = out["source"].astype(str)
    return out.drop_duplicates(["ticker", "mic", "date"], keep="last").reset_index(drop=True)
