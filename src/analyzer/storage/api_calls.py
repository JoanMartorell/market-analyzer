"""Tabla ``<schema>.api_calls``: llamadas hechas a cada proveedor por día.

Los planes gratuitos cuentan llamadas por día natural, y pasarse no avisa:
devuelve errores el resto del día. Aquí la llamada se apunta antes de
hacerse, no después: si el proceso muere entre reservar y llamar, queda
contada igual, y el cupo nunca se rebasa por un fallo nuestro. El contador
lo comparten todas las regiones y comandos porque es una tabla, no una
variable del proceso.

El día es el de UTC, que es cuando EODHD reinicia su cuota.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from analyzer.storage.table import Table


class ApiCallStore(Table):
    NAME = "api_calls"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        provider   VARCHAR   NOT NULL,
        day        DATE      NOT NULL,
        calls      INTEGER   NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        PRIMARY KEY (provider, day)
    )
    """

    def used(self, provider: str, day: date) -> int:
        """Llamadas apuntadas a ``provider`` en ``day``."""
        row = self._con.execute(
            f"SELECT calls FROM {self.table} WHERE provider = ? AND day = ?",  # noqa: S608
            [provider, day],
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def reserve(self, provider: str, day: date, limit: int) -> bool:
        """Apunta una llamada más si ``day`` no ha llegado a ``limit``; ``True`` si cabe."""
        with self.transaction():
            used = self.used(provider, day)
            if used >= limit:
                return False
            self._con.execute(
                f"INSERT OR REPLACE INTO {self.table} VALUES (?, ?, ?, ?)",  # noqa: S608
                [provider, day, used + 1, datetime.now(UTC).replace(tzinfo=None)],
            )
        return True
