"""Tablas ``<schema>.news_clusters`` y ``<schema>.news_cluster_members``: temas del día.

Un *tema* es un grupo de artículos que cuentan lo mismo. La tabla de temas
guarda el representante (el artículo que mejor lo cuenta), de dónde salió y
su sentimiento; la de miembros, qué artículos entraron y por qué nivel de la
cascada. Separadas porque una señal cita el tema, no sus quince copias, pero
hay que poder rastrear cada copia.

Como el panel, son point-in-time: se reescribe el día entero de la región en
cada ejecución, no se acumulan versiones. ``ticker``/``mic`` van vacíos en los
temas que salen de fuentes globales (portada de un medio).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from analyzer.storage.news import naive_utc
from analyzer.storage.table import Table

CLUSTER_COLUMNS = (
    "cluster_id",
    "ticker",
    "mic",
    "representative_id",
    "size",
    "method",
    "sources",
    "reliability",
    "language",
    "first_published_at",
    "last_published_at",
    "title",
    "summary",
    "url",
    "sentiment_label",
    "sentiment_score",
)
STORED_CLUSTER_COLUMNS = ("region", "as_of", *CLUSTER_COLUMNS, "created_at")
MEMBER_COLUMNS = ("cluster_id", "news_id", "method", "is_representative")
STORED_MEMBER_COLUMNS = ("region", "as_of", *MEMBER_COLUMNS, "created_at")

_TIMESTAMPS = ("first_published_at", "last_published_at")


class NewsClusterStore(Table):
    NAME = "news_clusters"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        region             VARCHAR   NOT NULL,
        as_of              DATE      NOT NULL,
        cluster_id         VARCHAR   NOT NULL,
        ticker             VARCHAR,
        mic                VARCHAR,
        representative_id  VARCHAR   NOT NULL,
        size               INTEGER   NOT NULL,
        method             VARCHAR   NOT NULL,
        sources            VARCHAR   NOT NULL,
        reliability        DOUBLE    NOT NULL,
        language           VARCHAR   NOT NULL,
        first_published_at TIMESTAMP NOT NULL,
        last_published_at  TIMESTAMP NOT NULL,
        title              VARCHAR   NOT NULL,
        summary            VARCHAR,
        url                VARCHAR   NOT NULL,
        sentiment_label    VARCHAR,          -- NULL si el modelo no está instalado
        sentiment_score    DOUBLE,           -- firmado: +confianza si positivo, - si negativo
        created_at         TIMESTAMP NOT NULL,
        PRIMARY KEY (region, as_of, cluster_id)
    )
    """

    def replace_day(self, region: str, as_of: date, clusters: pd.DataFrame) -> int:
        """Sustituye los temas de ``region`` en ``as_of``; la transacción la pone quien llama."""
        incoming = _prepare(clusters, CLUSTER_COLUMNS, region, as_of)
        for column in _TIMESTAMPS:
            incoming[column] = naive_utc(incoming[column])
        incoming["size"] = incoming["size"].astype("int32")
        incoming["reliability"] = incoming["reliability"].astype("float64")
        incoming["sentiment_score"] = incoming["sentiment_score"].astype("float64")
        self._delete_day(region, as_of)
        return self._insert_or_replace(incoming, STORED_CLUSTER_COLUMNS)

    def load(self, region: str, as_of: date) -> pd.DataFrame:
        """Temas de ``region`` en ``as_of``, del más reciente al más antiguo."""
        frame = self._con.execute(
            f"SELECT {', '.join(CLUSTER_COLUMNS)} FROM {self.table} "  # noqa: S608
            "WHERE region = ? AND as_of = ? ORDER BY last_published_at DESC, cluster_id",
            [region, as_of],
        ).df()
        for column in _TIMESTAMPS:
            frame[column] = pd.to_datetime(frame[column]).dt.tz_localize("UTC")
        return frame

    def _delete_day(self, region: str, as_of: date) -> None:
        self._con.execute(
            f"DELETE FROM {self.table} WHERE region = ? AND as_of = ?",  # noqa: S608
            [region, as_of],
        )


class NewsClusterMemberStore(Table):
    NAME = "news_cluster_members"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        region            VARCHAR   NOT NULL,
        as_of             DATE      NOT NULL,
        cluster_id        VARCHAR   NOT NULL,
        news_id           VARCHAR   NOT NULL,
        method            VARCHAR   NOT NULL,   -- nivel de la cascada que lo unió al tema
        is_representative BOOLEAN   NOT NULL,
        created_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (region, as_of, cluster_id, news_id)
    )
    """

    def replace_day(self, region: str, as_of: date, members: pd.DataFrame) -> int:
        """Sustituye los miembros de ``region`` en ``as_of``; la transacción es de quien llama."""
        incoming = _prepare(members, MEMBER_COLUMNS, region, as_of)
        incoming["is_representative"] = incoming["is_representative"].astype("bool")
        self._con.execute(
            f"DELETE FROM {self.table} WHERE region = ? AND as_of = ?",  # noqa: S608
            [region, as_of],
        )
        return self._insert_or_replace(incoming, STORED_MEMBER_COLUMNS)

    def load(self, region: str, as_of: date) -> pd.DataFrame:
        return self._con.execute(
            f"SELECT {', '.join(MEMBER_COLUMNS)} FROM {self.table} "  # noqa: S608
            "WHERE region = ? AND as_of = ? ORDER BY cluster_id, news_id",
            [region, as_of],
        ).df()


def _prepare(
    frame: pd.DataFrame, columns: tuple[str, ...], region: str, as_of: date
) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"faltan columnas de temas: {', '.join(missing)}")
    incoming = frame.loc[:, list(columns)].copy()
    incoming["region"] = region
    incoming["as_of"] = as_of
    incoming["created_at"] = datetime.now(UTC).replace(tzinfo=None)
    return incoming
