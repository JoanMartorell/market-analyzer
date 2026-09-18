"""Tabla ``<schema>.news``: un artículo por (fuente, valor, url).

Guarda lo que descarga el paso ``news`` tal cual llega, ya limpio de HTML.
``ticker``/``mic`` van vacíos en las fuentes globales (portada de un medio),
que no hablan de un valor concreto. ``id`` es el sha256 de fuente, valor y
url: repetir el ciclo no duplica nada, y un mismo artículo que aparece en
el feed de dos valores cuenta para los dos. La deduplicación por contenido
es del paso siguiente, no de esta tabla.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime

import pandas as pd

from analyzer.storage.table import Table

NEWS_COLUMNS = (
    "id",
    "ticker",
    "mic",
    "source",
    "reliability",
    "language",
    "published_at",
    "title",
    "summary",
    "url",
)
LOADED_NEWS_COLUMNS = (*NEWS_COLUMNS, "as_of", "fetched_at")


class NewsStore(Table):
    NAME = "news"
    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        id           VARCHAR   NOT NULL,
        ticker       VARCHAR,
        mic          VARCHAR,
        source       VARCHAR   NOT NULL,
        reliability  DOUBLE    NOT NULL,
        language     VARCHAR   NOT NULL,
        published_at TIMESTAMP NOT NULL,
        title        VARCHAR   NOT NULL,
        summary      VARCHAR,
        url          VARCHAR   NOT NULL,
        as_of        DATE      NOT NULL,
        fetched_at   TIMESTAMP NOT NULL,
        PRIMARY KEY (id)
    )
    """

    def upsert(self, frame: pd.DataFrame, as_of: date) -> int:
        missing = [c for c in NEWS_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"faltan columnas de noticias: {', '.join(missing)}")
        if frame.empty:
            return 0
        incoming = frame.loc[:, list(NEWS_COLUMNS)].copy()
        incoming["published_at"] = naive_utc(incoming["published_at"])
        incoming["reliability"] = incoming["reliability"].astype("float64")
        incoming["as_of"] = as_of
        incoming["fetched_at"] = datetime.now(UTC).replace(tzinfo=None)
        return self._insert_or_replace(incoming, LOADED_NEWS_COLUMNS)

    def existing_ids(self, ids: Iterable[str]) -> set[str]:
        """Qué ``ids`` ya están guardados."""
        wanted = pd.DataFrame({"id": sorted(set(ids))})
        if wanted.empty:
            return set()
        view = "wanted_news_ids"
        self._con.register(view, wanted)
        try:
            rows = self._con.execute(
                f"SELECT id FROM {self.table} WHERE id IN (SELECT id FROM {view})"  # noqa: S608
            ).fetchall()
        finally:
            self._con.unregister(view)
        return {str(r[0]) for r in rows}

    def load(
        self,
        since: datetime | None = None,
        keys: pd.DataFrame | None = None,
        *,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """Artículos publicados desde ``since``; con ``keys``, los de esos valores.

        Las fuentes globales (sin valor) entran salvo ``include_global=False``.
        ``published_at`` vuelve como datetime UTC consciente.
        """
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("published_at >= ?")
            params.append(since.astimezone(UTC).replace(tzinfo=None))
        with self._key_filter(keys) as key_clause:
            if key_clause:
                clauses.append(
                    f"({key_clause} OR ticker IS NULL)" if include_global else key_clause
                )
            elif not include_global:
                clauses.append("ticker IS NOT NULL")
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            frame = self._con.execute(
                f"SELECT {', '.join(LOADED_NEWS_COLUMNS)} FROM {self.table}{where} "  # noqa: S608
                "ORDER BY published_at DESC, id",
                params,
            ).df()
        frame["published_at"] = pd.to_datetime(frame["published_at"]).dt.tz_localize("UTC")
        return frame


def naive_utc(series: pd.Series) -> pd.Series:
    """UTC sin zona, que es como DuckDB guarda los TIMESTAMP de este proyecto."""
    stamps = pd.to_datetime(series, utc=True)
    return stamps.dt.tz_convert("UTC").dt.tz_localize(None)
