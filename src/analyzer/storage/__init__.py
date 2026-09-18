"""Almacenamiento local: un fichero DuckDB en ``data/``.

Dos esquemas con las mismas tablas: ``staging`` (ingesta cruda, la escribe
``ingest`` y nadie genera señales desde ahí) y ``prod`` (solo lo que ha pasado
``quality``). Promover datos es copiar filas de un esquema al otro.

``connect`` crea la carpeta de datos y los esquemas si no existen, para que
``data/`` no tenga que estar en git. Cada tabla es una subclase de ``Table``
que se crea (y migra) sola al instanciarse.
"""

from __future__ import annotations

import duckdb

from analyzer.storage.corporate_actions import ACTION_COLUMNS, CorporateActionStore
from analyzer.storage.indicators import BASE_COLUMNS, IndicatorStore
from analyzer.storage.news import LOADED_NEWS_COLUMNS, NEWS_COLUMNS, NewsStore
from analyzer.storage.prices import (
    LOADED_COLUMNS,
    OPTIONAL_PRICE_COLUMNS,
    PRICE_COLUMNS,
    UNKNOWN_MIC,
    PriceStore,
)
from analyzer.storage.snapshots import UNIVERSE_COLUMNS, SnapshotRunStore, SnapshotStore
from analyzer.storage.sql import check_identifier
from analyzer.storage.table import Table
from core.config import AppConfig


def connect(cfg: AppConfig) -> duckdb.DuckDBPyConnection:
    """Abre (o crea) la base de datos y garantiza los esquemas de settings."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.duckdb_path))
    for schema in (cfg.settings.data.staging_schema, cfg.settings.data.prod_schema):
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{check_identifier(schema)}"')
    return con


__all__ = [
    "ACTION_COLUMNS",
    "BASE_COLUMNS",
    "LOADED_COLUMNS",
    "LOADED_NEWS_COLUMNS",
    "NEWS_COLUMNS",
    "OPTIONAL_PRICE_COLUMNS",
    "PRICE_COLUMNS",
    "UNIVERSE_COLUMNS",
    "UNKNOWN_MIC",
    "CorporateActionStore",
    "IndicatorStore",
    "NewsStore",
    "PriceStore",
    "SnapshotRunStore",
    "SnapshotStore",
    "Table",
    "check_identifier",
    "connect",
]
