"""Formato del fichero de constituyentes y consultas sobre él.

Columnas:
- ticker        símbolo canónico con punto para clases (BRK.B), no guion
- name, sector, sub_industry   solo para miembros actuales; histórico puede ir vacío
- cik           identificador SEC, nullable
- mic           bolsa (XNYS, XNAS), nullable si la fuente no lo sabe
- currency      divisa de cotización
- start         primer día de pertenencia (NaT = anterior a los registros)
- end           primer día ya fuera del índice (NaT = sigue dentro)
- source        de dónde salió la fila
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

COLUMNS: tuple[str, ...] = (
    "ticker",
    "name",
    "sector",
    "sub_industry",
    "cik",
    "mic",
    "currency",
    "start",
    "end",
    "source",
)


def members_as_of(table: pd.DataFrame, day: date) -> pd.DataFrame:
    """Filas vigentes el día ``day``: ``start <= day < end``, con NaT como abierto."""
    moment = pd.Timestamp(day)
    started = table["start"].isna() | (table["start"] <= moment)
    not_ended = table["end"].isna() | (table["end"] > moment)
    members = table.loc[started & not_ended]
    # La clave es (ticker, bolsa): "SAN" es Santander en Madrid y Sanofi en París.
    return members.drop_duplicates(["ticker", "mic"]).sort_values("ticker").reset_index(drop=True)


def members_between(table: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Filas vigentes en algún momento de ``[start, end]`` (backfill sin sesgo de supervivencia)."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    started = table["start"].isna() | (table["start"] <= last)
    not_ended = table["end"].isna() | (table["end"] > first)
    members = table.loc[started & not_ended]
    return members.drop_duplicates(["ticker", "mic"]).sort_values("ticker").reset_index(drop=True)


def write_constituents(table: pd.DataFrame, path: Path, meta: dict[str, Any]) -> Path:
    """Escribe el parquet y un sidecar ``.meta.json`` con cómo se construyó."""
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"faltan columnas {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table[list(COLUMNS)].to_parquet(path, index=False)
    meta_path(path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_constituents(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"fichero de constituyentes no encontrado: {path}")
    table = pd.read_parquet(path)
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}")
    table["start"] = pd.to_datetime(table["start"])
    table["end"] = pd.to_datetime(table["end"])
    return table


def meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")
