"""Panel point-in-time: lo que el screener ve hoy, congelado con su hash.

``prod.prices`` y ``prod.indicators`` son "lo mejor que sabemos ahora": un
split regenera el histórico, un dividendo reajusta el ``adj_close`` de
todas las fechas anteriores y quality puede sacar un valor de cuarentena.
La fila de hoy que ve el screener puede no ser la misma dentro de un mes.
El panel es la foto de hoy tal cual se usó, y su hash la identifica: una
señal guarda ese hash, y repetir el día con los mismos datos da el mismo hash.

El panel es una fila por valor del universo con fila de indicadores en
``as_of``: atributos del universo (sector, divisa...) más las columnas de
``prod.indicators``. Sin fila de hoy no hay valor en el panel, así que los
valores en cuarentena quedan fuera solos. Si la cobertura baja del mínimo,
el día está roto y se falla en cerrado.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import structlog
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from analyzer.storage import (
    UNIVERSE_COLUMNS,
    UNKNOWN_MIC,
    IndicatorStore,
    SnapshotRunStore,
    SnapshotStore,
)

log = structlog.get_logger(__name__)

SHORT_HASH = 12  # caracteres del hash que se muestran en resúmenes


@dataclass(frozen=True)
class SnapshotReport:
    region_id: str
    as_of: date
    expected: int  # claves del universo
    rows: int  # valores en el panel
    coverage: float  # rows / expected
    snapshot_hash: str
    previous_hash: str | None  # hash guardado antes de esta ejecución para el mismo día
    panel: pd.DataFrame = field(repr=False, compare=False)

    @property
    def changed(self) -> bool | None:
        """``None`` si es el primer panel del día; si no, si difiere del guardado."""
        return None if self.previous_hash is None else self.previous_hash != self.snapshot_hash

    def summary(self) -> str:
        text = (
            f"{self.rows}/{self.expected} valores en el panel, cobertura {self.coverage:.1%}, "
            f"hash {self.snapshot_hash[:SHORT_HASH]}"
        )
        if self.changed is True:
            text += ", distinto del anterior"
        elif self.changed is False:
            text += ", igual que el anterior"
        return text


def run_snapshot(
    universe: pd.DataFrame,
    as_of: date,
    *,
    region_id: str,
    indicators: IndicatorStore,
    store: SnapshotStore,
    runs: SnapshotRunStore,
    min_coverage: float,
) -> SnapshotReport:
    members = universe_members(universe)
    today = indicators.load(start=as_of, end=as_of, keys=members)
    panel = build_panel(members, today)
    if panel.empty:
        raise ValueError(f"ningún valor del universo tiene indicadores de {as_of}")

    expected = len(members)
    coverage = len(panel) / expected
    if coverage < min_coverage:
        raise ValueError(
            f"cobertura del panel {coverage:.1%} ({len(panel)}/{expected}) "
            f"por debajo del mínimo {min_coverage:.0%}"
        )

    digest = panel_hash(panel, store.panel_columns)
    previous = runs.get(region_id, as_of)
    with store.transaction():
        rows = store.replace_day(region_id, as_of, panel)
        runs.upsert(region_id, as_of, snapshot_hash=digest, rows=rows, expected=expected)

    report = SnapshotReport(
        region_id=region_id,
        as_of=as_of,
        expected=expected,
        rows=rows,
        coverage=coverage,
        snapshot_hash=digest,
        previous_hash=None if previous is None else str(previous["snapshot_hash"]),
        panel=panel,
    )
    log.info("snapshot.frozen", detail=report.summary(), hash=digest)
    return report


def universe_members(universe: pd.DataFrame) -> pd.DataFrame:
    """Una fila por clave (ticker, mic) con los atributos que pasan al panel."""
    members = universe.copy()
    for column in UNIVERSE_COLUMNS:
        if column not in members.columns:
            members[column] = None
    members["ticker"] = members["ticker"].astype(str)
    members["mic"] = members["mic"].fillna(UNKNOWN_MIC).astype(str)
    columns = ["ticker", "mic", *UNIVERSE_COLUMNS]
    return members.loc[:, columns].drop_duplicates(["ticker", "mic"]).reset_index(drop=True)


def build_panel(members: pd.DataFrame, today: pd.DataFrame) -> pd.DataFrame:
    """Une atributos del universo con la fila de indicadores de hoy; sin fila, sin valor."""
    values = today.drop(columns=["computed_at"], errors="ignore")
    panel = members.merge(values, on=["ticker", "mic"], how="inner")
    return panel.sort_values(["ticker", "mic"]).reset_index(drop=True)


def panel_hash(panel: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """sha256 del contenido del panel, independiente del orden de filas y del índice.

    Cada columna se serializa de forma canónica: los números como float64
    little-endian con NaN y ceros normalizados, las fechas en ISO y el resto
    como texto con un marcador para nulos. Así el hash solo cambia si
    cambia un dato.
    """
    ordered = panel.sort_values(["ticker", "mic"]).reset_index(drop=True)
    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_canonical_bytes(ordered[column]))
        digest.update(b"\x01")
    return digest.hexdigest()


def _canonical_bytes(series: pd.Series) -> bytes:
    if is_datetime64_any_dtype(series):
        return "\x1f".join(series.dt.strftime("%Y-%m-%d").fillna("\x00")).encode("utf-8")
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        values = series.to_numpy(dtype="float64", na_value=np.nan) + 0.0  # -0.0 -> 0.0
        values = np.where(np.isnan(values), np.nan, values)  # un solo patrón de bits para NaN
        return values.astype("<f8").tobytes()
    text = series.astype("object").map(lambda v: "\x00" if pd.isna(v) else str(v))
    return "\x1f".join(text).encode("utf-8")
