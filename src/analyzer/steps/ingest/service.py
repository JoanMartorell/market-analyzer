"""Ingesta incremental de precios: qué falta, descargarlo, escribirlo en staging.

Lo comparten el paso diario y el comando ``analyzer ingest backfill``. No
sabe nada del motor de pasos ni de la configuración: recibe el universo, el
proveedor y el almacén, y devuelve un informe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import structlog

from analyzer.steps.ingest.providers.base import PriceProvider
from analyzer.storage import UNKNOWN_MIC, PriceStore

log = structlog.get_logger(__name__)

# Sesiones por año natural en la mayoría de bolsas; se usa para pasar de
# "N sesiones hacia atrás" a días de calendario con un margen.
SESSIONS_PER_YEAR = 252
LOOKBACK_MARGIN_DAYS = 7


@dataclass(frozen=True)
class IngestReport:
    provider: str
    as_of: date
    requested: int  # claves que había que actualizar
    up_to_date: int  # claves que ya tenían as_of
    received: int  # claves con al menos una vela nueva
    rows: int  # velas escritas en staging
    start: date | None  # inicio más antiguo pedido
    missing: tuple[str, ...]  # claves pedidas sin datos, como "ticker@mic"

    def summary(self) -> str:
        if self.requested == 0:
            return f"{self.up_to_date} valores ya al día a {self.as_of}"
        span = f" desde {self.start}" if self.start else ""
        text = (
            f"{self.received}/{self.requested} valores con datos{span}, {self.rows} velas a staging"
        )
        if self.up_to_date:
            text += f", {self.up_to_date} ya al día"
        if self.missing:
            text += f", sin datos: {len(self.missing)}"
        return text


def lookback_start(as_of: date, sessions: int) -> date:
    """Fecha de calendario que cubre ``sessions`` sesiones antes de ``as_of``."""
    days = math.ceil(sessions * 366 / SESSIONS_PER_YEAR) + LOOKBACK_MARGIN_DAYS
    return as_of - timedelta(days=days)


def universe_keys(universe: pd.DataFrame) -> pd.DataFrame:
    """Claves (ticker, mic) únicas del universo, con ``UNKNOWN_MIC`` donde no hay bolsa."""
    keys = universe.loc[:, ["ticker", "mic"]].copy()
    keys["ticker"] = keys["ticker"].astype(str)
    keys["mic"] = keys["mic"].fillna(UNKNOWN_MIC).astype(str)
    return keys.drop_duplicates().reset_index(drop=True)


def plan_fetch(
    keys: pd.DataFrame,
    last_dates: pd.DataFrame,
    as_of: date,
    default_start: date,
    force_start: date | None = None,
) -> pd.DataFrame:
    """Añade a ``keys`` la columna ``start``: desde cuándo pedir, o ``None`` si ya está al día.

    Sin datos previos se pide desde ``default_start``; con datos, desde el día
    siguiente al último guardado. ``force_start`` ignora lo guardado (backfill).
    """
    planned = keys.merge(last_dates, on=["ticker", "mic"], how="left")
    if force_start is not None:
        starts = pd.Series([force_start] * len(planned), index=planned.index, dtype="object")
    else:
        has_last = planned["last_date"].notna()
        starts = pd.Series([default_start] * len(planned), index=planned.index, dtype="object")
        starts[has_last] = planned.loc[has_last, "last_date"].map(lambda d: d + timedelta(days=1))
    up_to_date = starts.map(lambda d: d > as_of)
    starts[up_to_date] = None
    planned["start"] = starts
    return planned.drop(columns=["last_date"])


def ingest_prices(
    universe: pd.DataFrame,
    as_of: date,
    *,
    provider: PriceProvider,
    store: PriceStore,
    lookback_days: int,
    start: date | None = None,
) -> IngestReport:
    """Descarga lo que falta hasta ``as_of`` para el universo y lo escribe en ``store``."""
    keys = universe_keys(universe)
    planned = plan_fetch(
        keys, store.last_dates(), as_of, lookback_start(as_of, lookback_days), force_start=start
    )
    todo = planned[planned["start"].notna()]
    up_to_date = len(planned) - len(todo)

    frames: list[pd.DataFrame] = []
    starts: list[date] = sorted({d for d in todo["start"] if isinstance(d, date)})
    for start_day in starts:
        group = todo[todo["start"] == start_day]
        log.info(
            "ingest.fetch",
            provider=provider.name,
            keys=len(group),
            start=str(start_day),
            end=str(as_of),
        )
        frame = provider.fetch(group[["ticker", "mic"]], start_day, as_of)
        if not frame.empty:
            frames.append(frame)

    fetched = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "mic"])
    )
    fetched["source"] = provider.name
    rows = store.upsert(fetched) if frames else 0

    received_keys = set(zip(fetched["ticker"], fetched["mic"], strict=True))
    todo_keys = zip(todo["ticker"], todo["mic"], strict=True)
    missing = tuple(sorted(f"{t}@{m}" for t, m in todo_keys if (t, m) not in received_keys))
    return IngestReport(
        provider=provider.name,
        as_of=as_of,
        requested=len(todo),
        up_to_date=up_to_date,
        received=len(todo) - len(missing),
        rows=rows,
        start=min(starts) if starts else None,
        missing=missing,
    )
