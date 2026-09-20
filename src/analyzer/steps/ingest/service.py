"""Ingesta incremental de precios: qué falta, descargarlo, escribirlo en staging.

Lo comparten el paso diario y el comando ``analyzer ingest backfill``. No
sabe nada del motor de pasos ni de la configuración: recibe el universo, el
proveedor y el almacén, y devuelve un informe.

Con un proveedor de rescate (``fallback``), lo que el principal deje sin la
vela de ``as_of`` se vuelve a pedir a ese segundo proveedor, que decide
cuánto puede permitirse (EODHD lleva un cupo diario). Solo se rescatan
valores de los mercados que tuvieron sesión: pedir la vela de hoy a una
bolsa cerrada gastaría cupo para nada.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import structlog

from analyzer.steps.ingest.providers.base import ALL_FETCH_COLUMNS, PriceProvider
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
    stale: tuple[str, ...] = ()  # claves con datos pero sin la vela de as_of
    fallback: str | None = None  # proveedor de rescate, si lo hubo
    rescued: int = 0  # claves a las que el rescate dio la vela de as_of

    def summary(self) -> str:
        if self.requested == 0:
            return f"{self.up_to_date} valores ya al día a {self.as_of}"
        span = f" desde {self.start}" if self.start else ""
        text = (
            f"{self.received}/{self.requested} valores con datos{span}, {self.rows} velas a staging"
        )
        if self.up_to_date:
            text += f", {self.up_to_date} ya al día"
        if self.fallback:
            text += f", {self.rescued} rescatados por {self.fallback}"
        if self.stale:
            text += f", sin llegar a {self.as_of}: {len(self.stale)}"
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
    fallback: PriceProvider | None = None,
    fallback_markets: Collection[str] | None = None,
) -> IngestReport:
    """Descarga lo que falta hasta ``as_of`` para el universo y lo escribe en ``store``.

    ``fallback_markets`` limita el rescate a esos MIC (los que tuvieron
    sesión); ``None`` no limita.
    """
    keys = universe_keys(universe)
    planned = plan_fetch(
        keys, store.last_dates(), as_of, lookback_start(as_of, lookback_days), force_start=start
    )
    todo = planned[planned["start"].notna()]
    up_to_date = len(planned) - len(todo)

    fetched = _fetch(provider, todo, as_of)
    rows = _write(store, fetched, provider.name)

    rescued = 0
    if fallback is not None:
        short = _without_bar(todo, fetched, as_of)
        if fallback_markets is not None:
            short = short[short["mic"].isin(list(fallback_markets))]
        if not short.empty:
            log.info("ingest.rescue", provider=fallback.name, keys=len(short))
            extra = _fetch(fallback, short, as_of)
            rows += _write(store, extra, fallback.name)
            rescued = len(short) - len(_without_bar(short, extra, as_of))
            fetched = pd.concat([fetched, extra], ignore_index=True)

    received_keys = set(zip(fetched["ticker"], fetched["mic"], strict=True))
    todo_keys = list(zip(todo["ticker"], todo["mic"], strict=True))
    missing = tuple(sorted(f"{t}@{m}" for t, m in todo_keys if (t, m) not in received_keys))
    stale_frame = _without_bar(todo, fetched, as_of)
    stale = tuple(
        sorted(
            f"{t}@{m}"
            for t, m in zip(stale_frame["ticker"], stale_frame["mic"], strict=True)
            if (t, m) in received_keys
        )
    )
    starts: list[date] = sorted({d for d in todo["start"] if isinstance(d, date)})
    return IngestReport(
        provider=provider.name,
        as_of=as_of,
        requested=len(todo),
        up_to_date=up_to_date,
        received=len(todo) - len(missing),
        rows=rows,
        start=min(starts) if starts else None,
        missing=missing,
        stale=stale,
        fallback=fallback.name if fallback is not None else None,
        rescued=rescued,
    )


def _fetch(provider: PriceProvider, planned: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Pide a ``provider`` cada grupo de claves desde su ``start`` hasta ``as_of``."""
    frames: list[pd.DataFrame] = []
    for start_day in sorted({d for d in planned["start"] if isinstance(d, date)}):
        group = planned[planned["start"] == start_day]
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
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in ALL_FETCH_COLUMNS})
    return pd.concat(frames, ignore_index=True)


def _write(store: PriceStore, fetched: pd.DataFrame, source: str) -> int:
    if fetched.empty:
        return 0
    frame = fetched.copy()
    frame["source"] = source
    return store.upsert(frame)


def _without_bar(planned: pd.DataFrame, fetched: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Filas de ``planned`` cuya clave no tiene vela de ``as_of`` en ``fetched``."""
    if fetched.empty:
        return planned
    dates = pd.to_datetime(fetched["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    on_day = dates.dt.normalize() == pd.Timestamp(as_of)
    covered = set(zip(fetched.loc[on_day, "ticker"], fetched.loc[on_day, "mic"], strict=True))
    keep = [(t, m) not in covered for t, m in zip(planned["ticker"], planned["mic"], strict=True)]
    return planned.loc[keep]
