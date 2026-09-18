"""Cálculo incremental de indicadores sobre ``prod.prices``.

Cada día se carga la ventana de ``lookback_days`` sesiones de cada valor del
universo y se calcula el catálogo entero sobre ella: es barato (unos cientos
de valores por unos cientos de velas) y evita arrastrar estado entre días.
Lo incremental está en qué se escribe:

- Filas posteriores a la última guardada de cada valor (normalmente, la de hoy).
- Todo el histórico completo de los valores sin filas o marcados para
  recálculo (``corporate_actions`` reingestó su histórico y las filas viejas
  tienen un escalón).
- Nunca filas con menos velas previas que el ``warmup`` del catálogo: serían
  artefactos de la ventana. La excepción es un valor con menos histórico que
  el propio warmup (salida a bolsa reciente), donde los NaN son reales.

Los valores en cuarentena no tienen vela de hoy en prod, así que hoy no
reciben fila y el screener no los ve. Sin casos especiales.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd
import structlog

from analyzer.steps.indicators.registry import INDICATOR_COLUMNS, WARMUP, compute_for_key
from analyzer.steps.ingest.service import lookback_start, universe_keys
from analyzer.storage import BASE_COLUMNS, IndicatorStore, PriceStore

log = structlog.get_logger(__name__)

Key = tuple[str, str]
POSITION = "_position"  # velas previas de la clave dentro de la ventana
HISTORY = "_history"  # velas totales de la clave dentro de la ventana
OUTPUT_COLUMNS = ("ticker", "mic", "date", *BASE_COLUMNS, *INDICATOR_COLUMNS)


@dataclass(frozen=True)
class IndicatorsReport:
    as_of: date
    expected: int  # claves del universo
    computed: int  # claves con velas en la ventana
    with_today: int  # claves con fila de as_of
    recomputed: int  # claves recalculadas desde cero (sin filas previas o forzadas)
    rows: int  # filas escritas
    short_history: int  # claves con menos velas que el warmup del catálogo
    window_start: date

    def summary(self) -> str:
        text = f"{self.with_today}/{self.expected} valores con indicadores, {self.rows} filas"
        if self.recomputed:
            text += f", {self.recomputed} desde cero"
        if self.short_history:
            text += f", {self.short_history} con histórico corto"
        return text


def run_indicators(
    universe: pd.DataFrame,
    as_of: date,
    *,
    prices: PriceStore,
    store: IndicatorStore,
    lookback_days: int,
    full_recompute: Iterable[Key] = (),
) -> IndicatorsReport:
    keys = universe_keys(universe)
    start = lookback_start(as_of, lookback_days)
    window = prices.load(start=start, end=as_of, keys=keys)
    if window.empty:
        raise ValueError(f"no hay velas en prod entre {start} y {as_of}")

    computed = compute_indicators(window)
    with_today = int((computed["date"] == pd.Timestamp(as_of)).sum())
    if with_today == 0:
        raise ValueError(f"ninguna vela de {as_of} en prod: quality no promovió el día")

    forced = set(full_recompute)
    stored = store.last_dates(keys)
    rows = store.upsert(select_rows(computed, stored, forced, warmup=WARMUP))

    history = computed.groupby(["ticker", "mic"], sort=False)[HISTORY].first()
    known = set(zip(stored["ticker"], stored["mic"], strict=True))
    recomputed = {k for k in _keys_of(computed) if k in forced or k not in known}
    report = IndicatorsReport(
        as_of=as_of,
        expected=len(keys),
        computed=len(history),
        with_today=with_today,
        recomputed=len(recomputed),
        rows=rows,
        short_history=int((history <= WARMUP).sum()),
        window_start=start,
    )
    log.info("indicators.computed", detail=report.summary(), keys=report.computed)
    return report


def compute_indicators(window: pd.DataFrame) -> pd.DataFrame:
    """Catálogo entero para cada clave de ``window``, más posición e histórico en la ventana."""
    parts: list[pd.DataFrame] = []
    for _, bars in window.groupby(["ticker", "mic"], sort=True):
        bars = bars.sort_values("date").reset_index(drop=True)
        out = bars.loc[:, ["ticker", "mic", "date", *BASE_COLUMNS]].copy()
        out[POSITION] = range(len(bars))
        out[HISTORY] = len(bars)
        parts.append(pd.concat([out, compute_for_key(bars)], axis=1))
    if not parts:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, POSITION, HISTORY])
    return pd.concat(parts, ignore_index=True)


def select_rows(
    computed: pd.DataFrame, stored: pd.DataFrame, forced: set[Key], *, warmup: int
) -> pd.DataFrame:
    """Filas a escribir: completas (ver módulo) y posteriores a lo guardado, salvo recálculo."""
    if computed.empty:
        return computed
    complete = (computed[POSITION] >= warmup) | (computed[HISTORY] <= warmup)
    merged = computed.merge(stored, on=["ticker", "mic"], how="left")
    last_stored = pd.to_datetime(merged["last_date"].astype("object"), errors="coerce")
    last_stored.index = computed.index
    is_forced = pd.Series([k in forced for k in _keys_of(computed)], index=computed.index)
    fresh = last_stored.isna() | (computed["date"] > last_stored) | is_forced
    return computed.loc[complete & fresh]


def _keys_of(frame: pd.DataFrame) -> list[Key]:
    return list(zip(frame["ticker"].astype(str), frame["mic"].astype(str), strict=True))
