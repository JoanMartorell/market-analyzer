"""Detección de eventos corporativos sobre la ventana de velas. Sin I/O.

Dos detectores, del más fiable al menos:
- ``detect_from_feed``: columnas ``dividend`` y ``split`` que el proveedor
  trae con las velas. Autoritativo cuando existe.
- ``detect_splits_by_ratio``: para proveedores sin feed. Un salto del cierre
  ajustado que coincide con un ratio de split habitual (2:1, 3:1, 1:10...)
  y que el cierre real reproduce igual. Un movimiento de mercado real rara
  vez cae clavado en uno de esos ratios.

``needs_reingest`` decide si el histórico guardado de un valor es anterior al
evento: si lo es, el proveedor ya lo ha reescrito y hay que volver a bajarlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

type Day = date  # alias: dentro de CorporateAction el nombre ``date`` es el campo

KIND_SPLIT = "split"
KIND_DIVIDEND = "dividend"
SOURCE_FEED = "feed"
SOURCE_RATIO = "ratio"

# Acciones nuevas por antigua. Splits normales y contrasplits habituales.
KNOWN_SPLIT_RATIOS: tuple[float, ...] = (
    1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 50.0,
    1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 1 / 25, 1 / 50,
)  # fmt: skip
RATIO_TOLERANCE = 0.03  # el precio post-split se mueve algo en el día
MIN_RATIO_MOVE = 0.25  # por debajo de esto no se especula con splits
SAME_RUN_TOLERANCE = timedelta(hours=1)  # velas cargadas en la misma ejecución


@dataclass(frozen=True)
class CorporateAction:
    ticker: str
    mic: str
    date: Day
    kind: str  # KIND_SPLIT | KIND_DIVIDEND
    value: float  # ratio (split) o importe por acción (dividend)
    source: str  # SOURCE_FEED | SOURCE_RATIO

    @property
    def key(self) -> tuple[str, str]:
        return (self.ticker, self.mic)

    @property
    def identity(self) -> tuple[str, str, Day, str]:
        return (self.ticker, self.mic, self.date, self.kind)

    def describe(self) -> str:
        if self.kind == KIND_SPLIT:
            return f"{self.ticker} split {_ratio_text(self.value)} el {self.date}"
        return f"{self.ticker} dividendo {self.value:g} el {self.date}"


def detect_from_feed(window: pd.DataFrame) -> list[CorporateAction]:
    """Eventos declarados por el proveedor en las columnas ``split`` y ``dividend``."""
    found: list[CorporateAction] = []
    if "split" in window.columns:
        splits = window[window["split"].notna() & (window["split"] > 0) & (window["split"] != 1)]
        found.extend(_from_rows(splits, KIND_SPLIT, "split"))
    if "dividend" in window.columns:
        dividends = window[window["dividend"].notna() & (window["dividend"] > 0)]
        found.extend(_from_rows(dividends, KIND_DIVIDEND, "dividend"))
    return sorted(found, key=lambda a: a.identity)


def detect_splits_by_ratio(window: pd.DataFrame, day: date) -> list[CorporateAction]:
    """Splits inferidos en la vela de ``day`` para claves cuyo proveedor no trae feed."""
    found: list[CorporateAction] = []
    moment = pd.Timestamp(day)
    for (ticker, mic), group in window.groupby(["ticker", "mic"], sort=True):
        group = group.sort_values("date")
        current = group[group["date"] == moment]
        previous = group[group["date"] < moment]
        if current.empty or previous.empty:
            continue
        bar, prev = current.iloc[-1], previous.iloc[-1]
        if "split" in bar.index and pd.notna(bar["split"]):
            continue  # el feed manda
        ratio = _split_ratio(bar, prev)
        if ratio is not None:
            found.append(
                CorporateAction(str(ticker), str(mic), day, KIND_SPLIT, ratio, SOURCE_RATIO)
            )
    return found


def needs_reingest(window: pd.DataFrame, action: CorporateAction) -> bool:
    """¿Hay velas del valor anteriores al evento cargadas antes que la vela del evento?"""
    rows = window[(window["ticker"] == action.ticker) & (window["mic"] == action.mic)]
    event_rows = rows[rows["date"] == pd.Timestamp(action.date)]
    if event_rows.empty:
        return False
    older = rows[rows["date"] < pd.Timestamp(action.date)]
    if older.empty:
        return False
    event_loaded = pd.to_datetime(event_rows["ingested_at"]).iloc[-1]
    oldest_load = pd.to_datetime(older["ingested_at"]).min()
    return bool(oldest_load < event_loaded - SAME_RUN_TOLERANCE)


def _from_rows(rows: pd.DataFrame, kind: str, column: str) -> list[CorporateAction]:
    days = pd.to_datetime(rows["date"]).dt.date
    values = rows[column].astype("float64")
    return [
        CorporateAction(str(ticker), str(mic), day, kind, float(value), SOURCE_FEED)
        for ticker, mic, day, value in zip(rows["ticker"], rows["mic"], days, values, strict=True)
    ]


def _split_ratio(bar: pd.Series, prev: pd.Series) -> float | None:
    prev_adj, prev_close = float(prev["adj_close"]), float(prev["close"])
    if prev_adj <= 0 or prev_close <= 0:
        return None
    move_adj = float(bar["adj_close"]) / prev_adj
    move_close = float(bar["close"]) / prev_close
    if abs(move_adj - 1) < MIN_RATIO_MOVE:
        return None
    if abs(move_close / move_adj - 1) > RATIO_TOLERANCE:
        return None  # real y ajustado no cuentan lo mismo: no es un split limpio
    implied = 1 / move_adj  # el precio se divide por el ratio
    for ratio in KNOWN_SPLIT_RATIOS:
        if abs(implied / ratio - 1) <= RATIO_TOLERANCE:
            return ratio
    return None


def _ratio_text(value: float) -> str:
    if value >= 1:
        return f"{value:g}:1"
    return f"1:{1 / value:g}"
