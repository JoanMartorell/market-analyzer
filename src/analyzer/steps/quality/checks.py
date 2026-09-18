"""Comprobaciones puras sobre DataFrames de velas. Sin I/O.

Dos niveles:
- Globales (``feed_date``, ``coverage``): si fallan, no hay señales ese día.
- Por valor (``find_anomalies``): el valor queda en cuarentena para ese día
  y se informa; solo si la cuarentena hunde la cobertura se falla.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

AVG_VOLUME_WINDOW = 20
OHLC_TOLERANCE = 0.001  # redondeos del proveedor


@dataclass(frozen=True)
class Anomaly:
    ticker: str
    mic: str
    reason: str  # jump | zero_volume | invalid_ohlc
    detail: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.ticker, self.mic)


def keys_with_bar(window: pd.DataFrame, day: date) -> set[tuple[str, str]]:
    """Claves con una vela exactamente en ``day``."""
    bars = window[window["date"] == pd.Timestamp(day)]
    return set(zip(bars["ticker"], bars["mic"], strict=True))


def feed_last_date(window: pd.DataFrame) -> date | None:
    if window.empty:
        return None
    last = window["date"].max()
    return last.date() if isinstance(last, pd.Timestamp) else None


def find_anomalies(
    window: pd.DataFrame,
    day: date,
    *,
    max_daily_jump: float,
    zero_volume_min_avg_volume: int,
) -> list[Anomaly]:
    """Anomalías en la vela de ``day`` de cada clave, usando el histórico de ``window``."""
    anomalies: list[Anomaly] = []
    moment = pd.Timestamp(day)
    for (ticker, mic), group in window.groupby(["ticker", "mic"], sort=True):
        group = group.sort_values("date")
        current = group[group["date"] == moment]
        if current.empty:
            continue
        bar = current.iloc[-1]
        previous = group[group["date"] < moment]

        invalid = _invalid_ohlc(bar)
        if invalid:
            anomalies.append(Anomaly(str(ticker), str(mic), "invalid_ohlc", invalid))
            continue  # con una vela rota el resto de comprobaciones no dice nada

        if not previous.empty:
            prev_adj = float(previous["adj_close"].iloc[-1])
            if prev_adj > 0:
                move = float(bar["adj_close"]) / prev_adj - 1
                if abs(move) > max_daily_jump:
                    anomalies.append(
                        Anomaly(str(ticker), str(mic), "jump", f"{move:+.1%} ajustado en un día")
                    )
                    continue

        volume = float(bar["volume"]) if pd.notna(bar["volume"]) else 0.0
        if volume == 0 and len(previous) >= 1:
            avg = float(previous["volume"].tail(AVG_VOLUME_WINDOW).fillna(0).mean())
            if avg >= zero_volume_min_avg_volume:
                anomalies.append(
                    Anomaly(str(ticker), str(mic), "zero_volume", f"media 20d {avg:,.0f}")
                )
    return anomalies


def _invalid_ohlc(bar: pd.Series) -> str | None:
    close = float(bar["close"])
    if not close > 0:
        return f"cierre {close}"
    high, low = bar["high"], bar["low"]
    if pd.isna(high) or pd.isna(low):
        return None  # sin máximo/mínimo no se puede juzgar; el cierre ya es válido
    high, low = float(high), float(low)
    if high < low:
        return f"máximo {high} < mínimo {low}"
    if close > high * (1 + OHLC_TOLERANCE) or close < low * (1 - OHLC_TOLERANCE):
        return f"cierre {close} fuera de [{low}, {high}]"
    return None
