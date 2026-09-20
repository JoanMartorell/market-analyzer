"""Proveedor yfinance (Yahoo Finance, gratuito, sin clave).

Traduce (ticker, mic) al símbolo de Yahoo con un sufijo por bolsa y descarga
por lotes con ``auto_adjust=False`` para tener cierre real y ajustado, y con
``actions=True`` para que dividendos y splits vengan en la misma llamada. Los
símbolos que Yahoo no conoce vuelven como columnas NaN y se descartan aquí;
la cobertura la juzga el paso ``quality``, no este.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta

import pandas as pd
import structlog

from analyzer.steps.ingest.providers.base import ALL_FETCH_COLUMNS, empty_fetch
from analyzer.yahoo import yf_symbol

log = structlog.get_logger(__name__)

_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividend",
    "Stock Splits": "split",
}

Downloader = Callable[[list[str], date, date], pd.DataFrame]


def _yf_download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # los fallos los contamos nosotros
    frame: pd.DataFrame = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # end es exclusivo en Yahoo
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    return frame


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, downloader: Downloader | None = None, batch_size: int = 200) -> None:
        self._download = downloader or _yf_download
        self._batch_size = batch_size

    def fetch(self, keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        symbols: dict[str, tuple[str, str]] = {}
        unsupported: list[str] = []
        for ticker, mic in zip(keys["ticker"], keys["mic"], strict=True):
            symbol = yf_symbol(str(ticker), str(mic))
            if symbol is None:
                unsupported.append(f"{ticker}@{mic}")
            else:
                symbols[symbol] = (str(ticker), str(mic))
        if unsupported:
            log.warning(
                "yfinance.unsupported_market", count=len(unsupported), sample=unsupported[:5]
            )

        frames: list[pd.DataFrame] = []
        ordered = list(symbols)
        for batch in _chunks(ordered, self._batch_size):
            started = time.perf_counter()
            raw = self._download(batch, start, end)
            frame = reshape_download(raw, symbols)
            frames.append(frame)
            log.info(
                "yfinance.batch",
                symbols=len(batch),
                received=frame[["ticker", "mic"]].drop_duplicates().shape[0],
                rows=len(frame),
                seconds=round(time.perf_counter() - started, 2),
            )
        if not frames:
            return empty_fetch()
        return pd.concat(frames, ignore_index=True)


def reshape_download(raw: pd.DataFrame, symbols: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """De columnas MultiIndex (símbolo, campo) a filas largas ``ticker, mic, date, ...``."""
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return empty_fetch()
    parts: list[pd.DataFrame] = []
    for symbol in raw.columns.get_level_values(0).unique():
        key = symbols.get(str(symbol))
        if key is None:
            continue
        sub = raw[symbol].rename(columns=_FIELD_MAP)
        sub = sub.loc[:, [c for c in _FIELD_MAP.values() if c in sub.columns]]
        sub = sub.dropna(subset=["close"]) if "close" in sub.columns else sub.iloc[0:0]
        if sub.empty:
            continue
        part = sub.reset_index(names="date")
        part.insert(0, "mic", key[1])
        part.insert(0, "ticker", key[0])
        parts.append(part)
    if not parts:
        return empty_fetch()
    out = pd.concat(parts, ignore_index=True)
    for column in ALL_FETCH_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, list(ALL_FETCH_COLUMNS)]


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
