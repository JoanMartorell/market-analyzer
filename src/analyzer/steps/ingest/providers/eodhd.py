"""Proveedor EODHD (eodhd.com): rescate con cupo, no fuente principal.

El plan gratuito da 20 llamadas al día, cada una de un solo valor y rango,
y prohíbe las descargas por bolsa. Con eso no se alimenta un universo de
cientos de valores: se completan los que el proveedor principal dejó sin
la vela del día. Cada llamada se reserva en ``ApiCallStore`` antes de
hacerse; agotado el cupo, los valores que quedan no se piden y se anota
cuántos.

El endpoint ``eod`` devuelve cierre real y ajustado pero no dividendos ni
splits, así que esas columnas van vacías. Un 404 es un símbolo que EODHD
no conoce; un 401, 402, 403, 423 o 429 es un problema de plan o de cuota
y corta el resto de peticiones del ciclo: insistir solo gastaría cupo.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pandas as pd
import structlog

from analyzer.steps.ingest.providers.base import ALL_FETCH_COLUMNS, empty_fetch
from analyzer.storage import UNKNOWN_MIC
from analyzer.storage.api_calls import ApiCallStore

log = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "https://eodhd.com/api"
STOP_STATUSES = frozenset({401, 402, 403, 423, 429})  # plan o cuota: no insistir

# Código de bolsa de EODHD por MIC.
EODHD_EXCHANGE: Mapping[str, str] = {
    "XNYS": "US",
    "XNAS": "US",
    "BATS": "US",
    UNKNOWN_MIC: "US",
    "XLON": "LSE",
    "XETR": "XETRA",
    "XPAR": "PA",
    "XAMS": "AS",
    "XMAD": "MC",
    "XMIL": "MI",
    "XSWX": "SW",
    "XSTO": "ST",
    "XOSL": "OL",
    "XCSE": "CO",
    "XHEL": "HE",
    "XBRU": "BR",
    "XWBO": "VI",
    "XDUB": "IR",
    "XLIS": "LS",
    "XWAR": "WAR",
    "XTKS": "TSE",
    "XHKG": "HK",
    "XASX": "AU",
    "XKRX": "KO",
}

_FIELDS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjusted_close": "adj_close",
    "volume": "volume",
}


def eodhd_symbol(ticker: str, mic: str) -> str | None:
    """``BRK.B``/XNYS -> ``BRK-B.US``; ``SAP``/XETR -> ``SAP.XETRA``; bolsa no cubierta -> None."""
    exchange = EODHD_EXCHANGE.get(mic)
    if exchange is None:
        return None
    base = ticker.strip().upper().replace(" ", "-").replace(".", "-")
    return f"{base}.{exchange}" if base else None


class DailyBudget:
    """Cupo diario de un proveedor, llevado en la base de datos y no en memoria."""

    def __init__(
        self,
        store: ApiCallStore,
        provider: str,
        limit: int,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self.limit = limit
        self._today = today or (lambda: datetime.now(UTC).date())

    def reserve(self) -> bool:
        """Apunta una llamada; ``False`` si el cupo de hoy ya está agotado."""
        return self._store.reserve(self._provider, self._today(), self.limit)

    def used(self) -> int:
        return self._store.used(self._provider, self._today())


class _StopError(Exception):
    """Respuesta tras la que no conviene hacer más peticiones en este ciclo."""


class EodhdProvider:
    name = "eodhd"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        budget: DailyBudget | None = None,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.budget = budget
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def fetch(self, keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        pairs = [(str(t), str(m)) for t, m in zip(keys["ticker"], keys["mic"], strict=True)]
        frames: list[pd.DataFrame] = []
        unsupported: list[str] = []
        unknown: list[str] = []
        requested = 0
        skipped = 0
        for position, (ticker, mic) in enumerate(pairs):
            symbol = eodhd_symbol(ticker, mic)
            if symbol is None:
                unsupported.append(f"{ticker}@{mic}")
                continue
            if self.budget is not None and not self.budget.reserve():
                skipped = sum(1 for t, m in pairs[position:] if eodhd_symbol(t, m) is not None)
                log.warning("eodhd.budget_exhausted", limit=self.budget.limit, skipped=skipped)
                break
            requested += 1
            try:
                rows = self._get(symbol, start, end)
            except _StopError:
                break
            if not rows:
                unknown.append(symbol)
                continue
            frames.append(_to_frame(rows, ticker, mic))
        log.info(
            "eodhd.fetch",
            requested=requested,
            received=len(frames),
            unknown=len(unknown),
            unsupported=len(unsupported),
            skipped=skipped,
        )
        return pd.concat(frames, ignore_index=True) if frames else empty_fetch()

    def _get(self, symbol: str, start: date, end: date) -> list[dict[str, Any]] | None:
        params = {
            "api_token": self._api_key,
            "fmt": "json",
            "period": "d",
            "from": start.isoformat(),
            "to": end.isoformat(),
        }
        try:
            response = self._client.get(f"{self._base_url}/eod/{symbol}", params=params)
        except httpx.HTTPError as exc:
            # Solo el tipo: el mensaje podría llevar la URL con la clave.
            log.warning("eodhd.request_failed", symbol=symbol, error=type(exc).__name__)
            raise _StopError from exc
        if response.status_code == 404:
            return None
        if response.status_code in STOP_STATUSES:
            log.warning(
                "eodhd.rejected",
                symbol=symbol,
                status=response.status_code,
                detail=response.text[:120],
            )
            raise _StopError
        if response.status_code != 200:
            log.warning("eodhd.error", symbol=symbol, status=response.status_code)
            return None
        try:
            payload = response.json()
        except ValueError:
            log.warning("eodhd.bad_payload", symbol=symbol)
            return None
        return payload if isinstance(payload, list) else None


def _to_frame(rows: list[dict[str, Any]], ticker: str, mic: str) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    out = pd.DataFrame({"ticker": ticker, "mic": mic, "date": pd.to_datetime(raw["date"])})
    for source, target in _FIELDS.items():
        values = raw[source] if source in raw.columns else pd.Series(float("nan"), index=raw.index)
        out[target] = pd.to_numeric(values, errors="coerce")
    out["dividend"] = float("nan")
    out["split"] = float("nan")
    return out.loc[:, list(ALL_FETCH_COLUMNS)]
