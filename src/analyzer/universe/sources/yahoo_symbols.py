"""Enriquecedor: cambia los tickers que Yahoo no conoce por los que sí.

Las tablas de Wikipedia traen a veces códigos Reuters (AIRP, BNPP, CAGR),
la bolsa equivocada o empresas que ya no cotizan. El proveedor de precios
traduce (ticker, mic) al símbolo de Yahoo, así que un ticker que Yahoo no
reconoce es un valor sin precios, y una cobertura que no llega al mínimo
de ``quality``.

Se comprueba cada valor con una descarga corta; para los que no devuelven
nada se busca en Yahoo, primero por nombre (sin acentos) en su bolsa,
luego por el propio ticker en su bolsa, y por último por nombre en otra
bolsa de la región, cambiando bolsa y divisa. Lo que la búsqueda no
encuentra se saca de la composición: es casi siempre una empresa absorbida
o excluida que la tabla aún lista, y dejarla hundiría la cobertura. Si la
búsqueda falla (red, cuota), el valor se queda tal cual, sin decidir.

Las resoluciones se guardan en ``yahoo_symbols.json`` junto al fichero
del universo: la composición se reconstruye a diario, y sin memoria un
fallo puntual del buscador cambiaría el ticker de un día para otro.
"""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import structlog

from analyzer.universe.base import FetchContext, Notes
from analyzer.universe.helpers import DEFAULT_USER_AGENT, canonical_ticker
from analyzer.yahoo import YF_SUFFIX

log = structlog.get_logger(__name__)

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
CHECK_DAYS = 14  # dos semanas de velas bastan para saber si Yahoo conoce el símbolo
CACHE_FILE = "yahoo_symbols.json"
SEARCH_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

Quote = dict[str, Any]
Fetcher = Callable[[pd.DataFrame, date, date], pd.DataFrame]
Searcher = Callable[[httpx.Client, str], list[Quote] | None]  # None = la búsqueda falló
Resolution = tuple[str, str]  # (ticker, mic) resueltos


def yahoo_search(client: httpx.Client, query: str) -> list[Quote] | None:
    """Cotizaciones que Yahoo asocia a ``query``; ``None`` si la búsqueda no se pudo hacer."""
    params: dict[str, str | int] = {"q": query, "quotesCount": 10, "newsCount": 0, "listsCount": 0}
    for attempt in range(SEARCH_ATTEMPTS):
        if attempt:
            time.sleep(0.5 * 2**attempt)
        try:
            response = client.get(
                SEARCH_URL, params=params, headers={"User-Agent": DEFAULT_USER_AGENT}
            )
        except httpx.HTTPError as exc:
            log.warning("yahoo_symbols.search_failed", query=query, error=type(exc).__name__)
            continue
        if response.status_code in RETRY_STATUSES:
            log.warning("yahoo_symbols.search_retry", query=query, status=response.status_code)
            continue
        if response.status_code != 200:
            log.warning("yahoo_symbols.search_rejected", query=query, status=response.status_code)
            return None
        try:
            quotes = response.json().get("quotes", [])
        except ValueError:
            return None
        return [q for q in quotes if isinstance(q, dict)]
    return None


def plain_name(name: str) -> str:
    """Sin acentos ni espacios repetidos: Yahoo no encuentra "Crédit Agricole", sí "Credit"."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    return " ".join(stripped.split())


def _download(keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    # Importación tardía: el proveedor vive en analyzer.steps, que a su vez
    # importa el universo; cargarlo aquí arriba sería un ciclo.
    from analyzer.steps.ingest.providers.yfinance import YFinanceProvider

    return YFinanceProvider().fetch(keys, start, end)


@dataclass(frozen=True)
class YahooSymbols:
    markets: Mapping[str, str]  # mic -> divisa; solo se reubica dentro de estas bolsas
    fetch_prices: Fetcher = _download
    search: Searcher = yahoo_search
    pause_seconds: float = 0.2  # entre búsquedas, para no parecer un ataque
    name: str = "yahoo_symbols"

    def enrich(
        self, table: pd.DataFrame, client: httpx.Client, ctx: FetchContext
    ) -> tuple[pd.DataFrame, Notes]:
        out = table.copy()
        known = self._known(out, ctx.today)
        cache_path = ctx.universe_dir / CACHE_FILE if ctx.universe_dir is not None else None
        cache = _load_cache(cache_path)
        resolved: list[str] = []
        relocated: list[str] = []
        dropped: list[str] = []
        drop_rows: list[Any] = []
        failed: list[str] = []
        from_cache = 0
        for idx in out.index:
            ticker = str(out.loc[idx, "ticker"])
            raw_mic = out.loc[idx, "mic"]
            mic = raw_mic if isinstance(raw_mic, str) else None
            if mic is not None and (ticker, mic) in known:
                continue
            label = f"{ticker}@{mic or '?'}"
            match = cache.get(label)
            if match is not None:
                from_cache += 1
            else:
                name = out.loc[idx, "name"] if "name" in out.columns else None
                query = plain_name(name) if isinstance(name, str) and name else ""
                try:
                    match = self._resolve(client, query, ticker, mic)
                except _SearchFailedError:
                    failed.append(label)
                    continue
                finally:
                    time.sleep(self.pause_seconds)
                if match is None:
                    dropped.append(label)
                    drop_rows.append(idx)
                    continue
                cache[label] = match
            new_ticker, new_mic = match
            out.loc[idx, "ticker"] = new_ticker
            if new_mic != mic:
                out.loc[idx, "mic"] = new_mic
                if "currency" in out.columns:
                    out.loc[idx, "currency"] = self.markets[new_mic]
                relocated.append(f"{label} -> {new_mic}")
            resolved.append(f"{label} -> {new_ticker}@{new_mic}")
        out = out.drop(index=drop_rows)
        before = len(out)
        out = out.drop_duplicates(["ticker", "mic"]).reset_index(drop=True)
        _save_cache(cache_path, cache)
        notes: Notes = {
            "checked": len(out),
            "resolved": resolved,
            "from_cache": from_cache,
            "relocated": relocated,
            "dropped": dropped,
            "search_failed": failed,
            "duplicates_dropped": before - len(out),
        }
        log.info(
            "yahoo_symbols.done",
            resolved=len(resolved),
            from_cache=from_cache,
            relocated=len(relocated),
            dropped=len(dropped),
            search_failed=len(failed),
        )
        return out, notes

    def _known(self, table: pd.DataFrame, today: date) -> set[tuple[str, str]]:
        """Claves (ticker, mic) que Yahoo devuelve con velas recientes."""
        keys = table.loc[table["mic"].isin(list(YF_SUFFIX)), ["ticker", "mic"]].astype(str)
        if keys.empty:
            return set()
        bars = self.fetch_prices(keys.drop_duplicates(), today - timedelta(days=CHECK_DAYS), today)
        if bars.empty:
            return set()
        return set(zip(bars["ticker"].astype(str), bars["mic"].astype(str), strict=True))

    def _resolve(
        self, client: httpx.Client, query: str, ticker: str, mic: str | None
    ) -> Resolution | None:
        """Ticker y bolsa según Yahoo: por nombre en su bolsa, por ticker en su bolsa, o reubicado.

        Si la búsqueda no se puede hacer, lanza ``_SearchFailedError``.
        """
        wanted = YF_SUFFIX.get(mic) if mic is not None else None
        by_name = self._equities(client, query) if query else []
        if wanted and mic is not None:
            found = _first(by_name, {mic: wanted})
            if found is None:
                found = _first(self._equities(client, ticker), {mic: wanted})
            if found is not None:
                return found
        return _first(by_name, self._suffixes())

    def _equities(self, client: httpx.Client, query: str) -> list[str]:
        quotes = self.search(client, query)
        if quotes is None:
            raise _SearchFailedError(query)
        return [
            str(q["symbol"])
            for q in quotes
            if q.get("quoteType") == "EQUITY" and isinstance(q.get("symbol"), str)
        ]

    def _suffixes(self) -> dict[str, str]:
        """Sufijos de Yahoo de las bolsas de la región; el vacío (USA) nunca cuenta."""
        return {mic: YF_SUFFIX[mic] for mic in self.markets if YF_SUFFIX.get(mic)}


class _SearchFailedError(Exception):
    """La búsqueda no se pudo hacer: no se decide nada sobre ese valor."""


def _first(symbols: list[str], suffixes: Mapping[str, str]) -> Resolution | None:
    """Primer símbolo con uno de los sufijos, como (ticker canónico, mic)."""
    for symbol in symbols:
        for mic, suffix in suffixes.items():
            if suffix and symbol.endswith(suffix):
                base = canonical_ticker(symbol[: -len(suffix)])
                if base:
                    return base, mic
    return None


def _load_cache(path: Path | None) -> dict[str, Resolution]:
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        log.warning("yahoo_symbols.cache_unreadable", path=str(path))
        return {}
    cache: dict[str, Resolution] = {}
    for label, value in raw.items() if isinstance(raw, dict) else []:
        if isinstance(value, list) and len(value) == 2 and all(isinstance(v, str) for v in value):
            cache[str(label)] = (value[0], value[1])
    return cache


def _save_cache(path: Path | None, cache: Mapping[str, Resolution]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {label: list(pair) for label, pair in sorted(cache.items())}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
