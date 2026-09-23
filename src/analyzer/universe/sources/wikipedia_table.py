"""Fuente genérica: tablas de componentes en páginas de Wikipedia.

Solo da la composición actual, sin fechas. Se usa con ``history=False`` para
que ``build`` acumule el histórico a partir del primer build. Cada índice
describe su tabla con una ``WikipediaTable``; un universo puede unir varias
(por ejemplo, cuatro índices nacionales para Asia-Pacífico).

Las columnas se pueden dar como un nombre o como varios candidatos: los
editores de Wikipedia renombran cabeceras ("Ticker", "Symbol", "Code") y así
un cambio de nombre no deja la región sin universo. Si ninguna tabla tiene
columnas reconocibles, el error lista las cabeceras encontradas.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import httpx
import pandas as pd

from analyzer.universe.base import FetchContext
from analyzer.universe.helpers import DEFAULT_USER_AGENT, canonical_ticker

TickerCleaner = Callable[[object], str | None]
Columns = str | tuple[str, ...]  # un nombre, o candidatos en orden de preferencia


@dataclass(frozen=True)
class WikipediaTable:
    index_id: str
    url: str
    ticker_column: Columns
    name_column: Columns
    sector_column: Columns | None = None
    # Bolsa y divisa fijas para toda la tabla...
    mic: str | None = None
    currency: str | None = None
    # ...o derivadas de una columna (p. ej. "Country" en el STOXX 600).
    market_column: Columns | None = None
    market_map: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    clean_ticker: TickerCleaner = canonical_ticker
    concat_all: bool = False  # la lista está repartida en varias tablas (Nikkei por sector)


class WikipediaSnapshot:
    """Une una o varias tablas de Wikipedia en una composición actual."""

    def __init__(self, name: str, tables: Sequence[WikipediaTable]) -> None:
        self.name = name
        self.tables = tuple(tables)

    def fetch(self, client: httpx.Client, ctx: FetchContext) -> pd.DataFrame:
        headers = {"User-Agent": ctx.user_agent or DEFAULT_USER_AGENT}
        frames = []
        for spec in self.tables:
            response = client.get(spec.url, headers=headers)
            response.raise_for_status()
            frames.append(parse_wikipedia_table(response.text, spec))
        table = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "mic"])
        table["start"] = pd.NaT
        table["end"] = pd.NaT
        table["source"] = "wikipedia:" + table["index_id"]
        return table.drop(columns="index_id").reset_index(drop=True)


def parse_wikipedia_table(html: str, spec: WikipediaTable) -> pd.DataFrame:
    """HTML -> DataFrame[ticker, name, sector, mic, currency, index_id]."""
    tables = pd.read_html(io.StringIO(html))
    matches = [t for t in tables if _pick(t, spec.ticker_column) and _pick(t, spec.name_column)]
    if not matches:
        found = sorted({str(c) for t in tables for c in t.columns})
        raise ValueError(
            f"{spec.index_id}: no hay tabla con columnas {_names(spec.ticker_column)} y "
            f"{_names(spec.name_column)} en {spec.url}; cabeceras encontradas: {found}"
        )
    raw = pd.concat(matches if spec.concat_all else [max(matches, key=len)], ignore_index=True)

    ticker_col = _pick(raw, spec.ticker_column)
    name_col = _pick(raw, spec.name_column)
    if ticker_col is None or name_col is None:  # imposible: ``matches`` ya lo comprobó
        raise ValueError(f"{spec.index_id}: columnas de ticker o nombre ausentes")
    out = pd.DataFrame(
        {
            "ticker": raw[ticker_col].map(spec.clean_ticker),
            "name": raw[name_col].astype(str).str.strip(),
        }
    )
    sector_col = _pick(raw, spec.sector_column) if spec.sector_column else None
    out["sector"] = raw[sector_col].astype(str).str.strip() if sector_col else None
    if spec.market_column:
        market_col = _pick(raw, spec.market_column)
        if market_col is None:
            raise ValueError(f"{spec.index_id}: falta la columna {_names(spec.market_column)}")
        markets = raw[market_col].astype(str).str.strip().map(spec.market_map)
        out["mic"] = markets.map(lambda m: m[0] if isinstance(m, tuple) else None)
        out["currency"] = markets.map(lambda m: m[1] if isinstance(m, tuple) else None)
    else:
        out["mic"] = spec.mic
        out["currency"] = spec.currency
    out["index_id"] = spec.index_id
    return out.dropna(subset=["ticker"]).drop_duplicates(["ticker", "mic"]).reset_index(drop=True)


def _pick(table: pd.DataFrame, wanted: Columns | None) -> str | None:
    """Primera columna de ``wanted`` presente en ``table``, o ``None``."""
    if wanted is None:
        return None
    present = {str(c) for c in table.columns}
    return next((c for c in _names(wanted) if c in present), None)


def _names(wanted: Columns) -> list[str]:
    return [wanted] if isinstance(wanted, str) else list(wanted)


def code_ticker(width: int) -> TickerCleaner:
    """Limpiador para códigos numéricos o alfanuméricos de ancho fijo.

    ``"SEHK: 5"`` -> ``"0005"`` (Hong Kong, 4), ``"090430"`` (Corea, 6),
    ``"285A"`` (Tokio, 4; los códigos nuevos llevan letra).
    """

    def clean(raw: object) -> str | None:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        text = str(raw).split(":")[-1]
        code = re.sub(r"[^0-9A-Za-z]", "", text).upper()
        if code.isdigit():
            code = code.zfill(width)
        return code if len(code) == width else None

    return clean


def exchange_code_ticker(raw: object) -> str | None:
    """Código de bolsa sin prefijo ni separadores: ``"BMV: GFNORTE O"`` -> ``"GFNORTEO"``.

    Es como Yahoo escribe B3 y la BMV (``PETR4.SA``, ``GFNORTEO.MX``,
    ``PE&OLES.MX``): la serie va pegada al nombre, sin espacio ni punto.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    code = re.sub(r"[^0-9A-Za-z&]", "", str(raw).split(":")[-1]).upper()
    return code or None
