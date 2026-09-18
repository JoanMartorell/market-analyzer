"""Enriquecedor: bolsa de cotización (MIC) y CIK desde la SEC.

``company_tickers_exchange.json`` cubre todo valor registrado en EE. UU.,
así que este enriquecedor servirá igual para otros índices americanos.
La SEC exige un User-Agent con contacto; sin él no se consulta.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import structlog

from analyzer.universe.base import FetchContext, Notes
from analyzer.universe.helpers import canonical_ticker, merge_fill

SEC_EXCHANGES_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_EXCHANGE_TO_MIC = {"Nasdaq": "XNAS", "NYSE": "XNYS"}  # OTC y CBOE: fuera del universo

log = structlog.get_logger(__name__)


class SecExchanges:
    name = "sec:company_tickers_exchange"

    def enrich(
        self, table: pd.DataFrame, client: httpx.Client, ctx: FetchContext
    ) -> tuple[pd.DataFrame, Notes]:
        if not ctx.user_agent:
            log.warning("universe.sec_skipped", detail="sin SEC_EDGAR_USER_AGENT; mic queda vacío")
            return table, {"consulted": False, "reason": "sin SEC_EDGAR_USER_AGENT"}
        response = client.get(SEC_EXCHANGES_URL, headers={"User-Agent": ctx.user_agent})
        response.raise_for_status()
        return apply_sec(table, parse_sec_json(response.json()))


def parse_sec_json(payload: dict[str, Any]) -> pd.DataFrame:
    """``{"fields": [...], "data": [[...]]}`` -> DataFrame[ticker, cik, mic]."""
    raw = pd.DataFrame(payload["data"], columns=payload["fields"])
    out = pd.DataFrame(
        {
            "ticker": raw["ticker"].map(canonical_ticker),
            "cik": pd.to_numeric(raw["cik"], errors="coerce").astype("Int64"),
            "mic": raw["exchange"].map(_EXCHANGE_TO_MIC),
        }
    )
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def apply_sec(table: pd.DataFrame, sec: pd.DataFrame) -> tuple[pd.DataFrame, Notes]:
    """Lógica pura: rellena mic y cik por ticker sin pisar lo que ya haya."""
    table = merge_fill(table, sec, ["mic", "cik"])
    current = table["end"].isna()
    notes: Notes = {
        "consulted": True,
        "current_without_mic": int(table.loc[current, "mic"].isna().sum()),
    }
    return table, notes
