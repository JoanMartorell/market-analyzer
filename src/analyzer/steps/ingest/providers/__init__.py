"""Proveedores de precios, por id de ``config/sources.yaml``.

Añadir uno es crear su módulo y registrar aquí su fábrica. La región elige
el suyo en ``providers.prices`` y, si quiere, un ``prices_fallback`` que
rescate lo que el principal deje sin la vela del día. El cargador ya
comprobó que existan en sources.yaml; aquí se comprueba que además estén
implementados. Las fábricas reciben la configuración y la conexión porque
algunos proveedores necesitan clave y cupo diario; los que no, las ignoran.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb

from analyzer.steps.ingest.providers.base import (
    ALL_FETCH_COLUMNS,
    FETCH_COLUMNS,
    OPTIONAL_FETCH_COLUMNS,
    PriceProvider,
)
from analyzer.steps.ingest.providers.eodhd import DEFAULT_BASE_URL, DailyBudget, EodhdProvider
from analyzer.steps.ingest.providers.yfinance import YFinanceProvider
from analyzer.storage.api_calls import ApiCallStore
from core.config import AppConfig

Factory = Callable[[AppConfig | None, duckdb.DuckDBPyConnection | None], PriceProvider]

EODHD = "eodhd"


def _yfinance(_cfg: AppConfig | None, _con: duckdb.DuckDBPyConnection | None) -> PriceProvider:
    return YFinanceProvider()


def _eodhd(cfg: AppConfig | None, con: duckdb.DuckDBPyConnection | None) -> PriceProvider:
    if cfg is None or con is None:
        raise ValueError(
            f"{EODHD} necesita la configuración y la base de datos para llevar el cupo diario"
        )
    source = cfg.sources.providers.get(EODHD)
    if source is None:
        raise ValueError(f"{EODHD} no está en sources.yaml")
    env_name = source.api_key_env or "EODHD_API_KEY"
    api_key = cfg.env.get(env_name)
    if api_key is None:
        raise ValueError(f"falta {env_name} en .env")
    budget = None
    if source.daily_call_limit is not None:
        store = ApiCallStore(con, cfg.settings.data.staging_schema)
        budget = DailyBudget(store, EODHD, source.daily_call_limit)
    return EodhdProvider(api_key, base_url=source.base_url or DEFAULT_BASE_URL, budget=budget)


PROVIDERS: dict[str, Factory] = {
    "yfinance": _yfinance,
    EODHD: _eodhd,
}


def get_price_provider(
    provider_id: str,
    *,
    cfg: AppConfig | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> PriceProvider:
    factory = PROVIDERS.get(provider_id)
    if factory is None:
        raise ValueError(
            f"proveedor de precios {provider_id!r} no implementado; "
            f"disponibles: {', '.join(sorted(PROVIDERS))}"
        )
    return factory(cfg, con)


__all__ = [
    "ALL_FETCH_COLUMNS",
    "FETCH_COLUMNS",
    "OPTIONAL_FETCH_COLUMNS",
    "PROVIDERS",
    "DailyBudget",
    "EodhdProvider",
    "PriceProvider",
    "YFinanceProvider",
    "get_price_provider",
]
