"""Proveedores de precios, por id de ``config/sources.yaml``.

Añadir uno es crear su módulo y registrarlo aquí. La región elige el suyo en
``providers.prices``; el cargador ya comprobó que exista en sources.yaml, y
aquí se comprueba que además esté implementado.
"""

from __future__ import annotations

from collections.abc import Callable

from analyzer.steps.ingest.providers.base import (
    ALL_FETCH_COLUMNS,
    FETCH_COLUMNS,
    OPTIONAL_FETCH_COLUMNS,
    PriceProvider,
)
from analyzer.steps.ingest.providers.yfinance import YFinanceProvider

PROVIDERS: dict[str, Callable[[], PriceProvider]] = {
    "yfinance": YFinanceProvider,
}


def get_price_provider(provider_id: str) -> PriceProvider:
    factory = PROVIDERS.get(provider_id)
    if factory is None:
        raise ValueError(
            f"proveedor de precios {provider_id!r} no implementado; "
            f"disponibles: {', '.join(sorted(PROVIDERS))}"
        )
    return factory()


__all__ = [
    "ALL_FETCH_COLUMNS",
    "FETCH_COLUMNS",
    "OPTIONAL_FETCH_COLUMNS",
    "PROVIDERS",
    "PriceProvider",
    "YFinanceProvider",
    "get_price_provider",
]
