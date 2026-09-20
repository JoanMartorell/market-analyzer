"""Paso 1: ingesta EOD a la tabla de staging, nunca directo a producción.

Toma el universo que dejó el paso anterior, pide al proveedor de precios de
la región las velas que faltan hasta ``as_of`` y las escribe en
``staging.prices``. Si la región tiene ``prices_fallback``, lo que el
principal deje sin la vela del día se pide a ese segundo proveedor, solo
para los mercados que tuvieron sesión. No valida nada: eso es ``quality``.
Solo falla si no llega absolutamente nada, que es un feed roto y no una
cobertura baja.

Deja en ``ctx.data["ingest"]`` un ``IngestReport``.
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.ingest.providers import PriceProvider, get_price_provider
from analyzer.steps.ingest.service import ingest_prices
from analyzer.storage import PriceStore, connect


class Ingest:
    name = "ingest"

    def run(self, ctx: StepContext) -> StepOutcome:
        universe = ctx.data.get("universe")
        if universe is None or universe.empty:
            raise StepError("no hay universo en el contexto: el paso universe debe ir antes")

        try:
            with connect(ctx.cfg) as con:
                provider, fallback = _providers(ctx, con)
                store = PriceStore(con, ctx.cfg.settings.data.staging_schema)
                report = ingest_prices(
                    universe,
                    ctx.as_of,
                    provider=provider,
                    store=store,
                    lookback_days=ctx.cfg.settings.indicators.lookback_days,
                    fallback=fallback,
                    fallback_markets=ctx.data.get("open_markets"),
                )
        except StepError:
            raise
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"ingesta fallida: {exc}") from exc

        # Feed roto = no llega nada y tampoco había nada al día. Que unos pocos
        # valores crónicos sin datos no lleguen es cobertura, y eso lo juzga quality.
        if report.requested and not report.received and not report.up_to_date:
            raise StepError(
                f"{provider.name} no devolvió datos para ninguno de los "
                f"{report.requested} valores pedidos"
            )

        ctx.data["ingest"] = report
        return StepOutcome(message=report.summary())


def _providers(
    ctx: StepContext, con: duckdb.DuckDBPyConnection
) -> tuple[PriceProvider, PriceProvider | None]:
    """El proveedor principal de la región y, si lo tiene, el de rescate."""
    providers = ctx.region.providers
    try:
        provider = get_price_provider(providers.prices, cfg=ctx.cfg, con=con)
        fallback = (
            get_price_provider(providers.prices_fallback, cfg=ctx.cfg, con=con)
            if providers.prices_fallback
            else None
        )
    except ValueError as exc:
        raise StepError(str(exc)) from exc
    return provider, fallback
