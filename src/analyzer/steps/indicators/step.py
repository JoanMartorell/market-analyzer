"""Paso 4: indicadores incrementales sobre ``prod.prices``.

Deja en ``ctx.data["indicators"]`` un ``IndicatorsReport``. Respeta
``ctx.data["full_recompute"]`` (claves cuyo histórico reescribió
``corporate_actions``) si ``full_recompute_on_corporate_action`` está activo.
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.indicators.registry import INDICATOR_COLUMNS
from analyzer.steps.indicators.service import run_indicators
from analyzer.storage import IndicatorStore, PriceStore, connect


class Indicators:
    name = "indicators"

    def run(self, ctx: StepContext) -> StepOutcome:
        universe = ctx.data.get("universe")
        if universe is None or universe.empty:
            raise StepError("no hay universo en el contexto: el paso universe debe ir antes")

        settings = ctx.cfg.settings.indicators
        forced: set[tuple[str, str]] = set()
        if settings.full_recompute_on_corporate_action:
            forced = set(ctx.data.get("full_recompute", ()))

        data = ctx.cfg.settings.data
        try:
            with connect(ctx.cfg) as con:
                report = run_indicators(
                    universe,
                    ctx.as_of,
                    prices=PriceStore(con, data.prod_schema),
                    store=IndicatorStore(con, data.prod_schema, INDICATOR_COLUMNS),
                    lookback_days=settings.lookback_days,
                    full_recompute=forced,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"indicadores fallidos: {exc}") from exc

        ctx.data["indicators"] = report
        return StepOutcome(message=report.summary())
