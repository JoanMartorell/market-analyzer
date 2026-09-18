"""Paso 6: evalúa las reglas de la región sobre el panel y deja los candidatos.

Deja en ``ctx.data["screener"]`` un ``ScreenerReport`` y en
``ctx.data["candidates"]`` el DataFrame de candidatos (puede estar vacío:
un día sin candidatos es un resultado, no un fallo).
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.indicators.registry import INDICATOR_COLUMNS
from analyzer.steps.screener.service import run_screener
from analyzer.storage import IndicatorStore, connect


class Screener:
    name = "screener"

    def run(self, ctx: StepContext) -> StepOutcome:
        panel = ctx.data.get("panel")
        if panel is None or panel.empty:
            raise StepError("no hay panel en el contexto: el paso snapshot debe ir antes")

        settings = ctx.cfg.settings.screener
        prod = ctx.cfg.settings.data.prod_schema
        try:
            with connect(ctx.cfg) as con:
                report = run_screener(
                    panel,
                    ctx.as_of,
                    rules=ctx.cfg.rules_for(ctx.region),
                    indicators=IndicatorStore(con, prod, INDICATOR_COLUMNS),
                    max_candidates=settings.max_candidates_per_region,
                    min_score=settings.min_score,
                    min_avg_volume=ctx.region.universe.min_avg_volume_20,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"screener fallido: {exc}") from exc

        ctx.data["screener"] = report
        ctx.data["candidates"] = report.candidates
        return StepOutcome(message=report.summary())
