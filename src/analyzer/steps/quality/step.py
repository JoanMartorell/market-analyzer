"""Paso 2: validación de calidad. Falla en cerrado.

Comprueba que el feed llega hasta ``as_of``, que la cobertura del universo
supera el mínimo y que las velas del día son coherentes. Los valores con
anomalías quedan en cuarentena ese día. Si el día pasa, promueve la ventana
de staging a prod; si no, prod se queda como estaba y el pipeline se detiene.

Deja en ``ctx.data["quality"]`` un ``QualityReport`` y en
``ctx.data["quarantine"]`` un dict clave -> motivo que los pasos siguientes
deben respetar (``corporate_actions`` puede reducirlo con evidencia).
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.quality.service import run_quality
from analyzer.storage import PriceStore, connect


class Quality:
    name = "quality"

    def run(self, ctx: StepContext) -> StepOutcome:
        universe = ctx.data.get("universe")
        if universe is None or universe.empty:
            raise StepError("no hay universo en el contexto: el paso universe debe ir antes")

        data = ctx.cfg.settings.data
        try:
            with connect(ctx.cfg) as con:
                report = run_quality(
                    universe,
                    ctx.as_of,
                    staging=PriceStore(con, data.staging_schema),
                    prod=PriceStore(con, data.prod_schema),
                    settings=ctx.cfg.settings.quality,
                    lookback_days=ctx.cfg.settings.indicators.lookback_days,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"validación fallida: {exc}") from exc

        ctx.data["quality"] = report
        # clave -> motivo; corporate_actions puede levantar cuarentenas con evidencia
        ctx.data["quarantine"] = {a.key: a.reason for a in report.anomalies}
        if not report.ok:
            raise StepError(report.summary())
        return StepOutcome(message=report.summary())
