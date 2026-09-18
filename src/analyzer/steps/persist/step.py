"""Paso 10: guarda las señales del día con el hash de la regla y del panel.

Necesita a los tres pasos que deciden: ``screener`` (candidatos), ``snapshot``
(hash del panel) y ``llm`` (análisis, o ``None`` si se saltó). Un día sin
candidatos también pasa por aquí: retira las señales que hubiera dejado una
ejecución anterior del mismo día.

Deja en ``ctx.data["persist"]`` un ``PersistReport`` y en
``ctx.data["signals"]`` el DataFrame guardado, que es lo que se entrega.
"""

from __future__ import annotations

from datetime import date

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.sessions import next_session
from analyzer.steps.persist.service import run_persist
from analyzer.storage import SignalStore, connect


class Persist:
    name = "persist"

    def run(self, ctx: StepContext) -> StepOutcome:
        candidates = ctx.data.get("candidates")
        if candidates is None:
            raise StepError("no hay candidatos en el contexto: el paso screener debe ir antes")
        snapshot = ctx.data.get("snapshot")
        if snapshot is None:
            raise StepError("no hay panel en el contexto: el paso snapshot debe ir antes")
        if "analysis" not in ctx.data:
            raise StepError("no hay análisis en el contexto: el paso llm debe ir antes")

        analysis = ctx.data["analysis"]
        llm = ctx.data.get("llm")
        try:
            execute_on = {
                str(mic): self._execution(ctx, str(mic)) for mic in candidates["mic"].unique()
            }
            with connect(ctx.cfg) as con:
                report = run_persist(
                    candidates,
                    analysis,
                    ctx.as_of,
                    region_id=ctx.region.id,
                    snapshot_hash=str(snapshot.snapshot_hash),
                    execute_on=execute_on,
                    model=None if llm is None else str(llm.model),
                    store=SignalStore(con, ctx.cfg.settings.data.prod_schema),
                    panel=ctx.data.get("panel"),
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"persistencia fallida: {exc}") from exc

        ctx.data["persist"] = report
        ctx.data["signals"] = report.signals
        return StepOutcome(message=report.summary())

    @staticmethod
    def _execution(ctx: StepContext, mic: str) -> date:
        session = next_session(ctx.region, ctx.as_of, mic)
        if session is None:
            raise StepError(f"sin sesión posterior a {ctx.as_of} en {mic}: calendario roto")
        return session
