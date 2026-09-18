"""Paso 5: congela el panel point-in-time del universo con fecha y hash.

Deja en ``ctx.data["snapshot"]`` un ``SnapshotReport`` y en
``ctx.data["panel"]`` el DataFrame que verá el screener.
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.indicators.registry import INDICATOR_COLUMNS
from analyzer.steps.snapshot.service import run_snapshot
from analyzer.storage import IndicatorStore, SnapshotRunStore, SnapshotStore, connect


class Snapshot:
    name = "snapshot"

    def run(self, ctx: StepContext) -> StepOutcome:
        universe = ctx.data.get("universe")
        if universe is None or universe.empty:
            raise StepError("no hay universo en el contexto: el paso universe debe ir antes")

        prod = ctx.cfg.settings.data.prod_schema
        try:
            with connect(ctx.cfg) as con:
                report = run_snapshot(
                    universe,
                    ctx.as_of,
                    region_id=ctx.region.id,
                    indicators=IndicatorStore(con, prod, INDICATOR_COLUMNS),
                    store=SnapshotStore(con, prod, INDICATOR_COLUMNS),
                    runs=SnapshotRunStore(con, prod),
                    min_coverage=ctx.cfg.settings.snapshot.min_coverage,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"snapshot fallido: {exc}") from exc

        ctx.data["snapshot"] = report
        ctx.data["panel"] = report.panel
        return StepOutcome(message=report.summary())
