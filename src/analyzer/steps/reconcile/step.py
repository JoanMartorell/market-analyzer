"""Paso 12: apertura real vs cierre asumido. Corre al día siguiente, fuera del ciclo diario.

Su ``as_of`` es la sesión de ejecución, no la de las señales: concilia las
señales que se ejecutaban ese día (y las anteriores que quedaran sin
conciliar) con las velas que el ciclo diario de esa sesión dejó en prod.
Por eso ``core`` lo lanza justo después del ciclo, como pipeline aparte, y
``analyzer reconcile`` permite repetirlo a mano.

Deja en ``ctx.data["reconcile"]`` un ``ReconcileReport``.
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.reconcile.service import run_reconcile
from analyzer.storage import PriceStore, SignalStore, connect


class Reconcile:
    name = "reconcile"

    def run(self, ctx: StepContext) -> StepOutcome:
        data = ctx.cfg.settings.data
        try:
            with connect(ctx.cfg) as con:
                report = run_reconcile(
                    ctx.region.id,
                    ctx.as_of,
                    signals=SignalStore(con, data.prod_schema),
                    prices=PriceStore(con, data.prod_schema),
                    max_open_gap=ctx.cfg.settings.reconcile.max_open_gap,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"conciliación fallida: {exc}") from exc

        ctx.data["reconcile"] = report
        return StepOutcome(message=report.summary())
