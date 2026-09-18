"""Paso 3: eventos corporativos. Reingesta el histórico afectado y ajusta la cuarentena.

Deja en ``ctx.data["corporate_actions"]`` un ``CorporateActionsReport``,
reduce ``ctx.data["quarantine"]`` con los valores liberados y publica en
``ctx.data["full_recompute"]`` las claves cuyo histórico cambió, para que
``indicators`` las recalcule desde cero.
"""

from __future__ import annotations

import duckdb

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.steps.corporate_actions.service import reconcile_corporate_actions
from analyzer.steps.ingest.providers import get_price_provider
from analyzer.storage import CorporateActionStore, PriceStore, connect


class CorporateActions:
    name = "corporate_actions"

    def run(self, ctx: StepContext) -> StepOutcome:
        universe = ctx.data.get("universe")
        if universe is None or universe.empty:
            raise StepError("no hay universo en el contexto: el paso universe debe ir antes")
        quarantine: dict[tuple[str, str], str] = dict(ctx.data.get("quarantine", {}))

        try:
            provider = get_price_provider(ctx.region.providers.prices)
        except ValueError as exc:
            raise StepError(str(exc)) from exc

        data = ctx.cfg.settings.data
        try:
            with connect(ctx.cfg) as con:
                report = reconcile_corporate_actions(
                    universe,
                    ctx.as_of,
                    provider=provider,
                    staging=PriceStore(con, data.staging_schema),
                    prod=PriceStore(con, data.prod_schema),
                    actions=CorporateActionStore(con, data.prod_schema),
                    quarantine=quarantine,
                    lookback_days=ctx.cfg.settings.indicators.lookback_days,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"eventos corporativos fallidos: {exc}") from exc

        ctx.data["corporate_actions"] = report
        ctx.data["quarantine"] = {k: r for k, r in quarantine.items() if k not in report.released}
        ctx.data["full_recompute"] = set(report.reingested)
        return StepOutcome(message=report.summary())
