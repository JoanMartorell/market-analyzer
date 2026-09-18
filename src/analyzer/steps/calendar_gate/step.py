"""Paso 0: gate de calendario.

Festivos, medias sesiones y cierres especiales los resuelve exchange_calendars.
Nunca ``weekday() < 5``. Si ningún mercado de la región tuvo sesión, el
pipeline termina sin error y sin señales.

Deja en ``ctx.data["open_markets"]`` la lista de MIC con sesión ese día.
"""

from __future__ import annotations

from datetime import timedelta

import exchange_calendars as xcals
import pandas as pd

from analyzer.engine import StepContext, StepOutcome, StepStatus

_MARGIN = timedelta(days=10)  # ventana mínima para construir el calendario


class CalendarGate:
    name = "calendar_gate"

    def run(self, ctx: StepContext) -> StepOutcome:
        session = pd.Timestamp(ctx.as_of)
        start, end = ctx.as_of - _MARGIN, ctx.as_of + _MARGIN

        open_markets = [
            market.mic
            for market in ctx.region.markets
            if xcals.get_calendar(market.calendar, start=str(start), end=str(end)).is_session(
                session
            )
        ]

        if not open_markets:
            return StepOutcome(
                status=StepStatus.SKIPPED,
                message=f"{ctx.as_of}: sin sesión en {ctx.region.id}",
                stop_pipeline=True,
            )

        ctx.data["open_markets"] = open_markets
        return StepOutcome(message=f"sesión en {', '.join(open_markets)}")
