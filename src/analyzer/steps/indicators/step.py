"""Paso 4: Indicadores incrementales sobre los últimos ~250 días.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class Indicators:
    name = "indicators"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 4: indicators")
