"""Paso 6: Evalúa las reglas YAML sobre el panel completo: universo -> candidatos con score.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class Screener:
    name = "screener"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 6: screener")
