"""Paso 10: Guarda señal + hash de reglas + snapshot que la generó.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class Persist:
    name = "persist"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 10: persist")
