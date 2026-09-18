"""Paso 5: Congela el estado point-in-time del universo con fecha y hash.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class Snapshot:
    name = "snapshot"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 5: snapshot")
