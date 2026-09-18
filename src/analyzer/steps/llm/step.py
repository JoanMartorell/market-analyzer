"""Paso 9: Una llamada al LLM con prompt cacheado, vía Batch API si no hay prisa.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class Llm:
    name = "llm"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 9: llm")
