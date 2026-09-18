"""Paso 7: Descarga noticias solo de candidatos y posiciones abiertas.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class News:
    name = "news"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 7: news")
