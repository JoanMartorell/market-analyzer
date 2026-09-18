"""Paso 8: Cascada de deduplicación (hash, MinHash, embeddings) y FinBERT local.

Pendiente de implementar. Mientras tanto el motor lo reporta como
``not_implemented`` y detiene el pipeline, que es lo que debe pasar:
nunca se emiten señales con un paso a medias.
"""

from analyzer.engine import StepContext, StepOutcome


class DedupSentiment:
    name = "dedup_sentiment"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("paso 8: dedup_sentiment")
