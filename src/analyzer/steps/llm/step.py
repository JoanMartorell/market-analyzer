"""Paso 9: una llamada al modelo con los candidatos del día y sus noticias.

Es el único paso que gasta dinero, así que se salta en cuanto no hace falta:
sin candidatos no hay nada que analizar, y con ``llm.enabled`` a falso o
``max_calls_per_cycle`` a cero el pipeline sigue siendo puramente técnico.

Deja en ``ctx.data["llm"]`` un ``LlmReport`` y en ``ctx.data["analysis"]``
el ``Analysis`` validado que usarán el paso de persistencia y la entrega
(``None`` si se saltó).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from analyzer.engine import StepContext, StepError, StepOutcome, StepStatus
from analyzer.steps.llm.client import LlmError, build_client
from analyzer.steps.llm.service import run_llm
from analyzer.storage import CLUSTER_COLUMNS, LlmCallStore, connect

SKIPPED = StepStatus.SKIPPED


class Llm:
    name = "llm"

    def run(self, ctx: StepContext) -> StepOutcome:
        candidates = ctx.data.get("candidates")
        if candidates is None:
            raise StepError("no hay candidatos en el contexto: el paso screener debe ir antes")

        ctx.data["analysis"] = None
        settings = ctx.cfg.settings.llm
        if not settings.enabled:
            return StepOutcome(status=SKIPPED, message="desactivado en settings (llm.enabled)")
        if settings.max_calls_per_cycle == 0:
            return StepOutcome(status=SKIPPED, message="llm.max_calls_per_cycle es 0")
        if candidates.empty:
            return StepOutcome(status=SKIPPED, message="sin candidatos: nada que analizar")

        api_key = ctx.cfg.env.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise StepError("falta ANTHROPIC_API_KEY y el análisis del modelo está activado")

        prompt = self._prompt(ctx)
        clusters = ctx.data.get("clusters")
        if clusters is None:
            clusters = pd.DataFrame(columns=[*CLUSTER_COLUMNS])

        try:
            with connect(ctx.cfg) as con:
                report = run_llm(
                    candidates,
                    clusters,
                    ctx.as_of,
                    region_id=ctx.region.id,
                    currency=ctx.cfg.settings.base_currency,
                    settings=settings,
                    prompt=prompt,
                    client=build_client(settings, api_key),
                    store=LlmCallStore(con, ctx.cfg.settings.data.prod_schema),
                    panel=ctx.data.get("panel"),
                )
        except (OSError, duckdb.Error, ValueError, LlmError) as exc:
            raise StepError(f"análisis del modelo fallido: {exc}") from exc

        ctx.data["llm"] = report
        ctx.data["analysis"] = report.analysis
        return StepOutcome(message=report.summary())

    @staticmethod
    def _prompt(ctx: StepContext) -> str:
        path = ctx.cfg.prompt_path
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise StepError(f"no se puede leer el prompt {path}: {exc}") from exc
        if not text:
            raise StepError(f"el prompt {path} está vacío")
        return text
