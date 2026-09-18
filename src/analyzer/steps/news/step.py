"""Paso 7: descarga noticias solo de candidatos y posiciones abiertas.

Objetivos: los candidatos del screener más ``ctx.data["open_positions"]``
(ticker, mic) si algún paso anterior lo deja; hoy no hay gestor de
posiciones y llega vacío. Con ``only_candidates_and_open_positions`` a
falso se piden noticias de todo el panel. Sin objetivos, o en una región
sin fuentes (``technical_only``), el paso se salta sin error.

Deja en ``ctx.data["news"]`` un ``NewsReport`` y en ``ctx.data["articles"]``
el DataFrame de artículos de esta ejecución (vacío si se saltó).
"""

from __future__ import annotations

import duckdb
import pandas as pd
import structlog

from analyzer.engine import StepContext, StepError, StepOutcome, StepStatus
from analyzer.steps.news.service import SourceResult, run_news
from analyzer.steps.news.sources import ArticleSource, build_source, new_client
from analyzer.storage import NEWS_COLUMNS, NewsStore, connect

log = structlog.get_logger(__name__)


class News:
    name = "news"

    def run(self, ctx: StepContext) -> StepOutcome:
        candidates = ctx.data.get("candidates")
        if candidates is None:
            raise StepError("no hay candidatos en el contexto: el paso screener debe ir antes")

        targets = self._targets(ctx, candidates)
        ctx.data["news_targets"] = targets  # el paso dedup_sentiment pide por los mismos
        ctx.data["articles"] = pd.DataFrame(columns=list(NEWS_COLUMNS))
        if targets.empty:
            return StepOutcome(
                status=StepStatus.SKIPPED,
                message="sin candidatos ni posiciones abiertas: no se descargan noticias",
            )
        if not ctx.region.providers.news:
            return StepOutcome(status=StepStatus.SKIPPED, message="región sin fuentes de noticias")

        # Una fuente sin clave o sin implementar no para el día: se anota y se
        # sigue con las demás. config-check ya avisa de las claves que faltan.
        sources: list[ArticleSource] = []
        unavailable: list[SourceResult] = []
        for sid in ctx.region.providers.news:
            try:
                sources.append(build_source(sid, ctx.cfg.sources.news_sources[sid], ctx.cfg.env))
            except ValueError as exc:
                log.warning("news.source_unavailable", source=sid, detail=str(exc))
                reason = "sin clave" if "necesita" in str(exc) else "sin implementar"
                unavailable.append(SourceResult(sid, 0, reason=reason, error=str(exc)))
        if not sources:
            raise StepError(
                "ninguna fuente de noticias disponible: "
                + "; ".join(u.error or "" for u in unavailable)
            )

        try:
            with new_client() as http, connect(ctx.cfg) as con:
                report = run_news(
                    targets,
                    ctx.as_of,
                    sources=sources,
                    store=NewsStore(con, ctx.cfg.settings.data.prod_schema),
                    http=http,
                    lookback_hours=ctx.cfg.settings.news.lookback_hours,
                    unavailable=unavailable,
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"noticias fallidas: {exc}") from exc

        ctx.data["news"] = report
        ctx.data["articles"] = report.articles
        return StepOutcome(message=report.summary())

    @staticmethod
    def _targets(ctx: StepContext, candidates: pd.DataFrame) -> pd.DataFrame:
        if not ctx.cfg.settings.news.only_candidates_and_open_positions:
            panel: pd.DataFrame | None = ctx.data.get("panel")
            if panel is None:
                raise StepError("no hay panel en el contexto para pedir noticias de todo")
            return panel.loc[:, ["ticker", "mic"]]
        parts = [candidates.loc[:, ["ticker", "mic"]]]
        positions = ctx.data.get("open_positions")
        if isinstance(positions, pd.DataFrame) and not positions.empty:
            parts.append(positions.loc[:, ["ticker", "mic"]])
        targets: pd.DataFrame = pd.concat(parts, ignore_index=True).drop_duplicates()
        return targets
