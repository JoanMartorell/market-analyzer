"""Paso 8: agrupa las noticias en temas y les pone sentimiento.

Lee de ``prod.news`` lo publicado en las últimas ``cluster_window_hours``
para los valores que interesan, en vez de quedarse con lo que acaba de
descargar el paso anterior: así repetir el ciclo da el mismo resultado
aunque no haya llegado nada nuevo.

Deja en ``ctx.data["dedup"]`` un ``DedupReport`` y en ``ctx.data["clusters"]``
los temas que verá el LLM (vacío si se saltó).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pandas as pd

from analyzer.engine import StepContext, StepError, StepOutcome, StepStatus
from analyzer.steps.dedup_sentiment.service import run_dedup_sentiment
from analyzer.storage import (
    CLUSTER_COLUMNS,
    NewsClusterMemberStore,
    NewsClusterStore,
    NewsStore,
    connect,
)


class DedupSentiment:
    name = "dedup_sentiment"

    def run(self, ctx: StepContext) -> StepOutcome:
        targets = ctx.data.get("news_targets")
        if targets is None:
            raise StepError(
                "no hay objetivos de noticias en el contexto: el paso news debe ir antes"
            )

        ctx.data["clusters"] = pd.DataFrame(columns=[*CLUSTER_COLUMNS])
        if targets.empty:
            return StepOutcome(status=StepStatus.SKIPPED, message="sin valores con noticias")

        settings = ctx.cfg.settings.news
        since = datetime.now(UTC) - timedelta(hours=settings.dedup.cluster_window_hours)
        prod = ctx.cfg.settings.data.prod_schema
        try:
            with connect(ctx.cfg) as con:
                articles = NewsStore(con, prod).load(since=since, keys=targets)
                if articles.empty:
                    hours = settings.dedup.cluster_window_hours
                    return StepOutcome(
                        status=StepStatus.SKIPPED,
                        message=f"sin noticias de las últimas {hours}h",
                    )
                report = run_dedup_sentiment(
                    articles,
                    ctx.as_of,
                    region_id=ctx.region.id,
                    settings=settings,
                    store=NewsClusterStore(con, prod),
                    members=NewsClusterMemberStore(con, prod),
                )
        except (OSError, duckdb.Error, ValueError) as exc:
            raise StepError(f"deduplicación fallida: {exc}") from exc

        ctx.data["dedup"] = report
        ctx.data["clusters"] = report.clusters
        return StepOutcome(message=report.summary())
