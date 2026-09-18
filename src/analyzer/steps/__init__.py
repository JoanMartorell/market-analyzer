"""Orden de los pasos del pipeline.

Cada paso es un paquete independiente en esta carpeta. El orden vive aquí y
en ningún otro sitio. El paso 11 (entrega) no aparece: lo hace ``core`` con
el paquete ``delivery`` una vez terminado el pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

from analyzer.engine import Step
from analyzer.steps.calendar_gate import CalendarGate
from analyzer.steps.corporate_actions import CorporateActions
from analyzer.steps.dedup_sentiment import DedupSentiment
from analyzer.steps.indicators import Indicators
from analyzer.steps.ingest import Ingest
from analyzer.steps.llm import Llm
from analyzer.steps.news import News
from analyzer.steps.persist import Persist
from analyzer.steps.quality import Quality
from analyzer.steps.reconcile import Reconcile
from analyzer.steps.screener import Screener
from analyzer.steps.snapshot import Snapshot
from analyzer.steps.universe import Universe

DAILY_STEPS: tuple[Callable[[], Step], ...] = (
    CalendarGate,  # 0. ¿hoy cotizó el mercado? si no, fin sin error
    Universe,  # 0b. quién formaba el índice ese día (point-in-time)
    Ingest,  # 1. EOD a staging
    Quality,  # 2. validación; falla en cerrado
    CorporateActions,  # 3. splits y dividendos
    Indicators,  # 4. incremental, últimos ~250 días
    Snapshot,  # 5. point-in-time con hash
    Screener,  # 6. universo -> candidatos
    News,  # 7. solo candidatos y posiciones
    DedupSentiment,  # 8. cascada de dedup + FinBERT
    Llm,  # 9. una llamada, prompt cacheado
    Persist,  # 10. señal + hash de reglas + snapshot
)

RECONCILE_STEPS: tuple[Callable[[], Step], ...] = (Reconcile,)  # 12. al día siguiente


def build_daily_steps() -> list[Step]:
    return [factory() for factory in DAILY_STEPS]


def build_reconcile_steps() -> list[Step]:
    return [factory() for factory in RECONCILE_STEPS]
