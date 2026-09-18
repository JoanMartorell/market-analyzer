"""Motor de pasos.

Reglas:
- Los pasos se ejecutan en orden y comparten un ``StepContext``.
- Se falla en cerrado: la primera excepción detiene el pipeline. Nunca se
  sigue con datos a medias.
- Un paso puede detener el pipeline sin error (``stop_pipeline=True``), por
  ejemplo el gate de calendario cuando hoy no hubo sesión.
- ``NotImplementedError`` se reporta como ``not_implemented`` y también detiene.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, Protocol

import structlog

from core.config import AppConfig, Region

log = structlog.get_logger(__name__)


class StepError(Exception):
    """Fallo esperado de un paso con mensaje para el usuario (datos ausentes, feed roto...)."""


class StepStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass
class StepContext:
    """Estado compartido entre pasos de una ejecución."""

    cfg: AppConfig
    region: Region
    as_of: date
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepOutcome:
    status: StepStatus = StepStatus.OK
    message: str = ""
    stop_pipeline: bool = False


class Step(Protocol):
    name: str

    def run(self, ctx: StepContext) -> StepOutcome: ...


@dataclass(frozen=True)
class StepReport:
    name: str
    status: StepStatus
    message: str
    seconds: float


@dataclass
class PipelineResult:
    region_id: str
    as_of: date
    reports: list[StepReport] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def ok(self) -> bool:
        return all(r.status in (StepStatus.OK, StepStatus.SKIPPED) for r in self.reports)

    @property
    def failed_step(self) -> str | None:
        for r in self.reports:
            if r.status in (StepStatus.FAILED, StepStatus.NOT_IMPLEMENTED):
                return r.name
        return None

    @property
    def total_seconds(self) -> float:
        return sum(r.seconds for r in self.reports)

    def summary(self) -> str:
        lines = [f"{self.region_id} {self.as_of}: {'ok' if self.ok else 'FALLIDO'}"]
        for r in self.reports:
            detail = f"  {r.message}" if r.message else ""
            lines.append(f"  {r.status.value:16} {r.name:20} {r.seconds:6.2f}s{detail}")
        lines.append(f"  total {self.total_seconds:.2f}s")
        return "\n".join(lines)


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self._steps]

    def run(self, ctx: StepContext) -> PipelineResult:
        result = PipelineResult(region_id=ctx.region.id, as_of=ctx.as_of)
        for step in self._steps:
            report = self._run_step(step, ctx)
            result.reports.append(report)
            if report.status in (StepStatus.FAILED, StepStatus.NOT_IMPLEMENTED):
                break
            if _stopped(step, ctx, report):
                result.stopped_early = True
                break
        return result

    @staticmethod
    def _run_step(step: Step, ctx: StepContext) -> StepReport:
        started = time.perf_counter()
        slog = log.bind(step=step.name, region=ctx.region.id, as_of=str(ctx.as_of))
        try:
            outcome = step.run(ctx)
        except NotImplementedError as exc:
            slog.warning("step.not_implemented", detail=str(exc))
            return StepReport(step.name, StepStatus.NOT_IMPLEMENTED, str(exc), _elapsed(started))
        except StepError as exc:
            # Fallo esperado con mensaje para el usuario: sin traceback.
            slog.error("step.failed", detail=str(exc))
            return StepReport(step.name, StepStatus.FAILED, str(exc), _elapsed(started))
        except Exception as exc:
            slog.exception("step.crashed")
            return StepReport(
                step.name, StepStatus.FAILED, f"{type(exc).__name__}: {exc}", _elapsed(started)
            )

        ctx.data[_outcome_key(step)] = outcome
        slog.info("step.done", status=outcome.status.value, message=outcome.message)
        return StepReport(step.name, outcome.status, outcome.message, _elapsed(started))


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def _outcome_key(step: Step) -> str:
    return f"__outcome__{step.name}"


def _stopped(step: Step, ctx: StepContext, report: StepReport) -> bool:
    outcome = ctx.data.get(_outcome_key(step))
    return isinstance(outcome, StepOutcome) and outcome.stop_pipeline
