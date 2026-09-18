"""Planificador residente: ``analyzer serve``.

Lee ``schedule.run_at`` de cada región activa, espera hasta la más próxima,
lanza su ciclo y vuelve a planificar. Todo queda en el log con hora:
plan, espera, lanzamiento, resultado.

Reloj y ``sleep`` son inyectables para poder probarlo sin esperar.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Set
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog

from analyzer.engine import PipelineResult
from analyzer.sessions import last_closed_session
from core.app import build_delivery, run_region
from core.config import AppConfig

log = structlog.get_logger(__name__)

Clock = Callable[[], datetime]
Sleep = Callable[[float], None]
Runner = Callable[[str, date], PipelineResult]
RunKey = tuple[str, date]


@dataclass(frozen=True, order=True)
class ScheduledRun:
    at: datetime
    region_id: str


def next_runs(cfg: AppConfig, now: datetime, done: Set[RunKey] = frozenset()) -> list[ScheduledRun]:
    """Próximo lanzamiento de cada región activa, ordenados por hora.

    Un ``run_at`` de hoy que ya pasó sigue contando si está dentro del
    periodo de gracia y no se ha ejecutado: así arrancar el servidor unos
    minutos tarde no pierde el ciclo del día.
    """
    grace = timedelta(minutes=cfg.settings.scheduler.late_grace_minutes)
    runs: list[ScheduledRun] = []
    for region in cfg.enabled_regions():
        today_at = datetime.combine(now.date(), region.schedule.run_at, tzinfo=now.tzinfo)
        pending_today = (region.id, now.date()) not in done and today_at + grace > now
        runs.append(
            ScheduledRun(today_at if pending_today else today_at + timedelta(days=1), region.id)
        )
    return sorted(runs)


def serve(
    cfg: AppConfig,
    *,
    clock: Clock | None = None,
    sleep: Sleep = time.sleep,
    runner: Runner | None = None,
    max_runs: int | None = None,
) -> int:
    """Bucle principal. Devuelve el número de ciclos lanzados (útil con ``max_runs``)."""
    tz = ZoneInfo(cfg.settings.local_timezone)
    now_local: Clock = clock if clock is not None else (lambda: datetime.now(tz))
    if runner is None:
        dispatcher = build_delivery(cfg)  # falla ya si faltan credenciales de entrega

        def runner(region_id: str, as_of: date) -> PipelineResult:
            return run_region(cfg, region_id, as_of, dispatcher=dispatcher)

    regions = cfg.enabled_regions()
    if not regions:
        log.error("scheduler.no_regions", detail="ninguna región con enabled: true")
        return 0

    heartbeat = cfg.settings.scheduler.heartbeat_seconds
    done: set[RunKey] = set()
    completed = 0
    log.info(
        "scheduler.start",
        tz=cfg.settings.local_timezone,
        regions={r.id: r.schedule.run_at.strftime("%H:%M") for r in regions},
    )

    while max_runs is None or completed < max_runs:
        plan = next_runs(cfg, now_local(), done)
        for scheduled in plan:
            log.info("scheduler.plan", region=scheduled.region_id, at=_fmt(scheduled.at))
        upcoming = plan[0]

        _wait_until(upcoming, now_local, sleep, heartbeat)

        launched_at = now_local()
        done.add((upcoming.region_id, upcoming.at.date()))
        region = cfg.regions[upcoming.region_id]
        as_of = last_closed_session(region, launched_at)
        if as_of is None:
            log.warning("scheduler.no_session", region=region.id, at=_fmt(launched_at))
            continue

        log.info(
            "scheduler.launch",
            region=region.id,
            at=_fmt(launched_at),
            as_of=str(as_of),
            late_seconds=round((launched_at - upcoming.at).total_seconds()),
        )
        try:
            result = runner(region.id, as_of)
        except Exception:
            log.exception("scheduler.run_crashed", region=region.id)
        else:
            log.info(
                "scheduler.done",
                region=region.id,
                at=_fmt(now_local()),
                ok=result.ok,
                stopped=result.stopped_early,
                failed_step=result.failed_step,
                seconds=round(result.total_seconds, 2),
            )
        completed += 1

    log.info("scheduler.stop", runs=completed)
    return completed


def _wait_until(scheduled: ScheduledRun, clock: Clock, sleep: Sleep, heartbeat: int) -> None:
    while True:
        remaining = (scheduled.at - clock()).total_seconds()
        if remaining <= 0:
            return
        log.info(
            "scheduler.waiting",
            region=scheduled.region_id,
            at=_fmt(scheduled.at),
            remaining_min=round(remaining / 60, 1),
        )
        sleep(min(remaining, heartbeat))


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")
