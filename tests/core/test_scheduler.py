"""Planificador con reloj simulado: sin esperas reales."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from analyzer.engine import PipelineResult
from core.config import AppConfig
from core.scheduler import next_runs, serve

MADRID = ZoneInfo("Europe/Madrid")
DAY = date(2026, 9, 17)


def _clock(hour: int, minute: int = 0) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hour, minute, tzinfo=MADRID)


def _with_schedule(cfg: AppConfig, run_at: Mapping[str, time]) -> AppConfig:
    """Copia de la configuración con solo las regiones indicadas activas y sus horas.

    La entrega se fuerza a consola: el YAML puede tener telegram, y aquí no hay credenciales.
    """
    regions = {}
    for rid, region in cfg.regions.items():
        if rid in run_at:
            schedule = region.schedule.model_copy(update={"run_at": run_at[rid]})
            regions[rid] = region.model_copy(update={"enabled": True, "schedule": schedule})
        else:
            regions[rid] = region.model_copy(update={"enabled": False})
    delivery = cfg.settings.delivery.model_copy(update={"channels": ["console"]})
    settings = cfg.settings.model_copy(update={"delivery": delivery})
    return replace(cfg, regions=regions, settings=settings)


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


def test_next_runs_sorted_by_time(cfg: AppConfig) -> None:
    cfg = _with_schedule(cfg, {"europe": time(10, 30), "americas": time(10, 0)})
    plan = next_runs(cfg, _clock(9))

    assert [(r.region_id, r.at.time()) for r in plan] == [
        ("americas", time(10, 0)),
        ("europe", time(10, 30)),
    ]


def test_next_runs_keeps_late_run_within_grace(cfg: AppConfig) -> None:
    cfg = _with_schedule(cfg, {"americas": time(10, 0)})

    late = next_runs(cfg, _clock(10, 10))[0]
    assert late.at.date() == DAY  # 10 min tarde, dentro de la gracia de 30

    too_late = next_runs(cfg, _clock(11))[0]
    assert too_late.at.date() == DAY + timedelta(days=1)

    already_done = next_runs(cfg, _clock(10, 10), done={("americas", DAY)})[0]
    assert already_done.at.date() == DAY + timedelta(days=1)


def test_serve_launches_each_region_at_its_time(cfg: AppConfig) -> None:
    cfg = _with_schedule(cfg, {"americas": time(10, 0), "europe": time(10, 30)})
    clock = FakeClock(_clock(9))
    launches: list[tuple[str, date, datetime]] = []

    def runner(region_id: str, as_of: date) -> PipelineResult:
        launches.append((region_id, as_of, clock.now))
        return PipelineResult(region_id=region_id, as_of=as_of)

    completed = serve(cfg, clock=clock, sleep=clock.sleep, runner=runner, max_runs=2)

    assert completed == 2
    assert [(rid, at.time()) for rid, _, at in launches] == [
        ("americas", time(10, 0)),
        ("europe", time(10, 30)),
    ]
    assert launches[0][1] == date(2026, 9, 16)  # sesión USA del día anterior
    assert max(clock.sleeps) <= cfg.settings.scheduler.heartbeat_seconds


def test_serve_survives_a_crashing_run(cfg: AppConfig) -> None:
    cfg = _with_schedule(cfg, {"americas": time(10, 0), "europe": time(10, 30)})
    clock = FakeClock(_clock(9))
    calls: list[str] = []

    def runner(region_id: str, as_of: date) -> PipelineResult:
        calls.append(region_id)
        if region_id == "americas":
            raise RuntimeError("boom")
        return PipelineResult(region_id=region_id, as_of=as_of)

    assert serve(cfg, clock=clock, sleep=clock.sleep, runner=runner, max_runs=2) == 2
    assert calls == ["americas", "europe"]


def test_serve_without_regions_returns_zero(cfg: AppConfig) -> None:
    cfg = _with_schedule(cfg, {})
    assert serve(cfg, clock=FakeClock(_clock(9)), sleep=lambda _s: None, max_runs=1) == 0
