"""core.app: el ciclo diario encadena la conciliación y entrega un solo mensaje."""

from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from analyzer.engine import StepContext, StepError, StepOutcome, StepStatus
from core import app
from core.config import AppConfig, Env
from delivery import Dispatcher, Message

DAY = date(2026, 9, 18)


class FakeStep:
    def __init__(self, name: str, message: str = "", *, stop: bool = False, fail: bool = False):
        self.name = name
        self.message = message
        self.stop = stop
        self.fail = fail
        self.seen: list[date] = []

    def run(self, ctx: StepContext) -> StepOutcome:
        self.seen.append(ctx.as_of)
        if self.fail:
            raise StepError(self.message)
        status = StepStatus.SKIPPED if self.stop else StepStatus.OK
        return StepOutcome(status=status, message=self.message, stop_pipeline=self.stop)


class Outbox:
    name = "outbox"

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def send(self, message: Message) -> None:
        self.messages.append(message)


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Any) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def _wire(
    monkeypatch: pytest.MonkeyPatch, daily: FakeStep, reconcile: FakeStep
) -> tuple[Outbox, Dispatcher]:
    monkeypatch.setattr(app, "build_daily_steps", lambda: [daily])
    monkeypatch.setattr(app, "build_reconcile_steps", lambda: [reconcile])
    outbox = Outbox()
    return outbox, Dispatcher([outbox])


def test_the_cycle_reconciles_the_session_and_reports_both(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = FakeStep("persist", "11 señales para 2026-09-21")
    reconcile = FakeStep("reconcile", "11 señales del 2026-09-17: 11 conciliadas")
    outbox, dispatcher = _wire(monkeypatch, daily, reconcile)

    result = app.run_region(cfg_tmp, "americas", DAY, dispatcher=dispatcher)

    assert result.ok
    assert [r.name for r in result.reports] == ["persist", "reconcile"]
    assert reconcile.seen == [DAY]  # concilia la misma sesión que acaba de traer el ciclo
    assert len(outbox.messages) == 1
    assert "11 conciliadas" in outbox.messages[0].body


def test_a_cycle_that_stops_early_does_not_reconcile(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = FakeStep("calendar_gate", "sin sesión", stop=True)
    reconcile = FakeStep("reconcile")
    _, dispatcher = _wire(monkeypatch, daily, reconcile)

    result = app.run_region(cfg_tmp, "americas", DAY, dispatcher=dispatcher)

    assert result.stopped_early
    assert [r.name for r in result.reports] == ["calendar_gate"]
    assert reconcile.seen == []


def test_a_failed_cycle_does_not_reconcile(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = FakeStep("quality", "cobertura 0.90 < 0.98", fail=True)
    reconcile = FakeStep("reconcile")
    outbox, dispatcher = _wire(monkeypatch, daily, reconcile)

    result = app.run_region(cfg_tmp, "americas", DAY, dispatcher=dispatcher)

    assert not result.ok
    assert reconcile.seen == []
    assert outbox.messages[0].subject.endswith("pipeline fallido en quality")


def test_a_failed_reconciliation_fails_the_cycle(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = FakeStep("persist", "ok")
    reconcile = FakeStep("reconcile", "sin precios en prod", fail=True)
    outbox, dispatcher = _wire(monkeypatch, daily, reconcile)

    result = app.run_region(cfg_tmp, "americas", DAY, dispatcher=dispatcher)

    assert result.failed_step == "reconcile"
    assert outbox.messages[0].severity == "error"


def test_reconcile_region_runs_only_step_twelve(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = FakeStep("persist")
    reconcile = FakeStep("reconcile", "sin señales pendientes")
    _wire(monkeypatch, daily, reconcile)

    result = app.reconcile_region(cfg_tmp, "americas", DAY)

    assert [r.name for r in result.reports] == ["reconcile"]
    assert daily.seen == []
