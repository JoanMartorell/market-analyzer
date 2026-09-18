"""Gate de calendario contra el calendario real de NYSE."""

from datetime import date

import pytest

from analyzer.engine import StepContext, StepStatus
from analyzer.steps.calendar_gate import CalendarGate
from core.config import AppConfig


def _run(cfg: AppConfig, day: date) -> tuple[StepContext, StepStatus, bool]:
    ctx = StepContext(cfg=cfg, region=cfg.regions["americas"], as_of=day)
    outcome = CalendarGate().run(ctx)
    return ctx, outcome.status, outcome.stop_pipeline


def test_weekday_session_passes(cfg: AppConfig) -> None:
    ctx, status, stop = _run(cfg, date(2026, 9, 16))  # miércoles

    assert status == StepStatus.OK
    assert not stop
    assert set(ctx.data["open_markets"]) == {"XNYS", "XNAS"}


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 9, 13),  # sábado
        date(2026, 7, 3),  # 4 de julio cae en sábado: NYSE cierra el viernes 3
        date(2026, 12, 25),  # Navidad
    ],
)
def test_no_session_stops_without_error(cfg: AppConfig, day: date) -> None:
    ctx, status, stop = _run(cfg, day)

    assert status == StepStatus.SKIPPED
    assert stop
    assert "open_markets" not in ctx.data
