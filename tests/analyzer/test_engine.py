"""El motor: orden, fallo en cerrado y parada sin error."""

from datetime import date

from analyzer.engine import Pipeline, StepContext, StepOutcome, StepStatus
from core.config import AppConfig


class _Ok:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, ctx: StepContext) -> StepOutcome:
        ctx.data.setdefault("trace", []).append(self.name)
        return StepOutcome(message=f"{self.name} hecho")


class _Stop:
    name = "gate"

    def run(self, ctx: StepContext) -> StepOutcome:
        return StepOutcome(StepStatus.SKIPPED, "sin sesión", stop_pipeline=True)


class _Boom:
    name = "boom"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise RuntimeError("feed roto")


class _Todo:
    name = "todo"

    def run(self, ctx: StepContext) -> StepOutcome:
        raise NotImplementedError("pendiente")


def _ctx(cfg: AppConfig) -> StepContext:
    return StepContext(cfg=cfg, region=cfg.regions["americas"], as_of=date(2026, 9, 16))


def test_runs_all_steps_in_order(cfg: AppConfig) -> None:
    ctx = _ctx(cfg)
    result = Pipeline([_Ok("a"), _Ok("b"), _Ok("c")]).run(ctx)

    assert result.ok
    assert not result.stopped_early
    assert ctx.data["trace"] == ["a", "b", "c"]
    assert [r.status for r in result.reports] == [StepStatus.OK] * 3


def test_exception_fails_closed(cfg: AppConfig) -> None:
    ctx = _ctx(cfg)
    result = Pipeline([_Ok("a"), _Boom(), _Ok("c")]).run(ctx)

    assert not result.ok
    assert result.failed_step == "boom"
    assert ctx.data["trace"] == ["a"]  # c nunca corre
    assert "RuntimeError: feed roto" in result.reports[1].message


def test_not_implemented_is_reported_and_stops(cfg: AppConfig) -> None:
    result = Pipeline([_Todo(), _Ok("b")]).run(_ctx(cfg))

    assert not result.ok
    assert result.reports[0].status == StepStatus.NOT_IMPLEMENTED
    assert len(result.reports) == 1


def test_stop_pipeline_is_not_a_failure(cfg: AppConfig) -> None:
    ctx = _ctx(cfg)
    result = Pipeline([_Stop(), _Ok("b")]).run(ctx)

    assert result.ok
    assert result.stopped_early
    assert result.failed_step is None
    assert "trace" not in ctx.data


def test_summary_mentions_every_step(cfg: AppConfig) -> None:
    result = Pipeline([_Ok("a"), _Boom()]).run(_ctx(cfg))
    text = result.summary()

    assert "FALLIDO" in text
    assert "a " in text and "boom" in text
