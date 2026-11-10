"""core.app: el ciclo diario encadena la conciliación y entrega un solo mensaje."""

from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from analyzer.engine import PipelineResult, StepContext, StepError, StepOutcome, StepStatus
from analyzer.storage import PriceStore, connect
from analyzer.universe import write_constituents
from core import app
from core.config import AppConfig, ConfigError, Env
from core.operations import NO_CHART
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


class SignalStep(FakeStep):
    """Deja en el contexto lo que persist dejaría, para que el mensaje sea el resumen."""

    def run(self, ctx: StepContext) -> StepOutcome:
        ctx.data["signals"] = _signals()
        ctx.data["analysis"] = None
        return super().run(ctx)


def _signals(**overrides: Any) -> pd.DataFrame:
    row: dict[str, Any] = {
        "ticker": "AAA",
        "mic": "XNYS",
        "date": DAY,
        "rule_id": "pullback",
        "direction": "long",
        "score": 0.8,
        "conditions_met": "rsi_14 < 35",
        "close": 10.0,
        "pattern": None,
        "stop_price": None,
        "target_price": None,
        "currency": "USD",
        "execute_on": DAY,
        "verdict": None,
        "confidence": None,
        "headline": None,
        "rationale": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _double_bottom_bars(ticker: str) -> pd.DataFrame:
    """Velas de prod con un doble suelo que rompe en ``DAY``."""
    close = np.interp(np.arange(69), [0, 20, 40, 50, 60, 68], [120, 130, 100, 110, 100, 111.6])
    return pd.DataFrame(
        {
            "ticker": ticker,
            "mic": "XNYS",
            "date": [pd.Timestamp(DAY - timedelta(days=i)) for i in range(68, -1, -1)],
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
            "source": "test",
        }
    )


def test_a_good_cycle_delivers_the_digest_not_the_step_table(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    daily = SignalStep("persist", "1 señal")
    reconcile = FakeStep("reconcile", "sin señales pendientes de conciliar a 2026-09-18")
    outbox, dispatcher = _wire(monkeypatch, daily, reconcile)

    app.run_region(cfg_tmp, "americas", DAY, dispatcher=dispatcher)

    message = outbox.messages[0]
    assert message.subject == "[americas] 2026-09-18: 1 candidato sin veredicto"
    assert "• AAA · 10.00 USD · score 0.80 · rsi_14 < 35" in message.body
    assert "Ayer: sin señales pendientes de conciliar a 2026-09-18" in message.body
    assert "persist" not in message.body  # la tabla de pasos se queda en el log


def test_after_the_digest_each_operation_gets_its_own_message(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox, dispatcher = _wire(monkeypatch, SignalStep("persist", "1 señal"), FakeStep("reconcile"))

    app.run_region(cfg_tmp, "americas", DAY, dispatcher=dispatcher)

    assert [m.subject for m in outbox.messages] == [
        "[americas] 2026-09-18: 1 candidato sin veredicto",
        "[americas] COMPRAR AAA · pullback",
    ]
    # Sin velas en prod no hay gráfico, pero la operación se entrega y lo dice.
    assert outbox.messages[1].image is None
    assert outbox.messages[1].body.endswith(NO_CHART)


def test_operation_messages_can_be_switched_off(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivery = cfg_tmp.settings.delivery.model_copy(update={"notify_operations": False})
    quiet = replace(cfg_tmp, settings=cfg_tmp.settings.model_copy(update={"delivery": delivery}))
    outbox, dispatcher = _wire(monkeypatch, SignalStep("persist", "1 señal"), FakeStep("reconcile"))

    app.run_region(quiet, "americas", DAY, dispatcher=dispatcher)

    assert len(outbox.messages) == 1


def test_the_operation_chart_is_drawn_from_prod_candles(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        PriceStore(con, cfg_tmp.settings.data.prod_schema).upsert(_double_bottom_bars("AAA"))
    signals = _signals(
        rule_id="double_bottom",
        pattern="double_bottom",
        close=111.6,
        stop_price=98.9,
        target_price=121.5,
    )
    result = PipelineResult(region_id="americas", as_of=DAY, reports=[])

    (message,) = app.operation_messages(cfg_tmp, result, {"signals": signals, "analysis": None})

    assert message.subject == "[americas] COMPRAR AAA · doble suelo"
    assert message.image is not None
    assert message.image.startswith(b"\x89PNG")
    assert NO_CHART not in message.body


def test_pattern_stats_follow_the_patterns_of_the_current_universe(cfg_tmp: AppConfig) -> None:
    region = cfg_tmp.regions["americas"]
    members = pd.DataFrame(
        {
            "ticker": ["AAA", "GONE"],  # GONE salió del índice en junio
            "name": ["Alpha", "Gone"],
            "sector": ["IT", "IT"],
            "sub_industry": [None, None],
            "cik": pd.array([None, None], dtype="Int64"),
            "mic": ["XNYS", "XNYS"],
            "currency": ["USD", "USD"],
            "start": pd.to_datetime(pd.Series(["2000-01-03", "2000-01-03"])),
            "end": pd.to_datetime(pd.Series([None, "2026-06-01"])),
            "source": ["test", "test"],
        }
    )
    write_constituents(members, cfg_tmp.constituents_path(region), meta={})
    with pytest.raises(ValueError, match="no hay velas en prod"):
        app.pattern_stats(cfg_tmp, "americas", DAY)

    later = DAY + timedelta(days=1)  # un día después de la ruptura: ya hay apertura de entrada
    bars = _double_bottom_bars("AAA")
    after = bars.iloc[[-1]].assign(date=pd.Timestamp(later))
    outsider = _double_bottom_bars("ZZZ")  # con velas pero fuera del universo: no cuenta
    with connect(cfg_tmp) as con:
        store = PriceStore(con, cfg_tmp.settings.data.prod_schema)
        store.upsert(pd.concat([bars, after, outsider], ignore_index=True))

    stats = app.pattern_stats(cfg_tmp, "americas", later, horizon=10)

    assert stats.keys == 1
    assert stats.events["ticker"].tolist() == ["AAA"]
    assert stats.summary() == "1 patrón en 1 valores: doble suelo 1"

    # Con años se lee staging (donde deja las velas el backfill) y cuentan también
    # los valores que estuvieron en el índice durante el rango.
    with pytest.raises(ValueError, match="no hay velas en staging"):
        app.pattern_stats(cfg_tmp, "americas", later, years=1)
    gone = _double_bottom_bars("GONE")
    with connect(cfg_tmp) as con:
        staging = PriceStore(con, cfg_tmp.settings.data.staging_schema)
        staging.upsert(pd.concat([gone, gone.iloc[[-1]].assign(date=pd.Timestamp(later))]))

    history = app.pattern_stats(cfg_tmp, "americas", later, horizon=10, years=1)

    assert history.events["ticker"].tolist() == ["GONE"]
    assert history.yearly()[["año", "patron", "hallazgos"]].to_numpy().tolist() == [
        [2026, "doble suelo", 1]
    ]


def test_telegram_chats_needs_the_token(cfg_tmp: AppConfig) -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        app.telegram_chats(cfg_tmp)
