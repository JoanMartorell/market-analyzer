"""Paso reconcile: apertura real vs cierre asumido, cálculo, tabla, servicio y paso."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from typing import Any

import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError, StepStatus
from analyzer.steps.persist import build_signals
from analyzer.steps.reconcile import (
    RECONCILE_STATUSES,
    Reconcile,
    ReconcileReport,
    reconcile_signals,
    run_reconcile,
)
from analyzer.steps.screener.service import CANDIDATE_COLUMNS
from analyzer.storage import RECONCILE_COLUMNS, PriceStore, SignalStore, connect
from core.config import AppConfig, Env

SIGNAL_DAY = date(2026, 9, 17)  # jueves: la señal sale con este cierre
EXECUTION = date(2026, 9, 18)  # viernes: se ejecuta en esta apertura
REGION = "americas"
GAP = 0.02


# --- datos de prueba --------------------------------------------------------------------


def _signal(
    ticker: str = "AAA",
    close: float = 100.0,
    *,
    signal_day: date = SIGNAL_DAY,
    execute_on: date = EXECUTION,
    rule_id: str = "pullback",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "mic": "XNYS",
        "date": signal_day,
        "rule_id": rule_id,
        "close": close,
        "execute_on": execute_on,
    }


def _signals(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [_signal()])


def _candle(ticker: str, open_: float | None, day: date = EXECUTION) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "mic": "XNYS",
        "date": day,
        "open": open_,
        "high": open_,
        "low": open_,
        "close": open_ if open_ is not None else 1.0,
        "adj_close": open_,
        "volume": 1_000,
        "source": "test",
    }


def _candles(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _stored_signals(
    *tickers: str, close: float = 100.0, signal_day: date = SIGNAL_DAY
) -> pd.DataFrame:
    """Señales como las escribe persist, listas para ``replace_day``."""
    candidates = pd.DataFrame(
        [
            {
                "ticker": t,
                "mic": "XNYS",
                "rule_id": "pullback",
                "direction": "long",
                "score": 0.8,
                "conditions_met": "rsi_14 < 35",
                "close": close,
                "sector": "Industrials",
                "rule_hash": "regla-abc",
            }
            for t in tickers
        ],
        columns=[*CANDIDATE_COLUMNS],
    )
    execute_on = {"XNYS": EXECUTION if signal_day == SIGNAL_DAY else date(2026, 9, 17)}
    return build_signals(
        candidates,
        None,
        region_id=REGION,
        as_of=signal_day,
        snapshot_hash="a" * 64,
        execute_on=execute_on,
        model=None,
    )


# --- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Any) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


@pytest.fixture
def stores(cfg_tmp: AppConfig) -> Iterator[tuple[SignalStore, PriceStore]]:
    with connect(cfg_tmp) as con:
        schema = cfg_tmp.settings.data.prod_schema
        yield SignalStore(con, schema), PriceStore(con, schema)


# --- cálculo ----------------------------------------------------------------------------


def test_an_open_within_the_gap_reconciles_the_signal() -> None:
    out = reconcile_signals(_signals(), _candles(_candle("AAA", 101.0)), max_open_gap=GAP)

    assert list(RECONCILE_COLUMNS[:3]) == ["execution_open", "open_gap", "reconcile_status"]
    row = out.iloc[0]
    assert row["execution_open"] == 101.0
    assert row["open_gap"] == pytest.approx(0.01)
    assert row["reconcile_status"] == "conciliada"


def test_a_gap_beyond_the_limit_in_either_direction_deviates() -> None:
    signals = _signals(_signal("UP"), _signal("DOWN"), _signal("EDGE"))
    candles = _candles(_candle("UP", 103.0), _candle("DOWN", 97.0), _candle("EDGE", 102.0))

    out = reconcile_signals(signals, candles, max_open_gap=GAP)

    assert out["reconcile_status"].tolist() == ["desviada", "desviada", "conciliada"]
    assert out["open_gap"].round(4).tolist() == [0.03, -0.03, 0.02]  # el límite no desvía


def test_a_signal_without_a_candle_that_day_has_no_price() -> None:
    signals = _signals(_signal("AAA"), _signal("GONE"), _signal("ZERO"))
    candles = _candles(
        _candle("AAA", 100.0),
        _candle("GONE", 100.0, day=date(2026, 9, 17)),  # otra sesión, no vale
        _candle("ZERO", 0.0),
    )

    out = reconcile_signals(signals, candles, max_open_gap=GAP)

    assert out["reconcile_status"].tolist() == ["conciliada", "sin_precio", "sin_precio"]
    assert out["execution_open"].isna().tolist() == [False, True, True]
    assert out["open_gap"].isna().tolist() == [False, True, True]


def test_signals_from_different_days_use_their_own_execution_candle() -> None:
    monday = date(2026, 9, 21)
    signals = _signals(
        _signal("AAA", 100.0),
        _signal("AAA", 110.0, signal_day=EXECUTION, execute_on=monday),
    )
    candles = _candles(_candle("AAA", 101.0), _candle("AAA", 110.0, day=monday))

    out = reconcile_signals(signals, candles, max_open_gap=GAP)

    assert out["execution_open"].tolist() == [101.0, 110.0]
    assert out["open_gap"].round(4).tolist() == [0.01, 0.0]


def test_previous_reconciliation_columns_are_recomputed() -> None:
    signals = _signals().assign(
        execution_open=1.0, open_gap=-0.99, reconcile_status="desviada", reconciled_at=None
    )

    out = reconcile_signals(signals, _candles(_candle("AAA", 100.0)), max_open_gap=GAP)

    assert out["reconcile_status"].tolist() == ["conciliada"]
    assert "reconciled_at" not in out.columns


def test_the_calculation_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError, match="faltan columnas de las señales: execute_on"):
        reconcile_signals(_signals().drop(columns="execute_on"), _candles(), max_open_gap=GAP)
    with pytest.raises(ValueError, match="faltan columnas de las velas: open"):
        reconcile_signals(
            _signals(), _candles(_candle("AAA", 1.0)).drop(columns="open"), max_open_gap=GAP
        )
    with pytest.raises(ValueError, match="cierre de referencia no positivo: AAA"):
        reconcile_signals(
            _signals(_signal("AAA", 0.0)), _candles(_candle("AAA", 1.0)), max_open_gap=GAP
        )
    with pytest.raises(ValueError, match="max_open_gap"):
        reconcile_signals(_signals(), _candles(_candle("AAA", 1.0)), max_open_gap=1.5)


# --- tabla ------------------------------------------------------------------------------


def test_the_store_writes_the_reconciliation_and_keeps_nulls(
    stores: tuple[SignalStore, PriceStore],
) -> None:
    signals, _ = stores
    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals("AAA", "BBB"))
    reconciled = reconcile_signals(
        signals.load(REGION, SIGNAL_DAY), _candles(_candle("AAA", 103.0)), max_open_gap=GAP
    )

    assert signals.reconcile(reconciled) == 2

    loaded = signals.load(REGION, SIGNAL_DAY).set_index("ticker")
    assert loaded.loc["AAA", "reconcile_status"] == "desviada"
    assert loaded.loc["AAA", "open_gap"] == pytest.approx(0.03)
    assert loaded.loc["BBB", "reconcile_status"] == "sin_precio"
    assert loaded["execution_open"].isna().to_dict() == {"AAA": False, "BBB": True}
    assert loaded["reconciled_at"].notna().all()
    nulls = signals._con.execute(  # se comprueba el NULL real, no un NaN
        f"SELECT count(*) FROM {signals.table} WHERE execution_open IS NULL"
    ).fetchone()
    assert nulls == (1,)


def test_replacing_the_day_clears_the_reconciliation(
    stores: tuple[SignalStore, PriceStore],
) -> None:
    signals, _ = stores
    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals("AAA"))
    signals.reconcile(
        reconcile_signals(
            signals.load(REGION, SIGNAL_DAY), _candles(_candle("AAA", 100.0)), max_open_gap=GAP
        )
    )

    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals("AAA"))

    assert signals.load(REGION, SIGNAL_DAY)["reconcile_status"].isna().all()


def test_pending_takes_the_day_and_whatever_older_was_left_without_status(
    stores: tuple[SignalStore, PriceStore],
) -> None:
    signals, _ = stores
    wednesday = date(2026, 9, 16)
    signals.replace_day(REGION, wednesday, _stored_signals("OLD", "DONE", signal_day=wednesday))
    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals("NEW"))
    done = signals.load(REGION, wednesday)
    done = done[done["ticker"] == "DONE"]
    signals.reconcile(
        reconcile_signals(done, _candles(_candle("DONE", 100.0, day=SIGNAL_DAY)), max_open_gap=GAP)
    )

    pending = signals.pending_reconciliation(REGION, EXECUTION)

    assert pending["ticker"].tolist() == ["OLD", "NEW"]  # DONE ya está; OLD sigue sin estado
    # Repetir el 17 recalcula todas las que se ejecutaban ese día, conciliadas o no.
    assert signals.pending_reconciliation(REGION, SIGNAL_DAY)["ticker"].tolist() == ["DONE", "OLD"]
    assert signals.pending_reconciliation(REGION, wednesday).empty


def test_the_store_refuses_a_frame_without_the_columns(
    stores: tuple[SignalStore, PriceStore],
) -> None:
    signals, _ = stores
    with pytest.raises(ValueError, match="faltan columnas de la conciliación"):
        signals.reconcile(_signals())


# --- servicio ---------------------------------------------------------------------------


def _run(stores: tuple[SignalStore, PriceStore], as_of: date = EXECUTION) -> ReconcileReport:
    signals, prices = stores
    return run_reconcile(REGION, as_of, signals=signals, prices=prices, max_open_gap=GAP)


def test_the_service_reconciles_and_summarises(stores: tuple[SignalStore, PriceStore]) -> None:
    signals, prices = stores
    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals("AAA", "BBB", "CCC", "DDD"))
    prices.upsert(_candles(_candle("AAA", 101.0), _candle("BBB", 103.0), _candle("CCC", 96.0)))

    report = _run(stores)

    assert report.total == 4
    assert report.count("conciliada") == 1
    assert report.count("desviada") == 2
    assert report.count("sin_precio") == 1
    assert report.signal_dates == [SIGNAL_DAY]
    assert report.summary() == (
        f"4 señales del {SIGNAL_DAY}: 1 conciliadas, 2 desviadas más del 2.0% "
        "(CCC -4.0%, BBB +3.0%), 1 sin precio (DDD); gap medio +0.0%, máximo -4.0%"
    )
    saved = signals.load(REGION, SIGNAL_DAY).set_index("ticker")
    assert saved["reconcile_status"].to_dict() == {
        "AAA": "conciliada",
        "BBB": "desviada",
        "CCC": "desviada",
        "DDD": "sin_precio",
    }


def test_the_service_without_pending_signals_says_so(
    stores: tuple[SignalStore, PriceStore],
) -> None:
    report = _run(stores)

    assert report.total == 0
    assert report.summary() == f"sin señales pendientes de conciliar a {EXECUTION}"


def test_the_service_fails_closed_without_the_session_in_prod(
    stores: tuple[SignalStore, PriceStore],
) -> None:
    signals, prices = stores
    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals("AAA"))
    prices.upsert(_candles(_candle("AAA", 100.0, day=SIGNAL_DAY)))  # la de ayer, no la de hoy

    with pytest.raises(ValueError, match=f"sin precios en prod para {EXECUTION}"):
        _run(stores)


def test_the_service_names_at_most_five_deviated(stores: tuple[SignalStore, PriceStore]) -> None:
    signals, prices = stores
    tickers = [f"T{i}" for i in range(7)]
    signals.replace_day(REGION, SIGNAL_DAY, _stored_signals(*tickers))
    prices.upsert(_candles(*(_candle(t, 105.0) for t in tickers)))

    summary = _run(stores).summary()

    assert "7 desviadas más del 2.0% (" in summary
    assert "y 2 más)" in summary


def test_statuses_are_the_three_documented() -> None:
    assert RECONCILE_STATUSES == ("conciliada", "desviada", "sin_precio")


# --- paso -------------------------------------------------------------------------------


def _context(cfg: AppConfig, as_of: date = EXECUTION) -> StepContext:
    return StepContext(cfg=cfg, region=cfg.regions[REGION], as_of=as_of)


def test_the_step_fails_closed_without_prices(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        SignalStore(con, cfg_tmp.settings.data.prod_schema).replace_day(
            REGION, SIGNAL_DAY, _stored_signals("AAA")
        )

    with pytest.raises(StepError, match="conciliación fallida: sin precios en prod"):
        Reconcile().run(_context(cfg_tmp))


def test_the_step_reconciles_yesterdays_signals(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        schema = cfg_tmp.settings.data.prod_schema
        SignalStore(con, schema).replace_day(REGION, SIGNAL_DAY, _stored_signals("AAA", "BBB"))
        PriceStore(con, schema).upsert(_candles(_candle("AAA", 101.0), _candle("BBB", 97.0)))
    ctx = _context(cfg_tmp)

    outcome = Reconcile().run(ctx)

    assert outcome.status is StepStatus.OK
    assert outcome.message == (
        f"2 señales del {SIGNAL_DAY}: 1 conciliadas, 1 desviadas más del 2.0% (BBB -3.0%); "
        "gap medio -1.0%, máximo -3.0%"
    )
    assert ctx.data["reconcile"].count("desviada") == 1


def test_the_step_without_signals_is_ok(cfg_tmp: AppConfig) -> None:
    outcome = Reconcile().run(_context(cfg_tmp))

    assert outcome.status is StepStatus.OK
    assert outcome.message.startswith("sin señales pendientes")
