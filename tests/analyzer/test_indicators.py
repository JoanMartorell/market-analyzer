"""Paso indicators: catálogo, selección incremental de filas, servicio y paso."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.indicators import (
    INDICATOR_COLUMNS,
    WARMUP,
    Indicators,
    compute_for_key,
    compute_indicators,
    run_indicators,
    select_rows,
)
from analyzer.steps.indicators.service import HISTORY, POSITION
from analyzer.storage import IndicatorStore, PriceStore, connect
from core.config import AppConfig, Env

AS_OF = date(2026, 9, 17)


def _bars(ticker: str, days: int = 260, *, end: date = AS_OF, mic: str = "XNAS") -> pd.DataFrame:
    """``days`` velas hasta ``end`` con un paseo aleatorio reproducible por ticker."""
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    close = 100 + rng.standard_normal(days).cumsum()
    dates = [pd.Timestamp(end - timedelta(days=i)) for i in range(days - 1, -1, -1)]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "mic": mic,
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
            "source": "test",
        }
    )


# --- catálogo ----------------------------------------------------------------


def test_compute_for_key_returns_catalog_columns_with_expected_warmup() -> None:
    values = compute_for_key(_bars("AAPL", 260))

    assert tuple(values.columns) == INDICATOR_COLUMNS
    assert values["sma_200"].isna().sum() == 199
    assert values["sma_200"].iloc[199:].notna().all()
    assert values["rsi_14"].dropna().between(0, 100).all()
    assert values["macd_hist"].iloc[-1] == pytest.approx(
        values["macd"].iloc[-1] - values["macd_signal"].iloc[-1]
    )
    assert values["avg_volume_20"].iloc[-1] == pytest.approx(1_000_000.0)


def test_compute_for_key_handles_history_shorter_than_periods() -> None:
    values = compute_for_key(_bars("NEW", 10))  # pandas-ta devuelve None por debajo del periodo

    assert tuple(values.columns) == INDICATOR_COLUMNS
    assert len(values) == 10
    assert values["sma_200"].isna().all()
    assert values["macd"].isna().all()
    assert values["ret_1d"].notna().sum() == 9


def test_atr_uses_adjusted_scale() -> None:
    bars = _bars("SPL", 40)
    bars["adj_close"] = bars["close"] / 10  # split 10:1 aplicado a todo el histórico
    values = compute_for_key(bars)

    assert values["atr_14"].iloc[-1] == pytest.approx(0.2, abs=0.05)  # rango 2 en escala /10


# --- selección de filas ----------------------------------------------------------


def _stored(**last: date) -> pd.DataFrame:
    rows = [{"ticker": t, "mic": "XNAS", "last_date": d} for t, d in last.items()]
    return pd.DataFrame(rows, columns=["ticker", "mic", "last_date"])


def test_select_rows_first_run_keeps_only_complete_rows() -> None:
    computed = compute_indicators(_bars("AAPL", 260))

    kept = select_rows(computed, _stored(), set(), warmup=WARMUP)

    assert len(kept) == 260 - WARMUP
    assert kept["sma_200"].notna().all()
    assert kept["date"].max() == pd.Timestamp(AS_OF)


def test_select_rows_incremental_appends_after_last_stored() -> None:
    computed = compute_indicators(_bars("AAPL", 260))
    stored = _stored(AAPL=AS_OF - timedelta(days=2))

    kept = select_rows(computed, stored, set(), warmup=WARMUP)

    assert kept["date"].dt.date.tolist() == [AS_OF - timedelta(days=1), AS_OF]


def test_select_rows_forced_key_rewrites_everything_complete() -> None:
    computed = compute_indicators(pd.concat([_bars("AAPL", 260), _bars("MSFT", 260)]))
    stored = _stored(AAPL=AS_OF, MSFT=AS_OF)

    kept = select_rows(computed, stored, {("AAPL", "XNAS")}, warmup=WARMUP)

    assert set(kept["ticker"]) == {"AAPL"}
    assert len(kept) == 260 - WARMUP


def test_select_rows_short_history_is_kept_entirely() -> None:
    computed = compute_indicators(_bars("IPO", 30))

    kept = select_rows(computed, _stored(), set(), warmup=WARMUP)

    assert len(kept) == 30
    assert (kept[HISTORY] == 30).all()
    assert kept[POSITION].tolist() == list(range(30))


# --- servicio -------------------------------------------------------------------


@pytest.fixture
def stores() -> tuple[PriceStore, IndicatorStore]:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA prod")
    return PriceStore(con, "prod"), IndicatorStore(con, "prod", INDICATOR_COLUMNS)


def _universe(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers), "mic": ["XNAS"] * len(tickers)})


def test_run_indicators_is_incremental_and_recomputes_forced_keys(
    stores: tuple[PriceStore, IndicatorStore],
) -> None:
    prices, store = stores
    prices.upsert(pd.concat([_bars("AAPL", 260), _bars("MSFT", 260), _bars("IPO", 30)]))
    universe = _universe("AAPL", "MSFT", "IPO")

    first = run_indicators(universe, AS_OF, prices=prices, store=store, lookback_days=250)
    assert (
        first.summary()
        == "3/3 valores con indicadores, 152 filas, 3 desde cero, 1 con histórico corto"
    )
    assert first.with_today == 3

    again = run_indicators(universe, AS_OF, prices=prices, store=store, lookback_days=250)
    assert again.rows == 0
    assert again.recomputed == 0

    # Un día más de velas: se escribe solo la fila nueva de cada valor.
    tomorrow = AS_OF + timedelta(days=1)
    prices.upsert(
        pd.concat(
            [
                _bars("AAPL", 261, end=tomorrow),
                _bars("MSFT", 261, end=tomorrow),
                _bars("IPO", 31, end=tomorrow),
            ]
        )
    )
    daily = run_indicators(universe, tomorrow, prices=prices, store=store, lookback_days=250)
    assert daily.rows == 3
    assert daily.summary() == "3/3 valores con indicadores, 3 filas, 1 con histórico corto"

    # Histórico reescrito por un split: se recalcula el valor entero.
    forced = run_indicators(
        universe,
        tomorrow,
        prices=prices,
        store=store,
        lookback_days=250,
        full_recompute={("AAPL", "XNAS")},
    )
    assert forced.recomputed == 1
    assert forced.rows == 261 - WARMUP

    aapl = store.load(keys=_universe("AAPL"))
    assert aapl["date"].max() == pd.Timestamp(tomorrow)
    assert aapl["sma_200"].notna().all()
    assert aapl["close"].notna().all()


def test_run_indicators_fails_closed_without_today(
    stores: tuple[PriceStore, IndicatorStore],
) -> None:
    prices, store = stores
    prices.upsert(_bars("AAPL", 260, end=AS_OF - timedelta(days=1)))

    with pytest.raises(ValueError, match="ninguna vela de 2026-09-17"):
        run_indicators(_universe("AAPL"), AS_OF, prices=prices, store=store, lookback_days=250)


def test_indicator_store_migrates_new_catalog_columns() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA prod")
    IndicatorStore(con, "prod", ("rsi_14",))
    wider = IndicatorStore(con, "prod", ("rsi_14", "brand_new"))  # un indicador añadido después

    assert wider.load().columns.tolist()[-3:] == ["rsi_14", "brand_new", "computed_at"]
    with pytest.raises(ValueError, match="reservado"):
        IndicatorStore(con, "prod", ("close",))


# --- paso -----------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def test_step_reads_prod_and_honours_full_recompute(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        PriceStore(con, "prod").upsert(_bars("AAPL", 260))
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["universe"] = _universe("AAPL")
    ctx.data["full_recompute"] = {("AAPL", "XNAS")}

    outcome = Indicators().run(ctx)

    assert outcome.message.startswith("1/1 valores con indicadores")
    assert ctx.data["indicators"].recomputed == 1


def test_step_fails_closed_when_prod_is_empty(cfg_tmp: AppConfig) -> None:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["universe"] = _universe("AAPL")

    with pytest.raises(StepError, match="no hay velas en prod"):
        Indicators().run(ctx)
