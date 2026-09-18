"""Paso quality: comprobaciones puras, servicio con promoción y paso en el pipeline."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.quality import Quality, find_anomalies, run_quality
from analyzer.storage import PriceStore, connect
from core.config import AppConfig, Env
from core.config.schema import QualitySettings

AS_OF = date(2026, 9, 17)


def _series(
    ticker: str,
    mic: str,
    days: int = 30,
    *,
    last_close: float | None = None,
    last_volume: float | None = None,
    end: date = AS_OF,
    volume: float = 2_000_000.0,
) -> pd.DataFrame:
    """``days`` velas planas a 100 hasta ``end``; la última se puede alterar."""
    dates = [pd.Timestamp(end - timedelta(days=i)) for i in range(days - 1, -1, -1)]
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "mic": mic,
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": volume,
            "source": "test",
        }
    )
    if last_close is not None:
        frame.loc[frame.index[-1], ["close", "adj_close", "high", "low"]] = [
            last_close,
            last_close,
            max(last_close, 101.0),
            min(last_close, 99.0),
        ]
    if last_volume is not None:
        frame.loc[frame.index[-1], "volume"] = last_volume
    return frame


def _window(*parts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True)


# --- comprobaciones puras ------------------------------------------------------


def _anomalies(window: pd.DataFrame) -> dict[str, str]:
    found = find_anomalies(window, AS_OF, max_daily_jump=0.4, zero_volume_min_avg_volume=1_000_000)
    return {a.ticker: a.reason for a in found}


def test_clean_window_has_no_anomalies() -> None:
    assert _anomalies(_window(_series("AAPL", "XNAS"), _series("MSFT", "XNAS"))) == {}


def test_jump_is_flagged_but_small_move_is_not() -> None:
    window = _window(
        _series("JUMP", "XNAS", last_close=150.0), _series("OK", "XNAS", last_close=130.0)
    )
    assert _anomalies(window) == {"JUMP": "jump"}


def test_zero_volume_only_flagged_in_liquid_names() -> None:
    window = _window(
        _series("LIQ", "XNAS", last_volume=0.0),
        _series("THIN", "XNAS", last_volume=0.0, volume=50_000.0),
    )
    assert _anomalies(window) == {"LIQ": "zero_volume"}


def test_invalid_ohlc_is_flagged() -> None:
    bad = _series("BAD", "XNAS")
    bad.loc[bad.index[-1], "close"] = 0.0
    assert _anomalies(_window(bad)) == {"BAD": "invalid_ohlc"}


def test_split_adjusted_by_provider_is_not_a_jump() -> None:
    # Yahoo ya ajusta adj_close: el cierre real cae a la mitad, el ajustado no se mueve.
    split = _series("SPL", "XNAS")
    split.loc[split.index[-1], ["close", "high", "low"]] = [50.0, 50.5, 49.5]
    assert _anomalies(_window(split)) == {}


# --- servicio ----------------------------------------------------------------


@pytest.fixture
def stores() -> tuple[PriceStore, PriceStore]:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    con.execute("CREATE SCHEMA prod")
    return PriceStore(con, "staging"), PriceStore(con, "prod")


def _universe(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers), "mic": ["XNAS"] * len(tickers)})


def test_run_quality_promotes_without_quarantined_bars(
    stores: tuple[PriceStore, PriceStore],
) -> None:
    staging, prod = stores
    staging.upsert(_window(_series("AAPL", "XNAS"), _series("JUMP", "XNAS", last_close=200.0)))
    settings = QualitySettings(min_ticker_coverage=0.5)

    report = run_quality(
        _universe("AAPL", "JUMP"),
        AS_OF,
        staging=staging,
        prod=prod,
        settings=settings,
        lookback_days=50,
    )

    assert report.ok
    assert report.covered == 2
    assert report.quarantined == 1
    assert report.quarantined_keys() == {("JUMP", "XNAS")}
    assert report.promoted_rows == 59  # 60 velas menos la de JUMP en as_of
    promoted = prod.load()
    assert len(promoted) == 59
    assert promoted[(promoted["ticker"] == "JUMP")]["date"].max() == pd.Timestamp(
        AS_OF - timedelta(days=1)
    )

    # idempotente
    again = run_quality(
        _universe("AAPL", "JUMP"),
        AS_OF,
        staging=staging,
        prod=prod,
        settings=settings,
        lookback_days=50,
    )
    assert again.ok
    assert len(prod.load()) == 59


def test_run_quality_fails_on_low_coverage_and_leaves_prod_untouched(
    stores: tuple[PriceStore, PriceStore],
) -> None:
    staging, prod = stores
    staging.upsert(_series("AAPL", "XNAS"))

    report = run_quality(
        _universe("AAPL", "MSFT", "GOOG"),
        AS_OF,
        staging=staging,
        prod=prod,
        settings=QualitySettings(),
        lookback_days=50,
    )

    assert not report.ok
    assert any("mínimo 98%" in f for f in report.failures)
    assert report.promoted_rows == 0
    assert prod.stats()["rows"] == 0


def test_run_quality_fails_on_stale_feed(stores: tuple[PriceStore, PriceStore]) -> None:
    staging, prod = stores
    staging.upsert(_series("AAPL", "XNAS", end=AS_OF - timedelta(days=1)))

    report = run_quality(
        _universe("AAPL"),
        AS_OF,
        staging=staging,
        prod=prod,
        settings=QualitySettings(),
        lookback_days=50,
    )

    assert not report.ok
    assert any("feed llega hasta 2026-09-16" in f for f in report.failures)


def test_quarantine_can_sink_coverage(stores: tuple[PriceStore, PriceStore]) -> None:
    staging, prod = stores
    staging.upsert(_window(_series("AAPL", "XNAS"), _series("JUMP", "XNAS", last_close=200.0)))

    report = run_quality(
        _universe("AAPL", "JUMP"),
        AS_OF,
        staging=staging,
        prod=prod,
        settings=QualitySettings(),
        lookback_days=50,
    )

    assert report.usable == 1
    assert not report.ok


# --- paso --------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def test_step_promotes_and_reports(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        PriceStore(con, "staging").upsert(_window(_series("AAPL", "XNAS"), _series("MSFT", "XNAS")))
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["universe"] = _universe("AAPL", "MSFT")

    outcome = Quality().run(ctx)

    assert "cobertura 100.0%" in outcome.message
    assert ctx.data["quality"].ok
    with connect(cfg_tmp) as con:
        assert PriceStore(con, "prod").stats()["rows"] == 60


def test_step_fails_closed_with_message(cfg_tmp: AppConfig) -> None:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["universe"] = _universe("AAPL")

    with pytest.raises(StepError, match=r"cobertura 0.0%"):
        Quality().run(ctx)
    assert not ctx.data["quality"].ok


def test_step_requires_universe(cfg_tmp: AppConfig) -> None:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    with pytest.raises(StepError, match="universo"):
        Quality().run(ctx)
