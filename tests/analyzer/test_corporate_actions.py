"""Paso corporate_actions: detectores puros, reconciliación con reingesta y paso."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analyzer.engine import StepContext
from analyzer.steps.corporate_actions import (
    KIND_DIVIDEND,
    KIND_SPLIT,
    CorporateActions,
    detect_from_feed,
    detect_splits_by_ratio,
    needs_reingest,
    reconcile_corporate_actions,
)
from analyzer.steps.ingest.providers.base import ALL_FETCH_COLUMNS
from analyzer.storage import CorporateActionStore, PriceStore, connect
from core.config import AppConfig, Env

AS_OF = date(2026, 9, 17)
OLD_LOAD = datetime(2026, 1, 1, tzinfo=UTC)
PROVIDER_TARGET = "analyzer.steps.corporate_actions.step.get_price_provider"


def _bars(
    ticker: str,
    days: int = 30,
    *,
    close: float = 100.0,
    end: date = AS_OF,
    mic: str = "XNAS",
    with_feed: bool = True,
) -> pd.DataFrame:
    dates = [pd.Timestamp(end - timedelta(days=i)) for i in range(days - 1, -1, -1)]
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "mic": mic,
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
            "source": "test",
            "dividend": 0.0 if with_feed else float("nan"),
            "split": 0.0 if with_feed else float("nan"),
            "ingested_at": pd.Timestamp("2026-09-18 08:00:00"),
        }
    )
    return frame


def _set_last(frame: pd.DataFrame, **values: float) -> pd.DataFrame:
    for column, value in values.items():
        frame.loc[frame.index[-1], column] = value
    return frame


# --- detectores --------------------------------------------------------------


def test_detect_from_feed_reads_split_and_dividend_columns() -> None:
    nvda = _set_last(_bars("NVDA"), split=10.0, close=12.0, adj_close=12.0)
    aapl = _set_last(_bars("AAPL"), dividend=0.25)
    found = detect_from_feed(pd.concat([nvda, aapl, _bars("MSFT")], ignore_index=True))

    assert [(a.ticker, a.kind, a.value) for a in found] == [
        ("AAPL", KIND_DIVIDEND, 0.25),
        ("NVDA", KIND_SPLIT, 10.0),
    ]
    assert found[1].describe() == "NVDA split 10:1 el 2026-09-17"


def test_ratio_detector_only_without_feed_and_only_on_clean_ratios() -> None:
    halved = _set_last(_bars("SPL", with_feed=False), close=50.0, adj_close=50.0)
    real_move = _set_last(_bars("MOVE", with_feed=False), close=70.0, adj_close=70.0)
    with_feed = _set_last(_bars("FED"), close=50.0, adj_close=50.0)  # split=0.0: el feed manda
    reverse = _set_last(_bars("REV", with_feed=False), close=1000.0, adj_close=1000.0)

    found = detect_splits_by_ratio(
        pd.concat([halved, real_move, with_feed, reverse], ignore_index=True), AS_OF
    )

    assert {a.ticker: a.value for a in found} == {"SPL": 2.0, "REV": 0.1}
    assert all(a.source == "ratio" for a in found)


def test_needs_reingest_only_when_history_predates_the_event_load() -> None:
    stale = _set_last(_bars("NVDA"), split=10.0)
    stale.loc[stale.index[:-1], "ingested_at"] = pd.Timestamp("2026-01-01")
    fresh = _set_last(_bars("AAPL"), split=4.0)
    window = pd.concat([stale, fresh], ignore_index=True)
    actions = detect_from_feed(window)

    by_ticker = {a.ticker: needs_reingest(window, a) for a in actions}
    assert by_ticker == {"NVDA": True, "AAPL": False}


# --- reconciliación ------------------------------------------------------------


class FakeProvider:
    """Devuelve un histórico ya ajustado por el split 10:1 de NVDA."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def fetch(self, keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        self.calls.append(keys["ticker"].tolist())
        parts = [
            _set_last(_bars(t, close=10.0, mic=m), split=10.0)
            for t, m in zip(keys["ticker"], keys["mic"], strict=True)
        ]
        out = pd.concat(parts, ignore_index=True)
        return out.loc[:, list(ALL_FETCH_COLUMNS)]


@pytest.fixture
def stores() -> tuple[PriceStore, PriceStore, CorporateActionStore]:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    con.execute("CREATE SCHEMA prod")
    return PriceStore(con, "staging"), PriceStore(con, "prod"), CorporateActionStore(con, "prod")


def _universe(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers), "mic": ["XNAS"] * len(tickers)})


def test_reconcile_reingests_releases_and_records(
    stores: tuple[PriceStore, PriceStore, CorporateActionStore],
) -> None:
    staging, prod, actions = stores
    # Histórico antiguo sin ajustar (cierre 100) cargado en enero...
    history = _bars("NVDA", days=29, end=AS_OF - timedelta(days=1))
    staging.upsert(history, ingested_at=OLD_LOAD)
    prod.upsert(history, ingested_at=OLD_LOAD)
    # ...y la vela de hoy con el split, a 10, cargada ahora. quality la puso en cuarentena.
    today = _set_last(_bars("NVDA", days=1, close=10.0), split=10.0)
    staging.upsert(today)
    staging.upsert(_bars("AAPL"))
    prod.upsert(_bars("AAPL"))
    provider = FakeProvider()

    report = reconcile_corporate_actions(
        _universe("NVDA", "AAPL"),
        AS_OF,
        provider=provider,
        staging=staging,
        prod=prod,
        actions=actions,
        quarantine={("NVDA", "XNAS"): "jump"},
        lookback_days=50,
    )

    assert [a.describe() for a in report.detected] == ["NVDA split 10:1 el 2026-09-17"]
    assert report.reingested == (("NVDA", "XNAS"),)
    assert report.released == (("NVDA", "XNAS"),)
    assert provider.calls == [["NVDA"]]
    assert (
        "1 splits, 0 dividendos, 1 valores reingestados, 1 fuera de cuarentena" == report.summary()
    )

    nvda_prod = prod.load(keys=_universe("NVDA"))
    assert nvda_prod["close"].eq(10.0).all()  # histórico coherente, sin escalón
    assert nvda_prod["date"].max() == pd.Timestamp(AS_OF)  # la vela del día ya está en prod
    recorded = actions.load()
    assert recorded[["ticker", "kind", "value"]].to_numpy().tolist() == [["NVDA", "split", 10.0]]

    # Al día siguiente el evento ya está registrado: nada que hacer.
    again = reconcile_corporate_actions(
        _universe("NVDA", "AAPL"),
        AS_OF,
        provider=provider,
        staging=staging,
        prod=prod,
        actions=actions,
        quarantine={},
        lookback_days=50,
    )
    assert again.summary() == "sin eventos nuevos"
    assert provider.calls == [["NVDA"]]


def test_reconcile_keeps_unexplained_jump_in_quarantine(
    stores: tuple[PriceStore, PriceStore, CorporateActionStore],
) -> None:
    staging, prod, actions = stores
    staging.upsert(_set_last(_bars("BIO"), close=160.0, adj_close=160.0))  # +60% real, sin split
    provider = FakeProvider()

    report = reconcile_corporate_actions(
        _universe("BIO"),
        AS_OF,
        provider=provider,
        staging=staging,
        prod=prod,
        actions=actions,
        quarantine={("BIO", "XNAS"): "jump"},
        lookback_days=50,
    )

    assert report.released == ()
    assert provider.calls == []
    assert prod.stats()["rows"] == 0


def test_dividend_on_fresh_history_is_recorded_without_reingest(
    stores: tuple[PriceStore, PriceStore, CorporateActionStore],
) -> None:
    staging, prod, actions = stores
    staging.upsert(_set_last(_bars("AAPL"), dividend=0.25))
    provider = FakeProvider()

    report = reconcile_corporate_actions(
        _universe("AAPL"),
        AS_OF,
        provider=provider,
        staging=staging,
        prod=prod,
        actions=actions,
        quarantine={},
        lookback_days=50,
    )

    assert [a.kind for a in report.detected] == [KIND_DIVIDEND]
    assert report.reingested == ()
    assert len(actions.load()) == 1


# --- paso --------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def test_step_updates_quarantine_and_recompute_set(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PROVIDER_TARGET, lambda _id, **_: FakeProvider())
    with connect(cfg_tmp) as con:
        staging = PriceStore(con, "staging")
        staging.upsert(_bars("NVDA", days=29, end=AS_OF - timedelta(days=1)), ingested_at=OLD_LOAD)
        staging.upsert(_set_last(_bars("NVDA", days=1, close=10.0), split=10.0))
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["universe"] = _universe("NVDA")
    ctx.data["quarantine"] = {("NVDA", "XNAS"): "jump", ("OTHER", "XNAS"): "zero_volume"}

    outcome = CorporateActions().run(ctx)

    assert "1 splits" in outcome.message
    assert ctx.data["quarantine"] == {("OTHER", "XNAS"): "zero_volume"}
    assert ctx.data["full_recompute"] == {("NVDA", "XNAS")}
