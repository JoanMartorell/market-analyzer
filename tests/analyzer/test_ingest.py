"""Ingesta de precios: símbolos yfinance, reshape, plan incremental, servicio y paso."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.ingest import Ingest, ingest_prices, lookback_start
from analyzer.steps.ingest.providers import get_price_provider
from analyzer.steps.ingest.providers.base import ALL_FETCH_COLUMNS, FETCH_COLUMNS
from analyzer.steps.ingest.providers.yfinance import YFinanceProvider, reshape_download, yf_symbol
from analyzer.steps.ingest.service import plan_fetch
from analyzer.storage import UNKNOWN_MIC, PriceStore, connect
from core.config import AppConfig, Env

PROVIDER_TARGET = "analyzer.steps.ingest.step.get_price_provider"


# --- símbolos y reshape ------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "mic", "expected"),
    [
        ("AAPL", "XNAS", "AAPL"),
        ("BRK.B", "XNYS", "BRK-B"),
        ("CBOE", UNKNOWN_MIC, "CBOE"),
        ("SAN", "XMAD", "SAN.MC"),
        ("ATCO A", "XSTO", "ATCO-A.ST"),
        ("0005", "XHKG", "0005.HK"),
        ("7203", "XTKS", "7203.T"),
        ("X", "XXXX", None),
    ],
)
def test_yf_symbol(ticker: str, mic: str, expected: str | None) -> None:
    assert yf_symbol(ticker, mic) == expected


def _yf_frame(symbols: dict[str, list[float | None]], dates: list[str]) -> pd.DataFrame:
    """Imita la salida de yf.download(group_by='ticker'): columnas (símbolo, campo)."""
    columns = pd.MultiIndex.from_tuples(
        [(s, f) for s in symbols for f in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    )
    data: dict[tuple[str, str], list[float | None]] = {}
    for symbol, closes in symbols.items():
        for field in ("Open", "High", "Low", "Close", "Adj Close"):
            data[(symbol, field)] = closes
        data[(symbol, "Volume")] = [None if c is None else 1_000_000.0 for c in closes]
    frame = pd.DataFrame(data, index=pd.to_datetime(dates), columns=columns)
    frame.index.name = "Date"
    return frame


def test_reshape_download_drops_unknown_symbols_and_nan_rows() -> None:
    raw = _yf_frame(
        {"AAPL": [100.0, 101.0], "SAN.MC": [5.0, None], "NOPE": [None, None]},
        ["2026-09-16", "2026-09-17"],
    )
    symbols = {"AAPL": ("AAPL", "XNAS"), "SAN.MC": ("SAN", "XMAD"), "NOPE": ("NOPE", "XNYS")}

    out = reshape_download(raw, symbols)

    assert list(out.columns) == list(ALL_FETCH_COLUMNS)
    assert len(out) == 3
    assert set(zip(out["ticker"], out["mic"], strict=True)) == {("AAPL", "XNAS"), ("SAN", "XMAD")}
    assert out.loc[out["ticker"] == "SAN", "date"].dt.date.tolist() == [date(2026, 9, 16)]


def test_provider_batches_and_maps_back() -> None:
    calls: list[list[str]] = []

    def fake_download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
        calls.append(symbols)
        return _yf_frame({s: [10.0] for s in symbols}, ["2026-09-17"])

    provider = YFinanceProvider(downloader=fake_download, batch_size=2)
    keys = pd.DataFrame({"ticker": ["AAPL", "BRK.B", "SAN"], "mic": ["XNAS", "XNYS", "XMAD"]})

    out = provider.fetch(keys, date(2026, 9, 17), date(2026, 9, 17))

    assert calls == [["AAPL", "BRK-B"], ["SAN.MC"]]
    assert out["ticker"].tolist() == ["AAPL", "BRK.B", "SAN"]


# --- plan incremental --------------------------------------------------------


def test_lookback_start_covers_sessions() -> None:
    start = lookback_start(date(2026, 9, 17), 250)
    assert (date(2026, 9, 17) - start).days >= 365


def test_plan_fetch_new_incremental_and_up_to_date() -> None:
    keys = pd.DataFrame({"ticker": ["NEW", "OLD", "DONE"], "mic": ["XNAS"] * 3})
    last = pd.DataFrame(
        {
            "ticker": ["OLD", "DONE"],
            "mic": ["XNAS", "XNAS"],
            "last_date": [date(2026, 9, 10), date(2026, 9, 17)],
        }
    )

    planned = plan_fetch(keys, last, date(2026, 9, 17), default_start=date(2025, 9, 10))
    by_ticker = planned.set_index("ticker")["start"]

    assert by_ticker["NEW"] == date(2025, 9, 10)
    assert by_ticker["OLD"] == date(2026, 9, 11)
    assert by_ticker["DONE"] is None

    forced = plan_fetch(
        keys, last, date(2026, 9, 17), date(2025, 9, 10), force_start=date(2024, 1, 1)
    )
    assert set(forced["start"]) == {date(2024, 1, 1)}


# --- servicio ----------------------------------------------------------------


class FakeProvider:
    name = "fake"

    def __init__(self, closes: dict[str, float] | None = None) -> None:
        self.calls: list[tuple[list[str], date, date]] = []
        self.closes = closes if closes is not None else {"AAPL": 100.0, "SAN": 5.0}

    def fetch(self, keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        self.calls.append((keys["ticker"].tolist(), start, end))
        rows = [
            {
                "ticker": t,
                "mic": m,
                "date": pd.Timestamp(end),
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "adj_close": c,
                "volume": 1.0,
            }
            for t, m in zip(keys["ticker"], keys["mic"], strict=True)
            if (c := self.closes.get(t)) is not None
        ]
        return pd.DataFrame(rows, columns=list(FETCH_COLUMNS))


@pytest.fixture
def memory_store() -> PriceStore:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    return PriceStore(con, "staging")


UNIVERSE = pd.DataFrame(
    {"ticker": ["AAPL", "SAN", "GHOST"], "mic": ["XNAS", "XMAD", None], "name": ["a", "s", "g"]}
)


def test_ingest_prices_is_incremental(memory_store: PriceStore) -> None:
    provider = FakeProvider()

    first = ingest_prices(
        UNIVERSE, date(2026, 9, 16), provider=provider, store=memory_store, lookback_days=250
    )
    assert first.requested == 3
    assert first.received == 2
    assert first.rows == 2
    assert first.missing == (f"GHOST@{UNKNOWN_MIC}",)
    assert first.start == lookback_start(date(2026, 9, 16), 250)

    second = ingest_prices(
        UNIVERSE, date(2026, 9, 17), provider=provider, store=memory_store, lookback_days=250
    )
    # AAPL y SAN desde el 17; GHOST sigue sin datos y vuelve a pedirse desde el lookback
    starts = sorted(call[1] for call in provider.calls[1:])
    assert starts == [lookback_start(date(2026, 9, 17), 250), date(2026, 9, 17)]
    assert second.rows == 2

    third = ingest_prices(
        UNIVERSE, date(2026, 9, 17), provider=provider, store=memory_store, lookback_days=250
    )
    assert third.up_to_date == 2
    assert third.requested == 1  # solo GHOST
    assert "ya al día" in third.summary()


# --- paso --------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def _ctx(cfg: AppConfig, universe: pd.DataFrame | None = UNIVERSE) -> StepContext:
    ctx = StepContext(cfg=cfg, region=cfg.regions["americas"], as_of=date(2026, 9, 17))
    if universe is not None:
        ctx.data["universe"] = universe
    return ctx


def test_step_writes_to_staging_and_creates_data_dir(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(PROVIDER_TARGET, lambda _id: provider)
    ctx = _ctx(cfg_tmp)

    outcome = Ingest().run(ctx)

    assert "2/3 valores con datos" in outcome.message
    assert ctx.data["ingest"].rows == 2
    assert cfg_tmp.duckdb_path.is_file()
    with connect(cfg_tmp) as con:
        stats = PriceStore(con, "staging").stats()
    assert stats["rows"] == 2
    assert stats["last_date"] == date(2026, 9, 17)


def test_step_requires_universe(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="universo"):
        Ingest().run(_ctx(cfg_tmp, universe=None))


def test_step_fails_closed_when_nothing_arrives(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PROVIDER_TARGET, lambda _id: FakeProvider(closes={}))

    with pytest.raises(StepError, match="no devolvió datos"):
        Ingest().run(_ctx(cfg_tmp))


def test_step_rejects_unimplemented_provider(cfg_tmp: AppConfig) -> None:
    americas = cfg_tmp.regions["americas"]
    providers = americas.providers.model_copy(update={"prices": "eodhd"})
    region = americas.model_copy(update={"providers": providers})
    ctx = StepContext(cfg=cfg_tmp, region=region, as_of=date(2026, 9, 17))
    ctx.data["universe"] = UNIVERSE

    with pytest.raises(StepError, match="no implementado"):
        Ingest().run(ctx)


def test_registry_knows_yfinance() -> None:
    assert get_price_provider("yfinance").name == "yfinance"
    with pytest.raises(ValueError, match="disponibles"):
        get_price_provider("nope")
