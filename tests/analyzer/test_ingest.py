"""Ingesta de precios: símbolos yfinance, reshape, plan incremental, servicio y paso."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.ingest import Ingest, ingest_prices, lookback_start
from analyzer.steps.ingest.providers import DailyBudget, EodhdProvider, get_price_provider
from analyzer.steps.ingest.providers.base import ALL_FETCH_COLUMNS, FETCH_COLUMNS
from analyzer.steps.ingest.providers.eodhd import eodhd_symbol
from analyzer.steps.ingest.providers.yfinance import YFinanceProvider, reshape_download
from analyzer.steps.ingest.service import plan_fetch
from analyzer.storage import UNKNOWN_MIC, ApiCallStore, PriceStore, connect
from analyzer.yahoo import yf_symbol
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
    def __init__(
        self,
        closes: dict[str, float] | None = None,
        *,
        name: str = "fake",
        bar_dates: dict[str, date] | None = None,  # valores cuya última vela no llega a end
    ) -> None:
        self.name = name
        self.calls: list[tuple[list[str], date, date]] = []
        self.closes = closes if closes is not None else {"AAPL": 100.0, "SAN": 5.0}
        self.bar_dates = bar_dates or {}

    def fetch(self, keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        self.calls.append((keys["ticker"].tolist(), start, end))
        rows = [
            {
                "ticker": t,
                "mic": m,
                "date": pd.Timestamp(self.bar_dates.get(t, end)),
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
    monkeypatch.setattr(PROVIDER_TARGET, lambda _id, **_: provider)
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
    monkeypatch.setattr(PROVIDER_TARGET, lambda _id, **_: FakeProvider(closes={}))

    with pytest.raises(StepError, match="no devolvió datos"):
        Ingest().run(_ctx(cfg_tmp))


def test_step_rejects_unimplemented_provider(cfg_tmp: AppConfig) -> None:
    americas = cfg_tmp.regions["americas"]
    providers = americas.providers.model_copy(update={"prices": "nope"})
    region = americas.model_copy(update={"providers": providers})
    ctx = StepContext(cfg=cfg_tmp, region=region, as_of=date(2026, 9, 17))
    ctx.data["universe"] = UNIVERSE

    with pytest.raises(StepError, match="no implementado"):
        Ingest().run(ctx)


def test_registry_knows_yfinance() -> None:
    assert get_price_provider("yfinance").name == "yfinance"
    with pytest.raises(ValueError, match="disponibles"):
        get_price_provider("nope")


# --- rescate con un segundo proveedor ------------------------------------------


def test_fallback_rescues_keys_without_the_day_bar(memory_store: PriceStore) -> None:
    as_of = date(2026, 9, 17)
    main = FakeProvider(bar_dates={"SAN": date(2026, 9, 15)})  # SAN llega con dos días de retraso
    rescue = FakeProvider(closes={"SAN": 6.0}, name="rescue")

    report = ingest_prices(
        UNIVERSE,
        as_of,
        provider=main,
        store=memory_store,
        lookback_days=250,
        fallback=rescue,
        fallback_markets=["XMAD", "XNAS"],
    )

    assert [c[0] for c in rescue.calls] == [["SAN"]]  # GHOST no tiene bolsa con sesión
    assert report.rescued == 1
    assert report.stale == ()
    assert report.rows == 3
    assert "1 rescatados por rescue" in report.summary()
    saved = memory_store.load(keys=pd.DataFrame({"ticker": ["SAN"], "mic": ["XMAD"]}))
    by_day = dict(zip(saved["date"].dt.date, saved["source"], strict=True))
    assert by_day == {date(2026, 9, 15): "fake", as_of: "rescue"}


def test_fallback_only_asks_for_markets_with_a_session(memory_store: PriceStore) -> None:
    main = FakeProvider(bar_dates={"SAN": date(2026, 9, 15)})
    rescue = FakeProvider(closes={"SAN": 6.0}, name="rescue")

    report = ingest_prices(
        UNIVERSE,
        date(2026, 9, 17),
        provider=main,
        store=memory_store,
        lookback_days=250,
        fallback=rescue,
        fallback_markets=["XNAS"],
    )

    assert rescue.calls == []
    assert report.rescued == 0
    assert report.stale == ("SAN@XMAD",)
    assert "sin llegar a 2026-09-17: 1" in report.summary()


# --- eodhd -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "mic", "expected"),
    [
        ("SAP", "XETR", "SAP.XETRA"),
        ("BRK.B", "XNYS", "BRK-B.US"),
        ("ATCO A", "XSTO", "ATCO-A.ST"),
        ("7203", "XTKS", "7203.TSE"),
        ("X", "XXXX", None),
    ],
)
def test_eodhd_symbol(ticker: str, mic: str, expected: str | None) -> None:
    assert eodhd_symbol(ticker, mic) == expected


def _eodhd_rows(closes: list[float]) -> list[dict[str, float | str]]:
    return [
        {
            "date": f"2026-09-{15 + i:02d}",
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "adjusted_close": c - 1,
            "volume": 100,
        }
        for i, c in enumerate(closes)
    ]


def _budget(limit: int) -> DailyBudget:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    store = ApiCallStore(con, "staging")
    return DailyBudget(store, "eodhd", limit, today=lambda: date(2026, 9, 19))


def test_eodhd_maps_rows_and_stops_at_the_daily_budget() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.path.rsplit("/", 1)[-1]
        seen.append(symbol)
        assert request.url.params["api_token"] == "secret"
        if symbol == "NOPE.PA":
            return httpx.Response(404)
        return httpx.Response(200, json=_eodhd_rows([10.0, 11.0]))

    budget = _budget(limit=2)
    provider = EodhdProvider(
        "secret", budget=budget, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    keys = pd.DataFrame(
        {"ticker": ["SAP", "NOPE", "BMW", "X"], "mic": ["XETR", "XPAR", "XETR", "XXXX"]}
    )

    out = provider.fetch(keys, date(2026, 9, 15), date(2026, 9, 16))

    assert seen == ["SAP.XETRA", "NOPE.PA"]  # BMW se queda sin cupo; X no tiene bolsa
    assert budget.used() == 2
    assert list(out.columns) == list(ALL_FETCH_COLUMNS)
    assert out["ticker"].tolist() == ["SAP", "SAP"]
    assert out["mic"].tolist() == ["XETR", "XETR"]
    assert out["adj_close"].tolist() == [9.0, 10.0]
    assert out["dividend"].isna().all()


def test_eodhd_stops_after_a_plan_rejection() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(402, text="upgrade your plan")

    provider = EodhdProvider("secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    keys = pd.DataFrame({"ticker": ["SAP", "BMW"], "mic": ["XETR", "XETR"]})

    out = provider.fetch(keys, date(2026, 9, 15), date(2026, 9, 16))

    assert len(seen) == 1
    assert out.empty


def test_registry_builds_eodhd_with_its_budget(cfg_tmp: AppConfig) -> None:
    with pytest.raises(ValueError, match="cupo"):
        get_price_provider("eodhd")

    with connect(cfg_tmp) as con, pytest.raises(ValueError, match="EODHD_API_KEY"):
        get_price_provider("eodhd", cfg=cfg_tmp, con=con)

    with_key = replace(
        cfg_tmp, env=Env(_env_file=None, ma_data_dir=cfg_tmp.env.ma_data_dir, eodhd_api_key="k")
    )
    with connect(with_key) as con:
        provider = get_price_provider("eodhd", cfg=with_key, con=con)
    assert isinstance(provider, EodhdProvider)
    assert provider.budget is not None
    assert provider.budget.limit == 20
