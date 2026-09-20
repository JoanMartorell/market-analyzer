"""Fuente y enriquecedores del S&P 500, construcción genérica y consultas point-in-time."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from analyzer.universe import (
    COLUMNS,
    FetchContext,
    Notes,
    UniverseSpec,
    build_universe,
    get_universe,
    load_constituents,
    members_as_of,
)
from analyzer.universe.helpers import canonical_ticker, merge_fill
from analyzer.universe.sources.europe import EUROPE_MARKETS
from analyzer.universe.sources.sp500.github import parse_intervals_csv
from analyzer.universe.sources.sp500.sec import apply_sec, parse_sec_json
from analyzer.universe.sources.sp500.wikipedia import (
    SOURCE_NAME as WIKI,
)
from analyzer.universe.sources.sp500.wikipedia import (
    apply_wikipedia,
    parse_wikipedia_html,
)
from analyzer.universe.sources.yahoo_symbols import YahooSymbols
from core.config import AppConfig

INTERVALS_CSV = """ticker,start_date,end_date
AAPL,1996-01-02,
FB,2013-12-23,2022-06-09
META,2022-06-09,
brk-b,2010-02-16,
OLD,1996-01-02,2001-05-01
STALE,2019-01-01,
"""

WIKIPEDIA_HTML = """
<table class="wikitable"><thead><tr>
<th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
<th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th>
</tr></thead><tbody>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware</td>
<td>Cupertino</td><td>1982-11-30</td><td>320193</td><td>1977</td></tr>
<tr><td>META</td><td>Meta Platforms</td><td>Communication Services</td><td>Interactive Media</td>
<td>Menlo Park</td><td>2013-12-23[3]</td><td>1326801</td><td>2004</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td>
<td>Omaha</td><td>2010-02-16</td><td>1067983</td><td>1839</td></tr>
<tr><td>NEW</td><td>Newcomer Corp</td><td>Industrials</td><td>Machinery</td>
<td>Nowhere</td><td>2026-09-01</td><td>999</td><td>2000</td></tr>
</tbody></table>
"""

SEC_JSON = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [320193, "Apple Inc.", "AAPL", "Nasdaq"],
        [1067983, "Berkshire", "BRK-B", "NYSE"],
        [1326801, "Meta", "META", "Nasdaq"],
        [1, "Penny Co", "PNNY", "OTC"],
    ],
}

TODAY = date(2026, 9, 17)


# --- helpers ---------------------------------------------------------------


def test_canonical_ticker() -> None:
    assert canonical_ticker(" brk-b ") == "BRK.B"
    assert canonical_ticker(float("nan")) is None
    assert canonical_ticker("") is None


def test_merge_fill_does_not_overwrite_existing_values() -> None:
    table = pd.DataFrame({"ticker": ["A", "B"], "mic": ["XNYS", None]})
    other = pd.DataFrame({"ticker": ["A", "B"], "mic": ["XNAS", "XNAS"], "cik": [1, 2]})

    out = merge_fill(table, other, ["mic", "cik"])

    assert out["mic"].tolist() == ["XNYS", "XNAS"]
    assert out["cik"].tolist() == [1, 2]


# --- fuente y enriquecedores (lógica pura) ------------------------------------


def test_parse_intervals_csv() -> None:
    table = parse_intervals_csv(INTERVALS_CSV)

    assert list(table.columns) == ["ticker", "start", "end"]
    assert "BRK.B" in set(table["ticker"])
    fb = table.loc[table["ticker"] == "FB"].iloc[0]
    assert fb["start"] == pd.Timestamp("2013-12-23")
    assert fb["end"] == pd.Timestamp("2022-06-09")


def test_parse_wikipedia_html_strips_footnotes() -> None:
    table = parse_wikipedia_html(WIKIPEDIA_HTML)

    meta = table.loc[table["ticker"] == "META"].iloc[0]
    assert meta["date_added"] == pd.Timestamp("2013-12-23")
    assert meta["cik"] == 1326801
    assert meta["sector"] == "Communication Services"


def test_parse_wikipedia_html_without_table_fails() -> None:
    with pytest.raises(ValueError, match="tabla de componentes"):
        parse_wikipedia_html("<table><tr><th>Nope</th></tr><tr><td>1</td></tr></table>")


def test_parse_sec_json_maps_exchanges_to_mic() -> None:
    table = parse_sec_json(SEC_JSON).set_index("ticker")

    assert table.loc["AAPL", "mic"] == "XNAS"
    assert table.loc["BRK.B", "mic"] == "XNYS"
    assert pd.isna(table.loc["PNNY", "mic"])  # OTC queda fuera


def test_apply_wikipedia_adds_newcomers_and_reports_stale() -> None:
    intervals = parse_intervals_csv(INTERVALS_CSV)
    intervals["source"] = "test"

    table, notes = apply_wikipedia(intervals, parse_wikipedia_html(WIKIPEDIA_HTML), TODAY)

    new = table.loc[table["ticker"] == "NEW"].iloc[0]
    assert new["source"] == WIKI
    assert new["start"] == pd.Timestamp("2026-09-01")
    assert pd.isna(new["end"])
    assert notes == {"added": ["NEW"], "open_not_listed": ["STALE"]}
    assert table.loc[table["ticker"] == "STALE", "end"].isna().all()  # no se toca
    assert table.loc[table["ticker"] == "AAPL", "sector"].iloc[0] == "Information Technology"
    assert table.loc[table["ticker"] == "FB", "sector"].isna().all()  # histórico sin enriquecer


def test_apply_sec_fills_mic_and_cik() -> None:
    intervals = parse_intervals_csv(INTERVALS_CSV)

    table, notes = apply_sec(intervals, parse_sec_json(SEC_JSON))
    rows = table.set_index("ticker")

    assert rows.loc["AAPL", "mic"] == "XNAS"
    assert rows.loc["BRK.B", "mic"] == "XNYS"
    assert rows.loc["BRK.B", "cik"] == 1067983
    assert notes == {"consulted": True, "current_without_mic": 1}  # STALE


# --- construcción genérica con fuente y enriquecedores falsos ------------------


class _FakeSource:
    name = "fake:intervals"

    def fetch(self, client: httpx.Client, ctx: FetchContext) -> pd.DataFrame:
        return parse_intervals_csv(INTERVALS_CSV)


class _FakeWikipedia:
    name = WIKI

    def enrich(
        self, table: pd.DataFrame, client: httpx.Client, ctx: FetchContext
    ) -> tuple[pd.DataFrame, Notes]:
        return apply_wikipedia(table, parse_wikipedia_html(WIKIPEDIA_HTML), ctx.today)


class _FakeSec:
    name = "sec"

    def enrich(
        self, table: pd.DataFrame, client: httpx.Client, ctx: FetchContext
    ) -> tuple[pd.DataFrame, Notes]:
        return apply_sec(table, parse_sec_json(SEC_JSON))


FAKE_SPEC = UniverseSpec(
    id="sp500", currency="USD", source=_FakeSource(), enrichers=(_FakeWikipedia(), _FakeSec())
)


@pytest.fixture
def built(cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, object]:
    monkeypatch.setattr("analyzer.universe.build.get_universe", lambda _id: FAKE_SPEC)
    offline = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    path = tmp_path / "universe" / "sp500_pit.parquet"
    report = build_universe(
        cfg.regions["americas"], path, user_agent=None, client=offline, today=TODAY
    )
    return path, report


def test_build_writes_full_schema_and_report(built: tuple[Path, object]) -> None:
    path, report = built
    table = load_constituents(path)

    assert list(table.columns) == list(COLUMNS)
    assert path.with_suffix(".meta.json").is_file()
    assert (table["currency"] == "USD").all()
    assert getattr(report, "rows") == len(table) == 7  # noqa: B009
    assert getattr(report, "current_members") == 5  # noqa: B009  AAPL META BRK.B STALE NEW
    assert set(getattr(report, "notes")) == {WIKI, "sec"}  # noqa: B009


def test_members_as_of_respects_intervals(built: tuple[Path, object]) -> None:
    table = load_constituents(built[0])

    in_2020 = set(members_as_of(table, date(2020, 1, 1))["ticker"])
    assert "FB" in in_2020 and "META" not in in_2020 and "OLD" not in in_2020

    on_rename_day = set(members_as_of(table, date(2022, 6, 9))["ticker"])
    assert "FB" not in on_rename_day and "META" in on_rename_day  # end es exclusivo

    assert "NEW" not in set(members_as_of(table, date(2026, 8, 31))["ticker"])
    assert "NEW" in set(members_as_of(table, date(2026, 9, 1))["ticker"])


def test_build_without_enrichers_fills_defaults(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bare = replace(FAKE_SPEC, id="bare", enrichers=())
    monkeypatch.setattr("analyzer.universe.build.get_universe", lambda _id: bare)
    offline = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))

    build_universe(
        cfg.regions["americas"], tmp_path / "bare.parquet", user_agent=None, client=offline
    )

    table = load_constituents(tmp_path / "bare.parquet")
    assert table["mic"].isna().all()
    assert table["cik"].isna().all()
    assert (table["source"] == "fake:intervals").all()


def test_unknown_universe_is_rejected() -> None:
    with pytest.raises(ValueError, match="no soportado"):
        get_universe("nonexistent")


def test_load_missing_file_is_clear(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="constituyentes"):
        load_constituents(tmp_path / "nope.parquet")


def test_members_between_includes_departed_members() -> None:
    from analyzer.universe import members_between

    table = pd.DataFrame(
        {
            "ticker": ["OLD", "NEW", "GONE"],
            "mic": ["XNAS"] * 3,
            "start": pd.to_datetime(["2015-01-01", "2026-06-01", "2015-01-01"]),
            "end": pd.to_datetime(pd.Series([None, None, "2020-01-01"])),
        }
    )

    window = members_between(table, date(2023, 1, 1), date(2026, 9, 17))
    assert window["ticker"].tolist() == ["NEW", "OLD"]
    later = members_between(table, date(2021, 1, 1), date(2026, 9, 17))["ticker"].tolist()
    earlier = members_between(table, date(2019, 1, 1), date(2026, 9, 17))["ticker"].tolist()
    assert "GONE" not in later
    assert "GONE" in earlier


# --- símbolos de Yahoo --------------------------------------------------------


def _known_prices(keys: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Yahoo solo conoce SAP tal cual; el resto hay que buscarlo."""
    return pd.DataFrame({"ticker": ["SAP"], "mic": ["XETR"], "date": [pd.Timestamp(end)]})


QUOTES: dict[str, list[dict[str, str]] | None] = {
    "Air Liquide": [{"symbol": "AI.PA", "quoteType": "EQUITY"}, {"symbol": "AIL.F"}],
    "argenx": [
        {"symbol": "ARGX", "quoteType": "EQUITY"},  # sin sufijo = USA, nunca vale
        {"symbol": "ARGX.BR", "quoteType": "EQUITY"},
    ],
    "Hiscox": [
        {"symbol": "HCXLY", "quoteType": "EQUITY"},
        {"symbol": "HSX.L", "quoteType": "EQUITY"},
    ],
    "TUI Group": [],  # por nombre nada; por ticker sí
    "TUI": [{"symbol": "TUI1.DE", "quoteType": "EQUITY"}],
    "Credit Agricole": [{"symbol": "ACA.PA", "quoteType": "EQUITY"}],  # sin acento
    "Nadie": [{"symbol": "NADIE.PA", "quoteType": "ETF"}],
    "Fallo": None,
}


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["SAP", "AIRP", "ARGX", "HSX", "TUI", "CAGR", "ZZZ", "FFF"],
            "name": [
                "SAP SE",
                "Air Liquide",
                "argenx",
                "Hiscox",
                "TUI Group",
                "Crédit Agricole",
                "Nadie",
                "Fallo",
            ],
            "mic": ["XETR", "XPAR", "XAMS", None, "XETR", "XPAR", "XPAR", "XPAR"],
            "currency": ["EUR", "EUR", "EUR", None, "EUR", "EUR", "EUR", "EUR"],
        }
    )


def _enricher(searches: list[str]) -> YahooSymbols:
    def search(_client: httpx.Client, query: str) -> list[dict[str, str]] | None:
        searches.append(query)
        return QUOTES.get(query, [])

    return YahooSymbols(
        markets=EUROPE_MARKETS, fetch_prices=_known_prices, search=search, pause_seconds=0.0
    )


def test_yahoo_symbols_fixes_tickers_relocates_and_drops_the_dead(tmp_path: Path) -> None:
    searches: list[str] = []
    offline = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))

    out, notes = _enricher(searches).enrich(
        _table(), offline, FetchContext(today=TODAY, universe_dir=tmp_path)
    )

    by_name = out.set_index("name")
    assert by_name.loc["SAP SE", "ticker"] == "SAP"
    assert list(by_name.loc["Air Liquide", ["ticker", "mic"]]) == ["AI", "XPAR"]
    assert list(by_name.loc["argenx", ["ticker", "mic", "currency"]]) == ["ARGX", "XBRU", "EUR"]
    assert list(by_name.loc["Hiscox", ["ticker", "mic", "currency"]]) == ["HSX", "XLON", "GBP"]
    assert by_name.loc["TUI Group", "ticker"] == "TUI1"  # por ticker, en su bolsa
    assert by_name.loc["Crédit Agricole", "ticker"] == "ACA"  # buscado sin acento
    assert "Nadie" not in by_name.index  # búsqueda correcta sin resultado: fuera
    assert by_name.loc["Fallo", "ticker"] == "FFF"  # búsqueda fallida: se queda como estaba
    assert notes["dropped"] == ["ZZZ@XPAR"]
    assert notes["search_failed"] == ["FFF@XPAR"]
    assert notes["relocated"] == ["ARGX@XAMS -> XBRU", "HSX@? -> XLON"]
    assert notes["resolved"][0] == "AIRP@XPAR -> AI@XPAR"
    assert "Credit Agricole" in searches and "Crédit Agricole" not in searches
    assert searches.index("TUI Group") < searches.index("TUI")

    # La memoria guarda solo lo resuelto, y el siguiente build no vuelve a buscar.
    cache = json.loads((tmp_path / "yahoo_symbols.json").read_text(encoding="utf-8"))
    assert cache["AIRP@XPAR"] == ["AI", "XPAR"]
    assert "ZZZ@XPAR" not in cache and "FFF@XPAR" not in cache
    again: list[str] = []
    out2, notes2 = _enricher(again).enrich(
        _table(), offline, FetchContext(today=TODAY, universe_dir=tmp_path)
    )
    assert notes2["from_cache"] == 5
    assert sorted(again) == ["Fallo", "Nadie", "ZZZ"]  # ZZZ: por nombre y por ticker
    assert out2["ticker"].tolist() == out["ticker"].tolist()
