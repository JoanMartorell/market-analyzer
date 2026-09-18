"""Fuente genérica de Wikipedia, modo instantánea y universos de Europa y APAC."""

from datetime import date

import pandas as pd

from analyzer.universe import get_universe, members_as_of
from analyzer.universe.build import roll_snapshot
from analyzer.universe.sources.europe.stoxx600 import STOXX600
from analyzer.universe.sources.wikipedia_table import (
    WikipediaTable,
    code_ticker,
    parse_wikipedia_table,
)

STOXX_HTML = """
<table><tr><th>Ticker</th><th>Company</th><th>ICB Sector</th><th>Country</th>
<th>Headquarters</th></tr>
<tr><td>SAN</td><td>Banco Santander</td><td>Banks</td><td>Spain</td><td>Madrid</td></tr>
<tr><td>SAN</td><td>Sanofi</td><td>Health Care</td><td>France</td><td>Paris</td></tr>
<tr><td>SHEL</td><td>Shell plc</td><td>Energy</td><td>United Kingdom</td><td>London</td></tr>
<tr><td>TEVA</td><td>Teva</td><td>Health Care</td><td>Israel</td><td>Tel Aviv</td></tr>
</table>
"""

HSI_HTML = """
<table><tr><th>Ticker</th><th>Name</th><th>Sub-index</th></tr>
<tr><td>SEHK:&nbsp;5</td><td>HSBC Holdings plc</td><td>Finance</td></tr>
<tr><td>SEHK:&nbsp;388</td><td>HKEx Limited</td><td>Finance</td></tr>
</table>
"""

NIKKEI_HTML = """
<table><tr><th>年</th><th>除外</th><th>採用</th></tr><tr><td>1970年</td><td>a</td><td>b</td></tr></table>
<table><tr><th>証券コード</th><th>銘柄</th><th>備考</th></tr><tr><td>285A</td><td>Kioxia</td><td></td></tr></table>
<table><tr><th>証券コード</th><th>銘柄</th><th>備考</th></tr><tr><td>7203</td><td>Toyota</td><td></td></tr></table>
"""

HSI = WikipediaTable(
    index_id="hsi",
    url="x",
    ticker_column="Ticker",
    name_column="Name",
    sector_column="Sub-index",
    mic="XHKG",
    currency="HKD",
    clean_ticker=code_ticker(4),
)
NIKKEI = WikipediaTable(
    index_id="nikkei225",
    url="x",
    ticker_column="証券コード",
    name_column="銘柄",
    mic="XTKS",
    currency="JPY",
    clean_ticker=code_ticker(4),
    concat_all=True,
)


def test_code_ticker() -> None:
    clean = code_ticker(4)
    assert clean("SEHK:\xa05") == "0005"
    assert clean("285A") == "285A"
    assert clean("7203") == "7203"
    assert clean("12345") is None
    assert code_ticker(6)("090430") == "090430"


def test_stoxx_table_maps_country_to_market_and_keeps_collisions() -> None:
    table = parse_wikipedia_table(STOXX_HTML, STOXX600)

    santander = table.loc[(table["ticker"] == "SAN") & (table["mic"] == "XMAD")].iloc[0]
    sanofi = table.loc[(table["ticker"] == "SAN") & (table["mic"] == "XPAR")].iloc[0]
    assert santander["currency"] == "EUR" and sanofi["name"] == "Sanofi"
    assert table.loc[table["ticker"] == "SHEL", "currency"].iloc[0] == "GBP"
    teva = table.loc[table["ticker"] == "TEVA"].iloc[0]
    assert pd.isna(teva["mic"]) and pd.isna(teva["currency"])  # país sin bolsa mapeada
    assert (table["index_id"] == "stoxx600").all()


def test_hsi_table_cleans_exchange_prefix() -> None:
    table = parse_wikipedia_table(HSI_HTML, HSI)

    assert table["ticker"].tolist() == ["0005", "0388"]
    assert (table["mic"] == "XHKG").all()
    assert table["sector"].tolist() == ["Finance", "Finance"]


def test_nikkei_concatenates_sector_tables_and_ignores_others() -> None:
    table = parse_wikipedia_table(NIKKEI_HTML, NIKKEI)

    assert sorted(table["ticker"]) == ["285A", "7203"]
    assert table["sector"].isna().all()


def _fresh(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "name": list(tickers),
            "sector": None,
            "mic": "XETR",
            "currency": "EUR",
            "source": "wikipedia:test",
        }
    )


def test_roll_snapshot_first_build_opens_everything_with_unknown_start() -> None:
    table, notes = roll_snapshot(None, _fresh("SAP", "SIE"), date(2026, 9, 18))

    assert table["start"].isna().all() and table["end"].isna().all()
    assert notes["mode"] == "first_snapshot" and notes["members"] == 2


def test_roll_snapshot_closes_gone_and_opens_new() -> None:
    first, _ = roll_snapshot(None, _fresh("SAP", "SIE"), date(2026, 9, 18))
    first["start"] = pd.to_datetime(first["start"])
    first["end"] = pd.to_datetime(first["end"])

    second, notes = roll_snapshot(first, _fresh("SAP", "BMW"), date(2026, 10, 1))

    assert notes["new"] == ["BMW"] and notes["gone"] == ["SIE"]
    sie = second.loc[second["ticker"] == "SIE"].iloc[0]
    bmw = second.loc[second["ticker"] == "BMW"].iloc[0]
    assert sie["end"] == pd.Timestamp("2026-10-01")
    assert bmw["start"] == pd.Timestamp("2026-10-01")
    assert set(members_as_of(second, date(2026, 9, 20))["ticker"]) == {"SAP", "SIE"}
    assert set(members_as_of(second, date(2026, 10, 1))["ticker"]) == {"SAP", "BMW"}

    again, notes_again = roll_snapshot(second, _fresh("SAP", "BMW"), date(2026, 10, 1))
    assert notes_again["new"] == [] and notes_again["gone"] == []
    assert len(again) == len(second)  # idempotente el mismo día


def test_members_as_of_keeps_same_ticker_on_different_exchanges() -> None:
    table = parse_wikipedia_table(STOXX_HTML, STOXX600)
    table["start"] = pd.NaT
    table["end"] = pd.NaT

    members = members_as_of(table, date(2026, 9, 18))

    assert (members["ticker"] == "SAN").sum() == 2


def test_registry_has_all_three_universes() -> None:
    assert get_universe("sp500").history is True
    assert get_universe("stoxx600").history is False
    assert get_universe("apac_large_cap").history is False
