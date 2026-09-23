"""Universos del catálogo de Quantfury: lectura del CSV y coherencia con las regiones."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest

from analyzer.sessions import last_closed_session
from analyzer.universe import FetchContext, get_universe, members_as_of
from analyzer.universe.build import finalize, roll_snapshot
from analyzer.universe.sources.quantfury import CATALOGUE_FILE, QuantfuryCatalogue
from analyzer.universe.sources.quantfury.catalogue import read_catalogue
from analyzer.universe.sources.yahoo_symbols import YahooSymbols
from analyzer.yahoo import YF_SUFFIX
from core.config import AppConfig

QUANTFURY_REGIONS = ("quantfury_us", "quantfury_latam", "quantfury_europe")

CSV = """exchange,symbol,ticker,mic,currency,type,name
NYSE,BRK.B,BRK.B,XNYS,USD,Stocks,Berkshire Hathaway
NYSE,XLE,XLE,XNYS,USD,Etfs,Energy Select Sector SPDR Fund
NASDAQ,AAPL,AAPL,XNAS,USD,Stocks,Apple Inc.
B3,PETR4,PETR4,BVMF,BRL,Stocks,Petróleo Brasileiro S.A.
Cboe Europe,SAN,SAN,XMAD,EUR,Stocks,Banco Santander SA
Cboe Europe,SAN,SAN,XPAR,EUR,Stocks,Sanofi S.A.
Cboe Europe,RYAAY,RYA,XDUB,EUR,Stocks,Ryanair Holdings plc
"""


@pytest.fixture
def catalogue(tmp_path: Path) -> Path:
    path = tmp_path / "instruments.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def _ctx() -> FetchContext:
    return FetchContext(today=date(2026, 9, 18))


def test_catalogue_keeps_only_the_requested_exchanges(catalogue: Path) -> None:
    source = QuantfuryCatalogue(["NYSE", "NASDAQ"], path=catalogue)

    table = source.fetch(httpx.Client(), _ctx())

    assert sorted(table["ticker"]) == ["AAPL", "BRK.B", "XLE"]
    assert set(table["mic"]) == {"XNYS", "XNAS"}
    assert set(table["source"]) == {"quantfury:nyse", "quantfury:nasdaq"}
    assert source.name == "quantfury:nyse+nasdaq"


def test_catalogue_marks_etfs_as_sector(catalogue: Path) -> None:
    table = read_catalogue(catalogue, ["NYSE"])

    sectors = dict(zip(table["ticker"], table["sector"], strict=True))
    assert sectors["XLE"] == "ETF"
    assert pd.isna(sectors["BRK.B"])


def test_catalogue_uses_local_ticker_and_keeps_same_symbol_on_two_markets(
    catalogue: Path,
) -> None:
    table = read_catalogue(catalogue, ["Cboe Europe"])

    assert sorted(zip(table["ticker"], table["mic"], strict=True)) == [
        ("RYA", "XDUB"),
        ("SAN", "XMAD"),
        ("SAN", "XPAR"),
    ]
    assert set(table["currency"]) == {"EUR"}


def test_catalogue_rejects_unknown_exchange(catalogue: Path) -> None:
    with pytest.raises(ValueError, match="bolsas sin instrumentos"):
        read_catalogue(catalogue, ["NYSE", "Bovespa"])


def test_catalogue_rejects_repeated_keys(tmp_path: Path) -> None:
    path = tmp_path / "instruments.csv"
    path.write_text(CSV + "NYSE,AAPL,AAPL,XNAS,USD,Stocks,Apple otra vez\n", encoding="utf-8")

    with pytest.raises(ValueError, match="claves repetidas"):
        read_catalogue(path, ["NYSE", "NASDAQ"])


def test_catalogue_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "instruments.csv"
    path.write_text("exchange,symbol\nNYSE,AAPL\n", encoding="utf-8")

    with pytest.raises(ValueError, match="faltan columnas"):
        read_catalogue(path, ["NYSE"])


def test_snapshot_of_the_catalogue_is_valid_universe(catalogue: Path) -> None:
    spec = get_universe("quantfury_us")
    fresh = read_catalogue(catalogue, ["NYSE", "NASDAQ"])

    table, notes = roll_snapshot(None, fresh, date(2026, 9, 18))
    table = finalize(table, spec)

    assert notes["mode"] == "first_snapshot"
    assert len(members_as_of(table, date(2026, 9, 18))) == 3


@pytest.mark.parametrize("region_id", QUANTFURY_REGIONS)
def test_repo_catalogue_matches_region_markets(cfg: AppConfig, region_id: str) -> None:
    region = cfg.regions[region_id]
    spec = get_universe(region.universe.id)
    currency_of = {m.mic: m.currency for m in region.markets}

    table = spec.source.fetch(httpx.Client(), _ctx())

    assert not spec.history
    assert not table.empty
    assert set(table["mic"]) <= set(currency_of)  # cada valor, en una bolsa de la región
    assert (table["currency"] == table["mic"].map(currency_of)).all()
    for enricher in spec.enrichers:
        assert isinstance(enricher, YahooSymbols)
        assert dict(enricher.markets) == currency_of  # solo reubica dentro de la región
    assert set(currency_of) <= set(YF_SUFFIX)  # Yahoo sabe escribir todas sus bolsas


def test_repo_catalogue_has_the_broker_exchanges() -> None:
    raw = pd.read_csv(CATALOGUE_FILE, dtype=str, keep_default_na=False)

    assert set(raw["exchange"]) == {"NYSE", "NASDAQ", "Cboe Europe", "B3", "BMV"}
    assert set(raw["type"]) <= {"Stocks", "Etfs"}


def test_latam_waits_for_mexico_to_close(cfg: AppConfig) -> None:
    latam = cfg.regions["quantfury_latam"]
    madrid = ZoneInfo("Europe/Madrid")
    # jueves 22:30 Madrid: B3 ya cerró (22:00), la BMV cierra a las 23:00
    assert last_closed_session(latam, datetime(2026, 9, 17, 22, 30, tzinfo=madrid)) == date(
        2026, 9, 16
    )
    # viernes a las 07:00, la hora del ciclo: la sesión del jueves ya vale
    assert last_closed_session(latam, datetime(2026, 9, 18, 7, 0, tzinfo=madrid)) == date(
        2026, 9, 17
    )
