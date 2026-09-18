"""Paso snapshot: panel, hash canónico, servicio y paso."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.indicators import INDICATOR_COLUMNS
from analyzer.steps.snapshot import (
    Snapshot,
    SnapshotReport,
    build_panel,
    panel_hash,
    run_snapshot,
    universe_members,
)
from analyzer.storage import (
    UNIVERSE_COLUMNS,
    UNKNOWN_MIC,
    IndicatorStore,
    SnapshotRunStore,
    SnapshotStore,
    connect,
)
from core.config import AppConfig, Env

AS_OF = date(2026, 9, 17)
REGION = "americas"


def _universe(*tickers: str, mic: str | None = "XNAS") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "mic": [mic] * len(tickers),
            "name": [f"{t} Inc" for t in tickers],
            "sector": ["Financials" if t.startswith("F") else "Technology" for t in tickers],
            "sub_industry": [None] * len(tickers),
            "currency": ["USD"] * len(tickers),
        }
    )


def _indicator_rows(*tickers: str, day: date = AS_OF, mic: str = "XNAS") -> pd.DataFrame:
    """Una fila de ``prod.indicators`` por ticker con valores reproducibles."""
    rows = []
    for ticker in tickers:
        rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
        row: dict[str, object] = {"ticker": ticker, "mic": mic, "date": day}
        row.update({"close": 100.0, "adj_close": 99.0, "volume": 2_000_000})
        row.update({c: float(rng.random()) for c in INDICATOR_COLUMNS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[frame["ticker"] == "IPO", "sma_200"] = np.nan  # histórico corto
    return frame


@pytest.fixture
def stores() -> tuple[IndicatorStore, SnapshotStore, SnapshotRunStore]:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA prod")
    return (
        IndicatorStore(con, "prod", INDICATOR_COLUMNS),
        SnapshotStore(con, "prod", INDICATOR_COLUMNS),
        SnapshotRunStore(con, "prod"),
    )


# --- panel --------------------------------------------------------------------------


def test_universe_members_fills_unknown_mic_and_dedups() -> None:
    universe = pd.concat([_universe("AAPL", mic=None), _universe("AAPL", mic=None)])

    members = universe_members(universe)

    assert len(members) == 1
    assert members["mic"].iloc[0] == UNKNOWN_MIC
    assert list(members.columns) == ["ticker", "mic", *UNIVERSE_COLUMNS]


def test_build_panel_keeps_only_keys_with_today_row() -> None:
    members = universe_members(_universe("AAPL", "MSFT", "FRT"))
    today = _indicator_rows("MSFT", "AAPL")  # FRT en cuarentena: sin fila

    panel = build_panel(members, today)

    assert panel["ticker"].tolist() == ["AAPL", "MSFT"]
    assert panel["sector"].tolist() == ["Technology", "Technology"]
    assert set(INDICATOR_COLUMNS) <= set(panel.columns)


# --- hash ---------------------------------------------------------------------------


def test_panel_hash_ignores_row_order_and_index() -> None:
    columns = ("ticker", "mic", "date", "close", "sma_200")
    a = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "mic": ["XNAS", "XNAS"],
            "date": pd.to_datetime([AS_OF, AS_OF]),
            "close": [1.0, 2.0],
            "sma_200": [np.nan, 0.0],
        }
    )
    b = a.iloc[::-1].reset_index(drop=True)
    b.index = pd.Index([10, 20])
    b["sma_200"] = [-0.0, float("nan")]  # ceros y NaN con otro patrón de bits

    assert panel_hash(a, columns) == panel_hash(b, columns)


def test_panel_hash_changes_with_any_value() -> None:
    columns = ("ticker", "mic", "date", "close", "sector")
    base = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "mic": ["XNAS"],
            "date": pd.to_datetime([AS_OF]),
            "close": [100.0],
            "sector": ["Technology"],
        }
    )
    price = base.assign(close=[100.01])
    sector = base.assign(sector=pd.Series([None], dtype="object"))
    day = base.assign(date=pd.to_datetime([AS_OF + timedelta(days=1)]))

    hashes = {panel_hash(f, columns) for f in (base, price, sector, day)}
    assert len(hashes) == 4
    assert all(len(h) == 64 for h in hashes)


# --- servicio -----------------------------------------------------------------------


def test_run_snapshot_freezes_panel_and_detects_changes(
    stores: tuple[IndicatorStore, SnapshotStore, SnapshotRunStore],
) -> None:
    indicators, store, runs = stores
    indicators.upsert(_indicator_rows("AAPL", "MSFT", "IPO"))
    universe = _universe("AAPL", "MSFT", "IPO")

    def snapshot(members: pd.DataFrame, min_coverage: float = 0.9) -> SnapshotReport:
        return run_snapshot(
            members,
            AS_OF,
            region_id=REGION,
            indicators=indicators,
            store=store,
            runs=runs,
            min_coverage=min_coverage,
        )

    first = snapshot(universe)
    assert first.rows == 3
    assert first.changed is None
    assert first.summary() == (
        f"3/3 valores en el panel, cobertura 100.0%, hash {first.snapshot_hash[:12]}"
    )
    stored = store.load(REGION, AS_OF)
    assert stored["ticker"].tolist() == ["AAPL", "IPO", "MSFT"]
    assert stored["sector"].tolist() == ["Technology", "Technology", "Technology"]
    assert np.isnan(stored.loc[stored["ticker"] == "IPO", "sma_200"].iloc[0])
    header = runs.get(REGION, AS_OF)
    assert header is not None
    assert header["snapshot_hash"] == first.snapshot_hash

    same = snapshot(universe)
    assert same.snapshot_hash == first.snapshot_hash
    assert same.changed is False
    assert same.summary().endswith(", igual que el anterior")

    # Un dividendo reajusta el adj_close de AAPL y MSFT sale del universo.
    indicators.upsert(_indicator_rows("AAPL").assign(adj_close=98.0))
    changed = snapshot(_universe("AAPL", "IPO"))
    assert changed.changed is True
    assert changed.snapshot_hash != first.snapshot_hash
    assert store.load(REGION, AS_OF)["ticker"].tolist() == ["AAPL", "IPO"]  # MSFT ya no está
    header = runs.get(REGION, AS_OF)
    assert header is not None
    assert header["rows"] == 2


def test_run_snapshot_fails_closed_on_low_coverage(
    stores: tuple[IndicatorStore, SnapshotStore, SnapshotRunStore],
) -> None:
    indicators, store, runs = stores
    indicators.upsert(_indicator_rows("AAPL"))

    def snapshot(members: pd.DataFrame, min_coverage: float) -> SnapshotReport:
        return run_snapshot(
            members,
            AS_OF,
            region_id=REGION,
            indicators=indicators,
            store=store,
            runs=runs,
            min_coverage=min_coverage,
        )

    with pytest.raises(ValueError, match=r"cobertura del panel 50.0% \(1/2\)"):
        snapshot(_universe("AAPL", "MSFT"), 0.95)
    assert runs.get(REGION, AS_OF) is None  # no se escribe nada

    with pytest.raises(ValueError, match="ningún valor del universo tiene indicadores"):
        snapshot(_universe("MSFT"), 0.0)


def test_snapshot_store_rejects_reserved_names() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA prod")
    with pytest.raises(ValueError, match="reservado"):
        SnapshotStore(con, "prod", ("sector",))


# --- paso -----------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def test_step_leaves_panel_in_context(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        IndicatorStore(con, "prod", INDICATOR_COLUMNS).upsert(_indicator_rows("AAPL", "MSFT"))
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions[REGION], as_of=AS_OF)
    ctx.data["universe"] = _universe("AAPL", "MSFT")

    outcome = Snapshot().run(ctx)

    assert outcome.message.startswith("2/2 valores en el panel")
    assert ctx.data["panel"]["ticker"].tolist() == ["AAPL", "MSFT"]
    assert ctx.data["snapshot"].region_id == REGION


def test_step_fails_closed_without_indicators(cfg_tmp: AppConfig) -> None:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions[REGION], as_of=AS_OF)
    ctx.data["universe"] = _universe("AAPL")

    with pytest.raises(StepError, match="snapshot fallido: ningún valor"):
        Snapshot().run(ctx)
