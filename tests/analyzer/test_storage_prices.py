"""Tabla staging.prices en DuckDB: upsert idempotente, tipos y consultas."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from analyzer.storage import UNKNOWN_MIC, PriceStore, check_identifier


@pytest.fixture
def store() -> PriceStore:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    return PriceStore(con, "staging")


def _frame(rows: list[tuple[str, str | None, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "mic": [r[1] for r in rows],
            "date": pd.to_datetime([r[2] for r in rows]),
            "open": [r[3] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[3] for r in rows],
            "adj_close": [r[4] for r in rows],
            "volume": [1000.0] * len(rows),
            "source": ["test"] * len(rows),
        }
    )


def test_upsert_is_idempotent_and_replaces(store: PriceStore) -> None:
    first = _frame([("AAPL", "XNAS", "2026-09-16", 100.0, 99.0)])
    assert store.upsert(first) == 1
    assert store.upsert(first) == 1

    revised = _frame([("AAPL", "XNAS", "2026-09-16", 101.0, 100.0)])
    store.upsert(revised)

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded.loc[0, "close"] == 101.0
    assert loaded.loc[0, "volume"] == 1000


def test_rows_without_close_are_dropped_and_mic_filled(store: PriceStore) -> None:
    frame = _frame(
        [("AAPL", "XNAS", "2026-09-16", 100.0, 99.0), ("CBOE", None, "2026-09-16", 50.0, 50.0)]
    )
    frame.loc[0, "close"] = float("nan")

    assert store.upsert(frame) == 1
    loaded = store.load()
    assert loaded["ticker"].tolist() == ["CBOE"]
    assert loaded["mic"].tolist() == [UNKNOWN_MIC]


def test_last_dates_and_stats(store: PriceStore) -> None:
    store.upsert(
        _frame(
            [
                ("AAPL", "XNAS", "2026-09-15", 1.0, 1.0),
                ("AAPL", "XNAS", "2026-09-16", 1.0, 1.0),
                ("SAN", "XMAD", "2026-09-10", 1.0, 1.0),
                ("SAN", "XPAR", "2026-09-16", 1.0, 1.0),
            ]
        )
    )

    last = store.last_dates().set_index(["ticker", "mic"])["last_date"]
    assert last[("AAPL", "XNAS")] == date(2026, 9, 16)
    assert last[("SAN", "XMAD")] == date(2026, 9, 10)
    assert last[("SAN", "XPAR")] == date(2026, 9, 16)

    stats = store.stats()
    assert stats["rows"] == 4
    assert stats["keys"] == 3

    window = store.load(start=date(2026, 9, 16))
    assert len(window) == 2


def test_missing_columns_rejected(store: PriceStore) -> None:
    with pytest.raises(ValueError, match="faltan columnas"):
        store.upsert(pd.DataFrame({"ticker": ["AAPL"]}))


def test_schema_identifier_validated() -> None:
    with pytest.raises(ValueError, match="identificador"):
        check_identifier("staging; DROP TABLE x")
