"""Contador de llamadas por proveedor y día: se apunta antes de llamar y nunca pasa del cupo."""

from datetime import date

import duckdb

from analyzer.storage import ApiCallStore


def test_reserve_counts_until_the_limit_and_then_refuses() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    store = ApiCallStore(con, "staging")
    day = date(2026, 9, 19)

    assert store.used("eodhd", day) == 0
    assert store.reserve("eodhd", day, limit=2)
    assert store.reserve("eodhd", day, limit=2)
    assert not store.reserve("eodhd", day, limit=2)
    assert store.used("eodhd", day) == 2

    # Otro día y otro proveedor van aparte; otra instancia sobre la misma base ve lo mismo.
    assert store.reserve("eodhd", date(2026, 9, 20), limit=2)
    assert store.reserve("fmp", day, limit=1)
    assert ApiCallStore(con, "staging").used("eodhd", day) == 2
