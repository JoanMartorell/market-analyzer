"""Paso screener: ventana, filtros, score, límite de candidatos y paso."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.indicators import INDICATOR_COLUMNS
from analyzer.steps.screener import CANDIDATE_COLUMNS, Screener, run_screener
from analyzer.storage import UNIVERSE_COLUMNS, IndicatorStore, connect
from core.config import AppConfig, Env, Rule

AS_OF = date(2026, 9, 17)
DAYS = 8  # sesiones guardadas por valor; la regla usa ventana 5 y el cruce mira una más


def _history(
    ticker: str,
    *,
    rsi: list[float] | float = 50.0,
    close: float = 100.0,
    sma_200: float = 90.0,
    macd: list[float] | float = -1.0,
    avg_volume: float = 5_000_000.0,
    days: int = DAYS,
) -> pd.DataFrame:
    """``days`` filas de indicadores hasta AS_OF. Las listas van de la más antigua a hoy."""
    dates = [pd.Timestamp(AS_OF - timedelta(days=i)) for i in range(days - 1, -1, -1)]
    rows: list[dict[str, Any]] = []
    for i, day in enumerate(dates):
        row: dict[str, Any] = dict.fromkeys(INDICATOR_COLUMNS, 0.0)
        row.update(ticker=ticker, mic="XNAS", date=day, close=close, adj_close=close)
        row.update(volume=avg_volume, avg_volume_20=avg_volume, sma_200=sma_200, macd_signal=0.0)
        row["rsi_14"] = rsi[i] if isinstance(rsi, list) else rsi
        row["macd"] = macd[i] if isinstance(macd, list) else macd
        rows.append(row)
    return pd.DataFrame(rows)


def _panel(store: IndicatorStore, sectors: dict[str, str]) -> pd.DataFrame:
    today = store.load(start=AS_OF, end=AS_OF).drop(columns=["computed_at"])
    for column in UNIVERSE_COLUMNS:
        today[column] = None
    today["sector"] = today["ticker"].map(sectors)
    today["currency"] = "USD"
    return today


def _rule(**overrides: Any) -> Rule:
    data: dict[str, Any] = {
        "version": 3,
        "id": "oversold_uptrend",
        "window": 5,
        "conditions": [
            {"expr": "rsi_14 < 30", "weight": 0.4},
            {"expr": "close > sma_200", "weight": 0.3, "required": True},
            {"expr": "bullish_cross(macd, macd_signal)", "weight": 0.3},
        ],
        "universe_filters": ["sector != 'Financials'"],
        "output": {"score_threshold": 0.7},
        "source_hash": "abc123",
    }
    data.update(overrides)
    return Rule.model_validate(data)


@pytest.fixture
def store() -> IndicatorStore:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA prod")
    return IndicatorStore(con, "prod", INDICATOR_COLUMNS)


def _run(store: IndicatorStore, panel: pd.DataFrame, rules: list[Rule], **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "max_candidates": 20,
        "min_score": 0.0,
        "min_avg_volume": 1_000_000,
    }
    kwargs.update(overrides)
    return run_screener(panel, AS_OF, rules=rules, indicators=store, **kwargs)


# RSI en sobreventa hace tres sesiones y cruce alcista de MACD ayer: candidato completo.
OVERSOLD_RSI = [50.0, 50.0, 50.0, 50.0, 50.0, 28.0, 40.0, 45.0]
CROSS_MACD = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5, 0.6]


def test_scores_conditions_within_window_and_ranks_candidates(store: IndicatorStore) -> None:
    store.upsert(
        pd.concat(
            [
                _history("AAPL", rsi=OVERSOLD_RSI, macd=CROSS_MACD),  # todo: 1.0
                _history("MSFT", rsi=OVERSOLD_RSI),  # sin cruce: 0.7
                _history("NVDA", rsi=[28.0] + [50.0] * 7, macd=CROSS_MACD),  # RSI fuera de ventana
                _history("BAC", rsi=OVERSOLD_RSI, macd=CROSS_MACD),  # Financials: filtrado
                _history("PLTR", rsi=OVERSOLD_RSI, macd=CROSS_MACD, avg_volume=500_000.0),
            ]
        )
    )
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology"}
    sectors |= {"BAC": "Financials", "PLTR": "Technology"}
    panel = _panel(store, sectors)

    report = _run(store, panel, [_rule()])

    assert report.evaluated == 5
    assert report.illiquid == 1  # PLTR
    assert report.rules[0].eligible == 3  # BAC filtrado, PLTR ni se evalúa
    assert list(report.candidates.columns) == list(CANDIDATE_COLUMNS)
    assert report.candidates["ticker"].tolist() == ["AAPL", "MSFT"]
    assert report.candidates["score"].tolist() == [1.0, 0.7]
    assert report.candidates["conditions_met"].iloc[1] == "rsi_14 < 30; close > sma_200"
    assert report.candidates["rule_hash"].iloc[0] == "abc123"
    assert report.candidates["sector"].iloc[0] == "Technology"
    assert report.candidates["direction"].iloc[0] == "long"
    assert report.summary() == (
        "2 candidatos entre 5 valores (1 sin liquidez): oversold_uptrend 2/3; AAPL 1.00, MSFT 0.70"
    )


def test_required_condition_zeroes_the_score(store: IndicatorStore) -> None:
    store.upsert(_history("AAPL", rsi=OVERSOLD_RSI, macd=CROSS_MACD, close=80.0, sma_200=90.0))
    panel = _panel(store, {"AAPL": "Technology"})

    report = _run(store, panel, [_rule()])

    assert report.candidates.empty
    assert report.summary() == "sin candidatos entre 1 valores: oversold_uptrend 0/1"


def test_short_history_key_never_qualifies_and_nan_is_not_liquid(store: IndicatorStore) -> None:
    fresh = _history("IPO", rsi=[20.0, 20.0], macd=[-1.0, 1.0], days=2)
    fresh["avg_volume_20"] = np.nan
    store.upsert(fresh)
    panel = _panel(store, {"IPO": "Technology"})

    report = _run(store, panel, [_rule()])

    assert report.illiquid == 1
    assert report.candidates.empty


def test_max_candidates_and_min_score_apply_across_rules(store: IndicatorStore) -> None:
    store.upsert(
        pd.concat(
            [
                _history("AAPL", rsi=OVERSOLD_RSI, macd=CROSS_MACD),
                _history("MSFT", rsi=OVERSOLD_RSI),
            ]
        )
    )
    panel = _panel(store, {"AAPL": "Technology", "MSFT": "Technology"})
    momentum = _rule(
        id="momentum",
        window=1,
        conditions=[{"expr": "macd > macd_signal", "weight": 1.0}],
        universe_filters=[],
        output={"score_threshold": 0.5},
    )

    report = _run(store, panel, [_rule(), momentum], max_candidates=2, min_score=0.9)

    assert [tuple(r) for r in report.candidates[["ticker", "rule_id"]].itertuples(index=False)] == [
        ("AAPL", "momentum"),
        ("AAPL", "oversold_uptrend"),
    ]
    assert report.rules[1].candidates == 1  # momentum: solo AAPL está sobre la señal hoy


def test_unknown_variable_fails_before_evaluating(store: IndicatorStore) -> None:
    store.upsert(_history("AAPL"))
    panel = _panel(store, {"AAPL": "Technology"})
    rule = _rule(conditions=[{"expr": "pe_ratio < 15", "weight": 1.0}])

    with pytest.raises(
        ValueError, match="regla 'oversold_uptrend': variables desconocidas pe_ratio"
    ):
        _run(store, panel, [rule])


def test_panel_key_without_today_row_fails_closed(store: IndicatorStore) -> None:
    store.upsert(_history("AAPL"))
    panel = _panel(store, {"AAPL": "Technology"})
    panel = pd.concat([panel, panel.assign(ticker="GHOST")], ignore_index=True)

    with pytest.raises(ValueError, match="1 valores del panel sin indicadores de 2026-09-17"):
        _run(store, panel, [_rule()])


def test_region_without_rules_fails(store: IndicatorStore) -> None:
    store.upsert(_history("AAPL"))
    with pytest.raises(ValueError, match="no tiene reglas"):
        _run(store, _panel(store, {"AAPL": "Technology"}), [])


# --- paso -----------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def test_step_uses_region_rules_and_leaves_candidates(cfg_tmp: AppConfig) -> None:
    with connect(cfg_tmp) as con:
        store = IndicatorStore(con, "prod", INDICATOR_COLUMNS)
        store.upsert(_history("AAPL", rsi=OVERSOLD_RSI, macd=CROSS_MACD))
        panel = _panel(store, {"AAPL": "Information Technology"})
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["panel"] = panel

    outcome = Screener().run(ctx)

    assert outcome.message == "1 candidato entre 1 valores: oversold_uptrend 1/1; AAPL 1.00"
    assert (
        ctx.data["candidates"]["rule_hash"].iloc[0] == cfg_tmp.rules["oversold_uptrend"].source_hash
    )


def test_step_requires_panel(cfg_tmp: AppConfig) -> None:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)

    with pytest.raises(StepError, match="no hay panel"):
        Screener().run(ctx)
