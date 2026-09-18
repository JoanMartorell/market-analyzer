"""Paso persist: señales con procedencia y veredicto, tabla y paso."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from typing import Any

import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError, StepStatus
from analyzer.steps.llm import Analysis, LlmReport, TokenUsage, Verdict
from analyzer.steps.persist import Persist, PersistReport, build_signals, run_persist
from analyzer.steps.screener.service import CANDIDATE_COLUMNS
from analyzer.steps.snapshot import SnapshotReport
from analyzer.storage import SIGNAL_COLUMNS, SignalStore, connect
from core.config import AppConfig, Env

AS_OF = date(2026, 9, 17)
NEXT = date(2026, 9, 18)
REGION = "americas"
PANEL_HASH = "a" * 64
EXECUTE_ON = {"XNYS": NEXT, "XNAS": NEXT}


# --- datos de prueba --------------------------------------------------------------------


def _candidate(ticker: str = "AAA", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticker": ticker,
        "mic": "XNYS",
        "rule_id": "pullback",
        "direction": "long",
        "score": 0.8,
        "conditions_met": "rsi_14 < 35; close > sma_200",
        "close": 101.25,
        "sector": "Industrials",
        "rule_hash": "regla-abc",
    }
    row.update(overrides)
    return row


def _candidates(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [_candidate()], columns=[*CANDIDATE_COLUMNS])


def _verdict(ticker: str, name: str = "confirmar", confidence: float = 0.7) -> Verdict:
    return Verdict(
        ticker=ticker,
        verdict=name,
        confidence=confidence,
        headline="Buena pinta",
        rationale=f"{ticker} tiene buena pinta.",
        risks=["riesgo uno", "riesgo dos; con punto y coma"],
    )


def _analysis(*verdicts: Verdict) -> Analysis:
    return Analysis(summary="Día tranquilo.", verdicts=list(verdicts))


def _panel(*tickers: str, currency: str = "USD") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "mic": ["XNYS"] * len(tickers),
            "currency": [currency] * len(tickers),
        }
    )


def _build(candidates: pd.DataFrame, analysis: Analysis | None, **overrides: Any) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "region_id": REGION,
        "as_of": AS_OF,
        "snapshot_hash": PANEL_HASH,
        "execute_on": EXECUTE_ON,
        "model": "claude-sonnet-5",
    }
    kwargs.update(overrides)
    return build_signals(candidates, analysis, **kwargs)


# --- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Any) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


@pytest.fixture
def store(cfg_tmp: AppConfig) -> Iterator[SignalStore]:
    with connect(cfg_tmp) as con:
        yield SignalStore(con, cfg_tmp.settings.data.prod_schema)


# --- construcción -----------------------------------------------------------------------


def test_a_signal_carries_the_rule_the_panel_and_the_verdict() -> None:
    signals = _build(_candidates(), _analysis(_verdict("AAA")), panel=_panel("AAA"))

    assert list(signals.columns) == [c for c in SIGNAL_COLUMNS if c != "created_at"]
    row = signals.iloc[0]
    assert row["region"] == REGION
    assert row["date"] == AS_OF
    assert row["rule_hash"] == "regla-abc"
    assert row["snapshot_hash"] == PANEL_HASH
    assert row["execute_on"] == NEXT
    assert row["currency"] == "USD"
    assert row["verdict"] == "confirmar"
    assert row["confidence"] == 0.7
    assert row["headline"] == "Buena pinta"
    assert row["risks"] == ["riesgo uno", "riesgo dos; con punto y coma"]
    assert row["llm_model"] == "claude-sonnet-5"


def test_without_analysis_the_verdict_columns_are_null() -> None:
    signals = _build(_candidates(), None)

    row = signals.iloc[0]
    assert row["verdict"] is None
    assert row["confidence"] is None
    assert row["rationale"] is None
    assert row["risks"] is None
    assert row["llm_model"] is None  # aunque el modelo esté configurado, no opinó


def test_two_rules_on_the_same_ticker_share_its_verdict() -> None:
    candidates = _candidates(_candidate(), _candidate(rule_id="breakout", score=0.6))

    signals = _build(candidates, _analysis(_verdict("AAA", "vigilar")))

    assert signals["verdict"].tolist() == ["vigilar", "vigilar"]
    assert signals["rule_id"].tolist() == ["pullback", "breakout"]


def test_a_candidate_without_verdict_is_an_error() -> None:
    candidates = _candidates(_candidate("AAA"), _candidate("BBB"))

    with pytest.raises(ValueError, match="sin veredicto del modelo: BBB"):
        _build(candidates, _analysis(_verdict("AAA")))


def test_a_market_without_execution_session_is_an_error() -> None:
    with pytest.raises(ValueError, match="sin sesión de ejecución para los mercados: XNYS"):
        _build(_candidates(), None, execute_on={"XNAS": NEXT})


def test_the_panel_hash_is_required() -> None:
    with pytest.raises(ValueError, match="hash del panel"):
        _build(_candidates(), None, snapshot_hash="")


def test_a_ticker_missing_from_the_panel_has_no_currency() -> None:
    signals = _build(_candidates(_candidate("AAA"), _candidate("BBB")), None, panel=_panel("AAA"))

    assert signals["currency"].tolist() == ["USD", None]


def test_no_candidates_gives_no_signals() -> None:
    signals = _build(_candidates().iloc[0:0], None)

    assert signals.empty
    assert list(signals.columns) == [c for c in SIGNAL_COLUMNS if c != "created_at"]


# --- tabla ------------------------------------------------------------------------------


def test_the_store_round_trips_lists_and_nulls(store: SignalStore) -> None:
    candidates = _candidates(_candidate("AAA"), _candidate("BBB", score=0.9))
    signals = _build(candidates, _analysis(_verdict("AAA"), _verdict("BBB", "descartar", 0.2)))

    assert store.replace_day(REGION, AS_OF, signals) == (0, 2)

    loaded = store.load(REGION, AS_OF)
    assert loaded["ticker"].tolist() == ["BBB", "AAA"]  # de mayor a menor score
    assert loaded["risks"].iloc[0] == ["riesgo uno", "riesgo dos; con punto y coma"]
    assert loaded["execute_on"].iloc[0] == pd.Timestamp(NEXT)
    assert loaded["verdict"].tolist() == ["descartar", "confirmar"]


def test_repeating_the_day_replaces_its_signals(store: SignalStore) -> None:
    store.replace_day(
        REGION, AS_OF, _build(_candidates(_candidate("AAA"), _candidate("BBB")), None)
    )
    before = date(2026, 9, 16)
    store.replace_day(REGION, before, _build(_candidates(_candidate("CCC")), None, as_of=before))

    assert store.replace_day(REGION, AS_OF, _build(_candidates(_candidate("AAA")), None)) == (2, 1)

    assert store.load(REGION, AS_OF)["ticker"].tolist() == ["AAA"]
    assert store.load(REGION, before)["ticker"].tolist() == ["CCC"]  # otro día, intacto


def test_a_day_without_candidates_clears_the_previous_run(store: SignalStore) -> None:
    store.replace_day(REGION, AS_OF, _build(_candidates(), None))

    assert store.replace_day(REGION, AS_OF, _build(_candidates().iloc[0:0], None)) == (1, 0)
    assert store.load(REGION, AS_OF).empty


def test_the_store_refuses_a_frame_without_the_columns(store: SignalStore) -> None:
    with pytest.raises(ValueError, match="faltan columnas de las señales"):
        store.replace_day(REGION, AS_OF, _candidates())


# --- servicio ---------------------------------------------------------------------------


def _run(store: SignalStore, candidates: pd.DataFrame, analysis: Analysis | None) -> PersistReport:
    return run_persist(
        candidates,
        analysis,
        AS_OF,
        region_id=REGION,
        snapshot_hash=PANEL_HASH,
        execute_on=EXECUTE_ON,
        model="claude-sonnet-5",
        store=store,
    )


def test_the_service_saves_and_summarises(store: SignalStore) -> None:
    candidates = _candidates(_candidate("AAA"), _candidate("BBB"), _candidate("CCC"))
    analysis = _analysis(_verdict("AAA"), _verdict("BBB"), _verdict("CCC", "vigilar"))

    report = _run(store, candidates, analysis)

    assert report.saved == 3
    assert report.replaced == 0
    assert report.count("confirmar") == 2
    assert report.summary() == (
        f"3 señales para {NEXT}: 2 confirmar, 1 vigilar; panel {PANEL_HASH[:12]}"
    )
    assert len(store.load(REGION, AS_OF)) == 3


def test_the_summary_without_analysis_says_so(store: SignalStore) -> None:
    report = _run(store, _candidates(), None)

    assert report.summary() == f"1 señal para {NEXT} sin veredicto del modelo; panel {'a' * 12}"


def test_the_summary_counts_what_it_replaced(store: SignalStore) -> None:
    _run(store, _candidates(_candidate("AAA"), _candidate("BBB")), None)

    report = _run(store, _candidates().iloc[0:0], None)

    assert report.summary() == f"sin señales; panel {'a' * 12}; sustituyen a 2 anteriores"


# --- paso -------------------------------------------------------------------------------


def _snapshot() -> SnapshotReport:
    return SnapshotReport(
        region_id=REGION,
        as_of=AS_OF,
        expected=1,
        rows=1,
        coverage=1.0,
        snapshot_hash=PANEL_HASH,
        previous_hash=None,
        panel=_panel("AAA"),
    )


def _llm(analysis: Analysis) -> LlmReport:
    return LlmReport(
        region_id=REGION,
        as_of=AS_OF,
        model="claude-sonnet-5",
        candidates=1,
        usage=TokenUsage(),
        cost_eur=0.0,
        month_cost_eur=0.0,
        budget_eur=25.0,
        batch=True,
        reused=True,
        analysis=analysis,
    )


def _context(cfg: AppConfig, **data: Any) -> StepContext:
    ctx = StepContext(cfg=cfg, region=cfg.regions[REGION], as_of=AS_OF)
    ctx.data.update(data)
    return ctx


def test_the_step_needs_the_screener(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="paso screener debe ir antes"):
        Persist().run(_context(cfg_tmp))


def test_the_step_needs_the_snapshot(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="paso snapshot debe ir antes"):
        Persist().run(_context(cfg_tmp, candidates=_candidates()))


def test_the_step_needs_the_llm_step_even_if_it_skipped(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="paso llm debe ir antes"):
        Persist().run(_context(cfg_tmp, candidates=_candidates(), snapshot=_snapshot()))


def test_the_step_saves_the_signals_for_the_next_session(cfg_tmp: AppConfig) -> None:
    analysis = _analysis(_verdict("AAA"))
    ctx = _context(
        cfg_tmp,
        candidates=_candidates(),
        snapshot=_snapshot(),
        panel=_panel("AAA"),
        analysis=analysis,
        llm=_llm(analysis),
    )

    outcome = Persist().run(ctx)

    assert outcome.status is StepStatus.OK
    assert outcome.message == f"1 señal para {NEXT}: 1 confirmar; panel {'a' * 12}"
    assert ctx.data["signals"]["currency"].tolist() == ["USD"]
    assert ctx.data["signals"]["llm_model"].tolist() == ["claude-sonnet-5"]
    with connect(cfg_tmp) as con:
        saved = SignalStore(con, cfg_tmp.settings.data.prod_schema).load(REGION, AS_OF)
    assert saved["execute_on"].iloc[0] == pd.Timestamp(NEXT)  # jueves -> viernes


def test_the_step_on_a_friday_executes_on_monday(cfg_tmp: AppConfig) -> None:
    friday = date(2026, 9, 18)
    ctx = _context(cfg_tmp, candidates=_candidates(), snapshot=_snapshot(), analysis=None)
    ctx.as_of = friday

    outcome = Persist().run(ctx)

    assert "para 2026-09-21" in outcome.message
    assert "sin veredicto del modelo" in outcome.message


def test_the_step_without_candidates_is_ok(cfg_tmp: AppConfig) -> None:
    ctx = _context(cfg_tmp, candidates=_candidates().iloc[0:0], snapshot=_snapshot(), analysis=None)

    outcome = Persist().run(ctx)

    assert outcome.status is StepStatus.OK
    assert outcome.message.startswith("sin señales")
    assert ctx.data["signals"].empty
