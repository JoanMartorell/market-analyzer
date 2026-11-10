"""Mensajes por operación y su gráfico: qué se recomienda, qué dice el texto, qué se dibuja."""

import struct
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from analyzer.steps.llm import Analysis, Verdict
from analyzer.steps.persist import build_signals
from analyzer.steps.screener.service import CANDIDATE_COLUMNS
from core.charts import render_chart
from core.operations import (
    NO_CHART,
    build_operation_messages,
    find_match,
    recommended_operations,
)

AS_OF = date(2026, 9, 17)
NEXT = date(2026, 9, 18)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Doble suelo cuya ruptura (cierre sobre 110,5) es la última vela.
BREAKOUT = [(0, 120), (20, 130), (40, 100), (50, 110), (60, 100), (68, 111.6)]


def _bars(waypoints: list[tuple[int, float]], *, mirror: float | None = None) -> pd.DataFrame:
    xs, ys = zip(*waypoints, strict=True)
    close = np.interp(np.arange(xs[-1] + 1), xs, ys)
    if mirror is not None:
        close = mirror - close
    days = len(close)
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(AS_OF - timedelta(days=i)) for i in range(days - 1, -1, -1)],
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
        }
    )


def _candidate(ticker: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticker": ticker,
        "mic": "XNYS",
        "rule_id": "oversold_uptrend",
        "direction": "long",
        "score": 0.8,
        "conditions_met": "rsi_14 < 30; close > sma_200",
        "close": 111.6,
        "sector": "Industrials",
        "rule_hash": "regla-abc",
    }
    row.update(overrides)
    return row


def _figure(ticker: str, **overrides: Any) -> dict[str, Any]:
    levels = {
        "rule_id": "double_bottom",
        "pattern": "double_bottom",
        "stop_price": 98.9,
        "target_price": 121.5,
    }
    return _candidate(ticker, **(levels | overrides))


def _verdict(ticker: str, name: str = "confirmar", confidence: float = 0.7) -> Verdict:
    return Verdict(
        ticker=ticker,
        verdict=name,
        confidence=confidence,
        headline="Ruptura con volumen y sin noticias en contra",
        rationale="Nada en contra.",
        risks=[],
    )


def _signals(candidates: list[dict[str, Any]], analysis: Analysis | None) -> pd.DataFrame:
    frame = pd.DataFrame(candidates, columns=[*CANDIDATE_COLUMNS])
    return build_signals(
        frame,
        analysis,
        region_id="americas",
        as_of=AS_OF,
        snapshot_hash="a" * 64,
        execute_on={"XNYS": NEXT},
        model="claude-sonnet-5",
        panel=frame.loc[:, ["ticker", "mic"]].assign(currency="USD"),
    )


def _size(png: bytes) -> tuple[int, int]:
    """Ancho y alto del PNG, de su cabecera IHDR."""
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


# --- qué se recomienda ----------------------------------------------------------------


def test_only_confirmed_signals_are_operations_most_confident_first() -> None:
    analysis = Analysis(
        summary="",
        verdicts=[
            _verdict("LOW", confidence=0.6),
            _verdict("TOP", confidence=0.9),
            _verdict("WAIT", "vigilar"),
            _verdict("BAD", "descartar"),
        ],
    )
    signals = _signals([_candidate(t) for t in ("LOW", "TOP", "WAIT", "BAD")], analysis)

    operations = recommended_operations(signals, analysed=True)

    assert operations["ticker"].tolist() == ["TOP", "LOW"]


def test_a_value_flagged_by_two_rules_is_one_operation_the_one_with_a_pattern() -> None:
    analysis = Analysis(summary="", verdicts=[_verdict("AAA")])
    signals = _signals([_candidate("AAA", score=0.9), _figure("AAA", score=0.6)], analysis)

    operations = recommended_operations(signals, analysed=True)

    assert operations["rule_id"].tolist() == ["double_bottom"]


def test_without_the_model_every_candidate_is_an_operation() -> None:
    signals = _signals([_candidate("AAA", score=0.7), _figure("BBB", score=0.8)], None)

    messages = build_operation_messages(signals, region_id="americas", analysed=False, bars={})

    assert [m.subject for m in messages] == [
        "[americas] COMPRAR BBB · doble suelo",
        "[americas] COMPRAR AAA · oversold_uptrend",
    ]
    assert "Sin veredicto del modelo: lo recomienda la regla." in messages[0].body


def test_no_signals_no_messages() -> None:
    assert (
        build_operation_messages(_signals([], None), region_id="x", analysed=False, bars={}) == []
    )


# --- el mensaje -----------------------------------------------------------------------


def test_the_message_of_a_pattern_operation_carries_levels_and_the_chart() -> None:
    analysis = Analysis(summary="", verdicts=[_verdict("AAA")])
    signals = _signals([_figure("AAA")], analysis)

    (message,) = build_operation_messages(
        signals,
        region_id="americas",
        analysed=True,
        bars={("AAA", "XNYS"): _bars(BREAKOUT)},
        names={("AAA", "XNYS"): "Alpha Inc"},
    )

    assert message.subject == "[americas] COMPRAR AAA · doble suelo"
    assert message.body.splitlines() == [
        "Alpha Inc · XNYS",
        "Cierre 111.60 USD del 2026-09-17 · se ejecuta en la apertura del 2026-09-18",
        "Stop 98.90 (-11.4%) · objetivo 121.50 (+8.9%)",
        "Modelo 70%: Ruptura con volumen y sin noticias en contra",
        "Regla double_bottom · score 0.80",
    ]
    assert message.image is not None
    assert message.image.startswith(PNG_MAGIC)


def test_a_bearish_pattern_is_announced_as_a_short() -> None:
    analysis = Analysis(summary="", verdicts=[_verdict("ZZZ")])
    short = _figure(
        "ZZZ",
        rule_id="double_top",
        pattern="double_top",
        direction="short",
        close=118.4,
        stop_price=131.1,
        target_price=108.5,
    )

    (message,) = build_operation_messages(
        _signals([short], analysis),
        region_id="americas",
        analysed=True,
        bars={("ZZZ", "XNYS"): _bars(BREAKOUT, mirror=230.0)},
    )

    assert message.subject == "[americas] VENDER EN CORTO ZZZ · doble techo"
    assert "Stop 131.10 (+10.7%) · objetivo 108.50 (-8.4%)" in message.body
    assert message.image is not None


def test_an_operation_without_candles_still_goes_out_and_says_so() -> None:
    signals = _signals([_figure("AAA")], None)

    (message,) = build_operation_messages(signals, region_id="americas", analysed=False, bars={})

    assert message.image is None
    assert message.body.splitlines()[-1] == NO_CHART


def test_a_rule_without_pattern_gets_the_plain_price_chart() -> None:
    signals = _signals([_candidate("AAA")], None)

    (message,) = build_operation_messages(
        signals, region_id="americas", analysed=False, bars={("AAA", "XNYS"): _bars(BREAKOUT)}
    )

    assert message.image is not None
    assert "Stop" not in message.body  # sin patrón no hay niveles que dar


# --- la figura y el dibujo ------------------------------------------------------------


def test_find_match_returns_the_pattern_completed_on_the_last_bar() -> None:
    bars = _bars(BREAKOUT)

    match = find_match(bars, "double_bottom")

    assert match is not None
    assert match.bar == len(bars) - 1
    assert find_match(bars, "double_top") is None
    assert find_match(bars.iloc[:-1], "double_bottom") is None  # ayer todavía no había roto


def test_the_chart_is_a_png_of_fixed_size_with_or_without_figure() -> None:
    bars = _bars(BREAKOUT)
    match = find_match(bars, "double_bottom")

    plain = render_chart(bars, title="AAA · precio")
    drawn = render_chart(bars, title="AAA · doble suelo · compra", subtitle="Cierre", match=match)

    assert plain.startswith(PNG_MAGIC)
    assert _size(plain) == _size(drawn) == (1200, 675)
    assert plain != drawn


def test_the_chart_survives_adjusted_prices_and_rejects_empty_input() -> None:
    bars = _bars(BREAKOUT)
    bars["adj_close"] = bars["close"] * 0.5  # un split aplicado a todo el histórico

    assert render_chart(bars, title="AAA", match=find_match(bars, "double_bottom")).startswith(
        PNG_MAGIC
    )
    with pytest.raises(ValueError, match="no hay velas"):
        render_chart(bars.iloc[:0], title="AAA")
