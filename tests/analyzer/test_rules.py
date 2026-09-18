"""Motor de reglas: compilación segura, evaluación vectorizada y funciones del registro."""

import numpy as np
import pandas as pd
import pytest

from analyzer.rules import ExpressionError, compile_expression


def _frame() -> pd.DataFrame:
    """Dos claves con tres sesiones; MACD cruza la señal en la última de AAPL."""
    index = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "XNAS", pd.Timestamp("2026-09-15")),
            ("AAPL", "XNAS", pd.Timestamp("2026-09-16")),
            ("AAPL", "XNAS", pd.Timestamp("2026-09-17")),
            ("MSFT", "XNAS", pd.Timestamp("2026-09-15")),
            ("MSFT", "XNAS", pd.Timestamp("2026-09-16")),
            ("MSFT", "XNAS", pd.Timestamp("2026-09-17")),
        ],
        names=["ticker", "mic", "date"],
    )
    return pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 20.0, np.nan, 22.0],
            "sma_200": [9.0, 9.0, 9.0, 25.0, 25.0, 25.0],
            "rsi_14": [25.0, 35.0, 45.0, 60.0, 61.0, 62.0],
            "macd": [-1.0, -0.5, 0.5, 1.0, 1.0, 1.0],
            "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sector": ["Technology", "Technology", "Technology", "Financials", None, "Financials"],
        },
        index=index,
    )


def _scope(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {c: frame[c] for c in frame.columns}


def test_arithmetic_comparisons_and_boolean_operators() -> None:
    frame = _frame()
    expr = compile_expression(
        "close > sma_200 * 1.15 and (rsi_14 < 30 or not sector == 'Financials')"
    )

    assert expr.variables == {"close", "sma_200", "rsi_14", "sector"}
    assert expr.functions == frozenset()
    assert expr.lookback == 0
    assert list(expr(_scope(frame))) == [False, True, True, False, False, False]


def test_chained_comparison_and_nan_is_false() -> None:
    frame = _frame()
    expr = compile_expression("20 <= close < 22")

    result = expr(_scope(frame))
    assert list(result) == [False, False, False, True, False, False]  # NaN de MSFT no cumple


def test_constants_and_unary_minus() -> None:
    expr = compile_expression("-1_000 + 2 * 500 == 0")

    assert bool(expr({})) is True


def test_string_comparison_with_null_counts_as_different() -> None:
    frame = _frame()
    expr = compile_expression("sector != 'Financials'")

    assert list(expr(_scope(frame))) == [True, True, True, False, True, False]


def test_bullish_cross_does_not_leak_across_keys() -> None:
    frame = _frame()
    expr = compile_expression("bullish_cross(macd, macd_signal)")

    assert expr.functions == {"bullish_cross"}
    assert expr.lookback == 1
    # AAPL cruza el día 17; MSFT ya estaba encima y su primera fila no mira a AAPL.
    assert list(expr(_scope(frame))) == [False, False, True, False, False, False]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("close.mean()", "no se permite"),
        ("close[0]", "no se permite"),
        ("(lambda: 1)()", "no se permite"),
        ("__import__('os')", "función desconocida"),
        ("bullish_cross(macd)", "espera 2 argumentos"),
        ("bullish_cross(macd, macd_signal, 1)", "espera 2 argumentos"),
        ("bullish_cross(fast=macd, slow=macd_signal)", "no se permite"),
        ("close in sma_200", "comparación no permitida"),
        ("close **", "expresión inválida"),
        ("None", "no se permite"),
    ],
)
def test_disallowed_expressions_fail_at_compile_time(source: str, message: str) -> None:
    with pytest.raises(ExpressionError, match=message):
        compile_expression(source)


def test_unknown_variable_fails_at_evaluation_with_expression_in_message() -> None:
    expr = compile_expression("foo > 1")

    with pytest.raises(ExpressionError, match="'foo > 1': variable desconocida 'foo'"):
        expr({"close": pd.Series([1.0])})
