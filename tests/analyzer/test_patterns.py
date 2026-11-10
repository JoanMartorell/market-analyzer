"""Patrones chartistas: pivotes, figuras, causalidad, escala ajustada y evaluación."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analyzer.patterns import (
    DEFAULT_PARAMS,
    PATTERN_COLUMNS,
    PATTERN_WARMUP,
    ZigZag,
    average_true_range,
    find_patterns,
    pattern_columns,
    scan,
)
from analyzer.patterns.evaluate import OPEN, STOP, TARGET, VOID, evaluate_key, evaluate_prices
from analyzer.patterns.pivots import Array

END = date(2026, 9, 17)
Waypoints = list[tuple[int, float]]

# Caída desde 130, suelo en 100, rebote a 110, segundo suelo en 100 y ruptura de 110.
DOUBLE_BOTTOM: Waypoints = [(0, 120), (20, 130), (40, 100), (50, 110), (60, 100), (70, 115)]
# Caída desde 135, hombro en 105, cabeza en 95, hombro en 105 y ruptura de la línea de 115.
INVERSE_HS: Waypoints = [
    (0, 125),
    (20, 135),
    (35, 105),
    (45, 115),
    (60, 95),
    (75, 115),
    (85, 105),
    (100, 125),
]


def _bars(waypoints: Waypoints, *, mirror: float | None = None) -> pd.DataFrame:
    """Velas que unen los puntos con rectas; ``mirror`` las refleja (suelos -> techos)."""
    xs, ys = zip(*waypoints, strict=True)
    close = np.interp(np.arange(xs[-1] + 1), xs, ys)
    if mirror is not None:
        close = mirror - close
    return _frame(close)


def _frame(close: Array) -> pd.DataFrame:
    days = len(close)
    dates = [pd.Timestamp(END - timedelta(days=i)) for i in range(days - 1, -1, -1)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
        }
    )


def _walk(seed: int, days: int) -> pd.DataFrame:
    """Paseo aleatorio con tramos largos: da para varios pivotes y algún patrón."""
    rng = np.random.default_rng(seed)
    return _frame(100 + (rng.standard_normal(days) * 1.5).cumsum())


# --- pivotes --------------------------------------------------------------------


def test_zigzag_confirms_alternating_pivots_only_after_the_reversal() -> None:
    bars = _bars(DOUBLE_BOTTOM)
    high, low, close = (bars[c].to_numpy(dtype="float64") for c in ("high", "low", "close"))
    zigzag = ZigZag(high, low, average_true_range(high, low, close, 14), 3.0)
    for t in range(len(bars)):
        zigzag.step(t)

    # El primero es el arranque de la serie (primera vela con ATR); después, los giros.
    kinds = [(p.kind, p.index) for p in zigzag.pivots]
    assert kinds == [("L", 13), ("H", 20), ("L", 40), ("H", 50), ("L", 60)]
    assert all(p.confirmed_at > p.index for p in zigzag.pivots)
    assert zigzag.pivots[2].price == pytest.approx(99.5)  # el mínimo de la vela, no el cierre


def test_zigzag_ignores_bars_without_atr_or_with_gaps_in_the_data() -> None:
    bars = _bars(DOUBLE_BOTTOM)
    bars.loc[45, ["high", "low"]] = np.nan  # una vela incompleta en mitad del rebote

    matches = find_patterns(bars)

    assert [m.name for m in matches] == ["double_bottom"]


# --- figuras --------------------------------------------------------------------


def test_double_bottom_completes_on_the_close_above_the_bounce_high() -> None:
    (match,) = find_patterns(_bars(DOUBLE_BOTTOM))

    assert (match.name, match.direction, match.bar) == ("double_bottom", "long", 68)
    assert match.entry == pytest.approx(110.5)  # máximo del rebote
    assert match.target == pytest.approx(110.5 + 11.0)  # la altura de la figura, proyectada
    assert 97.0 < match.stop < 99.5  # bajo los suelos, con holgura
    assert [p.index for p in match.pivots] == [20, 40, 50, 60]


def test_double_top_is_the_mirror_image_and_goes_short() -> None:
    (match,) = find_patterns(_bars(DOUBLE_BOTTOM, mirror=230.0))

    assert (match.name, match.direction, match.bar) == ("double_top", "short", 68)
    assert match.entry == pytest.approx(119.5)
    assert match.target == pytest.approx(119.5 - 11.0)
    assert 130.5 < match.stop < 133.0


def test_inverse_head_and_shoulders_breaks_the_neckline() -> None:
    (match,) = find_patterns(_bars(INVERSE_HS))

    assert (match.name, match.direction, match.bar) == ("inv_head_shoulders", "long", 93)
    assert match.entry == pytest.approx(115.5)
    assert match.target == pytest.approx(115.5 + 21.0)  # de la cabeza a la línea clavicular
    assert 102.0 < match.stop < 104.5  # bajo el hombro derecho, no bajo la cabeza
    assert len(match.pivots) == 6


def test_head_and_shoulders_is_the_mirror_image_and_goes_short() -> None:
    (match,) = find_patterns(_bars(INVERSE_HS, mirror=230.0))

    assert (match.name, match.direction, match.bar) == ("head_shoulders", "short", 93)
    assert match.target == pytest.approx(114.5 - 21.0)


@pytest.mark.parametrize(
    ("waypoints", "why"),
    [
        ([(0, 120), (20, 130), (40, 100), (50, 110), (60, 100), (70, 109)], "no rompe"),
        ([(0, 120), (20, 130), (40, 100), (50, 110), (60, 104), (70, 115)], "suelos desiguales"),
        ([(0, 100), (20, 112), (40, 100), (50, 110), (60, 100), (70, 115)], "sin caída previa"),
        ([(0, 120), (20, 130), (40, 100), (43, 110), (46, 100), (60, 115)], "suelos pegados"),
    ],
)
def test_near_misses_are_not_a_double_bottom(waypoints: Waypoints, why: str) -> None:
    assert find_patterns(_bars(waypoints)) == [], why


def test_a_breakout_that_already_reached_the_target_is_not_a_signal() -> None:
    bars = _bars([(0, 120), (20, 130), (40, 100), (50, 110), (60, 100), (67, 110)])
    gap = bars.iloc[[-1]].copy()
    gap[["open", "close", "adj_close"]] = 125.0  # salta la ruptura y el objetivo de una vez
    gap["high"], gap["low"] = 125.5, 124.5
    gap["date"] = gap["date"] + pd.Timedelta(days=1)

    assert find_patterns(pd.concat([bars, gap], ignore_index=True)) == []


# --- sin mirar al futuro --------------------------------------------------------


def test_what_is_found_on_a_day_does_not_change_when_later_days_arrive() -> None:
    found = 0
    for seed in range(6):
        bars = _walk(seed, 420)
        full = find_patterns(bars)
        found += len(full)
        for cut in range(60, len(bars), 45):
            assert find_patterns(bars.iloc[:cut]) == [m for m in full if m.bar < cut]
    assert found > 0  # el paseo da patrones: la comprobación no pasa en vacío


def test_rows_past_the_warmup_do_not_depend_on_where_the_window_starts() -> None:
    compared = 0
    for seed in range(12):
        bars = _walk(seed, 520)
        full = pattern_columns(bars)
        window = 270
        tail = pattern_columns(bars.iloc[-window:].reset_index(drop=True))
        kept = window - PATTERN_WARMUP  # las filas que el servicio escribiría
        expected = full.iloc[-kept:].reset_index(drop=True)
        pd.testing.assert_frame_equal(tail.iloc[-kept:].reset_index(drop=True), expected)
        compared += int(expected.loc[:, list(PATTERN_COLUMNS[:4])].to_numpy().sum())
    assert compared > 0


# --- columnas y escala ----------------------------------------------------------


def test_pattern_columns_flag_only_the_breakout_bar() -> None:
    values = pattern_columns(_bars(DOUBLE_BOTTOM))

    assert tuple(values.columns) == PATTERN_COLUMNS
    assert values["pat_double_bottom"].sum() == 1.0
    assert values["pat_double_bottom"].iloc[68] == 1.0
    assert values.loc[:, ["pat_double_top", "pat_head_shoulders"]].to_numpy().sum() == 0.0
    assert values["pat_stop"].notna().sum() == 1
    assert values["pat_entry"].iloc[68] == pytest.approx(110.5)
    assert values["pat_target"].iloc[68] == pytest.approx(121.5)


def test_levels_come_back_in_the_real_price_of_the_breakout_day() -> None:
    bars = _bars(DOUBLE_BOTTOM)
    bars["adj_close"] = bars["close"] * 0.5  # todo el histórico ajustado a la mitad

    values = pattern_columns(bars)
    (match,) = find_patterns(bars)

    assert match.entry == pytest.approx(55.25)  # la figura se busca en escala ajustada
    assert values["pat_entry"].iloc[68] == pytest.approx(110.5)  # y se devuelve al precio real
    assert values["pat_target"].iloc[68] == pytest.approx(121.5)


def test_short_history_gives_no_patterns_and_no_errors() -> None:
    values = pattern_columns(_frame(np.linspace(100, 105, 10, dtype=np.float64)))

    assert len(values) == 10
    assert values.loc[:, list(PATTERN_COLUMNS[:4])].to_numpy().sum() == 0.0
    assert values["pat_stop"].isna().all()


def test_scan_takes_plain_arrays() -> None:
    bars = _bars(DOUBLE_BOTTOM)
    high, low, close = (bars[c].to_numpy(dtype="float64") for c in ("high", "low", "close"))

    assert [m.name for m in scan(high, low, close, DEFAULT_PARAMS)] == ["double_bottom"]


# --- evaluación -----------------------------------------------------------------


def _outcome(after: Waypoints) -> pd.Series:
    """Desenlace del doble suelo según cómo siga el precio tras la ruptura (vela 68)."""
    events = evaluate_key(_bars([*DOUBLE_BOTTOM[:-1], *after]), horizon=40)
    assert len(events) == 1
    return events.iloc[0]


def test_evaluation_follows_the_trade_from_the_next_open() -> None:
    hit = _outcome([(70, 115), (80, 130)])
    assert (hit["pattern"], hit["outcome"]) == ("double_bottom", TARGET)
    assert hit["entry"] == pytest.approx(113.5)  # apertura de la vela 69, no el cierre de la 68
    assert hit["ret"] == pytest.approx(121.5 / 113.5 - 1)

    stopped = _outcome([(70, 115), (90, 90)])
    assert stopped["outcome"] == STOP
    assert stopped["ret"] < 0

    drifting = _outcome([(70, 115), (120, 116)])
    assert (drifting["outcome"], drifting["sessions"]) == (OPEN, 40)


def test_evaluation_voids_a_trade_when_the_open_gaps_past_a_level() -> None:
    bars = _bars([*DOUBLE_BOTTOM[:-1], (68, 111.6)])  # la ruptura es la última vela
    gap = bars.iloc[[-1]].copy()
    gap[["open", "close", "adj_close"]] = 125.0
    gap["high"], gap["low"] = 125.5, 124.5
    gap["date"] = gap["date"] + pd.Timedelta(days=1)

    events = evaluate_key(pd.concat([bars, gap], ignore_index=True))

    assert events["outcome"].tolist() == [VOID]


def test_evaluation_skips_a_breakout_on_the_last_bar() -> None:
    assert evaluate_key(_bars([*DOUBLE_BOTTOM[:-1], (68, 111.6)])).empty


def test_stats_summarise_each_pattern_across_the_universe() -> None:
    up = _bars([*DOUBLE_BOTTOM, (80, 130)]).assign(ticker="AAA", mic="XNAS")
    down = _bars([*DOUBLE_BOTTOM, (90, 90)]).assign(ticker="BBB", mic="XNAS")
    flat = _frame(np.full(60, 50.0)).assign(ticker="CCC", mic="XNAS")

    stats = evaluate_prices(pd.concat([up, down, flat], ignore_index=True))

    assert stats.summary() == "2 patrones en 3 valores: doble suelo 2 (50% al objetivo)"
    row = stats.table().iloc[0]
    assert (row["hallazgos"], row["objetivo"], row["stop"]) == (2, 1, 1)
    assert evaluate_prices(flat).summary() == "sin patrones en 1 valores"
    with pytest.raises(ValueError, match="horizonte"):
        evaluate_prices(flat, horizon=0)
