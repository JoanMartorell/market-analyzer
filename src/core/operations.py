"""Un mensaje por operación recomendada, con el gráfico de lo que se ha visto.

El resumen del día dice qué hay; estos mensajes son los que se abren para
operar: un valor por mensaje, con su dirección (COMPRAR o VENDER EN CORTO),
el precio de referencia, los niveles del patrón y la imagen de la figura.

Qué cuenta como operación recomendada:

- Con veredicto del modelo, las señales confirmadas. Las de vigilar y las
  descartadas se quedan en el resumen.
- Sin modelo, todos los candidatos de las reglas, y el mensaje lo dice.

Un valor marcado por dos reglas en la misma dirección sale una vez, por la
regla que trae patrón. El gráfico se vuelve a calcular aquí con las mismas
velas y la misma ventana que usó el paso de indicadores: el detector es
determinista, así que la figura dibujada es la que originó la señal. Si no
se puede dibujar, el mensaje sale igual y lo avisa: una operación no se
calla por un gráfico.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import structlog

from analyzer.patterns import PatternMatch, find_patterns
from analyzer.steps.llm.schema import CONFIRM
from core.charts import render_chart
from core.digest import HEADINGS, TAGS, _brief, pattern_label
from delivery import Message

log = structlog.get_logger(__name__)

Key = tuple[str, str]
NO_CHART = "Sin gráfico: no se ha podido dibujar."


def recommended_operations(signals: pd.DataFrame, analysed: bool) -> pd.DataFrame:
    """Las señales que piden operar, una por valor y dirección, la más segura primero."""
    if signals.empty:
        return signals
    rows = signals.loc[signals["verdict"] == CONFIRM] if analysed else signals
    by = ["confidence", "score"] if analysed else ["score"]
    rows = rows.assign(_figure=rows["pattern"].notna())
    preferred = rows.sort_values(["_figure", *by], ascending=False, kind="stable")
    unique = preferred.drop_duplicates(["ticker", "mic", "direction"])
    ordered = unique.sort_values([*by, "ticker"], ascending=[False] * len(by) + [True])
    return ordered.drop(columns="_figure").reset_index(drop=True)


def build_operation_messages(
    signals: pd.DataFrame,
    *,
    region_id: str,
    analysed: bool,
    bars: Mapping[Key, pd.DataFrame],
    names: Mapping[Key, str] | None = None,
) -> list[Message]:
    """Un mensaje por operación de ``recommended_operations``.

    ``bars`` son las velas de prod de cada valor hasta el día de la señal,
    ordenadas por fecha; sin velas de un valor, su mensaje sale sin gráfico.
    """
    messages: list[Message] = []
    for _, row in recommended_operations(signals, analysed).iterrows():
        key = (str(row["ticker"]), str(row["mic"]))
        image = _chart(row, bars.get(key))
        name = (names or {}).get(key)
        messages.append(_message(row, region_id, analysed, image, name))
    return messages


def find_match(bars: pd.DataFrame, pattern: str) -> PatternMatch | None:
    """El patrón ``pattern`` que se completa en la última vela de ``bars``, si lo hay."""
    last = len(bars) - 1
    return next((m for m in find_patterns(bars) if m.bar == last and m.name == pattern), None)


def _chart(row: pd.Series, bars: pd.DataFrame | None) -> bytes | None:
    ticker = str(row["ticker"])
    if bars is None or bars.empty:
        log.error("operations.chart_failed", ticker=ticker, error="sin velas")
        return None
    label = pattern_label(row)
    try:
        match = find_match(bars, str(row["pattern"])) if label else None
        if label and match is None:
            # No debería pasar con las mismas velas; se dibuja el precio y se deja rastro.
            log.warning("operations.pattern_not_found", ticker=ticker, pattern=row["pattern"])
        return render_chart(bars, title=_title(row, label), subtitle=_subtitle(row), match=match)
    except Exception as exc:  # un fallo de dibujo no puede dejar sin enviar la operación
        log.error("operations.chart_failed", ticker=ticker, error=str(exc))
        return None


def _message(
    row: pd.Series, region_id: str, analysed: bool, image: bytes | None, name: str | None
) -> Message:
    direction = str(row["direction"])
    label = pattern_label(row) or str(row["rule_id"])
    heading = HEADINGS.get(direction, direction)
    lines = [f"{name} · {row['mic']}" if name else str(row["mic"])]
    lines.append(
        f"Cierre {_money(row['close'], row.get('currency'))} del {_day(row['date'])} · "
        f"se ejecuta en la apertura del {_day(row['execute_on'])}"
    )
    levels = _levels(row)
    if levels:
        lines.append(levels)
    if analysed:
        headline = row.get("headline")
        why = headline if isinstance(headline, str) and headline else row.get("rationale")
        lines.append(f"Modelo {float(row['confidence']):.0%}: {_brief(str(why or ''))}")
    else:
        lines.append("Sin veredicto del modelo: lo recomienda la regla.")
    lines.append(f"Regla {row['rule_id']} · score {float(row['score']):.2f}")
    if image is None:
        lines.append(NO_CHART)
    return Message(
        subject=f"[{region_id}] {heading} {row['ticker']} · {label}",
        body="\n".join(lines),
        image=image,
    )


def _levels(row: pd.Series) -> str:
    """``Stop 95.10 (-4.1%) · objetivo 108.40 (+9.3%)``, con la distancia al cierre."""
    close = float(row["close"])
    parts = []
    for name, column in (("Stop", "stop_price"), ("objetivo", "target_price")):
        level = row.get(column)
        if level is not None and pd.notna(level):
            parts.append(f"{name} {float(level):.2f} ({float(level) / close - 1:+.1%})")
    return " · ".join(parts)


def _title(row: pd.Series, label: str | None) -> str:
    direction = str(row["direction"])
    parts = [str(row["ticker"]), label or str(row["rule_id"]), TAGS.get(direction, direction)]
    return " · ".join(parts)


def _subtitle(row: pd.Series) -> str:
    return f"Cierre {_money(row['close'], row.get('currency'))} · {_day(row['date'])}"


def _money(value: Any, currency: Any) -> str:
    price = f"{float(value):.2f}"
    return f"{price} {currency}" if isinstance(currency, str) and currency else price


def _day(value: Any) -> str:
    return str(pd.Timestamp(value).date())
