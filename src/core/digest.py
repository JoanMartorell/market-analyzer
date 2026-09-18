"""El mensaje del día: qué comprar, qué vender, qué vigilar, y por qué en una línea.

Es lo que llega al móvil cuando el ciclo termina bien, así que tiene que
caber en una pantalla: una lista por decisión, cada valor con su precio de
referencia, la confianza del modelo y el motivo en una línea (el
``headline`` que se le pide al modelo; el motivo largo se queda en la tabla
de señales). La tabla de pasos y los tiempos se quedan en el log; aquí solo
va un pie con lo que cuesta y lo que dio de sí la conciliación de ayer.

Sin veredicto del modelo (``llm.enabled: false`` o paso saltado) se listan
los candidatos por puntuación de la regla con las condiciones cumplidas
como motivo, y el mensaje lo dice.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd

from analyzer.engine import PipelineResult
from analyzer.steps.llm.schema import CONFIRM, DISCARD, WATCH
from delivery import Message

HEADINGS = {"long": "COMPRAR", "short": "VENDER"}
TAGS = {"long": "compra", "short": "venta"}
WHY_LIMIT = 160  # caracteres del motivo por valor; más no cabe en una línea del móvil
BULLET = "•"


def build_digest(result: PipelineResult, data: Mapping[str, Any]) -> Message:
    """Mensaje de un ciclo que terminó bien. Sin señales en el contexto, la tabla de pasos."""
    signals = data.get("signals")
    if not isinstance(signals, pd.DataFrame):
        return Message(subject=f"[{result.region_id}] ciclo {result.as_of}", body=result.summary())

    analysis = data.get("analysis")
    analysed = analysis is not None
    lines = _headline(signals, result.as_of)
    lines.extend(_decisions(signals, analysed))
    reading = getattr(analysis, "summary", None) if analysed else None
    if isinstance(reading, str) and reading:
        lines.extend(["", f"Lectura del día: {reading}"])
    lines.extend(_footer(result, data))
    subject = f"[{result.region_id}] {result.as_of}: {_counts(signals, analysed)}"
    return Message(subject=subject, body="\n".join(lines))


def _headline(signals: pd.DataFrame, as_of: date) -> list[str]:
    if signals.empty:
        return [f"Sin señales el {as_of}."]
    when = ", ".join(sorted({str(_day(d)) for d in signals["execute_on"]}))
    return [f"{_count(len(signals), 'señal', 'señales')} del {as_of}, a ejecutar el {when}"]


def _decisions(signals: pd.DataFrame, analysed: bool) -> list[str]:
    if signals.empty:
        return []
    lines: list[str] = []
    ordered = _ordered(signals, analysed)
    if analysed:
        confirmed = ordered[ordered["verdict"] == CONFIRM]
        for direction, heading in HEADINGS.items():
            rows = confirmed[confirmed["direction"] == direction]
            if len(rows):
                lines.extend(["", heading, *(_bullet(r, analysed) for _, r in rows.iterrows())])
        watch = ordered[ordered["verdict"] == WATCH]
        if len(watch):
            lines.extend(
                ["", "VIGILAR", *(_bullet(r, analysed, tag=True) for _, r in watch.iterrows())]
            )
        discarded = ordered[ordered["verdict"] == DISCARD]
        if len(discarded):
            names = ", ".join(sorted(set(discarded["ticker"].astype(str))))
            lines.extend(["", f"Descartadas por el modelo: {names}"])
        return lines
    for direction, heading in HEADINGS.items():
        rows = ordered[ordered["direction"] == direction]
        if len(rows):
            lines.append("")
            lines.append(f"{heading} (candidatos de las reglas, sin veredicto del modelo)")
            lines.extend(_bullet(r, analysed) for _, r in rows.iterrows())
    return lines


def _ordered(signals: pd.DataFrame, analysed: bool) -> pd.DataFrame:
    """Un valor por fila aunque lo marquen dos reglas; el más seguro primero."""
    by = ["confidence", "score"] if analysed else ["score"]
    ordered = signals.sort_values([*by, "ticker"], ascending=[False] * len(by) + [True])
    return ordered.drop_duplicates(["ticker", "mic"])


def _bullet(row: pd.Series, analysed: bool, *, tag: bool = False) -> str:
    parts = [str(row["ticker"])]
    if tag:
        parts[0] += f" ({TAGS.get(str(row['direction']), row['direction'])})"
    price = f"{float(row['close']):.2f}"
    currency = row.get("currency")
    parts.append(f"{price} {currency}" if isinstance(currency, str) else price)
    if analysed:
        parts.append(f"{float(row['confidence']):.0%}")
        # La línea corta que escribió el modelo; si no la hay, la primera frase del motivo.
        headline = row.get("headline")
        why = _brief(
            headline if isinstance(headline, str) and headline else str(row.get("rationale") or "")
        )
    else:
        parts.append(f"score {float(row['score']):.2f}")
        why = _brief(str(row.get("conditions_met") or ""))
    if why:
        parts.append(why)
    return f"{BULLET} {' · '.join(parts)}"


def _brief(text: str, limit: int = WHY_LIMIT) -> str:
    """La primera frase del motivo, y como mucho ``limit`` caracteres."""
    text = " ".join(text.split())
    for mark in (". ", "; "):
        cut = text.find(mark)
        if 0 < cut < limit:
            text = text[: cut + 1].rstrip(";")
            break
    if len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        text = text[: cut if cut > 0 else limit].rstrip(",;:") + "…"
    return text


def _footer(result: PipelineResult, data: Mapping[str, Any]) -> list[str]:
    lines = [""]
    reconciled = next((r.message for r in result.reports if r.name == "reconcile"), None)
    if reconciled:
        lines.append(f"Ayer: {reconciled}")
    tail = [f"ciclo {result.total_seconds:.0f} s"]
    llm = data.get("llm")
    if llm is not None:
        cost = "respuesta reutilizada" if llm.reused else f"{llm.cost_eur:.2f} EUR"
        tail.append(
            f"modelo {cost} ({llm.month_cost_eur:.2f} de {llm.budget_eur:.0f} EUR este mes)"
        )
    lines.append(" · ".join(tail))
    return lines


def _counts(signals: pd.DataFrame, analysed: bool) -> str:
    if signals.empty:
        return "sin señales"
    if not analysed:
        return f"{_count(signals['ticker'].nunique(), 'candidato', 'candidatos')} sin veredicto"
    one = signals.drop_duplicates(["ticker", "mic"])
    parts = []
    for direction, heading in HEADINGS.items():
        n = int(((one["verdict"] == CONFIRM) & (one["direction"] == direction)).sum())
        if n:
            parts.append(f"{n} {heading.lower()}")
    for verdict in (WATCH, DISCARD):
        n = int((one["verdict"] == verdict).sum())
        if n:
            parts.append(f"{n} {verdict}")
    return ", ".join(parts)


def _day(value: Any) -> date:
    return pd.Timestamp(value).date()


def _count(number: int, singular: str, plural: str) -> str:
    return f"{number} {singular if number == 1 else plural}"
