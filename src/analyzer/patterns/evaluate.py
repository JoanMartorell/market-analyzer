"""Qué pasó después de cada patrón: ¿llegó antes al objetivo o al stop?

Pasa el detector por el histórico de cada valor y sigue cada hallazgo como
lo seguiría la señal: se entra en la apertura de la sesión siguiente a la
ruptura y se sale en el primero de los dos niveles que toque el precio.
Sirve para decidir qué patrones merecen regla y con qué umbrales, antes de
fiarse de ninguno.

Los criterios son los prudentes:

- Si una misma vela toca stop y objetivo, cuenta como stop.
- Si la apertura de entrada ya está más allá de uno de los dos niveles, la
  operación queda ``anulada``: el hueco se comió el planteamiento.
- Lo que no toca ningún nivel en ``horizon`` sesiones queda ``abierta`` y
  se valora al último cierre.

El retorno es bruto, a favor de la dirección de la señal y sin costes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analyzer.patterns.detect import adjusted_ohlc, scan
from analyzer.patterns.model import DEFAULT_PARAMS, PATTERNS, PatternMatch, PatternParams
from analyzer.patterns.pivots import Array

TARGET, STOP, OPEN, VOID = "objetivo", "stop", "abierta", "anulada"
OUTCOMES = (TARGET, STOP, OPEN, VOID)
DEFAULT_HORIZON = 40  # sesiones que se sigue una operación antes de darla por abierta
EVENT_COLUMNS = (
    "ticker",
    "mic",
    "date",
    "pattern",
    "direction",
    "entry",
    "stop",
    "target",
    "outcome",
    "sessions",
    "ret",
)


@dataclass(frozen=True)
class PatternStats:
    keys: int  # valores recorridos
    horizon: int
    events: pd.DataFrame = field(repr=False, compare=False)

    def table(self) -> pd.DataFrame:
        """Una fila por patrón: cuántos, cómo acabaron y qué rindieron de media."""
        rows = []
        for spec in PATTERNS:
            found = self.events.loc[self.events["pattern"] == spec.name]
            traded = found.loc[found["outcome"] != VOID]
            closed = int(traded["outcome"].isin([TARGET, STOP]).sum())
            hits = int((traded["outcome"] == TARGET).sum())
            rows.append(
                {
                    "patron": spec.label,
                    "direccion": spec.direction,
                    "hallazgos": len(found),
                    "objetivo": hits,
                    "stop": int((traded["outcome"] == STOP).sum()),
                    "abiertas": int((traded["outcome"] == OPEN).sum()),
                    "anuladas": len(found) - len(traded),
                    "acierto": hits / closed if closed else float("nan"),
                    "retorno_medio": float(traded["ret"].mean()) if len(traded) else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        if self.events.empty:
            return f"sin patrones en {self.keys} valores"
        parts = []
        for row in self.table().to_dict("records"):
            if not row["hallazgos"]:
                continue
            text = f"{row['patron']} {row['hallazgos']}"
            if not np.isnan(row["acierto"]):
                text += f" ({row['acierto']:.0%} al objetivo)"
            parts.append(text)
        noun = "patrón" if len(self.events) == 1 else "patrones"
        return f"{len(self.events)} {noun} en {self.keys} valores: {', '.join(parts)}"


def evaluate_prices(
    prices: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    params: PatternParams = DEFAULT_PARAMS,
) -> PatternStats:
    """Detecta y sigue los patrones de todas las claves de ``prices`` (velas de prod)."""
    if horizon < 1:
        raise ValueError(f"el horizonte debe ser de al menos una sesión: {horizon}")
    parts: list[pd.DataFrame] = []
    keys = 0
    for (ticker, mic), bars in prices.groupby(["ticker", "mic"], sort=True):
        keys += 1
        events = evaluate_key(bars.sort_values("date"), horizon=horizon, params=params)
        if not events.empty:
            parts.append(events.assign(ticker=str(ticker), mic=str(mic)))
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    events = events.reindex(columns=list(EVENT_COLUMNS))
    return PatternStats(keys=keys, horizon=horizon, events=events)


def evaluate_key(
    bars: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    params: PatternParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Los patrones de un valor con su desenlace; sin ``ticker`` ni ``mic``."""
    ohlc = adjusted_ohlc(bars.reset_index(drop=True))
    opens, highs, lows, closes = (
        ohlc[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close")
    )
    dates = pd.to_datetime(bars["date"]).to_numpy()
    rows = []
    for match in scan(highs, lows, closes, params):
        outcome = _follow(match, opens, highs, lows, closes, horizon)
        if outcome is None:
            continue  # ruptura en la última vela: todavía no hay sesión de entrada
        entry, result, sessions, ret = outcome
        rows.append(
            {
                "date": dates[match.bar],
                "pattern": match.name,
                "direction": match.direction,
                "entry": entry,
                "stop": match.stop,
                "target": match.target,
                "outcome": result,
                "sessions": sessions,
                "ret": ret,
            }
        )
    return pd.DataFrame(rows)


def _follow(
    match: PatternMatch,
    opens: Array,
    highs: Array,
    lows: Array,
    closes: Array,
    horizon: int,
) -> tuple[float, str, int, float] | None:
    """(entrada, desenlace, sesiones, retorno) de entrar en la apertura siguiente a la ruptura."""
    first = match.bar + 1
    if first >= len(opens) or not opens[first] > 0:
        return None
    sign = 1.0 if match.direction == "long" else -1.0
    entry = float(opens[first])
    # En largo el stop queda por debajo y el objetivo por encima; en corto, al revés.
    if sign * entry <= sign * match.stop or sign * entry >= sign * match.target:
        return entry, VOID, 0, float("nan")
    last = min(first + horizon, len(opens)) - 1
    for t in range(first, last + 1):
        adverse = lows[t] if sign > 0 else highs[t]
        favourable = highs[t] if sign > 0 else lows[t]
        if sign * adverse <= sign * match.stop:
            return entry, STOP, t - first + 1, sign * (match.stop / entry - 1.0)
        if sign * favourable >= sign * match.target:
            return entry, TARGET, t - first + 1, sign * (match.target / entry - 1.0)
    return entry, OPEN, last - first + 1, sign * (float(closes[last]) / entry - 1.0)
