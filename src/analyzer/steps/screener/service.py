"""Screener: reglas YAML sobre el panel -> candidatos con score.

Una regla tiene filtros de universo, condiciones ponderadas y una ventana.
Los filtros se evalúan sobre la fila de hoy y dejan fuera al valor entero
(liquidez, sector...). Cada condición se evalúa sobre las últimas ``window``
sesiones del valor y se cumple si es cierta en alguna de ellas: "RSI bajo
30 en los últimos cinco días" y "cruce alcista en los últimos cinco días"
pueden ser de sesiones distintas. El score es la suma de pesos de las
condiciones cumplidas; si falla una ``required``, es 0. Candidato si el
score llega al umbral de la regla.

El panel (paso ``snapshot``) dice qué valores se evalúan y aporta los
atributos del universo; las sesiones anteriores salen de ``prod.indicators``.
Las funciones del registro pueden necesitar sesiones previas a la ventana
(un cruce mira la sesión anterior): se cargan de más y no cuentan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import structlog

from analyzer.rules import KEY_LEVELS, Expression, Scope, compile_expression
from analyzer.steps.ingest.service import lookback_start
from analyzer.storage import UNIVERSE_COLUMNS, IndicatorStore
from core.config import Rule

log = structlog.get_logger(__name__)

RECENCY = "_recency"  # 0 = sesión de as_of, 1 = la anterior...
CANDIDATE_COLUMNS = (
    "ticker",
    "mic",
    "rule_id",
    "direction",
    "score",
    "conditions_met",
    "close",
    "sector",
    "rule_hash",
)
TOP_IN_SUMMARY = 5


@dataclass(frozen=True)
class CompiledRule:
    rule: Rule
    filters: tuple[Expression, ...]
    conditions: tuple[Expression, ...]

    @property
    def lookback(self) -> int:
        return max((e.lookback for e in (*self.filters, *self.conditions)), default=0)

    @property
    def sessions(self) -> int:
        """Sesiones que hay que cargar por valor: la ventana más lo que miran las funciones."""
        return self.rule.window + self.lookback

    @property
    def variables(self) -> frozenset[str]:
        names: set[str] = set()
        for expression in (*self.filters, *self.conditions):
            names |= expression.variables
        return frozenset(names)


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    eligible: int  # valores que pasan los filtros de universo
    candidates: int


@dataclass(frozen=True)
class ScreenerReport:
    as_of: date
    evaluated: int  # valores del panel
    illiquid: int  # descartados por el volumen medio mínimo de la región
    rules: tuple[RuleResult, ...]
    candidates: pd.DataFrame = field(repr=False, compare=False)

    def summary(self) -> str:
        count = len(self.candidates)
        head = {0: "sin candidatos", 1: "1 candidato"}.get(count, f"{count} candidatos")
        text = f"{head} entre {self.evaluated} valores"
        if self.illiquid:
            text += f" ({self.illiquid} sin liquidez)"
        text += ": " + ", ".join(f"{r.rule_id} {r.candidates}/{r.eligible}" for r in self.rules)
        if count:
            top = self.candidates.head(TOP_IN_SUMMARY)
            listed = ", ".join(
                f"{t} {s:.2f}" for t, s in zip(top["ticker"], top["score"], strict=True)
            )
            text += f"; {listed}"
            if count > TOP_IN_SUMMARY:
                text += ", ..."
        return text


def run_screener(
    panel: pd.DataFrame,
    as_of: date,
    *,
    rules: Sequence[Rule],
    indicators: IndicatorStore,
    max_candidates: int,
    min_score: float,
    min_avg_volume: int,
) -> ScreenerReport:
    if not rules:
        raise ValueError("la región no tiene reglas")
    compiled = [compile_rule(rule) for rule in rules]
    check_variables(compiled, set(panel.columns))

    liquid = panel["avg_volume_20"] >= min_avg_volume  # NaN (histórico corto) queda fuera
    illiquid = int((~liquid).sum())
    frame = load_window(indicators, panel.loc[liquid], as_of, max(c.sessions for c in compiled))

    results: list[RuleResult] = []
    found: list[pd.DataFrame] = []
    for item in compiled:
        scores = evaluate_rule(item, frame)
        threshold = max(item.rule.output.score_threshold, min_score)
        chosen = scores.loc[scores["eligible"] & (scores["score"] >= threshold)]
        results.append(RuleResult(item.rule.id, int(scores["eligible"].sum()), len(chosen)))
        found.append(_as_candidates(chosen, item.rule, frame))

    candidates = pd.concat(found, ignore_index=True) if found else _empty_candidates()
    candidates = candidates.sort_values(
        ["score", "ticker", "mic", "rule_id"], ascending=[False, True, True, True]
    ).head(max_candidates)
    report = ScreenerReport(
        as_of=as_of,
        evaluated=len(panel),
        illiquid=illiquid,
        rules=tuple(results),
        candidates=candidates.reset_index(drop=True),
    )
    log.info("screener.done", detail=report.summary())
    return report


def compile_rule(rule: Rule) -> CompiledRule:
    return CompiledRule(
        rule=rule,
        filters=tuple(compile_expression(f) for f in rule.universe_filters),
        conditions=tuple(compile_expression(c.expr) for c in rule.conditions),
    )


def check_variables(compiled: Sequence[CompiledRule], available: set[str]) -> None:
    """Falla antes de evaluar si una regla usa una columna que el panel no tiene."""
    for item in compiled:
        unknown = sorted(item.variables - available)
        if unknown:
            raise ValueError(f"regla {item.rule.id!r}: variables desconocidas {', '.join(unknown)}")


def load_window(
    indicators: IndicatorStore, panel: pd.DataFrame, as_of: date, sessions: int
) -> pd.DataFrame:
    """Últimas ``sessions`` filas de cada valor del panel, indexadas por (ticker, mic, date).

    Lleva los atributos del universo del panel y ``_recency`` (0 en ``as_of``).
    Todo valor del panel debe tener fila de ``as_of``; si no, prod cambió
    entre pasos y se falla en cerrado.
    """
    keys = panel.loc[:, ["ticker", "mic"]]
    history = indicators.load(start=lookback_start(as_of, sessions), end=as_of, keys=keys)
    history = history.drop(columns=["computed_at"]).sort_values([*KEY_LEVELS, "date"])
    grouped = history.groupby(KEY_LEVELS, sort=False)
    history = grouped.tail(sessions).copy()
    history[RECENCY] = history.groupby(KEY_LEVELS, sort=False).cumcount(ascending=False)
    history["volume"] = history["volume"].astype("float64")  # sin enteros nullable en las reglas

    with_today = history.loc[history[RECENCY] == 0, KEY_LEVELS]
    missing = keys.merge(with_today, how="left", indicator=True)
    stale = missing.loc[missing["_merge"] != "both"]
    if not stale.empty:
        raise ValueError(f"{len(stale)} valores del panel sin indicadores de {as_of} en prod")
    if not (history.loc[history[RECENCY] == 0, "date"] == pd.Timestamp(as_of)).all():
        raise ValueError(f"hay valores cuya última sesión en prod no es {as_of}")

    attrs = panel.loc[:, ["ticker", "mic", *UNIVERSE_COLUMNS]]
    frame = history.merge(attrs, on=KEY_LEVELS, how="inner")
    return frame.set_index([*KEY_LEVELS, "date"]).sort_index()


def evaluate_rule(item: CompiledRule, frame: pd.DataFrame) -> pd.DataFrame:
    """Por valor: ``eligible`` (filtros hoy), ``score`` y ``conditions_met`` (texto)."""
    scope: Scope = {column: frame[column] for column in frame.columns}
    today = frame[RECENCY] == 0
    in_window = frame[RECENCY] < item.rule.window
    index = frame.index
    if not isinstance(index, pd.MultiIndex):
        raise ValueError("la ventana debe ir indexada por (ticker, mic, date)")
    keys = index.droplevel("date").unique()

    eligible = pd.Series(True, index=keys)
    for expression in item.filters:
        passes = _as_bool(expression(scope), frame.index)
        eligible &= passes.loc[today].droplevel("date").reindex(keys, fill_value=False)

    score = pd.Series(0.0, index=keys)
    blocked = pd.Series(False, index=keys)
    met_labels: dict[str, pd.Series] = {}
    for condition, expression in zip(item.rule.conditions, item.conditions, strict=True):
        hit = _as_bool(expression(scope), frame.index) & in_window
        met = hit.groupby(level=KEY_LEVELS, sort=False).any().reindex(keys, fill_value=False)
        score += condition.weight * met
        if condition.required:
            blocked |= ~met
        met_labels[condition.expr] = met
    score[blocked] = 0.0

    labels = pd.DataFrame(met_labels, index=keys)
    conditions_met = pd.Series(
        ["; ".join(labels.columns[list(row)]) for row in labels.itertuples(index=False)],
        index=keys,
        dtype=object,
    )
    return pd.DataFrame(
        {"eligible": eligible, "score": score.round(6), "conditions_met": conditions_met}
    )


def _as_bool(value: object, index: pd.Index) -> pd.Series:
    """Resultado de una expresión como serie booleana sobre ``index``; NaN cuenta como falso."""
    if isinstance(value, pd.Series):
        return pd.Series(np.asarray(value.fillna(False), dtype=bool), index=value.index)
    return pd.Series(bool(value), index=index)


def _as_candidates(chosen: pd.DataFrame, rule: Rule, frame: pd.DataFrame) -> pd.DataFrame:
    if chosen.empty:
        return _empty_candidates()
    today = frame.loc[frame[RECENCY] == 0, ["close", "sector"]].droplevel("date")
    rows = chosen.join(today, how="left").reset_index()
    rows["rule_id"] = rule.id
    rows["direction"] = rule.output.direction
    rows["rule_hash"] = rule.source_hash
    return rows.loc[:, list(CANDIDATE_COLUMNS)]


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))
