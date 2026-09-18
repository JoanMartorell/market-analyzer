"""Validación y promoción: staging -> prod.

Lee de staging la ventana de ``lookback_days`` sesiones hasta ``as_of`` para
las claves del universo, aplica las comprobaciones y, si el día es válido,
escribe la ventana en prod sin las velas en cuarentena. Idempotente: promover
dos veces el mismo día deja prod igual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import structlog

from analyzer.steps.ingest.service import lookback_start, universe_keys
from analyzer.steps.quality.checks import Anomaly, feed_last_date, find_anomalies, keys_with_bar
from analyzer.storage import PriceStore
from core.config.schema import QualitySettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class QualityReport:
    as_of: date
    expected: int  # claves del universo
    covered: int  # con vela en as_of
    quarantined: int  # con vela pero anómala
    usable: int  # covered - quarantined
    coverage: float  # usable / expected
    feed_last_date: date | None
    anomalies: tuple[Anomaly, ...] = ()
    failures: tuple[str, ...] = ()
    promoted_rows: int = 0
    window_start: date | None = None
    notes: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def quarantined_keys(self) -> set[tuple[str, str]]:
        return {a.key for a in self.anomalies}

    def summary(self) -> str:
        text = f"cobertura {self.coverage:.1%} ({self.usable}/{self.expected})"
        if self.quarantined:
            by_reason: dict[str, int] = {}
            for a in self.anomalies:
                by_reason[a.reason] = by_reason.get(a.reason, 0) + 1
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items()))
            text += f", cuarentena {self.quarantined} ({reasons})"
        if self.ok:
            text += f", {self.promoted_rows} velas a prod"
        else:
            text += "; " + "; ".join(self.failures)
        return text


def run_quality(
    universe: pd.DataFrame,
    as_of: date,
    *,
    staging: PriceStore,
    prod: PriceStore,
    settings: QualitySettings,
    lookback_days: int,
) -> QualityReport:
    keys = universe_keys(universe)
    start = lookback_start(as_of, lookback_days)
    window = staging.load(start=start, end=as_of, keys=keys)

    expected = len(keys)
    with_bar = keys_with_bar(window, as_of)
    last_date = feed_last_date(window)
    anomalies = find_anomalies(
        window,
        as_of,
        max_daily_jump=settings.max_daily_jump,
        zero_volume_min_avg_volume=settings.zero_volume_min_avg_volume,
    )
    quarantined = {a.key for a in anomalies}
    usable = len(with_bar - quarantined)
    coverage = usable / expected if expected else 0.0

    failures: list[str] = []
    if settings.feed_date_must_be_last_session and last_date != as_of:
        failures.append(f"el feed llega hasta {last_date}, se esperaba {as_of}")
    if coverage < settings.min_ticker_coverage:
        failures.append(f"cobertura por debajo del mínimo {settings.min_ticker_coverage:.0%}")
    for anomaly in anomalies:
        log.warning(
            "quality.quarantine",
            ticker=anomaly.ticker,
            mic=anomaly.mic,
            reason=anomaly.reason,
            detail=anomaly.detail,
        )

    promoted = 0
    if not failures:
        promoted = promote(window, as_of, quarantined, prod)

    return QualityReport(
        as_of=as_of,
        expected=expected,
        covered=len(with_bar),
        quarantined=len(quarantined),
        usable=usable,
        coverage=coverage,
        feed_last_date=last_date,
        anomalies=tuple(anomalies),
        failures=tuple(failures),
        promoted_rows=promoted,
        window_start=start,
        notes={"window_rows": len(window)},
    )


def promote(
    window: pd.DataFrame, as_of: date, quarantined: set[tuple[str, str]], prod: PriceStore
) -> int:
    """Escribe la ventana en prod, sin la vela de ``as_of`` de las claves en cuarentena."""
    if window.empty:
        return 0
    if quarantined:
        is_day = window["date"] == pd.Timestamp(as_of)
        in_quarantine = pd.Series(
            [k in quarantined for k in zip(window["ticker"], window["mic"], strict=True)],
            index=window.index,
        )
        window = window.loc[~(is_day & in_quarantine)]
    return prod.upsert(window)
