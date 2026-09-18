"""Señales del día: candidatos con su procedencia y su veredicto, guardados.

El paso no decide nada nuevo: junta lo que decidieron los anteriores y lo
deja donde se pueda consultar dentro de un año. Cada señal lleva el hash de
la regla que la marcó y el del panel del que salió; si el modelo la analizó,
su veredicto. Los que el modelo descarta se guardan igual, con su
``descartar``: son la única forma de medir después si el modelo acierta.

Se falla en cerrado si algo no cuadra entre pasos: un candidato sin
veredicto habiendo análisis, o un mercado sin sesión siguiente. Antes que
guardar una señal a medias, mejor no guardar ninguna.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import structlog

from analyzer.steps.llm.schema import VERDICTS, Analysis, Verdict
from analyzer.steps.screener.service import CANDIDATE_COLUMNS
from analyzer.steps.snapshot.service import SHORT_HASH
from analyzer.storage import SIGNAL_COLUMNS, SignalStore

log = structlog.get_logger(__name__)

SIGNAL = ("señal", "señales")


@dataclass(frozen=True)
class PersistReport:
    region_id: str
    as_of: date
    snapshot_hash: str
    saved: int
    replaced: int  # señales del mismo día que había de una ejecución anterior
    analysed: bool  # llevan veredicto del modelo
    signals: pd.DataFrame = field(repr=False, compare=False)

    def count(self, verdict: str) -> int:
        return int((self.signals["verdict"] == verdict).sum()) if self.saved else 0

    @property
    def execute_on(self) -> list[date]:
        """Sesiones en las que se ejecutan las señales, normalmente una."""
        return sorted(self.signals["execute_on"].unique()) if self.saved else []

    def summary(self) -> str:
        if not self.saved:
            text = "sin señales"
        else:
            when = ", ".join(str(d) for d in self.execute_on)
            text = f"{_count(self.saved, SIGNAL)} para {when}"
            if self.analysed:
                counts = ", ".join(f"{self.count(v)} {v}" for v in VERDICTS if self.count(v))
                text += f": {counts}"
            else:
                text += " sin veredicto del modelo"
        text += f"; panel {self.snapshot_hash[:SHORT_HASH]}"
        if self.replaced:
            text += f"; sustituyen a {self.replaced} anteriores"
        return text


def run_persist(
    candidates: pd.DataFrame,
    analysis: Analysis | None,
    as_of: date,
    *,
    region_id: str,
    snapshot_hash: str,
    execute_on: Mapping[str, date],
    model: str | None,
    store: SignalStore,
    panel: pd.DataFrame | None = None,
) -> PersistReport:
    signals = build_signals(
        candidates,
        analysis,
        region_id=region_id,
        as_of=as_of,
        snapshot_hash=snapshot_hash,
        execute_on=execute_on,
        model=model,
        panel=panel,
    )
    replaced, saved = store.replace_day(region_id, as_of, signals)
    report = PersistReport(
        region_id=region_id,
        as_of=as_of,
        snapshot_hash=snapshot_hash,
        saved=saved,
        replaced=replaced,
        analysed=analysis is not None,
        signals=signals,
    )
    log.info("persist.done", detail=report.summary())
    return report


def build_signals(
    candidates: pd.DataFrame,
    analysis: Analysis | None,
    *,
    region_id: str,
    as_of: date,
    snapshot_hash: str,
    execute_on: Mapping[str, date],
    model: str | None,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Una fila de ``signals`` por candidato, con las columnas de la tabla menos ``created_at``."""
    missing = [c for c in CANDIDATE_COLUMNS if c not in candidates.columns]
    if missing:
        raise ValueError(f"faltan columnas de los candidatos: {', '.join(missing)}")
    if not snapshot_hash:
        raise ValueError("no hay hash del panel: la señal no se podría explicar después")

    signals = candidates.loc[:, list(CANDIDATE_COLUMNS)].copy()
    signals["ticker"] = signals["ticker"].astype(str)
    signals["mic"] = signals["mic"].astype(str)
    signals["region"] = region_id
    signals["date"] = as_of
    signals["snapshot_hash"] = snapshot_hash
    signals["currency"] = _currencies(signals, panel)
    signals["execute_on"] = _execution_dates(signals["mic"], execute_on)

    verdicts = _verdicts(signals["ticker"], analysis)
    signals["verdict"] = [None if v is None else v.verdict for v in verdicts]
    signals["confidence"] = [None if v is None else v.confidence for v in verdicts]
    signals["rationale"] = [None if v is None else v.rationale for v in verdicts]
    signals["risks"] = pd.Series(
        [None if v is None else list(v.risks) for v in verdicts], index=signals.index, dtype=object
    )
    signals["llm_model"] = model if analysis is not None else None

    columns = [c for c in SIGNAL_COLUMNS if c != "created_at"]
    return signals.loc[:, columns].reset_index(drop=True)


def _verdicts(tickers: pd.Series, analysis: Analysis | None) -> list[Verdict | None]:
    """El veredicto de cada candidato. Con análisis, todos tienen que tenerlo."""
    if analysis is None:
        return [None] * len(tickers)
    by_ticker = {v.ticker: v for v in analysis.verdicts}
    unanswered = sorted(set(tickers) - set(by_ticker))
    if unanswered:
        raise ValueError(f"candidatos sin veredicto del modelo: {', '.join(unanswered)}")
    return [by_ticker[str(t)] for t in tickers]


def _execution_dates(mics: pd.Series, execute_on: Mapping[str, date]) -> list[date]:
    unknown = sorted(set(mics) - set(execute_on))
    if unknown:
        raise ValueError(f"sin sesión de ejecución para los mercados: {', '.join(unknown)}")
    return [execute_on[str(m)] for m in mics]


def _currencies(signals: pd.DataFrame, panel: pd.DataFrame | None) -> pd.Series:
    """Divisa de cada valor según el panel; sin panel, o sin el valor en él, nula."""
    if panel is None or panel.empty or "currency" not in panel.columns:
        return pd.Series([None] * len(signals), index=signals.index, dtype=object)
    attrs = panel.loc[:, ["ticker", "mic", "currency"]].drop_duplicates(["ticker", "mic"])
    merged = signals.loc[:, ["ticker", "mic"]].merge(attrs, on=["ticker", "mic"], how="left")
    currency = merged["currency"].astype(object).where(merged["currency"].notna(), None)
    return pd.Series(list(currency), index=signals.index, dtype=object)


def _count(number: int, names: tuple[str, str]) -> str:
    singular, plural = names
    return f"{number} {singular if number == 1 else plural}"
