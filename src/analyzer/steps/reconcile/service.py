"""Apertura real frente a cierre asumido: a qué precio se ejecutó de verdad la señal.

Una señal del día D se emite con el cierre de D como referencia y se
ejecuta en la apertura de la siguiente sesión de su mercado. Entre medias
hay una noche: resultados, noticias, el gap de apertura. Este paso mide
ese gap con la apertura que trajo el ciclo de la sesión de ejecución y lo
deja escrito en la propia señal, con un estado:

- ``conciliada``: la apertura quedó dentro del gap admitido; la señal se
  ejecutó en las condiciones que la regla vio.
- ``desviada``: el gap supera ``reconcile.max_open_gap`` en cualquier
  sentido. El precio ya era otro, y con él la señal.
- ``sin_precio``: no hay vela de esa sesión en prod para el valor
  (cuarentena, feed sin datos, valor que salió del índice).

El gap es ``apertura / cierre - 1`` con signo y sin mirar la dirección:
una apertura muy por debajo también es otro escenario aunque a un largo le
convenga. Sin este registro, regla y modelo se medirían después con precios
a los que nunca se entró.

Se falla en cerrado si no hay ninguna vela de la sesión que se concilia:
el ciclo diario de esa sesión no ha pasado y no hay nada real que comparar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import structlog

from analyzer.storage import RECONCILE_COLUMNS, SIGNAL_KEY, PriceStore, SignalStore

log = structlog.get_logger(__name__)

RECONCILED, DEVIATED, NO_PRICE = "conciliada", "desviada", "sin_precio"
RECONCILE_STATUSES = (RECONCILED, DEVIATED, NO_PRICE)
REQUIRED_SIGNAL_COLUMNS = (*SIGNAL_KEY, "close", "execute_on")
REQUIRED_CANDLE_COLUMNS = ("ticker", "mic", "date", "open")
SIGNAL = ("señal", "señales")
NAMED = 5  # valores que se citan en el resumen antes de pasar a "y N más"
GAP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ReconcileReport:
    region_id: str
    as_of: date  # sesión de ejecución conciliada
    max_open_gap: float
    signals: pd.DataFrame = field(repr=False, compare=False)  # con las columnas de conciliación

    @property
    def total(self) -> int:
        return len(self.signals)

    def count(self, status: str) -> int:
        return int((self.signals["reconcile_status"] == status).sum()) if self.total else 0

    @property
    def signal_dates(self) -> list[date]:
        """Días de las señales conciliadas: normalmente la sesión anterior a ``as_of``."""
        if not self.total:
            return []
        return sorted(pd.to_datetime(self.signals["date"]).dt.date.unique())

    def summary(self) -> str:
        if not self.total:
            return f"sin señales pendientes de conciliar a {self.as_of}"
        when = ", ".join(str(d) for d in self.signal_dates)
        parts = [f"{self.count(RECONCILED)} conciliadas"]
        deviated = self.signals[self.signals["reconcile_status"] == DEVIATED]
        if len(deviated):
            limit = f"más del {self.max_open_gap:.1%}"
            parts.append(f"{len(deviated)} desviadas {limit} ({_named_gaps(deviated)})")
        no_price = self.signals[self.signals["reconcile_status"] == NO_PRICE]
        if len(no_price):
            parts.append(f"{len(no_price)} sin precio ({_named(no_price['ticker'])})")
        text = f"{_count(self.total, SIGNAL)} del {when}: {', '.join(parts)}"
        gaps = self.signals["open_gap"].dropna()
        if len(gaps):
            worst = float(gaps.iloc[int(gaps.abs().argmax())])
            text += f"; gap medio {gaps.mean():+.1%}, máximo {worst:+.1%}"
        return text


def run_reconcile(
    region_id: str,
    as_of: date,
    *,
    signals: SignalStore,
    prices: PriceStore,
    max_open_gap: float,
) -> ReconcileReport:
    """Concilia las señales de ``region_id`` pendientes a ``as_of`` con las velas de prod."""
    pending = signals.pending_reconciliation(region_id, as_of)
    if pending.empty:
        reconciled = pending
    else:
        first = pending["execute_on"].min().date()
        candles = prices.load(start=first, end=as_of, keys=pending)
        executing_today = bool((pending["execute_on"] == pd.Timestamp(as_of)).any())
        has_today = bool((candles["date"] == pd.Timestamp(as_of)).any()) if len(candles) else False
        if candles.empty or (executing_today and not has_today):
            raise ValueError(
                f"sin precios en prod para {as_of}: el ciclo diario de esa sesión debe ir antes"
            )
        reconciled = reconcile_signals(pending, candles, max_open_gap=max_open_gap)
        updated = signals.reconcile(reconciled)
        if updated != len(reconciled):
            raise ValueError(
                f"conciliadas {len(reconciled)} señales pero se actualizaron {updated}: "
                "la tabla cambió durante el paso"
            )
    report = ReconcileReport(
        region_id=region_id, as_of=as_of, max_open_gap=max_open_gap, signals=reconciled
    )
    log.info("reconcile.done", detail=report.summary())
    return report


def reconcile_signals(
    signals: pd.DataFrame, candles: pd.DataFrame, *, max_open_gap: float
) -> pd.DataFrame:
    """Añade a cada señal la apertura real, el gap y su estado de conciliación.

    ``candles`` son velas de prod de las sesiones de ejecución; una señal
    sin vela ese día (o con apertura no positiva) queda ``sin_precio``. Las
    columnas de conciliación que trajeran las señales se recalculan.
    """
    _check_columns(signals, REQUIRED_SIGNAL_COLUMNS, "las señales")
    _check_columns(candles, REQUIRED_CANDLE_COLUMNS, "las velas")
    if not 0.0 < max_open_gap < 1.0:
        raise ValueError(f"max_open_gap debe estar entre 0 y 1: {max_open_gap}")

    out = signals.drop(columns=list(RECONCILE_COLUMNS), errors="ignore").reset_index(drop=True)
    close = out["close"].astype("float64")
    if not (close > 0).all():
        bad = sorted(set(out.loc[~(close > 0), "ticker"].astype(str)))
        raise ValueError(f"señales con cierre de referencia no positivo: {', '.join(bad)}")

    keys = out.loc[:, ["ticker", "mic"]].astype(str)
    keys["execute_on"] = pd.to_datetime(out["execute_on"]).dt.normalize()
    merged = keys.merge(_opens(candles), on=["ticker", "mic", "execute_on"], how="left")

    opens = merged["open"].astype("float64").to_numpy()
    execution_open = np.where(opens > 0, opens, np.nan)  # apertura no positiva = sin precio
    gap = execution_open / close.to_numpy() - 1.0
    # Un gap justo en el límite no se desvía: la tolerancia absorbe el ruido de coma flotante.
    beyond = (np.abs(gap) - max_open_gap) > GAP_TOLERANCE
    status = np.where(np.isnan(gap), NO_PRICE, np.where(beyond, DEVIATED, RECONCILED))
    out["execution_open"] = execution_open
    out["open_gap"] = gap
    out["reconcile_status"] = status
    return out


def _opens(candles: pd.DataFrame) -> pd.DataFrame:
    """Apertura por (ticker, mic, execute_on), con la fecha de la vela como sesión de ejecución."""
    opens = candles.loc[:, ["ticker", "mic", "open"]].copy()
    opens["ticker"] = opens["ticker"].astype(str)
    opens["mic"] = opens["mic"].astype(str)
    opens["execute_on"] = pd.to_datetime(candles["date"]).dt.normalize()
    return opens.drop_duplicates(["ticker", "mic", "execute_on"], keep="last")


def _check_columns(frame: pd.DataFrame, required: Sequence[str], what: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"faltan columnas de {what}: {', '.join(missing)}")


def _named_gaps(deviated: pd.DataFrame) -> str:
    """``GE +3.1%, SPG -2.4%`` de mayor a menor desvío; un valor con dos reglas se cita una vez."""
    rows = deviated.drop_duplicates("ticker")
    rows = rows.iloc[rows["open_gap"].abs().argsort()[::-1]]
    names = [f"{t} {g:+.1%}" for t, g in zip(rows["ticker"], rows["open_gap"], strict=True)]
    return _shorten(names)


def _named(tickers: pd.Series) -> str:
    return _shorten(sorted(set(tickers.astype(str))))


def _shorten(names: list[str]) -> str:
    if len(names) <= NAMED:
        return ", ".join(names)
    return f"{', '.join(names[:NAMED])} y {len(names) - NAMED} más"


def _count(number: int, names: tuple[str, str]) -> str:
    singular, plural = names
    return f"{number} {singular if number == 1 else plural}"
