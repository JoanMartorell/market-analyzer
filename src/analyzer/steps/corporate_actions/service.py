"""Reconciliación de eventos corporativos.

Para el universo del día:
1. Detecta eventos en la ventana de staging (feed del proveedor y, si no
   hay feed, ratio del salto).
2. Descarta los ya registrados en ``prod.corporate_actions``.
3. Si el histórico guardado de un valor es anterior a un evento nuevo, vuelve
   a bajar su ventana completa: el proveedor ya la reescribió y la nuestra
   tiene un escalón ficticio en la frontera.
4. Levanta la cuarentena de ``quality`` a los valores cuyo salto explica un
   split del mismo día.
5. Escribe lo reingestado en staging y prod, y registra los eventos.

Todo es idempotente: al día siguiente los eventos ya están registrados y no
se vuelve a bajar nada.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd
import structlog

from analyzer.steps.corporate_actions.detect import (
    KIND_SPLIT,
    CorporateAction,
    detect_from_feed,
    detect_splits_by_ratio,
    needs_reingest,
)
from analyzer.steps.ingest.providers.base import PriceProvider
from analyzer.steps.ingest.service import lookback_start, universe_keys
from analyzer.storage import ACTION_COLUMNS, CorporateActionStore, PriceStore

log = structlog.get_logger(__name__)

Key = tuple[str, str]
QUARANTINE_REASON_JUMP = "jump"  # el motivo que un split puede explicar


@dataclass(frozen=True)
class CorporateActionsReport:
    as_of: date
    detected: tuple[CorporateAction, ...]  # eventos nuevos (no registrados antes)
    reingested: tuple[Key, ...]  # claves cuyo histórico se volvió a bajar
    released: tuple[Key, ...]  # claves que salen de cuarentena
    refreshed_rows: int  # velas reescritas en prod

    def summary(self) -> str:
        if not self.detected and not self.released:
            return "sin eventos nuevos"
        splits = sum(a.kind == KIND_SPLIT for a in self.detected)
        dividends = len(self.detected) - splits
        parts = [f"{splits} splits, {dividends} dividendos"]
        if self.reingested:
            parts.append(f"{len(self.reingested)} valores reingestados")
        if self.released:
            parts.append(f"{len(self.released)} fuera de cuarentena")
        return ", ".join(parts)


def reconcile_corporate_actions(
    universe: pd.DataFrame,
    as_of: date,
    *,
    provider: PriceProvider,
    staging: PriceStore,
    prod: PriceStore,
    actions: CorporateActionStore,
    quarantine: Mapping[Key, str],
    lookback_days: int,
) -> CorporateActionsReport:
    keys = universe_keys(universe)
    start = lookback_start(as_of, lookback_days)
    window = staging.load(start=start, end=as_of, keys=keys)

    detected = detect_from_feed(window) + detect_splits_by_ratio(window, as_of)
    known = _known_identities(actions.load(start=start, end=as_of, keys=keys))
    new = [a for a in detected if a.identity not in known]

    released = _released(quarantine, detected, as_of)
    still_quarantined = set(quarantine) - released
    stale = {a.key for a in new if needs_reingest(window, a)}

    for action in new:
        # Un split es raro y relevante; los dividendos van a cientos en la primera carga.
        emit = log.info if action.kind == KIND_SPLIT else log.debug
        emit("corporate_action.detected", detail=action.describe(), source=action.source)

    refreshed = pd.DataFrame()
    if stale:
        refreshed = _reingest(stale, start, as_of, provider=provider, staging=staging)
    to_promote = _rows_to_promote(window, refreshed, released - stale, as_of, still_quarantined)
    promoted = prod.upsert(to_promote) if not to_promote.empty else 0

    if new:
        actions.upsert(pd.DataFrame([_as_row(a) for a in new], columns=list(ACTION_COLUMNS)))

    return CorporateActionsReport(
        as_of=as_of,
        detected=tuple(new),
        reingested=tuple(sorted(stale)),
        released=tuple(sorted(released)),
        refreshed_rows=promoted,
    )


def _known_identities(recorded: pd.DataFrame) -> set[tuple[str, str, date, str]]:
    days = pd.to_datetime(recorded["date"]).dt.date
    return set(
        zip(
            recorded["ticker"].astype(str),
            recorded["mic"].astype(str),
            days,
            recorded["kind"].astype(str),
            strict=True,
        )
    )


def _released(
    quarantine: Mapping[Key, str], detected: list[CorporateAction], as_of: date
) -> set[Key]:
    """Claves en cuarentena por salto que un split del mismo día explica."""
    split_today = {a.key for a in detected if a.kind == KIND_SPLIT and a.date == as_of}
    return {
        k
        for k, reason in quarantine.items()
        if reason == QUARANTINE_REASON_JUMP and k in split_today
    }


def _reingest(
    keys: set[Key], start: date, end: date, *, provider: PriceProvider, staging: PriceStore
) -> pd.DataFrame:
    frame = pd.DataFrame(sorted(keys), columns=["ticker", "mic"])
    log.info("corporate_action.reingest", keys=len(frame), start=str(start), end=str(end))
    fresh = provider.fetch(frame, start, end)
    if fresh.empty:
        log.warning("corporate_action.reingest_empty", keys=len(frame))
        return fresh
    fresh = fresh.copy()
    fresh["source"] = provider.name
    staging.upsert(fresh)
    return fresh


def _rows_to_promote(
    window: pd.DataFrame,
    refreshed: pd.DataFrame,
    released_not_refreshed: set[Key],
    as_of: date,
    still_quarantined: set[Key],
) -> pd.DataFrame:
    """Histórico reingestado y vela del día de los liberados; nunca la de los aún en cuarentena."""
    parts: list[pd.DataFrame] = []
    if not refreshed.empty:
        parts.append(refreshed)
    if released_not_refreshed:
        is_day = window["date"] == pd.Timestamp(as_of)
        is_released = _key_mask(window, released_not_refreshed)
        parts.append(window.loc[is_day & is_released])
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    drop = (rows["date"] == pd.Timestamp(as_of)) & _key_mask(rows, still_quarantined)
    return rows.loc[~drop]


def _key_mask(frame: pd.DataFrame, keys: set[Key]) -> pd.Series:
    return pd.Series(
        [k in keys for k in zip(frame["ticker"], frame["mic"], strict=True)], index=frame.index
    )


def _as_row(action: CorporateAction) -> dict[str, object]:
    return {
        "ticker": action.ticker,
        "mic": action.mic,
        "date": action.date,
        "kind": action.kind,
        "value": action.value,
        "source": action.source,
    }
