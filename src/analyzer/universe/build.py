"""Construcción genérica del fichero de constituyentes de una región.

No sabe qué es el S&P 500. Busca la spec del universo en el registro,
pide los intervalos a la fuente, aplica los enriquecedores en orden,
completa las columnas que falten y escribe el parquet con su sidecar.

Universos sin histórico (``history=False``): la fuente devuelve solo la
composición actual y ``roll_snapshot`` la compara con el fichero anterior.
Quien desaparece se cierra hoy; quien aparece se abre hoy. Con el tiempo el
fichero se convierte en point-in-time desde la fecha del primer build.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import structlog

from analyzer.universe.base import SOURCE_COLUMNS, FetchContext, Notes, UniverseSpec
from analyzer.universe.constituents import COLUMNS, load_constituents, write_constituents
from analyzer.universe.registry import get_universe
from core.config import Region

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BuildReport:
    region_id: str
    universe_id: str
    path: Path
    built_at: datetime
    source: str
    enrichers: tuple[str, ...]
    history: bool
    rows: int
    current_members: int
    notes: dict[str, Notes]


def build_universe(
    region: Region,
    out_path: Path,
    *,
    user_agent: str | None,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> BuildReport:
    spec = get_universe(region.universe.id)
    ctx = FetchContext(
        today=today or datetime.now(UTC).date(),
        user_agent=user_agent,
        universe_dir=out_path.parent,
    )

    context = (
        nullcontext(client)
        if client is not None
        else httpx.Client(timeout=60.0, follow_redirects=True)
    )
    notes: dict[str, Notes] = {}
    with context as http:
        log.info("universe.source", universe=spec.id, source=spec.source.name)
        table = spec.source.fetch(http, ctx)
        _require_columns(table, SOURCE_COLUMNS, spec.source.name)
        if "source" not in table.columns:
            table["source"] = spec.source.name

        for enricher in spec.enrichers:
            log.info("universe.enrich", universe=spec.id, enricher=enricher.name)
            table, enricher_notes = enricher.enrich(table, http, ctx)
            notes[enricher.name] = enricher_notes

    if not spec.history:
        previous = load_constituents(out_path) if out_path.is_file() else None
        table, notes["snapshot"] = roll_snapshot(previous, table, ctx.today)

    table = finalize(table, spec)
    report = BuildReport(
        region_id=region.id,
        universe_id=spec.id,
        path=out_path,
        built_at=datetime.now(UTC),
        source=spec.source.name,
        enrichers=tuple(e.name for e in spec.enrichers),
        history=spec.history,
        rows=len(table),
        current_members=int(table["end"].isna().sum()),
        notes=notes,
    )
    write_constituents(table, out_path, meta=asdict(report))
    log.info(
        "universe.built",
        universe=spec.id,
        path=str(out_path),
        rows=report.rows,
        current=report.current_members,
    )
    return report


def roll_snapshot(
    previous: pd.DataFrame | None, fresh: pd.DataFrame, today: date
) -> tuple[pd.DataFrame, Notes]:
    """Compara la composición actual con el fichero anterior y ajusta intervalos.

    Primer build: todo abierto con ``start`` desconocido (NaT). Builds
    siguientes: los abiertos que ya no aparecen se cierran con ``end=today``;
    los nuevos se abren con ``start=today``. Repetir el build el mismo día no
    cambia nada.
    """
    fresh = fresh.copy()
    fresh["start"] = pd.NaT
    fresh["end"] = pd.NaT
    if previous is None or previous.empty:
        return fresh, {"mode": "first_snapshot", "members": len(fresh), "new": 0, "gone": 0}

    moment = pd.Timestamp(today)
    prev = previous.copy()
    is_open = prev["end"].isna()
    open_keys = set(_keys(prev.loc[is_open]))
    fresh_keys = set(_keys(fresh))

    gone = is_open & ~_keys(prev).isin(fresh_keys)
    prev.loc[gone, "end"] = moment

    new_rows = fresh.loc[~_keys(fresh).isin(open_keys)].copy()
    new_rows["start"] = moment

    result = pd.concat([prev, new_rows], ignore_index=True)
    notes: Notes = {
        "mode": "rolled",
        "members": int(result["end"].isna().sum()),
        "new": sorted(new_rows["ticker"]),
        "gone": sorted(prev.loc[gone, "ticker"]),
    }
    return result, notes


def _keys(table: pd.DataFrame) -> pd.Series:
    mic = (
        table["mic"] if "mic" in table.columns else pd.Series([""] * len(table), index=table.index)
    )
    return table["ticker"].astype(str) + "|" + mic.fillna("").astype(str)


def finalize(table: pd.DataFrame, spec: UniverseSpec) -> pd.DataFrame:
    """Garantiza todas las columnas del formato, tipos y orden."""
    out = table.copy()
    for column in ("name", "sector", "sub_industry", "mic"):
        if column not in out.columns:
            out[column] = pd.Series([None] * len(out), dtype="object")
    if "cik" not in out.columns:
        out["cik"] = pd.array([None] * len(out), dtype="Int64")
    out["cik"] = out["cik"].astype("Int64")
    if "currency" not in out.columns:
        out["currency"] = spec.currency
    out["currency"] = out["currency"].fillna(spec.currency)
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])
    return out[list(COLUMNS)].sort_values(["ticker", "start"]).reset_index(drop=True)


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], who: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"{who}: faltan columnas {missing}")
