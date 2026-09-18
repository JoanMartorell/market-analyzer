"""Paso 0b: universo point-in-time.

Lee el fichero de constituyentes de la región y se queda con los valores
vigentes el día ``as_of``. Si el gate de calendario dejó ``open_markets``,
descarta los valores cuya bolsa (cuando se conoce) no tuvo sesión.

El fichero se construye solo si no existe, así ``data/`` no necesita
existir para arrancar. Los universos de instantánea (sin histórico propio)
se reconstruyen una vez al día: es lo que va acumulando su historial.

Deja en ``ctx.data["universe"]`` un DataFrame con una fila por ticker.
El filtro por volumen medio (``universe.min_avg_volume_20``) no va aquí:
necesita precios y lo aplica el screener.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import structlog

from analyzer.engine import StepContext, StepError, StepOutcome
from analyzer.universe import build_universe, get_universe, load_constituents, members_as_of
from analyzer.universe.constituents import meta_path

log = structlog.get_logger(__name__)


class Universe:
    name = "universe"

    def run(self, ctx: StepContext) -> StepOutcome:
        path = ctx.cfg.constituents_path(ctx.region)
        reason = build_reason(path, ctx.region.universe.id)
        if reason:
            self._build(ctx, path, reason)

        try:
            table = load_constituents(path)
        except FileNotFoundError as exc:
            raise StepError(str(exc)) from exc

        members = members_as_of(table, ctx.as_of)
        open_markets = ctx.data.get("open_markets")
        if open_markets:
            mic_known = members["mic"].notna()
            members = members.loc[~mic_known | members["mic"].isin(open_markets)]

        if members.empty:
            raise StepError(f"universo {ctx.region.universe.id!r} vacío a {ctx.as_of}")

        ctx.data["universe"] = members.reset_index(drop=True)
        built = " (construido ahora)" if reason else ""
        return StepOutcome(
            message=f"{len(members)} valores en {ctx.region.universe.id} a {ctx.as_of}{built}"
        )

    @staticmethod
    def _build(ctx: StepContext, path: Path, reason: str) -> None:
        log.info("universe.build", region=ctx.region.id, reason=reason, path=str(path))
        try:
            build_universe(ctx.region, path, user_agent=ctx.cfg.env.get("SEC_EDGAR_USER_AGENT"))
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise StepError(
                f"no se pudo construir el universo {ctx.region.universe.id!r}: {exc}"
            ) from exc


def build_reason(path: Path, universe_id: str) -> str | None:
    """Por qué hay que (re)construir el fichero, o ``None`` si sirve tal cual."""
    if not path.is_file():
        return "no existe"
    if get_universe(universe_id).history:
        return None  # la fuente ya trae el histórico; se reconstruye a mano
    built_at = _built_at(path)
    if built_at is None:
        return "sin metadatos de construcción"
    if built_at.date() < datetime.now(UTC).date():
        return f"instantánea del {built_at.date()}"
    return None


def _built_at(path: Path) -> datetime | None:
    meta = meta_path(path)
    if not meta.is_file():
        return None
    try:
        raw = json.loads(meta.read_text(encoding="utf-8"))["built_at"]
        return datetime.fromisoformat(str(raw))
    except (ValueError, KeyError, TypeError):
        return None
