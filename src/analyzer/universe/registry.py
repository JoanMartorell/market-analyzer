"""Universos disponibles, por id (el ``universe.id`` del YAML de región)."""

from __future__ import annotations

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.apac import APAC
from analyzer.universe.sources.europe import EUROPE
from analyzer.universe.sources.latam import LATAM
from analyzer.universe.sources.sp500 import SP500

UNIVERSES: dict[str, UniverseSpec] = {
    SP500.id: SP500,
    EUROPE.id: EUROPE,
    APAC.id: APAC,
    LATAM.id: LATAM,
}


def get_universe(universe_id: str) -> UniverseSpec:
    spec = UNIVERSES.get(universe_id)
    if spec is None:
        available = ", ".join(sorted(UNIVERSES))
        raise ValueError(f"universo {universe_id!r} no soportado; disponibles: {available}")
    return spec
