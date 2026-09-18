"""Utilidades mínimas de SQL."""

from __future__ import annotations

import re

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_identifier(name: str) -> str:
    """Nombre de esquema o tabla seguro para interpolar en SQL (viene de config, no del usuario)."""
    if not _IDENT.match(name):
        raise ValueError(f"identificador SQL inválido: {name!r}")
    return name
