"""Utilidades compartidas por fuentes y enriquecedores."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

# Wikipedia rechaza (403) User-Agents sin URL o contacto identificable.
DEFAULT_USER_AGENT = "market-analyzer/0.1 (+https://github.com/market-analyzer)"


def canonical_ticker(raw: object) -> str | None:
    """Mayúsculas y punto para clases de acciones: ``brk-b`` -> ``BRK.B``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    ticker = str(raw).strip().upper().replace("-", ".")
    return ticker or None


def merge_fill(table: pd.DataFrame, other: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Rellena ``columns`` de ``table`` con ``other`` por ticker, sin pisar valores ya presentes."""
    lookup = other.drop_duplicates("ticker").set_index("ticker")
    out = table.copy()
    for column in columns:
        incoming = out["ticker"].map(lookup[column])
        if column in out.columns:
            out[column] = out[column].where(out[column].notna(), incoming)
        else:
            out[column] = incoming
    return out
