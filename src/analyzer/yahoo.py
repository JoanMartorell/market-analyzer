"""Símbolos de Yahoo Finance: cómo se escribe un (ticker, mic) para Yahoo.

Lo comparten el proveedor de precios yfinance y el enriquecedor de universos
que corrige tickers. Vive aparte de ``analyzer.steps`` para que el universo
pueda usarlo sin arrastrar todos los pasos del pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping

from analyzer.storage.prices import UNKNOWN_MIC

# Sufijo de Yahoo por MIC. Cadena vacía = sin sufijo (mercado USA).
YF_SUFFIX: Mapping[str, str] = {
    "XNYS": "",
    "XNAS": "",
    "BATS": "",
    UNKNOWN_MIC: "",  # sin bolsa conocida se prueba como símbolo USA
    "XLON": ".L",
    "XETR": ".DE",
    "XPAR": ".PA",
    "XAMS": ".AS",
    "XMAD": ".MC",
    "XMIL": ".MI",
    "XSWX": ".SW",
    "XSTO": ".ST",
    "XOSL": ".OL",
    "XCSE": ".CO",
    "XHEL": ".HE",
    "XBRU": ".BR",
    "XWBO": ".VI",
    "XDUB": ".IR",
    "XLIS": ".LS",
    "XWAR": ".WA",
    "XTKS": ".T",
    "XHKG": ".HK",
    "XASX": ".AX",
    "XKRX": ".KS",
}


def yf_symbol(ticker: str, mic: str) -> str | None:
    """``BRK.B``/XNYS -> ``BRK-B``; ``ATCO A``/XSTO -> ``ATCO-A.ST``; bolsa no cubierta -> None."""
    suffix = YF_SUFFIX.get(mic)
    if suffix is None:
        return None
    base = ticker.strip().upper().replace(" ", "-").replace(".", "-")
    return f"{base}{suffix}" if base else None
