"""Entrada del modelo: candidatos del screener con sus temas de noticias.

Todo lo que entra aquí se paga en tokens, así que se manda lo justo: el
valor, por qué lo marcó la regla, y los temas que el paso de deduplicación
dejó para ese valor, recortados. Nada de urls, ids ni artículos sueltos.

La construcción es determinista —mismo día, mismos datos, mismo JSON byte a
byte— porque de ese JSON sale el hash que decide si hay que volver a pagar
una llamada o vale la que ya está guardada.

El texto de las noticias viaja como dato dentro del JSON; el prompt del
sistema le dice al modelo que no obedezca instrucciones que vengan ahí.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable
from datetime import date
from typing import Any

import pandas as pd

Row = dict[Hashable, Any]  # una fila tal como la devuelve ``DataFrame.to_dict("records")``

MAX_TOPICS = 5  # temas por valor: los más recientes; el resto es ruido repetido
MAX_GENERAL_TOPICS = 5
SUMMARY_CHARS = 400  # un resumen más largo no aporta y multiplica el coste
CONDITION_SEPARATOR = "; "  # como las une el screener en ``conditions_met``
PRICE_DECIMALS = 4
SCORE_DECIMALS = 3
SENTIMENT_DECIMALS = 2


def build_payload(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    region_id: str,
    as_of: date,
    currency: str,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """JSON que se manda al modelo: los candidatos del día con su contexto."""
    names = _attributes(panel)
    topics = _topics_by_ticker(clusters)
    return {
        "region": region_id,
        "fecha": as_of.isoformat(),
        "moneda": currency,
        "candidatos": [_candidate(row, names, topics) for row in candidates.to_dict("records")],
        "temas_generales": _general_topics(clusters),
    }


def payload_hash(payload: dict[str, Any]) -> str:
    """Huella de la entrada: si no cambia, la respuesta guardada sigue valiendo."""
    return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()


def dumps(payload: dict[str, Any]) -> str:
    """Serialización estable: claves ordenadas y sin espacios de relleno."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --- piezas -----------------------------------------------------------------------------


def _candidate(
    row: Row,
    names: dict[tuple[str, str], dict[str, Any]],
    topics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    ticker = str(row["ticker"])
    attributes = names.get((ticker, str(row["mic"])), {})
    return {
        "ticker": ticker,
        "nombre": attributes.get("name") or ticker,
        "sector": _text(row.get("sector")),
        "cierre": _number(row.get("close"), PRICE_DECIMALS),
        "divisa": attributes.get("currency"),
        "regla": str(row["rule_id"]),
        "direccion": str(row["direction"]),
        "score": _number(row.get("score"), SCORE_DECIMALS),
        "condiciones": _conditions(row.get("conditions_met")),
        "temas": topics.get(ticker, []),
    }


def _conditions(raw: object) -> list[str]:
    text = _text(raw)
    return [part for part in text.split(CONDITION_SEPARATOR) if part] if text else []


def _topics_by_ticker(clusters: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Temas de cada valor, del más reciente al más antiguo y como mucho ``MAX_TOPICS``."""
    if clusters.empty:
        return {}
    with_ticker = clusters.loc[clusters["ticker"].notna()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _by_recency(with_ticker):
        bucket = grouped.setdefault(str(row["ticker"]), [])
        if len(bucket) < MAX_TOPICS:
            bucket.append(_topic(row))
    return grouped


def _general_topics(clusters: pd.DataFrame) -> list[dict[str, Any]]:
    if clusters.empty:
        return []
    globals_only = clusters.loc[clusters["ticker"].isna()]
    return [_topic(row) for row in _by_recency(globals_only)[:MAX_GENERAL_TOPICS]]


def _by_recency(clusters: pd.DataFrame) -> list[Row]:
    if clusters.empty:
        return []
    ordered = clusters.sort_values(["last_published_at", "cluster_id"], ascending=[False, True])
    return list(ordered.to_dict("records"))


def _topic(row: Row) -> dict[str, Any]:
    return {
        "titular": _text(row.get("title")),
        "resumen": _clip(_text(row.get("summary")), SUMMARY_CHARS),
        "fuentes": _text(row.get("sources")),
        "sentimiento": _number(row.get("sentiment_score"), SENTIMENT_DECIMALS),
        "publicado": _moment(row.get("last_published_at")),
        "articulos": int(row.get("size") or 1),
    }


# --- normalización ----------------------------------------------------------------------


def _missing(value: Any) -> bool:
    """Ausente, NaN o NaT: pandas devuelve las tres cosas según la columna."""
    return value is None or bool(pd.isna(value))


def _text(value: Any) -> str:
    return "" if _missing(value) else str(value).strip()


def _number(value: Any, decimals: int) -> float | None:
    """``None`` en vez de NaN: el JSON no tiene NaN y ``null`` se lee como 'no hay dato'."""
    return None if _missing(value) else round(float(value), decimals)


def _moment(value: Any) -> str | None:
    return None if _missing(value) else pd.Timestamp(value).strftime("%Y-%m-%dT%H:%MZ")


def _clip(text: str, limit: int) -> str:
    """Recorta por palabra entera; el modelo no necesita el resumen completo."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut.rstrip('.,;:')}..."


def _attributes(panel: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Nombre y divisa de cada valor, del panel. Sin panel, el ticker hace de nombre."""
    if panel is None or panel.empty:
        return {}
    return {
        (str(row["ticker"]), str(row["mic"])): {
            "name": _text(row.get("name")) or None,
            "currency": _text(row.get("currency")) or None,
        }
        for row in panel.to_dict("records")
    }
