"""Forma de la respuesta del modelo.

El esquema se le pasa a la API como *structured output*: la respuesta llega
ya como JSON válido contra estos modelos, sin pedirle por prompt que "responda
solo JSON" y sin recortar texto alrededor. Aun así se valida al recibirla:
que el JSON encaje en el esquema no garantiza que hable de los candidatos
que se enviaron, y eso se comprueba aparte (ver ``service.check_answer``).

Los nombres de campo van en inglés como el resto del código; el contenido que
escribe el modelo va en español, que es lo que lee el usuario.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Name = Literal["confirmar", "vigilar", "descartar"]
CONFIRM, WATCH, DISCARD = get_args(Name)
VERDICTS: tuple[str, ...] = get_args(Name)  # el orden es el de la lectura: mejor a peor

# ``output_config.format`` rechaza los límites numéricos: "For 'number' type,
# properties maximum, minimum are not supported". Se quitan del esquema que va
# a la API; el rango se le dice al modelo en la descripción del campo y lo
# sigue comprobando pydantic al validar la respuesta.
UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Verdict(Answer):
    """Lectura del modelo sobre un candidato del screener."""

    ticker: str
    verdict: Name
    confidence: float = Field(ge=0.0, le=1.0, description="De 0 a 1.")
    headline: str = Field(
        description="El motivo en una sola línea, doce palabras como mucho: es lo que se lee "
        "en el móvil."
    )
    rationale: str
    risks: list[str]


class Analysis(Answer):
    """Respuesta completa de un ciclo: la lectura del día y un veredicto por candidato."""

    summary: str
    verdicts: list[Verdict]

    @property
    def tickers(self) -> list[str]:
        return [v.ticker for v in self.verdicts]

    def by_verdict(self, name: str) -> list[Verdict]:
        return [v for v in self.verdicts if v.verdict == name]


def json_schema() -> dict[str, Any]:
    """Esquema JSON de ``Analysis`` para ``output_config.format``."""
    return _prune(Analysis.model_json_schema())


def _prune(node: dict[str, Any]) -> dict[str, Any]:
    return {key: _value(value) for key, value in node.items() if key not in UNSUPPORTED}


def _value(value: Any) -> Any:
    if isinstance(value, dict):
        return _prune(value)
    if isinstance(value, list):
        return [_value(item) for item in value]
    return value
