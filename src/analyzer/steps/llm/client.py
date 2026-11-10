"""Llamada al modelo: directa o por lotes.

La Batch API cuesta la mitad y encaja con un análisis nocturno, pero no
promete cuándo responde: la mayoría de lotes terminan en menos de una hora y
el máximo son veinticuatro. El ciclo diario no puede esperar tanto, así que
se espera hasta ``batch_wait_minutes``, y si el lote no ha terminado se
cancela y se repite la llamada en directo. Se paga el precio completo ese
día, que es mucho mejor que quedarse sin análisis.

La respuesta llega como *structured output*: se manda el esquema JSON y la
API garantiza que el texto encaja. Lo que no garantiza es que hable de los
candidatos correctos; de eso se ocupa el servicio.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import anthropic
import structlog
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchResult
from anthropic.types.messages.batch_create_params import Request

from analyzer.steps.llm.schema import json_schema
from core.config import LLMSettings

log = structlog.get_logger(__name__)

CUSTOM_ID = "analisis"  # un lote de una sola petición: el id no necesita ser único
POLL_SECONDS = 10.0
ENDED = "ended"
CLIENT_TIMEOUT = 300.0  # una llamada con razonamiento largo, no un ciclo entero


class LlmError(Exception):
    """La llamada no dio una respuesta utilizable (error de API, corte, rechazo)."""


@dataclass(frozen=True)
class TokenUsage:
    """Tokens de una llamada, separados por lo que cuesta cada uno."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class Reply:
    text: str  # JSON conforme al esquema
    usage: TokenUsage
    batch: bool  # si se cobró con el descuento del lote


class LlmClient(Protocol):
    def ask(self, system: str, payload: str) -> Reply: ...


class AnthropicClient:
    """Cliente sobre el SDK de Anthropic con los parámetros de ``settings.llm``."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: LLMSettings,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._settings = settings
        self._sleep = sleep

    def ask(self, system: str, payload: str) -> Reply:
        params = self._params(system, payload)
        if not self._settings.use_batch_api:
            return self._ask_now(params)
        try:
            return self._ask_batch(params)
        except TimeoutError as exc:
            log.warning("llm.batch_timeout", detail=str(exc))
            return self._ask_now(params)

    # --- peticiones ---------------------------------------------------------------------

    def _params(self, system: str, payload: str) -> dict[str, Any]:
        """Los mismos parámetros para las dos vías: lo único que cambia es el transporte."""
        block: dict[str, Any] = {"type": "text", "text": system}
        if self._settings.prompt_caching:
            # El prompt del analista no cambia entre ciclos; lo que se paga de
            # nuevo cada día es el payload. La caché solo ahorra si otra región
            # llama dentro de la ventana de unos minutos.
            block["cache_control"] = {"type": "ephemeral"}
        return {
            "model": self._settings.model,
            "max_tokens": self._settings.max_tokens,
            "system": [block],
            "messages": [{"role": "user", "content": payload}],
            "output_config": {
                "effort": self._settings.effort,
                "format": {"type": "json_schema", "schema": json_schema()},
            },
        }

    def _ask_now(self, params: dict[str, Any]) -> Reply:
        try:
            message = self._client.with_options(timeout=CLIENT_TIMEOUT).messages.create(**params)
        except anthropic.APIError as exc:
            raise LlmError(f"la API respondió con error: {exc}") from exc
        return _reply(message, batch=False)

    def _ask_batch(self, params: dict[str, Any]) -> Reply:
        request = Request(
            custom_id=CUSTOM_ID,
            params=cast(MessageCreateParamsNonStreaming, params),
        )
        try:
            batch = self._client.messages.batches.create(requests=[request])
            log.info("llm.batch_created", batch=batch.id)
            self._wait(batch.id)
            return self._collect(batch.id)
        except anthropic.APIError as exc:
            raise LlmError(f"la API respondió con error: {exc}") from exc

    def _wait(self, batch_id: str) -> None:
        """Espera a que el lote termine.

        La espera se cuenta en consultas, no en reloj: ``batch_wait_minutes``
        es un tope aproximado, que es todo lo que hace falta para decidir si
        se sigue esperando o se llama en directo.
        """
        polls = max(1, round(self._settings.batch_wait_minutes * 60 / POLL_SECONDS))
        status = "unknown"
        for _ in range(polls):
            status = self._client.messages.batches.retrieve(batch_id).processing_status
            if status == ENDED:
                return
            self._sleep(POLL_SECONDS)
        if self._cancel(batch_id):
            return
        raise TimeoutError(
            f"el lote {batch_id} sigue en {status} tras "
            f"{self._settings.batch_wait_minutes} min: se cancela y se llama en directo"
        )

    def _cancel(self, batch_id: str) -> bool:
        """Cancela el lote; devuelve ``True`` si resulta que ya había terminado.

        Entre la última consulta y la cancelación el lote puede terminar, y la
        API rechaza cancelar un lote terminado con un 400. En ese caso se
        vuelve a consultar: si está terminado, sus resultados valen y no hace
        falta la llamada directa. Cualquier otro rechazo sigue siendo un error.
        """
        try:
            self._client.messages.batches.cancel(batch_id)
        except anthropic.BadRequestError:
            status = self._client.messages.batches.retrieve(batch_id).processing_status
            if status != ENDED:
                raise
            log.info("llm.batch_ended_on_cancel", batch=batch_id)
            return True
        return False

    def _collect(self, batch_id: str) -> Reply:
        for result in self._client.messages.batches.results(batch_id):
            if result.custom_id != CUSTOM_ID:
                continue
            if result.result.type != "succeeded":
                raise LlmError(f"el lote falló: {_batch_error(result.result)}")
            return _reply(result.result.message, batch=True)
        raise LlmError(f"el lote {batch_id} no devolvió la petición {CUSTOM_ID!r}")


def build_client(settings: LLMSettings, api_key: str) -> AnthropicClient:
    return AnthropicClient(anthropic.Anthropic(api_key=api_key), settings)


def _batch_error(result: MessageBatchResult) -> str:
    """El motivo que da la API, que es lo único que sirve para arreglarlo."""
    if result.type == "errored":
        return f"{result.error.error.type}: {result.error.error.message}"
    return result.type  # canceled o expired: no hay más detalle


def _reply(message: anthropic.types.Message, *, batch: bool) -> Reply:
    if message.stop_reason == "max_tokens":
        raise LlmError(
            f"la respuesta se cortó en max_tokens ({message.usage.output_tokens} tokens): "
            "sube llm.max_tokens o baja llm.effort"
        )
    if message.stop_reason == "refusal":
        raise LlmError("el modelo rechazó la petición")
    text = next((b.text for b in message.content if b.type == "text"), "")
    if not text:
        raise LlmError("la respuesta no trae texto")
    return Reply(text=text, usage=_usage(message.usage), batch=batch)


def _usage(usage: anthropic.types.Usage) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )
