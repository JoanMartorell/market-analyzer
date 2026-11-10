"""Telegram Bot API: https://core.telegram.org/bots/api#sendmessage

Sin librería intermedia: una llamada HTTP con httpx es todo lo que hace falta.
El cliente es inyectable para poder probar sin red.

Un mensaje con imagen va por ``sendPhoto`` con el texto como pie. El pie
tiene un límite más corto que un mensaje: si el texto no cabe, la foto sale
con el asunto y el texto completo va detrás como mensaje normal.
"""

from __future__ import annotations

from typing import ClassVar

import httpx

from delivery.base import Credentials, DeliveryError, Message

TELEGRAM_MAX_LEN = 4096  # límite de la API por mensaje
TELEGRAM_CAPTION_MAX_LEN = 1024  # límite del pie de una foto
_ICONS = {"info": "📊", "warning": "⚠️", "error": "❌"}


class TelegramProvider:
    name: ClassVar[str] = "telegram"
    REQUIRED_ENV: ClassVar[tuple[str, ...]] = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base = f"https://api.telegram.org/bot{token}"
        self._chat_id = chat_id
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    @classmethod
    def from_credentials(cls, credentials: Credentials) -> TelegramProvider:
        token = credentials.get("TELEGRAM_BOT_TOKEN")
        chat_id = credentials.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise DeliveryError("telegram: faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return cls(token=token, chat_id=chat_id)

    def send(self, message: Message) -> None:
        heading = f"{_ICONS[message.severity]} {message.subject}"
        text = f"{heading}\n\n{message.body}"
        if message.image is not None:
            if len(text) <= TELEGRAM_CAPTION_MAX_LEN:
                self._post_photo(message.image, text)
                return
            self._post_photo(message.image, heading[:TELEGRAM_CAPTION_MAX_LEN])
        for chunk in split_text(text, TELEGRAM_MAX_LEN):
            self._post_text(chunk)

    def _post_text(self, text: str) -> None:
        self._call(
            "sendMessage",
            json={"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True},
        )

    def _post_photo(self, image: bytes, caption: str) -> None:
        self._call(
            "sendPhoto",
            data={"chat_id": self._chat_id, "caption": caption},
            files={"photo": ("grafico.png", image, "image/png")},
        )

    def _call(
        self,
        method: str,
        *,
        json: dict[str, object] | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> None:
        try:
            # El error de httpx lleva la URL, y la URL lleva el token: se informa
            # del método y del estado, nunca del texto de la excepción.
            response = self._client.post(
                f"{self._base}/{method}", json=json, data=data, files=files
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeliveryError(f"telegram: {method} devolvió {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise DeliveryError(f"telegram: {method} falló ({type(exc).__name__})") from None
        payload = response.json()
        if not payload.get("ok", False):
            raise DeliveryError(f"telegram: {payload.get('description', 'respuesta no ok')}")


def split_text(text: str, limit: int) -> list[str]:
    """Trocea por el último salto de línea antes del límite; si no hay, corta seco."""
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks
