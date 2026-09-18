"""Telegram Bot API: https://core.telegram.org/bots/api#sendmessage

Sin librería intermedia: una llamada HTTP con httpx es todo lo que hace falta.
El cliente es inyectable para poder probar sin red.
"""

from __future__ import annotations

from typing import ClassVar

import httpx

from delivery.base import Credentials, DeliveryError, Message

TELEGRAM_MAX_LEN = 4096  # límite de la API por mensaje
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
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
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
        text = f"{_ICONS[message.severity]} {message.subject}\n\n{message.body}"
        for chunk in split_text(text, TELEGRAM_MAX_LEN):
            self._post(chunk)

    def _post(self, text: str) -> None:
        try:
            response = self._client.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DeliveryError(f"telegram: {exc}") from exc
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
