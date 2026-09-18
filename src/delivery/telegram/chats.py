"""Descubrir el chat: https://core.telegram.org/bots/api#getupdates

Un bot no puede escribir a nadie que no le haya escrito antes, y el id del
chat no aparece en la app. La forma de obtenerlo es pedir al bot los
mensajes que ha recibido y leer de ahí el ``chat``. Telegram guarda esos
mensajes 24 horas y solo mientras no haya un webhook configurado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from delivery.base import DeliveryError

# Tipos de actualización que traen un ``chat`` del que el bot puede aprender el id.
_WITH_CHAT = ("message", "edited_message", "channel_post", "my_chat_member")


@dataclass(frozen=True)
class TelegramChat:
    id: int
    kind: str  # private, group, supergroup, channel
    title: str  # nombre de la persona o del grupo
    last_text: str  # último mensaje recibido, para reconocer el chat


def telegram_chats(
    token: str, *, client: httpx.Client | None = None, timeout: float = 10.0
) -> list[TelegramChat]:
    """Chats que han escrito al bot, del más reciente al más antiguo."""
    http = client if client is not None else httpx.Client(timeout=timeout)
    try:
        response = http.get(f"https://api.telegram.org/bot{token}/getUpdates")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DeliveryError(f"telegram: {exc}") from exc
    payload = response.json()
    if not payload.get("ok", False):
        raise DeliveryError(f"telegram: {payload.get('description', 'respuesta no ok')}")

    found: dict[int, TelegramChat] = {}
    for update in payload.get("result", []):
        for kind in _WITH_CHAT:
            chat = _chat(update.get(kind))
            if chat is not None:
                # Se reinserta para que el orden del dict sea el del último mensaje.
                found.pop(chat.id, None)
                found[chat.id] = chat
    return list(reversed(found.values()))


def _chat(event: Any) -> TelegramChat | None:
    if not isinstance(event, dict) or not isinstance(event.get("chat"), dict):
        return None
    chat = event["chat"]
    if not isinstance(chat.get("id"), int):
        return None
    name = " ".join(
        part for part in (chat.get("first_name"), chat.get("last_name")) if isinstance(part, str)
    )
    title = chat.get("title") or name or chat.get("username") or "(sin nombre)"
    if isinstance(chat.get("username"), str):
        title = f"{title} (@{chat['username']})"
    text = event.get("text")
    return TelegramChat(
        id=chat["id"],
        kind=str(chat.get("type", "?")),
        title=str(title),
        last_text=text if isinstance(text, str) else "",
    )
