"""Proveedor de consola: imprime el mensaje en stdout. Para desarrollo."""

from __future__ import annotations

import sys
from typing import ClassVar, TextIO

from delivery.base import Credentials, Message

_ICONS = {"info": "i", "warning": "!", "error": "x"}


class ConsoleProvider:
    name: ClassVar[str] = "console"
    REQUIRED_ENV: ClassVar[tuple[str, ...]] = ()

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    @classmethod
    def from_credentials(cls, credentials: Credentials) -> ConsoleProvider:
        return cls()

    def send(self, message: Message) -> None:
        icon = _ICONS[message.severity]
        text = f"[{icon}] {message.subject}\n{message.body}\n"
        if message.image is not None:
            # La consola no pinta imágenes: deja constancia de que el mensaje la lleva.
            text += f"(imagen adjunta, {len(message.image) / 1024:.0f} kB)\n"
        self._stream.write(text)
        self._stream.flush()
