"""Contratos del paquete delivery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

Severity = Literal["info", "warning", "error"]
Credentials = Mapping[str, str | None]


class DeliveryError(Exception):
    """Canal desconocido, credenciales ausentes o fallo al enviar."""


@dataclass(frozen=True)
class Message:
    subject: str
    body: str
    severity: Severity = "info"


class DeliveryProvider(Protocol):
    @property
    def name(self) -> str: ...

    def send(self, message: Message) -> None: ...


class RegisteredProvider(Protocol):
    """Lo que necesita el registro para construir un proveedor a partir del entorno."""

    name: ClassVar[str]
    REQUIRED_ENV: ClassVar[tuple[str, ...]]

    @classmethod
    def from_credentials(cls, credentials: Credentials) -> DeliveryProvider: ...
