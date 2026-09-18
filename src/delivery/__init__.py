"""delivery: entrega de mensajes por uno o varios proveedores.

Este paquete no conoce la configuración del proyecto. Recibe una lista de
canales y un mapping de credenciales, y devuelve un ``Dispatcher``. A partir
de ahí el resto del sistema solo usa ``Dispatcher.send(Message)``.

Añadir un proveedor: crear su módulo o subcarpeta, implementar
``DeliveryProvider`` con ``REQUIRED_ENV`` y ``from_credentials``, y
registrarlo en ``REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Iterable

from delivery.base import (
    Credentials,
    DeliveryError,
    DeliveryProvider,
    Message,
    RegisteredProvider,
    Severity,
)
from delivery.console import ConsoleProvider
from delivery.dispatcher import DeliveryReport, Dispatcher
from delivery.telegram import TelegramProvider

REGISTRY: dict[str, type[RegisteredProvider]] = {
    ConsoleProvider.name: ConsoleProvider,
    TelegramProvider.name: TelegramProvider,
}


def available_providers() -> list[str]:
    return sorted(REGISTRY)


def required_env(channels: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Variables de entorno que exige cada canal. Lanza ``DeliveryError`` si alguno no existe."""
    return {name: _lookup(name).REQUIRED_ENV for name in channels}


def build_dispatcher(channels: Iterable[str], credentials: Credentials) -> Dispatcher:
    providers: list[DeliveryProvider] = []
    for name in channels:
        spec = _lookup(name)
        missing = [env for env in spec.REQUIRED_ENV if not credentials.get(env)]
        if missing:
            raise DeliveryError(f"canal {name!r}: faltan credenciales {', '.join(missing)}")
        providers.append(spec.from_credentials(credentials))
    return Dispatcher(providers)


def _lookup(name: str) -> type[RegisteredProvider]:
    spec = REGISTRY.get(name)
    if spec is None:
        options = ", ".join(available_providers())
        raise DeliveryError(f"canal de entrega desconocido {name!r}; disponibles: {options}")
    return spec


__all__ = [
    "REGISTRY",
    "ConsoleProvider",
    "Credentials",
    "DeliveryError",
    "DeliveryProvider",
    "DeliveryReport",
    "Dispatcher",
    "Message",
    "Severity",
    "TelegramProvider",
    "available_providers",
    "build_dispatcher",
    "required_env",
]
