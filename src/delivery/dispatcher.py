"""Despachador: envía un mensaje por todos los proveedores configurados.

Un proveedor que falla no impide que los demás envíen. El fallo se registra
y se devuelve en el informe; el llamante decide qué hacer con él.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from delivery.base import DeliveryProvider, Message

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DeliveryReport:
    provider: str
    ok: bool
    error: str = ""


class Dispatcher:
    def __init__(self, providers: Sequence[DeliveryProvider]) -> None:
        self._providers = list(providers)

    @property
    def providers(self) -> list[str]:
        return [p.name for p in self._providers]

    def send(self, message: Message) -> list[DeliveryReport]:
        reports: list[DeliveryReport] = []
        for provider in self._providers:
            try:
                provider.send(message)
            except Exception as exc:
                log.error("delivery.failed", provider=provider.name, error=str(exc))
                reports.append(DeliveryReport(provider.name, ok=False, error=str(exc)))
            else:
                log.info("delivery.sent", provider=provider.name, subject=message.subject)
                reports.append(DeliveryReport(provider.name, ok=True))
        if not self._providers:
            log.warning("delivery.no_providers", subject=message.subject)
        return reports
