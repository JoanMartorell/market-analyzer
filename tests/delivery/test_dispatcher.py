"""Despachador y registro de proveedores."""

import io

import pytest

from delivery import (
    ConsoleProvider,
    DeliveryError,
    Dispatcher,
    Message,
    available_providers,
    build_dispatcher,
    required_env,
)


class _Recorder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


class _Broken:
    name = "broken"

    def send(self, message: Message) -> None:
        raise ConnectionError("sin red")


def test_sends_to_every_provider() -> None:
    a, b = _Recorder("a"), _Recorder("b")
    reports = Dispatcher([a, b]).send(Message("asunto", "cuerpo"))

    assert [r.ok for r in reports] == [True, True]
    assert a.sent == b.sent == [Message("asunto", "cuerpo")]


def test_one_failure_does_not_block_the_rest() -> None:
    ok = _Recorder("ok")
    reports = Dispatcher([_Broken(), ok]).send(Message("asunto", "cuerpo"))

    assert [(r.provider, r.ok) for r in reports] == [("broken", False), ("ok", True)]
    assert "sin red" in reports[0].error
    assert len(ok.sent) == 1


def test_registry_lists_console_and_telegram() -> None:
    assert available_providers() == ["console", "telegram"]
    assert required_env(["console"]) == {"console": ()}
    assert required_env(["telegram"]) == {"telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}


def test_unknown_channel_is_rejected() -> None:
    with pytest.raises(DeliveryError, match="desconocido 'fax'"):
        build_dispatcher(["fax"], {})


def test_missing_credentials_are_rejected_at_build_time() -> None:
    with pytest.raises(DeliveryError, match="TELEGRAM_CHAT_ID"):
        build_dispatcher(["telegram"], {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": ""})


def test_build_console_dispatcher() -> None:
    dispatcher = build_dispatcher(["console"], {})
    assert dispatcher.providers == ["console"]


def test_console_provider_writes_to_stream() -> None:
    stream = io.StringIO()
    ConsoleProvider(stream).send(Message("Asunto", "línea 1\nlínea 2", severity="error"))

    assert stream.getvalue() == "[x] Asunto\nlínea 1\nlínea 2\n"
