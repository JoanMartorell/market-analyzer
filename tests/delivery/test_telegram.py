"""Proveedor Telegram sin red, con un transporte simulado de httpx."""

import json
from collections.abc import Callable

import httpx
import pytest

from delivery import DeliveryError, Message
from delivery.telegram import TelegramProvider
from delivery.telegram.provider import TELEGRAM_MAX_LEN, split_text

Handler = Callable[[httpx.Request], httpx.Response]


def _provider(handler: Handler) -> tuple[TelegramProvider, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return TelegramProvider("TOKEN", "12345", client=client), calls


def _ok(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/botTOKEN/sendMessage"
    return httpx.Response(200, json={"ok": True, "result": {}})


def test_sends_subject_and_body_to_chat() -> None:
    provider, calls = _provider(_ok)
    provider.send(Message("Asunto", "cuerpo", severity="warning"))

    assert len(calls) == 1
    assert calls[0]["chat_id"] == "12345"
    assert str(calls[0]["text"]).endswith("Asunto\n\ncuerpo")


def test_long_message_is_split_in_chunks() -> None:
    provider, calls = _provider(_ok)
    body = "\n".join(f"línea {i}" for i in range(800))  # > 4096 caracteres
    provider.send(Message("Largo", body))

    assert len(calls) >= 2
    assert all(len(str(c["text"])) <= TELEGRAM_MAX_LEN for c in calls)


def test_api_error_becomes_delivery_error() -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "chat not found"})

    provider, _ = _provider(bad)
    with pytest.raises(DeliveryError, match="chat not found"):
        provider.send(Message("a", "b"))


def test_http_error_becomes_delivery_error() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502)

    provider, _ = _provider(down)
    with pytest.raises(DeliveryError, match="502"):
        provider.send(Message("a", "b"))


def test_from_credentials_requires_both_values() -> None:
    with pytest.raises(DeliveryError):
        TelegramProvider.from_credentials({"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": None})


def test_split_prefers_line_breaks() -> None:
    assert split_text("abc", 10) == ["abc"]
    assert split_text("aaaa\nbbbb\ncccc", 8) == ["aaaa", "bbbb", "cccc"]
    assert split_text("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]
