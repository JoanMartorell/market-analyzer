"""Proveedor Telegram sin red, con un transporte simulado de httpx."""

import json
from collections.abc import Callable

import httpx
import pytest

from delivery import DeliveryError, Message
from delivery.telegram import TelegramProvider, telegram_chats
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


# --- descubrir el chat --------------------------------------------------------------


def _updates(handler: Handler) -> httpx.Client:
    def wrapped(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/botTOKEN/getUpdates"
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(wrapped))


def test_chats_come_from_the_updates_most_recent_first() -> None:
    updates = [
        {
            "update_id": 1,
            "message": {
                "chat": {"id": 111, "type": "private", "first_name": "Joan", "username": "jm"},
                "text": "/start",
            },
        },
        {
            "update_id": 2,
            "my_chat_member": {"chat": {"id": -222, "type": "group", "title": "Señales"}},
        },
        {"update_id": 3, "message": {"chat": {"id": 111, "type": "private"}, "text": "hola"}},
    ]
    client = _updates(lambda _r: httpx.Response(200, json={"ok": True, "result": updates}))

    chats = telegram_chats("TOKEN", client=client)

    assert [(c.id, c.kind, c.title, c.last_text) for c in chats] == [
        (111, "private", "(sin nombre)", "hola"),
        (-222, "group", "Señales", ""),
    ]


def test_chats_without_updates_is_empty() -> None:
    client = _updates(lambda _r: httpx.Response(200, json={"ok": True, "result": []}))
    assert telegram_chats("TOKEN", client=client) == []


def test_chats_with_a_bad_token_is_a_delivery_error() -> None:
    client = _updates(
        lambda _r: httpx.Response(401, json={"ok": False, "description": "Unauthorized"})
    )
    with pytest.raises(DeliveryError, match="401"):
        telegram_chats("TOKEN", client=client)


# --- mensajes con imagen ------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"datos-de-la-imagen"


def _photo_provider() -> tuple[TelegramProvider, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TelegramProvider("TOKEN", "12345", client=client), requests


def test_a_message_with_image_goes_as_a_photo_with_the_text_as_caption() -> None:
    provider, requests = _photo_provider()
    provider.send(Message("COMPRAR AAA", "Cierre 101.25 USD", image=PNG))

    assert [r.url.path for r in requests] == ["/botTOKEN/sendPhoto"]
    body = requests[0].content
    assert requests[0].headers["content-type"].startswith("multipart/form-data")
    assert PNG in body
    assert b'name="caption"' in body
    assert b"COMPRAR AAA" in body
    assert b"Cierre 101.25 USD" in body
    assert b'name="chat_id"' in body
    assert b"12345" in body


def test_a_text_too_long_for_a_caption_follows_the_photo_as_a_message() -> None:
    provider, requests = _photo_provider()
    body = "\n".join(f"línea {i}" for i in range(300))  # más de 1024 caracteres
    provider.send(Message("COMPRAR AAA", body, image=PNG))

    assert [r.url.path for r in requests] == ["/botTOKEN/sendPhoto", "/botTOKEN/sendMessage"]
    assert "línea 299" not in requests[0].content.decode("utf-8", errors="ignore")
    assert str(json.loads(requests[1].content)["text"]).endswith("línea 299")


def test_errors_never_carry_the_token() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=request)

    for handler in (down, unreachable):
        provider, _ = _provider(handler)
        with pytest.raises(DeliveryError) as raised:
            provider.send(Message("a", "b"))
        assert "TOKEN" not in str(raised.value)
        assert "sendMessage" in str(raised.value)
