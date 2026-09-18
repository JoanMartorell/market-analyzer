"""Proveedor Telegram: envía mensajes a un chat vía Bot API y descubre el id del chat."""

from delivery.telegram.chats import TelegramChat, telegram_chats
from delivery.telegram.provider import TelegramProvider

__all__ = ["TelegramChat", "TelegramProvider", "telegram_chats"]
