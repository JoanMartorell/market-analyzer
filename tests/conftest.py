"""Fixtures compartidas."""

from pathlib import Path

import pytest

from core.config import AppConfig, Env, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

ALL_KEYS = [
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
    "EODHD_API_KEY",
    "FMP_API_KEY",
    "SEC_EDGAR_USER_AGENT",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]


@pytest.fixture
def empty_env(monkeypatch: pytest.MonkeyPatch) -> Env:
    """Entorno sin ninguna clave ni .env, para que los tests no dependan de la máquina."""
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Env(_env_file=None)


@pytest.fixture
def cfg(empty_env: Env) -> AppConfig:
    """La configuración real del repositorio, cargada sin credenciales."""
    return load_config(CONFIG_DIR, env=empty_env)
