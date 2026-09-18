"""Variables de entorno y .env.

Aquí solo viven secretos y rutas propias de la máquina. Cualquier parámetro
de comportamiento (umbrales, horarios, modelos) va en config/*.yaml para que
quede versionado.
"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- claves de API -----------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    fred_api_key: SecretStr | None = None
    eodhd_api_key: SecretStr | None = None
    fmp_api_key: SecretStr | None = None
    sec_edgar_user_agent: str | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # --- entorno de ejecución ---------------------------------------------
    ma_env: Literal["dev", "prod"] = "dev"
    ma_config_dir: Path = Path("config")
    ma_data_dir: Path | None = None
    ma_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def get(self, env_name: str) -> str | None:
        """Valor de una variable por su nombre de entorno, p. ej. ``"FINNHUB_API_KEY"``.

        Devuelve ``None`` si no está definida o está vacía. Los nombres que no
        correspondan a ningún campo también devuelven ``None``.
        """
        value = getattr(self, env_name.lower(), None)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value or None
        return None

    def has(self, env_name: str) -> bool:
        return self.get(env_name) is not None
